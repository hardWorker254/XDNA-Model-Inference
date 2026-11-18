//! Whisper's model: the encoder stack and the decoder stack.
//!
//! # Whisper is NOT BERT
//!
//! Three things differ, and all three of them are silent when you get one
//! of them wrong:
//!
//! * the normalisation ORDER: Whisper is pre-LN (residual, LayerNorm,
//!   attention, add) and BERT is post-LN;
//! * the LayerNorm itself: Whisper's accumulates the mean and the variance
//!   in DOUBLE and rounds them to float32 once, and BERT's sums a row in
//!   eight lanes with a fixed tree. The epsilon differs too -- 1e-5 against
//!   1e-12 -- and the difference is visible in the last bits of a 32-layer
//!   stack;
//! * the attention's exponential: Whisper's is `libm`'s `exp` and BERT's
//!   is a degree-7 polynomial on the fractional part.
//!
//! So the kernels here are Whisper's own and not the BERT ones reused: the
//! arithmetic is the same KIND of arithmetic and a different set of
//! numbers.
//!
//! # The encoder's shape
//!
//! The encoder's real input is `(1500, d)`: conv2's stride 2 turns 3000 mel
//! frames into 1500 positions, and 1500 is `max_source_positions` for
//! every shipped size. The design set was exported for a FIXED row count
//! (512 for whisper-tiny), so the reference walks the tensor in row chunks --
//! 512, 512, 476 -- and the last chunk's padded rows are zeroed.
//!
//! The chunking does not change a single number: the GEMM reduces over K
//! per row, and every row's reduction is the same whether it is in a
//! chunk or not. So this implementation walks the whole tensor at once and
//! gets the same bytes.
//!
//! # The position table is added BEFORE the first LayerNorm
//!
//! HF adds the position to the convolutional output, not after the first
//! norm, and the difference is not a rounding one.

use crate::kernels;
use crate::whisper_features::{MelSpec, gelu_erf_exact};
use npum_format::Container;
use std::path::Path;

/// The model's geometry, as the container states it.
#[derive(Debug, Clone)]
pub struct Geometry {
    /// The model's width.
    pub d_model: usize,
    /// How many heads.
    pub heads: usize,
    /// One head's width.
    pub head_dim: usize,
    /// How many encoder layers.
    pub enc_layers: usize,
    /// How many decoder layers.
    pub dec_layers: usize,
    /// The encoder's FFN width.
    pub enc_intermediate: usize,
    /// The decoder's FFN width.
    pub dec_intermediate: usize,
    /// How many positions the encoder's weights cover.
    pub max_seq: usize,
    /// How many target positions the decoder's weights cover.
    pub max_target: usize,
    /// The vocabulary's size.
    pub vocab: usize,
    /// How many mel bands.
    pub mel_bins: usize,
    /// The transform's hop, in samples.
    pub hop_length: usize,
    /// The LayerNorm's epsilon.
    pub ln_eps: f64,
    /// The attention's scale.
    pub attn_scale: f64,
    /// Whether the scale is already folded into Q's weight and bias.
    pub qkv_scale_folded: bool,
}

impl Geometry {
    /// Read the geometry out of a container.
    ///
    /// Every key is REQUIRED. A container that does not carry the geometry
    /// is a container this runtime cannot run, and a default here would be
    /// a model nobody chose.
    pub fn read(container: &Container) -> Option<Geometry> {
        let config = container.config();
        let geometry = Geometry {
            d_model: config.get_i64("d_model")? as usize,
            heads: config.get_i64("num_heads")? as usize,
            head_dim: config.get_i64("head_dim")? as usize,
            enc_layers: config.get_i64("num_encoder_layers")? as usize,
            dec_layers: config.get_i64("num_decoder_layers")? as usize,
            enc_intermediate: config.get_i64("encoder_intermediate")? as usize,
            dec_intermediate: config.get_i64("decoder_intermediate")? as usize,
            max_seq: config.get_i64("max_seq_len")? as usize,
            max_target: config.get_i64("max_target_positions")? as usize,
            vocab: config.get_i64("vocab_size")? as usize,
            mel_bins: config.get_i64("num_mel_bins")? as usize,
            hop_length: config.get_i64("hop_length")? as usize,
            ln_eps: config.get_f64("layer_norm_eps")?,
            attn_scale: config.get_f64("attention_scale")?,
            qkv_scale_folded: config.get_str("qkv_scale_folded") != Some("false"),
        };
        if geometry.heads * geometry.head_dim != geometry.d_model {
            return None;
        }
        Some(geometry)
    }
}

/// One encoder layer's weights.
struct EncoderLayer {
    qkv: Vec<f32>,
    qkv_bias: Vec<f32>,
    attn_out: Vec<f32>,
    attn_out_bias: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_up_bias: Vec<f32>,
    ffn_down: Vec<f32>,
    ffn_down_bias: Vec<f32>,
    ln1_gamma: Vec<f32>,
    ln1_beta: Vec<f32>,
    ln2_gamma: Vec<f32>,
    ln2_beta: Vec<f32>,
}

/// The encoder stack.
pub struct Encoder {
    geometry: Geometry,
    layers: Vec<EncoderLayer>,
    embed_positions: Vec<f32>,
    final_gamma: Vec<f32>,
    final_beta: Vec<f32>,
}

impl Encoder {
    /// Read the encoder's weights out of a container.
    pub fn open(path: &Path) -> Result<Encoder, String> {
        let container = Container::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let geometry = Geometry::read(&container).ok_or_else(|| {
            format!(
                "{}: the container does not carry the geometry",
                path.display()
            )
        })?;
        let mut layers = Vec::with_capacity(geometry.enc_layers);
        for l in 0..geometry.enc_layers {
            let p = format!("encoder.layers.{l}.");
            layers.push(EncoderLayer {
                qkv: tensor(&container, &format!("{p}qkv"))?,
                qkv_bias: tensor(&container, &format!("{p}qkv.bias"))?,
                attn_out: tensor(&container, &format!("{p}attn_out"))?,
                attn_out_bias: tensor(&container, &format!("{p}attn_out.bias"))?,
                ffn_up: tensor(&container, &format!("{p}ffn_up"))?,
                ffn_up_bias: tensor(&container, &format!("{p}ffn_up.bias"))?,
                ffn_down: tensor(&container, &format!("{p}ffn_down"))?,
                ffn_down_bias: tensor(&container, &format!("{p}ffn_down.bias"))?,
                ln1_gamma: tensor(&container, &format!("{p}ln1.weight"))?,
                ln1_beta: tensor(&container, &format!("{p}ln1.bias"))?,
                ln2_gamma: tensor(&container, &format!("{p}ln2.weight"))?,
                ln2_beta: tensor(&container, &format!("{p}ln2.bias"))?,
            });
        }
        Ok(Encoder {
            embed_positions: tensor(&container, "encoder.embed_positions")?,
            final_gamma: tensor(&container, "encoder.layer_norm.weight")?,
            final_beta: tensor(&container, "encoder.layer_norm.bias")?,
            layers,
            geometry,
        })
    }

    /// The model's geometry.
    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    /// The encoder's forward pass.
    ///
    /// `conv_out` is the front end's `(n_src, d_model)` output with no
    /// positional term yet, and `n_src` is how many positions of it are
    /// real. The result is `(n_src, d_model)` AFTER the final LayerNorm,
    /// which is what transformers hands the decoder.
    pub fn run(&self, conv_out: &[f32], n_src: usize) -> Vec<f32> {
        let g = &self.geometry;
        let d = g.d_model;

        // + embed_positions, before the first LayerNorm: HF adds the position
        // to the convolutional output, not after the first norm.
        let mut x = conv_out[..n_src * d].to_vec();
        for t in 0..n_src {
            for j in 0..d {
                x[t * d + j] += self.embed_positions[t * d + j];
            }
        }

        // The attention's scale: it is already inside Q's weight and Q's
        // bias unless the container was packed with --no-fold-scale. Applying
        // it here when it is folded would be applying it twice, which is a
        // different model.
        let scale = if g.qkv_scale_folded {
            1.0f32
        } else {
            g.attn_scale as f32
        };

        for layer in &self.layers {
            // ---- self-attention, pre-LN ----
            let mut norm = x.clone();
            for row in norm.chunks_exact_mut(d) {
                layernorm_row(row, &layer.ln1_gamma, &layer.ln1_beta, g.ln_eps);
            }
            let mut qkv = vec![0.0f32; n_src * 3 * d];
            kernels::gemm_nt(
                &norm,
                n_src,
                &layer.qkv,
                d,
                3 * d,
                &layer.qkv_bias,
                &mut qkv,
            );
            let mut ctx = vec![0.0f32; n_src * d];
            attention(
                &qkv,
                3 * d,
                &qkv[d..],
                3 * d,
                n_src,
                n_src,
                d,
                g.heads,
                g.head_dim,
                scale,
                &mut ctx,
            );
            let mut proj = vec![0.0f32; n_src * d];
            kernels::gemm_nt(
                &ctx,
                n_src,
                &layer.attn_out,
                d,
                d,
                &layer.attn_out_bias,
                &mut proj,
            );
            for i in 0..n_src * d {
                x[i] += proj[i];
            }

            // ---- feed-forward, pre-LN ----
            let mut norm = x.clone();
            for row in norm.chunks_exact_mut(d) {
                layernorm_row(row, &layer.ln2_gamma, &layer.ln2_beta, g.ln_eps);
            }
            let mut up = vec![0.0f32; n_src * g.enc_intermediate];
            kernels::gemm_nt(
                &norm,
                n_src,
                &layer.ffn_up,
                d,
                g.enc_intermediate,
                &layer.ffn_up_bias,
                &mut up,
            );
            for v in up.iter_mut() {
                *v = gelu_erf_exact(*v);
            }
            let mut down = vec![0.0f32; n_src * d];
            kernels::gemm_nt(
                &up,
                n_src,
                &layer.ffn_down,
                g.enc_intermediate,
                d,
                &layer.ffn_down_bias,
                &mut down,
            );
            for i in 0..n_src * d {
                x[i] += down[i];
            }
        }

        // Whisper's encoder output is AFTER the final LayerNorm -- that is
        // what transformers hands the decoder.
        for row in x.chunks_exact_mut(d) {
            layernorm_row(row, &self.final_gamma, &self.final_beta, g.ln_eps);
        }
        x
    }
}

// ---------------------------------------------------------------------------
// The decoder
// ---------------------------------------------------------------------------

/// One decoder layer's weights.
struct DecoderLayer {
    self_qkv: Vec<f32>,
    self_qkv_bias: Vec<f32>,
    self_attn_out: Vec<f32>,
    self_attn_out_bias: Vec<f32>,
    cross_q: Vec<f32>,
    cross_q_bias: Vec<f32>,
    cross_kv: Vec<f32>,
    cross_kv_bias: Vec<f32>,
    cross_attn_out: Vec<f32>,
    cross_attn_out_bias: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_up_bias: Vec<f32>,
    ffn_down: Vec<f32>,
    ffn_down_bias: Vec<f32>,
    ln1_gamma: Vec<f32>,
    ln1_beta: Vec<f32>,
    ln2_gamma: Vec<f32>,
    ln2_beta: Vec<f32>,
    ln3_gamma: Vec<f32>,
    ln3_beta: Vec<f32>,
}

/// The decoder stack: greedy autoregression over the encoder's output,
/// with a KV cache per layer.
///
/// The cache is exact, not approximate: a cached K/V row is bit-identical
/// to what recomputing it from the same state and the same weights would
/// produce, because the GEMM, the operand and the bias are the same
/// operations either way.
///
/// Positions are structural, not masked: the cache only ever holds
/// positions at or before the current one, so causality cannot be
/// violated by a caller and there is no mask to get wrong.
pub struct Decoder {
    geometry: Geometry,
    layers: Vec<DecoderLayer>,
    embed_positions: Vec<f32>,
    embed_tokens: Vec<f32>,
    final_gamma: Vec<f32>,
    final_beta: Vec<f32>,
    /// The per-layer cross-attention K|V cache, `[n_src, 2*d_model]` each.
    cross_kv: Vec<Vec<f32>>,
    /// The per-layer self-attention K|V cache, `[n_cached, 2*d_model]` each.
    self_kv: Vec<Vec<f32>>,
    /// How many positions the self-attention cache holds.
    n_cached: usize,
    /// How many positions the cross-attention cache holds.
    n_src: usize,
    /// How many tokens this decoder has GENERATED, which is what the
    /// policy's "begin" list is keyed on.
    n_generated: usize,
    /// Forbidden at every step.
    suppress: Vec<i32>,
    /// Forbidden at the first generated token only.
    begin_suppress: Vec<i32>,
}

impl Decoder {
    /// Read the decoder's weights out of a container.
    pub fn open(path: &Path) -> Result<Decoder, String> {
        let container = Container::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let geometry = Geometry::read(&container).ok_or_else(|| {
            format!(
                "{}: the container does not carry the geometry",
                path.display()
            )
        })?;
        let mut layers = Vec::with_capacity(geometry.dec_layers);
        for l in 0..geometry.dec_layers {
            let p = format!("decoder.layers.{l}.");
            layers.push(DecoderLayer {
                self_qkv: tensor(&container, &format!("{p}self_qkv"))?,
                self_qkv_bias: tensor(&container, &format!("{p}self_qkv.bias"))?,
                self_attn_out: tensor(&container, &format!("{p}self_attn_out"))?,
                self_attn_out_bias: tensor(&container, &format!("{p}self_attn_out.bias"))?,
                cross_q: tensor(&container, &format!("{p}cross_q"))?,
                cross_q_bias: tensor(&container, &format!("{p}cross_q.bias"))?,
                cross_kv: tensor(&container, &format!("{p}cross_kv"))?,
                cross_kv_bias: tensor(&container, &format!("{p}cross_kv.bias"))?,
                cross_attn_out: tensor(&container, &format!("{p}cross_attn_out"))?,
                cross_attn_out_bias: tensor(&container, &format!("{p}cross_attn_out.bias"))?,
                ffn_up: tensor(&container, &format!("{p}ffn_up"))?,
                ffn_up_bias: tensor(&container, &format!("{p}ffn_up.bias"))?,
                ffn_down: tensor(&container, &format!("{p}ffn_down"))?,
                ffn_down_bias: tensor(&container, &format!("{p}ffn_down.bias"))?,
                ln1_gamma: tensor(&container, &format!("{p}ln1.weight"))?,
                ln1_beta: tensor(&container, &format!("{p}ln1.bias"))?,
                ln2_gamma: tensor(&container, &format!("{p}ln2.weight"))?,
                ln2_beta: tensor(&container, &format!("{p}ln2.bias"))?,
                ln3_gamma: tensor(&container, &format!("{p}ln3.weight"))?,
                ln3_beta: tensor(&container, &format!("{p}ln3.bias"))?,
            });
        }
        // The decoding policy. Both lists are REQUIRED: a container packed
        // before they were stored is refused rather than run with a policy of
        // its own, because the difference shows up only in the text and
        // nowhere else.
        let suppress = int_list(&container, "decoder.suppress_tokens")?;
        let begin_suppress = int_list(&container, "decoder.begin_suppress_tokens")?;
        if suppress.is_empty() && begin_suppress.is_empty() {
            return Err(format!(
                "{}: no decoding policy. The container carries neither decoder.suppress_tokens nor decoder.begin_suppress_tokens, so a greedy step here would pick a different token than the reference implementation does.",
                path.display()
            ));
        }
        for list in [&suppress, &begin_suppress] {
            for &id in list {
                if id < 0 || id as usize >= geometry.vocab {
                    return Err(format!(
                        "{}: the decoding policy names id {id}, outside this container's {}-entry vocabulary -- the lists were packed against a different checkpoint",
                        path.display(),
                        geometry.vocab
                    ));
                }
            }
        }
        Ok(Decoder {
            embed_positions: tensor(&container, "decoder.embed_positions")?,
            embed_tokens: tensor(&container, "decoder.embed_tokens")?,
            final_gamma: tensor(&container, "decoder.layer_norm.weight")?,
            final_beta: tensor(&container, "decoder.layer_norm.bias")?,
            cross_kv: vec![Vec::new(); geometry.dec_layers],
            self_kv: vec![Vec::new(); geometry.dec_layers],
            n_cached: 0,
            n_src: 0,
            n_generated: 0,
            suppress,
            begin_suppress,
            layers,
            geometry,
        })
    }

    /// The model's geometry.
    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    /// Build the per-layer cross-attention K|V cache from the encoder's
    /// output.
    ///
    /// Must be called before the first step, and again whenever the source
    /// changes. The cache is built ONCE per transcription: the encoder's
    /// output does not change while the decoder runs, so 1500 positions at
    /// 256 rows is six dispatches per layer instead of 1500.
    pub fn set_source(&mut self, enc: &[f32], n_src: usize) -> Result<(), String> {
        let d = self.geometry.d_model;
        if n_src == 0 || enc.len() < n_src * d {
            return Err(format!(
                "whisper decoder: encoder output is {} rows, {n_src} wanted",
                enc.len() / d
            ));
        }
        self.n_src = n_src;
        for (l, layer) in self.layers.iter().enumerate() {
            let mut cache = vec![0.0f32; n_src * 2 * d];
            kernels::gemm_nt(
                enc,
                n_src,
                &layer.cross_kv,
                d,
                2 * d,
                &layer.cross_kv_bias,
                &mut cache,
            );
            self.cross_kv[l] = cache;
        }
        self.reset();
        Ok(())
    }

    /// Forget every cached K/V row. A new transcription starts here.
    pub fn reset(&mut self) {
        self.n_cached = 0;
        self.n_generated = 0;
        for kv in self.self_kv.iter_mut() {
            kv.clear();
        }
    }

    /// How many positions the self-attention cache holds.
    pub fn cached(&self) -> usize {
        self.n_cached
    }

    /// One step: run `token` at `position` and return the greedy next id.
    ///
    /// `hidden` receives the final-LayerNorm state and `logits` the
    /// tied-embedding projection of it.
    pub fn step(
        &mut self,
        token: i32,
        position: usize,
        hidden: Option<&mut Vec<f32>>,
        logits: Option<&mut Vec<f32>>,
    ) -> Result<i32, String> {
        let g = &self.geometry;
        let d = g.d_model;
        if token < 0 || token as usize >= g.vocab {
            return Err(format!(
                "whisper decoder: token {token} is outside the vocabulary ({} ids)",
                g.vocab
            ));
        }
        // Positions are the cache's own index. Accepting a token at any
        // other position would produce a fluent transcript with the wrong
        // position embedding, which is the worst kind of wrong.
        if position != self.n_cached {
            return Err(format!(
                "whisper decoder: token at position {position} but the cache holds {} positions. Tokens must be fed in order, or reset() first.",
                self.n_cached
            ));
        }
        if self.n_cached >= g.max_target {
            return Err(format!(
                "whisper decoder: position {position} is past max_target_positions {}",
                g.max_target
            ));
        }
        let scale = if g.qkv_scale_folded {
            1.0f32
        } else {
            g.attn_scale as f32
        };

        // embed_tokens + embed_positions, no norm in between: HF's decoder
        // adds the position to the token embedding and enters the first
        // pre-LN block.
        let mut h = vec![0.0f32; d];
        let et = &self.embed_tokens[token as usize * d..][..d];
        let pe = &self.embed_positions[position * d..][..d];
        for j in 0..d {
            h[j] = et[j] + pe[j];
        }

        let mut qkv = vec![0.0f32; 3 * d];
        let mut proj = vec![0.0f32; d];
        let mut up = vec![0.0f32; g.dec_intermediate];
        let mut down = vec![0.0f32; d];
        let mut ctx = vec![0.0f32; d];

        for (l, layer) in self.layers.iter().enumerate() {
            // ---- self-attention over the cache, one new row ----
            let mut norm = h.clone();
            layernorm_row(&mut norm, &layer.ln1_gamma, &layer.ln1_beta, g.ln_eps);
            kernels::gemm_nt(
                &norm,
                1,
                &layer.self_qkv,
                d,
                3 * d,
                &layer.self_qkv_bias,
                &mut qkv,
            );
            // The K|V half of the fused QKV joins the cache; the Q half is
            // this step's query.
            let cache = &mut self.self_kv[l];
            cache.extend_from_slice(&qkv[d..3 * d]);
            attention(
                &qkv,
                3 * d,
                cache,
                2 * d,
                1,
                self.n_cached + 1,
                d,
                g.heads,
                g.head_dim,
                scale,
                &mut ctx,
            );
            kernels::gemm_nt(
                &ctx,
                1,
                &layer.self_attn_out,
                d,
                d,
                &layer.self_attn_out_bias,
                &mut proj,
            );
            for j in 0..d {
                h[j] += proj[j];
            }

            // ---- cross-attention over the encoder output ----
            let mut norm = h.clone();
            layernorm_row(&mut norm, &layer.ln2_gamma, &layer.ln2_beta, g.ln_eps);
            kernels::gemm_nt(
                &norm,
                1,
                &layer.cross_q,
                d,
                d,
                &layer.cross_q_bias,
                &mut qkv,
            );
            // The Q row is all this step needs from cross_q: its K|V half was
            // folded into a separate operand precisely so the two could be
            // fed from different A operands.
            attention(
                &qkv,
                d,
                &self.cross_kv[l],
                2 * d,
                1,
                self.n_src,
                d,
                g.heads,
                g.head_dim,
                scale,
                &mut ctx,
            );
            kernels::gemm_nt(
                &ctx,
                1,
                &layer.cross_attn_out,
                d,
                d,
                &layer.cross_attn_out_bias,
                &mut proj,
            );
            for j in 0..d {
                h[j] += proj[j];
            }

            // ---- feed-forward ----
            let mut norm = h.clone();
            layernorm_row(&mut norm, &layer.ln3_gamma, &layer.ln3_beta, g.ln_eps);
            kernels::gemm_nt(
                &norm,
                1,
                &layer.ffn_up,
                d,
                g.dec_intermediate,
                &layer.ffn_up_bias,
                &mut up,
            );
            for v in up.iter_mut() {
                *v = gelu_erf_exact(*v);
            }
            kernels::gemm_nt(
                &up,
                1,
                &layer.ffn_down,
                g.dec_intermediate,
                d,
                &layer.ffn_down_bias,
                &mut down,
            );
            for j in 0..d {
                h[j] += down[j];
            }
        }

        self.n_cached += 1;
        layernorm_row(&mut h, &self.final_gamma, &self.final_beta, g.ln_eps);
        let mut logit_vector = logits_from(&h, &self.embed_tokens, d, g.vocab);
        let next = self.apply_policy(&mut logit_vector);
        if let Some(out) = hidden {
            *out = h;
        }
        if let Some(out) = logits {
            *out = logit_vector;
        }
        self.n_generated += 1;
        Ok(next)
    }

    /// The policy's two lists, the argmax, and the tie-break towards the
    /// lower id.
    ///
    /// transformers installs TWO processors and they OVERLAP on the first
    /// step: `SuppressTokensLogitsProcessor(suppress_tokens)` at every step,
    /// and `SuppressTokensAtBeginLogitsProcessor(begin_suppress_tokens)` at
    /// the first generated step only. Applying only the begin list there --
    /// which is the obvious reading of "begin" -- lets a token the full list
    /// forbids win the first step: on a 3 s tone that is 50362 instead of
    /// 50259, and the whole transcript after it is a different language.
    ///
    /// The policy is applied to the VECTOR and not to the argmax, so that a
    /// caller reading the logits sees -3.4e+38 exactly where the choice was
    /// forbidden.
    fn apply_policy(&self, logits: &mut [f32]) -> i32 {
        let vocab = logits.len();
        for &id in &self.suppress {
            if id >= 0 && (id as usize) < vocab {
                logits[id as usize] = -3.4e38f32;
            }
        }
        if self.n_generated == 0 {
            for &id in &self.begin_suppress {
                if id >= 0 && (id as usize) < vocab {
                    logits[id as usize] = -3.4e38f32;
                }
            }
        }
        // Ties broken by the lower id, so the result does not depend on the
        // order the ids were looked at: the argmax is a function of the
        // logits and not of the schedule.
        let mut best_id = 0usize;
        let mut best = f32::NEG_INFINITY;
        let mut have = false;
        for (v, &x) in logits.iter().enumerate() {
            if !have || x > best {
                best = x;
                best_id = v;
                have = true;
            }
        }
        best_id as i32
    }

    /// Greedy continuation of `prompt` (fed at positions 0..n-1), up to
    /// `stop_id`, `max_new` generated tokens, or the checkpoint's own
    /// max_target_positions -- whichever comes first.
    ///
    /// The cap matters for more than tidiness: a reference implementation is
    /// compared against this with ITS max_new_tokens, and a 31 s tone makes
    /// the model emit the same token until it runs out of positions, so an
    /// uncapped chain and a capped reference are not the same string.
    pub fn greedy(
        &mut self,
        prompt: &[i32],
        stop_id: i32,
        max_new: usize,
    ) -> Result<Vec<i32>, String> {
        if prompt.is_empty() {
            return Err("whisper decoder: greedy() needs at least one prompt token (the control tokens are the model's own)".into());
        }
        if prompt.len() > self.geometry.max_target {
            return Err(format!(
                "whisper decoder: prompt of {} tokens exceeds max_target_positions {}",
                prompt.len(),
                self.geometry.max_target
            ));
        }
        self.reset();
        let mut next = 0;
        for (i, &token) in prompt.iter().enumerate() {
            next = self.step(token, i, None, None)?;
            if next == stop_id {
                return Ok(Vec::new());
            }
        }
        let mut out = vec![next];
        while out.last() != Some(&stop_id)
            && self.n_cached < self.geometry.max_target
            && (max_new == 0 || out.len() < max_new)
        {
            let Some(&token) = out.last() else {
                break;
            };
            let position = self.n_cached;
            out.push(self.step(token, position, None, None)?);
        }
        if out.last() == Some(&stop_id) {
            out.pop();
        }
        Ok(out)
    }
}

/// The vocabulary projection: `logits = h @ embed_tokens^T`.
///
/// The checkpoint has no `proj_out`: the logit matrix is the tied token
/// embedding. 66 MB of weights are read for 66 MFLOP of work, which is why
/// the HOST does it.
///
/// The accumulation is EIGHT lane accumulators with an FMA per element,
/// finished with the same eight-lane horizontal sum the attention uses --
/// and not the GEMM's own reduction, which walks K in order for each
/// output. The two are different trees and the two are different numbers.
pub fn logits_from(h: &[f32], embed_tokens: &[f32], d: usize, vocab: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; vocab];
    for (v, slot) in out.iter_mut().enumerate() {
        let e = &embed_tokens[v * d..][..d];
        let mut lanes = [0.0f32; 8];
        for l in 0..8 {
            lanes[l] = h[l] * e[l];
        }
        let mut j = 8;
        while j + 8 <= d {
            for l in 0..8 {
                lanes[l] = h[j + l].mul_add(e[j + l], lanes[l]);
            }
            j += 8;
        }
        let mut acc = kernels::hsum8(lanes);
        while j < d {
            acc += h[j] * e[j];
            j += 1;
        }
        *slot = acc;
    }
    out
}

/// LayerNorm over one row, as the reference's `layernorm_rows` computes it.
///
/// The mean and the variance are accumulated in DOUBLE and rounded to
/// float32 once, and the epsilon is the container's -- 1e-5 for Whisper,
/// not BERT's 1e-12.
///
/// The epilogue is an FMA: `(x - mean) * inv * gamma + beta` contracts to
/// one rounding, and the reference's compiler contracts it.
pub fn layernorm_row(row: &mut [f32], gamma: &[f32], beta: &[f32], eps: f64) {
    let d = row.len();
    let mut sm = 0.0f64;
    for v in row.iter() {
        sm += *v as f64;
    }
    let mean = (sm / d as f64) as f32;
    let mut sv = 0.0f64;
    for v in row.iter() {
        let dd = *v as f64 - mean as f64;
        sv += dd * dd;
    }
    let var = (sv / d as f64) as f32;
    let inv = 1.0f32 / (var + eps as f32).sqrt();
    for (j, v) in row.iter_mut().enumerate() {
        *v = ((*v - mean) * inv).mul_add(gamma[j], beta[j]);
    }
}

/// Attention over a K|V block, with the queries in a separate operand.
///
/// `q` is `[n_q, d_model]` with row stride `q_stride`; `kv` is
/// `[n_kv, 2*d_model]` (K first, then V) with row stride `kv_stride`. The
/// strides are what let one function serve all three of Whisper's
/// attentions: the encoder's fused `[Q|K|V]` operand is a q block and a kv
/// block with stride `3*d_model`, the decoder's self-attention cache is
/// `[K|V]` with stride `2*d_model`, and the cross-attention cache is the
/// same shape built once from the encoder output.
///
/// `scale` is 1.0 when the packer folded `1/sqrt(head_dim)` into the Q
/// weight and the Q bias, and the real attention scale when it did not.
///
/// There is NO mask: the encoder attends over every position it was given,
/// and the decoder's cache holds only positions at or before the current
/// one, so causality is structural rather than applied.
pub fn attention(
    q: &[f32],
    q_stride: usize,
    kv: &[f32],
    kv_stride: usize,
    n_q: usize,
    n_kv: usize,
    d_model: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
) {
    let nv = head_dim / 8;
    let mut scores = vec![0.0f32; n_kv];
    for i in 0..n_q {
        for h in 0..heads {
            // QK^T, one row of scores at a time. The accumulation is eight
            // lane accumulators with an FMA per element, finished with the
            // same eight-lane horizontal sum the reference's `hsum256` does.
            for (j, slot) in scores.iter_mut().enumerate().take(n_kv) {
                let qh = &q[i * q_stride + h * head_dim..][..head_dim];
                let kj = &kv[j * kv_stride + h * head_dim..][..head_dim];
                let mut lanes = [0.0f32; 8];
                for l in 0..8 {
                    lanes[l] = qh[l] * kj[l];
                }
                for v in 1..nv {
                    for l in 0..8 {
                        lanes[l] = qh[v * 8 + l].mul_add(kj[v * 8 + l], lanes[l]);
                    }
                }
                *slot = kernels::hsum8(lanes) * scale;
            }

            // Softmax with the maximum subtracted, the same reduction HF's
            // own attention does. The exponential is `libm`'s `exp` -- NOT
            // the BERT path's polynomial -- and the sum is a plain float32
            // accumulation in order.
            let mut mx = scores[0];
            for s in scores.iter().take(n_kv).skip(1) {
                mx = mx.max(*s);
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut().take(n_kv) {
                *s = (*s - mx).exp();
                sum += *s;
            }
            let inv = if sum > 0.0 { 1.0f32 / sum } else { 0.0f32 };

            // softmax.V, into this head's slice of the output row. One
            // accumulator per eight of the head's width -- the reference's
            // `__m256 acc[NV]`, and not one accumulator for the whole head.
            // The difference is the ORDER of the additions, and the order is
            // the answer.
            let o = &mut out[i * d_model + h * head_dim..][..head_dim];
            let mut acc = vec![0.0f32; head_dim];
            for (j, &aj) in scores.iter().enumerate().take(n_kv) {
                let a = aj * inv;
                let vj = &kv[j * kv_stride + d_model + h * head_dim..][..head_dim];
                for lane in 0..nv {
                    for l in 0..8 {
                        acc[lane * 8 + l] = a.mul_add(vj[lane * 8 + l], acc[lane * 8 + l]);
                    }
                }
            }
            o.copy_from_slice(&acc);
        }
    }
}

/// A tensor's values.
fn tensor(container: &Container, name: &str) -> Result<Vec<f32>, String> {
    let bytes = container
        .tensor_bytes(name)
        .map_err(|_| format!("the container holds no tensor named {name}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// A tensor's values, as int32.
///
/// A tensor the container does not have is an empty list, not an error: the
/// policy lists are optional in the sense that one of them may be absent,
/// and the caller checks that at least one is not.
fn int_list(container: &Container, name: &str) -> Result<Vec<i32>, String> {
    if container.directory().tensor(name).is_none() {
        return Ok(Vec::new());
    }
    let bytes = container
        .tensor_bytes(name)
        .map_err(|e| format!("{name}: {e}"))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{name}: {} bytes is not a whole number of int32",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// The front end's output, ready for the encoder.
pub fn front_end(
    mel: &MelSpec,
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    hidden: usize,
) -> Vec<f32> {
    crate::whisper_features::conv_front_end(mel, w1, b1, w2, b2, hidden)
}
