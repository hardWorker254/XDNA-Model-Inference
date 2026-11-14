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
//! (512 for whisper-tiny), so the C++ walks the tensor in row chunks --
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
use crate::whisper_features::{gelu_erf_exact, MelSpec};
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
        let geometry = Geometry::read(&container)
            .ok_or_else(|| format!("{}: the container does not carry the geometry", path.display()))?;
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
            kernels::gemm_nt(&norm, n_src, &layer.qkv, d, 3 * d, &layer.qkv_bias, &mut qkv);
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
            kernels::gemm_nt(&ctx, n_src, &layer.attn_out, d, d, &layer.attn_out_bias, &mut proj);
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

/// LayerNorm over one row, as the C++'s `layernorm_rows` computes it.
///
/// The mean and the variance are accumulated in DOUBLE and rounded to
/// float32 once, and the epsilon is the container's -- 1e-5 for Whisper,
/// not BERT's 1e-12.
///
/// The epilogue is an FMA: `(x - mean) * inv * gamma + beta` contracts to
/// one rounding, and the C++'s compiler contracts it.
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
            // same eight-lane horizontal sum the C++'s `hsum256` does.
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
            // accumulator per eight of the head's width -- the C++'s
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

