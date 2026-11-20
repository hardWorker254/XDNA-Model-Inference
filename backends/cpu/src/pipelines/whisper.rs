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
use crate::parallel;
use crate::whisper_features::{gelu_erf_exact, MelSpec};
use npum_format::Container;
use npum_model::array::{ArrayEltwise, ArrayText, ArrayTextRef, AttentionShape, EltOp, };
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
    /// The encoder's projection names, in the order the run dispatches them.
    gemm_names: Vec<[String; 4]>,
    embed_positions: Vec<f32>,
    final_gamma: Vec<f32>,
    final_beta: Vec<f32>,
}

/// Reusable scratch buffers for an encoder forward pass.
///
/// The pass touches buffers of `n_src * d_model` (and `n_src * intermediate`)
/// floats in every layer. Allocating them per layer is measurable for a
/// 1500-position encoder, and a transcription runs the encoder once per window.
/// A caller that keeps one of these can reuse the buffers across layers AND
/// across calls with `Encoder::run_into`.
#[derive(Debug, Clone, Default)]
pub struct EncoderWorkspace {
    x: Vec<f32>,
    norm: Vec<f32>,
    qkv: Vec<f32>,
    ctx: Vec<f32>,
    proj: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    scores: Vec<f32>,
    acc: Vec<f32>,
    /// gamma|beta for the array's LayerNorm, and the zero attention mask.
    ln_params: Vec<f32>,
    mask: Vec<f32>,
}

impl EncoderWorkspace {
    /// A fresh, empty workspace.
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure(&mut self, g: &Geometry, n_src: usize, score_cols: usize) {
        let d = g.d_model;
        let rows = n_src;
        resize_f32(&mut self.x, rows * d);
        resize_f32(&mut self.norm, rows * d);
        resize_f32(&mut self.qkv, rows * 3 * d);
        resize_f32(&mut self.ctx, rows * d);
        resize_f32(&mut self.proj, rows * d);
        resize_f32(&mut self.up, rows * g.enc_intermediate);
        resize_f32(&mut self.down, rows * d);
        // The host's row is one entry per source position; the array's (and
        // the array softmax's) row is the design's padded width.
        resize_f32(&mut self.scores, rows * g.heads * score_cols);
        resize_f32(&mut self.acc, g.head_dim);
        resize_f32(&mut self.ln_params, 2 * d);
        // Whisper's encoder has no mask; the array's attention takes one
        // value per query, all zero.
        resize_f32(&mut self.mask, rows);
        self.mask.fill(0.0);
    }
}

#[inline]
fn resize_f32(buf: &mut Vec<f32>, len: usize) {
    if buf.len() != len {
        buf.resize(len, 0.0);
    }
}

/// One projection: the array's seam when asked, the host's kernel otherwise.
///
/// The host arm is the ORIGINAL call, byte for byte; the array arm multiplies
/// the same weights in bf16 and is the same network with a coarser operand.
#[allow(clippy::too_many_arguments)]
fn gemm_seam(
    array: Option<&mut (dyn ArrayText + '_)>,
    gemm_on: bool,
    weight: &str,
    a: &[f32],
    rows: usize,
    b: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
    threads: usize,
) -> Result<(), String> {
    match array.filter(|_| gemm_on) {
        Some(backend) => backend
            .gemm(weight, a, rows, k, n, Some(bias), out)
            .map_err(|e| e.to_string()),
        None => {
            kernels::gemm_nt_rows(a, rows, b, k, n, bias, out, threads);
            Ok(())
        }
    }
}

/// One LayerNorm with a separate destination: copy and normalise on the
/// array, or the host's fused copy-and-normalise kernel.
#[allow(clippy::too_many_arguments)]
fn layernorm_seam(
    eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
    ln_on: bool,
    params: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    src: &[f32],
    dst: &mut [f32],
    rows: usize,
    width: usize,
    eps: f64,
) -> Result<(), String> {
    if let Some(eltwise) = eltwise.filter(|_| ln_on) {
        let len = rows * width;
        dst[..len].copy_from_slice(&src[..len]);
        params[..width].copy_from_slice(gamma);
        params[width..2 * width].copy_from_slice(beta);
        eltwise
            .layernorm(&params[..2 * width], &mut dst[..len], rows)
            .map_err(|e| e.to_string())
    } else {
        layernorm_rows(src, dst, rows, width, gamma, beta, eps);
        Ok(())
    }
}

/// One decoder attention: the array's whole block when asked, the host's
/// attention otherwise (with the eltwise softmax when only `softm` was named).
///
/// The array's attention takes Q|K|V in one buffer. The decoder keeps K|V in a
/// cache of its own, so the caller's reusable buffers are filled with Q in the
/// first row and K|V from the cache in every row. Rows past the first are
/// queried and discarded -- they carry no answer, but they let the array read
/// one [Q|K|V] block instead of three separate operands.
#[allow(clippy::too_many_arguments)]
fn decoder_attention(
    array: Option<&mut (dyn ArrayText + '_)>,
    attn_on: bool,
    sm_on: bool,
    q: &[f32],
    kv: &[f32],
    kv_stride: usize,
    n_kv: usize,
    d: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
    scores: &mut [f32],
    sc_cols: usize,
    acc: &mut [f32],
    attn_qkv: &mut [f32],
    attn_ctx: &mut [f32],
    mask: &mut [f32],
    softmax: Option<&mut (dyn ArrayEltwise + '_)>,
) -> Result<(), String> {
    if attn_on {
        let backend = array.ok_or_else(|| {
            "the array attention was asked for but no handle was given".to_string()
        })?;
        for j in 0..n_kv {
            let row = &mut attn_qkv[j * 3 * d..][..3 * d];
            row.fill(0.0);
            if j == 0 {
                row[..d].copy_from_slice(q);
            }
            row[d..2 * d].copy_from_slice(&kv[j * kv_stride..][..d]);
            row[2 * d..3 * d].copy_from_slice(&kv[j * kv_stride + d..][..d]);
        }
        let shape = AttentionShape {
            batch: 1,
            seq: n_kv,
            heads,
            head_dim,
            hidden: d,
        };
        let m = &mut mask[..n_kv];
        m.fill(0.0);
        backend
            .attention(
                &attn_qkv[..n_kv * 3 * d],
                shape,
                m,
                softmax,
                &mut attn_ctx[..n_kv * d],
            )
            .map_err(|e| e.to_string())?;
        out[..d].copy_from_slice(&attn_ctx[..d]);
        Ok(())
    } else if sm_on {
        let softmax = softmax
            .ok_or_else(|| "the array softmax was asked for but no handle was given".to_string())?;
        attention_eltwise_into(
            q, d, kv, kv_stride, 1, n_kv, d, heads, head_dim, scale, out, scores, sc_cols, acc,
            softmax,
        )
    } else {
        attention_into(
            q, d, kv, kv_stride, 1, n_kv, d, heads, head_dim, scale, out, scores, acc,
        );
        Ok(())
    }
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
        let mut gemm_names = Vec::with_capacity(geometry.enc_layers);
        for l in 0..geometry.enc_layers {
            let p = format!("encoder.layers.{l}.");
            gemm_names.push([
                format!("{p}qkv"),
                format!("{p}attn_out"),
                format!("{p}ffn_up"),
                format!("{p}ffn_down"),
            ]);
            layers.push(EncoderLayer {
                qkv: operand(&container, &format!("{p}qkv"))?,
                qkv_bias: tensor(&container, &format!("{p}qkv.bias"))?,
                attn_out: operand(&container, &format!("{p}attn_out"))?,
                attn_out_bias: tensor(&container, &format!("{p}attn_out.bias"))?,
                ffn_up: operand(&container, &format!("{p}ffn_up"))?,
                ffn_up_bias: tensor(&container, &format!("{p}ffn_up.bias"))?,
                ffn_down: operand(&container, &format!("{p}ffn_down"))?,
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
            gemm_names,
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
    pub fn run(
        &self,
        conv_out: &[f32],
        n_src: usize,
        array: ArrayTextRef<'_, '_>,
        eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
    ) -> Result<Vec<f32>, String> {
        let mut ws = EncoderWorkspace::new();
        Ok(self
            .run_into(conv_out, n_src, &mut ws, array, eltwise)?
            .to_vec())
    }

    /// The same pass, with the scratch buffers supplied by the caller.
    ///
    /// The returned slice borrows `workspace`, not `self`: a caller that
    /// transcribes many windows can keep one workspace and hand it back on
    /// every call. Every buffer is overwritten before it is read, so the
    /// workspace needs no clearing between calls.
    ///
    /// `array` and `eltwise` are the array's seams. With both `None` every
    /// call below is the ORIGINAL host call, byte for byte. A pass named by
    /// `--npu-ops` goes through the trait instead, and an array that refuses
    /// a dispatch is an error -- never a silent host computation.
    pub fn run_into<'ws>(
        &self,
        conv_out: &[f32],
        n_src: usize,
        ws: &'ws mut EncoderWorkspace,
        mut array: ArrayTextRef<'_, '_>,
        mut eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
    ) -> Result<&'ws [f32], String> {
        let g = &self.geometry;
        let d = g.d_model;

        // THE SCORE ROW'S WIDTH. The array's softmax normalises the DESIGN's
        // padded key count, so the scores must be laid out at that width --
        // and a design narrower than the sequence cannot be used at all.
        let sc_cols = match eltwise.as_deref().and_then(|e| e.softmax_cols()) {
            Some(cols) => {
                if cols < n_src {
                    return Err(format!(
                        "the softmax design's score row is {cols} wide, but this encoder has {n_src} positions"
                    ));
                }
                cols
            }
            None => n_src,
        };
        let gemm_on = array.as_deref().is_some_and(|a| a.gemm_on_array());
        let attn_on = array.as_deref().is_some_and(|a| a.on_array());
        if attn_on {
            let attn_cols = array
                .as_deref()
                .and_then(|a| a.score_cols())
                .unwrap_or(n_src);
            if let Some(cols) = eltwise.as_deref().and_then(|e| e.softmax_cols()) {
                if cols != attn_cols {
                    return Err(format!(
                        "the softmax design's score row is {cols} wide and the attention's is {attn_cols} wide"
                    ));
                }
            }
        }
        let ln_on = eltwise
            .as_deref()
            .is_some_and(|e| e.on_array(EltOp::LayerNorm));
        let gelu_on = eltwise.as_deref().is_some_and(|e| e.on_array(EltOp::Gelu));
        let sm_on = eltwise
            .as_deref()
            .is_some_and(|e| e.on_array(EltOp::Softmax));

        ws.ensure(g, n_src, sc_cols);
        let EncoderWorkspace {
            x,
            norm,
            qkv,
            ctx,
            proj,
            up,
            down,
            scores,
            acc,
            ln_params,
            mask,
        } = ws;

        // + embed_positions, before the first LayerNorm: HF adds the position
        // to the convolutional output, not after the first norm.
        x.copy_from_slice(&conv_out[..n_src * d]);
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

        let threads = parallel::threads();
        for (li, layer) in self.layers.iter().enumerate() {
            let names = &self.gemm_names[li];
            // ---- self-attention, pre-LN ----
            layernorm_seam(
                eltwise.as_deref_mut(),
                ln_on,
                &mut ln_params[..2 * d],
                &layer.ln1_gamma,
                &layer.ln1_beta,
                x,
                norm,
                n_src,
                d,
                g.ln_eps,
            )?;
            gemm_seam(
                array.as_deref_mut(),
                gemm_on,
                &names[0],
                norm,
                n_src,
                &layer.qkv,
                d,
                3 * d,
                &layer.qkv_bias,
                qkv,
                threads,
            )?;
            if attn_on {
                if let Some(backend) = array.as_deref_mut() {
                    let shape = AttentionShape {
                        batch: 1,
                        seq: n_src,
                        heads: g.heads,
                        head_dim: g.head_dim,
                        hidden: d,
                    };
                    let softmax = if sm_on { eltwise.as_deref_mut() } else { None };
                    backend
                        .attention(qkv, shape, &mask[..n_src], softmax, ctx)
                        .map_err(|e| e.to_string())?;
                }
            } else if sm_on {
                if let Some(softmax) = eltwise.as_deref_mut() {
                    attention_eltwise_into(
                        qkv,
                        3 * d,
                        &qkv[d..],
                        3 * d,
                        n_src,
                        n_src,
                        d,
                        g.heads,
                        g.head_dim,
                        scale,
                        ctx,
                        &mut scores[..n_src * g.heads * sc_cols],
                        sc_cols,
                        acc,
                        softmax,
                    )?;
                }
            } else {
                attention_into(
                    qkv,
                    3 * d,
                    &qkv[d..],
                    3 * d,
                    n_src,
                    n_src,
                    d,
                    g.heads,
                    g.head_dim,
                    scale,
                    ctx,
                    scores,
                    acc,
                );
            }
            gemm_seam(
                array.as_deref_mut(),
                gemm_on,
                &names[1],
                ctx,
                n_src,
                &layer.attn_out,
                d,
                d,
                &layer.attn_out_bias,
                proj,
                threads,
            )?;
            kernels::add_inplace(x, &proj[..n_src * d]);

            // ---- feed-forward, pre-LN ----
            layernorm_seam(
                eltwise.as_deref_mut(),
                ln_on,
                &mut ln_params[..2 * d],
                &layer.ln2_gamma,
                &layer.ln2_beta,
                x,
                norm,
                n_src,
                d,
                g.ln_eps,
            )?;
            gemm_seam(
                array.as_deref_mut(),
                gemm_on,
                &names[2],
                norm,
                n_src,
                &layer.ffn_up,
                d,
                g.enc_intermediate,
                &layer.ffn_up_bias,
                up,
                threads,
            )?;
            if gelu_on {
                if let Some(backend) = eltwise.as_deref_mut() {
                    backend
                        .gelu(&mut up[..n_src * g.enc_intermediate])
                        .map_err(|e| e.to_string())?;
                }
            } else {
                gelu_erf_rows(&mut up[..n_src * g.enc_intermediate]);
            }
            gemm_seam(
                array.as_deref_mut(),
                gemm_on,
                &names[3],
                up,
                n_src,
                &layer.ffn_down,
                g.enc_intermediate,
                d,
                &layer.ffn_down_bias,
                down,
                threads,
            )?;
            kernels::add_inplace(x, &down[..n_src * d]);
        }

        // Whisper's encoder output is AFTER the final LayerNorm -- that is
        // what transformers hands the decoder.
        if ln_on {
            if let Some(backend) = eltwise.as_deref_mut() {
                ln_params[..d].copy_from_slice(&self.final_gamma);
                ln_params[d..2 * d].copy_from_slice(&self.final_beta);
                backend
                    .layernorm(&ln_params[..2 * d], &mut x[..n_src * d], n_src)
                    .map_err(|e| e.to_string())?;
            }
        } else {
            layernorm_rows_inplace(x, n_src, d, &self.final_gamma, &self.final_beta, g.ln_eps);
        }
        Ok(x)
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

/// Per-step scratch for the decoder.
///
/// A step runs several layers back to back, each with the same set of
/// intermediate widths. One set of buffers serves every layer and every
/// step: each is overwritten before it is read, so there is nothing to
/// clear between calls. This struct is private because the decoder owns it
/// -- unlike the encoder, the decoder already takes `&mut self`.
struct DecoderScratch {
    h: Vec<f32>,
    norm: Vec<f32>,
    qkv: Vec<f32>,
    proj: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    ctx: Vec<f32>,
    scores: Vec<f32>,
    acc: Vec<f32>,
    logits: Vec<f32>,
    /// gamma|beta for the array's LayerNorm, and the array attention's mask.
    ln_params: Vec<f32>,
    mask: Vec<f32>,
    /// Q|K|V rows and contexts assembled for the array's attention.
    attn_qkv: Vec<f32>,
    attn_ctx: Vec<f32>,
}

impl DecoderScratch {
    fn new(g: &Geometry) -> Self {
        let d = g.d_model;
        Self {
            h: vec![0.0f32; d],
            norm: vec![0.0f32; d],
            qkv: vec![0.0f32; 3 * d],
            proj: vec![0.0f32; d],
            up: vec![0.0f32; g.dec_intermediate],
            down: vec![0.0f32; d],
            ctx: vec![0.0f32; d],
            // Self-attention sees at most max_target positions; cross-
            // attention sees at most the encoder's position count.
            scores: vec![0.0f32; g.max_target.max(g.max_seq)],
            acc: vec![0.0f32; g.head_dim],
            logits: vec![0.0f32; g.vocab],
            ln_params: vec![0.0f32; 2 * g.d_model],
            mask: vec![0.0f32; g.max_target.max(g.max_seq)],
            attn_qkv: vec![0.0f32; g.max_target.max(g.max_seq) * 3 * g.d_model],
            attn_ctx: vec![0.0f32; g.max_target.max(g.max_seq) * g.d_model],
        }
    }
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
    /// The decoder's projection names, in the order a step dispatches them.
    gemm_names: Vec<[String; 6]>,
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
    /// Scratch buffers reused by every step.
    scratch: DecoderScratch,
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
        let mut gemm_names = Vec::with_capacity(geometry.dec_layers);
        for l in 0..geometry.dec_layers {
            let p = format!("decoder.layers.{l}.");
            gemm_names.push([
                format!("{p}self_qkv"),
                format!("{p}self_attn_out"),
                format!("{p}cross_q"),
                format!("{p}cross_attn_out"),
                format!("{p}ffn_up"),
                format!("{p}ffn_down"),
            ]);
            layers.push(DecoderLayer {
                self_qkv: operand(&container, &format!("{p}self_qkv"))?,
                self_qkv_bias: tensor(&container, &format!("{p}self_qkv.bias"))?,
                self_attn_out: operand(&container, &format!("{p}self_attn_out"))?,
                self_attn_out_bias: tensor(&container, &format!("{p}self_attn_out.bias"))?,
                cross_q: operand(&container, &format!("{p}cross_q"))?,
                cross_q_bias: tensor(&container, &format!("{p}cross_q.bias"))?,
                cross_kv: operand(&container, &format!("{p}cross_kv"))?,
                cross_kv_bias: tensor(&container, &format!("{p}cross_kv.bias"))?,
                cross_attn_out: operand(&container, &format!("{p}cross_attn_out"))?,
                cross_attn_out_bias: tensor(&container, &format!("{p}cross_attn_out.bias"))?,
                ffn_up: operand(&container, &format!("{p}ffn_up"))?,
                ffn_up_bias: tensor(&container, &format!("{p}ffn_up.bias"))?,
                ffn_down: operand(&container, &format!("{p}ffn_down"))?,
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
        let d = geometry.d_model;
        Ok(Decoder {
            embed_positions: tensor(&container, "decoder.embed_positions")?,
            embed_tokens: operand(&container, "decoder.embed_tokens")?,
            final_gamma: tensor(&container, "decoder.layer_norm.weight")?,
            final_beta: tensor(&container, "decoder.layer_norm.bias")?,
            cross_kv: vec![Vec::new(); geometry.dec_layers],
            // Reserve the whole self-attention cache once. Appending a row per
            // token would otherwise reallocate (and copy) every layer's cache
            // O(log max_target) times during autoregression. `reset()` clears
            // without dropping the capacity, so a second transcription reuses
            // the same allocation.
            self_kv: (0..geometry.dec_layers)
                .map(|_| Vec::with_capacity(geometry.max_target * 2 * d))
                .collect(),
            n_cached: 0,
            n_src: 0,
            n_generated: 0,
            suppress,
            begin_suppress,
            scratch: DecoderScratch::new(&geometry),
            layers,
            gemm_names,
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
        // The cross-attention score row is as wide as the source. The
        // geometry-sized scratch normally covers it, but growing it here
        // keeps an unusual caller (set_source with more rows than max_seq)
        // from tripping over a short buffer at step time.
        if self.scratch.scores.len() < n_src {
            self.scratch.scores.resize(n_src, 0.0);
        }
        for (l, layer) in self.layers.iter().enumerate() {
            let cache = &mut self.cross_kv[l];
            cache.resize(n_src * 2 * d, 0.0);
            kernels::gemm_nt_rows(
                enc,
                n_src,
                &layer.cross_kv,
                d,
                2 * d,
                &layer.cross_kv_bias,
                cache,
                parallel::threads(),
            );
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
        mut array: ArrayTextRef<'_, '_>,
        mut eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
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

        // The array's softmax and attention normalise the DESIGN's padded key
        // count. A decoder step sees a variable number of keys (the cache
        // grows, and cross-attention sees the source), so the design must be
        // wide enough for the largest of them.
        let widest = self.n_src.max(g.max_target).max(self.n_cached + 1);
        let sc_cols = match eltwise.as_deref().and_then(|e| e.softmax_cols()) {
            Some(cols) => {
                if cols < widest {
                    return Err(format!(
                        "the softmax design's score row is {cols} wide, but this decoder sees {widest} keys"
                    ));
                }
                cols
            }
            None => widest,
        };
        let gemm_on = array.as_deref().is_some_and(|a| a.gemm_on_array());
        let attn_on = array.as_deref().is_some_and(|a| a.on_array());
        if attn_on {
            let attn_cols = array
                .as_deref()
                .and_then(|a| a.score_cols())
                .unwrap_or(widest);
            if attn_cols < widest {
                return Err(format!(
                    "the attention design's score row is {attn_cols} wide, but this decoder sees {widest} keys"
                ));
            }
            if let Some(cols) = eltwise.as_deref().and_then(|e| e.softmax_cols()) {
                if cols != attn_cols {
                    return Err(format!(
                        "the softmax design's score row is {cols} wide and the attention's is {attn_cols} wide"
                    ));
                }
            }
        }
        let ln_on = eltwise
            .as_deref()
            .is_some_and(|e| e.on_array(EltOp::LayerNorm));
        let gelu_on = eltwise.as_deref().is_some_and(|e| e.on_array(EltOp::Gelu));
        let sm_on = eltwise
            .as_deref()
            .is_some_and(|e| e.on_array(EltOp::Softmax));

        if self.scratch.scores.len() < g.heads * sc_cols {
            self.scratch.scores.resize(g.heads * sc_cols, 0.0);
        }

        let DecoderScratch {
            h,
            norm,
            qkv,
            proj,
            up,
            down,
            ctx,
            scores,
            acc,
            logits: scratch_logits,
            ln_params,
            mask,
            attn_qkv,
            attn_ctx,
        } = &mut self.scratch;

        // embed_tokens + embed_positions, no norm in between: HF's decoder
        // adds the position to the token embedding and enters the first
        // pre-LN block.
        let et = &self.embed_tokens[token as usize * d..][..d];
        let pe = &self.embed_positions[position * d..][..d];
        for j in 0..d {
            h[j] = et[j] + pe[j];
        }

        // A single-token GEMM falls back to the serial kernel inside
        // `gemm_nt_rows`; the width is still useful if a caller decodes a
        // batch, so go through the same seam as the encoder.
        let threads = parallel::threads();
        for (l, layer) in self.layers.iter().enumerate() {
            let names = &self.gemm_names[l];
            // ---- self-attention over the cache, one new row ----
            layernorm_seam(
                eltwise.as_deref_mut(),
                ln_on,
                &mut ln_params[..2 * d],
                &layer.ln1_gamma,
                &layer.ln1_beta,
                h,
                norm,
                1,
                d,
                g.ln_eps,
            )?;
            gemm_seam(
                array.as_deref_mut(),
                gemm_on,
                &names[0],
                norm,
                1,
                &layer.self_qkv,
                d,
                3 * d,
                &layer.self_qkv_bias,
                qkv,
                threads,
            )?;
            // The K|V half of the fused QKV joins the cache; the Q half is
            // this step's query.
            let cache = &mut self.self_kv[l];
            let n_kv = self.n_cached + 1;
            cache.extend_from_slice(&qkv[d..3 * d]);
            let softmax = if sm_on { eltwise.as_deref_mut() } else { None };
            decoder_attention(
                array.as_deref_mut(),
                attn_on,
                sm_on,
                &qkv[..d],
                cache,
                2 * d,
                n_kv,
                d,
                g.heads,
                g.head_dim,
                scale,
                &mut ctx[..d],
                &mut scores[..g.heads * sc_cols],
                sc_cols,
                acc,
                attn_qkv,
                attn_ctx,
                mask,
                softmax,
            )?;
            gemm_seam(
                array.as_deref_mut(),
                gemm_on,
                &names[1],
                ctx,
                1,
                &layer.self_attn_out,
                d,
                d,
                &layer.self_attn_out_bias,
                proj,
                threads,
            )?;
            kernels::add_inplace(&mut h[..d], &proj[..d]);

            // ---- cross-attention over the encoder output ----
            layernorm_seam(
                eltwise.as_deref_mut(),
                ln_on,
                &mut ln_params[..2 * d],
                &layer.ln2_gamma,
                &layer.ln2_beta,
                h,
                norm,
                1,
                d,
                g.ln_eps,
            )?;
            gemm_seam(
                array.as_deref_mut(),
                gemm_on,
                &names[2],
                norm,
                1,
                &layer.cross_q,
                d,
                d,
                &layer.cross_q_bias,
                qkv,
                threads,
            )?;
            // The Q row is all this step needs from cross_q: its K|V half was
            // folded into a separate operand precisely so the two could be
            // fed from different A operands.
            let softmax = if sm_on { eltwise.as_deref_mut() } else { None };
            decoder_attention(
                array.as_deref_mut(),
                attn_on,
                sm_on,
                &qkv[..d],
                &self.cross_kv[l],
                2 * d,
                self.n_src,
                d,
                g.heads,
                g.head_dim,
                scale,
                &mut ctx[..d],
                &mut scores[..g.heads * sc_cols],
                sc_cols,
                acc,
                attn_qkv,
                attn_ctx,
                mask,
                softmax,
            )?;
            gemm_seam(
                array.as_deref_mut(),
                gemm_on,
                &names[3],
                ctx,
                1,
                &layer.cross_attn_out,
                d,
                d,
                &layer.cross_attn_out_bias,
                proj,
                threads,
            )?;
            kernels::add_inplace(&mut h[..d], &proj[..d]);

            // ---- feed-forward ----
            layernorm_seam(
                eltwise.as_deref_mut(),
                ln_on,
                &mut ln_params[..2 * d],
                &layer.ln3_gamma,
                &layer.ln3_beta,
                h,
                norm,
                1,
                d,
                g.ln_eps,
            )?;
            gemm_seam(
                array.as_deref_mut(),
                gemm_on,
                &names[4],
                norm,
                1,
                &layer.ffn_up,
                d,
                g.dec_intermediate,
                &layer.ffn_up_bias,
                up,
                threads,
            )?;
            if gelu_on {
                if let Some(backend) = eltwise.as_deref_mut() {
                    backend
                        .gelu(&mut up[..g.dec_intermediate])
                        .map_err(|e| e.to_string())?;
                }
            } else {
                gelu_erf_rows(&mut up[..g.dec_intermediate]);
            }
            gemm_seam(
                array.as_deref_mut(),
                gemm_on,
                &names[5],
                up,
                1,
                &layer.ffn_down,
                g.dec_intermediate,
                d,
                &layer.ffn_down_bias,
                down,
                threads,
            )?;
            kernels::add_inplace(&mut h[..d], &down[..d]);
        }

        self.n_cached += 1;
        if ln_on {
            if let Some(backend) = eltwise.as_deref_mut() {
                ln_params[..d].copy_from_slice(&self.final_gamma);
                ln_params[d..2 * d].copy_from_slice(&self.final_beta);
                backend
                    .layernorm(&ln_params[..2 * d], &mut h[..d], 1)
                    .map_err(|e| e.to_string())?;
            }
        } else {
            layernorm_row(&mut h[..d], &self.final_gamma, &self.final_beta, g.ln_eps);
        }
        logits_from_into(
            &h[..d],
            &self.embed_tokens,
            d,
            g.vocab,
            &mut scratch_logits[..g.vocab],
        );
        let next = Self::apply_policy(
            &mut scratch_logits[..g.vocab],
            &self.suppress,
            &self.begin_suppress,
            self.n_generated,
        );
        if let Some(out) = hidden {
            out.resize(d, 0.0);
            out.copy_from_slice(&h[..d]);
        }
        if let Some(out) = logits {
            out.resize(g.vocab, 0.0);
            out.copy_from_slice(&scratch_logits[..g.vocab]);
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
    fn apply_policy(
        logits: &mut [f32],
        suppress: &[i32],
        begin_suppress: &[i32],
        n_generated: usize,
    ) -> i32 {
        let vocab = logits.len();
        for &id in suppress {
            if id >= 0 && (id as usize) < vocab {
                logits[id as usize] = -3.4e38f32;
            }
        }
        if n_generated == 0 {
            for &id in begin_suppress {
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
        mut array: ArrayTextRef<'_, '_>,
        mut eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
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
            next = self.step(
                token,
                i,
                None,
                None,
                array.as_deref_mut(),
                eltwise.as_deref_mut(),
            )?;
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
            out.push(self.step(
                token,
                position,
                None,
                None,
                array.as_deref_mut(),
                eltwise.as_deref_mut(),
            )?);
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
    // `logits_from_into` overwrites every entry, so no zero pass is needed.
    let mut out = kernels::scratch_f32(vocab);
    logits_from_into(h, embed_tokens, d, vocab, &mut out);
    out
}

/// The same projection, into a caller-owned buffer.
///
/// `out` must hold at least `vocab` floats; every entry of `out[..vocab]` is
/// overwritten. This is the form the decoder's step uses, so the 50k-entry
/// logits vector is not rebuilt on every token.
pub fn logits_from_into(h: &[f32], embed_tokens: &[f32], d: usize, vocab: usize, out: &mut [f32]) {
    let h = &h[..d];
    let out = &mut out[..vocab];
    let work = (vocab as i64) * (d as i64);
    let Some(chunk) = parallel::chunk_rows(vocab, work, parallel::threads()) else {
        for (v, slot) in out.iter_mut().enumerate() {
            *slot = kernels::dot8(h, &embed_tokens[v * d..][..d]);
        }
        return;
    };
    // Each vocabulary row is one independent dot product: splitting the rows
    // changes which worker computes a logit, never the reduction order inside
    // it, so the projection stays bit-identical to the serial one. `dot8`
    // keeps the first-eight-products / eight-FMA-lanes / `hsum8` tree.
    parallel::for_each_chunk_mut(out, chunk, |ci, part| {
        let first = ci * chunk;
        for (v, slot) in part.iter_mut().enumerate() {
            *slot = kernels::dot8(h, &embed_tokens[(first + v) * d..][..d]);
        }
    });
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

/// LayerNorm from one row into another, in one pass.
///
/// Identical arithmetic to [`layernorm_row`] -- the same f64 mean and
/// variance, the same single rounding, the same FMA epilogue -- but it reads
/// `src` while writing `dst`. The callers used to do `dst.copy_from_slice(src)`
/// and then normalise `dst`, which walked each row twice; the first walk was
/// pure overhead because every byte it copied was overwritten immediately.
#[inline]
pub fn layernorm_into(src: &[f32], dst: &mut [f32], gamma: &[f32], beta: &[f32], eps: f64) {
    let d = dst.len();
    let src = &src[..d];
    let mut sm = 0.0f64;
    for v in src.iter() {
        sm += *v as f64;
    }
    let mean = (sm / d as f64) as f32;
    let mut sv = 0.0f64;
    for v in src.iter() {
        let dd = *v as f64 - mean as f64;
        sv += dd * dd;
    }
    let var = (sv / d as f64) as f32;
    let inv = 1.0f32 / (var + eps as f32).sqrt();
    for (j, slot) in dst.iter_mut().enumerate() {
        *slot = ((src[j] - mean) * inv).mul_add(gamma[j], beta[j]);
    }
}

/// LayerNorm of `rows` rows from `src` into `dst`, split over the pool.
///
/// A row is normalised from its own values only, so the split crosses no
/// reduction and is bit-identical to running [`layernorm_into`] row by row.
pub fn layernorm_rows(
    src: &[f32],
    dst: &mut [f32],
    rows: usize,
    width: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f64,
) {
    let dst = &mut dst[..rows * width];
    let Some(chunk) =
        parallel::chunk_rows(rows, (rows as i64) * (width as i64), parallel::threads())
    else {
        for i in 0..rows {
            layernorm_into(
                &src[i * width..][..width],
                &mut dst[i * width..][..width],
                gamma,
                beta,
                eps,
            );
        }
        return;
    };
    parallel::for_each_chunk_mut(dst, chunk * width, |ci, part| {
        let first = ci * chunk;
        for (i, row) in part.chunks_exact_mut(width).enumerate() {
            layernorm_into(&src[(first + i) * width..][..width], row, gamma, beta, eps);
        }
    });
}

/// The in-place form of [`layernorm_rows`], for a stack's final norm.
pub fn layernorm_rows_inplace(
    x: &mut [f32],
    rows: usize,
    width: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f64,
) {
    let x = &mut x[..rows * width];
    let Some(chunk) =
        parallel::chunk_rows(rows, (rows as i64) * (width as i64), parallel::threads())
    else {
        for row in x.chunks_exact_mut(width) {
            layernorm_row(row, gamma, beta, eps);
        }
        return;
    };
    parallel::for_each_chunk_mut(x, chunk * width, |_, part| {
        for row in part.chunks_exact_mut(width) {
            layernorm_row(row, gamma, beta, eps);
        }
    });
}

/// Whisper's exact-erf GELU over a slice, split over the pool.
///
/// `gelu_erf_exact` evaluates `erf` in f64 and rounds once; that is far more
/// work per element than a polynomial GELU, so the 1.5 M elements of the
/// encoder's FFNs are worth splitting even though each one is independent.
pub fn gelu_erf_rows(x: &mut [f32]) {
    let Some(chunk) = parallel::chunk_rows(x.len(), x.len() as i64, parallel::threads()) else {
        for v in x.iter_mut() {
            *v = gelu_erf_exact(*v);
        }
        return;
    };
    parallel::for_each_chunk_mut(x, chunk, |_, part| {
        for v in part.iter_mut() {
            *v = gelu_erf_exact(*v);
        }
    });
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
    // Both are overwritten in full by `attention_rows` before any read.
    let mut scores = kernels::scratch_f32(n_kv);
    let mut acc = kernels::scratch_f32(head_dim);
    attention_into(
        q,
        q_stride,
        kv,
        kv_stride,
        n_q,
        n_kv,
        d_model,
        heads,
        head_dim,
        scale,
        out,
        &mut scores,
        &mut acc,
    );
}

/// The same attention, with the score row and the per-head accumulator
/// supplied by the caller.
///
/// `scores` must hold at least `n_kv` floats and `acc` at least `head_dim`.
/// Both are overwritten before being read. The encoder and the decoder call
/// this form so a head's accumulator is not a fresh allocation for every
/// query and every layer.
pub fn attention_into(
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
    scores: &mut [f32],
    acc: &mut [f32],
) {
    if n_q == 0 || n_kv == 0 || heads == 0 {
        return;
    }
    let out = &mut out[..n_q * d_model];
    let work = (n_q as i64) * (heads as i64) * (n_kv as i64) * (head_dim as i64) * 2;

    // OVER QUERY ROWS first. One query's scores and output depend on no other
    // query's, so a row split crosses no reduction: it is a choice of WHICH
    // worker computes a row, never of what the row contains. Each chunk owns
    // its score row and head accumulator, so workers cannot race on scratch.
    if let Some(chunk) = parallel::chunk_rows(n_q, work, parallel::threads()) {
        parallel::for_each_chunk_mut(out, chunk * d_model, |ci, part| {
            let first = ci * chunk;
            let rows = part.len() / d_model;
            let mut scores = kernels::scratch_f32(n_kv);
            let mut acc = kernels::scratch_f32(head_dim);
            attention_rows(
                q,
                q_stride,
                kv,
                kv_stride,
                first,
                rows,
                n_kv,
                d_model,
                0,
                heads,
                head_dim,
                scale,
                part,
                &mut scores,
                &mut acc,
            );
        });
        return;
    }

    // One query has no row to split, so split the HEADS: each head writes its
    // own `head_dim`-wide slice of the output row and reads only its own Q/K/V
    // columns. This is what makes single-token decode use more than one core.
    if n_q == 1 {
        let row = &mut out[..d_model];
        if let Some(chunk) = parallel::chunk_rows(heads, work, parallel::threads()) {
            parallel::for_each_chunk_mut(row, chunk * head_dim, |hi, part| {
                let first = hi * chunk;
                let count = part.len() / head_dim;
                let mut scores = kernels::scratch_f32(n_kv);
                let mut acc = kernels::scratch_f32(head_dim);
                attention_rows(
                    q,
                    q_stride,
                    kv,
                    kv_stride,
                    0,
                    1,
                    n_kv,
                    d_model,
                    first,
                    first + count,
                    head_dim,
                    scale,
                    part,
                    &mut scores,
                    &mut acc,
                );
            });
            return;
        }
    }

    // Too small for a worker: the barrier costs more than the arithmetic.
    // The caller's reusable scratch is used only on this path.
    attention_rows(
        q,
        q_stride,
        kv,
        kv_stride,
        0,
        n_q,
        n_kv,
        d_model,
        0,
        heads,
        head_dim,
        scale,
        out,
        &mut scores[..n_kv],
        &mut acc[..head_dim],
    );
}

/// The attention with the softmax on the array but its two GEMMs on the host.
///
/// The array's softmax normalises a row of the DESIGN's width -- the padded
/// key count -- so the scores have to be laid out at that width even though
/// the host computed them. Padding columns start at -1e30 and exponentiate to
/// zero. The host-only [`attention_into`] is untouched by this path.
#[allow(clippy::too_many_arguments)]
pub fn attention_eltwise_into(
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
    scores: &mut [f32],
    cols: usize,
    acc: &mut [f32],
    softmax: &mut (dyn ArrayEltwise + '_),
) -> Result<(), String> {
    if n_q == 0 || n_kv == 0 || heads == 0 {
        return Ok(());
    }
    if cols < n_kv {
        return Err(format!(
            "the softmax design's score row is {cols} wide, but this attention reads {n_kv} keys"
        ));
    }
    let rows = n_q * heads;
    for i in 0..n_q {
        for h in 0..heads {
            let row = &mut scores[(i * heads + h) * cols..][..cols];
            row.fill(-1e30);
            for (j, slot) in row[..n_kv].iter_mut().enumerate() {
                let qh = &q[i * q_stride + h * head_dim..][..head_dim];
                let kj = &kv[j * kv_stride + h * head_dim..][..head_dim];
                *slot = kernels::dot8(qh, kj) * scale;
            }
        }
    }
    softmax
        .softmax(&mut scores[..rows * cols], rows)
        .map_err(|e| e.to_string())?;
    let out = &mut out[..n_q * d_model];
    let acc = &mut acc[..head_dim];
    for i in 0..n_q {
        for h in 0..heads {
            let row = &scores[(i * heads + h) * cols..][..n_kv];
            let o = &mut out[i * d_model + h * head_dim..][..head_dim];
            acc.fill(0.0);
            for (j, &a) in row.iter().enumerate() {
                let vj = &kv[j * kv_stride + d_model + h * head_dim..][..head_dim];
                kernels::weighted_accumulate(acc, vj, a);
            }
            o.copy_from_slice(acc);
        }
    }
    Ok(())
}

/// One slice of attention: queries `i0..i0+rows`, heads `h0..h1`.
///
/// `out` is `rows` rows of `d_model` floats. The score dot product and the
/// softmax-weighted value accumulation go through the bit-exact eight-lane
/// helpers in [`kernels`], so the parallel path produces the same bytes the
/// old hand-rolled eight-lane loops did.
#[allow(clippy::too_many_arguments)]
fn attention_rows(
    q: &[f32],
    q_stride: usize,
    kv: &[f32],
    kv_stride: usize,
    i0: usize,
    rows: usize,
    n_kv: usize,
    d_model: usize,
    h0: usize,
    h1: usize,
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
    scores: &mut [f32],
    acc: &mut [f32],
) {
    let scores = &mut scores[..n_kv];
    let acc = &mut acc[..head_dim];
    for i in i0..i0 + rows {
        for h in h0..h1 {
            // QK^T, one row of scores at a time: eight lane accumulators with
            // an FMA per element and the reference's fixed `hsum8` tree.
            for (j, slot) in scores.iter_mut().enumerate() {
                let qh = &q[i * q_stride + h * head_dim..][..head_dim];
                let kj = &kv[j * kv_stride + h * head_dim..][..head_dim];
                *slot = kernels::dot8(qh, kj) * scale;
            }

            // Softmax with the maximum subtracted, the same reduction HF's
            // own attention does. The exponential is `libm`'s `exp` -- NOT
            // the BERT path's polynomial -- and the sum is a plain float32
            // accumulation in order.
            let mut mx = scores[0];
            for s in scores.iter().skip(1) {
                mx = mx.max(*s);
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            let inv = if sum > 0.0 { 1.0f32 / sum } else { 0.0f32 };

            // softmax.V, into this head's slice of the output row. One
            // accumulator per eight of the head's width -- the reference's
            // `__m256 acc[NV]`, and not one accumulator for the whole head.
            // The difference is the ORDER of the additions, and the order is
            // the answer.
            let o = &mut out[(i - i0) * d_model + h * head_dim..][..head_dim];
            acc.fill(0.0);
            for (j, &aj) in scores.iter().enumerate() {
                let a = aj * inv;
                let vj = &kv[j * kv_stride + d_model + h * head_dim..][..head_dim];
                kernels::weighted_accumulate(acc, vj, a);
            }
            o.copy_from_slice(acc);
        }
    }
}

/// A GEMM operand: F32 rows in a host container, de-tiled BF16 panels in one
/// packed for the array. The same forward pass runs on both.
fn operand(container: &Container, name: &str) -> Result<Vec<f32>, String> {
    super::operand::read_operand(container, name)
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

#[cfg(test)]
mod tests {
    use super::*;
    use npum_model::array::{ArrayAttention, ArrayGemm};

    /// Deterministic, non-trivial f32 data in [-1, 1].
    fn noise(len: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761) ^ 0x9e37_79b9;
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn geometry() -> Geometry {
        Geometry {
            d_model: 16,
            heads: 2,
            head_dim: 8,
            enc_layers: 1,
            dec_layers: 1,
            enc_intermediate: 32,
            dec_intermediate: 32,
            max_seq: 8,
            max_target: 8,
            vocab: 8,
            mel_bins: 80,
            hop_length: 160,
            ln_eps: 1e-5,
            attn_scale: 0.353_553_4,
            qkv_scale_folded: true,
        }
    }

    #[test]
    fn logits_into_matches_fresh() {
        let (d, vocab) = (16, 8);
        let h = noise(d, 3);
        let embed = noise(vocab * d, 4);
        let want = logits_from(&h, &embed, d, vocab);
        // A longer buffer: only [..vocab] may be touched.
        let mut out = vec![f32::NAN; vocab + 3];
        logits_from_into(&h, &embed, d, vocab, &mut out);
        assert_eq!(&out[..vocab], want.as_slice());
        assert!(out[vocab..].iter().all(|v| v.is_nan()));
    }

    #[test]
    fn attention_into_matches_fresh() {
        let (d, heads, head_dim) = (16, 2, 8);
        let (n_q, n_kv) = (3, 4);
        let (q_stride, kv_stride) = (3 * d, 2 * d);
        let q = noise((n_q - 1) * q_stride + heads * head_dim, 5);
        let kv = noise((n_kv - 1) * kv_stride + 2 * d, 6);
        let mut want = vec![0.0f32; n_q * d];
        attention(
            &q, q_stride, &kv, kv_stride, n_q, n_kv, d, heads, head_dim, 0.25, &mut want,
        );
        // Reused scratch, deliberately poisoned before every call.
        let mut scores = vec![f32::NAN; n_kv];
        let mut acc = vec![f32::NAN; head_dim];
        for _ in 0..3 {
            scores.fill(f32::NAN);
            acc.fill(f32::NAN);
            let mut got = vec![0.0f32; n_q * d];
            attention_into(
                &q,
                q_stride,
                &kv,
                kv_stride,
                n_q,
                n_kv,
                d,
                heads,
                head_dim,
                0.25,
                &mut got,
                &mut scores,
                &mut acc,
            );
            assert_eq!(got, want);
        }
    }

    fn encoder_layer(g: &Geometry, seed: u32) -> EncoderLayer {
        let d = g.d_model;
        let f = g.enc_intermediate;
        EncoderLayer {
            qkv: noise(d * 3 * d, seed),
            qkv_bias: noise(3 * d, seed + 1),
            attn_out: noise(d * d, seed + 2),
            attn_out_bias: noise(d, seed + 3),
            ffn_up: noise(d * f, seed + 4),
            ffn_up_bias: noise(f, seed + 5),
            ffn_down: noise(f * d, seed + 6),
            ffn_down_bias: noise(d, seed + 7),
            ln1_gamma: noise(d, seed + 8).iter().map(|v| v.abs() + 0.5).collect(),
            ln1_beta: noise(d, seed + 9),
            ln2_gamma: noise(d, seed + 10).iter().map(|v| v.abs() + 0.5).collect(),
            ln2_beta: noise(d, seed + 11),
        }
    }

    #[test]
    fn encoder_workspace_matches_fresh() {
        let g = geometry();
        let d = g.d_model;
        let encoder = Encoder {
            layers: vec![encoder_layer(&g, 10)],
            gemm_names: vec![[
                "encoder.layers.0.qkv".into(),
                "encoder.layers.0.attn_out".into(),
                "encoder.layers.0.ffn_up".into(),
                "encoder.layers.0.ffn_down".into(),
            ]],
            geometry: g.clone(),
            embed_positions: noise(g.max_seq * d, 20),
            final_gamma: noise(d, 21).iter().map(|v| v.abs() + 0.5).collect(),
            final_beta: noise(d, 22),
        };
        let conv = noise(g.max_seq * d, 30);
        let mut ws = EncoderWorkspace::new();
        // Reuse at the same shape, then at a smaller one (the buffers shrink),
        // then back to the first shape.
        for &n_src in &[8usize, 8, 4, 8] {
            let want = encoder.run(&conv[..n_src * d], n_src, None, None).unwrap();
            let got = encoder
                .run_into(&conv[..n_src * d], n_src, &mut ws, None, None)
                .unwrap();
            assert_eq!(got, want.as_slice(), "n_src = {n_src}");
        }
    }

    fn decoder_layer(g: &Geometry, seed: u32) -> DecoderLayer {
        let d = g.d_model;
        let f = g.dec_intermediate;
        DecoderLayer {
            self_qkv: noise(d * 3 * d, seed),
            self_qkv_bias: noise(3 * d, seed + 1),
            self_attn_out: noise(d * d, seed + 2),
            self_attn_out_bias: noise(d, seed + 3),
            cross_q: noise(d * d, seed + 4),
            cross_q_bias: noise(d, seed + 5),
            cross_kv: noise(d * 2 * d, seed + 6),
            cross_kv_bias: noise(2 * d, seed + 7),
            cross_attn_out: noise(d * d, seed + 8),
            cross_attn_out_bias: noise(d, seed + 9),
            ffn_up: noise(d * f, seed + 10),
            ffn_up_bias: noise(f, seed + 11),
            ffn_down: noise(f * d, seed + 12),
            ffn_down_bias: noise(d, seed + 13),
            ln1_gamma: noise(d, seed + 14).iter().map(|v| v.abs() + 0.5).collect(),
            ln1_beta: noise(d, seed + 15),
            ln2_gamma: noise(d, seed + 16).iter().map(|v| v.abs() + 0.5).collect(),
            ln2_beta: noise(d, seed + 17),
            ln3_gamma: noise(d, seed + 18).iter().map(|v| v.abs() + 0.5).collect(),
            ln3_beta: noise(d, seed + 19),
        }
    }

    fn decoder(g: &Geometry) -> Decoder {
        let d = g.d_model;
        Decoder {
            geometry: g.clone(),
            layers: vec![decoder_layer(g, 40)],
            gemm_names: vec![[
                "decoder.layers.0.self_qkv".into(),
                "decoder.layers.0.self_attn_out".into(),
                "decoder.layers.0.cross_q".into(),
                "decoder.layers.0.cross_attn_out".into(),
                "decoder.layers.0.ffn_up".into(),
                "decoder.layers.0.ffn_down".into(),
            ]],
            embed_positions: noise(g.max_target * d, 50),
            embed_tokens: noise(g.vocab * d, 51),
            final_gamma: noise(d, 52).iter().map(|v| v.abs() + 0.5).collect(),
            final_beta: noise(d, 53),
            cross_kv: vec![Vec::new()],
            self_kv: vec![Vec::with_capacity(g.max_target * 2 * d)],
            n_cached: 0,
            n_src: 0,
            n_generated: 0,
            suppress: vec![1],
            begin_suppress: vec![2],
            scratch: DecoderScratch::new(g),
        }
    }

    #[test]
    fn decoder_scratch_matches_fresh() {
        let g = geometry();
        let d = g.d_model;
        let enc = noise(g.max_seq * d, 60);
        let mut reused = decoder(&g);
        let mut fresh = decoder(&g);
        reused.set_source(&enc, g.max_seq).unwrap();
        fresh.set_source(&enc, g.max_seq).unwrap();

        for pos in 0..4 {
            // The reference decoder gets a brand-new scratch every step.
            fresh.scratch = DecoderScratch::new(&g);
            let token = (pos as i32 + 3) % g.vocab as i32;
            let mut h_fresh = Vec::new();
            let mut lg_fresh = Vec::new();
            let next_fresh = fresh
                .step(
                    token,
                    pos,
                    Some(&mut h_fresh),
                    Some(&mut lg_fresh),
                    None,
                    None,
                )
                .unwrap();
            let mut h_reused = Vec::new();
            let mut lg_reused = Vec::new();
            let next_reused = reused
                .step(
                    token,
                    pos,
                    Some(&mut h_reused),
                    Some(&mut lg_reused),
                    None,
                    None,
                )
                .unwrap();
            assert_eq!(next_reused, next_fresh, "position {pos}");
            assert_eq!(h_reused, h_fresh, "hidden at position {pos}");
            assert_eq!(lg_reused, lg_fresh, "logits at position {pos}");
        }
    }

    use std::cell::RefCell;
    use std::rc::Rc;

    /// What the family asked the array to do, recorded by the mocks below.
    #[derive(Default)]
    struct SeamLog {
        gemms: Vec<String>,
        attention: usize,
        ln: usize,
        gelu: usize,
        softmax: usize,
    }

    struct MockText {
        log: Rc<RefCell<SeamLog>>,
        cols: usize,
    }

    impl ArrayGemm for MockText {
        fn name(&self) -> &'static str {
            "mock-array"
        }
        fn gemm_on_array(&self) -> bool {
            true
        }
        fn dispatches(&self) -> usize {
            self.log.borrow().gemms.len()
        }
        fn staged_bytes(&self) -> usize {
            0
        }
        fn gemm(
            &mut self,
            weight: &str,
            _a: &[f32],
            _rows: usize,
            _k: usize,
            _n: usize,
            _bias: Option<&[f32]>,
            out: &mut [f32],
        ) -> npum_model::array::Result<()> {
            self.log.borrow_mut().gemms.push(weight.to_string());
            out.fill(0.0);
            Ok(())
        }
    }

    impl ArrayAttention for MockText {
        fn name(&self) -> &'static str {
            "mock-array"
        }
        fn on_array(&self) -> bool {
            true
        }
        fn score_cols(&self) -> Option<usize> {
            Some(self.cols)
        }
        fn attn_dispatches(&self) -> usize {
            self.log.borrow().attention
        }
        fn attn_seconds(&self) -> (f64, f64) {
            (0.0, 0.0)
        }
        fn attention(
            &mut self,
            _qkv: &[f32],
            _shape: AttentionShape,
            _mask: &[f32],
            _softmax: Option<&mut (dyn ArrayEltwise + '_)>,
            ctx: &mut [f32],
        ) -> npum_model::array::Result<()> {
            self.log.borrow_mut().attention += 1;
            ctx.fill(0.0);
            Ok(())
        }
    }

    impl ArrayText for MockText {}
use npum_model::array::ArrayFrontend;
    impl ArrayFrontend for MockText {}

    struct MockEltwise {
        log: Rc<RefCell<SeamLog>>,
        cols: usize,
    }

    impl ArrayEltwise for MockEltwise {
        fn on_array(&self, _op: EltOp) -> bool {
            true
        }
        fn layernorm(
            &mut self,
            _params: &[f32],
            _x: &mut [f32],
            _n_rows: usize,
        ) -> npum_model::array::Result<()> {
            self.log.borrow_mut().ln += 1;
            Ok(())
        }
        fn gelu(&mut self, _x: &mut [f32]) -> npum_model::array::Result<()> {
            self.log.borrow_mut().gelu += 1;
            Ok(())
        }
        fn softmax(&mut self, _x: &mut [f32], _n_rows: usize) -> npum_model::array::Result<()> {
            self.log.borrow_mut().softmax += 1;
            Ok(())
        }
        fn softmax_cols(&self) -> Option<usize> {
            Some(self.cols)
        }
        fn dispatches(&self) -> usize {
            let l = self.log.borrow();
            l.ln + l.gelu + l.softmax
        }
    }

    #[test]
    fn the_encoder_dispatches_the_dense_seams() {
        let g = geometry();
        let d = g.d_model;
        let encoder = Encoder {
            layers: vec![encoder_layer(&g, 10)],
            gemm_names: vec![[
                "encoder.layers.0.qkv".into(),
                "encoder.layers.0.attn_out".into(),
                "encoder.layers.0.ffn_up".into(),
                "encoder.layers.0.ffn_down".into(),
            ]],
            geometry: g.clone(),
            embed_positions: noise(g.max_seq * d, 20),
            final_gamma: noise(d, 21).iter().map(|v| v.abs() + 0.5).collect(),
            final_beta: noise(d, 22),
        };
        let conv = noise(g.max_seq * d, 30);
        let log = Rc::new(RefCell::new(SeamLog::default()));
        let mut text = MockText {
            log: log.clone(),
            cols: g.max_seq,
        };
        let mut eltwise = MockEltwise {
            log: log.clone(),
            cols: g.max_seq,
        };

        encoder
            .run(&conv, g.max_seq, Some(&mut text), Some(&mut eltwise))
            .expect("the array path runs");

        let recorded = log.borrow();
        assert_eq!(
            recorded.gemms,
            [
                "encoder.layers.0.qkv",
                "encoder.layers.0.attn_out",
                "encoder.layers.0.ffn_up",
                "encoder.layers.0.ffn_down"
            ]
        );
        assert_eq!(recorded.attention, 1);
        assert!(recorded.ln >= 3, "ln1, ln2 and the final norm");
        assert_eq!(recorded.gelu, 1);
        // The array's attention owns the softmax; the eltwise softmax is not
        // dispatched a second time.
        assert_eq!(recorded.softmax, 0);
    }
}
