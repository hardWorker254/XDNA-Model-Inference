//! The ViT: the patch embedding, the pre-LN encoder and the classifier
//! head.
//!
//! # WHAT IS DIFFERENT FROM BERT, AND FROM WHISPER
//!
//! * the INPUT is pixels turned into patches, not tokens: the front end
//!   produces a `[n_patches, patch_dim]` matrix and the first GEMM is the
//!   patch embedding;
//! * the ORDER is pre-LN -- residual, LayerNorm, attention, add -- like
//!   Whisper's and not like BERT's;
//! * the LayerNorm is the DOUBLE-accumulating one (the same function
//!   Whisper's encoder uses), and the epsilon is the CONTAINER's;
//! * the attention has NO MASK: a ViT attends over every position it was
//!   given, and every position is a real one;
//! * the attention's scale is FOLDED into Q's weight and Q's bias, so
//!   applying it again would be applying it twice -- which is a different
//!   model;
//! * the output is a LABEL and a probability, not a vector.
//!
//! # THE FINAL LAYERNORM IS NOT A DETAIL
//!
//! transformers' ViTModel applies a LayerNorm after the encoder, and that
//! normalised CLS row is what the head consumes. Skipping it changes the
//! logits by a scale, which changes which label wins, without changing any
//! shape -- so it is not a detail.

use crate::kernels;
use crate::whisper::attention;
use crate::whisper::layernorm_row;
use crate::whisper_features::gelu_erf_exact;
use crate::vit_image::Resample;
use npum_format::Container;
use std::path::Path;

/// The ViT's geometry, as the container states it.
#[derive(Debug, Clone)]
pub struct Geometry {
    /// The container's name.
    pub model_name: String,
    /// The model's width.
    pub hidden: i64,
    /// How many layers.
    pub layers: i64,
    /// How many heads.
    pub heads: i64,
    /// One head's width.
    pub head_dim: i64,
    /// The input's side, in pixels.
    pub image_size: i64,
    /// The patch's side, in pixels.
    pub patch_size: i64,
    /// The positions, the patches and the CLS row.
    pub n_pos: i64,
    /// How many patches there are.
    pub n_patches: i64,
    /// The patch's width: the channels times the patch's area.
    pub patch_dim: i64,
    /// How many labels.
    pub num_labels: i64,
    /// The layer norm's epsilon.
    pub ln_eps: f64,
    /// The attention's scale, when it is not folded.
    pub attn_scale: f64,
    /// Whether the scale is already folded into Q's weight and bias.
    pub qkv_scale_folded: bool,
    /// The per-channel means.
    pub mean: Vec<f32>,
    /// The per-channel standard deviations.
    pub std_dev: Vec<f32>,
    /// The resample code.
    pub resample: Resample,
    /// How many channels.
    pub num_channels: i64,
}

impl Geometry {
    /// Read the geometry out of a container.
    ///
    /// Every key is REQUIRED except the ones with a measured default: a
    /// container that does not carry the geometry is refused by the C++'s
    /// reader, and a default here would be a classifier that answers about
    /// a model nobody chose.
    pub fn read(container: &Container, path: &Path) -> Option<Geometry> {
        let config = container.config();
        let model_name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let mean = config
            .get("image_mean")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_f64()).collect::<Vec<f64>>())?;
        let std_dev = config
            .get("image_std")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_f64()).collect::<Vec<f64>>())?;
        let resample = Resample::from_code(config.get_i64("resample")?)?;
        Some(Geometry {
            model_name,
            hidden: config.get_i64("hidden")?,
            layers: config.get_i64("num_layers")?,
            heads: config.get_i64("num_heads")?,
            head_dim: config.get_i64("head_dim")?,
            image_size: config.get_i64("image_size")?,
            patch_size: config.get_i64("patch_size")?,
            n_pos: config.get_i64("max_seq_len")?,
            n_patches: config.get_i64("n_patches")?,
            patch_dim: config.get_i64("patch_dim")?,
            num_labels: config.get_i64("num_labels")?,
            ln_eps: config.get_f64("layer_norm_eps").unwrap_or(1e-12),
            attn_scale: config.get_f64("attention_scale").unwrap_or(0.125),
            qkv_scale_folded: config.get_str("qkv_scale_folded") != Some("false"),
            mean: mean.into_iter().map(|v| v as f32).collect(),
            std_dev: std_dev.into_iter().map(|v| v as f32).collect(),
            resample,
            num_channels: config.get_i64("num_channels").unwrap_or(3),
        })
    }
}

/// One layer's weights.
struct Layer {
    ln1_gamma: Vec<f32>,
    ln1_beta: Vec<f32>,
    qkv: Vec<f32>,
    qkv_bias: Vec<f32>,
    attn_out: Vec<f32>,
    attn_out_bias: Vec<f32>,
    ln2_gamma: Vec<f32>,
    ln2_beta: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_up_bias: Vec<f32>,
    ffn_down: Vec<f32>,
    ffn_down_bias: Vec<f32>,
}

/// The ViT: the encoder stack and the head.
pub struct Vit {
    geometry: Geometry,
    layers: Vec<Layer>,
    positions: Vec<f32>,
    cls_token: Vec<f32>,
    patch_embed: Vec<f32>,
    patch_embed_bias: Vec<f32>,
    final_gamma: Vec<f32>,
    final_beta: Vec<f32>,
    head: Vec<f32>,
    head_bias: Vec<f32>,
    labels: Vec<String>,
}

impl Vit {
    /// Read the weights out of a container.
    pub fn open(path: &Path) -> Result<Vit, String> {
        let container = Container::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let geometry = Geometry::read(&container, path)
            .ok_or_else(|| format!("{}: the container does not carry the ViT's geometry", path.display()))?;
        let mut layers = Vec::with_capacity(geometry.layers as usize);
        for l in 0..geometry.layers {
            let p = format!("layer.{l}.");
            layers.push(Layer {
                ln1_gamma: tensor(&container, &format!("{p}ln1.weight"))?,
                ln1_beta: tensor(&container, &format!("{p}ln1.bias"))?,
                qkv: tensor(&container, &format!("{p}qkv"))?,
                qkv_bias: tensor(&container, &format!("{p}qkv.bias"))?,
                attn_out: tensor(&container, &format!("{p}attn_out"))?,
                attn_out_bias: tensor(&container, &format!("{p}attn_out.bias"))?,
                ln2_gamma: tensor(&container, &format!("{p}ln2.weight"))?,
                ln2_beta: tensor(&container, &format!("{p}ln2.bias"))?,
                ffn_up: tensor(&container, &format!("{p}ffn_up"))?,
                ffn_up_bias: tensor(&container, &format!("{p}ffn_up.bias"))?,
                ffn_down: tensor(&container, &format!("{p}ffn_down"))?,
                ffn_down_bias: tensor(&container, &format!("{p}ffn_down.bias"))?,
            });
        }
        // The labels: the vocabulary the head's OUTPUTS are named by. A
        // head with more outputs than names would print label 999's weight
        // as if it were all of them -- a finite, confident, wrong answer.
        let labels: Vec<String> = container
            .tensor_bytes("labels.table")
            .map(|b| {
                String::from_utf8_lossy(b)
                    .split('\n')
                    .filter(|l| !l.is_empty())
                    .map(|l| l.to_string())
                    .collect()
            })
            .unwrap_or_default();
        if labels.len() as i64 != geometry.num_labels {
            return Err(format!(
                "{}: labels.table has {} names and the head has {} outputs. \
                 Printing label {} as '{}' would name a class that is not the one \
                 the argmax chose.",
                path.display(),
                labels.len(),
                geometry.num_labels,
                labels.len().saturating_sub(1),
                labels.last().map(|l| l.as_str()).unwrap_or("")
            ));
        }
        Ok(Vit {
            positions: tensor(&container, "frontend.position_embeddings")?,
            cls_token: tensor(&container, "frontend.cls_token")?,
            patch_embed: tensor(&container, "frontend.patch_embed")?,
            patch_embed_bias: tensor(&container, "frontend.patch_embed.bias")?,
            final_gamma: tensor(&container, "layernorm.weight")?,
            final_beta: tensor(&container, "layernorm.bias")?,
            head: tensor(&container, "classifier.weight")?,
            head_bias: tensor(&container, "classifier.bias")?,
            layers,
            geometry,
            labels,
        })
    }

    /// The model's geometry.
    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    /// The labels, in the head's order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// The encoder: the patches to the normalised positions.
    ///
    /// `patches` is the front end's `[n_patches, patch_dim]` matrix.
    pub fn encode(&self, patches: &[f32]) -> Vec<f32> {
        let g = &self.geometry;
        let d = g.hidden as usize;
        let n_pos = g.n_pos as usize;
        let n_patches = g.n_patches as usize;

        // The patch embedding: a conv rewritten as an exact GEMM, because
        // the stride equals the kernel, so im2col is a permutation.
        let mut patch_out = vec![0.0f32; n_patches * d];
        kernels::gemm_nt(
            patches,
            n_patches,
            &self.patch_embed,
            g.patch_dim as usize,
            d,
            &self.patch_embed_bias,
            &mut patch_out,
        );

        // The CLS row and the position table. Positions are added AFTER
        // the concatenation and BEFORE the first LayerNorm, which is HF's
        // order: pos_embeddings[:, 1:] + patch, then cat((cls, patches)),
        // then + pos_embeddings[:, :1]. Same sum, same tensor, and the CLS
        // row therefore holds position 0 like any other.
        let mut x = vec![0.0f32; n_pos * d];
        x[..d].copy_from_slice(&self.cls_token[..d]);
        for p in 0..n_patches {
            x[(p + 1) * d..(p + 2) * d].copy_from_slice(&patch_out[p * d..(p + 1) * d]);
        }
        for t in 0..n_pos {
            let pe = &self.positions[t * d..(t + 1) * d];
            let row = &mut x[t * d..(t + 1) * d];
            for j in 0..d {
                row[j] += pe[j];
            }
        }

        // The scale is inside the Q weight and the Q bias unless the
        // container was packed with --no-fold-scale. Applying it here as
        // well would be applying it twice, which is a different model.
        let scale = if g.qkv_scale_folded {
            1.0f32
        } else {
            g.attn_scale as f32
        };
        let inter = g.hidden as usize; // placeholder, replaced below
        let _ = inter;

        for layer in &self.layers {
            // ---- self-attention, pre-LN ----
            let mut norm = x.clone();
            for row in norm.chunks_exact_mut(d) {
                layernorm_row(row, &layer.ln1_gamma, &layer.ln1_beta, g.ln_eps);
            }
            let mut qkv = vec![0.0f32; n_pos * 3 * d];
            kernels::gemm_nt(&norm, n_pos, &layer.qkv, d, 3 * d, &layer.qkv_bias, &mut qkv);
            // The fused [Q|K|V] row is Q at offset 0 and a K|V block at
            // offset d, both with row stride 3*d_model -- the same layout
            // the Whisper's encoder reads. There is NO mask.
            let mut ctx = vec![0.0f32; n_pos * d];
            attention(
                &qkv,
                3 * d,
                &qkv[d..],
                3 * d,
                n_pos,
                n_pos,
                d,
                g.heads as usize,
                g.head_dim as usize,
                scale,
                &mut ctx,
            );
            let mut proj = vec![0.0f32; n_pos * d];
            kernels::gemm_nt(&ctx, n_pos, &layer.attn_out, d, d, &layer.attn_out_bias, &mut proj);
            for i in 0..n_pos * d {
                x[i] += proj[i];
            }

            // ---- feed-forward, pre-LN ----
            let mut norm = x.clone();
            for row in norm.chunks_exact_mut(d) {
                layernorm_row(row, &layer.ln2_gamma, &layer.ln2_beta, g.ln_eps);
            }
            let width = layer.ffn_up.len() / d;
            let mut up = vec![0.0f32; n_pos * width];
            kernels::gemm_nt(&norm, n_pos, &layer.ffn_up, d, width, &layer.ffn_up_bias, &mut up);
            for v in up.iter_mut() {
                *v = gelu_erf_exact(*v);
            }
            let mut down = vec![0.0f32; n_pos * d];
            kernels::gemm_nt(&up, n_pos, &layer.ffn_down, width, d, &layer.ffn_down_bias, &mut down);
            for i in 0..n_pos * d {
                x[i] += down[i];
            }
        }

        // The final LayerNorm, over the whole stack and not per chunk.
        for row in x.chunks_exact_mut(d) {
            layernorm_row(row, &self.final_gamma, &self.final_beta, g.ln_eps);
        }
        x
    }

    /// The head: the CLS row to the logits.
    ///
    /// One `d x num_labels` matvec on the host. The labels are the FAST
    /// axis of the weight, and the accumulation is in DOUBLE -- the same
    /// arithmetic as the C++'s `head_matvec`.
    pub fn classify(&self, cls_row: &[f32]) -> Vec<f32> {
        let g = &self.geometry;
        let d = g.hidden as usize;
        let nlab = g.num_labels as usize;
        let mut logits = vec![0.0f32; nlab];
        for j in 0..nlab {
            let mut acc = self.head_bias[j] as f64;
            for t in 0..d {
                acc += cls_row[t] as f64 * self.head[t * nlab + j] as f64;
            }
            logits[j] = acc as f32;
        }
        logits
    }
}

/// A prediction: the label, its name and its probability.
pub struct Prediction {
    /// The winning label's index.
    pub label: usize,
    /// The winning label's name.
    pub name: String,
    /// The winning label's probability.
    pub p: f32,
    /// The full logit row.
    pub logits: Vec<f32>,
}

/// The argmax, and then the softmax over the FULL logit row.
///
/// The max is taken first so the exponential cannot overflow, and the
/// probability is computed from the same row rather than from the top few,
/// because a truncated softmax is not a probability and reporting it as
/// one is how a confident-looking 0.99 comes out of a model that is
/// guessing.
pub fn predict(logits: &[f32], labels: &[String]) -> Result<Prediction, String> {
    if logits.is_empty() {
        return Err("the head produced no logits".into());
    }
    let mut best = 0usize;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    let mx = logits[best];
    let mut sum = 0.0f64;
    for v in logits {
        sum += (*v as f64 - mx as f64).exp();
    }
    let p = if sum > 0.0 { (1.0f64 / sum) as f32 } else { 0.0f32 };
    Ok(Prediction {
        label: best,
        name: labels.get(best).cloned().unwrap_or_default(),
        p,
        logits: logits.to_vec(),
    })
}

/// The prediction, as the JSON object.
///
/// The C++'s shape, in the C++'s order: the image, the label, its name,
/// the probability, the runners-up (when more than one was asked for),
/// the three timings and the two dispatch counts.
///
/// The numbers are `%.9g`, and the probability is a float32 printed with
/// nine significant digits -- which is still far more than a softmax
/// carries and is what the golden transcript holds.
pub fn prediction_json(
    prediction: &Prediction,
    image: &str,
    top_k: usize,
    front_end_s: f64,
    encoder_s: f64,
    head_s: f64,
) -> String {
    let mut out = format!(
        "{{\"image\": \"{}\", \"label\": {}, \"name\": \"{}\", \"p\": {}",
        json_escape(image),
        prediction.label,
        json_escape(&prediction.name),
        format_g(prediction.p as f64, 9),
    );
    if top_k > 1 && !prediction.logits.is_empty() {
        // The runners-up, by the same argmax the label came from, and out of
        // the SAME softmax -- so their probabilities sum with `p` to 1 rather
        // than being independently normalised numbers.
        let mx = prediction
            .logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for v in &prediction.logits {
            sum += (*v as f64 - mx as f64).exp();
        }
        let mut order: Vec<usize> = (0..prediction.logits.len()).collect();
        order.sort_by(|a, b| {
            prediction.logits[*b]
                .partial_cmp(&prediction.logits[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let want = top_k.min(order.len());
        out.push_str(", \"top_k\": [");
        for i in 0..want {
            if i > 0 {
                out.push_str(", ");
            }
            let p = if sum > 0.0 {
                (prediction.logits[order[i]] as f64 - mx as f64).exp() / sum
            } else {
                0.0
            };
            out.push_str(&format!(
                "{{\"label\": {}, \"p\": {}}}",
                order[i],
                format_g(p, 9)
            ));
        }
        out.push(']');
    }
    out.push_str(&format!(
        ", \"front_end_s\": {}, \"encoder_s\": {}, \"head_s\": {}, \
         \"dispatches\": 0, \"elt_dispatches\": 0}}",
        format_g(front_end_s, 9),
        format_g(encoder_s, 9),
        format_g(head_s, 9),
    ));
    out
}

/// The JSON escaping the C++'s server uses.
///
/// `"`, `\\`, the three named control characters, and `\\uXXXX` for the rest
/// of the control range.
pub fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// `%.9g`: nine significant digits, C's rules.
///
/// The probability is a float32 printed with nine significant digits, and
/// the golden transcript holds those digits, so the formatting is the
/// contract and not a detail.
pub fn format_g(value: f64, precision: usize) -> String {
    if value.is_nan() {
        return if value.is_sign_negative() {
            "-nan".to_string()
        } else {
            "nan".to_string()
        };
    }
    if value.is_infinite() {
        return if value < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0".to_string()
        } else {
            "0".to_string()
        };
    }
    let exponent = value.abs().log10().floor() as i32;
    if exponent < -4 || exponent >= precision as i32 {
        let text = format!("{:.*e}", precision - 1, value);
        let (mantissa, exp) = text.split_once('e').expect("Rust's exponential form");
        let exp: i32 = exp.parse().expect("Rust's exponential form");
        format!(
            "{}e{}{:02}",
            trim_zeros(mantissa),
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        )
    } else {
        let decimals = (precision as i32 - 1 - exponent).max(0) as usize;
        trim_zeros(&format!("{value:.decimals$}"))
    }
}

/// The trailing zeros of a decimal, and the point that follows them.
fn trim_zeros(text: &str) -> String {
    if !text.contains('.') {
        return text.to_string();
    }
    let trimmed = text.trim_end_matches('0');
    let trimmed = trimmed.strip_suffix('.').unwrap_or(trimmed);
    trimmed.to_string()
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
