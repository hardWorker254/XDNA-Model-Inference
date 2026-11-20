//! The EmbeddingGemma forward pass (arch `gemma3_mqa_rope_geglu`), on the host.
//!
//! Ported from `reference/encoder_gemma.py`, which was checked against
//! transformers' Gemma3TextModel. Numerical traps that still look plausible
//! when wrong are spelled out here:
//! * Gemma's RMSNorm gain is ALREADY `1 + w` in the packed container. Multiplying
//!   by the stored weight is the whole formula; adding one again is a disaster.
//! * Q/K norms run after projection and before RoPE; the attention scale is
//!   `query_pre_attn_scalar^-0.5` on the SCORES. Folding it into Wq is illegal:
//!   q_norm is scale invariant and would annihilate it.
//! * RoPE is rotate_half (halves concatenated, not interleaved), base 1e6 on
//!   every sixth layer and 1e4 on the rest.
//! * Attention is bidirectional. Up to the sliding window (512) the window
//!   predicate never excludes a key, so the only mask here is padding.
//! * The FFN is GeGLU on `gate|up` halves with GELU's tanh approximation.

use crate::pipelines::operand;
use npum_format::Container;
use npum_model::array::{ArrayEltwise, ArrayTextRef, EltOp};
use npum_model::Ops;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("the container at {path} cannot be read: {source}")]
    Container {
        path: String,
        #[source]
        source: npum_format::Error,
    },
    #[error("the container's config has no {0}")]
    NoConfig(&'static str),
    #[error("the container has no tensor named {0}")]
    NoTensor(String),
    #[error("the tensor {name} holds {found} values, and the model needs {expected}")]
    WrongSize {
        name: String,
        expected: usize,
        found: usize,
    },
    #[error("this architecture is {arch}, and this forward pass implements {implemented}")]
    WrongArch {
        arch: String,
        implemented: &'static str,
    },
    #[error("the tensor {name} cannot be read as a GEMM operand: {reason}")]
    BadOperand { name: String, reason: String },
    #[error("the array refused the projection: {0}")]
    Array(#[from] npum_model::array::ArrayError),
    #[error("the softmax design's score row is {cols} wide and this model has {seq} keys")]
    SoftmaxTooNarrow { cols: usize, seq: usize },
    #[error("a sequence of {seq} exceeds the sliding window of {window}")]
    TooLong { seq: usize, window: usize },
    #[error(transparent)]
    Kernel(#[from] npum_model::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
pub const IMPLEMENTED_ARCH: &str = "gemma3_mqa_rope_geglu";

/// The name the CLI uses for this architecture.
pub type GemmaModel = Model;

#[derive(Debug, Clone)]
pub struct Spec {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub dense_hidden: usize,
    pub vocab: usize,
    pub max_seq_len: usize,
    pub eps: f64,
    pub rope_theta: f64,
    pub rope_local_base: f64,
    pub sliding_window: usize,
    pub window_pattern: usize,
    pub attn_scale: f64,
    pub qkv_n: usize,
}

impl Spec {
    pub fn from_config(config: &npum_format::directory::Config) -> Result<Self> {
        let arch = config.get_str("arch").unwrap_or("");
        if arch != IMPLEMENTED_ARCH && !arch.contains("gemma") {
            return Err(Error::WrongArch {
                arch: arch.to_string(),
                implemented: IMPLEMENTED_ARCH,
            });
        }
        let int = |key: &'static str| -> Result<usize> {
            config
                .get_i64(key)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or(Error::NoConfig(key))
        };
        let float = |key: &'static str| -> Result<f64> {
            config.get_f64(key).ok_or(Error::NoConfig(key))
        };
        Ok(Self {
            layers: int("num_layers")?,
            hidden: int("hidden")?,
            heads: int("num_heads")?,
            kv_heads: int("num_key_value_heads")?,
            head_dim: int("head_dim")?,
            ffn: int("intermediate")?,
            dense_hidden: int("dense_hidden")?,
            vocab: int("vocab_size")?,
            max_seq_len: int("max_seq_len")?,
            eps: float("rms_norm_eps")?,
            rope_theta: float("rope_theta")?,
            rope_local_base: float("rope_local_base_freq")?,
            sliding_window: int("sliding_window")?,
            window_pattern: int("sliding_window_pattern")?,
            attn_scale: float("query_pre_attn_scalar")?.powf(-0.5),
            qkv_n: int("qkv_n")?,
        })
    }

    fn global(&self, layer: usize) -> bool {
        self.window_pattern != 0 && (layer + 1) % self.window_pattern == 0
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Timings {
    pub host_layernorm: f64,
    pub host_softmax: f64,
    pub host_attn: f64,
    pub host_gelu: f64,
    pub array_gemm: f64,
    pub array_softmax: f64,
}

impl Timings {
    pub fn busy(&self) -> f64 {
        self.host_layernorm
            + self.host_softmax
            + self.host_attn
            + self.host_gelu
            + self.array_gemm
            + self.array_softmax
    }
}

struct LayerWeights {
    qkv: Vec<f32>,
    qkv_bias: Vec<f32>,
    attn_out: Vec<f32>,
    attn_out_bias: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_up_bias: Vec<f32>,
    ffn_down: Vec<f32>,
    ffn_down_bias: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ln_in: Vec<f32>,
    ln_post_attn: Vec<f32>,
    ln_pre_ffn: Vec<f32>,
    ln_post_ffn: Vec<f32>,
}

pub struct Model {
    pub spec: Spec,
    embed: Vec<f32>,
    final_norm: Vec<f32>,
    layers: Vec<LayerWeights>,
    dense2: Vec<f32>,
    dense3: Vec<f32>,
    embed_scale: f32,
}

fn expect(name: &str, got: Vec<f32>, expected: usize) -> Result<Vec<f32>> {
    if got.len() != expected {
        Err(Error::WrongSize {
            name: name.to_string(),
            expected,
            found: got.len(),
        })
    } else {
        Ok(got)
    }
}

impl Model {
    pub fn from_container(container: &Container) -> Result<Self> {
        let spec = Spec::from_config(container.config())?;
        if spec.heads % spec.kv_heads != 0 {
            return Err(Error::WrongArch {
                arch: "gemma with incompatible MQA head counts".to_string(),
                implemented: IMPLEMENTED_ARCH,
            });
        }
        let f32_tensor = |name: &str| -> Result<Vec<f32>> {
            let entry = container
                .directory()
                .tensor(name)
                .ok_or_else(|| Error::NoTensor(name.to_string()))?;
            if entry.dtype != "F32" {
                return Err(Error::BadOperand {
                    name: name.to_string(),
                    reason: format!("expected F32, found {}", entry.dtype),
                });
            }
            let bytes = container
                .tensor_bytes(name)
                .map_err(|e| Error::BadOperand {
                    name: name.to_string(),
                    reason: e.to_string(),
                })?;
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        let op = |name: &str| -> Result<Vec<f32>> {
            if container.directory().tensor(name).is_none() {
                return Err(Error::NoTensor(name.to_string()));
            }
            operand::read_operand(container, name).map_err(|reason| Error::BadOperand {
                name: name.to_string(),
                reason,
            })
        };
        // Gemma's RMSNorm is `x/rms * (1 + w)`. The ONNX weights used by
        // reference/encoder_gemma.py carry the `1 +` already folded in; the
        // .npue packer writes the raw safetensors weight instead. Folding it
        // HERE is what makes the two agree -- adding it in rms_into too would
        // double-count for the reference and is exactly the trap that file
        // warns about.
        let gain = |name: String, n: usize| -> Result<Vec<f32>> {
            let mut v = expect(&name, f32_tensor(&name)?, n)?;
            for g in v.iter_mut() {
                *g += 1.0;
            }
            Ok(v)
        };
        let embed = expect(
            "embed_tokens.weight",
            f32_tensor("embed_tokens.weight")?,
            spec.vocab * spec.hidden,
        )?;
        let final_norm = gain("norm.weight".to_string(), spec.hidden)?;
        let mut layers = Vec::with_capacity(spec.layers);
        for i in 0..spec.layers {
            let p = format!("layer.{i}.");
            layers.push(LayerWeights {
                qkv: op(&format!("{p}qkv"))?,
                qkv_bias: expect(&format!("{p}qkv.bias"), f32_tensor(&format!("{p}qkv.bias"))?, spec.qkv_n)?,
                attn_out: op(&format!("{p}attn_out"))?,
                attn_out_bias: expect(&format!("{p}attn_out.bias"), f32_tensor(&format!("{p}attn_out.bias"))?, spec.hidden)?,
                ffn_up: op(&format!("{p}ffn_up"))?,
                ffn_up_bias: expect(&format!("{p}ffn_up.bias"), f32_tensor(&format!("{p}ffn_up.bias"))?, 2 * spec.ffn)?,
                ffn_down: op(&format!("{p}ffn_down"))?,
                ffn_down_bias: expect(&format!("{p}ffn_down.bias"), f32_tensor(&format!("{p}ffn_down.bias"))?, spec.hidden)?,
                q_norm: gain(format!("{p}q_norm.weight"), spec.head_dim)?,
                k_norm: gain(format!("{p}k_norm.weight"), spec.head_dim)?,
                ln_in: gain(format!("{p}input_layernorm.weight"), spec.hidden)?,
                ln_post_attn: gain(format!("{p}post_attention_layernorm.weight"), spec.hidden)?,
                ln_pre_ffn: gain(format!("{p}pre_feedforward_layernorm.weight"), spec.hidden)?,
                ln_post_ffn: gain(format!("{p}post_feedforward_layernorm.weight"), spec.hidden)?,
            });
        }
        let dense2 = expect(
            "dense2.weight",
            op("dense2.weight")?,
            spec.hidden * spec.dense_hidden,
        )?;
        let dense3 = expect(
            "dense3.weight",
            op("dense3.weight")?,
            spec.dense_hidden * spec.hidden,
        )?;
        let embed_scale = (spec.hidden as f64).sqrt() as f32;
        Ok(Self {
            spec,
            embed,
            final_norm,
            layers,
            dense2,
            dense3,
            embed_scale,
        })
    }
}

#[derive(Debug, Default)]
pub struct Workspace {
    x: Vec<f32>,
    normed: Vec<f32>,
    qkv: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    scores: Vec<f32>,
    probs: Vec<f32>,
    ctx: Vec<f32>,
    proj: Vec<f32>,
    up: Vec<f32>,
    gate: Vec<f32>,
    down: Vec<f32>,
}

fn grown(buf: &mut Vec<f32>, n: usize) -> &mut [f32] {
    if buf.len() < n {
        buf.resize(n, 0.0);
    }
    &mut buf[..n]
}

fn rms_into(x: &[f32], gain: &[f32], out: &mut [f32], eps: f64) {
    let sum: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let scale = 1.0 / (sum / x.len() as f64 + eps).sqrt();
    for (o, (v, g)) in out.iter_mut().zip(x.iter().zip(gain)) {
        *o = (*v as f64 * scale * *g as f64) as f32;
    }
}

/// RMSNorm over `rows` independent rows of `width` values.
fn rms_rows(x: &[f32], gain: &[f32], out: &mut [f32], eps: f64, rows: usize, width: usize) {
    for r in 0..rows {
        rms_into(
            &x[r * width..(r + 1) * width],
            gain,
            &mut out[r * width..(r + 1) * width],
            eps,
        );
    }
}

fn gelu_tanh(v: f32) -> f32 {
    const C: f64 = 0.7978845608028654;
    let x = v as f64;
    let inner = C * (x + 0.044715 * x * x * x);
    (0.5 * x * (1.0 + inner.tanh())) as f32
}

fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    for row in x.chunks_exact_mut(cols).take(rows) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for v in row.iter_mut() {
            *v = ((*v as f64 - max as f64).exp()) as f32;
            sum += *v as f64;
        }
        for v in row.iter_mut() {
            *v = (*v as f64 / sum) as f32;
        }
    }
}

fn rope_tables(seq: usize, head_dim: usize, base: f64) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; seq * head_dim];
    let mut sin = vec![0.0f32; seq * head_dim];
    for pos in 0..seq {
        for j in 0..half {
            let inv = base.powf(-2.0 * j as f64 / head_dim as f64);
            let angle = pos as f64 * inv;
            let (c, s) = (angle.cos() as f32, angle.sin() as f32);
            cos[pos * head_dim + j] = c;
            cos[pos * head_dim + half + j] = c;
            sin[pos * head_dim + j] = s;
            sin[pos * head_dim + half + j] = s;
        }
    }
    (cos, sin)
}

fn rotate_half(v: &[f32], out: &mut [f32]) {
    let half = v.len() / 2;
    for i in 0..half {
        out[i] = -v[half + i];
        out[half + i] = v[i];
    }
}

impl Model {
    /// Open a model from a container path, as the other families do.
    pub fn open(path: &Path) -> Result<Self> {
        let container = Container::open(path).map_err(|e| Error::Container {
            path: path.display().to_string(),
            source: e,
        })?;
        Self::from_container(&container)
    }

    /// Diagnostic: the scaled embedding row of one token, before any layer.
    pub fn debug_embedding(&self, id: usize) -> Vec<f32> {
        (0..self.spec.hidden)
            .map(|d| (self.embed[id * self.spec.hidden + d] as f64 * self.embed_scale as f64) as f32)
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn embed_timed_with(
        &self,
        ops: &mut dyn Ops,
        timings: &mut Timings,
        ids: &[u32],
        mask: &[u32],
        batch: usize,
        seq: usize,
        array: ArrayTextRef<'_, '_>,
        eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
    ) -> Result<Vec<f32>> {
        self.embed_timed_with_taps(ops, timings, ids, mask, batch, seq, array, eltwise,
                                   &mut HashMap::new())
    }

    /// The same run, with named intermediate tensors kept for diagnosis.
    #[allow(clippy::too_many_arguments)]
    pub fn embed_timed_with_taps(
        &self,
        ops: &mut dyn Ops,
        timings: &mut Timings,
        ids: &[u32],
        mask: &[u32],
        batch: usize,
        seq: usize,
        mut array: ArrayTextRef<'_, '_>,
        mut eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
        taps: &mut HashMap<String, Vec<f32>>,
    ) -> Result<Vec<f32>> {
        if seq > self.spec.sliding_window {
            return Err(Error::TooLong { seq, window: self.spec.sliding_window });
        }
        if ids.len() < batch * seq || mask.len() < batch * seq {
            return Err(Error::WrongSize {
                name: "input ids / mask".to_string(),
                expected: batch * seq,
                found: ids.len().min(mask.len()),
            });
        }
        let sp = &self.spec;
        let rows = batch * seq;
        let hidden = sp.hidden;
        let hd = sp.head_dim;
        let heads = sp.heads;
        let rep = heads / sp.kv_heads;
        let mut ws = Workspace::default();

        let x = grown(&mut ws.x, rows * hidden);
        for (r, &id) in ids[..rows].iter().enumerate() {
            let row = id as usize * hidden;
            for d in 0..hidden {
                x[r * hidden + d] = (self.embed[row + d] as f64 * self.embed_scale as f64) as f32;
            }
        }

        if !taps.is_empty() {
            taps.insert("emb.scaled".to_string(), x.to_vec());
        }
        let sm_cols = eltwise
            .as_deref()
            .filter(|e| e.on_array(EltOp::Softmax))
            .and_then(|e| e.softmax_cols())
            .unwrap_or(0);
        if sm_cols != 0 && sm_cols < seq {
            return Err(Error::SoftmaxTooNarrow { cols: sm_cols, seq });
        }
        let score_cols = sm_cols.max(seq);
        let normed = grown(&mut ws.normed, rows * hidden);
        let qkv = grown(&mut ws.qkv, rows * sp.qkv_n);
        let q = grown(&mut ws.q, rows * heads * hd);
        let k = grown(&mut ws.k, rows * sp.kv_heads * hd);
        let v = grown(&mut ws.v, rows * sp.kv_heads * hd);
        let mut scores = grown(&mut ws.scores, batch * heads * seq * score_cols);
        let mut probs = grown(&mut ws.probs, batch * heads * seq * score_cols);
        let ctx = grown(&mut ws.ctx, rows * hidden);
        let proj = grown(&mut ws.proj, rows * hidden);
        let up = grown(&mut ws.up, rows * 2 * sp.ffn);
        let gate = grown(&mut ws.gate, rows * sp.ffn);
        let down = grown(&mut ws.down, rows * hidden);
        let mut rotated = vec![0.0f32; hd];
        let mut head_tmp = vec![0.0f32; hd];
        let (cos_g, sin_g) = rope_tables(seq, hd, sp.rope_theta);
        let (cos_l, sin_l) = rope_tables(seq, hd, sp.rope_local_base);

        for (li, layer) in self.layers.iter().enumerate() {
            let t = std::time::Instant::now();
            rms_rows(x, &layer.ln_in, normed, sp.eps, rows, hidden);
            timings.host_layernorm += t.elapsed().as_secs_f64();

            match array.as_deref_mut().filter(|a| a.gemm_on_array()) {
                Some(a) => {
                    let t = std::time::Instant::now();
                    a.gemm(&format!("layer.{li}.qkv"), normed, rows, hidden, sp.qkv_n, None, qkv)?;
                    timings.array_gemm += t.elapsed().as_secs_f64();
                }
                None => ops.gemm_nt(normed, rows, &layer.qkv, hidden, sp.qkv_n, &layer.qkv_bias, qkv)?,
            }

            if !taps.is_empty() && li == 0 {
                taps.insert("L0.ln_in".to_string(), normed.to_vec());
                taps.insert("L0.qkv".to_string(), qkv.to_vec());
            }
            let qw = heads * hd;
            let kvw = sp.kv_heads * hd;
            for r in 0..rows {
                q[r * qw..(r + 1) * qw].copy_from_slice(&qkv[r * sp.qkv_n..r * sp.qkv_n + qw]);
                k[r * kvw..(r + 1) * kvw]
                    .copy_from_slice(&qkv[r * sp.qkv_n + qw..r * sp.qkv_n + qw + kvw]);
                v[r * kvw..(r + 1) * kvw]
                    .copy_from_slice(&qkv[r * sp.qkv_n + qw + kvw..r * sp.qkv_n + qw + 2 * kvw]);
            }
            let (cos, sin) = if sp.global(li) { (&cos_g, &sin_g) } else { (&cos_l, &sin_l) };
            for r in 0..rows {
                let pos = r % seq;
                for h in 0..heads {
                    let at = r * qw + h * hd;
                    head_tmp.copy_from_slice(&q[at..at + hd]);
                    rms_into(&head_tmp, &layer.q_norm, &mut q[at..at + hd], sp.eps);
                }
                for h in 0..sp.kv_heads {
                    let at = r * kvw + h * hd;
                    head_tmp.copy_from_slice(&k[at..at + hd]);
                    rms_into(&head_tmp, &layer.k_norm, &mut k[at..at + hd], sp.eps);
                }
            if !taps.is_empty() && li == 0 && r == 0 {
                taps.insert("L0.q_normed".to_string(), q[..heads * hd].to_vec());
                taps.insert("L0.k_normed".to_string(), k[..sp.kv_heads * hd].to_vec());
            }
                for h in 0..heads {
                    let at = r * qw + h * hd;
                    rotate_half(&q[at..at + hd], &mut rotated);
                    for d in 0..hd {
                        q[at + d] = (q[at + d] as f64 * cos[pos * hd + d] as f64
                            + rotated[d] as f64 * sin[pos * hd + d] as f64) as f32;
                    }
                }
                for h in 0..sp.kv_heads {
                    let at = r * kvw + h * hd;
                    rotate_half(&k[at..at + hd], &mut rotated);
                    for d in 0..hd {
                        k[at + d] = (k[at + d] as f64 * cos[pos * hd + d] as f64
                            + rotated[d] as f64 * sin[pos * hd + d] as f64) as f32;
                    }
                }
            }

            for b in 0..batch {
                for h in 0..heads {
                    let score_base = (b * heads + h) * seq;
                    for t in 0..seq {
                        let row = &mut scores[(score_base + t) * score_cols..];
                        let qrow = (b * seq + t) * qw + h * hd;
                        for u in 0..seq {
                            let krow = (b * seq + u) * kvw + (h / rep) * hd;
                            let mut dot = 0.0f64;
                            for d in 0..hd {
                                dot += q[qrow + d] as f64 * k[krow + d] as f64;
                            }
                            let masked = if mask[b * seq + u] == 0 { f32::MIN } else { 0.0 };
                            row[u] = (dot * sp.attn_scale) as f32 + masked;
                        }
                        for u in seq..score_cols {
                            row[u] = -1.0e30;
                        }
                    }
                }
            }

            if !taps.is_empty() && li == 0 {
                taps.insert("L0.q_rope".to_string(), q[..batch * seq * heads * hd].to_vec());
                taps.insert("L0.k_rope".to_string(), k[..batch * seq * sp.kv_heads * hd].to_vec());
                taps.insert("L0.scores".to_string(), scores[..batch * heads * seq * seq].to_vec());
            }
            let t = std::time::Instant::now();
            match eltwise.as_deref_mut().filter(|e| e.on_array(EltOp::Softmax)) {
                Some(sm) => {
                    sm.softmax(scores, batch * heads * seq)?;
                    probs.copy_from_slice(scores);
                    timings.array_softmax += t.elapsed().as_secs_f64();
                }
                None => {
                    softmax_rows(scores, batch * heads * seq, score_cols);
                    std::mem::swap(&mut probs, &mut scores);
                    timings.host_softmax += t.elapsed().as_secs_f64();
                }
            }

            if !taps.is_empty() && li == 0 {
                taps.insert("L0.probs".to_string(), probs[..batch * heads * seq * seq].to_vec());
            }
            for b in 0..batch {
                for h in 0..heads {
                    let score_base = (b * heads + h) * seq;
                    for t in 0..seq {
                        let prow = (score_base + t) * score_cols;
                        let orow = (b * seq + t) * qw + h * hd;
                        for d in 0..hd {
                            let mut acc = 0.0f64;
                            for u in 0..seq {
                                let vrow = (b * seq + u) * kvw + (h / rep) * hd;
                                acc += probs[prow + u] as f64 * v[vrow + d] as f64;
                            }
                            ctx[orow + d] = acc as f32;
                        }
                    }
                }
            }

            if !taps.is_empty() && li == 0 {
                taps.insert("L0.ctx".to_string(), ctx.to_vec());
            }
            match array.as_deref_mut().filter(|a| a.gemm_on_array()) {
                Some(a) => {
                    let t = std::time::Instant::now();
                    a.gemm(&format!("layer.{li}.attn_out"), ctx, rows, hidden, hidden, None, proj)?;
                    timings.array_gemm += t.elapsed().as_secs_f64();
                }
                None => ops.gemm_nt(ctx, rows, &layer.attn_out, hidden, hidden, &layer.attn_out_bias, proj)?,
            }
            let t = std::time::Instant::now();
            rms_rows(proj, &layer.ln_post_attn, normed, sp.eps, rows, hidden);
            timings.host_layernorm += t.elapsed().as_secs_f64();
            for (o, a) in x.iter_mut().zip(normed.iter()) {
                *o += *a;
            }

            if !taps.is_empty() && li == 0 {
                taps.insert("L0.resid1".to_string(), x.to_vec());
            }
            let t = std::time::Instant::now();
            rms_rows(x, &layer.ln_pre_ffn, normed, sp.eps, rows, hidden);
            timings.host_layernorm += t.elapsed().as_secs_f64();
            match array.as_deref_mut().filter(|a| a.gemm_on_array()) {
                Some(a) => {
                    let t = std::time::Instant::now();
                    a.gemm(&format!("layer.{li}.ffn_up"), normed, rows, hidden, 2 * sp.ffn, None, up)?;
                    timings.array_gemm += t.elapsed().as_secs_f64();
                }
                None => ops.gemm_nt(normed, rows, &layer.ffn_up, hidden, 2 * sp.ffn, &layer.ffn_up_bias, up)?,
            }
            let t = std::time::Instant::now();
            for r in 0..rows {
                for d in 0..sp.ffn {
                    gate[r * sp.ffn + d] = (gelu_tanh(up[r * 2 * sp.ffn + d]) as f64
                        * up[r * 2 * sp.ffn + sp.ffn + d] as f64) as f32;
                }
            }
            timings.host_gelu += t.elapsed().as_secs_f64();
            match array.as_deref_mut().filter(|a| a.gemm_on_array()) {
                Some(a) => {
                    let t = std::time::Instant::now();
                    a.gemm(&format!("layer.{li}.ffn_down"), gate, rows, sp.ffn, hidden, None, down)?;
                    timings.array_gemm += t.elapsed().as_secs_f64();
                }
                None => ops.gemm_nt(gate, rows, &layer.ffn_down, sp.ffn, hidden, &layer.ffn_down_bias, down)?,
            }
            let t = std::time::Instant::now();
            rms_rows(down, &layer.ln_post_ffn, normed, sp.eps, rows, hidden);
            timings.host_layernorm += t.elapsed().as_secs_f64();
            for (o, a) in x.iter_mut().zip(normed.iter()) {
                *o += *a;
            }
        }

        if let Some(l) = self.layers.first() {
            let _ = l;
        }
        let t = std::time::Instant::now();
        rms_rows(x, &self.final_norm, normed, sp.eps, rows, hidden);
        timings.host_layernorm += t.elapsed().as_secs_f64();
        let mut pooled = vec![0.0f32; batch * hidden];
        for b in 0..batch {
            let denom = mask[b * seq..(b + 1) * seq]
                .iter()
                .filter(|&&m| m != 0)
                .count()
                .max(1) as f64;
            for d in 0..hidden {
                let mut sum = 0.0f64;
                for s in 0..seq {
                    if mask[b * seq + s] != 0 {
                        sum += normed[(b * seq + s) * hidden + d] as f64;
                    }
                }
                pooled[b * hidden + d] = (sum / denom) as f32;
            }
        }
        let mut dense = vec![0.0f32; batch * sp.dense_hidden];
        let zeros2 = vec![0.0f32; sp.dense_hidden];
        ops.gemm_nt(&pooled, batch, &self.dense2, hidden, sp.dense_hidden, &zeros2, &mut dense)?;
        let mut out = vec![0.0f32; batch * hidden];
        let zeros3 = vec![0.0f32; hidden];
        ops.gemm_nt(&dense, batch, &self.dense3, sp.dense_hidden, hidden, &zeros3, &mut out)?;
        for row in out.chunks_exact_mut(hidden) {
            let norm = row
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>()
                .max(1e-24)
                .sqrt();
            for v in row.iter_mut() {
                *v = (*v as f64 / norm) as f32;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_applies_the_gain_verbatim() {
        // The `1 + w` folding happens once, when the container is loaded;
        // the kernel itself must not add one a second time. RMS is over the
        // last axis: sqrt((3^2 + 4^2)/2) = sqrt(12.5).
        let mut out = [0.0f32; 2];
        rms_into(&[3.0, 4.0], &[2.0, 2.0], &mut out, 1e-6);
        let rms = 12.5f64.sqrt();
        assert!((out[0] as f64 - 3.0 * 2.0 / rms).abs() < 1e-4, "{:?}", out);
        assert!((out[1] as f64 - 4.0 * 2.0 / rms).abs() < 1e-4, "{:?}", out);
    }

    #[test]
    fn rope_tables_have_the_rotate_half_symmetry() {
        let (cos, sin) = rope_tables(2, 8, 1e4);
        for pos in 0..2 {
            for j in 0..4 {
                assert_eq!(cos[pos * 8 + j], cos[pos * 8 + 4 + j]);
                assert_eq!(sin[pos * 8 + j], sin[pos * 8 + 4 + j]);
            }
        }
    }
}
