//! The GTE forward pass (arch `gte_new_rope_geglu`), on the host.
//!
//! Ported from `reference/encoder_gte.py` and matched to the packed container
//! layout. Post-LN, RoPE with NTK-scaled inv_freq (baked in config), gated GEGLU
//! FFN with GELU on the gate half, CLS pooling, no final norm.

use crate::kernels;
use npum_model::Ops;
use npum_model::array::{ArrayEltwise, ArrayTextRef};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("the container at {path} cannot be read: {source}")]
    Container {
        path: String,
        #[source]
        source: npum_format::Error,
    },
    #[error("the container's config has no {key}")]
    NoConfig { key: &'static str },
    #[error("the container has no tensor named {0}")]
    NoTensor(String),
    #[error("the tensor {name} holds {found} values, and the model's shapes need {expected}")]
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
    #[error(
        "the hidden size {hidden} is not a multiple of eight, and the host kernels read eight floats at a time"
    )]
    UnalignedHidden { hidden: usize },
    #[error(
        "the head size {head_dim} is not a multiple of eight, and the host kernels read eight floats at a time"
    )]
    UnalignedHead { head_dim: usize },
    #[error("the input is {found} ids for {batch} sequences of {seq}, and it should be {expected}")]
    WrongInput {
        batch: usize,
        seq: usize,
        expected: usize,
        found: usize,
    },
    #[error("the tensor {name} cannot be read as a GEMM operand: {reason}")]
    BadOperand { name: String, reason: String },
    #[error("the array refused the projection: {0}")]
    Array(#[from] npum_model::array::ArrayError),
    #[error(transparent)]
    Kernel(#[from] npum_model::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
pub const IMPLEMENTED_ARCH: &str = "gte_new_rope_geglu";

#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    pub host_layernorm: f64,
    pub host_softmax: f64,
    pub host_gelu: f64,
    pub attention: f64,
    pub qk: f64,
    pub av: f64,
    pub array_gemm: f64,
    pub gpu: f64,
}

impl Timings {
    pub fn busy(&self) -> f64 {
        self.host_layernorm
            + self.host_softmax
            + self.host_gelu
            + self.attention
            + self.array_gemm
            + self.gpu
    }
}

#[derive(Debug, Clone)]
pub struct GteSpec {
    pub layers: usize,
    pub heads: usize,
    pub hidden: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub max_seq_len: usize,
    pub cls_pooling: bool,
    pub l2_normalize: bool,
    pub eps: f32,
    pub rope_inv_freq: Vec<f32>,
}

impl GteSpec {
    pub fn from_config(config: &npum_format::directory::Config) -> Result<Self> {
        let get = |key: &'static str| -> Result<i64> {
            config.get_i64(key).ok_or(Error::NoConfig { key })
        };
        let layers = get("num_layers")? as usize;
        let heads = get("num_heads")? as usize;
        let hidden = get("hidden")? as usize;
        let head_dim = get("head_dim")? as usize;
        let ffn = get("intermediate")? as usize;
        let vocab = get("vocab_size")? as usize;
        let max_seq_len = get("max_seq_len")? as usize;
        let pooling = config.get_str("pooling").unwrap_or("cls");
        let eps = config.get_f64("layer_norm_eps").unwrap_or(1e-12) as f32;
        let rope_inv_freq = if let Some(v) = config.get("rope_inv_freq").and_then(|v| v.as_array())
        {
            v.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect()
        } else {
            let theta = config.get_f64("rope_theta").unwrap_or(20000.0) as f32;
            let factor = config
                .get("rope_scaling")
                .and_then(|s| s.get("factor"))
                .and_then(|v| v.as_f64())
                .unwrap_or(8.0) as f32;
            let mut inv = vec![0.0f32; head_dim / 2];
            let scaled_base = theta * factor;
            for i in 0..(head_dim / 2) {
                let exp = (2.0 * i as f32) / head_dim as f32;
                inv[i] = 1.0 / scaled_base.powf(exp) / factor.powf(2.0 / head_dim as f32);
            }
            inv
        };
        Ok(Self {
            layers,
            heads,
            hidden,
            head_dim,
            ffn,
            vocab,
            max_seq_len,
            cls_pooling: pooling == "cls",
            l2_normalize: config.get_bool("l2_normalize").unwrap_or(true),
            eps,
            rope_inv_freq,
        })
    }
}

fn read_f32_tensor(container: &npum_format::Container, name: &str) -> Result<Vec<f32>> {
    let entry = container
        .directory()
        .tensor(name)
        .ok_or_else(|| Error::NoTensor(name.to_string()))?;
    if entry.dtype != "F32" {
        return Err(Error::BadOperand {
            name: name.to_string(),
            reason: format!("expected an F32 tensor, found {}", entry.dtype),
        });
    }
    let bytes = container
        .tensor_bytes(name)
        .map_err(|_| Error::NoTensor(name.to_string()))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_operand(
    container: &npum_format::Container,
    name: &str,
) -> std::result::Result<Vec<f32>, String> {
    let entry = container
        .directory()
        .tensor(name)
        .ok_or_else(|| format!("the container has no tensor named {name}"))?;
    let bytes = container
        .tensor_bytes(name)
        .map_err(|e| format!("{name}: {e}"))?;
    match entry.dtype.as_str() {
        "F32" => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        "BF16" => {
            let shape: Vec<i64> = entry.padded_shape.clone();
            if shape.len() != 2 {
                return Err(format!(
                    "{name}: a GEMM operand has a {}-dimensional padded shape; a dense operand is [K, N]",
                    shape.len()
                ));
            }
            if !entry.logical_shape.is_empty()
                && (entry.logical_shape[0] != shape[0] || entry.logical_shape[1] != shape[1])
            {
                return Err(format!(
                    "{name}: logical [{},{}] but padded [{},{}]. The host builds [K,N] from the padded shape and the caller indexes it with the logical one, so the two must agree.",
                    entry.logical_shape[0], entry.logical_shape[1], shape[0], shape[1]
                ));
            }
            let (k, n) = (shape[0] as usize, shape[1] as usize);
            if k == 0 || n == 0 {
                return Err(format!("{name}: a GEMM operand with a zero dimension"));
            }
            let layout = entry.layout.clone().ok_or_else(|| {
                format!("{name}: the payload carries no layout. The layout is what says where each element sits, so without it these bytes are a right count in an unknown order.")
            })?;
            let layout: npum_format::layout::BLayout = serde_json::from_value(layout)
                .map_err(|e| format!("{name}: the recorded layout is unreadable: {e}"))?;
            if layout.kind != "block_panel"
                || layout.tile_k <= 0
                || layout.tile_n <= 0
                || layout.mac_s <= 0
                || layout.mac_t <= 0
            {
                return Err(format!(
                    "{name}: the payload carries no usable block_panel layout (kind '{}').",
                    layout.kind
                ));
            }
            let tiled = npum_format::tiling::untile_b(
                bytes,
                k,
                n,
                layout.tile_k as usize,
                layout.tile_n as usize,
                2,
                Some(layout.mac_s as usize),
                Some(layout.mac_t as usize),
                npum_format::tiling::TileOrder::Kn,
            )
            .map_err(|e| format!("{name}: the panel does not de-tile: {}", e.message))?;
            let mut values = vec![0f32; k * n];
            npum_format::bf16::bf16_to_f32_bytes(&tiled, &mut values);
            Ok(values)
        }
        other => Err(format!(
            "{name}: a {other} operand cannot be run on the host. This build widens BF16 and copies F32; anything else would mean inventing a conversion, and a guessed conversion is a wrong number in a plausible range."
        )),
    }
}

fn expect(name: &str, values: Vec<f32>, want: usize) -> Result<Vec<f32>> {
    if values.len() != want {
        return Err(Error::WrongSize {
            name: name.to_string(),
            expected: want,
            found: values.len(),
        });
    }
    Ok(values)
}

#[derive(Debug, Clone)]
struct LayerWeights {
    qkv: Vec<f32>,
    qkv_bias: Vec<f32>,
    attn_out: Vec<f32>,
    attn_out_bias: Vec<f32>,
    ln1_gamma: Vec<f32>, // attn_ln
    ln1_beta: Vec<f32>,
    ffn_up: Vec<f32>, // fused up_gate [hidden, 2*ffn]
    ffn_down: Vec<f32>,
    ffn_down_bias: Vec<f32>,
    ln2_gamma: Vec<f32>, // mlp_ln
    ln2_beta: Vec<f32>,
}

#[derive(Debug, Clone)]
struct Weights {
    word: Vec<f32>,
    token_type: Vec<f32>,
    ln_gamma: Vec<f32>,
    ln_beta: Vec<f32>,
    layers: Vec<LayerWeights>,
}

#[derive(Debug, Clone)]
pub struct GteModel {
    pub spec: GteSpec,
    weights: Weights,
}

impl GteModel {
    pub fn open(path: &Path) -> Result<Self> {
        let container = npum_format::Container::open(path).map_err(|source| Error::Container {
            path: path.display().to_string(),
            source,
        })?;
        let arch = container.config_str("arch").unwrap_or("").to_string();
        if arch != IMPLEMENTED_ARCH {
            return Err(Error::WrongArch {
                arch,
                implemented: IMPLEMENTED_ARCH,
            });
        }
        let spec = GteSpec::from_config(container.config())?;
        if spec.hidden % 8 != 0 {
            return Err(Error::UnalignedHidden {
                hidden: spec.hidden,
            });
        }
        if spec.head_dim % 8 != 0 {
            return Err(Error::UnalignedHead {
                head_dim: spec.head_dim,
            });
        }
        let hidden = spec.hidden;
        let ffn = spec.ffn;
        let three_hidden = 3 * hidden;

        let read_op = |name: &str| -> Result<Vec<f32>> {
            if container.directory().tensor(name).is_some() {
                // try direct f32 first? read_operand handles both
                return read_operand(&container, name).map_err(|reason| Error::BadOperand {
                    name: name.to_string(),
                    reason,
                });
            }
            Err(Error::NoTensor(name.to_string()))
        };

        let word = expect(
            "embeddings.word",
            read_f32_tensor(&container, "embeddings.word")?,
            spec.vocab * hidden,
        )?;
        let token_type = expect(
            "embeddings.token_type",
            read_f32_tensor(&container, "embeddings.token_type")?,
            hidden,
        )?;
        let ln_gamma = expect(
            "embeddings.ln.weight",
            read_f32_tensor(&container, "embeddings.ln.weight")?,
            hidden,
        )?;
        let ln_beta = expect(
            "embeddings.ln.bias",
            read_f32_tensor(&container, "embeddings.ln.bias")?,
            hidden,
        )?;

        let mut layers = Vec::with_capacity(spec.layers);
        for l in 0..spec.layers {
            let p = format!("layer.{l}");
            layers.push(LayerWeights {
                qkv: expect(
                    &format!("{p}.qkv"),
                    read_op(&format!("{p}.qkv"))?,
                    hidden * three_hidden,
                )?,
                qkv_bias: expect(
                    &format!("{p}.qkv.bias"),
                    read_f32_tensor(&container, &format!("{p}.qkv.bias"))?,
                    three_hidden,
                )?,
                attn_out: expect(
                    &format!("{p}.attn_out"),
                    read_op(&format!("{p}.attn_out"))?,
                    hidden * hidden,
                )?,
                attn_out_bias: expect(
                    &format!("{p}.attn_out.bias"),
                    read_f32_tensor(&container, &format!("{p}.attn_out.bias"))?,
                    hidden,
                )?,
                ln1_gamma: expect(
                    &format!("{p}.ln1.weight"),
                    read_f32_tensor(&container, &format!("{p}.ln1.weight"))?,
                    hidden,
                )?,
                ln1_beta: expect(
                    &format!("{p}.ln1.bias"),
                    read_f32_tensor(&container, &format!("{p}.ln1.bias"))?,
                    hidden,
                )?,
                ffn_up: expect(
                    &format!("{p}.ffn_up"),
                    read_op(&format!("{p}.ffn_up"))?,
                    hidden * 2 * ffn,
                )?,
                ffn_down: expect(
                    &format!("{p}.ffn_down"),
                    read_op(&format!("{p}.ffn_down"))?,
                    ffn * hidden,
                )?,
                ffn_down_bias: expect(
                    &format!("{p}.ffn_down.bias"),
                    read_f32_tensor(&container, &format!("{p}.ffn_down.bias"))?,
                    hidden,
                )?,
                ln2_gamma: expect(
                    &format!("{p}.ln2.weight"),
                    read_f32_tensor(&container, &format!("{p}.ln2.weight"))?,
                    hidden,
                )?,
                ln2_beta: expect(
                    &format!("{p}.ln2.bias"),
                    read_f32_tensor(&container, &format!("{p}.ln2.bias"))?,
                    hidden,
                )?,
            });
        }

        Ok(Self {
            spec,
            weights: Weights {
                word,
                token_type,
                ln_gamma,
                ln_beta,
                layers,
            },
        })
    }

    pub fn embed_timed_with(
        &self,
        ops: &mut dyn Ops,
        timings: &mut Timings,
        ids: &[u32],
        mask: &[u32],
        batch: usize,
        seq: usize,
        array: ArrayTextRef<'_, '_>,
        _eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
    ) -> Result<Vec<f32>> {
        let rows = batch * seq;
        if ids.len() != rows || mask.len() != rows {
            return Err(Error::WrongInput {
                batch,
                seq,
                expected: rows,
                found: ids.len().max(mask.len()),
            });
        }
        if seq > self.spec.max_seq_len {
            return Err(Error::WrongInput {
                batch,
                seq,
                expected: self.spec.max_seq_len,
                found: seq,
            });
        }
        let hidden = self.spec.hidden;

        // Embeddings: word + token_type[0], then embeddings.LayerNorm.
        // No absolute position addition (RoPE). position_embedding tensor in container is zeroed.
        let mut x = vec![0.0f32; rows * hidden];
        for r in 0..rows {
            let id = ids[r] as usize;
            let wv = &self.weights.word[id * hidden..(id + 1) * hidden];
            let tv = &self.weights.token_type[..hidden];
            let dst = &mut x[r * hidden..(r + 1) * hidden];
            for c in 0..hidden {
                dst[c] = wv[c] + tv[c];
            }
        }

        let mut add_mask = vec![0.0f32; rows];
        let mut attn_mask = vec![0.0f32; rows];
        for r in 0..rows {
            if mask[r] > 0 {
                attn_mask[r] = 1.0;
            } else {
                add_mask[r] = -1.0e30;
            }
        }

        let t0 = std::time::Instant::now();
        ops.layernorm_rows(
            &mut x,
            rows,
            hidden,
            &self.weights.ln_gamma,
            &self.weights.ln_beta,
        )?;
        timings.host_layernorm += t0.elapsed().as_secs_f64();

        let cos_sin = rope_cos_sin(seq, &self.spec.rope_inv_freq, self.spec.head_dim);
        let h = self.forward(ops, &mut x, &add_mask, batch, seq, timings, array, &cos_sin)?;
        Ok(self.pool(&h, &attn_mask, batch, seq))
    }

    fn forward(
        &self,
        ops: &mut dyn Ops,
        x: &mut [f32],
        add_mask: &[f32],
        batch: usize,
        seq: usize,
        timings: &mut Timings,
        mut array: ArrayTextRef<'_, '_>,
        cos_sin: &(Vec<f32>, Vec<f32>),
    ) -> Result<Vec<f32>> {
        let spec = &self.spec;
        let hidden = spec.hidden;
        let rows = batch * seq;
        let three_hidden = 3 * hidden;
        let score_rows = batch * spec.heads * seq;
        let gemm_on = array.as_deref().is_some_and(|a| a.gemm_on_array());

        let mut qkv = vec![0.0f32; rows * three_hidden];
        let mut ctx = vec![0.0f32; rows * hidden];
        let mut proj = vec![0.0f32; rows * hidden];
        let mut up_gate = vec![0.0f32; rows * 2 * spec.ffn];
        let mut act = vec![0.0f32; rows * spec.ffn];
        let mut down = vec![0.0f32; rows * hidden];
        let mut scores = vec![0.0f32; score_rows * seq];

        for (index, layer) in self.weights.layers.iter().enumerate() {
            // QKV projection
            match array.as_deref_mut().filter(|_| gemm_on) {
                Some(array) => {
                    let t = std::time::Instant::now();
                    array.gemm(
                        &format!("layer.{index}.qkv"),
                        x,
                        rows,
                        hidden,
                        three_hidden,
                        Some(&layer.qkv_bias),
                        &mut qkv,
                    )?;
                    timings.array_gemm += t.elapsed().as_secs_f64();
                }
                None => {
                    let t = std::time::Instant::now();
                    ops.gemm_nt(
                        x,
                        rows,
                        &layer.qkv,
                        hidden,
                        three_hidden,
                        &layer.qkv_bias,
                        &mut qkv,
                    )?;
                    if ops.place() != npum_model::Placement::Host {
                        timings.gpu += t.elapsed().as_secs_f64();
                    }
                }
            }
            // Apply RoPE to Q and K
            apply_rope_qkv(
                &mut qkv,
                batch,
                seq,
                spec.heads,
                spec.head_dim,
                hidden,
                cos_sin,
            );

            // Attention (host path, correct semantics)
            let t_qk = std::time::Instant::now();
            self.qk(&qkv, &mut scores, batch, seq, add_mask);
            timings.qk += t_qk.elapsed().as_secs_f64();
            let t_sm = std::time::Instant::now();
            ops.softmax_rows(&mut scores, score_rows, seq)?;
            timings.host_softmax += t_sm.elapsed().as_secs_f64();
            let t_av = std::time::Instant::now();
            self.av(&scores, &qkv, &mut ctx, batch, seq);
            timings.av += t_av.elapsed().as_secs_f64();
            timings.attention += t_qk.elapsed().as_secs_f64() + t_av.elapsed().as_secs_f64();

            // Output projection + residual
            match array.as_deref_mut().filter(|_| gemm_on) {
                Some(array) => {
                    let t = std::time::Instant::now();
                    array.gemm(
                        &format!("layer.{index}.attn_out"),
                        &ctx,
                        rows,
                        hidden,
                        hidden,
                        Some(&layer.attn_out_bias),
                        &mut proj,
                    )?;
                    timings.array_gemm += t.elapsed().as_secs_f64();
                }
                None => {
                    let t = std::time::Instant::now();
                    ops.gemm_nt(
                        &ctx,
                        rows,
                        &layer.attn_out,
                        hidden,
                        hidden,
                        &layer.attn_out_bias,
                        &mut proj,
                    )?;
                    if ops.place() != npum_model::Placement::Host {
                        timings.gpu += t.elapsed().as_secs_f64();
                    }
                }
            }
            for i in 0..x.len() {
                x[i] += proj[i];
            }
            // attn_ln (post-LN)
            let t_ln = std::time::Instant::now();
            ops.layernorm_rows(x, rows, hidden, &layer.ln1_gamma, &layer.ln1_beta)?;
            timings.host_layernorm += t_ln.elapsed().as_secs_f64();

            // Gated GEGLU MLP: up_gate = x @ W (no bias), split [up|gate], gelu(gate)*up, down
            let zeros_bias = vec![0.0f32; 2 * spec.ffn];
            match array.as_deref_mut().filter(|_| gemm_on) {
                Some(array) => {
                    let t = std::time::Instant::now();
                    array.gemm(
                        &format!("layer.{index}.ffn_up"),
                        x,
                        rows,
                        hidden,
                        2 * spec.ffn,
                        None,
                        &mut up_gate,
                    )?;
                    timings.array_gemm += t.elapsed().as_secs_f64();
                }
                None => {
                    let t = std::time::Instant::now();
                    ops.gemm_nt(
                        x,
                        rows,
                        &layer.ffn_up,
                        hidden,
                        2 * spec.ffn,
                        &zeros_bias,
                        &mut up_gate,
                    )?;
                    if ops.place() != npum_model::Placement::Host {
                        timings.gpu += t.elapsed().as_secs_f64();
                    }
                }
            }
            let inter = spec.ffn;
            for r in 0..rows {
                let base = r * 2 * inter;
                for j in 0..inter {
                    let upv = up_gate[base + j];
                    let gate = up_gate[base + inter + j];
                    act[r * inter + j] = upv * gelu_erf_f32(gate);
                }
            }
            match array.as_deref_mut().filter(|_| gemm_on) {
                Some(array) => {
                    let t = std::time::Instant::now();
                    array.gemm(
                        &format!("layer.{index}.ffn_down"),
                        &act,
                        rows,
                        inter,
                        hidden,
                        Some(&layer.ffn_down_bias),
                        &mut down,
                    )?;
                    timings.array_gemm += t.elapsed().as_secs_f64();
                }
                None => {
                    let t = std::time::Instant::now();
                    ops.gemm_nt(
                        &act,
                        rows,
                        &layer.ffn_down,
                        inter,
                        hidden,
                        &layer.ffn_down_bias,
                        &mut down,
                    )?;
                    if ops.place() != npum_model::Placement::Host {
                        timings.gpu += t.elapsed().as_secs_f64();
                    }
                }
            }
            for i in 0..x.len() {
                x[i] += down[i];
            }
            // mlp_ln (post-LN)
            let t_ln = std::time::Instant::now();
            ops.layernorm_rows(x, rows, hidden, &layer.ln2_gamma, &layer.ln2_beta)?;
            timings.host_layernorm += t_ln.elapsed().as_secs_f64();
        }
        Ok(x.to_vec())
    }

    fn qk(&self, qkv: &[f32], scores: &mut [f32], batch: usize, seq: usize, add_mask: &[f32]) {
        let spec = &self.spec;
        let hidden = spec.hidden;
        let head_dim = spec.head_dim;
        let nv = head_dim / 8;
        for b in 0..batch {
            for h in 0..spec.heads {
                for i in 0..seq {
                    let q = &qkv[(b * seq + i) * 3 * hidden + h * head_dim..][..head_dim];
                    let row_base = ((b * spec.heads + h) * seq + i) * seq;
                    for j in 0..seq {
                        let k =
                            &qkv[(b * seq + j) * 3 * hidden + hidden + h * head_dim..][..head_dim];
                        let mut lanes = [0.0f32; 8];
                        for l in 0..8 {
                            lanes[l] = q[l] * k[l];
                        }
                        for v in 1..nv {
                            let off = v * 8;
                            for l in 0..8 {
                                lanes[l] = q[off + l].mul_add(k[off + l], lanes[l]);
                            }
                        }
                        scores[row_base + j] = kernels::hsum8(lanes) + add_mask[b * seq + j];
                    }
                }
            }
        }
    }

    fn av(&self, scores: &[f32], qkv: &[f32], ctx: &mut [f32], batch: usize, seq: usize) {
        let spec = &self.spec;
        let hidden = spec.hidden;
        let head_dim = spec.head_dim;
        for b in 0..batch {
            for h in 0..spec.heads {
                for i in 0..seq {
                    let a = &scores[((b * spec.heads + h) * seq + i) * seq..][..seq];
                    let o = &mut ctx[(b * seq + i) * hidden + h * head_dim..][..head_dim];
                    for v in o.iter_mut() {
                        *v = 0.0;
                    }
                    for j in 0..seq {
                        let aj = a[j];
                        let vrow = &qkv[(b * seq + j) * 3 * hidden + 2 * hidden + h * head_dim..]
                            [..head_dim];
                        for d in 0..head_dim {
                            o[d] = aj.mul_add(vrow[d], o[d]);
                        }
                    }
                }
            }
        }
    }

    fn pool(&self, h: &[f32], mask: &[f32], batch: usize, seq: usize) -> Vec<f32> {
        let hidden = self.spec.hidden;
        let mut out = vec![0.0f32; batch * hidden];
        for b in 0..batch {
            if self.spec.cls_pooling {
                if mask[b * seq] > 0.0 {
                    let src = &h[(b * seq) * hidden..(b * seq + 1) * hidden];
                    out[b * hidden..(b + 1) * hidden].copy_from_slice(src);
                }
            } else {
                let mut acc = vec![0.0f64; hidden];
                let mut count = 0.0f64;
                for s in 0..seq {
                    if mask[b * seq + s] > 0.0 {
                        count += 1.0;
                        let row = &h[(b * seq + s) * hidden..(b * seq + s + 1) * hidden];
                        for c in 0..hidden {
                            acc[c] += row[c] as f64;
                        }
                    }
                }
                let denom = if count > 0.0 { count } else { 1.0 };
                for c in 0..hidden {
                    out[b * hidden + c] = (acc[c] / denom) as f32;
                }
            }
            if self.spec.l2_normalize {
                let mut n2 = 0.0f64;
                for c in 0..hidden {
                    let v = out[b * hidden + c] as f64;
                    n2 += v * v;
                }
                let inv = if n2 > 0.0 { 1.0 / n2.sqrt() } else { 0.0 };
                for c in 0..hidden {
                    out[b * hidden + c] = (out[b * hidden + c] as f64 * inv) as f32;
                }
            }
        }
        out
    }
}

fn rope_cos_sin(seq: usize, inv_freq: &[f32], head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; seq * head_dim];
    let mut sin = vec![0.0f32; seq * head_dim];
    for t in 0..seq {
        for i in 0..half {
            let angle = (t as f32) * inv_freq[i];
            let c = angle.cos();
            let s = angle.sin();
            cos[t * head_dim + i] = c;
            cos[t * head_dim + i + half] = c;
            sin[t * head_dim + i] = s;
            sin[t * head_dim + i + half] = s;
        }
    }
    (cos, sin)
}

fn apply_rope_qkv(
    qkv: &mut [f32],
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    hidden: usize,
    cos_sin: &(Vec<f32>, Vec<f32>),
) {
    let (cos, sin) = cos_sin;
    let half = head_dim / 2;
    for b in 0..batch {
        for s in 0..seq {
            let c = &cos[s * head_dim..(s + 1) * head_dim];
            let sn = &sin[s * head_dim..(s + 1) * head_dim];
            for h in 0..heads {
                let base = (b * seq + s) * 3 * hidden;
                let q = &mut qkv[base + h * head_dim..base + (h + 1) * head_dim];
                rotate_inplace(q, c, sn, half);
                let k = &mut qkv[base + hidden + h * head_dim..base + hidden + (h + 1) * head_dim];
                rotate_inplace(k, c, sn, half);
            }
        }
    }
}

fn rotate_inplace(x: &mut [f32], cos: &[f32], sin: &[f32], half: usize) {
    let mut tmp = vec![0.0f32; x.len()];
    for i in 0..half {
        let x1 = x[i];
        let x2 = x[i + half];
        tmp[i] = -x2;
        tmp[i + half] = x1;
    }
    for i in 0..x.len() {
        x[i] = x[i] * cos[i] + tmp[i] * sin[i];
    }
}

fn gelu_erf_f32(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x / std::f32::consts::SQRT_2))
}
