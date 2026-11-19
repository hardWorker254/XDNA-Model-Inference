//! The GTE forward pass (arch `gte_new_rope_geglu`), on the host.
//!
//! Ported from `reference/encoder_gte.py` and matched to the packed container
//! layout. Post-LN, RoPE with NTK-scaled inv_freq (baked in config), gated GEGLU
//! FFN with GELU on the gate half, CLS pooling, no final norm.

use crate::kernels;
use crate::parallel;
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
    /// The four op names the array backend looks up, built once at open.
    /// They used to be `format!`ed inside the layer loop of every request.
    names: Vec<[String; 4]>,
}

/// The scratch a GTE forward pass needs, owned by the caller and reused across
/// requests.
///
/// A host forward used to allocate seven multi-megabyte buffers and both RoPE
/// tables on every call. The model's shapes are fixed once it is opened, so
/// every one of those allocations can be grown once and then kept. The server
/// holds one of these per engine and lends it to each batch; a short-lived
/// caller (the CLI, a test) builds a local one and drops it afterwards.
#[derive(Debug, Clone, Default)]
pub struct GteWorkspace {
    x: Vec<f32>,
    add_mask: Vec<f32>,
    attn_mask: Vec<f32>,
    qkv: Vec<f32>,
    ctx: Vec<f32>,
    proj: Vec<f32>,
    up_gate: Vec<f32>,
    act: Vec<f32>,
    down: Vec<f32>,
    scores: Vec<f32>,
    zeros_bias: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    /// How many positions of `cos`/`sin` are already filled. Positions are
    /// computed from the first one onwards, so a later, longer sequence only
    /// computes the NEW tail and a shorter one reuses the prefix.
    rope_seq: usize,
    pool_acc: Vec<f64>,
}

/// Grow `buf` to `len`, keeping what is already there.
///
/// Every buffer here is either overwritten in full by its kernel or, in the
/// case of `zeros_bias`, only ever read as zeros. `resize` zero-fills only
/// newly grown elements, so the steady state is a no-op and a shape change is
/// still safe.
#[inline]
fn grown(buf: &mut Vec<f32>, len: usize) -> &mut [f32] {
    buf.resize(len, 0.0);
    buf
}

/// The RoPE tables up to `seq`, cached across forward passes.
///
/// `cos`/`sin` are a pure function of the token position, the model's baked
/// inverse frequencies and the head size: none of them changes between
/// requests. The server narrows `seq` per batch, so positions arrive in an
/// order that a single "recompute if different" flag would rebuild every
/// time; growing the table monotonically and keeping the prefix avoids that.
fn ensure_rope<'a>(
    cos: &'a mut Vec<f32>,
    sin: &'a mut Vec<f32>,
    cached: &mut usize,
    seq: usize,
    inv_freq: &[f32],
    head_dim: usize,
) -> (&'a [f32], &'a [f32]) {
    let half = head_dim / 2;
    if seq > *cached {
        cos.resize(seq * head_dim, 0.0);
        sin.resize(seq * head_dim, 0.0);
        for t in *cached..seq {
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
        *cached = seq;
    }
    (&cos[..seq * head_dim], &sin[..seq * head_dim])
}

impl GteModel {
    pub fn open(path: &Path) -> Result<Self> {
        let container = npum_format::Container::open(path).map_err(|source| Error::Container {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_container(&container)
    }

    /// Read the model out of an ALREADY OPEN container.
    ///
    /// The embed command opens one container and threads it through the arch
    /// check, the status block, the tokenizer and the model; opening it again
    /// here would map the same multi-GB file a second time.
    pub fn from_container(container: &npum_format::Container) -> Result<Self> {
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

        let names = (0..spec.layers)
            .map(|l| {
                let p = format!("layer.{l}");
                [
                    format!("{p}.qkv"),
                    format!("{p}.attn_out"),
                    format!("{p}.ffn_up"),
                    format!("{p}.ffn_down"),
                ]
            })
            .collect();
        Ok(Self {
            spec,
            weights: Weights {
                word,
                token_type,
                ln_gamma,
                ln_beta,
                layers,
            },
            names,
        })
    }

    /// Embed with a fresh workspace. Kept for callers that run one batch and
    /// are gone (the CLI, tests); a server should keep a workspace and use
    /// [`GteModel::embed_timed_with_workspace`] instead.
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
        let mut workspace = GteWorkspace::default();
        self.embed_timed_with_workspace(
            ops,
            timings,
            ids,
            mask,
            batch,
            seq,
            array,
            eltwise,
            &mut workspace,
        )
    }

    /// Embed, borrowing scratch from `workspace` so the buffers survive the
    /// call and the next batch pays no allocation for them.
    pub fn embed_timed_with_workspace(
        &self,
        ops: &mut dyn Ops,
        timings: &mut Timings,
        ids: &[u32],
        mask: &[u32],
        batch: usize,
        seq: usize,
        array: ArrayTextRef<'_, '_>,
        _eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
        workspace: &mut GteWorkspace,
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
        let h = self.forward(ops, ids, mask, batch, seq, timings, array, workspace)?;
        Ok(self.pool(&h, batch, seq, workspace))
    }

    fn forward(
        &self,
        ops: &mut dyn Ops,
        ids: &[u32],
        mask: &[u32],
        batch: usize,
        seq: usize,
        timings: &mut Timings,
        mut array: ArrayTextRef<'_, '_>,
        ws: &mut GteWorkspace,
    ) -> Result<Vec<f32>> {
        let spec = &self.spec;
        let hidden = spec.hidden;
        let rows = batch * seq;
        let three_hidden = 3 * hidden;
        let score_rows = batch * spec.heads * seq;
        let gemm_on = array.as_deref().is_some_and(|a| a.gemm_on_array());

        // Embeddings: word + token_type[0], then embeddings.LayerNorm.
        // No absolute position addition (RoPE). position_embedding tensor in container is zeroed.
        let x = grown(&mut ws.x, rows * hidden);
        for r in 0..rows {
            let id = ids[r] as usize;
            let wv = &self.weights.word[id * hidden..(id + 1) * hidden];
            let tv = &self.weights.token_type[..hidden];
            let dst = &mut x[r * hidden..(r + 1) * hidden];
            for c in 0..hidden {
                dst[c] = wv[c] + tv[c];
            }
        }
        let add_mask = grown(&mut ws.add_mask, rows);
        let attn_mask = grown(&mut ws.attn_mask, rows);
        for r in 0..rows {
            if mask[r] > 0 {
                attn_mask[r] = 1.0;
            } else {
                add_mask[r] = -1.0e30;
            }
        }
        let t0 = std::time::Instant::now();
        ops.layernorm_rows(
            x,
            rows,
            hidden,
            &self.weights.ln_gamma,
            &self.weights.ln_beta,
        )?;
        timings.host_layernorm += t0.elapsed().as_secs_f64();

        // The RoPE tables are their own cache: growing them mutably borrows
        // only these three fields, leaving every scratch buffer below free.
        let (cos, sin) = ensure_rope(
            &mut ws.cos,
            &mut ws.sin,
            &mut ws.rope_seq,
            seq,
            &spec.rope_inv_freq,
            spec.head_dim,
        );

        // Every one of these was a fresh multi-megabyte allocation per call.
        // They are grown in place and reused; each is written IN FULL before
        // it is read. `zeros_bias` is the exception -- it is READ as a bias --
        // so it is only ever extended with zeros and never written.
        let qkv = grown(&mut ws.qkv, rows * three_hidden);
        let ctx = grown(&mut ws.ctx, rows * hidden);
        let proj = grown(&mut ws.proj, rows * hidden);
        let up_gate = grown(&mut ws.up_gate, rows * 2 * spec.ffn);
        let act = grown(&mut ws.act, rows * spec.ffn);
        let down = grown(&mut ws.down, rows * hidden);
        let scores = grown(&mut ws.scores, score_rows * seq);
        let zeros_bias = grown(&mut ws.zeros_bias, 2 * spec.ffn);

        for (index, layer) in self.weights.layers.iter().enumerate() {
            let names = &self.names[index];
            // QKV projection
            match array.as_deref_mut().filter(|_| gemm_on) {
                Some(array) => {
                    let t = std::time::Instant::now();
                    array.gemm(
                        &names[0],
                        x,
                        rows,
                        hidden,
                        three_hidden,
                        Some(&layer.qkv_bias),
                        qkv,
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
                        qkv,
                    )?;
                    if ops.place() != npum_model::Placement::Host {
                        timings.gpu += t.elapsed().as_secs_f64();
                    }
                }
            }
            // Apply RoPE to Q and K
            apply_rope_qkv(qkv, batch, seq, spec.heads, spec.head_dim, hidden, cos, sin);

            // ATTENTION: one fused host kernel, or the three separate passes
            // when a device owns the softmax.
            let t_attn = std::time::Instant::now();
            if ops.place() == npum_model::Placement::Host {
                // QK^T + softmax + softmax.V in one pass over a single score
                // row. Bit-identical to the three passes below (the same
                // dot8, the same softmax body, the same V accumulation), but
                // it never materialises the seq-by-seq score matrix.
                kernels::fused_attention(
                    qkv,
                    ctx,
                    add_mask,
                    batch,
                    seq,
                    spec.heads,
                    spec.head_dim,
                    hidden,
                )?;
                timings.attention += t_attn.elapsed().as_secs_f64();
            } else {
                let t_qk = std::time::Instant::now();
                self.qk(qkv, scores, batch, seq, add_mask);
                let qk_s = t_qk.elapsed().as_secs_f64();
                timings.qk += qk_s;
                let t_sm = std::time::Instant::now();
                ops.softmax_rows(scores, score_rows, seq)?;
                timings.host_softmax += t_sm.elapsed().as_secs_f64();
                let t_av = std::time::Instant::now();
                self.av(scores, qkv, ctx, batch, seq);
                let av_s = t_av.elapsed().as_secs_f64();
                timings.av += av_s;
                timings.attention += qk_s + av_s;
            }

            // Output projection + residual
            match array.as_deref_mut().filter(|_| gemm_on) {
                Some(array) => {
                    let t = std::time::Instant::now();
                    array.gemm(
                        &names[1],
                        &ctx,
                        rows,
                        hidden,
                        hidden,
                        Some(&layer.attn_out_bias),
                        proj,
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
                        proj,
                    )?;
                    if ops.place() != npum_model::Placement::Host {
                        timings.gpu += t.elapsed().as_secs_f64();
                    }
                }
            }
            ops.add_inplace(x, &proj[..x.len()])?;
            // attn_ln (post-LN)
            let t_ln = std::time::Instant::now();
            ops.layernorm_rows(x, rows, hidden, &layer.ln1_gamma, &layer.ln1_beta)?;
            timings.host_layernorm += t_ln.elapsed().as_secs_f64();

            // Gated GEGLU MLP: up_gate = x @ W (no bias), split [up|gate], gelu(gate)*up, down
            match array.as_deref_mut().filter(|_| gemm_on) {
                Some(array) => {
                    let t = std::time::Instant::now();
                    array.gemm(&names[2], x, rows, hidden, 2 * spec.ffn, None, up_gate)?;
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
                        up_gate,
                    )?;
                    if ops.place() != npum_model::Placement::Host {
                        timings.gpu += t.elapsed().as_secs_f64();
                    }
                }
            }
            let inter = spec.ffn;
            let geglu = |r: usize, out: &mut [f32]| {
                let base = r * 2 * inter;
                for (j, slot) in out.iter_mut().enumerate() {
                    let upv = up_gate[base + j];
                    let gate = up_gate[base + inter + j];
                    *slot = upv * gelu_erf_f32(gate);
                }
            };
            // Every output element is one `erff` on its own gate, so a row
            // split crosses no reduction and keeps the exact libm result.
            match parallel::chunk_rows(rows, (rows as i64) * (inter as i64), parallel::threads()) {
                Some(chunk) => {
                    parallel::for_each_chunk_mut(
                        &mut act[..rows * inter],
                        chunk * inter,
                        |ci, part| {
                            let first = ci * chunk;
                            for (i, row) in part.chunks_exact_mut(inter).enumerate() {
                                geglu(first + i, row);
                            }
                        },
                    );
                }
                None => {
                    for (r, part) in act[..rows * inter].chunks_exact_mut(inter).enumerate() {
                        geglu(r, part);
                    }
                }
            }
            match array.as_deref_mut().filter(|_| gemm_on) {
                Some(array) => {
                    let t = std::time::Instant::now();
                    array.gemm(
                        &names[3],
                        &act,
                        rows,
                        inter,
                        hidden,
                        Some(&layer.ffn_down_bias),
                        down,
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
                        down,
                    )?;
                    if ops.place() != npum_model::Placement::Host {
                        timings.gpu += t.elapsed().as_secs_f64();
                    }
                }
            }
            ops.add_inplace(x, &down[..x.len()])?;
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
        let rows = batch * spec.heads * seq;
        let work = (rows as i64) * (seq as i64) * (head_dim as i64) * 2;
        let qk_row = |row: usize, dst: &mut [f32]| {
            let p = row / seq;
            let i = row % seq;
            let b = p / spec.heads;
            let h = p % spec.heads;
            let q = &qkv[(b * seq + i) * 3 * hidden + h * head_dim..][..head_dim];
            for (j, slot) in dst.iter_mut().enumerate() {
                let k = &qkv[(b * seq + j) * 3 * hidden + hidden + h * head_dim..][..head_dim];
                // `dot8` keeps the first-eight-products / FMA-lanes / hsum8
                // order of the old hand-rolled loop, bit for bit.
                *slot = kernels::dot8(q, k) + add_mask[b * seq + j];
            }
        };
        match parallel::chunk_rows(rows, work, parallel::threads()) {
            Some(chunk) => {
                parallel::for_each_chunk_mut(&mut scores[..rows * seq], chunk * seq, |ci, part| {
                    let first = ci * chunk;
                    for (i, row) in part.chunks_exact_mut(seq).enumerate() {
                        qk_row(first + i, row);
                    }
                });
            }
            None => {
                for (row, part) in scores[..rows * seq].chunks_exact_mut(seq).enumerate() {
                    qk_row(row, part);
                }
            }
        }
    }

    fn av(&self, scores: &[f32], qkv: &[f32], ctx: &mut [f32], batch: usize, seq: usize) {
        let spec = &self.spec;
        let hidden = spec.hidden;
        let head_dim = spec.head_dim;
        let rows = batch * seq;
        let work = (rows as i64) * (spec.heads as i64) * (seq as i64) * (head_dim as i64) * 2;
        let av_row = |row: usize, out: &mut [f32]| {
            let b = row / seq;
            let i = row % seq;
            for h in 0..spec.heads {
                let a = &scores[((b * spec.heads + h) * seq + i) * seq..][..seq];
                let o = &mut out[h * head_dim..][..head_dim];
                o.fill(0.0);
                for (j, &aj) in a.iter().enumerate() {
                    let vrow =
                        &qkv[(b * seq + j) * 3 * hidden + 2 * hidden + h * head_dim..][..head_dim];
                    kernels::weighted_accumulate(o, vrow, aj);
                }
            }
        };
        match parallel::chunk_rows(rows, work, parallel::threads()) {
            Some(chunk) => {
                parallel::for_each_chunk_mut(
                    &mut ctx[..rows * hidden],
                    chunk * hidden,
                    |ci, part| {
                        let first = ci * chunk;
                        for (i, row) in part.chunks_exact_mut(hidden).enumerate() {
                            av_row(first + i, row);
                        }
                    },
                );
            }
            None => {
                for (row, part) in ctx[..rows * hidden].chunks_exact_mut(hidden).enumerate() {
                    av_row(row, part);
                }
            }
        }
    }

    fn pool(&self, h: &[f32], batch: usize, seq: usize, ws: &mut GteWorkspace) -> Vec<f32> {
        let hidden = self.spec.hidden;
        let mask = &ws.attn_mask[..batch * seq];
        let mut out = vec![0.0f32; batch * hidden];
        for b in 0..batch {
            if self.spec.cls_pooling {
                if mask[b * seq] > 0.0 {
                    let src = &h[(b * seq) * hidden..(b * seq + 1) * hidden];
                    out[b * hidden..(b + 1) * hidden].copy_from_slice(src);
                }
            } else {
                let acc = &mut ws.pool_acc;
                acc.resize(hidden, 0.0);
                acc.fill(0.0);
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

fn apply_rope_qkv(
    qkv: &mut [f32],
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    hidden: usize,
    cos: &[f32],
    sin: &[f32],
) {
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
    // The old form built a `tmp` Vec on EVERY call (twice per head per
    // layer). `tmp[i] = -x2` and `tmp[i + half] = x1` are two scalars; keep
    // them as scalars and keep the exact multiply/add order -- in particular
    // `(-x2) * sin`, which is what the temporary encoded. No FMA is introduced.
    for i in 0..half {
        let x1 = x[i];
        let x2 = x[i + half];
        let j = i + half;
        x[i] = x1 * cos[i] + (-x2) * sin[i];
        x[j] = x2 * cos[j] + x1 * sin[j];
    }
}

fn gelu_erf_f32(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x / std::f32::consts::SQRT_2))
}

#[cfg(test)]
mod rope_cache_tests {
    use super::*;

    #[test]
    fn the_rope_cache_grows_without_recomputing_or_changing_the_prefix() {
        let inv_freq = vec![0.5f32, 0.25, 0.125, 0.0625];
        let head_dim = 8;
        let (mut cos, mut sin, mut cached) = (Vec::new(), Vec::new(), 0usize);

        let (c8, s8) = ensure_rope(&mut cos, &mut sin, &mut cached, 8, &inv_freq, head_dim);
        assert_eq!(cached, 8);
        let first = c8.to_vec();
        assert_eq!(
            c8[0],
            (0.0f32 * inv_freq[0]).cos(),
            "position zero is cos(0)"
        );
        assert_eq!(c8[1], c8[5], "the cos halves are duplicated");
        assert_eq!(s8[0], 0.0, "position zero is sin(0)");

        // A later, longer sequence keeps the prefix BIT for bit and only
        // fills the new positions.
        let (c16, _) = ensure_rope(&mut cos, &mut sin, &mut cached, 16, &inv_freq, head_dim);
        assert_eq!(cached, 16);
        assert_eq!(&c16[..8 * head_dim], &first, "the prefix was not recomputed");
        assert_eq!(c16[8 * head_dim], (8.0f32 * inv_freq[0]).cos());
        let c16_copy = c16.to_vec();

        // A shorter sequence reads a prefix of the same table.
        let (c4, _) = ensure_rope(&mut cos, &mut sin, &mut cached, 4, &inv_freq, head_dim);
        assert_eq!(cached, 16, "shrinking does not rebuild the table");
        assert_eq!(c4.len(), 4 * head_dim);
        assert_eq!(c4, &c16_copy[..4 * head_dim]);
    }
}
