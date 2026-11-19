//! The BERT-family forward pass, on the host.
//!
//! # This is a port, not a reimplementation
//!
//! The reference runtime's `BertEncoder::run()` is the definition of what a BERT
//! container means at run time, and the golden transcripts were recorded
//! through it. So the shape of the pass below is the reference's shape, including
//! the parts that are not how a transformer is usually written:
//!
//! * the LayerNorm comes AFTER the attention and AFTER the FFN (post-LN),
//!   and the embedding LayerNorm comes before the first layer;
//! * the residual is added BEFORE the LayerNorm, not after;
//! * the attention's scale is NOT applied here: the packer folded
//!   `1/sqrt(head_dim)` into Q's weight and bias, and applying it again
//!   would compute a different model;
//! * the mean pooling accumulates in DOUBLE and the L2 normalisation is a
//!   double square root;
//! * the GELU is the degree-8 polynomial, not the exact erf.
//!
//! # What is NOT here
//!
//! The rotary variants (arch=2, nomic) and the gated FFN are the same walk
//! with a RoPE step and a SwiGLU in the middle. They are not implemented
//! yet, and the arch check says so by name rather than running the wrong
//! arithmetic for them.

use crate::kernels;
use crate::kernels::Isa;
use crate::ops::CpuOps;
use npum_model::Ops;
use npum_model::array::{ArrayEltwise, ArrayTextRef, AttentionShape};
use npum_model::dispatcher::Placement;
use std::path::Path;

/// What can go wrong while running a BERT container.
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

    #[error(
        "the softmax design was built with cols {cols}, but this container's sequence is {seq} \
             wide. The design normalises whatever row width it was compiled for and cannot be \
             handed a longer one, so the scores would be computed at the wrong length. Re-export \
             it with sm_cols >= seq."
    )]
    SoftmaxTooNarrow { cols: usize, seq: usize },

    #[error(
        "the softmax design's rows are {softmax} wide, and the attention's score row is {attn} \
             wide. The softmax design reduces along the whole row, so it has to be the width of \
             the score row it is handed."
    )]
    SoftmaxWidthMismatch { softmax: usize, attn: usize },

    #[error(transparent)]
    Kernel(#[from] npum_model::Error),
}

/// The result type of this module.
pub type Result<T> = std::result::Result<T, Error>;

/// The architectures this forward pass implements.
pub const IMPLEMENTED_ARCH: &str = "bert_abs_gelu_postln";

/// What the forward pass spent its time on.
///
/// The reference's `BertEncoder::reset_timers` / `timings()` pair exists for one
/// reason: the status block's TIME SPLIT is the measurement every
/// optimisation question about this path starts from, and it is only
/// readable after the pass. The Rust measures the same four things -- the
/// host LayerNorm, the host softmax, the host GELU and the attention -- and
/// leaves the array's own rows at zero, because the array dispatched
/// nothing.
///
/// The attention is counted TWICE on purpose: `attn` is the whole block and
/// `qk` and `av` are its halves, and the reference says so in its own comment --
/// "count it once" -- because counting the block and its halves together
/// double-counts every attention pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    /// The host LayerNorm, in seconds.
    ///
    /// Zero when the LayerNorm is on the array: the row says "host", and a
    /// row that reports the array's time under the host's name is a row
    /// that lies about where the work happened.
    pub host_layernorm: f64,
    /// The host softmax, in seconds.
    ///
    /// Zero when the softmax is on the array, for the same reason.
    pub host_softmax: f64,
    /// The host GELU, in seconds.
    ///
    /// Zero when the GELU is on the array, for the same reason.
    pub host_gelu: f64,
    /// The whole attention block, in seconds.
    ///
    /// THE reference's `t_attn` IS QK^T AND softmax.V, and not the softmax: the
    /// reference counts the softmax in its own row, and counting it here as well
    /// counts it twice -- once in this row and once in the softmax's.
    pub attention: f64,
    /// The QK^T half of the attention, in seconds.
    pub qk: f64,
    /// The softmax.V half of the attention, in seconds.
    pub av: f64,
    /// The array's own time, in seconds.
    ///
    /// The projections and the elementwise passes, measured around the
    /// call. It is zero on a host run -- which is what the status block's
    /// split is read against -- and the reference's own split puts the
    /// elementwise passes in the same row as the projections', because
    /// both are the array's time.
    pub array_gemm: f64,
    /// The GPU's own time, in seconds.
    ///
    /// Zero unless the run put the arithmetic on the GPU. The row it
    /// belongs to is the GPU's, and the host's rows stay zero in that
    /// case: a row named "host LayerNorm" that holds the GPU's time is a
    /// row that lies about where the work happened.
    pub gpu: f64,
}

impl Timings {
    /// The sum of the rows the status block prints.
    ///
    /// The reference's `busy`: the array's conversion, upload, dispatch, download
    /// and bias rows -- all zero on the host path -- plus the host rows and
    /// the attention.
    pub fn busy(&self) -> f64 {
        self.host_layernorm
            + self.host_softmax
            + self.host_gelu
            + self.attention
            + self.array_gemm
            + self.gpu
    }
}

/// The BERT family's shapes, as the container's config states them.
#[derive(Debug, Clone)]
pub struct BertSpec {
    /// How many transformer layers.
    pub layers: usize,
    /// How many attention heads.
    pub heads: usize,
    /// The model's width.
    pub hidden: usize,
    /// One head's width.
    pub head_dim: usize,
    /// The FFN's width.
    pub ffn: usize,
    /// The vocabulary's size.
    pub vocab: usize,
    /// How many positions the model has.
    pub max_seq_len: usize,
    /// Whether the pooling takes the `[CLS]` row instead of the mean.
    pub cls_pooling: bool,
    /// Whether the vectors are L2-normalised.
    pub l2_normalize: bool,
}

impl BertSpec {
    /// Read the shapes from a container's config.
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
        let pooling = config.get_str("pooling").unwrap_or("mean");
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
        })
    }
}

/// Reusable buffers for a BERT forward pass.
///
/// The forward pass touches several large scratch buffers whose sizes depend
/// on `batch * seq` (and the model's shapes). Allocating them on every call
/// is measurable for server and batched workloads, so a caller that runs the
/// same model repeatedly can keep one workspace and let the methods below
/// grow it as needed.
#[derive(Debug, Clone, Default)]
pub struct BertWorkspace {
    x: Vec<f32>,
    add_mask: Vec<f32>,
    attn_mask: Vec<f32>,
    qkv: Vec<f32>,
    ctx: Vec<f32>,
    proj: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    scores: Vec<f32>,
    pool_acc: Vec<f64>,
}

impl BertWorkspace {
    /// A fresh, empty workspace.
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure(&mut self, spec: &BertSpec, batch: usize, seq: usize) {
        let rows = batch * seq;
        let hidden = spec.hidden;
        Self::grow_f32(&mut self.x, rows * hidden);
        Self::grow_f32(&mut self.add_mask, rows);
        Self::grow_f32(&mut self.attn_mask, rows);
        Self::grow_f32(&mut self.qkv, rows * 3 * hidden);
        Self::grow_f32(&mut self.ctx, rows * hidden);
        Self::grow_f32(&mut self.proj, rows * hidden);
        Self::grow_f32(&mut self.up, rows * spec.ffn);
        Self::grow_f32(&mut self.down, rows * hidden);
        Self::grow_f64(&mut self.pool_acc, hidden);
    }

    fn ensure_scores(&mut self, score_rows: usize, sc_cols: usize) {
        Self::grow_f32(&mut self.scores, score_rows.saturating_mul(sc_cols));
    }

    /// Grow a scratch buffer without zeroing the new tail.
    ///
    /// # SAFETY: why the unwritten tail is never read
    ///
    /// Every field of `BertWorkspace` is a scratch buffer whose used range is
    /// defined by the current batch and sequence, and every pass writes that
    /// range in full BEFORE any pass reads it:
    ///
    /// * `x` -- the embedding loop writes all `rows * hidden` values; the
    ///   residual adds, both LayerNorms and the pooling only read what was
    ///   written.
    /// * `add_mask` / `attn_mask` -- both are assigned for every row in the
    ///   mask loop before the forward pass.
    /// * `qkv`, `proj`, `up`, `down` -- GEMMs write their entire output range.
    /// * `ctx` -- the host's `av` zeroes each head's slice with `o.fill(0.0)`
    ///   before accumulating into it; the array's attention writes its whole
    ///   output.
    /// * `scores` -- `qk` writes the `seq` real keys and fills the padded tail
    ///   with `-1e30`; on the array path the buffer is not read at all.
    /// * `pool_acc` -- `pool_into` starts every batch with `acc.fill(0.0)`.
    ///
    /// Callers always slice to the logical length (`[..rows * hidden]`,
    /// `[..score_rows * sc_cols]`, `[..hidden]`), so a buffer left longer than
    /// the current batch is never observed. `f32`/`f64` have no drop glue and
    /// accept every bit pattern; `reserve` guarantees the allocation exists
    /// before `set_len` publishes it.
    ///
    /// Shrinking is a no-op: the bytes stay allocated and are overwritten on
    /// the next, larger batch instead of being zero-filled and reallocated.
    #[inline]
    fn grow_f32(buf: &mut Vec<f32>, len: usize) {
        if buf.len() < len {
            buf.reserve(len - buf.len());
            // SAFETY: see the audit above; the published range is written in
            // full before it is read.
            unsafe { buf.set_len(len) };
        }
    }

    #[inline]
    fn grow_f64(buf: &mut Vec<f64>, len: usize) {
        if buf.len() < len {
            buf.reserve(len - buf.len());
            // SAFETY: see the audit above.
            unsafe { buf.set_len(len) };
        }
    }
}

/// A GEMM operand as the host reads it: `[K, N]` in f32, row-major.
///
/// # THE CONTAINER STORES IT TWO WAYS
///
/// * an fp32 row-major filter -- the container's default, and the one the
///   CPU backend has always read;
/// * a bf16 panel, already padded to the design's widths and already tiled
///   into the MMAC's byte order -- the container's ARRAY operand, and the
///   one a container packed for the board carries.
///
/// The second is read by DE-TILING it with its own recorded layout and
/// widening it back to f32. So a host run of a bf16 container is the same
/// network as the array's run of it -- the same numbers, rounded -- and not
/// a second network.
///
/// # WHY THE HOST READS A PANEL AT ALL
///
/// Because the array and the host must be able to run the SAME container.
/// A container that only the array can read is a container whose host path
/// is a different model, and the two would disagree by exactly the width of
/// bf16 -- which is a difference nobody would notice until the numbers
/// mattered.
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
            // THE SHAPE: the PADDED one, which is the panel's own.
            let shape: Vec<i64> = entry.padded_shape.clone();
            if shape.len() != 2 {
                return Err(format!(
                    "{name}: a GEMM operand has a {}-dimensional padded shape; a dense operand \
                     is [K, N]",
                    shape.len()
                ));
            }
            if !entry.logical_shape.is_empty()
                && (entry.logical_shape[0] != shape[0] || entry.logical_shape[1] != shape[1])
            {
                return Err(format!(
                    "{name}: logical [{},{}] but padded [{},{}]. The host builds [K,N] from the \
                     padded shape and the caller indexes it with the logical one, so the two \
                     must agree.",
                    entry.logical_shape[0], entry.logical_shape[1], shape[0], shape[1]
                ));
            }
            let (k, n) = (shape[0] as usize, shape[1] as usize);
            if k == 0 || n == 0 {
                return Err(format!("{name}: a GEMM operand with a zero dimension"));
            }
            // THE LAYOUT, WHICH IS THE PANEL'S OWN.
            let layout = entry
                .layout
                .clone()
                .ok_or_else(|| format!("{name}: the payload carries no layout. The layout is what says where each element sits, so without it these bytes are a right count in an unknown order."))?;
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
            "{name}: a {other} operand cannot be run on the host. This build widens BF16 and \
             copies F32; anything else would mean inventing a conversion, and a guessed \
             conversion is a wrong number in a plausible range."
        )),
    }
}

/// One layer's weights.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// The fused QKV weight, `[hidden, 3 * hidden]`.
    pub qkv: Vec<f32>,
    /// The fused QKV bias, `[3 * hidden]`.
    pub qkv_bias: Vec<f32>,
    /// The attention output projection, `[hidden, hidden]`.
    pub attn_out: Vec<f32>,
    /// The attention output bias, `[hidden]`.
    pub attn_out_bias: Vec<f32>,
    /// The first LayerNorm's gain.
    pub ln1_gamma: Vec<f32>,
    /// The first LayerNorm's offset.
    pub ln1_beta: Vec<f32>,
    /// The first LayerNorm's gain and offset concatenated, `[gamma | beta]`.
    pub ln1_params: Vec<f32>,
    /// The FFN's up projection, `[hidden, ffn]`.
    pub ffn_up: Vec<f32>,
    /// The FFN's up bias.
    pub ffn_up_bias: Vec<f32>,
    /// The FFN's down projection, `[ffn, hidden]`.
    pub ffn_down: Vec<f32>,
    /// The FFN's down bias.
    pub ffn_down_bias: Vec<f32>,
    /// The second LayerNorm's gain.
    pub ln2_gamma: Vec<f32>,
    /// The second LayerNorm's offset.
    pub ln2_beta: Vec<f32>,
    /// The second LayerNorm's gain and offset concatenated, `[gamma | beta]`.
    pub ln2_params: Vec<f32>,
}

/// The model's weights, read out of a container.
#[derive(Debug, Clone)]
pub struct BertWeights {
    /// The token embedding, `[vocab, hidden]`.
    pub word: Vec<f32>,
    /// The position embedding, `[max_seq_len, hidden]`.
    pub position: Vec<f32>,
    /// The token-type embedding, `[2, hidden]`.
    pub token_type: Vec<f32>,
    /// The embedding LayerNorm's gain.
    pub ln_gamma: Vec<f32>,
    /// The embedding LayerNorm's offset.
    pub ln_beta: Vec<f32>,
    /// The embedding LayerNorm's gain and offset concatenated, `[gamma | beta]`.
    pub ln_params: Vec<f32>,
    /// The layers, in order.
    pub layers: Vec<LayerWeights>,
}

/// A BERT model, ready to run.
#[derive(Debug, Clone)]
pub struct BertModel {
    /// The shapes.
    pub spec: BertSpec,
    /// The weights.
    pub weights: BertWeights,
    gemm_names_qkv: Vec<String>,
    gemm_names_attn_out: Vec<String>,
    gemm_names_ffn_up: Vec<String>,
    gemm_names_ffn_down: Vec<String>,
}

impl BertModel {
    /// Open a container and read the model out of it.
    ///
    /// Every shape is read from the container and every weight is checked
    /// against the shape: a container that is missing a tensor is an error
    /// that names the tensor, and one whose tensor is the wrong size is an
    /// error that names both sizes. A default here would be
    /// indistinguishable from a correct value, and this runtime has shipped
    /// six bugs of exactly that shape.
    pub fn open(path: &Path) -> Result<Self> {
        let container = npum_format::Container::open(path).map_err(|source| Error::Container {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_container(&container)
    }

    /// Read the model out of an ALREADY OPEN container.
    ///
    /// The CLI opens one container per command and threads it through the
    /// arch check, the status block, the tokenizer and the model. Re-opening
    /// it here would map a multi-GB file again for every command.
    pub fn from_container(container: &npum_format::Container) -> Result<Self> {
        let arch = container.config_str("arch").unwrap_or("").to_string();
        if arch != IMPLEMENTED_ARCH {
            return Err(Error::WrongArch {
                arch,
                implemented: IMPLEMENTED_ARCH,
            });
        }
        let spec = BertSpec::from_config(container.config())?;
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

        let read = |name: &str| -> Result<Vec<f32>> {
            let bytes = container
                .tensor_bytes(name)
                .map_err(|_| Error::NoTensor(name.to_string()))?;
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        // THE GEMM OPERANDS ARE READ DIFFERENTLY, because the container
        // stores them two ways.
        let read_operand = |name: &str| -> Result<Vec<f32>> {
            read_operand(&container, name).map_err(|reason| Error::BadOperand {
                name: name.to_string(),
                reason,
            })
        };
        let expect = |name: &str, values: Vec<f32>, want: usize| -> Result<Vec<f32>> {
            if values.len() != want {
                return Err(Error::WrongSize {
                    name: name.to_string(),
                    expected: want,
                    found: values.len(),
                });
            }
            Ok(values)
        };

        let hidden = spec.hidden;
        let ffn = spec.ffn;
        let three_hidden = 3 * hidden;

        let word = expect(
            "embeddings.word",
            read("embeddings.word")?,
            spec.vocab * hidden,
        )?;
        let position = expect(
            "embeddings.position",
            read("embeddings.position")?,
            spec.max_seq_len * hidden,
        )?;
        let token_type = expect(
            "embeddings.token_type",
            read("embeddings.token_type")?,
            2 * hidden,
        )?;
        let ln_gamma = expect(
            "embeddings.ln.weight",
            read("embeddings.ln.weight")?,
            hidden,
        )?;
        let ln_beta = expect("embeddings.ln.bias", read("embeddings.ln.bias")?, hidden)?;
        let ln_params = layernorm_params(&ln_gamma, &ln_beta);

        let mut layers = Vec::with_capacity(spec.layers);
        for l in 0..spec.layers {
            let p = format!("layer.{l}");
            let ln1_gamma = expect(
                &format!("{p}.ln1.weight"),
                read(&format!("{p}.ln1.weight"))?,
                hidden,
            )?;
            let ln1_beta = expect(
                &format!("{p}.ln1.bias"),
                read(&format!("{p}.ln1.bias"))?,
                hidden,
            )?;
            let ln2_gamma = expect(
                &format!("{p}.ln2.weight"),
                read(&format!("{p}.ln2.weight"))?,
                hidden,
            )?;
            let ln2_beta = expect(
                &format!("{p}.ln2.bias"),
                read(&format!("{p}.ln2.bias"))?,
                hidden,
            )?;
            layers.push(LayerWeights {
                qkv: expect(
                    &format!("{p}.qkv"),
                    read_operand(&format!("{p}.qkv"))?,
                    hidden * three_hidden,
                )?,
                qkv_bias: expect(
                    &format!("{p}.qkv.bias"),
                    read(&format!("{p}.qkv.bias"))?,
                    three_hidden,
                )?,
                attn_out: expect(
                    &format!("{p}.attn_out"),
                    read_operand(&format!("{p}.attn_out"))?,
                    hidden * hidden,
                )?,
                attn_out_bias: expect(
                    &format!("{p}.attn_out.bias"),
                    read(&format!("{p}.attn_out.bias"))?,
                    hidden,
                )?,
                ln1_params: layernorm_params(&ln1_gamma, &ln1_beta),
                ln1_gamma,
                ln1_beta,
                ffn_up: expect(
                    &format!("{p}.ffn_up"),
                    read_operand(&format!("{p}.ffn_up"))?,
                    hidden * ffn,
                )?,
                ffn_up_bias: expect(
                    &format!("{p}.ffn_up.bias"),
                    read(&format!("{p}.ffn_up.bias"))?,
                    ffn,
                )?,
                ffn_down: expect(
                    &format!("{p}.ffn_down"),
                    read_operand(&format!("{p}.ffn_down"))?,
                    ffn * hidden,
                )?,
                ffn_down_bias: expect(
                    &format!("{p}.ffn_down.bias"),
                    read(&format!("{p}.ffn_down.bias"))?,
                    hidden,
                )?,
                ln2_params: layernorm_params(&ln2_gamma, &ln2_beta),
                ln2_gamma,
                ln2_beta,
            });
        }

        let gemm_names_qkv = (0..spec.layers).map(|i| format!("layer.{i}.qkv")).collect();
        let gemm_names_attn_out = (0..spec.layers)
            .map(|i| format!("layer.{i}.attn_out"))
            .collect();
        let gemm_names_ffn_up = (0..spec.layers)
            .map(|i| format!("layer.{i}.ffn_up"))
            .collect();
        let gemm_names_ffn_down = (0..spec.layers)
            .map(|i| format!("layer.{i}.ffn_down"))
            .collect();

        Ok(Self {
            spec,
            weights: BertWeights {
                word,
                position,
                token_type,
                ln_gamma,
                ln_beta,
                ln_params,
                layers,
            },
            gemm_names_qkv,
            gemm_names_attn_out,
            gemm_names_ffn_up,
            gemm_names_ffn_down,
        })
    }

    /// Embed a batch of tokenized sequences.
    ///
    /// `ids` and `mask` are `batch * seq` long: the token ids and the
    /// attention mask, 1 for a real token and 0 for padding. The result is
    /// `batch * hidden` long: the pooled, normalised vectors.
    ///
    /// `ops` is WHERE the arithmetic happens: the host's own kernels
    /// (`crate::ops::CpuOps`) or the GPU's (`npum_vulkan::ops::VulkanOps`).
    /// The walk below is the same either way -- only the four operations
    /// change hands.
    pub fn embed(
        &self,
        ids: &[u32],
        mask: &[u32],
        batch: usize,
        seq: usize,
        ops: &mut dyn Ops,
        array: ArrayTextRef<'_, '_>,
        eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
    ) -> Result<Vec<f32>> {
        let mut timings = Timings::default();
        self.embed_timed_with(ops, &mut timings, ids, mask, batch, seq, array, eltwise)
    }

    /// Embed a batch, and report where the time went.
    pub fn embed_timed(
        &self,
        timings: &mut Timings,
        ids: &[u32],
        mask: &[u32],
        batch: usize,
        seq: usize,
        ops: &mut dyn Ops,
        array: ArrayTextRef<'_, '_>,
        eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
    ) -> Result<Vec<f32>> {
        self.embed_timed_with(ops, timings, ids, mask, batch, seq, array, eltwise)
    }

    /// Embed a batch of tokenized sequences, on a named ISA.
    ///
    /// The ISAs must agree BYTE FOR BYTE -- that is the gate the plan's
    /// 5.1a asks for, and `embed_on` is how it asks it: the same input
    /// through every ISA this host has, and the bytes compared.
    pub fn embed_on(
        &self,
        isa: Isa,
        ids: &[u32],
        mask: &[u32],
        batch: usize,
        seq: usize,
        array: ArrayTextRef<'_, '_>,
        eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
    ) -> Result<Vec<f32>> {
        let mut timings = Timings::default();
        self.embed_timed_with(
            &mut CpuOps::on(isa),
            &mut timings,
            ids,
            mask,
            batch,
            seq,
            array,
            eltwise,
        )
    }

    /// Embed a batch on a named ISA, and report where the time went.
    pub fn embed_timed_on(
        &self,
        isa: Isa,
        timings: &mut Timings,
        ids: &[u32],
        mask: &[u32],
        batch: usize,
        seq: usize,
        array: ArrayTextRef<'_, '_>,
        eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
    ) -> Result<Vec<f32>> {
        self.embed_timed_with(
            &mut CpuOps::on(isa),
            timings,
            ids,
            mask,
            batch,
            seq,
            array,
            eltwise,
        )
    }

    /// Embed a batch, with the arithmetic going through `ops`.
    ///
    /// This is the one that does the work. The two wrappers above are the
    /// host's own: one asks for the host's best ISA, the other for a named
    /// one, and the gate that compares the ISAs byte for byte uses the
    /// second.
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
        let mut ws = BertWorkspace::new();
        let mut out = vec![0.0f32; batch * self.spec.hidden];
        self.embed_timed_with_into(
            ops, timings, ids, mask, batch, seq, array, eltwise, &mut ws, &mut out,
        )?;
        Ok(out)
    }

    /// Embed a batch, reusing `workspace` and writing pooled vectors into `out`.
    ///
    /// `out` is resized to `batch * hidden`.
    pub fn embed_into(
        &self,
        ids: &[u32],
        mask: &[u32],
        batch: usize,
        seq: usize,
        ops: &mut dyn Ops,
        array: ArrayTextRef<'_, '_>,
        eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
        workspace: &mut BertWorkspace,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        let mut timings = Timings::default();
        self.embed_timed_with_into(
            ops,
            &mut timings,
            ids,
            mask,
            batch,
            seq,
            array,
            eltwise,
            workspace,
            out,
        )
    }

    /// Embed a batch, reusing `workspace`, writing pooled vectors into `out`,
    /// and reporting where the time went.
    ///
    /// `out` is resized to `batch * hidden`.
    pub fn embed_timed_into(
        &self,
        timings: &mut Timings,
        ids: &[u32],
        mask: &[u32],
        batch: usize,
        seq: usize,
        ops: &mut dyn Ops,
        array: ArrayTextRef<'_, '_>,
        eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
        workspace: &mut BertWorkspace,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        self.embed_timed_with_into(
            ops, timings, ids, mask, batch, seq, array, eltwise, workspace, out,
        )
    }

    /// Embed a batch, reusing `workspace`, with the arithmetic going through `ops`.
    ///
    /// This is the core implementation: scratch buffers come from `workspace`
    /// and pooled vectors are written into `out` (`batch * hidden`).
    pub fn embed_timed_with_into(
        &self,
        ops: &mut dyn Ops,
        timings: &mut Timings,
        ids: &[u32],
        mask: &[u32],
        batch: usize,
        seq: usize,
        array: ArrayTextRef<'_, '_>,
        mut eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
        workspace: &mut BertWorkspace,
        out: &mut Vec<f32>,
    ) -> Result<()> {
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
        workspace.ensure(&self.spec, batch, seq);
        if out.len() != batch * hidden {
            out.resize(batch * hidden, 0.0);
        }

        // 1. The embeddings: the token, the position and the token type,
        //    added in that order.
        for r in 0..rows {
            let s = r % seq;
            let id = ids[r] as usize;
            let wv = &self.weights.word[id * hidden..id * hidden + hidden];
            let pv = &self.weights.position[s * hidden..s * hidden + hidden];
            let tv = &self.weights.token_type[..hidden];
            let dst = &mut workspace.x[r * hidden..(r + 1) * hidden];
            for c in 0..hidden {
                dst[c] = wv[c] + pv[c] + tv[c];
            }
        }

        // 2. The additive mask: 0 for a real token, -1e30 for padding.
        for r in 0..rows {
            if mask[r] > 0 {
                workspace.attn_mask[r] = 1.0;
                workspace.add_mask[r] = 0.0;
            } else {
                workspace.attn_mask[r] = 0.0;
                workspace.add_mask[r] = -1.0e30;
            }
        }

        // 3. The forward pass.
        self.forward(
            ops,
            workspace,
            batch,
            seq,
            timings,
            array,
            eltwise.as_deref_mut(),
        )?;

        // 4. The pooling.
        self.pool_into(
            &workspace.x[..rows * hidden],
            &workspace.attn_mask[..rows],
            batch,
            seq,
            &mut workspace.pool_acc[..hidden],
            out,
        );
        Ok(())
    }

    /// The forward pass over `[rows, hidden]`, in place in `workspace.x`.
    ///
    /// The reference's `BertEncoder::run()`, step for step.
    fn forward(
        &self,
        ops: &mut dyn Ops,
        workspace: &mut BertWorkspace,
        batch: usize,
        seq: usize,
        timings: &mut Timings,
        mut array: ArrayTextRef<'_, '_>,
        mut eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
    ) -> Result<()> {
        let spec = &self.spec;
        let hidden = spec.hidden;
        let rows = batch * seq;
        let three_hidden = 3 * hidden;

        // THE SCORE MATRIX'S ROW WIDTH, AND WHO DECIDES IT.
        //
        // The host's softmax normalises a row of `seq` keys -- the real
        // key count, unpadded. The array's softmax normalises a row of
        // the DESIGN's width, which is the PADDED key count, and the
        // family has to lay the scores out at that width: a softmax that
        // normalises a different number of keys than the family laid out
        // is a softmax over the wrong row.
        //
        // And a design compiled for a NARROWER row than the sequence is
        // refused, because the design cannot be handed a longer row and
        // computing at the wrong length is worse than not computing.
        let sc_cols = match eltwise.as_deref().and_then(|e| e.softmax_cols()) {
            Some(cols) => {
                if cols < seq {
                    return Err(Error::SoftmaxTooNarrow { cols, seq });
                }
                cols
            }
            None => seq,
        };
        let score_rows = batch * spec.heads * seq;
        workspace.ensure_scores(score_rows, sc_cols);

        // WHICH PASSES ARE ON THE ARRAY, ASKED ONCE.
        //
        // The flag names the passes one by one, so a run can put the
        // LayerNorm on the array and leave the GELU and the softmax on the
        // host -- and the call site has to know which, because the host's
        // pass and the array's pass are different arithmetic and the
        // timings are counted in different rows.
        let ln_on = eltwise
            .as_deref()
            .is_some_and(|e| e.on_array(npum_model::array::EltOp::LayerNorm));
        let gelu_on = eltwise
            .as_deref()
            .is_some_and(|e| e.on_array(npum_model::array::EltOp::Gelu));
        let sm_on = eltwise
            .as_deref()
            .is_some_and(|e| e.on_array(npum_model::array::EltOp::Softmax));

        // THE ATTENTION, ON THE ARRAY OR ON THE HOST.
        //
        // The attention is the one pass whose two halves are the ARRAY's
        // decision: `attn` moves the two GEMMs and `softm` moves the
        // softmax, and a run that names one has not named the other.
        //
        // And when the attention is on the array, the SCORE ROWS are the
        // array's: they are at the PADDED key count, because the array's
        // score row has a column for every key the design was compiled for.
        // The softmax inside the attention is at that width too -- the
        // array's, when the softmax is on the array, and the host's at the
        // same width when it is not.
        let attn_on = array.as_deref().is_some_and(|a| a.on_array());
        // THE PROJECTIONS, ON THE ARRAY OR ON THE HOST.
        //
        // Asked and not assumed: the handle covers the projections AND the
        // attention, so a run that asked only for the attention has a
        // handle and no panels -- and the projections stay on the host.
        let gemm_on = array.as_deref().is_some_and(|a| a.gemm_on_array());
        if attn_on {
            let attn_cols = array.as_deref().and_then(|a| a.score_cols()).unwrap_or(seq);
            if let Some(cols) = eltwise.as_deref().and_then(|e| e.softmax_cols()) {
                if cols != attn_cols {
                    return Err(Error::SoftmaxWidthMismatch {
                        softmax: cols,
                        attn: attn_cols,
                    });
                }
            }
        }

        // The embedding LayerNorm.
        let t0 = std::time::Instant::now();
        if let Some(eltwise) = eltwise.as_deref_mut().filter(|_| ln_on) {
            eltwise.layernorm(
                &self.weights.ln_params,
                &mut workspace.x[..rows * hidden],
                rows,
            )?;
            // THE ARRAY'S TIME, IN THE ARRAY'S ROW.
            timings.array_gemm += t0.elapsed().as_secs_f64();
        } else {
            ops.layernorm_rows(
                &mut workspace.x[..rows * hidden],
                rows,
                hidden,
                &self.weights.ln_gamma,
                &self.weights.ln_beta,
            )?;
            let spent = t0.elapsed().as_secs_f64();
            match ops.place() {
                Placement::Host => timings.host_layernorm += spent,
                _ => timings.gpu += spent,
            }
        }

        for (index, layer) in self.weights.layers.iter().enumerate() {
            let (x_buf, qkv_buf, ctx_buf, proj_buf, up_buf, down_buf, scores_buf, add_mask_buf) = (
                &mut workspace.x[..rows * hidden],
                &mut workspace.qkv[..rows * three_hidden],
                &mut workspace.ctx[..rows * hidden],
                &mut workspace.proj[..rows * hidden],
                &mut workspace.up[..rows * spec.ffn],
                &mut workspace.down[..rows * hidden],
                &mut workspace.scores[..score_rows * sc_cols],
                &workspace.add_mask[..rows],
            );

            // The fused QKV projection.
            //
            // THE FOUR PROJECTIONS OF A LAYER ARE A CHAIN, NOT A BATCH. QKV
            // feeds the attention, whose output feeds the output projection,
            // whose residual feeds LayerNorm and then ffn_up, whose GELU feeds
            // ffn_down. There is no independent pair here for the array's
            // `gemm_batch` to overlap; keep these calls sequential. The batch
            // path is for families with genuinely independent projections.
            //
            // THE SEAM: the array, or the host, decided ONCE per projection
            // and in one place. The two paths are not the same numbers --
            // the array multiplies the weights in bf16 and the host in f32
            // -- so a run on the array is the same network with a coarser
            // weight, and the difference is the width of bf16.
            match array.as_deref_mut().filter(|_| gemm_on) {
                Some(array) => {
                    let t = std::time::Instant::now();
                    array.gemm(
                        &self.gemm_names_qkv[index],
                        x_buf,
                        rows,
                        hidden,
                        three_hidden,
                        Some(&layer.qkv_bias),
                        qkv_buf,
                    )?;
                    timings.array_gemm += t.elapsed().as_secs_f64();
                }
                None => {
                    let t = std::time::Instant::now();
                    ops.gemm_nt(
                        x_buf,
                        rows,
                        &layer.qkv,
                        hidden,
                        three_hidden,
                        &layer.qkv_bias,
                        qkv_buf,
                    )?;
                    // The host's own GEMM has NO row in the reference's split --
                    // the row that holds the projections' time is the
                    // array's -- so the host's projection time is not
                    // recorded. The GPU's is, because the GPU's row is
                    // its own.
                    if ops.place() != Placement::Host {
                        timings.gpu += t.elapsed().as_secs_f64();
                    }
                }
            }

            // THE ATTENTION: QK^T, the softmax, and softmax.V.
            //
            // TWO PATHS, AND THEY ARE ALTERNATIVES.
            //
            // The array's path takes the WHOLE block -- the two GEMMs and
            // the softmax between them -- because the array's score row is
            // at the PADDED key count and the host's is at the sequence, and
            // a softmax that normalises a different number of keys than the
            // row holds is a softmax over the wrong row. The host's path
            // below is the reference the array's one is measured against,
            // and not a fallback for it.
            if attn_on {
                let t_attn = std::time::Instant::now();
                let before = array
                    .as_deref()
                    .map(|a| a.attn_seconds())
                    .unwrap_or((0.0, 0.0));
                let shape = AttentionShape {
                    batch,
                    seq,
                    heads: spec.heads,
                    head_dim: spec.head_dim,
                    hidden,
                };
                if let Some(array) = array.as_deref_mut() {
                    // The softmax handle is the family's, and it is passed
                    // ONLY when the softmax is on the array: the attention
                    // asks the eltwise backend for a softmax, and a backend
                    // with no softmax design would refuse.
                    let softmax = if sm_on { eltwise.as_deref_mut() } else { None };
                    array.attention(qkv_buf, shape, add_mask_buf, softmax, ctx_buf)?;
                }
                let after = array
                    .as_deref()
                    .map(|a| a.attn_seconds())
                    .unwrap_or((0.0, 0.0));
                // The array's own sub-items: the QK^T half and the softmax.V
                // half. The block's own time is more than their sum -- the
                // panels are built on the host, once per head, and they are
                // in neither half.
                timings.qk += after.0 - before.0;
                timings.av += after.1 - before.1;
                timings.attention += t_attn.elapsed().as_secs_f64();
            } else {
                let t_qk = std::time::Instant::now();
                self.qk(qkv_buf, scores_buf, batch, seq, sc_cols);
                let qk_s = t_qk.elapsed().as_secs_f64();
                timings.qk += qk_s;
                let t_sm = std::time::Instant::now();
                // THE SOFTMAX, ON THE ARRAY OR ON THE HOST.
                //
                // The mask is added to the row before the softmax, because
                // the softmax's maximum has to see it -- and the mask covers
                // the REAL keys only, so the padding columns keep the -1e30
                // the score matrix was laid out with.
                if let Some(eltwise) = eltwise.as_deref_mut().filter(|_| sm_on) {
                    for r in 0..score_rows {
                        let row = &mut scores_buf[r * sc_cols..][..sc_cols];
                        let mk = &add_mask_buf[(r / (spec.heads * seq)) * seq..][..seq];
                        kernels::add_inplace(row, mk);
                    }
                    eltwise.softmax(scores_buf, score_rows)?;
                    // THE ARRAY'S TIME, IN THE ARRAY'S ROW.
                    timings.array_gemm += t_sm.elapsed().as_secs_f64();
                } else {
                    // THE MASK, THEN THE SOFTMAX.
                    //
                    // The mask goes in first, because the softmax's maximum
                    // has to see it -- and the mask covers the REAL keys
                    // only, so the padding columns keep the -1e30 the score
                    // matrix was laid out with.
                    for r in 0..score_rows {
                        let row = &mut scores_buf[r * seq..(r + 1) * seq];
                        let mk = &add_mask_buf[(r / (spec.heads * seq)) * seq..][..seq];
                        kernels::add_inplace(row, mk);
                    }
                    // ONE CALL FOR THE WHOLE MATRIX.
                    //
                    // The rows are the rows: the host's softmax normalises a
                    // row of `seq` keys, and the GPU's normalises a row of
                    // the same width. The matrix is laid out at the
                    // SEQUENCE's width here -- the array's padded width is
                    // the array's path, and the host's path never sees it.
                    ops.softmax_rows(scores_buf, score_rows, seq)?;
                    let spent = t_sm.elapsed().as_secs_f64();
                    match ops.place() {
                        Placement::Host => timings.host_softmax += spent,
                        _ => timings.gpu += spent,
                    }
                }
                let t_av = std::time::Instant::now();
                self.av(scores_buf, qkv_buf, ctx_buf, batch, seq, sc_cols);
                let av_s = t_av.elapsed().as_secs_f64();
                timings.av += av_s;
                // THE ATTENTION BLOCK IS QK^T AND softmax.V, and not the
                // softmax: the reference's `t_attn` is those two, and the softmax
                // is counted in its own row. Counting it here as well counts
                // it twice.
                timings.attention += qk_s + av_s;
            }

            // The attention output projection, and the residual.
            match array.as_deref_mut().filter(|_| gemm_on) {
                Some(array) => {
                    let t = std::time::Instant::now();
                    array.gemm(
                        &self.gemm_names_attn_out[index],
                        ctx_buf,
                        rows,
                        hidden,
                        hidden,
                        Some(&layer.attn_out_bias),
                        proj_buf,
                    )?;
                    timings.array_gemm += t.elapsed().as_secs_f64();
                }
                None => {
                    let t = std::time::Instant::now();
                    ops.gemm_nt(
                        ctx_buf,
                        rows,
                        &layer.attn_out,
                        hidden,
                        hidden,
                        &layer.attn_out_bias,
                        proj_buf,
                    )?;
                    if ops.place() != Placement::Host {
                        timings.gpu += t.elapsed().as_secs_f64();
                    }
                }
            }
            kernels::add_inplace(x_buf, proj_buf);

            // The first LayerNorm, then the FFN.
            let t_ln = std::time::Instant::now();
            if let Some(eltwise) = eltwise.as_deref_mut().filter(|_| ln_on) {
                eltwise.layernorm(&layer.ln1_params, x_buf, rows)?;
                timings.array_gemm += t_ln.elapsed().as_secs_f64();
            } else {
                ops.layernorm_rows(x_buf, rows, hidden, &layer.ln1_gamma, &layer.ln1_beta)?;
                let spent = t_ln.elapsed().as_secs_f64();
                match ops.place() {
                    Placement::Host => timings.host_layernorm += spent,
                    _ => timings.gpu += spent,
                }
            }
            match array.as_deref_mut().filter(|_| gemm_on) {
                Some(array) => {
                    let t = std::time::Instant::now();
                    array.gemm(
                        &self.gemm_names_ffn_up[index],
                        x_buf,
                        rows,
                        hidden,
                        spec.ffn,
                        Some(&layer.ffn_up_bias),
                        up_buf,
                    )?;
                    timings.array_gemm += t.elapsed().as_secs_f64();
                }
                None => {
                    let t = std::time::Instant::now();
                    ops.gemm_nt(
                        x_buf,
                        rows,
                        &layer.ffn_up,
                        hidden,
                        spec.ffn,
                        &layer.ffn_up_bias,
                        up_buf,
                    )?;
                    if ops.place() != Placement::Host {
                        timings.gpu += t.elapsed().as_secs_f64();
                    }
                }
            }
            let t_gelu = std::time::Instant::now();
            if let Some(eltwise) = eltwise.as_deref_mut().filter(|_| gelu_on) {
                eltwise.gelu(up_buf)?;
                timings.array_gemm += t_gelu.elapsed().as_secs_f64();
            } else {
                ops.gelu(up_buf)?;
                let spent = t_gelu.elapsed().as_secs_f64();
                match ops.place() {
                    Placement::Host => timings.host_gelu += spent,
                    _ => timings.gpu += spent,
                }
            }
            match array.as_deref_mut().filter(|_| gemm_on) {
                Some(array) => {
                    let t = std::time::Instant::now();
                    array.gemm(
                        &self.gemm_names_ffn_down[index],
                        up_buf,
                        rows,
                        spec.ffn,
                        hidden,
                        Some(&layer.ffn_down_bias),
                        down_buf,
                    )?;
                    timings.array_gemm += t.elapsed().as_secs_f64();
                }
                None => {
                    let t = std::time::Instant::now();
                    ops.gemm_nt(
                        up_buf,
                        rows,
                        &layer.ffn_down,
                        spec.ffn,
                        hidden,
                        &layer.ffn_down_bias,
                        down_buf,
                    )?;
                    if ops.place() != Placement::Host {
                        timings.gpu += t.elapsed().as_secs_f64();
                    }
                }
            }
            kernels::add_inplace(x_buf, down_buf);

            // The second LayerNorm.
            let t_ln = std::time::Instant::now();
            if let Some(eltwise) = eltwise.as_deref_mut().filter(|_| ln_on) {
                eltwise.layernorm(&layer.ln2_params, x_buf, rows)?;
                timings.array_gemm += t_ln.elapsed().as_secs_f64();
            } else {
                ops.layernorm_rows(x_buf, rows, hidden, &layer.ln2_gamma, &layer.ln2_beta)?;
                let spent = t_ln.elapsed().as_secs_f64();
                match ops.place() {
                    Placement::Host => timings.host_layernorm += spent,
                    _ => timings.gpu += spent,
                }
            }
        }
        Ok(())
    }

    /// `QK^T`, as the reference's `qk_impl` computes it.
    ///
    /// For every (sequence, head, query position): the query against every
    /// key of the same sequence and head. The dot product is eight lanes
    /// wide: the FIRST eight products are a plain multiplication and the rest
    /// are FMAs onto it, and the eight lanes are then folded by `hsum8`.
    ///
    /// THE ROW'S STRIDE IS `sc_cols`, not the sequence length: the row is
    /// as wide as the softmax that will read it, and the columns past the
    /// sequence are not keys -- they are the padding, and they are filled
    /// with -1e30 so that the softmax's exponentials underflow them to
    /// zero. A zero there would be a key attending to everything.
    fn qk(&self, qkv: &[f32], scores: &mut [f32], batch: usize, seq: usize, sc_cols: usize) {
        let spec = &self.spec;
        let hidden = spec.hidden;
        let head_dim = spec.head_dim;
        let nv = head_dim / 8;
        let pairs = batch * spec.heads;
        for p in 0..pairs {
            let b = p / spec.heads;
            let h = p % spec.heads;
            for i in 0..seq {
                let q = &qkv[(b * seq + i) * 3 * hidden + h * head_dim..][..head_dim];
                let row = (p * seq + i) * sc_cols;
                let dst = &mut scores[row..][..seq];
                for (j, slot) in dst.iter_mut().enumerate() {
                    let k = &qkv[(b * seq + j) * 3 * hidden + hidden + h * head_dim..][..head_dim];
                    let mut lanes = [0.0f32; 8];
                    for l in 0..8 {
                        lanes[l] = q[l] * k[l];
                    }
                    for v in 1..nv {
                        for l in 0..8 {
                            lanes[l] = q[v * 8 + l].mul_add(k[v * 8 + l], lanes[l]);
                        }
                    }
                    *slot = kernels::hsum8(lanes);
                }
                // The padding column, only when the array's softmax is what
                // will read this row and it was compiled wider than the
                // sequence.
                scores[row + seq..][..sc_cols - seq].fill(-1.0e30);
            }
        }
    }

    /// `softmax.V`, as the reference's `av_impl` computes it.
    ///
    /// Eight lane accumulators, one per eight of the head's width, and an
    /// FMA per key.
    ///
    /// THE ROW'S STRIDE IS `sc_cols` and the row's KEYS are the first
    /// `seq` of it: the padded tail is not a key and is not read.
    fn av(
        &self,
        scores: &[f32],
        qkv: &[f32],
        ctx: &mut [f32],
        batch: usize,
        seq: usize,
        sc_cols: usize,
    ) {
        let spec = &self.spec;
        let hidden = spec.hidden;
        let head_dim = spec.head_dim;
        let nv = head_dim / 8;
        let pairs = batch * spec.heads;
        for p in 0..pairs {
            let b = p / spec.heads;
            let h = p % spec.heads;
            for i in 0..seq {
                let a = &scores[(p * seq + i) * sc_cols..][..seq];
                let o = &mut ctx[(b * seq + i) * hidden + h * head_dim..][..head_dim];
                o.fill(0.0);
                for (j, &aj) in a.iter().enumerate() {
                    let v =
                        &qkv[(b * seq + j) * 3 * hidden + 2 * hidden + h * head_dim..][..head_dim];
                    for lane in 0..nv {
                        for l in 0..8 {
                            let idx = lane * 8 + l;
                            o[idx] = aj.mul_add(v[idx], o[idx]);
                        }
                    }
                }
            }
        }
    }

    /// The pooling, as the reference's `pool_rows` computes it.
    ///
    /// The accumulator is a DOUBLE, and so is the norm. The mean pooling
    /// divides by the number of unmasked positions, and the `[CLS]` pooling
    /// takes the first row -- and refuses a masked one, because a masked
    /// `[CLS]` means an empty sequence and zeros would be a silently
    /// plausible answer.
    fn pool_into(
        &self,
        h: &[f32],
        mask: &[f32],
        batch: usize,
        seq: usize,
        acc: &mut [f64],
        out: &mut [f32],
    ) {
        let hidden = self.spec.hidden;
        for b in 0..batch {
            let amb = &mask[b * seq..(b + 1) * seq];
            let hb = &h[b * seq * hidden..(b + 1) * seq * hidden];
            acc.fill(0.0);
            if self.spec.cls_pooling {
                if amb[0] == 0.0 {
                    // The reference throws here. A refusal is the Rust's shape for
                    // the same condition: an empty sequence has no pooled
                    // vector, and zero is a plausible answer to a question
                    // nobody asked.
                    out[b * hidden..(b + 1) * hidden].fill(0.0);
                    continue;
                }
                for c in 0..hidden {
                    acc[c] = hb[c] as f64;
                }
            } else {
                let mut denom = 0.0f32;
                for s in 0..seq {
                    denom += amb[s];
                }
                let denom = denom.max(1e-9);
                for s in 0..seq {
                    let m = amb[s];
                    if m == 0.0 {
                        continue;
                    }
                    let hr = &hb[s * hidden..(s + 1) * hidden];
                    for c in 0..hidden {
                        acc[c] += (hr[c] * m) as f64;
                    }
                }
                for c in 0..hidden {
                    acc[c] /= denom as f64;
                }
            }
            let o = &mut out[b * hidden..(b + 1) * hidden];
            if self.spec.l2_normalize {
                let mut nrm = 0.0f64;
                for c in 0..hidden {
                    nrm += acc[c] * acc[c];
                }
                let nrm = nrm.max(1e-24).sqrt();
                for c in 0..hidden {
                    o[c] = (acc[c] / nrm) as f32;
                }
            } else {
                for c in 0..hidden {
                    o[c] = acc[c] as f32;
                }
            }
        }
    }
}

/// Layer normalisation over every row of a buffer.
///
/// A LayerNorm's parameters as the array's parameter buffer wants them:
/// the gain first, then the offset, both f32.
///
/// The order is the design's, and it is not the container's -- the
/// container stores the two apart, and the design reads them as one
/// buffer of `2 * cols` values.
fn layernorm_params(gamma: &[f32], beta: &[f32]) -> Vec<f32> {
    let mut params = Vec::with_capacity(gamma.len() + beta.len());
    params.extend_from_slice(gamma);
    params.extend_from_slice(beta);
    params
}
