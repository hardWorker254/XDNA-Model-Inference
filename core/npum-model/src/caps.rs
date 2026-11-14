//! What a backend can do, and which operation is which.
//!
//! # The operation list
//!
//! Nine operations, the same nine the C++ runtime had: `gemm`, `gelu`,
//! `layernorm`, `softmax`, `attn`, `conv`, `mproj`, `fft`, `logit`.
//!
//! The list is deliberately the one the `--npu-ops` flag already accepted, so
//! that flag's meaning does not change when the runtime moves. A user with
//! `--npu-ops gelu,layernorm` in a script keeps the same meaning.

use std::fmt;

/// One operation the runtime can dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// A general matrix multiply.
    Gemm,
    /// The GELU activation.
    Gelu,
    /// Layer normalisation.
    Layernorm,
    /// Softmax.
    Softmax,
    /// Scaled dot-product attention.
    Attn,
    /// A convolution.
    Conv,
    /// The multimodal projection.
    Mproj,
    /// A fast Fourier transform.
    Fft,
    /// The logit projection.
    Logit,
}

impl OpKind {
    /// Every operation, in the order the flag's documentation lists them.
    pub const ALL: [OpKind; 9] = [
        OpKind::Gemm,
        OpKind::Gelu,
        OpKind::Layernorm,
        OpKind::Softmax,
        OpKind::Attn,
        OpKind::Conv,
        OpKind::Mproj,
        OpKind::Fft,
        OpKind::Logit,
    ];

    /// The operation's name, as `--npu-ops` spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            OpKind::Gemm => "gemm",
            OpKind::Gelu => "gelu",
            OpKind::Layernorm => "layernorm",
            OpKind::Softmax => "softmax",
            OpKind::Attn => "attn",
            OpKind::Conv => "conv",
            OpKind::Mproj => "mproj",
            OpKind::Fft => "fft",
            OpKind::Logit => "logit",
        }
    }

    /// Parse an operation's name.
    ///
    /// An unknown name is `None` and not a default. A misspelled
    /// `--npu-ops gelu,layernrom` that silently meant `layernorm` would
    /// dispatch an operation the user did not ask for and leave one they did
    /// ask for on the host -- and the user would see a number, not an error.
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|op| op.as_str() == name)
    }
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What a backend can do.
///
/// # What this is for
///
/// The C++ runtime decided "host or NPU" with a scattering of flags and
/// boolean fields: `--npu-ops`, a `host_*` flag per operation, and a separate
/// `npu_ops.cpp` per family. Three backends make the decision explicit, and
/// this is where the explicit decision is described.
///
/// The tolerance question is the interesting part. Two backends that both
/// claim to compute f32 do NOT compute the same f32: the GPU rounds
/// differently, and the array's bf16 datapath is a different precision
/// altogether. So a backend's capabilities say what precision it works in,
/// and the tolerances matrix in `docs/en/BACKENDS.md` says what a comparison
/// between two of them is allowed to show.
#[derive(Debug, Clone)]
pub struct Caps {
    /// The operations this backend can run.
    pub ops: Vec<OpKind>,
    /// The precision this backend works in.
    pub precision: Precision,
    /// The NPU generation this backend targets, if it is an array backend.
    pub generation: Option<npum_format::layout::Generation>,
    /// Whether this backend can run an operation and still produce the same
    /// bits as the CPU reference.
    pub bit_exact_with_cpu: bool,
}

/// The precision a backend works in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// 32-bit float.
    F32,
    /// Brain float.
    Bf16,
    /// 8-bit integer.
    I8,
}

impl Precision {
    /// The precision's name.
    pub fn as_str(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::Bf16 => "bf16",
            Precision::I8 => "i8",
        }
    }
}

impl Caps {
    /// Whether this backend can run the named operation.
    pub fn can(&self, op: OpKind) -> bool {
        self.ops.contains(&op)
    }
}
