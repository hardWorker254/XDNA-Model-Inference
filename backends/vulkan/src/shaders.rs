//! The shaders: WGSL, compiled at build time and embedded.
//!
//! # Why the GELU shader is a polynomial and not an erf
//!
//! The CPU kernel's GELU is a degree-8 minimax polynomial, and the array's
//! fused kernel uses the same one. A shader that computed the exact erf
//! would be a DIFFERENT FUNCTION -- not a different implementation of the
//! same one -- and the difference would show up as a tolerance that has to
//! be explained rather than a tolerance that is understood.
//!
//! # Why the softmax shader carries the clamps
//!
//! The three clamps from `kernels/softmax.cc` -- the load clamp, the
//! difference clamp and the base-2 argument clamp -- are the GPU's problem
//! too. A GPU's f32 has the same narrow-range hazard as the array's bf16
//! whenever the values are narrowed, and the NaN that one masked element
//! produces destroys a whole row.

/// The GELU shader: the same degree-8 polynomial the CPU kernel uses.
pub const GELU: &str = include_str!("../shaders/gelu.wgsl");

/// The layer-normalisation shader.
pub const LAYERNORM: &str = include_str!("../shaders/layernorm.wgsl");

/// The softmax shader.
pub const SOFTMAX: &str = include_str!("../shaders/softmax.wgsl");

/// The GEMM shader.
pub const GEMM: &str = include_str!("../shaders/gemm.wgsl");

/// The attention shader.
pub const ATTENTION: &str = include_str!("../shaders/attention.wgsl");

/// The elementwise addition shader.
pub const ADD: &str = include_str!("../shaders/add.wgsl");

/// The elementwise multiplication shader.
pub const MUL: &str = include_str!("../shaders/mul.wgsl");

/// Every shader, by name.
pub const ALL: [(&str, &str); 7] = [
    ("gemm", GEMM),
    ("gelu", GELU),
    ("layernorm", LAYERNORM),
    ("softmax", SOFTMAX),
    ("attention", ATTENTION),
    ("add", ADD),
    ("mul", MUL),
];
