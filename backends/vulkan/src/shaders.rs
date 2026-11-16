//! The shaders: WGSL compiled to SPIR-V at build time and embedded.
//!
//! # Where the source is
//!
//! `backends/vulkan/shaders/*.wgsl`. `build.rs` compiles each one with naga
//! and writes the words into `OUT_DIR/shaders_spv.rs`, which is included
//! below: a shader that does not parse or does not validate is a build
//! failure, not a run-time one.
//!
//! # Why the constants are `&[u32]`
//!
//! `vkCreateShaderModule` reads the module as 32-bit words, and a byte slice
//! from `include_bytes!` is not guaranteed to be four-byte aligned. A
//! generated source of `u32` literals has no alignment question at all.
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

include!(concat!(env!("OUT_DIR"), "/shaders_spv.rs"));

/// Every shader, by name.
pub const ALL: [(&str, &[u32]); 7] = [
    ("gemm", GEMM),
    ("gelu", GELU),
    ("layernorm", LAYERNORM),
    ("softmax", SOFTMAX),
    ("attention", ATTENTION),
    ("add", ADD),
    ("mul", MUL),
];
