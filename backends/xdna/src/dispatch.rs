//! Dispatching to the array.
//!
//! # Why every function here says "not yet"
//!
//! The reference runtime's dispatch code was 4 000 lines of XRT calls and buffer
//! arithmetic, and it is the part of the port that carries the most risk and
//! the least insight. It is written LAST, against a gate that compares the
//! array's numbers with the CPU reference's, and not before.
//!
//! What is here now is the SHAPE: every operation the trait declares has a
//! dispatch function, each one says which transport it needs, and each one
//! refuses with a message that says what is missing rather than panicking.
//!
//! # WHERE THE REAL EXECUTION LIVES
//!
//! None of these functions is on a running model's path. Do not look here for
//! the code that dispatches a BERT projection or a detector convolution:
//!
//! * projections (and the attention's two GEMMs) run through
//!   [`crate::array_gemms::ArrayGemms`], using the trait seam in
//!   `npum_model::array::ArrayGemm` / `ArrayText`;
//! * dense convolutions run through [`crate::array_convs::ArrayConvs`] and
//!   [`crate::conv`];
//! * the elementwise designs live in [`crate::eltwise`].
//!
//! All of them dispatch through [`crate::array::ArrayDesign`], which is the
//! only place that allocates, binds and starts buffers. This module is the
//! OLD layer: it describes the operations the trait knows about, not the
//! operations the array set actually runs. Filling these stubs in would
//! duplicate the array path and give the runtime two answers to the same
//! question; the right change here is to keep them honest refusals.
//!
//! # The transport boundary
//!
//! Everything below this line is transport-agnostic. `xrt.rs` and
//! `direct.rs` both implement the same three calls -- allocate a buffer,
//! upload, dispatch -- and neither of them is visible from the code above.

use npum_model::{Result, Tensor};

use crate::XdnaBackend;

/// The error a not-yet-implemented dispatch returns.
///
/// It is an error and not a panic: a runtime that panics on an operation the
/// array does not have yet is a runtime that takes the whole process down
/// with it, and the reference runtime's contract was to say what was wrong and
/// stop.
fn not_yet(op: npum_model::OpKind) -> npum_model::Error {
    npum_model::Error::NotImplemented {
        op: op.as_str(),
        dtype: "f32".into(),
    }
}

/// A general matrix multiply on the array.
pub fn gemm(
    _backend: &mut XdnaBackend,
    _a: &Tensor,
    _b: &Tensor,
    _c: &mut Tensor,
    _acc: Option<&mut Tensor>,
) -> Result<()> {
    Err(not_yet(npum_model::OpKind::Gemm))
}

/// The GELU activation on the array.
pub fn gelu(_backend: &mut XdnaBackend, _x: &mut Tensor) -> Result<()> {
    Err(not_yet(npum_model::OpKind::Gelu))
}

/// Layer normalisation on the array.
pub fn layernorm(
    _backend: &mut XdnaBackend,
    _x: &mut Tensor,
    _gamma: &Tensor,
    _beta: &Tensor,
    _eps: f32,
) -> Result<()> {
    Err(not_yet(npum_model::OpKind::Layernorm))
}

/// Softmax on the array.
pub fn softmax(_backend: &mut XdnaBackend, _x: &mut Tensor) -> Result<()> {
    Err(not_yet(npum_model::OpKind::Softmax))
}

/// Attention on the array.
pub fn attention(
    _backend: &mut XdnaBackend,
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _out: &mut Tensor,
) -> Result<()> {
    Err(not_yet(npum_model::OpKind::Attn))
}

/// A convolution on the array.
pub fn conv1d(
    _backend: &mut XdnaBackend,
    _x: &mut Tensor,
    _w: &Tensor,
    _b: Option<&Tensor>,
    _stride: i64,
    _pad: i64,
) -> Result<()> {
    Err(not_yet(npum_model::OpKind::Conv))
}

/// A two-dimensional convolution on the array.
pub fn conv2d(
    _backend: &mut XdnaBackend,
    _x: &mut Tensor,
    _w: &Tensor,
    _b: Option<&Tensor>,
    _stride: (i64, i64),
    _pad: (i64, i64),
) -> Result<()> {
    Err(not_yet(npum_model::OpKind::Conv))
}

/// The multimodal projection on the array.
pub fn mproj(_backend: &mut XdnaBackend, _x: &Tensor, _out: &mut Tensor) -> Result<()> {
    Err(not_yet(npum_model::OpKind::Mproj))
}

/// The FFT on the array.
pub fn fft(_backend: &mut XdnaBackend, _x: &Tensor, _out: &mut Tensor) -> Result<()> {
    Err(not_yet(npum_model::OpKind::Fft))
}

/// The logit projection on the array.
pub fn logit(_backend: &mut XdnaBackend, _x: &Tensor, _out: &mut Tensor) -> Result<()> {
    Err(not_yet(npum_model::OpKind::Logit))
}
