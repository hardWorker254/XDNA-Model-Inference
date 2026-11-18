//! The arithmetic seam: the operations a family's forward pass goes through.
//!
//! # Why this is a trait and not a call into the CPU crate
//!
//! A family's forward pass is a sequence of GEMMs, LayerNorms, GELUs and
//! softmaxes. Until now every one of them was a call into
//! `npum_cpu::kernels`, which made the CPU the only place arithmetic could
//! happen -- and a GPU backend that nothing can call is a GPU backend that
//! nothing uses.
//!
//! So the seam is here, in the crate that owns the model's shape, and the
//! backends implement it: `npum_cpu` implements it with the host's own
//! kernels (the ISA-selected ones, which are the reference), and
//! `npum_vulkan` implements it with the GPU.
//!
//! # When this is the canonical layer
//!
//! Use [`Ops`] for family/reference forward passes that operate on flat host or
//! device-backed buffers with explicit dimensions and need truthful placement/
//! dispatch reporting (`place`, `device`, `dispatches`). Use [`crate::Backend`]
//! for tensor/graph-oriented backend execution and planning.
//!
//! # What the trait does NOT decide
//!
//! * **The arithmetic.** Each implementation is free to compute the same
//!   function its own way, and the GPU's f32 is not the host's f32. The
//!   difference is a tolerance, and the tolerances live in the gates.
//! * **The shapes.** The shapes are the family's, and the family checks
//!   them before it calls.
//! * **Where the model runs.** This trait is about the arithmetic of ONE
//!   forward pass, not about a model's whole life. A model that runs on the
//!   array and on the host uses both: the array through
//!   `crate::array`, the host through this.
//!
//! # The signatures are the host's, on purpose
//!
//! The methods take flat slices and explicit widths, because that is what
//! the host's kernels take and what the family's code already has in hand.
//! A trait that took `Tensor`s would have meant copying every activation
//! into a tensor to call it, and the copy would have cost more than the
//! operation.

use crate::Result;
use crate::dispatcher::Placement;

/// The arithmetic a family's forward pass goes through.
///
/// The four operations are the ones the host kernels provide, and the
/// families call them by these names: a GEMM, a LayerNorm, a GELU and a
/// softmax.
pub trait Ops {
    /// `out[m, n] = a[m, k] * b[k, n] + bias[n]`, with `a` row-major
    /// `[m, k]`, `b` row-major `[k, n]` and `out` row-major `[m, n]`.
    ///
    /// The bias is added AFTER the reduction, in the epilogue -- the host's
    /// order, and the array's too. An implementation that folds the bias
    /// into the reduction would be a different number.
    fn gemm_nt(
        &mut self,
        a: &[f32],
        m: usize,
        b: &[f32],
        k: usize,
        n: usize,
        bias: &[f32],
        out: &mut [f32],
    ) -> Result<()>;

    /// The LayerNorm over `rows` rows of `width` values, in place.
    ///
    /// `gamma` and `beta` are the row's scale and shift, each `width` long.
    fn layernorm_rows(
        &mut self,
        values: &mut [f32],
        rows: usize,
        width: usize,
        gamma: &[f32],
        beta: &[f32],
    ) -> Result<()>;

    /// The GELU over a span of activations, in place.
    ///
    /// The span is flat and has no row structure: the host's GELU is
    /// element-wise over the activations, and the array's "row" is the whole
    /// span.
    fn gelu(&mut self, x: &mut [f32]) -> Result<()>;

    /// The softmax over `rows` rows of `width` values, in place.
    ///
    /// A row is a row: the maximum, the exponentials and the sum are the
    /// row's, and one row's sum never touches another's.
    fn softmax_rows(&mut self, values: &mut [f32], rows: usize, width: usize) -> Result<()>;

    /// Where these operations ran.
    ///
    /// A claim about the run rather than a name of a backend: the family
    /// counts its measurements into the rows that own them, and a row that
    /// says "the host" when the work happened on the GPU is a row that
    /// lies.
    fn place(&self) -> Placement;

    /// The device's name, when the arithmetic runs on one.
    ///
    /// The host has no name to give -- it is the host -- and the GPU has
    /// one that a reader recognises.
    fn device(&self) -> Option<&str> {
        None
    }

    /// How many operations went to the device.
    ///
    /// Zero on the host: the host does not dispatch, it computes. On the
    /// GPU it is the number of dispatches the device made, and it is the
    /// number a reader can check the claim "the GPU ran it" against.
    fn dispatches(&self) -> usize {
        0
    }
}
