//! The seam between a family's host path and the array.
//!
//! # WHY A TRAIT AND NOT A FLAG
//!
//! A family's path is a straight line of host code -- an im2col, a GEMM, a
//! transpose -- and the array's path is the same line with one call
//! different. The difference has to be made ONCE, in one place, and it has
//! to be made by the CALLER: the family knows which convolution it is
//! running and what its own shape is, and only the family can say that.
//!
//! So the family calls a trait, and the trait has two implementations that
//! matter: the array's, and the host's. The host's implementation is not a
//! fallback bolted on afterwards -- it is the path every run took before the
//! array existed, and it stays bit for bit the same, because a port that
//! moves the numbers is not a port.
//!
//! # WHY IT IS IN THIS CRATE
//!
//! The same reason the `Backend` trait is: the family calls it and the
//! backend implements it, and a trait that only one side can name is a trait
//! the other side cannot use. `npum-cpu` and `npum-xdna` both depend on
//! this crate and neither on the other, which is what keeps the CPU
//! reference comparable against anything.
//!
//! # WHAT THE ARRAY'S SIDE IS TOLD
//!
//! The family's own shape -- the rows, the reduction width, the output
//! width -- and the convolution's identity. The identity is the WEIGHT's
//! name in the container ("det.conv.0.w"), because that is the name the
//! family's graph already carries, and because the panel the array
//! multiplies by is the weight's own: the same numbers, rounded to bf16,
//! padded to the design's widths and tiled into the MMAC's byte order.
//!
//! The array's side knows the widths IT wants -- the design's padded ones --
//! and it does the padding. The family must not: a family that padded its
//! own rows would have to know the design's shape, and the design's shape
//! is the design's business.

use std::fmt;

/// What went wrong while a family dispatched work to the array.
///
/// The message is the whole of it: the array's failures are reported to
/// the person who asked for the array, and a refusal that names the
/// convolution and the shape is worth ten that name the module.
#[derive(Debug, Clone)]
pub struct ArrayError(pub String);

impl fmt::Display for ArrayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ArrayError {}

/// The result of an array operation.
pub type Result<T> = std::result::Result<T, ArrayError>;

/// One convolution the array will run.
///
/// This is the family's description of a convolution, and it is here
/// because both sides need it: the family knows which convolutions it runs
/// and what their own shapes are, and the array's side knows what it can
/// dispatch. The shape in here is the CONVOLUTION's own -- the reduction
/// width and the output width -- and the design's padded widths are the
/// array's business.
#[derive(Debug, Clone)]
pub struct ConvSpec {
    /// The weight's name in the container -- "det.conv.0.w".
    ///
    /// It is the identity: the panel is the weight's own, and two
    /// convolutions of the same shape are two different weights.
    pub weight: String,

    /// The design's stream for it -- "conv64x128" -- as the packer recorded
    /// it in the graph's op list.
    ///
    /// It is the SHAPE the convolution is dispatched against, and two
    /// convolutions of the same shape share it -- which is exactly why it
    /// cannot be the panel's key.
    pub stream: String,

    /// The convolution's own reduction width: `cin * kh * kw`.
    pub k: usize,

    /// The convolution's own output width: its output channels.
    pub n: usize,

    /// The convolution's own bias, if it has one.
    pub bias: Option<Vec<f32>>,
}

/// A borrowed handle on the array's convolutions.
///
/// THE TWO LIFETIMES ARE SEPARATE ON PURPOSE. A `&mut` is invariant in the
/// lifetime of the thing it points at, so a handle that tied the trait
/// object's lifetime to the borrow's would refuse every REBORROW -- and a
/// reborrow is exactly what a walk over a graph does, once per op, with the
/// same handle.
pub type ArrayRef<'a, 'b> = Option<&'a mut (dyn ArrayConvs + 'b)>;

/// The array's side of a family's dense convolution.
///
/// A family's dense convolution is an im2col on the host and a matrix
/// multiplication on the array, and this trait is the second half. The
/// depthwise convolutions, the transposed ones and the projections stay on
/// the host: a depthwise filter reduces WITHIN one channel and has no
/// `[M, N]` product in it at all, and a transposed convolution's output is
/// a sum of shifted contributions, which is not a matrix product either.
pub trait ArrayConvs {
    /// The backend's name, as it appears in the status block.
    fn name(&self) -> &'static str;

    /// How many convolutions this backend has dispatched.
    ///
    /// The count is the RUN'S, and it is the number that says whether the
    /// flag did what it said. A run that asked for the array and dispatched
    /// nothing is a run that lied, and this is where the lie shows.
    fn dispatches(&self) -> usize;

    /// How many bytes of weight panels are staged on the board.
    ///
    /// It is the load's number and not the run's: the panels are staged
    /// once, at load, and every run of the model reuses them.
    fn staged_bytes(&self) -> usize;

    /// Run one dense convolution on the array.
    ///
    /// `weight` is the convolution's weight name in the container --
    /// "det.conv.0.w" -- and the array's side finds the panel that belongs
    /// to it. `a` holds the host's im2col rows, `rows` of `k` values;
    /// `out` receives `rows` of `n` values.
    ///
    /// The bias is NOT an argument: the panel carries the convolution's own
    /// bias, padded with zeros to the design's width, and adding it here
    /// would add it twice.
    fn conv(
        &mut self,
        weight: &str,
        a: &[f32],
        rows: usize,
        k: usize,
        n: usize,
        out: &mut [f32],
    ) -> Result<()>;
}
