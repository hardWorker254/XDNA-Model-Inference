//! The model graph, the operation dispatcher, and the trait a backend
//! implements.
//!
//! # Where the Backend trait lives, and why it is here
//!
//! The trait is in this crate and not in `backends/*`. A graph walks itself
//! and calls a trait, and the trait has to be nameable from both sides: the
//! graph calls it, and the three backends implement it. If the trait lived
//! in `backends/xdna`, then `core/npum-model` would have to depend on
//! `backends/xdna` -- and then a core crate could not be tested without a
//! backend, and the CPU reference could not be compared against anything.
//!
//! Each backend crate re-exports this trait, so `npum_xdna::Backend` and
//! `npum_model::Backend` are the same item and a reader finds it where the
//! plan's tree says it is.
//!
//! # What a backend is told and what it is asked
//!
//! `prepare` is where the backend decides what it can do with a model: it
//! reads the container, checks the geometry against its own capabilities, and
//! returns a `Plan` -- the list of operations it will run and how. The graph
//! then walks the operations and calls the trait's methods.
//!
//! The split exists so that a refusal happens ONCE, at prepare time, with
//! the whole model in hand. A backend that discovered at the middle of a
//! forward pass that it could not run the model would have done half the work
//! first, and the user would see the failure after the wait.

pub mod array;
pub mod caps;
pub mod dispatcher;
pub mod graph;
pub mod tensor;

pub use array::{
    ArrayAttention, ArrayAttentionRef, ArrayConvs, ArrayEltwise, ArrayError, ArrayGemm,
    ArrayGemmRef, ArrayText, ArrayTextRef, AttentionShape, ConvSpec, EltOp,
};
pub use caps::{Caps, OpKind};
pub use dispatcher::{Dispatcher, Placement};
pub use graph::{Graph, ModelSpec, Op};
pub use tensor::{DType, Tensor, TensorError};

use std::path::PathBuf;

/// Everything that can go wrong while running a model.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no backend named {0}")]
    UnknownBackend(String),

    #[error("the container at {path} cannot be read: {source}")]
    Container {
        path: PathBuf,
        #[source]
        source: npum_format::Error,
    },

    #[error("this backend cannot run {model}: {reason}")]
    Unsupported { model: String, reason: String },

    #[error("operation {op} on a tensor of {dtype} is not implemented yet")]
    NotImplemented {
        op: &'static str,
        dtype: String,
    },

    #[error(transparent)]
    Tensor(#[from] TensorError),
}

/// The result type of this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// A place where arithmetic happens.
///
/// # The nine operations
///
/// The same nine the C++ runtime had, with the same names: `gemm`, `gelu`,
/// `layernorm`, `softmax`, `attn`, `conv`, `mproj`, `fft`, `logit`. The
/// count is small on purpose -- every operation is a place where a backend's
/// arithmetic can differ from the host's, and a tenth would be a tenth more
/// places for the tolerances matrix to have a row.
pub trait Backend {
    /// The backend's name, as it appears in `--backend`.
    fn name(&self) -> &'static str;

    /// What this backend can do.
    fn caps(&self) -> Caps;

    /// Look at a container and decide whether this backend can run it.
    ///
    /// Returns the plan: the operations that will be dispatched and the ones
    /// that will not. A backend that refuses returns `Err` with a reason
    /// that names what was wrong, because a refusal that says only "no" is
    /// the same message as every other refusal and teaches the reader
    /// nothing.
    fn prepare(&mut self, spec: &ModelSpec) -> Result<Plan>;

    /// A general matrix multiply, optionally accumulating into `acc`.
    fn gemm(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        c: &mut Tensor,
        acc: Option<&mut Tensor>,
    ) -> Result<()>;

    /// The GELU activation, in place.
    fn gelu(&mut self, x: &mut Tensor) -> Result<()>;

    /// Layer normalisation, in place.
    fn layernorm(&mut self, x: &mut Tensor, gamma: &Tensor, beta: &Tensor, eps: f32)
    -> Result<()>;

    /// Softmax, in place, along the last axis.
    fn softmax(&mut self, x: &mut Tensor) -> Result<()>;

    /// Scaled dot-product attention.
    fn attention(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        out: &mut Tensor,
    ) -> Result<()>;

    /// A one-dimensional convolution.
    fn conv1d(
        &mut self,
        x: &mut Tensor,
        w: &Tensor,
        b: Option<&Tensor>,
        stride: i64,
        pad: i64,
    ) -> Result<()>;

    /// A two-dimensional convolution.
    fn conv2d(
        &mut self,
        x: &mut Tensor,
        w: &Tensor,
        b: Option<&Tensor>,
        stride: (i64, i64),
        pad: (i64, i64),
    ) -> Result<()>;

    /// The multimodal projection, into `out`.
    fn mproj(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()>;

    /// A fast Fourier transform, into `out`.
    fn fft(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()>;

    /// The logit projection, into `out`.
    fn logit(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()>;
}

/// What a backend decided to do with a model.
///
/// The plan is the backend's answer to "which of these operations do I run,
/// and how". It is not an optimisation hint: the dispatcher follows it.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    /// The operations this backend will run, by name.
    pub dispatched: Vec<String>,
    /// The operations this backend will NOT run, by name.
    pub host_side: Vec<String>,
    /// A note about how the plan was reached.
    pub note: String,
}

impl Plan {
    /// Whether the backend runs the named operation.
    pub fn runs(&self, op: &str) -> bool {
        self.dispatched.iter().any(|o| o == op)
    }
}
