//! The CPU backend: the accuracy reference.
//!
//! # Why this backend exists at all
//!
//! Every other backend is measured against this one. The GPU's f32 is not the
//! CPU's f32 -- it rounds differently -- and the array's bf16 is a different
//! precision altogether, so "the same numbers" is not a question that has one
//! answer. It has an answer per PAIR of backends, and the tolerances matrix
//! in `docs/en/BACKENDS.md` is that table.
//!
//! This backend is the one whose answer every other row is written relative
//! to. If it moves, the whole matrix moves with it.
//!
//! # The kernels
//!
//! `kernels.rs` holds the arithmetic, and it is a PORT of the C++ runtime's
//! host kernels rather than a fresh implementation of the mathematics. The
//! reasons are in `kernels.rs` itself; the short version is that the golden
//! transcripts are a contract, and a contract is a number, not a shape.
//!
//! The rules the port follows:
//!
//! * **The FMA is explicit.** The C++'s AVX2 paths contract the multiply
//!   and the add into ONE rounding, and `f32::mul_add` is the Rust spelling
//!   of exactly that. A contraction the compiler does on its own is not
//!   relied upon, because a compiler is free to stop doing it.
//!
//! * **The summation orders are the C++'s.** Floating-point addition is not
//!   associative, so "sum the row" is not a function until the order is
//!   named. The orders are named in `kernels.rs`.
//!
//! * **bf16 rounding is explicit.** Round-to-nearest-even, implemented here,
//!   not taken from `std::bf16` until a gate has confirmed that the standard
//!   library's semantics are the hardware's. The ties rule matters:
//!   round-half-to-even and round-half-away differ on every exact .5.

pub mod bert;
pub mod hands;
pub mod hands_decode;
pub mod kernels;
pub mod pose;
pub mod pose_decode;
pub mod vit;
pub mod vit_image;
pub mod whisper;
pub mod whisper_features;
pub mod whisper_session;
pub mod whisper_tokenizer;

use npum_model::{Backend, Caps, ModelSpec, OpKind, Plan, Result, Tensor};
use npum_model::caps::Precision as CapsPrecision;

pub use bert::BertModel;

pub use kernels::{gelu_f32, layernorm_row, softmax_row};

/// The CPU backend.
pub struct CpuBackend {
    plan: Plan,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackend {
    /// A CPU backend that runs everything.
    pub fn new() -> Self {
        Self {
            plan: Plan {
                dispatched: Vec::new(),
                host_side: OpKind::ALL.iter().map(|o| o.as_str().to_string()).collect(),
                note: "the CPU backend runs every operation on the host".into(),
            },
        }
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn caps(&self) -> Caps {
        Caps {
            ops: OpKind::ALL.to_vec(),
            precision: CapsPrecision::F32,
            generation: None,
            bit_exact_with_cpu: true,
        }
    }

    fn prepare(&mut self, spec: &ModelSpec) -> Result<Plan> {
        self.plan.note = format!("the CPU backend runs {} layers of {}", spec.layers, spec.arch_name);
        Ok(self.plan.clone())
    }

    fn gemm(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        c: &mut Tensor,
        acc: Option<&mut Tensor>,
    ) -> Result<()> {
        kernels::gemm(a, b, c, acc)?;
        Ok(())
    }

    fn gelu(&mut self, x: &mut Tensor) -> Result<()> {
        kernels::gelu_(x)?;
        Ok(())
    }

    fn layernorm(&mut self, x: &mut Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Result<()> {
        kernels::layernorm_(x, gamma, beta, eps)?;
        Ok(())
    }

    fn softmax(&mut self, x: &mut Tensor) -> Result<()> {
        kernels::softmax_(x)?;
        Ok(())
    }

    fn attention(&mut self, q: &Tensor, k: &Tensor, v: &Tensor, out: &mut Tensor) -> Result<()> {
        kernels::attention(q, k, v, out)?;
        Ok(())
    }

    fn conv1d(
        &mut self,
        x: &mut Tensor,
        w: &Tensor,
        b: Option<&Tensor>,
        stride: i64,
        pad: i64,
    ) -> Result<()> {
        kernels::conv1d(x, w, b, stride, pad)?;
        Ok(())
    }

    fn conv2d(
        &mut self,
        x: &mut Tensor,
        w: &Tensor,
        b: Option<&Tensor>,
        stride: (i64, i64),
        pad: (i64, i64),
    ) -> Result<()> {
        kernels::conv2d(x, w, b, stride, pad)?;
        Ok(())
    }

    fn mproj(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        kernels::mproj(x, out)?;
        Ok(())
    }

    fn fft(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        kernels::fft(x, out)?;
        Ok(())
    }

    fn logit(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        kernels::logit(x, out)?;
        Ok(())
    }
}
