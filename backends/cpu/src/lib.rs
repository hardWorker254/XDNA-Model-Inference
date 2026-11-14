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
//! `kernels.rs` holds the arithmetic. The rules it follows:
//!
//! * **No FMA contraction.** Rust does not contract `a * b + c` into an FMA
//!   by default, and the C++ host kernels did not either -- and the gate
//!   `runtime/tests/test_int8_host_kernels.cpp` proved that the scalar
//!   path and the AVX2 path agree bit for bit under exactly that rule. A
//!   contraction would change the last bit of every dot product and every
//!   tolerance measured against this backend would be measuring the wrong
//!   thing.
//!
//! * **AVX2 and scalar must agree bit for bit.** They are two spellings of
//!   the same arithmetic, and a difference between them is a bug in one of
//!   them, not a rounding difference. The AVX2 path is an optimisation; the
//!   scalar path is the definition.
//!
//! * **bf16 rounding is explicit.** Round-to-nearest-even, implemented here,
//!   not taken from `std::bf16` until a gate has confirmed that the standard
//!   library's semantics are the hardware's. The ties rule matters:
//!   round-half-to-even and round-half-away differ on every exact .5.

pub mod kernels;

use npum_model::{Backend, Caps, ModelSpec, OpKind, Plan, Result, Tensor};
use npum_model::caps::Precision as CapsPrecision;

pub use kernels::{gelu_f32, layernorm_f32, softmax_f32};

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
