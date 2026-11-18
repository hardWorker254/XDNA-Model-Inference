//! The host's end of the arithmetic seam.
//!
//! # This is the reference
//!
//! Every other implementation of `npum_model::Ops` is measured against this
//! one, and this one is the port of the reference runtime's host kernels. So the
//! implementation here is not "a way to compute a GEMM" -- it is THE way,
//! down to the FMA in the reduction and the order the row kernels fold
//! their lanes in.
//!
//! # The ISA is a property of this object, not of the caller
//!
//! The GEMM has three paths on this host -- the portable one, AVX2 and
//! AVX-512 -- and they agree BYTE FOR BYTE (that is what the plan's 5.1a
//! asks for, and `tests/isa.rs` is the gate). Which one runs is therefore
//! not a question with a different answer: it is a question about speed.
//!
//! The ISA lives here so that the gate can still ASK for a named one: the
//! same object, with a different ISA inside, and the bytes compared.

use crate::kernels::{self, Isa};
use npum_model::dispatcher::Placement;
use npum_model::{Ops, Result};

/// The host's arithmetic, on a named ISA.
pub struct CpuOps {
    isa: Isa,
}

impl CpuOps {
    /// The host's own choice: the widest GEMM this machine has.
    pub fn best() -> Self {
        Self {
            isa: kernels::gemm_isa(),
        }
    }

    /// A named ISA.
    ///
    /// For the gate that compares the ISAs byte for byte. On a host that
    /// does not have the ISA asked for, the GEMM falls back to the portable
    /// path -- which computes the same bytes, and is a great deal slower.
    pub fn on(isa: Isa) -> Self {
        Self { isa }
    }
}

impl Default for CpuOps {
    fn default() -> Self {
        Self::best()
    }
}

impl Ops for CpuOps {
    fn gemm_nt(
        &mut self,
        a: &[f32],
        m: usize,
        b: &[f32],
        k: usize,
        n: usize,
        bias: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        if !kernels::gemm_nt_on(self.isa, a, m, b, k, n, bias, out) {
            // The host does not have the ISA that was asked for. The
            // portable path computes the same bytes -- the gate says so --
            // and the alternative, leaving the output alone, is a run that
            // answers with zeros and calls it a result.
            kernels::gemm_nt_on(Isa::Portable, a, m, b, k, n, bias, out);
        }
        Ok(())
    }

    fn layernorm_rows(
        &mut self,
        values: &mut [f32],
        rows: usize,
        width: usize,
        gamma: &[f32],
        beta: &[f32],
    ) -> Result<()> {
        let span = rows * width;
        if values.len() < span {
            return Err(npum_model::TensorError::WrongLength {
                expected: span,
                found: values.len(),
            }
            .into());
        }
        for row in values[..span].chunks_mut(width) {
            kernels::layernorm_row_on(self.isa, row, gamma, beta)?;
        }
        Ok(())
    }

    fn gelu(&mut self, x: &mut [f32]) -> Result<()> {
        kernels::gelu_row_on(self.isa, x);
        Ok(())
    }

    fn softmax_rows(&mut self, values: &mut [f32], rows: usize, width: usize) -> Result<()> {
        let span = rows * width;
        if values.len() < span {
            return Err(npum_model::TensorError::WrongLength {
                expected: span,
                found: values.len(),
            }
            .into());
        }
        for row in values[..span].chunks_mut(width) {
            kernels::softmax_row_on(self.isa, row)?;
        }
        Ok(())
    }

    fn place(&self) -> Placement {
        Placement::Host
    }

    fn device(&self) -> Option<&str> {
        None
    }

    fn dispatches(&self) -> usize {
        0
    }
}
