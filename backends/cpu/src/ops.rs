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
//!
//! # Parallelism is a property of this object too
//!
//! Splitting rows across the shared pool changes which worker computes a
//! value, but not the value: each row's reduction is over K in the same order
//! on every worker. The gate therefore still compares bytes. `on(isa)` is
//! deliberately serial -- it is what the byte-comparison gate uses --
//! while `best()` uses the pool above the shared work threshold.

use crate::kernels::{self, Isa};
use crate::parallel;
use npum_model::dispatcher::Placement;
use npum_model::{Ops, Result, TensorError};

/// The host's arithmetic, on a named ISA, with a chosen worker count.
pub struct CpuOps {
    isa: Isa,
    /// `1` means the strictly serial reference. Anything above it splits
    /// independent rows over the shared pool, subject to the work threshold.
    threads: usize,
}

impl CpuOps {
    /// The host's own choice: the widest GEMM this machine has, row-parallel
    /// when there is enough work to make the split worthwhile.
    pub fn best() -> Self {
        Self::with_threads(kernels::gemm_isa(), parallel::threads())
    }

    /// A named ISA, strictly serial.
    ///
    /// For the gate that compares the ISAs byte for byte. On a host that
    /// does not have the ISA asked for, the GEMM falls back to the portable
    /// path -- which computes the same bytes, and is a great deal slower.
    pub fn on(isa: Isa) -> Self {
        Self { isa, threads: 1 }
    }

    /// A named ISA with a requested number of workers.
    ///
    /// A worker count above one is honoured only when the host can actually
    /// run the named ISA; otherwise the object stays serial and the named
    /// kernel's own portable fallback applies.
    pub fn with_threads(isa: Isa, threads: usize) -> Self {
        let usable = kernels::available_isas().contains(&isa);
        Self {
            isa,
            threads: if usable { threads.max(1) } else { 1 },
        }
    }

    /// The byte-exact reference, never parallel, on the host's best ISA.
    pub fn serial() -> Self {
        Self::on(kernels::gemm_isa())
    }

    /// The worker count this object was built with.
    pub fn threads(&self) -> usize {
        self.threads
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
        if !kernels::gemm_nt_rows_on(self.isa, a, m, b, k, n, bias, out, self.threads) {
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
            return Err(TensorError::WrongLength {
                expected: span,
                found: values.len(),
            }
            .into());
        }
        // The row kernels now handle a short tail, so an attention sequence
        // whose length is not a multiple of eight no longer has to be padded
        // (quadratic work in a short batch) or rejected at this seam.
        let work = (rows as i64) * (width as i64);
        match parallel::chunk_rows(rows, work, self.threads) {
            Some(chunk) => {
                parallel::for_each_chunk_mut(&mut values[..span], chunk * width, |_, part| {
                    for row in part.chunks_mut(width) {
                        let _ = kernels::layernorm_row_on(self.isa, row, gamma, beta);
                    }
                });
            }
            None => {
                for row in values[..span].chunks_mut(width) {
                    kernels::layernorm_row_on(self.isa, row, gamma, beta)?;
                }
            }
        }
        Ok(())
    }

    fn gelu(&mut self, x: &mut [f32]) -> Result<()> {
        // GELU is elementwise, so any split is bit-exact: each worker runs the
        // same polynomial on its own slice. `gelu_row_on` now has a genuinely
        // eight-wide AVX2 body, so combining the two gives width AND cores.
        let work = x.len() as i64;
        let Some(chunk) = parallel::chunk_rows(x.len(), work, self.threads) else {
            kernels::gelu_row_on(self.isa, x);
            return Ok(());
        };
        parallel::for_each_chunk_mut(x, chunk, |_, part| {
            kernels::gelu_row_on(self.isa, part);
        });
        Ok(())
    }

    fn softmax_rows(&mut self, values: &mut [f32], rows: usize, width: usize) -> Result<()> {
        let span = rows * width;
        if values.len() < span {
            return Err(TensorError::WrongLength {
                expected: span,
                found: values.len(),
            }
            .into());
        }
        let work = (rows as i64) * (width as i64);
        match parallel::chunk_rows(rows, work, self.threads) {
            Some(chunk) => {
                parallel::for_each_chunk_mut(&mut values[..span], chunk * width, |_, part| {
                    for row in part.chunks_mut(width) {
                        let _ = kernels::softmax_row_on(self.isa, row);
                    }
                });
            }
            None => {
                for row in values[..span].chunks_mut(width) {
                    kernels::softmax_row_on(self.isa, row)?;
                }
            }
        }
        Ok(())
    }

    fn add_inplace(&mut self, values: &mut [f32], other: &[f32]) -> Result<()> {
        // Each element depends only on the two values at the same index, so a
        // chunk boundary crosses no reduction: every worker computes exactly
        // the sum the serial loop would. GELU already uses this split; the
        // residual add and the additive attention mask are just as safe to
        // split, and they run on every layer of every inference.
        let work = values.len().min(other.len()) as i64;
        let Some(chunk) = parallel::chunk_rows(values.len(), work, self.threads) else {
            kernels::add_inplace(values, other);
            return Ok(());
        };
        parallel::for_each_chunk_mut(values, chunk, |ci, part| {
            let start = ci * chunk;
            let end = start + part.len();
            // `add_inplace` clamps to the shorter slice; a short `other` must
            // leave the tail alone, chunked or not.
            let src = &other[start.min(other.len())..end.min(other.len())];
            kernels::add_inplace(part, src);
        });
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
