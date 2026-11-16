//! The GPU's end of the arithmetic seam.
//!
//! # What this is
//!
//! The family calls four operations -- a GEMM, a LayerNorm, a GELU and a
//! softmax -- and this object answers them on the GPU. It is the same
//! arithmetic the host does, computed by a different machine: the GPU's f32
//! is not the host's f32, and the difference is a tolerance, not a bug.
//!
//! # The bias is added on the host
//!
//! The GEMM shader has no bias: it computes `a * b` and the epilogue adds
//! the bias, in the host's order -- a plain addition AFTER the whole
//! reduction. Folding the bias into the reduction would be a different
//! number, and the array's GEMM adds it in the epilogue too.
//!
//! # The dispatch count
//!
//! Every operation that went to the device is counted, because "the GPU ran
//! it" is a claim a reader can check: the count is in the status block, and
//! a run that dispatched nothing while claiming the GPU would show zero
//! there.

use crate::VulkanBackend;
use npum_model::dispatcher::Placement;
use npum_model::{Ops, Result};

/// The GPU's arithmetic.
///
/// The four operations can be selected individually (`--vk-ops`): an
/// operation not in the selection runs on the host's kernels, so a run
/// can mix GPU and host arithmetic per op. "Nothing selected" and "all
/// selected" are the same run -- the whole supported set on the GPU.
pub struct VulkanOps {
    backend: VulkanBackend,
    dispatches: usize,
    gemm: bool,
    layn: bool,
    gelu: bool,
    softm: bool,
}

impl Default for VulkanOps {
    fn default() -> Self {
        Self::new()
    }
}

impl VulkanOps {
    /// A GPU with every supported operation enabled.
    pub fn new() -> Self {
        Self::with_ops(true, true, true, true)
    }

    /// A GPU with a selected set of operations enabled.
    ///
    /// Operations with `false` run on the host's own kernels, so the
    /// arithmetic stays correct and the dispatch count reflects only the
    /// work that actually went to the device.
    pub fn with_ops(gemm: bool, layn: bool, gelu: bool, softm: bool) -> Self {
        Self {
            backend: VulkanBackend::new(),
            dispatches: 0,
            gemm,
            layn,
            gelu,
            softm,
        }
    }

    /// Whether a GPU was opened.
    pub fn has_gpu(&self) -> bool {
        self.backend.has_gpu()
    }

    /// The device's name, when there is one.
    pub fn device_name(&self) -> Option<&str> {
        self.backend.device_name()
    }

    /// How many operations went to the GPU.
    pub fn dispatches(&self) -> usize {
        self.dispatches
    }

    /// A backend failure that names the operation.
    fn failed(op: &str, detail: String) -> npum_model::Error {
        npum_model::Error::Backend {
            backend: "vulkan",
            detail: format!("{op}: {detail}"),
        }
    }
}

impl Ops for VulkanOps {
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
        if !self.gemm {
            npum_cpu::kernels::gemm_nt(a, m, b, k, n, bias, out);
            return Ok(());
        }
        let Some(gpu) = self.backend.device_mut() else {
            npum_cpu::kernels::gemm_nt(a, m, b, k, n, bias, out);
            return Ok(());
        };
        let mut values = gpu
            .gemm(a, b, m, k, n)
            .map_err(|d| Self::failed("gemm", d))?;
        // The bias, in the epilogue: after the whole reduction, as a plain
        // addition. The host does the same, and the array's GEMM does too.
        //
        // AND IT IS PER ROW: the bias has `n` values and every row of the
        // output gets all of them. A bias added to the first row only is
        // a different network, and it is the kind of difference that looks
        // like a precision problem until it is measured.
        for row in values.chunks_mut(n) {
            for (value, b) in row.iter_mut().zip(bias) {
                *value += *b;
            }
        }
        out.copy_from_slice(&values);
        self.dispatches += 1;
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
        if !self.layn {
            for row in values.chunks_mut(width) {
                npum_cpu::kernels::layernorm_row(row, gamma, beta)?;
            }
            return Ok(());
        }
        let Some(gpu) = self.backend.device_mut() else {
            for row in values.chunks_mut(width) {
                npum_cpu::kernels::layernorm_row(row, gamma, beta)?;
            }
            return Ok(());
        };
        gpu.layernorm(
            values,
            rows,
            width,
            (!gamma.is_empty()).then_some(gamma),
            (!beta.is_empty()).then_some(beta),
        )
        .map_err(|d| Self::failed("layernorm", d))?;
        self.dispatches += 1;
        Ok(())
    }

    fn gelu(&mut self, x: &mut [f32]) -> Result<()> {
        if !self.gelu {
            for v in x.iter_mut() {
                *v = npum_cpu::kernels::gelu_f32(*v);
            }
            return Ok(());
        }
        let Some(gpu) = self.backend.device_mut() else {
            for v in x.iter_mut() {
                *v = npum_cpu::kernels::gelu_f32(*v);
            }
            return Ok(());
        };
        gpu.gelu(x).map_err(|d| Self::failed("gelu", d))?;
        self.dispatches += 1;
        Ok(())
    }

    fn softmax_rows(&mut self, values: &mut [f32], rows: usize, width: usize) -> Result<()> {
        if !self.softm {
            for row in values.chunks_mut(width) {
                npum_cpu::kernels::softmax_row(row)?;
            }
            return Ok(());
        }
        let Some(gpu) = self.backend.device_mut() else {
            for row in values.chunks_mut(width) {
                npum_cpu::kernels::softmax_row(row)?;
            }
            return Ok(());
        };
        gpu.softmax(values, rows, width)
            .map_err(|d| Self::failed("softmax", d))?;
        self.dispatches += 1;
        Ok(())
    }

    fn place(&self) -> Placement {
        // THE PLACE IS WHERE THE ARITHMETIC ACTUALLY HAPPENS.
        //
        // A machine with no Vulkan device gets this object's operations on
        // the host, and the place says the host: a status block that said
        // "the GPU" about a run the host computed would be a block that
        // lies, and the CLI refuses `--backend vulkan` on such a machine
        // anyway -- this is the truth underneath that refusal.
        if self.backend.has_gpu() {
            Placement::Vulkan
        } else {
            Placement::Host
        }
    }

    fn device(&self) -> Option<&str> {
        self.backend.device_name()
    }

    fn dispatches(&self) -> usize {
        self.dispatches
    }
}
