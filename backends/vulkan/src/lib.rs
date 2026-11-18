//! The GPU backend: Vulkan.
//!
//! # What this backend is FOR
//!
//! Not throughput. An iGPU is usually idle while the NPU is busy, and the
//! hybrid -- "the array does the GEMM, the GPU does the attention and the
//! softmax" -- is a real win, because attention is where the array's
//! dispatch overhead hurts most.
//!
//! # The precision, stated plainly
//!
//! A GPU's f32 is NOT the CPU's f32. It rounds differently, it may fuse
//! multiply-add, and its transcendental functions differ. Two backends that
//! both claim to compute f32 do not compute the same f32, and this backend's
//! gates are therefore gates WITH TOLERANCES, not bit-for-bit comparisons.
//!
//! The tolerance matrix per operation is published in `docs/en/BACKENDS.md`,
//! and the approach is not new to this project: `reference/precision_study.py`
//! already measured the f32 spread between backends.
//!
//! # What runs on the GPU, and what does not yet
//!
//! | operation | where |
//! |---|---|
//! | `gemm`, `gelu`, `layernorm`, `softmax`, `attn` | the GPU, in f32 |
//! | `conv`, `mproj`, `fft`, `logit` | the host, through `npum_cpu::kernels` |
//!
//! The four host-side operations have shaders in `shaders/` but no dispatch
//! path yet: convolution is the one that needs a layout decision (the array's
//! folded convs are 3x3 stride-1 and use im2col; the GPU would want a tiled
//! winograd or a direct gather), and a GPU path that is slower than the host
//! path is not worth having. Until it is measured, they stay where they are
//! and the plan says so rather than claiming nine of nine.
//!
//! # Fallbacks, and when they happen
//!
//! * **No Vulkan device.** Everything runs on `npum_cpu::kernels`, and the
//!   plan's note says the GPU was not opened and why. A machine without a
//!   GPU runs the same numbers as the CPU backend, because it IS the CPU
//!   backend.
//! * **A tensor that is not f32.** The GPU path is f32; bf16 and i8 operands
//!   come back to the host for that operation. `Caps::precision` says f32, so
//!   this is the container's doing, not the shader's.
//! * **Attention longer than 1024 keys.** The attention shader keeps one
//!   score row in private memory, which is 1024 f32; a longer sequence falls
//!   back to the host rather than computing a truncated softmax.

pub mod device;
pub mod ops;
pub mod shaders;

use npum_model::caps::Precision as CapsPrecision;
use npum_model::{Backend, Caps, DType, Error, ModelSpec, OpKind, Plan, Result, Tensor};

use device::Gpu;

/// The longest key sequence the attention shader can take.
pub const MAX_ATTENTION_KEYS: usize = 1024;

/// The Vulkan backend.
pub struct VulkanBackend {
    /// `None` when no device could be opened, in which case every operation
    /// runs on the host and the plan says why.
    gpu: Option<Gpu>,
    caps: Caps,
    plan: Plan,
}

impl Default for VulkanBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl VulkanBackend {
    /// A Vulkan backend: the device if there is one, the host otherwise.
    pub fn new() -> Self {
        let gpu = match Gpu::open() {
            Ok(gpu) => Some(gpu),
            Err(reason) => {
                tracing::warn!(reason, "no Vulkan device; the host runs every operation");
                None
            }
        };
        let caps = Caps {
            ops: OpKind::ALL.to_vec(),
            precision: CapsPrecision::F32,
            generation: None,
            // NOT bit-exact with the CPU, and the gates say so.
            bit_exact_with_cpu: false,
        };
        Self {
            gpu,
            caps,
            plan: Plan::default(),
        }
    }

    /// Whether a GPU was opened.
    pub fn has_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// The device's name, when there is one.
    pub fn device_name(&self) -> Option<&str> {
        self.gpu.as_ref().map(Gpu::name)
    }

    /// The operations the GPU runs itself.
    pub fn gpu_ops() -> [OpKind; 5] {
        [
            OpKind::Gemm,
            OpKind::Gelu,
            OpKind::Layernorm,
            OpKind::Softmax,
            OpKind::Attn,
        ]
    }

    /// The GPU, when this operation is one it runs for these tensors.
    fn gpu(&mut self) -> Option<&mut Gpu> {
        self.gpu.as_mut()
    }

    /// The device itself, when there is one.
    ///
    /// The seam (`crate::ops`) works on flat slices, and the device's own
    /// methods are the slice-shaped ones -- so the seam borrows the device
    /// rather than wrapping every activation in a tensor and copying it
    /// twice.
    pub fn device_mut(&mut self) -> Option<&mut Gpu> {
        self.gpu.as_mut()
    }

    /// A backend failure that names the operation.
    fn failed(op: &str, detail: String) -> Error {
        Error::Backend {
            backend: "vulkan",
            detail: format!("{op}: {detail}"),
        }
    }

    /// Whether every one of these tensors is f32, which is what the GPU runs.
    fn all_f32(tensors: &[&Tensor]) -> bool {
        tensors.iter().all(|t| t.dtype() == DType::F32)
    }
}

impl Backend for VulkanBackend {
    fn name(&self) -> &'static str {
        "vulkan"
    }

    fn caps(&self) -> Caps {
        self.caps.clone()
    }

    fn prepare(&mut self, spec: &ModelSpec) -> Result<Plan> {
        let (dispatched, note) = match self.gpu.as_ref() {
            Some(gpu) => (
                Self::gpu_ops()
                    .iter()
                    .map(|o| o.as_str().to_string())
                    .collect(),
                format!(
                    "the GPU backend runs gemm, gelu, layernorm, softmax and attn of {} \
                     on {}; conv, mproj, fft and logit stay on the host",
                    spec.arch_name,
                    gpu.name()
                ),
            ),
            None => (
                Vec::new(),
                "no Vulkan device was opened: every operation runs on the host".to_string(),
            ),
        };
        self.plan = Plan {
            dispatched,
            host_side: OpKind::ALL
                .iter()
                .map(|o| o.as_str().to_string())
                .filter(|o| !self.plan.runs(o))
                .collect(),
            note,
        };
        Ok(self.plan.clone())
    }

    fn gemm(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        c: &mut Tensor,
        acc: Option<&mut Tensor>,
    ) -> Result<()> {
        if !Self::all_f32(&[a, b, c]) {
            return npum_cpu::kernels::gemm(a, b, c, acc);
        }
        let (m, k) = (a.shape()[0] as usize, a.shape()[1] as usize);
        let n = c.shape().last().copied().unwrap_or(0) as usize;
        let Some(gpu) = self.gpu() else {
            return npum_cpu::kernels::gemm(a, b, c, acc);
        };
        let mut values = gpu
            .gemm(&a.to_f32(), &b.to_f32(), m, k, n)
            .map_err(|d| Self::failed("gemm", d))?;
        if let Some(acc) = acc {
            let summed = gpu
                .add(&values, &acc.to_f32())
                .map_err(|d| Self::failed("gemm", d))?;
            values = summed;
        }
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        c.bytes_mut().copy_from_slice(&bytes);
        Ok(())
    }

    fn gelu(&mut self, x: &mut Tensor) -> Result<()> {
        if x.dtype() != DType::F32 {
            return npum_cpu::kernels::gelu_(x);
        }
        let Some(gpu) = self.gpu() else {
            return npum_cpu::kernels::gelu_(x);
        };
        let mut values = x.to_f32();
        gpu.gelu(&mut values).map_err(|d| Self::failed("gelu", d))?;
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        x.bytes_mut().copy_from_slice(&bytes);
        Ok(())
    }

    fn layernorm(&mut self, x: &mut Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Result<()> {
        if !Self::all_f32(&[x, gamma, beta]) {
            return npum_cpu::kernels::layernorm_(x, gamma, beta, eps);
        }
        let width = x.shape().last().copied().unwrap_or(0) as usize;
        if width == 0 {
            return Ok(());
        }
        let Some(gpu) = self.gpu() else {
            return npum_cpu::kernels::layernorm_(x, gamma, beta, eps);
        };
        let rows = x.len() / width;
        let mut values = x.to_f32();
        let g = gamma.to_f32();
        let b = beta.to_f32();
        gpu.layernorm(
            &mut values,
            rows,
            width,
            (!g.is_empty()).then_some(&g[..]),
            (!b.is_empty()).then_some(&b[..]),
        )
        .map_err(|d| Self::failed("layernorm", d))?;
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        x.bytes_mut().copy_from_slice(&bytes);
        Ok(())
    }

    fn softmax(&mut self, x: &mut Tensor) -> Result<()> {
        if x.dtype() != DType::F32 {
            return npum_cpu::kernels::softmax_(x);
        }
        let width = x.shape().last().copied().unwrap_or(0) as usize;
        if width == 0 {
            return Ok(());
        }
        let Some(gpu) = self.gpu() else {
            return npum_cpu::kernels::softmax_(x);
        };
        let rows = x.len() / width;
        let mut values = x.to_f32();
        gpu.softmax(&mut values, rows, width)
            .map_err(|d| Self::failed("softmax", d))?;
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        x.bytes_mut().copy_from_slice(&bytes);
        Ok(())
    }

    fn attention(&mut self, q: &Tensor, k: &Tensor, v: &Tensor, out: &mut Tensor) -> Result<()> {
        if !Self::all_f32(&[q, k, v]) {
            return npum_cpu::kernels::attention(q, k, v, out);
        }
        let d = q.shape().last().copied().unwrap_or(0) as usize;
        if d == 0 {
            return Ok(());
        }
        // The shader's score row is a private array of MAX_ATTENTION_KEYS.
        if k.len() / d > MAX_ATTENTION_KEYS {
            return npum_cpu::kernels::attention(q, k, v, out);
        }
        let Some(gpu) = self.gpu() else {
            return npum_cpu::kernels::attention(q, k, v, out);
        };
        let values = gpu
            .attention(&q.to_f32(), &k.to_f32(), &v.to_f32(), d)
            .map_err(|d| Self::failed("attn", d))?;
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in &values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        out.bytes_mut().copy_from_slice(&bytes);
        Ok(())
    }

    // The four operations with no dispatch path yet: the host computes them,
    // which is what `prepare`'s plan says.
    fn conv1d(
        &mut self,
        x: &mut Tensor,
        w: &Tensor,
        b: Option<&Tensor>,
        stride: i64,
        pad: i64,
    ) -> Result<()> {
        npum_cpu::kernels::conv1d(x, w, b, stride, pad)
    }

    fn conv2d(
        &mut self,
        x: &mut Tensor,
        w: &Tensor,
        b: Option<&Tensor>,
        stride: (i64, i64),
        pad: (i64, i64),
    ) -> Result<()> {
        npum_cpu::kernels::conv2d(x, w, b, stride, pad)
    }

    fn mproj(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        npum_cpu::kernels::mproj(x, out)
    }

    fn fft(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        npum_cpu::kernels::fft(x, out)
    }

    fn logit(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        npum_cpu::kernels::logit(x, out)
    }
}
