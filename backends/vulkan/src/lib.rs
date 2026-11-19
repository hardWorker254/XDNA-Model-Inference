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
    /// Reusable host staging for tensor -> f32 conversions. The tensor API
    /// used to allocate a fresh `Vec` per operand per call; these four slots
    /// cover the widest op (a, b, accumulator/gamma/beta) and grow once.
    scratch_a: Vec<f32>,
    scratch_b: Vec<f32>,
    scratch_c: Vec<f32>,
    /// Reusable staging for f32 -> tensor byte assembly.
    bytes: Vec<u8>,
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
            scratch_a: Vec::new(),
            scratch_b: Vec::new(),
            scratch_c: Vec::new(),
            bytes: Vec::new(),
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

    /// Decode a tensor into one of the reusable f32 staging buffers.
    fn decode<'a>(tensor: &Tensor, out: &'a mut Vec<f32>) -> &'a mut Vec<f32> {
        tensor.to_f32_into(out);
        out
    }

    /// Encode f32 values into a tensor's bytes, reusing the byte buffer.
    fn encode(tensor: &mut Tensor, values: &[f32], bytes: &mut Vec<u8>) {
        bytes.clear();
        bytes.reserve(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        tensor.bytes_mut().copy_from_slice(bytes);
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
        let Some(gpu) = self.gpu.as_mut() else {
            return npum_cpu::kernels::gemm(a, b, c, acc);
        };
        let a_values = Self::decode(a, &mut self.scratch_a);
        let b_values = Self::decode(b, &mut self.scratch_b);
        let mut values = gpu
            .gemm(a_values, b_values, m, k, n)
            .map_err(|d| Self::failed("gemm", d))?;
        if let Some(acc) = acc {
            let acc_values = Self::decode(acc, &mut self.scratch_c);
            values = gpu
                .add(&values, acc_values)
                .map_err(|d| Self::failed("gemm", d))?;
        }
        Self::encode(c, &values, &mut self.bytes);
        Ok(())
    }

    fn gelu(&mut self, x: &mut Tensor) -> Result<()> {
        if x.dtype() != DType::F32 {
            return npum_cpu::kernels::gelu_(x);
        }
        let Some(gpu) = self.gpu.as_mut() else {
            return npum_cpu::kernels::gelu_(x);
        };
        let values = Self::decode(x, &mut self.scratch_a);
        gpu.gelu(values).map_err(|d| Self::failed("gelu", d))?;
        Self::encode(x, values, &mut self.bytes);
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
        let Some(gpu) = self.gpu.as_mut() else {
            return npum_cpu::kernels::layernorm_(x, gamma, beta, eps);
        };
        let rows = x.len() / width;
        let values = Self::decode(x, &mut self.scratch_a);
        let g = Self::decode(gamma, &mut self.scratch_b);
        let b = Self::decode(beta, &mut self.scratch_c);
        gpu.layernorm(
            values,
            rows,
            width,
            (!g.is_empty()).then_some(&g[..]),
            (!b.is_empty()).then_some(&b[..]),
        )
        .map_err(|d| Self::failed("layernorm", d))?;
        Self::encode(x, values, &mut self.bytes);
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
        let Some(gpu) = self.gpu.as_mut() else {
            return npum_cpu::kernels::softmax_(x);
        };
        let rows = x.len() / width;
        let values = Self::decode(x, &mut self.scratch_a);
        gpu.softmax(values, rows, width)
            .map_err(|d| Self::failed("softmax", d))?;
        Self::encode(x, values, &mut self.bytes);
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
        let Some(gpu) = self.gpu.as_mut() else {
            return npum_cpu::kernels::attention(q, k, v, out);
        };
        let q_values = Self::decode(q, &mut self.scratch_a);
        let k_values = Self::decode(k, &mut self.scratch_b);
        let v_values = Self::decode(v, &mut self.scratch_c);
        let values = gpu
            .attention(q_values, k_values, v_values, d)
            .map_err(|d| Self::failed("attn", d))?;
        Self::encode(out, &values, &mut self.bytes);
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
