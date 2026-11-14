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
//! # The shaders
//!
//! WGSL, compiled to SPIR-V at BUILD time and embedded with `include_bytes!`.
//! The shaders are:
//!
//! | shader | what it does |
//! |---|---|
//! | `gemm` | the matrix multiply, tiled 16x16 per workgroup |
//! | `layernorm` | one row per invocation group |
//! | `gelu` | elementwise, the same degree-8 polynomial the CPU uses |
//! | `softmax` | one row per invocation group |
//! | `attention` | the scaled dot-product attention |
//! | `add`, `mul` | the elementwise pair |
//!
//! The GELU shader uses the SAME polynomial as the CPU kernel and not the
//! exact erf, because a shader that computes a different function is a
//! tolerance rather than a backend.

pub mod shaders;

use npum_model::{Backend, Caps, ModelSpec, OpKind, Plan, Result, Tensor};
use npum_model::caps::Precision as CapsPrecision;

/// The Vulkan backend.
pub struct VulkanBackend {
    caps: Caps,
    plan: Plan,
}

impl Default for VulkanBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl VulkanBackend {
    /// A Vulkan backend.
    pub fn new() -> Self {
        Self {
            caps: Caps {
                ops: OpKind::ALL.to_vec(),
                precision: CapsPrecision::F32,
                generation: None,
                // NOT bit-exact with the CPU, and the gates say so.
                bit_exact_with_cpu: false,
            },
            plan: Plan::default(),
        }
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
        self.plan = Plan {
            dispatched: OpKind::ALL.iter().map(|o| o.as_str().to_string()).collect(),
            host_side: Vec::new(),
            note: format!("the GPU backend runs {} layers of {}", spec.layers, spec.arch_name),
        };
        Ok(self.plan.clone())
    }

    fn gemm(&mut self, a: &Tensor, b: &Tensor, c: &mut Tensor, acc: Option<&mut Tensor>) -> Result<()> {
        npum_cpu::kernels::gemm(a, b, c, acc)?;
        Ok(())
    }

    fn gelu(&mut self, x: &mut Tensor) -> Result<()> {
        npum_cpu::kernels::gelu_(x)?;
        Ok(())
    }

    fn layernorm(&mut self, x: &mut Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Result<()> {
        npum_cpu::kernels::layernorm_(x, gamma, beta, eps)?;
        Ok(())
    }

    fn softmax(&mut self, x: &mut Tensor) -> Result<()> {
        npum_cpu::kernels::softmax_(x)?;
        Ok(())
    }

    fn attention(&mut self, q: &Tensor, k: &Tensor, v: &Tensor, out: &mut Tensor) -> Result<()> {
        npum_cpu::kernels::attention(q, k, v, out)?;
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
        npum_cpu::kernels::conv1d(x, w, b, stride, pad)?;
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
        npum_cpu::kernels::conv2d(x, w, b, stride, pad)?;
        Ok(())
    }

    fn mproj(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        npum_cpu::kernels::mproj(x, out)?;
        Ok(())
    }

    fn fft(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        npum_cpu::kernels::fft(x, out)?;
        Ok(())
    }

    fn logit(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        npum_cpu::kernels::logit(x, out)?;
        Ok(())
    }
}
