//! The NPU backend.
//!
//! # Two transports, one API
//!
//! * **`xrt`** (feature `xrt`, the default): a small C shim over XRT's reference
//!   API, compiled by `build.rs`. This is what the project has always used.
//!
//! * **`direct`** (feature `direct`): ioctls of the `amdxdna` driver on
//!   `/dev/accel/accel0`. No XRT, no reference at all, and -- the part that
//!   matters for a runtime that is being rewritten in Rust -- no Python.
//!   The reference runtime needed `PYTHONPATH=/opt/xilinx/xrt/python`, which meant
//!   the runtime required Python to run. That is not a fact about the
//!   hardware; it is a fact about XRT's packaging, and this backend does not
//!   have to inherit it.
//!
//! Which transport is in use is a property of the BUILD and not of the
//! runtime's behaviour: the two answer the same calls.
//!
//! # Two generations, and they are not the same
//!
//! XDNA1 (`aie2`) and XDNA2 (`aie2p`) differ in ways that are invisible
//! until they are wrong:
//!
//! | what | npu1 | npu2 |
//! |---|---|---|
//! | the B panel's sub-tile (bf16) | (8, 4) | (8, 8) |
//! | the instruction set | aie2 | aie2p |
//! | the hardware context budget | known | **not yet measured** |
//!
//! The sub-tile is the dangerous one: a container packed for one board and
//! read by a design built for the other is NOT rejected anywhere. The byte
//! count agrees, the shapes agree, and both sides derive the same
//! `layout_hash` from the same wrong constant -- and every product is
//! plausible. See `geometry.rs`.
//!
//! The context budget is the other one, and it is a budget of NOTHING rather
//! than a budget of zero: `Option<Budget>` with `None` meaning "this cannot
//! be compared", so that an unmeasured budget is not silently read as "there
//! is room".

pub mod array;
pub mod array_convs;
pub mod array_gemms;
pub mod attention;
pub mod conv;
pub mod design;
pub mod device;
pub mod eltwise;
pub mod gemm;
pub mod geometry;
pub mod manifest;
pub mod transport;

// The direct transport is compiled WHATEVER the build has: it needs no XRT
// and no shim, and the driver queries it holds -- which hardware contexts
// exist, right now -- are useful to a build that speaks through XRT too.
pub mod direct;
#[cfg(feature = "xrt")]
pub mod xrt;

pub mod contention;
pub mod dispatch;
pub mod xclbin;

pub use array::{ArrayDesign, ArrayError};
pub use design::{Design, DesignSet};
pub use device::{Device, DeviceError, Generation};
pub use geometry::Geometry;
pub use manifest::DesignInfo;
pub use transport::{BufferFlags, Transport, TransportError};

use npum_model::caps::Precision as CapsPrecision;
use npum_model::{Backend, Caps, ModelSpec, OpKind, Plan, Result, Tensor};

/// The NPU backend.
pub struct XdnaBackend {
    device: Device,
    caps: Caps,
    plan: Plan,
}

impl XdnaBackend {
    /// A backend for the device this process can see.
    pub fn open() -> std::result::Result<Self, DeviceError> {
        let device = Device::detect()?;
        let generation = device.generation();
        let caps = Caps {
            ops: OpKind::ALL.to_vec(),
            precision: CapsPrecision::Bf16,
            generation: Some(generation),
            // NOT bit-exact with the CPU, and never will be: the array's
            // datapath is bf16 and its intermediates are bf16. The tolerance
            // between them is in docs/en/BACKENDS.md.
            bit_exact_with_cpu: false,
        };
        Ok(Self {
            device,
            caps,
            plan: Plan::default(),
        })
    }

    /// The device this backend is bound to.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl Backend for XdnaBackend {
    fn name(&self) -> &'static str {
        "xdna"
    }

    fn caps(&self) -> Caps {
        self.caps.clone()
    }

    fn prepare(&mut self, spec: &ModelSpec) -> Result<Plan> {
        let generation = self.device.generation();
        let geometry = Geometry::for_generation(generation);
        self.plan = Plan {
            dispatched: OpKind::ALL.iter().map(|o| o.as_str().to_string()).collect(),
            host_side: Vec::new(),
            note: format!(
                "{} on {}: B panel sub-tile (s={}, t={})",
                spec.arch_name,
                generation.as_str(),
                geometry.mac_s(),
                geometry.mac_t()
            ),
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
        dispatch::gemm(self, a, b, c, acc)
    }

    fn gelu(&mut self, x: &mut Tensor) -> Result<()> {
        dispatch::gelu(self, x)
    }

    fn layernorm(&mut self, x: &mut Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Result<()> {
        dispatch::layernorm(self, x, gamma, beta, eps)
    }

    fn softmax(&mut self, x: &mut Tensor) -> Result<()> {
        dispatch::softmax(self, x)
    }

    fn attention(&mut self, q: &Tensor, k: &Tensor, v: &Tensor, out: &mut Tensor) -> Result<()> {
        dispatch::attention(self, q, k, v, out)
    }

    fn conv1d(
        &mut self,
        x: &mut Tensor,
        w: &Tensor,
        b: Option<&Tensor>,
        stride: i64,
        pad: i64,
    ) -> Result<()> {
        dispatch::conv1d(self, x, w, b, stride, pad)
    }

    fn conv2d(
        &mut self,
        x: &mut Tensor,
        w: &Tensor,
        b: Option<&Tensor>,
        stride: (i64, i64),
        pad: (i64, i64),
    ) -> Result<()> {
        dispatch::conv2d(self, x, w, b, stride, pad)
    }

    fn mproj(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        dispatch::mproj(self, x, out)
    }

    fn fft(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        dispatch::fft(self, x, out)
    }

    fn logit(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        dispatch::logit(self, x, out)
    }
}
