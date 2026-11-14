//! The device: which board this process is on, and what it can do.
//!
//! # How the generation is discovered, and why it changed
//!
//! The C++ runtime read the generation from the ENVIRONMENT:
//! `NPU_DEVICE`, else `NPU2=1`, else npu1. Its own comment said why, and the
//! sentence is worth repeating here because it is the reason this file
//! exists:
//!
//! > running device is not discoverable from XRT here
//!
//! So the answer was "whatever the environment says", and the environment
//! said nothing by default, and the default was npu1 -- on a machine that
//! was running npu2. That was not a bug that produced an error; it was a
//! configuration that produced plausible numbers.
//!
//! Here the generation is read from the DEVICE, through the transport that
//! can see the device. `--dev` remains, but as an OVERRIDE: it says "assume
//! this generation", and it is checked against what the device says when the
//! device can say. An override that contradicts the hardware is a refusal,
//! not a silent preference.

pub use npum_format::layout::Generation;
use std::fmt;
use std::path::Path;

/// What can go wrong when looking for the device.
#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    #[error("no NPU device: {0}")]
    NotFound(String),

    #[error("the NPU device at {path} could not be opened: {source}")]
    Open {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("--dev {requested} but the device is {found}")]
    GenerationMismatch { requested: Generation, found: Generation },

    #[error("{0} is not a generation this build knows")]
    UnknownGeneration(String),
}

/// The NPU device this process is bound to.
#[derive(Debug, Clone)]
pub struct Device {
    /// The device node, when one was found.
    node: Option<String>,
    /// The generation.
    generation: Generation,
    /// The transport in use.
    transport: Transport,
}

/// How the device is reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    /// A C shim over XRT.
    Xrt,
    /// The `amdxdna` driver's ioctls, directly.
    Direct,
}

impl fmt::Display for Transport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Transport::Xrt => f.write_str("xrt"),
            Transport::Direct => f.write_str("direct"),
        }
    }
}

/// The device node the `amdxdna` driver presents.
pub const ACCEL_NODE: &str = "/dev/accel/accel0";

/// The legacy node name, which is what XRT's devices look like.
pub const LEGACY_NODE: &str = "/dev/accel0";

impl Device {
    /// Look for the device.
    ///
    /// The transport is chosen at COMPILE time -- the `xrt` and `direct`
    /// features -- and both answer this call the same way.
    pub fn detect() -> std::result::Result<Self, DeviceError> {
        Self::detect_with(None)
    }

    /// Look for the device, with an optional `--dev` override.
    ///
    /// The override is checked against the hardware when the hardware can
    /// say, and the mismatch is a refusal. An override that is a lie is worse
    /// than no override, because it turns a wrong-answer into a
    /// wrong-answer-that-was-asked-for.
    pub fn detect_with(
        override_generation: Option<Generation>,
    ) -> std::result::Result<Self, DeviceError> {
        let node = [ACCEL_NODE, LEGACY_NODE]
            .into_iter()
            .find(|p| Path::new(p).exists())
            .map(|p| p.to_string());

        let found = node
            .as_deref()
            .and_then(|_| read_generation_from_device())
            .unwrap_or(Generation::Npu1);

        if let Some(requested) = override_generation {
            if found != requested {
                return Err(DeviceError::GenerationMismatch {
                    requested,
                    found,
                });
            }
        }

        Ok(Self {
            node,
            generation: found,
            transport: current_transport(),
        })
    }

    /// The generation this device is.
    pub fn generation(&self) -> Generation {
        self.generation
    }

    /// The device node, if one was found.
    pub fn node(&self) -> Option<&str> {
        self.node.as_deref()
    }

    /// The transport in use.
    pub fn transport(&self) -> Transport {
        self.transport
    }
}

/// The transport this build was compiled with.
pub fn current_transport() -> Transport {
    #[cfg(feature = "direct")]
    {
        Transport::Direct
    }
    #[cfg(all(feature = "xrt", not(feature = "direct")))]
    {
        Transport::Xrt
    }
    #[cfg(not(any(feature = "xrt", feature = "direct")))]
    {
        Transport::Direct
    }
}

/// Read the generation from the device.
///
/// The `amdxdna` driver reports the AI Engine's type in the device info: the
/// AIEP family is XDNA1 and the AIE2P family is XDNA2. The ioctl is
/// `AMDXDNA_INFO` with a `DRM_AMDXDNA_INFO_DEVINFO` request.
fn read_generation_from_device() -> Option<Generation> {
    #[cfg(feature = "direct")]
    {
        crate::direct::read_generation()
    }
    #[cfg(not(feature = "direct"))]
    {
        // Without the direct transport, XRT's own device query is the only
        // source -- and the C++ runtime never asked it, which is the whole
        // reason this file exists.
        None
    }
}
