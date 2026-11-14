//! The XRT transport: a C shim over XRT's C++ API.
//!
//! # Why a C shim and not a crate
//!
//! XRT's API is C++ and its ABI is not stable. There is no Rust binding for
//! it, and there will not be one, so the options are a C shim or a C++
//! binary spoken to over a pipe. The shim wins:
//!
//! * it is about 400 lines, against a pipe protocol that would have to
//!   describe buffers, barriers and error codes;
//! * it is compiled by `build.rs`, which means the build stays offline and
//!   needs no crate registry entry for a C++ library;
//! * the C++ runtime did exactly this, so the shim's semantics are known.
//!
//! # What the shim must NOT do
//!
//! It must not contain any policy. Everything above this file -- the
//! geometry, the design check, the placement -- is Rust, and the shim's job
//! is to turn Rust's calls into XRT's calls. A shim that decides anything is
//! a second implementation of the runtime's rules, and the two would drift.
//!
//! # The Python problem this solves
//!
//! XRT's device discovery lives in its Python bindings. The C++ runtime
//! found the device through `xrt._NPU_RUNTIME`, which is why it needed
//! `PYTHONPATH=/opt/xilinx/xrt/python` at RUN time. The shim links against
//! the C++ library directly, so the Python is a build-time dependency of the
//! toolchain and not a runtime dependency of the product.

use std::ffi::c_void;

/// The opaque device handle the shim owns.
#[derive(Debug)]
pub struct XrtDevice {
    handle: *mut c_void,
}

// SAFETY: the handle is owned by this struct and is only used from the
// thread that owns it. The shim's device calls are thread-safe in XRT, but
// the runtime dispatches one queue per thread, so the handle is not shared.
unsafe impl Send for XrtDevice {}

impl XrtDevice {
    /// Open the first device.
    pub fn open() -> Option<Self> {
        let handle = unsafe { xrt_shim_open() };
        if handle.is_null() {
            None
        } else {
            Some(Self { handle })
        }
    }

    /// The underlying handle.
    pub fn handle(&self) -> *mut c_void {
        self.handle
    }
}

impl Drop for XrtDevice {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { xrt_shim_close(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

unsafe extern "C" {
    fn xrt_shim_open() -> *mut c_void;
    fn xrt_shim_close(handle: *mut c_void);
}
