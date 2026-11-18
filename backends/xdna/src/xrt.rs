//! The XRT transport: the C shim over XRT's reference API.
//!
//! # Why a C shim and not a crate
//!
//! XRT's API is reference and its ABI is not stable. There is no Rust binding for
//! it, and there will not be one, so the options are a C shim or a reference
//! binary spoken to over a pipe. The shim wins:
//!
//! * it is about 300 lines, against a pipe protocol that would have to
//!   describe buffers, barriers and error codes;
//! * it is compiled by `build.rs`, which means the build stays offline and
//!   needs no crate registry entry for a reference library;
//! * the reference runtime did exactly this, so the shim's semantics are known.
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
//! XRT's device discovery lives in its Python bindings. The reference runtime
//! found the device through `xrt._NPU_RUNTIME`, which is why it needed
//! `PYTHONPATH=/opt/xilinx/xrt/python` at RUN time. The shim links against
//! the reference library directly, so the Python is a build-time dependency of the
//! toolchain and not a runtime dependency of the product.
//!
//! # THE HANDLE IS OWNED BY THE RUST SIDE
//!
//! Every object the shim creates is freed by the shim's own close function,
//! and the Rust struct that owns it does the freeing. The shim allocates with
//! `new` and Rust frees with the matching close, and nothing is allocated on
//! one side and freed on the other.

use std::ffi::{CString, c_char, c_void};

use crate::transport::{BufferFlags, Run, Transport, TransportError};

/// The shim's flag for a cached, device-side buffer.
const SHIM_BO_CACHEABLE: i32 = 1;
/// The shim's flag for a host-visible buffer with a device address.
const SHIM_BO_HOST_ONLY: i32 = 2;

/// The ERT state of a run that completed.
///
/// The ERT's own numbering, which is NOT a 0-based enumeration: NEW is 1,
/// RUNNING is 3 and COMPLETED is 4. The number is from
/// `xrt/detail/ert.h`, and a runtime that assumed it started at zero would
/// read a running kernel as a completed one.
const ERT_CMD_STATE_COMPLETED: i32 = 4;

#[cfg(all(feature = "xrt", xrt_shim))]
unsafe extern "C" {
    fn shim_last_error() -> *const c_char;
    fn shim_device_open(index: u32, out_device: *mut *mut c_void) -> i32;
    fn shim_device_close(device: *mut c_void);
    fn shim_context_open(
        device: *mut c_void,
        xclbin: *const u8,
        bytes: usize,
        out_context: *mut *mut c_void,
    ) -> i32;
    fn shim_context_close(context: *mut c_void);
    fn shim_kernel_open(
        context: *mut c_void,
        name: *const c_char,
        out_kernel: *mut *mut c_void,
    ) -> i32;
    fn shim_kernel_close(kernel: *mut c_void);
    fn shim_kernel_group_id(kernel: *mut c_void, argno: i32) -> i32;
    fn shim_bo_alloc(
        device: *mut c_void,
        bytes: u64,
        flags: i32,
        group: i32,
        out_bo: *mut *mut c_void,
    ) -> i32;
    fn shim_bo_write(bo: *mut c_void, data: *const c_void, bytes: u64, offset: u64) -> i32;
    fn shim_bo_read(bo: *mut c_void, data: *mut c_void, bytes: u64, offset: u64) -> i32;
    fn shim_bo_sync(bo: *mut c_void, to_device: i32, bytes: u64, offset: u64) -> i32;
    fn shim_bo_size(bo: *mut c_void) -> u64;
    fn shim_bo_free(bo: *mut c_void);
    fn shim_run_start(
        kernel: *mut c_void,
        instr_bo: *mut c_void,
        words: u32,
        args: *mut *mut c_void,
        n_args: i32,
        out_run: *mut *mut c_void,
    ) -> i32;
    fn shim_run_wait(run: *mut c_void) -> i32;
    fn shim_run_free(run: *mut c_void);
}

/// The error the shim last reported.
///
/// The driver's own words, not a paraphrase of them. `err=-22` says that
/// something was refused and not what; the shim's text is the difference
/// between "the device is busy" and "the xclbin has no kernel with that
/// name".
fn last_error() -> TransportError {
    #[cfg(all(feature = "xrt", xrt_shim))]
    {
        let text = unsafe {
            let ptr = shim_last_error();
            if ptr.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        if text.is_empty() {
            TransportError::new("the XRT call failed without a message")
        } else {
            TransportError::new(text)
        }
    }
    #[cfg(not(all(feature = "xrt", xrt_shim)))]
    {
        TransportError::new("this build has no XRT shim")
    }
}

/// A buffer on the device.
pub struct XrtBuffer {
    handle: *mut c_void,
}

// SAFETY: the handle is owned by this struct and is used by the thread that
// owns it. The runtime dispatches one queue per thread, so a buffer is never
// touched by two threads at once.
unsafe impl Send for XrtBuffer {}

impl Drop for XrtBuffer {
    fn drop(&mut self) {
        #[cfg(all(feature = "xrt", xrt_shim))]
        unsafe {
            if !self.handle.is_null() {
                shim_bo_free(self.handle);
            }
        }
        self.handle = std::ptr::null_mut();
    }
}

/// The transport over XRT.
pub struct XrtTransport {
    device: *mut c_void,
    context: *mut c_void,
    kernel: *mut c_void,
}

// SAFETY: the handles are owned by this struct and are used by the thread
// that owns it.
unsafe impl Send for XrtTransport {}

impl Drop for XrtTransport {
    fn drop(&mut self) {
        #[cfg(all(feature = "xrt", xrt_shim))]
        unsafe {
            if !self.kernel.is_null() {
                shim_kernel_close(self.kernel);
            }
            if !self.context.is_null() {
                shim_context_close(self.context);
            }
            if !self.device.is_null() {
                shim_device_close(self.device);
            }
        }
        self.kernel = std::ptr::null_mut();
        self.context = std::ptr::null_mut();
        self.device = std::ptr::null_mut();
    }
}

impl Transport for XrtTransport {
    type Buffer = XrtBuffer;

    fn open(index: u32) -> Result<Self, TransportError> {
        #[cfg(not(all(feature = "xrt", xrt_shim)))]
        {
            let _ = index;
            return Err(TransportError::new(
                "this build has no XRT shim: it was compiled without the `xrt` \
                 feature, or without XRT installed at build time",
            ));
        }
        #[cfg(all(feature = "xrt", xrt_shim))]
        {
            let mut device: *mut c_void = std::ptr::null_mut();
            let rc = unsafe { shim_device_open(index, &mut device) };
            if rc != 0 {
                return Err(last_error());
            }
            Ok(Self {
                device,
                context: std::ptr::null_mut(),
                kernel: std::ptr::null_mut(),
            })
        }
    }

    fn open_context(&mut self, xclbin: &[u8]) -> Result<(), TransportError> {
        #[cfg(not(all(feature = "xrt", xrt_shim)))]
        {
            let _ = xclbin;
            return Err(TransportError::new("this build has no XRT shim"));
        }
        #[cfg(all(feature = "xrt", xrt_shim))]
        {
            let mut context: *mut c_void = std::ptr::null_mut();
            let rc = unsafe {
                shim_context_open(self.device, xclbin.as_ptr(), xclbin.len(), &mut context)
            };
            if rc != 0 {
                return Err(last_error());
            }
            self.context = context;
            Ok(())
        }
    }

    fn open_kernel(&mut self, name: &str) -> Result<(), TransportError> {
        #[cfg(not(all(feature = "xrt", xrt_shim)))]
        {
            let _ = name;
            return Err(TransportError::new("this build has no XRT shim"));
        }
        #[cfg(all(feature = "xrt", xrt_shim))]
        {
            let c_name = CString::new(name)
                .map_err(|_| TransportError::new("a kernel name with a NUL in it"))?;
            let mut kernel: *mut c_void = std::ptr::null_mut();
            let rc = unsafe { shim_kernel_open(self.context, c_name.as_ptr(), &mut kernel) };
            if rc != 0 {
                return Err(last_error());
            }
            self.kernel = kernel;
            Ok(())
        }
    }

    fn group_id(&self, argno: i32) -> Result<i32, TransportError> {
        #[cfg(not(all(feature = "xrt", xrt_shim)))]
        {
            let _ = argno;
            return Err(TransportError::new("this build has no XRT shim"));
        }
        #[cfg(all(feature = "xrt", xrt_shim))]
        {
            let group = unsafe { shim_kernel_group_id(self.kernel, argno) };
            if group < 0 {
                return Err(last_error());
            }
            Ok(group)
        }
    }

    fn alloc(
        &mut self,
        bytes: u64,
        flags: BufferFlags,
        group: i32,
    ) -> Result<Self::Buffer, TransportError> {
        #[cfg(not(all(feature = "xrt", xrt_shim)))]
        {
            let _ = (bytes, flags, group);
            return Err(TransportError::new("this build has no XRT shim"));
        }
        #[cfg(all(feature = "xrt", xrt_shim))]
        {
            let shim_flags = match flags {
                BufferFlags::Cacheable => SHIM_BO_CACHEABLE,
                BufferFlags::HostOnly => SHIM_BO_HOST_ONLY,
            };
            let mut bo: *mut c_void = std::ptr::null_mut();
            let rc = unsafe { shim_bo_alloc(self.device, bytes, shim_flags, group, &mut bo) };
            if rc != 0 {
                return Err(last_error());
            }
            Ok(XrtBuffer { handle: bo })
        }
    }

    fn write(bo: &Self::Buffer, data: &[u8], offset: u64) -> Result<(), TransportError> {
        #[cfg(not(all(feature = "xrt", xrt_shim)))]
        {
            let _ = (bo, data, offset);
            return Err(TransportError::new("this build has no XRT shim"));
        }
        #[cfg(all(feature = "xrt", xrt_shim))]
        {
            let rc = unsafe {
                shim_bo_write(
                    bo.handle,
                    data.as_ptr() as *const c_void,
                    data.len() as u64,
                    offset,
                )
            };
            if rc != 0 {
                return Err(last_error());
            }
            Ok(())
        }
    }

    fn size(bo: &Self::Buffer) -> u64 {
        #[cfg(all(feature = "xrt", xrt_shim))]
        unsafe {
            shim_bo_size(bo.handle)
        }
        #[cfg(not(all(feature = "xrt", xrt_shim)))]
        {
            let _ = bo;
            0
        }
    }

    fn read(bo: &Self::Buffer, data: &mut [u8], offset: u64) -> Result<(), TransportError> {
        #[cfg(not(all(feature = "xrt", xrt_shim)))]
        {
            let _ = (bo, data, offset);
            return Err(TransportError::new("this build has no XRT shim"));
        }
        #[cfg(all(feature = "xrt", xrt_shim))]
        {
            let rc = unsafe {
                shim_bo_read(
                    bo.handle,
                    data.as_mut_ptr() as *mut c_void,
                    data.len() as u64,
                    offset,
                )
            };
            if rc != 0 {
                return Err(last_error());
            }
            Ok(())
        }
    }

    fn sync_to_device(bo: &Self::Buffer, bytes: u64, offset: u64) -> Result<(), TransportError> {
        #[cfg(not(all(feature = "xrt", xrt_shim)))]
        {
            let _ = (bo, bytes, offset);
            return Err(TransportError::new("this build has no XRT shim"));
        }
        #[cfg(all(feature = "xrt", xrt_shim))]
        {
            let rc = unsafe { shim_bo_sync(bo.handle, 1, bytes, offset) };
            if rc != 0 {
                return Err(last_error());
            }
            Ok(())
        }
    }

    fn sync_from_device(bo: &Self::Buffer, bytes: u64, offset: u64) -> Result<(), TransportError> {
        #[cfg(not(all(feature = "xrt", xrt_shim)))]
        {
            let _ = (bo, bytes, offset);
            return Err(TransportError::new("this build has no XRT shim"));
        }
        #[cfg(all(feature = "xrt", xrt_shim))]
        {
            let rc = unsafe { shim_bo_sync(bo.handle, 0, bytes, offset) };
            if rc != 0 {
                return Err(last_error());
            }
            Ok(())
        }
    }

    fn start(
        &mut self,
        instr: &Self::Buffer,
        words: u32,
        args: &[&Self::Buffer],
    ) -> Result<Run, TransportError> {
        #[cfg(not(all(feature = "xrt", xrt_shim)))]
        {
            let _ = (instr, words, args);
            return Err(TransportError::new("this build has no XRT shim"));
        }
        #[cfg(all(feature = "xrt", xrt_shim))]
        {
            let mut handles: Vec<*mut c_void> = args.iter().map(|b| b.handle).collect();
            let mut run: *mut c_void = std::ptr::null_mut();
            let rc = unsafe {
                shim_run_start(
                    self.kernel,
                    instr.handle,
                    words,
                    handles.as_mut_ptr(),
                    handles.len() as i32,
                    &mut run,
                )
            };
            if rc != 0 {
                return Err(last_error());
            }
            Ok(Run::new(run, shim_run_free))
        }
    }

    fn wait(run: &Run) -> Result<(), TransportError> {
        #[cfg(not(all(feature = "xrt", xrt_shim)))]
        {
            let _ = run;
            return Err(TransportError::new("this build has no XRT shim"));
        }
        #[cfg(all(feature = "xrt", xrt_shim))]
        {
            let state = unsafe { shim_run_wait(run.handle()) };
            if state < 0 {
                return Err(last_error());
            }
            if state != ERT_CMD_STATE_COMPLETED {
                return Err(TransportError::new(format!(
                    "the kernel did not complete: ERT state {state}"
                )));
            }
            Ok(())
        }
    }
}
