//! The direct transport: the `amdxdna` driver's ioctls.
//!
//! # What this buys
//!
//! The XRT transport needs a C++ shim, a C++ toolchain at build time, and --
//! this is the part that shaped the whole project -- a Python installation:
//! XRT ships its Python bindings in `/opt/xilinx/xrt/python`, and the C++
//! runtime found the device through them. A runtime that cannot start without
//! Python is a runtime that has a Python dependency, and this project is
//! trying to have none.
//!
//! The driver itself needs none of that. `/dev/accel/accel0` and a set of
//! ioctls are enough.
//!
//! # The ioctls, and what each one is for
//!
//! | ioctl | what it does |
//! |---|---|
//! | `AMDXDNA_INFO` | the device's properties, including the AI Engine type |
//! | `AMDXDNA_CTX_CREATE` | a hardware context |
//! | `AMDXDNA_BO_ALLOC` | a buffer object |
//! | `AMDXDNA_BO_SYNC` | make a buffer visible to the device, or to the host |
//! | `AMDXDNA_QUEUE_CREATE` | a command queue |
//! | `AMDXDNA_CMD_WRITE` | a packet of commands |
//!
//! # The generation, read from the device
//!
//! `DRM_AMDXDNA_INFO_DEVINFO` reports the AI Engine's type. The AIEP family
//! is XDNA1 and the AIE2P family is XDNA2, which is how `read_generation()`
//! answers the question the C++ runtime could not.

use crate::device::Generation;

/// The `AMDXDNA_INFO` ioctl's number.
#[cfg(target_os = "linux")]
const AMDXDNA_INFO: u64 = 0x4600;

/// The `DRM_AMDXDNA_INFO_DEVINFO` request.
#[cfg(target_os = "linux")]
const DRM_AMDXDNA_INFO_DEVINFO: u32 = 0;

/// The AI Engine type the driver reports for XDNA1.
#[cfg(target_os = "linux")]
const AIE_TYPE_AIE: u32 = 1;

/// The AI Engine type the driver reports for XDNA2.
#[cfg(target_os = "linux")]
const AIE_TYPE_AIE2P: u32 = 2;

/// Read the generation from the device.
///
/// `None` when the device cannot be asked -- which is not the same as "the
/// device is npu1", and the caller must not read it that way.
#[cfg(target_os = "linux")]
pub fn read_generation() -> Option<Generation> {
    use std::os::fd::AsRawFd;

    let file = std::fs::File::open(crate::device::ACCEL_NODE).ok()?;
    // The ioctl's argument is a struct whose first field is the request.
    //
    //   struct drm_amdxna_info {
    //       __u32 ctx_id;      // context id, or zero for device info
    //       __u32 dev_type;    // the request
    //       __u64 arg;         // the request's argument
    //   };
    //
    // For DEVINFO the argument points at a drm_amdxna_device_info, whose
    // first field is the AI Engine's type.
    #[repr(C)]
    struct Info {
        ctx_id: u32,
        dev_type: u32,
        arg: u64,
    }
    #[repr(C)]
    struct DeviceInfo {
        aie_type: u32,
        _pad: [u8; 60],
    }

    let mut device_info = DeviceInfo {
        aie_type: 0,
        _pad: [0; 60],
    };
    let mut info = Info {
        ctx_id: 0,
        dev_type: DRM_AMDXDNA_INFO_DEVINFO,
        arg: (&mut device_info as *mut DeviceInfo) as u64,
    };
    let rc = unsafe { libc_ioctl(file.as_raw_fd(), AMDXDNA_INFO, &mut info) };
    if rc < 0 {
        return None;
    }
    match device_info.aie_type {
        AIE_TYPE_AIE2P => Some(Generation::Npu2),
        AIE_TYPE_AIE => Some(Generation::Npu1),
        _ => None,
    }
}

#[cfg(not(target_os = "linux"))]
pub fn read_generation() -> Option<Generation> {
    None
}

/// The raw ioctl, without a libc dependency.
///
/// The crate deliberately does not depend on `libc`: this is the only place
/// that needs a raw syscall, and the alternative -- a dependency on `libc`
/// for one call -- is a dependency that has to be kept working on every
/// platform this project supports, including the two Windows builds.
#[cfg(target_os = "linux")]
unsafe fn libc_ioctl(fd: i32, request: u64, arg: *mut std::ffi::c_void) -> i32 {
    // `ioctl` is variadic, so it cannot be called through a plain `extern
    // "C"` declaration without a variadic marker. The syscall number for
    // ioctl on x86_64 is 16, and the argument order is (fd, request, arg).
    #[cfg(target_arch = "x86_64")]
    const SYS_IOCTL: i64 = 16;
    #[cfg(target_arch = "aarch64")]
    const SYS_IOCTL: i64 = 29;

    let ret: i64;
    unsafe {
        std::arch::asm!(
            "syscall",
            inlateout("rax") SYS_IOCTL => ret,
            in("rdi") fd as i64,
            in("rsi") request,
            in("rdx") arg,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack)
        );
    }
    ret as i32
}
