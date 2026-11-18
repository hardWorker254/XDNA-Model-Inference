//! The transport boundary: what the array path asks of the hardware, and
//! nothing more.
//!
//! # TWO TRANSPORTS, ONE API
//!
//! The array can be reached two ways: through XRT's reference library (`xrt`) or
//! through the `amdxdna` driver's ioctls (`direct`). They are different
//! transports to the same board, and everything above this file -- the
//! designs, the packing, the op kernels, the families -- is written against
//! THIS interface, not against either transport.
//!
//! The reason is the same one that put the C shim between Rust and XRT: a
//! rule that exists in two places is a rule that will disagree. The rule here
//! is small ("a board, a context, a kernel, some buffers, a run"), and the two
//! implementations of it are the only two places where the two transports are
//! allowed to differ.
//!
//! # WHAT THE INTERFACE DOES NOT HAVE
//!
//! No widths, no packing, no scheduling, no decisions. Those live in the
//! design layer and the dispatch layer, where they can be read and tested
//! without a board.
//!
//! # THE HANDLE IS NOT Send BY ACCIDENT
//!
//! A transport is used by one thread at a time -- the runtime dispatches one
//! queue per thread -- so the handles are `Send` and not `Sync`. A handle
//! that were `Sync` would let two threads race on one buffer, and the symptom
//! of that race is a number that is wrong once in ten thousand runs.

use std::fmt;

/// What went wrong at the transport level.
#[derive(Debug, Clone)]
pub struct TransportError {
    /// The message, as the driver or the shim gave it.
    pub message: String,
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TransportError {}

impl TransportError {
    /// An error with a message of our own.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// How a buffer is allocated.
///
/// The instruction stream and the data are different kinds of memory, and the
/// difference is not cosmetic: the instruction stream is read by the device's
/// instruction fetch and wants to be CACHED, while the data is written by the
/// host and read by the device and wants to be HOST-VISIBLE with a device
/// address of its own. The reference runtime made the same distinction, with the
/// same two modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferFlags {
    /// The instruction stream: device-side and cached.
    Cacheable,
    /// The data: host-visible with a device address.
    HostOnly,
}

/// One run of a kernel.
///
/// Opaque on purpose: what a run is made of differs between the transports
/// (XRT's `xrt::run`, the driver's command packet), and what the caller
/// needs from it is one thing -- whether it finished, and finished well.
pub struct Run {
    handle: *mut std::ffi::c_void,
    /// How this transport frees a run.
    ///
    /// The freeing is transport-specific -- XRT's run is an `xrt::run`, the
    /// driver's is a command packet -- so the transport hands its own
    /// destructor in rather than the interface guessing. A run that is dropped
    /// without being waited for is still freed: the board is left with a
    /// finished-or-not run, not with a leaked one.
    free: unsafe extern "C" fn(*mut std::ffi::c_void),
}

// SAFETY: the handle is owned by this struct and is used by the thread that
// owns it. The runtime dispatches one queue per thread.
unsafe impl Send for Run {}

impl Run {
    /// Take ownership of a transport's run handle.
    pub(crate) fn new(
        handle: *mut std::ffi::c_void,
        free: unsafe extern "C" fn(*mut std::ffi::c_void),
    ) -> Self {
        Self { handle, free }
    }

    /// The handle, for the transport that owns it.
    pub(crate) fn handle(&self) -> *mut std::ffi::c_void {
        self.handle
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.free)(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// A transport to the array.
pub trait Transport: Sized {
    /// The transport's buffer handle.
    type Buffer;

    /// Open the board.
    ///
    /// `index` is the device number, and zero is the only board this runtime
    /// has ever used.
    fn open(index: u32) -> Result<Self, TransportError>;

    /// The hardware context for one xclbin.
    ///
    /// The xclbin is handed over as BYTES: a design set that lives inside a
    /// container has no file to point at, and the reference runtime solved the same
    /// problem by handing the blob to XRT.
    fn open_context(&mut self, xclbin: &[u8]) -> Result<(), TransportError>;

    /// Open the kernel with this name in the current context.
    fn open_kernel(&mut self, name: &str) -> Result<(), TransportError>;

    /// The memory-bank group id of one kernel argument.
    ///
    /// The instruction stream is group 1 and the data arguments are 3, 4, 5:
    /// the numbers the design's own kernel signature uses. Getting them from
    /// the kernel rather than from a literal here is what keeps the two
    /// transports from disagreeing about where a buffer lands.
    fn group_id(&self, argno: i32) -> Result<i32, TransportError>;

    /// Allocate a buffer of `bytes`, in the given group.
    fn alloc(
        &mut self,
        bytes: u64,
        flags: BufferFlags,
        group: i32,
    ) -> Result<Self::Buffer, TransportError>;

    /// Write into a buffer.
    fn write(bo: &Self::Buffer, data: &[u8], offset: u64) -> Result<(), TransportError>;

    /// The buffer's own size, in bytes.
    ///
    /// The manifest says how big a buffer should be and the board says how
    /// big it is; when they differ the board is right, and the check that
    /// catches it is the one that reads the real size.
    fn size(bo: &Self::Buffer) -> u64;

    /// Read out of a buffer.
    ///
    /// Only meaningful after a `sync_from_device`: a buffer that the device
    /// has written and the host has not read back holds whatever the host
    /// last put there.
    fn read(bo: &Self::Buffer, data: &mut [u8], offset: u64) -> Result<(), TransportError>;

    /// Make a buffer's contents visible to the device.
    fn sync_to_device(bo: &Self::Buffer, bytes: u64, offset: u64) -> Result<(), TransportError>;

    /// Make the device's writes to a buffer visible to the host.
    fn sync_from_device(bo: &Self::Buffer, bytes: u64, offset: u64) -> Result<(), TransportError>;

    /// Start a run: the instruction stream, its length in 32-bit words, and
    /// the design's data buffers.
    fn start(
        &mut self,
        instr: &Self::Buffer,
        words: u32,
        args: &[&Self::Buffer],
    ) -> Result<Run, TransportError>;

    /// Wait for a run to finish.
    ///
    /// A run that did not COMPLETE is an error and not a number: the design
    /// may have written half an output, and half an output read back as a
    /// plausible tensor is exactly the failure this exists to catch.
    fn wait(run: &Run) -> Result<(), TransportError>;
}

/// Whether this build has a transport at all.
///
/// A build without XRT installed still builds the runtime for the CPU and
/// Vulkan backends, and the array path then says so instead of failing to
/// link.
pub const fn transport_available() -> bool {
    cfg!(all(feature = "xrt", xrt_shim)) || cfg!(feature = "direct")
}

/// The transport this build speaks.
///
/// The same rule for every dependent: the direct transport, when the build
/// has it, is the one that speaks; XRT is what a build without it uses. A
/// build can have both -- the direct transport does not need XRT to be
/// absent, only to be unused -- and the direct one wins, because it is the
/// one the build was asked for.
///
/// It lives HERE and not in each dependent so that the binary, the tests
/// and any future tool all speak through the same transport: a test that
/// named XRT itself would quietly keep testing XRT in a direct build.
#[cfg(feature = "direct")]
pub type SelectedTransport = crate::direct::DirectTransport;

/// The transport this build speaks, when it has no direct transport.
#[cfg(all(feature = "xrt", not(feature = "direct")))]
pub type SelectedTransport = crate::xrt::XrtTransport;

/// Open the board, through whichever transport this build speaks.
#[cfg(any(feature = "xrt", feature = "direct"))]
pub fn open_selected(index: u32) -> Result<SelectedTransport, TransportError> {
    SelectedTransport::open(index)
}
