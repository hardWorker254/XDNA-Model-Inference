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
//! # THE SEQUENCE, AND WHY IT IS THIS SEQUENCE
//!
//! The whole path is eleven calls, and every one of them is here:
//!
//! ```text
//! CREATE_BO      the device heap: where the driver's own memory comes from
//! GET_INFO       what the AIE is: its columns, its rows, its version
//! CREATE_HWCTX   a context, and the syncobj its jobs' timeline lives on
//! CREATE_BO      the PDI: the program, in device memory
//! CONFIG_HWCTX   the CU's configuration: the PDI's handle, and which CU
//! CREATE_BO      the command buffer: where the packet is written
//!
//!   ... and then, per dispatch ...
//!
//! EXEC_CMD       the packet, submitted
//! SYNCOBJ_TIMELINE_WAIT   the point on the timeline, waited for
//! ```
//!
//! and the teardown undoes them: the buffers are closed, the syncobj is
//! destroyed, the context is destroyed, and the heap is closed last,
//! because the device memory the other buffers live in is carved out of it.
//!
//! # THE PACKET
//!
//! The command packet is the ERT's: a header word, a CU mask, and the CU's
//! register map. The header says what the packet is --
//!
//! ```text
//!   0  u32   the state (NEW), the payload's length in words,
//!            the opcode (START_CU) and the type (CU)
//!   4  u32   the CU mask: the first CU
//!   8  ...   the CU's register map
//! ```
//!
//! -- and the register map is the one the xclbin's metadata describes: the
//! `opcode`, the instruction stream's address and length, and the addresses
//! of the buffers the kernel was called with. Where each of those sits is
//! read out of the metadata's XML, because the design says so and not this
//! file.
//!
//! # TWO ADDRESSES, AND THE DIFFERENCE MATTERS
//!
//! The instruction stream is DEVICE memory: the driver gives back a device
//! address, and that is what the packet carries. The data buffers are HOST
//! memory: the driver gives back a mapping, and the mapping's address is
//! what the packet carries. Mixing them up is a dispatch that runs the
//! wrong program over the wrong bytes, and nothing about it looks wrong.
//!
//! # THE WAIT
//!
//! A dispatch is done when the point on the context's timeline is
//! signalled, which is when the job is over -- and the driver says so
//! twice: the timeline says it, and the packet's own header word carries
//! the state. Both are read, because a run that did not COMPLETE is an
//! error and not a number: the design may have written half an output, and
//! half an output read back is a plausible tensor.

use std::ffi::c_void;
use std::os::fd::{AsRawFd, RawFd};
use std::sync::Arc;

use crate::device::Generation;
use crate::transport::{BufferFlags, Run, Transport, TransportError};
use crate::xclbin::{Kernel, Xclbin};

/// The DRM's command base: the driver's own ioctl numbers start here.
const DRM_COMMAND_BASE: u64 = 0x40;

/// The DRM's ioctl type byte: `d`.
const DRM_IOCTL_TYPE: u64 = b'd' as u64;

const IOC_WRITE: u64 = 1;
const IOC_READ: u64 = 2;

/// The ioctl number, as the kernel's `_IOC` macro computes it.
const fn ioc(dir: u64, nr: u64, size: u64) -> u64 {
    (dir << 30) | (DRM_IOCTL_TYPE << 8) | nr | (size << 16)
}

/// A read-write ioctl.
const fn iowr(nr: u64, size: u64) -> u64 {
    ioc(IOC_READ | IOC_WRITE, nr, size)
}

/// A write-only ioctl.
const fn iow(nr: u64, size: u64) -> u64 {
    ioc(IOC_WRITE, nr, size)
}

/// Create a hardware context.
const IOCTL_CREATE_HWCTX: u64 = iowr(DRM_COMMAND_BASE, size_of::<CreateHwctx>() as u64);
/// Destroy a hardware context.
const IOCTL_DESTROY_HWCTX: u64 =
    iowr(DRM_COMMAND_BASE + 1, size_of::<DestroyHwctx>() as u64);
/// Configure a hardware context.
const IOCTL_CONFIG_HWCTX: u64 = iowr(DRM_COMMAND_BASE + 2, size_of::<ConfigHwctx>() as u64);
/// Create a buffer object.
const IOCTL_CREATE_BO: u64 = iowr(DRM_COMMAND_BASE + 3, size_of::<CreateBo>() as u64);
/// Ask about a buffer object.
const IOCTL_GET_BO_INFO: u64 = iowr(DRM_COMMAND_BASE + 4, size_of::<GetBoInfo>() as u64);
/// Make a buffer's contents visible to the other side.
const IOCTL_SYNC_BO: u64 = iowr(DRM_COMMAND_BASE + 5, size_of::<SyncBo>() as u64);
/// Submit a command.
const IOCTL_EXEC_CMD: u64 = iowr(DRM_COMMAND_BASE + 6, size_of::<ExecCmd>() as u64);
/// Ask the device something.
const IOCTL_GET_INFO: u64 = iowr(DRM_COMMAND_BASE + 7, size_of::<GetInfo>() as u64);
/// Close a buffer object's handle.
const IOCTL_GEM_CLOSE: u64 = iow(0x09, size_of::<GemClose>() as u64);
/// Destroy a syncobj.
const IOCTL_SYNCOBJ_DESTROY: u64 = iowr(0xC0, size_of::<SyncobjDestroy>() as u64);
/// Wait for a point on a syncobj's timeline.
const IOCTL_SYNCOBJ_TIMELINE_WAIT: u64 =
    iowr(0xCA, size_of::<SyncobjTimelineWait>() as u64);

/// The buffer type of a host-visible buffer.
const BO_TYPE_SHARE: u32 = 1;
/// The buffer type of the device heap.
const BO_TYPE_DEV_HEAP: u32 = 2;
/// The buffer type of device memory.
const BO_TYPE_DEV: u32 = 3;
/// The buffer type of a command buffer.
const BO_TYPE_CMD: u32 = 4;

/// How big the device heap is.
///
/// The heap is the pool the driver's own allocations come from -- the
/// command buffers the context keeps, and the device memory the user asks
/// for. Sixty-four megabytes is what the C++ runtime's transport used, and
/// it is more than the array path needs.
const DEVICE_HEAP_BYTES: u64 = 64 * 1024 * 1024;

/// How big the command buffer is.
///
/// The driver's own command buffers are four kilobytes each, and the ERT's
/// packet for one CU is a few dozen words.
const COMMAND_BUFFER_BYTES: u64 = 4096;

/// The state an ERT packet is in when it is submitted.
const ERT_STATE_NEW: u32 = 1;
/// The state an ERT packet is in when it has run.
const ERT_STATE_COMPLETED: u32 = 4;

/// The ERT opcode of a packet that starts a CU.
const ERT_START_CU: u32 = 0;
/// The ERT type of a packet that is a compute unit's.
const ERT_CU: u32 = 3;

/// The CU's opcode for "run the DPU with this instruction stream".
///
/// The AIE2 compute unit takes an opcode as the first word of its register
/// map, and 3 is the one that starts the DPU. XRT's caller passes the same
/// number, and the C++ runtime went through XRT, so this is the number the
/// board has always been given.
const CU_OPCODE_RUN: u64 = 3;

/// The AIE version parameter of `GET_INFO`.
const QUERY_AIE_VERSION: u32 = 2;
/// The AIE metadata parameter of `GET_INFO`.
const QUERY_AIE_METADATA: u32 = 1;

/// `struct amdxdna_drm_create_hwctx`.
#[repr(C)]
struct CreateHwctx {
    ext: u64,
    ext_flags: u64,
    qos_p: u64,
    umq_bo: u32,
    log_buf_bo: u32,
    max_opc: u32,
    num_tiles: u32,
    mem_size: u32,
    umq_doorbell: u32,
    handle: u32,
    syncobj_handle: u32,
}

/// `struct amdxdna_drm_destroy_hwctx`.
#[repr(C)]
struct DestroyHwctx {
    handle: u32,
    pad: u32,
}

/// `struct amdxdna_drm_config_hwctx`.
#[repr(C)]
struct ConfigHwctx {
    handle: u32,
    param_type: u32,
    param_val: u64,
    param_val_size: u32,
    pad: u32,
}

/// `struct amdxdna_cu_config`.
#[repr(C)]
#[derive(Default)]
struct CuConfig {
    cu_bo: u32,
    cu_func: u8,
    pad: [u8; 3],
}

/// `struct amdxdna_hwctx_param_config_cu`.
#[repr(C)]
struct HwctxParamConfigCu {
    num_cus: u16,
    pad: [u8; 3],
    cu_configs: [CuConfig; 1],
}

/// `struct amdxdna_drm_create_bo`.
#[repr(C)]
struct CreateBo {
    flags: u64,
    vaddr: u64,
    size: u64,
    bo_type: u32,
    handle: u32,
}

/// `struct amdxdna_drm_get_bo_info`.
#[repr(C)]
struct GetBoInfo {
    ext: u64,
    ext_flags: u64,
    handle: u32,
    pad: u32,
    map_offset: u64,
    vaddr: u64,
    xdna_addr: u64,
}

/// `struct amdxdna_drm_sync_bo`.
#[repr(C)]
struct SyncBo {
    handle: u32,
    direction: u32,
    offset: u64,
    size: u64,
}

/// `struct amdxdna_drm_exec_cmd`.
#[repr(C)]
struct ExecCmd {
    ext: u64,
    ext_flags: u64,
    hwctx: u32,
    cmd_type: u32,
    cmd_handles: u64,
    args: u64,
    cmd_count: u32,
    arg_count: u32,
    seq: u64,
}

/// `struct amdxdna_drm_get_info`.
#[repr(C)]
struct GetInfo {
    param: u32,
    buffer_size: u32,
    buffer: u64,
}

/// `struct drm_gem_close`.
#[repr(C)]
struct GemClose {
    handle: u64,
}

/// `struct drm_syncobj_destroy`.
#[repr(C)]
struct SyncobjDestroy {
    handle: u64,
}

/// `struct drm_syncobj_timeline_wait`.
#[repr(C)]
struct SyncobjTimelineWait {
    handles: u64,
    points: u64,
    timeout_nsec: i64,
    count_handles: u32,
    flags: u32,
    first_signaled: u32,
    pad: u32,
    deadline_nsec: i64,
}

/// `struct amdxdna_drm_query_aie_version`.
#[repr(C)]
struct AieVersion {
    major: u32,
    minor: u32,
}

/// `struct amdxdna_qos_info`.
///
/// The QoS hints are the user's, and this runtime has none to give: the
/// numbers are left at zero, and the one field with a meaning of its own
/// -- the column the partition starts at -- is left at the value that says
/// "the driver decides".
#[repr(C)]
struct QosInfo {
    gops: u32,
    fps: u32,
    dma_bandwidth: u32,
    latency: u32,
    frame_exec_time: u32,
    priority: u32,
    user_start_col: u32,
    reserved: u32,
}

/// The value of `user_start_col` that leaves the placement to the driver.
const USER_START_COL_NOT_REQUESTED: u32 = 0xff;

/// `struct amdxdna_drm_query_aie_tile_metadata`.
#[repr(C)]
#[derive(Default)]
struct AieTileMetadata {
    row_count: u16,
    row_start: u16,
    dma_channel_count: u16,
    lock_count: u16,
    event_reg_count: u16,
    pad: [u16; 3],
}

/// `struct amdxdna_drm_query_aie_metadata`.
#[repr(C)]
struct AieMetadata {
    col_size: u32,
    cols: u16,
    rows: u16,
    version: AieVersion,
    core: AieTileMetadata,
    mem: AieTileMetadata,
    shim: AieTileMetadata,
}

/// The board, and the file descriptor it is reached through.
///
/// The descriptor is shared: the buffers hold a reference to the board, so
/// that a buffer freed after the transport is still freed against a
/// descriptor that is open.
struct Board {
    file: std::fs::File,
}

impl Board {
    fn fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}

/// The device heap: the pool the driver's own memory comes from.
///
/// The buffers carved out of it -- the PDI, the instruction streams -- hold
/// a reference to the heap, so the heap outlives them. A heap closed while
/// its memory is still in use is a heap the driver cannot account for.
struct Heap {
    board: Arc<Board>,
    handle: u32,
    /// The mapping, kept for as long as the heap: the device memory carved
    /// out of the heap is addressed by this process's address for it.
    _map: Mapping,
}

impl Drop for Heap {
    fn drop(&mut self) {
        close_bo(&self.board, self.handle);
    }
}

/// A mapping of a buffer into this process's address space.
struct Mapping {
    ptr: *mut u8,
    len: usize,
}

impl Mapping {
    /// A mapping, or the driver's refusal for one.
    fn from_ptr(ptr: *mut c_void, len: usize, what: &str) -> Result<Self, TransportError> {
        let address = ptr as isize;
        if address < 0 && address > -4096 {
            let errno = -address as i32;
            return Err(TransportError::new(format!(
                "cannot map {what}: {}",
                errno_text(errno)
            )));
        }
        if ptr.is_null() {
            return Err(TransportError::new(format!(
                "cannot map {what}: the driver refused"
            )));
        }
                Ok(Self {
            ptr: ptr as *mut u8,
            len,
        })
    }

    /// Map a buffer at an address that is a multiple of `align`.
    ///
    /// The heap has to be placed this way: the driver hands the firmware
    /// the heap by its address in this process, in chunks of the device
    /// memory size (64 MiB), and the firmware refuses a heap that does not
    /// start on a chunk boundary. The driver checks the heap's *size* is a
    /// multiple of the chunk but not its address, so an address the kernel
    /// picks freely -- which is aligned to 2 MiB at best -- is refused
    /// with EINVAL, as a host buffer the firmware will not map.
    fn new_aligned(
        fd: RawFd,
        offset: u64,
        len: usize,
        align: usize,
    ) -> Result<Self, TransportError> {
        const PROT_NONE: i32 = 0;
        const PROT_READ: i32 = 1;
        const PROT_WRITE: i32 = 2;
        const MAP_PRIVATE: i32 = 2;
        const MAP_FIXED: i32 = 0x10;
        const MAP_ANONYMOUS: i32 = 0x20;
        const MAP_SHARED: i32 = 1;
        const MAP_LOCKED: i32 = 0x2000;
        const MAP_FAILED: isize = -1;

        // A window to find an aligned address in, then the buffer placed
        // there. Nothing is ever stored in the window itself.
        let window = unsafe {
            libc_mmap(
                std::ptr::null_mut(),
                len + align,
                PROT_NONE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if window as isize == MAP_FAILED {
            return Err(TransportError::new(
                "cannot reserve an address for the heap",
            ));
        }
        let base = ((window as usize) + align - 1) & !(align - 1);
        unsafe { libc_munmap(window, len + align) };

        let ptr = unsafe {
            libc_mmap(
                base as *mut std::ffi::c_void,
                len,
                PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_FIXED | MAP_LOCKED,
                fd,
                offset as i64,
            )
        };
        Self::from_ptr(ptr, len, "the heap")
    }

    fn new(fd: RawFd, offset: u64, len: usize) -> Result<Self, TransportError> {
        // PROT_READ | PROT_WRITE, MAP_SHARED: the buffer is memory that
        // both the host and the device see.
        const PROT_READ: i32 = 1;
        const PROT_WRITE: i32 = 2;
        const MAP_SHARED: i32 = 1;
        let ptr = unsafe {
            libc_mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                offset as i64,
            )
        };
        Self::from_ptr(ptr, len, "the buffer")
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe { libc_munmap(self.ptr as *mut c_void, self.len) };
    }
}

/// A buffer on the device.
///
/// The two kinds are different in what they are and in what they are
/// addressed by: device memory is reached through a device address, and
/// host memory through the address this process sees it at.
pub struct DirectBuffer {
    board: Arc<Board>,
    /// The heap the device memory was carved out of, when it is device
    /// memory.
    _heap: Option<Arc<Heap>>,
    /// The driver's handle for the buffer.
    handle: u32,
    /// How big the buffer is.
    size: u64,
    /// Where this process sees the buffer.
    host: *mut u8,
    /// The device's address for the buffer, when it is device memory.
    device_addr: u64,
    /// The mapping this process made, when it made one.
    map: Option<Mapping>,
}

// SAFETY: the handle is owned by this struct and is used by the thread that
// owns it. The runtime dispatches one queue per thread, so a buffer is never
// touched by two threads at once.
unsafe impl Send for DirectBuffer {}

impl Drop for DirectBuffer {
    fn drop(&mut self) {
        // The mapping goes first: the driver's handle is the last thing to
        // be let go of.
        self.map = None;
        close_bo(&self.board, self.handle);
    }
}

/// One run of a kernel, on this transport.
struct DirectRun {
    /// The board the run was submitted to.
    board: Arc<Board>,
    /// The context's syncobj: the timeline the run's point is on.
    syncobj: u32,
    /// The run's point on that timeline.
    point: u64,
    /// The command buffer the packet was written into.
    ///
    /// The driver writes the run's state into the packet's header word, so
    /// the run holds the buffer -- and the mapping it lives in -- for as long
    /// as the state can be read. A run dropped without being waited for holds
    /// it until it is freed; a run that outlived the buffer would be reading
    /// a freed page for a number.
    command: Arc<Command>,
}

/// Free a run.
unsafe extern "C" fn free_run(ptr: *mut c_void) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr as *mut DirectRun)) };
    }
}

/// The transport over the driver's ioctls.
pub struct DirectTransport {
    /// The board.
    board: Arc<Board>,
    /// The device heap.
    heap: Option<Arc<Heap>>,
    /// The hardware context, when one is open.
    context: Option<Context>,
    /// The design's kernel, when a context is open.
    kernel: Option<Kernel>,
    /// The PDI, when a context is open.
    ///
    /// The CU's program stays in the memory the context was configured with:
    /// the driver hands the heap's space back out the moment the buffer is
    /// closed, and the next buffer carved out of the heap would land on the
    /// program. It is held for as long as the context lives.
    pdi: Option<DirectBuffer>,
    /// The command buffer, when a context is open.
    command: Option<Arc<Command>>,
}

/// A hardware context, and the syncobj its jobs' timeline lives on.
struct Context {
    /// The board the context is on.
    board: Arc<Board>,
    /// The driver's handle for the context.
    handle: u32,
    /// The syncobj the context's jobs are timed against.
    syncobj: u32,
}

impl Drop for Context {
    fn drop(&mut self) {
        // The syncobj first, then the context: the context's jobs are
        // points on that timeline, and a point on a timeline that is gone
        // is a point nobody can wait for.
        destroy_syncobj(&self.board, self.syncobj);
        let mut args = DestroyHwctx {
            handle: self.handle,
            pad: 0,
        };
        let _ = ioctl(
            &self.board,
            IOCTL_DESTROY_HWCTX,
            &mut args as *mut DestroyHwctx as *mut c_void,
        );
    }
}

/// The command buffer: where the packet is written.
struct Command {
    board: Arc<Board>,
    /// The driver's handle for the buffer.
    handle: u32,
    /// Where this process sees the buffer.
    map: Mapping,
}

impl Drop for Command {
    fn drop(&mut self) {
        close_bo(&self.board, self.handle);
    }
}

// SAFETY: the handles are owned by this struct and are used by the thread
// that owns it.
unsafe impl Send for DirectTransport {}

impl Drop for DirectTransport {
    fn drop(&mut self) {
        // The order matters: the context goes before the command buffer it
        // was submitting to, and the heap goes after everything that was
        // carved out of it.
        self.context = None;
        self.pdi = None;
        self.command = None;
        self.heap = None;
    }
}

impl Transport for DirectTransport {
    type Buffer = DirectBuffer;

    fn open(index: u32) -> Result<Self, TransportError> {
        let mut last = std::io::Error::other("no device node");
        for path in device_nodes(index) {
            // Read and write: the buffers are mapped shared, with write
            // access, and a mapping of a file opened for reading alone is
            // refused with EACCES.
            match std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
            {
                Ok(file) => {
                    let board = Arc::new(Board { file });
                    return Ok(Self {
                        board,
                        heap: None,
                        context: None,
                        kernel: None,
                        pdi: None,
                        command: None,
                    });
                }
                Err(error) => last = error,
            }
        }
        Err(TransportError::new(format!(
            "cannot open the NPU device: {last}"
        )))
    }

    fn open_context(&mut self, xclbin: &[u8]) -> Result<(), TransportError> {
        let xclbin = Xclbin::parse(xclbin).map_err(TransportError::new)?;
        let kernel = xclbin.kernel().map_err(TransportError::new)?;
        let partition = xclbin.partition().map_err(TransportError::new)?;
        let pdi = xclbin.pdi().map_err(TransportError::new)?;

        // The device heap.
        //
        // The driver's own allocations -- and the device memory the user
        // asks for -- come out of it, and the context will not be created
        // without one.
        if self.heap.is_none() {
            let handle = self.create_bo(DEVICE_HEAP_BYTES, BO_TYPE_DEV_HEAP)?;
            // The heap is mapped, and mapped on a chunk boundary: the
            // driver gives the firmware the heap's address in this
            // process, and the firmware takes only an address that is a
            // multiple of the device memory size.
            let mapped = self.bo_info(handle).and_then(|info| {
                Mapping::new_aligned(
                    self.board.fd(),
                    info.map_offset,
                    DEVICE_HEAP_BYTES as usize,
                    DEVICE_HEAP_BYTES as usize,
                )
            });
            let map = match mapped {
                Ok(map) => map,
                Err(error) => {
                    close_bo(&self.board, handle);
                    return Err(error);
                }
            };
            self.heap = Some(Arc::new(Heap {
                board: self.board.clone(),
                handle,
                _map: map,
            }));
        }

        // THE AIE'S OWN SHAPE.
        //
        // The context's tile count is in the device's rows: the partition
        // says how many columns it occupies, and the device says how many
        // rows a column has. The driver divides one by the other, and gets
        // the number of columns the design asked for.
        let metadata = self.aie_metadata()?;
        if metadata.core.row_count == 0 {
            return Err(TransportError::new(
                "the device reports no compute rows, and a context cannot be made without them",
            ));
        }
        let num_tiles = partition
            .column_width
            .checked_mul(metadata.core.row_count as u32)
            .ok_or_else(|| {
                TransportError::new("the design's column count overflows the tile count")
            })?;

        // THE CONTEXT.
        //
        // The QoS hints are a pointer, and the driver reads them whatever
        // they say: a null pointer is a fault, not a "no hints".
        let qos = QosInfo {
            gops: 0,
            fps: 0,
            dma_bandwidth: 0,
            latency: 0,
            frame_exec_time: 0,
            priority: 0,
            user_start_col: USER_START_COL_NOT_REQUESTED,
            reserved: 0,
        };
        let mut args = CreateHwctx {
            ext: 0,
            ext_flags: 0,
            qos_p: &qos as *const QosInfo as u64,
            umq_bo: 0,
            log_buf_bo: 0,
            max_opc: partition.operations_per_cycle,
            num_tiles,
            mem_size: 0,
            umq_doorbell: 0,
            handle: 0,
            syncobj_handle: 0,
        };
        ioctl(
            &self.board,
            IOCTL_CREATE_HWCTX,
            &mut args as *mut CreateHwctx as *mut c_void,
        )?;
        let context = Context {
            board: self.board.clone(),
            handle: args.handle,
            syncobj: args.syncobj_handle,
        };

        // THE PDI: the program, in device memory.
        let pdi_buffer = self.alloc_device(pdi.len() as u64)?;
        <DirectTransport as Transport>::write(&pdi_buffer, pdi, 0)?;

        // THE CU'S CONFIGURATION: the PDI is the CU's program.
        let mut cu = HwctxParamConfigCu {
            num_cus: 1,
            pad: [0; 3],
            cu_configs: [CuConfig {
                cu_bo: pdi_buffer.handle,
                cu_func: 0,
                pad: [0; 3],
            }],
        };
        let mut config = ConfigHwctx {
            handle: context.handle,
            param_type: 0,
            param_val: &mut cu as *mut HwctxParamConfigCu as u64,
            param_val_size: size_of::<HwctxParamConfigCu>() as u32,
            pad: 0,
        };
        ioctl(
            &self.board,
            IOCTL_CONFIG_HWCTX,
            &mut config as *mut ConfigHwctx as *mut c_void,
        )?;

        // THE COMMAND BUFFER: where the packet goes.
        let command = self.create_command()?;

        self.context = Some(context);
        self.kernel = Some(kernel);
        self.pdi = Some(pdi_buffer);
        self.command = Some(Arc::new(command));
        Ok(())
    }

    fn open_kernel(&mut self, name: &str) -> Result<(), TransportError> {
        let kernel = self
            .kernel
            .as_ref()
            .ok_or_else(|| TransportError::new("no context is open"))?;
        if kernel.name != name {
            return Err(TransportError::new(format!(
                "the design asks for the kernel `{name}`, and this xclbin's kernel is `{}`",
                kernel.name
            )));
        }
        Ok(())
    }

    fn group_id(&self, argno: i32) -> Result<i32, TransportError> {
        // The direct transport has no memory banks: the driver's buffer
        // TYPES decide where a buffer lands -- device memory for an
        // instruction stream, host memory for the data -- and the flags
        // say which. The number is still answered, because the design
        // layer asks every transport the same question, and the answer
        // here is the design's own numbering.
        if (0..=7).contains(&argno) {
            Ok(argno)
        } else {
            Err(TransportError::new(format!(
                "the design's kernel has no argument {argno}"
            )))
        }
    }

    fn alloc(
        &mut self,
        bytes: u64,
        flags: BufferFlags,
        _group: i32,
    ) -> Result<Self::Buffer, TransportError> {
        match flags {
            // THE INSTRUCTION STREAM: device memory, read by the device's
            // instruction fetch.
            BufferFlags::Cacheable => self.alloc_device(bytes),
            // THE DATA: host memory, written by the host and read by the
            // device.
            BufferFlags::HostOnly => self.alloc_host(bytes),
        }
    }

    fn write(bo: &Self::Buffer, data: &[u8], offset: u64) -> Result<(), TransportError> {
        let at = offset as usize;
        if at + data.len() > bo.size as usize {
            return Err(TransportError::new(format!(
                "{} bytes at offset {offset}, and the buffer is {} bytes",
                data.len(),
                bo.size
            )));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), bo.host.add(at), data.len());
        }
        Ok(())
    }

    fn size(bo: &Self::Buffer) -> u64 {
        bo.size
    }

    fn read(bo: &Self::Buffer, data: &mut [u8], offset: u64) -> Result<(), TransportError> {
        let at = offset as usize;
        if at + data.len() > bo.size as usize {
            return Err(TransportError::new(format!(
                "{} bytes at offset {offset}, and the buffer is {} bytes",
                data.len(),
                bo.size
            )));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(bo.host.add(at), data.as_mut_ptr(), data.len());
        }
        Ok(())
    }

    fn sync_to_device(
        bo: &Self::Buffer,
        bytes: u64,
        offset: u64,
    ) -> Result<(), TransportError> {
        sync_bo(&bo.board, bo.handle, offset, bytes, SYNC_DIRECT_TO_DEVICE)
    }

    fn sync_from_device(
        bo: &Self::Buffer,
        bytes: u64,
        offset: u64,
    ) -> Result<(), TransportError> {
        sync_bo(&bo.board, bo.handle, offset, bytes, SYNC_DIRECT_FROM_DEVICE)
    }

    fn start(
        &mut self,
        instr: &Self::Buffer,
        words: u32,
        args: &[&Self::Buffer],
    ) -> Result<Run, TransportError> {
        let kernel = self
            .kernel
            .as_ref()
            .ok_or_else(|| TransportError::new("no context is open"))?;
        let command = self
            .command
            .clone()
            .ok_or_else(|| TransportError::new("no command buffer is open"))?;
        let context = self
            .context
            .as_ref()
            .ok_or_else(|| TransportError::new("no context is open"))?;

        // THE PACKET.
        let packet = build_packet(kernel, instr, words, args)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                packet.as_ptr(),
                command.map.ptr,
                packet.len().min(command.map.len),
            );
        }

        // THE SUBMISSION.
        let mut exec = ExecCmd {
            ext: 0,
            ext_flags: 0,
            hwctx: context.handle,
            cmd_type: 0,
            cmd_handles: command.handle as u64,
            args: 0,
            cmd_count: 1,
            arg_count: 0,
            seq: 0,
        };
        ioctl(
            &self.board,
            IOCTL_EXEC_CMD,
            &mut exec as *mut ExecCmd as *mut c_void,
        )?;

        let run = Box::new(DirectRun {
            board: self.board.clone(),
            syncobj: context.syncobj,
            point: exec.seq,
            command,
        });
        Ok(Run::new(Box::into_raw(run) as *mut c_void, free_run))
    }

    fn wait(run: &Run) -> Result<(), TransportError> {
        let run = unsafe { &*(run.handle() as *const DirectRun) };

        // THE TIMELINE: the point the job was given, waited for.
        let mut handles = [run.syncobj as u64];
        let mut points = [run.point];
        let mut wait = SyncobjTimelineWait {
            handles: handles.as_mut_ptr() as u64,
            points: points.as_mut_ptr() as u64,
            timeout_nsec: i64::MAX,
            count_handles: 1,
            flags: 0,
            first_signaled: 0,
            pad: 0,
            deadline_nsec: 0,
        };
        ioctl(
            &run.board,
            IOCTL_SYNCOBJ_TIMELINE_WAIT,
            &mut wait as *mut SyncobjTimelineWait as *mut c_void,
        )?;

        // AND THE STATE: the driver writes it into the packet's own header
        // word, and a run that did not complete is an error and not a
        // number.
        let header = run.command.map.ptr as *mut u32;
        let state = unsafe { std::ptr::read_volatile(header) } & 0xf;
        if state != ERT_STATE_COMPLETED {
            return Err(TransportError::new(format!(
                "the kernel did not complete: the driver's state is {state}"
            )));
        }
        Ok(())
    }
}

impl DirectTransport {
    /// Create a buffer object and return its handle.
    fn create_bo(&self, bytes: u64, bo_type: u32) -> Result<u32, TransportError> {
        let mut args = CreateBo {
            flags: 0,
            vaddr: 0,
            size: bytes,
            bo_type,
            handle: 0,
        };
        ioctl(
            &self.board,
            IOCTL_CREATE_BO,
            &mut args as *mut CreateBo as *mut c_void,
        )?;
        Ok(args.handle)
    }

    /// Ask the driver about a buffer object, closing it if it cannot say.
    fn bo_info_or_close(&self, handle: u32) -> Result<GetBoInfo, TransportError> {
        let info = self.bo_info(handle);
        if info.is_err() {
            close_bo(&self.board, handle);
        }
        info
    }

    /// Ask the driver about a buffer object.
    fn bo_info(&self, handle: u32) -> Result<GetBoInfo, TransportError> {
        let mut args = GetBoInfo {
            ext: 0,
            ext_flags: 0,
            handle,
            pad: 0,
            map_offset: 0,
            vaddr: 0,
            xdna_addr: 0,
        };
        ioctl(
            &self.board,
            IOCTL_GET_BO_INFO,
            &mut args as *mut GetBoInfo as *mut c_void,
        )?;
        Ok(args)
    }

    /// A buffer in device memory.
    ///
    /// The driver maps the memory into this process's address space and
    /// gives back both that address and the device's own. The host writes
    /// through the first and the device reads at the second.
    fn alloc_device(&self, bytes: u64) -> Result<DirectBuffer, TransportError> {
        let heap = self
            .heap
            .clone()
            .ok_or_else(|| TransportError::new("the device heap is not open"))?;
        let handle = self.create_bo(bytes, BO_TYPE_DEV)?;
        let info = self.bo_info_or_close(handle)?;
        if info.vaddr == u64::MAX {
            close_bo(&self.board, handle);
            return Err(TransportError::new(
                "the driver gave no address for the device memory",
            ));
        }
        Ok(DirectBuffer {
            board: self.board.clone(),
            _heap: Some(heap),
            handle,
            size: bytes,
            host: info.vaddr as *mut u8,
            device_addr: info.xdna_addr,
            map: None,
        })
    }

    /// A buffer in host memory.
    ///
    /// The driver gives back a mapping's offset and not the mapping
    /// itself: this process maps it, and the address it gets is the
    /// address the packet carries.
    fn alloc_host(&self, bytes: u64) -> Result<DirectBuffer, TransportError> {
        let handle = self.create_bo(bytes, BO_TYPE_SHARE)?;
        let info = self.bo_info_or_close(handle)?;
        if info.map_offset == u64::MAX {
            close_bo(&self.board, handle);
            return Err(TransportError::new(
                "the driver gave no mapping for the host memory",
            ));
        }
        let map = match Mapping::new(self.board.fd(), info.map_offset, bytes as usize) {
            Ok(map) => map,
            Err(error) => {
                close_bo(&self.board, handle);
                return Err(error);
            }
        };
        let ptr = map.ptr;
        Ok(DirectBuffer {
            board: self.board.clone(),
            _heap: None,
            handle,
            size: bytes,
            host: ptr,
            device_addr: 0,
            map: Some(map),
        })
    }

    /// The command buffer, mapped.
    fn create_command(&self) -> Result<Command, TransportError> {
        let handle = self.create_bo(COMMAND_BUFFER_BYTES, BO_TYPE_CMD)?;
        let info = self.bo_info_or_close(handle)?;
        let map = match Mapping::new(
            self.board.fd(),
            info.map_offset,
            COMMAND_BUFFER_BYTES as usize,
        ) {
            Ok(map) => map,
            Err(error) => {
                close_bo(&self.board, handle);
                return Err(error);
            }
        };
        Ok(Command {
            board: self.board.clone(),
            handle,
            map,
        })
    }

    /// The device's own description of its array.
    fn aie_metadata(&self) -> Result<AieMetadata, TransportError> {
        let mut metadata = AieMetadata {
            col_size: 0,
            cols: 0,
            rows: 0,
            version: AieVersion { major: 0, minor: 0 },
            core: AieTileMetadata::default(),
            mem: AieTileMetadata::default(),
            shim: AieTileMetadata::default(),
        };
        let mut args = GetInfo {
            param: QUERY_AIE_METADATA,
            buffer_size: size_of::<AieMetadata>() as u32,
            buffer: &mut metadata as *mut AieMetadata as u64,
        };
        ioctl(
            &self.board,
            IOCTL_GET_INFO,
            &mut args as *mut GetInfo as *mut c_void,
        )?;
        Ok(metadata)
    }
}

/// The device nodes to try, in order.
///
/// The `amdxdna` driver presents `/dev/accel/accelN`; the older name is
/// what a machine with an older driver has, and the zero index is the
/// board this runtime has ever used.
fn device_nodes(index: u32) -> Vec<String> {
    if index == 0 {
        vec![
            crate::device::ACCEL_NODE.to_string(),
            crate::device::LEGACY_NODE.to_string(),
        ]
    } else {
        vec![format!("/dev/accel/accel{index}")]
    }
}

/// The direction of a buffer's synchronisation.
const SYNC_DIRECT_TO_DEVICE: u32 = 0;
/// The direction of a buffer's synchronisation.
const SYNC_DIRECT_FROM_DEVICE: u32 = 1;

/// Build the command packet.
///
/// The header says what the packet is, the CU mask says which CU runs it,
/// and the register map says what it runs on. The map's layout is the
/// design's own: the offsets come from the xclbin's metadata.
fn build_packet(
    kernel: &Kernel,
    instr: &DirectBuffer,
    words: u32,
    args: &[&DirectBuffer],
) -> Result<Vec<u8>, TransportError> {
    let map = kernel.map_size() as usize;
    let payload_words = 1 + map.div_ceil(4);
    let mut packet = vec![0u8; 4 + payload_words * 4];

    // THE HEADER: the state, the payload's length in words, the opcode and
    // the type.
    let header = ERT_STATE_NEW
        | ((payload_words as u32) << 12)
        | (ERT_START_CU << 23)
        | (ERT_CU << 28);
    packet[0..4].copy_from_slice(&header.to_le_bytes());

    // THE CU MASK: the first CU.
    packet[4..8].copy_from_slice(&1u32.to_le_bytes());

    let mut put = |name: &str, value: u64| -> Result<(), TransportError> {
        let arg = kernel.arg(name).ok_or_else(|| {
            TransportError::new(format!("the design's kernel has no `{name}` argument"))
        })?;
        let at = 8 + arg.offset as usize;
        let bytes = value.to_le_bytes();
        let width = (arg.size as usize).min(8);
        if at + width > packet.len() {
            return Err(TransportError::new(format!(
                "the design's kernel puts `{name}` outside the packet"
            )));
        }
        packet[at..at + width].copy_from_slice(&bytes[..width]);
        Ok(())
    };

    put("opcode", CU_OPCODE_RUN)?;
    put("instr", instr.device_addr)?;
    put("ninstr", words as u64)?;
    for (index, arg) in args.iter().enumerate() {
        let name = format!("bo{index}");
        put(&name, arg.host as u64)?;
    }
    Ok(packet)
}

/// An ioctl, with the driver's own words for a refusal.
fn ioctl(board: &Board, request: u64, arg: *mut c_void) -> Result<(), TransportError> {
    let rc = unsafe { libc_ioctl(board.fd(), request, arg) };
    if rc < 0 {
        return Err(TransportError::new(format!(
            "the driver refused ({}): {}",
            errno_text(-rc),
            ioctl_name(request)
        )));
    }
    Ok(())
}

/// Close a buffer object's handle.
fn close_bo(board: &Board, handle: u32) {
    let mut args = GemClose { handle: handle as u64 };
    unsafe {
        libc_ioctl(
            board.fd(),
            IOCTL_GEM_CLOSE,
            &mut args as *mut GemClose as *mut c_void,
        );
    }
}

/// Destroy a syncobj.
fn destroy_syncobj(board: &Board, handle: u32) {
    let mut args = SyncobjDestroy { handle: handle as u64 };
    unsafe {
        libc_ioctl(
            board.fd(),
            IOCTL_SYNCOBJ_DESTROY,
            &mut args as *mut SyncobjDestroy as *mut c_void,
        );
    }
}

/// Make a buffer's contents visible to the other side.
fn sync_bo(
    board: &Board,
    handle: u32,
    offset: u64,
    size: u64,
    direction: u32,
) -> Result<(), TransportError> {
    let mut args = SyncBo {
        handle,
        direction,
        offset,
        size,
    };
    ioctl(
        board,
        IOCTL_SYNC_BO,
        &mut args as *mut SyncBo as *mut c_void,
    )
}

/// The name of an ioctl, for the message a refusal carries.
fn ioctl_name(request: u64) -> &'static str {
    match request {
        IOCTL_CREATE_HWCTX => "CREATE_HWCTX",
        IOCTL_DESTROY_HWCTX => "DESTROY_HWCTX",
        IOCTL_CONFIG_HWCTX => "CONFIG_HWCTX",
        IOCTL_CREATE_BO => "CREATE_BO",
        IOCTL_GET_BO_INFO => "GET_BO_INFO",
        IOCTL_SYNC_BO => "SYNC_BO",
        IOCTL_EXEC_CMD => "EXEC_CMD",
        IOCTL_GET_INFO => "GET_INFO",
        IOCTL_GEM_CLOSE => "GEM_CLOSE",
        IOCTL_SYNCOBJ_DESTROY => "SYNCOBJ_DESTROY",
        IOCTL_SYNCOBJ_TIMELINE_WAIT => "SYNCOBJ_TIMELINE_WAIT",
        _ => "an unknown ioctl",
    }
}

/// The name of an error number, for the message a refusal carries.
fn errno_text(errno: i32) -> String {
    let name = match errno {
        1 => "EPERM",
        11 => "EAGAIN",
        12 => "ENOMEM",
        13 => "EACCES",
        14 => "EFAULT",
        16 => "EBUSY",
        19 => "ENODEV",
        22 => "EINVAL",
        25 => "ENOTTY",
        28 => "ENOSPC",
        95 => "EOPNOTSUPP",
        _ => "an unknown error",
    };
    format!("{name} ({errno})")
}

/// Read the generation from the device.
///
/// The driver answers `GET_INFO` with the AIE's version: the AIE1 family
/// is XDNA1, and the AIE2P family is XDNA2. The C++ runtime could not ask
/// -- it read the environment -- which is why this is here.
///
/// The npu2's number is not verifiable on an npu1: what the board says is
/// that its AIE reports 1.1, and the rule is that the AIE1 family is npu1.
#[cfg(target_os = "linux")]
pub fn read_generation() -> Option<Generation> {
    use std::os::fd::AsRawFd;

    let file = std::fs::File::open(crate::device::ACCEL_NODE).ok()?;
    let mut version = AieVersion { major: 0, minor: 0 };
    let mut info = GetInfo {
        param: QUERY_AIE_VERSION,
        buffer_size: size_of::<AieVersion>() as u32,
        buffer: &mut version as *mut AieVersion as u64,
    };
    let rc = unsafe {
        libc_ioctl(
            file.as_raw_fd(),
            IOCTL_GET_INFO,
            &mut info as *mut GetInfo as *mut c_void,
        )
    };
    if rc < 0 {
        return None;
    }
    match version.major {
        1 => Some(Generation::Npu1),
        2 => Some(Generation::Npu2),
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

/// The raw mmap, without a libc dependency.
///
/// The mapping is the driver's own: a buffer's `map_offset` is a fake
/// offset that the driver turns into the buffer's pages.
#[cfg(target_os = "linux")]
unsafe fn libc_mmap(
    addr: *mut std::ffi::c_void,
    length: usize,
    prot: i32,
    flags: i32,
    fd: i32,
    offset: i64,
) -> *mut std::ffi::c_void {
    #[cfg(target_arch = "x86_64")]
    const SYS_MMAP: i64 = 9;
    #[cfg(target_arch = "aarch64")]
    const SYS_MMAP: i64 = 222;

    let ret: i64;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::asm!(
            "syscall",
            inlateout("rax") SYS_MMAP => ret,
            in("rdi") addr,
            in("rsi") length,
            in("rdx") prot,
            in("r10") flags,
            in("r8") fd,
            in("r9") offset,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack)
        );
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!(
            "syscall",
            inlateout("rax") SYS_MMAP => ret,
            in("x0") addr,
            in("x1") length,
            in("x2") prot,
            in("x3") flags,
            in("x4") fd,
            in("x5") offset,
            options(nostack)
        );
    }
    ret as *mut std::ffi::c_void
}

/// The raw munmap, without a libc dependency.
#[cfg(target_os = "linux")]
unsafe fn libc_munmap(addr: *mut std::ffi::c_void, length: usize) -> i32 {
    #[cfg(target_arch = "x86_64")]
    const SYS_MUNMAP: i64 = 11;
    #[cfg(target_arch = "aarch64")]
    const SYS_MUNMAP: i64 = 215;

    let ret: i64;
    unsafe {
        std::arch::asm!(
            "syscall",
            inlateout("rax") SYS_MUNMAP => ret,
            in("rdi") addr,
            in("rsi") length,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack)
        );
    }
    ret as i32
}

#[cfg(not(target_os = "linux"))]
unsafe fn libc_mmap(
    _addr: *mut std::ffi::c_void,
    _length: usize,
    _prot: i32,
    _flags: i32,
    _fd: i32,
    _offset: i64,
) -> *mut std::ffi::c_void {
    std::ptr::null_mut()
}

#[cfg(not(target_os = "linux"))]
unsafe fn libc_munmap(_addr: *mut std::ffi::c_void, _length: usize) -> i32 {
    0
}
