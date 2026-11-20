//! The device: an instance, a queue, buffers, and a dispatch.
//!
//! # What this layer is, and what it is not
//!
//! It takes `&[f32]` and returns `&[f32]`. It knows nothing about tensors,
//! dtypes, models or the `Backend` trait -- that is `lib.rs`'s job. Keeping
//! the split means a shader's arithmetic can be tested against
//! `npum_cpu::kernels` with no model in sight, which is what
//! `tests/tolerance.rs` does.
//!
//! # The one descriptor set layout
//!
//! Every shader in this backend uses the SAME layout: bindings 0-4 are
//! storage buffers. A shader may use fewer bindings than the set declares --
//! Vulkan only requires the ones the shader reads to be valid -- so one
//! layout serves all shaders and a dispatch is a list of buffers, not a
//! reflection pass. Parameters travel as PUSH CONSTANTS (16 bytes), not as a
//! uniform buffer: a dispatch no longer maps, writes and unmaps a buffer
//! just to hand four words to a shader.
//!
//! # Host-visible memory, deliberately
//!
//! Every buffer here is `HOST_VISIBLE | HOST_COHERENT` storage. On the
//! integrated GPU this backend exists for, that IS device memory, so there
//! is no staging copy to save. Each buffer is mapped ONCE, when it is made,
//! and stays mapped until it is freed: map/unmap are driver calls, and a
//! forward pass used to make hundreds of them.
//!
//! # A ring of command buffers, and a finite wait
//!
//! `FRAMES` command buffers and fences are kept in a ring. A frame's fence
//! is waited on before that frame is reused, with a FINITE timeout: an
//! unbounded `u64::MAX` wait turns a driver bug into a hang that lasts
//! longer than the machine. Operations may be grouped into a batch: the
//! dispatches are recorded into one command buffer, separated by barriers,
//! and submitted once. Host reads happen only after the batch's fence.

use std::collections::{HashMap, VecDeque};
use std::ffi::{CStr, c_char};

use ash::vk;

use crate::shaders;

/// The number of storage bindings every shader's set declares.
const STORAGE_BINDINGS: u32 = 5;

/// Command-buffer/fence slots in the ring. Two are enough to overlap one
/// frame with the next; three let the host stay a frame ahead.
const FRAMES: usize = 3;

/// Descriptor sets per frame. One set is needed per recorded dispatch,
/// because descriptor writes are observed at EXECUTION time: reusing a set
/// inside one command buffer would make every dispatch see the last write.
const SETS_PER_FRAME: usize = 64;

/// A finite wait: 60 seconds is long enough for a large software-rendered
/// dispatch and short enough to report a wedged driver instead of hanging.
const WAIT_TIMEOUT_NS: u64 = 60_000_000_000;

/// A weight cache entry's identity. `ptr` is the address of the immutable
/// model buffer the caller handed us; the fingerprint catches the accidental
/// reuse of a scratch buffer at the same address.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeightId {
    ptr: usize,
    len: usize,
    fingerprint: [u32; 3],
}

impl WeightId {
    /// Identify a weight by its address, length and a cheap fingerprint.
    ///
    /// The model's weights are long-lived `Vec<f32>`s, so the pointer is a
    /// stable identity across forward passes. Hashing the whole matrix on
    /// every call would cost more than the upload it saves; three words
    /// sampled across the matrix make an accidental collision implausible.
    fn new(values: &[f32]) -> Self {
        let len = values.len();
        let word = |i: usize| values.get(i).copied().unwrap_or(0.0).to_bits();
        Self {
            ptr: values.as_ptr() as usize,
            len,
            fingerprint: [word(0), word(len / 2), word(len.saturating_sub(1))],
        }
    }

    /// Whether `values` still looks like the weight this id was made from.
    fn matches(&self, values: &[f32]) -> bool {
        self.len == values.len() && Self::new(values).fingerprint == self.fingerprint
    }
}

/// A cached weight buffer.
struct WeightEntry {
    buf: Buf,
    /// The number of f32 values actually written (may be `len` of the id).
    len: usize,
}

/// One device buffer and the memory behind it.
///
/// The mapping is permanent: `free` is the only place that unmaps.
pub struct Buf {
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    /// The permanent host mapping of `mem`.
    ptr: *mut u8,
    /// The requested size, used to return the buffer to the right pool bucket.
    size: u64,
}

// A `Buf` is an opaque handle: no Rust reference ever borrows the memory
// behind it. The pointer stays valid until `Gpu::free` unmaps and frees it.
unsafe impl Send for Buf {}

/// One frame: its command buffer, fence, and one descriptor set per dispatch.
struct Frame {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    sets: Vec<vk::DescriptorSet>,
    /// Dispatches recorded but not submitted yet.
    pending: usize,
    /// The fence is in an unsignalled, reusable state.
    busy: bool,
}

/// An open Vulkan device with a compute queue.
pub struct Gpu {
    /// The loader stays open for as long as the instance does.
    _entry: ash::Entry,
    instance: ash::Instance,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    device: ash::Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    frames: Vec<Frame>,
    /// The frame currently being recorded into.
    frame: usize,
    /// True between `begin_batch` and `end_batch`.
    batching: bool,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipelines: HashMap<&'static str, vk::Pipeline>,
    /// A valid one-word buffer for the bindings a shader does not use.
    dummy: Buf,
    /// One descriptor pool and its sets, reused for the device's whole life.
    desc_pool: vk::DescriptorPool,
    /// Weights already resident on the device, keyed by immutable identity.
    weights: HashMap<WeightId, WeightEntry>,
    /// FIFO order for evicting weights when the cache grows too large.
    weight_order: VecDeque<WeightId>,
    /// Bytes held by the weight cache, so eviction has a hard bound.
    weight_bytes: u64,
    /// Retired buffers, bucketed by size, for reuse by the next op.
    pool: HashMap<u64, Vec<Buf>>,
    /// Buffers released DURING a batch. Descriptors recorded earlier in the
    /// command buffer still refer to them, so they cannot go back to the
    /// pool until the batch has been submitted and waited on.
    batch_retired: Vec<Buf>,
    /// Buffers in flight until `end_batch` reads them back.
    deferred: Vec<Deferred>,
    /// Reusable host staging for the GEMM bias epilogue, padded to `n`.
    bias_scratch: Vec<f32>,
    /// Reusable host staging for the GEMM accumulator, padded to `m*n`.
    /// A pool buffer recycled from another op still holds its old bytes, and
    /// the shader reads every one of those `m*n` floats.
    acc_scratch: Vec<f32>,
    name: String,
}

/// A device buffer whose contents a batch will hand back to the host.
struct Deferred {
    /// The host destination, valid until `end_batch`.
    out: usize,
    len: usize,
    buf: Buf,
}

impl Gpu {
    /// Open the first device that has a compute queue.
    pub fn open() -> Result<Self, String> {
        unsafe {
            let entry = ash::Entry::load().map_err(|e| format!("no Vulkan loader: {e}"))?;
            let app = vk::ApplicationInfo::default()
                .application_name(c"npumembeddings")
                .api_version(vk::make_api_version(0, 1, 1, 0));
            let create = vk::InstanceCreateInfo::default().application_info(&app);
            let instance = entry
                .create_instance(&create, None)
                .map_err(|e| format!("no Vulkan instance: {e}"))?;

            let devices = match instance.enumerate_physical_devices() {
                Ok(d) => d,
                Err(e) => {
                    instance.destroy_instance(None);
                    return Err(format!("cannot enumerate Vulkan devices: {e}"));
                }
            };

            let mut pick = None;
            for pd in devices {
                let families = instance.get_physical_device_queue_family_properties(pd);
                let compute = families.iter().position(|f| {
                    f.queue_flags.contains(vk::QueueFlags::COMPUTE) && f.queue_count > 0
                });
                if let Some(family) = compute {
                    pick = Some((pd, family as u32));
                    break;
                }
            }
            let Some((pd, family)) = pick else {
                instance.destroy_instance(None);
                return Err("no Vulkan device with a compute queue".to_string());
            };
            Self::build(entry, instance, pd, family)
        }
    }

    /// The instance is open and a device was chosen: build the rest.
    unsafe fn build(
        entry: ash::Entry,
        instance: ash::Instance,
        pd: vk::PhysicalDevice,
        family: u32,
    ) -> Result<Self, String> {
        let mem_props = unsafe { instance.get_physical_device_memory_properties(pd) };
        let out = unsafe { Self::build_inner(&instance, pd, family, &mem_props) };
        match out {
            Ok(rest) => {
                let props = unsafe { instance.get_physical_device_properties(pd) };
                let name = unsafe {
                    CStr::from_ptr(props.device_name.as_ptr() as *const c_char)
                        .to_string_lossy()
                        .into_owned()
                };
                let mut gpu = Self {
                    _entry: entry,
                    instance,
                    mem_props,
                    device: rest.device,
                    queue: rest.queue,
                    cmd_pool: rest.cmd_pool,
                    frames: rest.frames,
                    frame: 0,
                    batching: false,
                    set_layout: rest.set_layout,
                    pipeline_layout: rest.pipeline_layout,
                    pipelines: rest.pipelines,
                    dummy: rest.dummy,
                    desc_pool: rest.desc_pool,
                    weights: HashMap::new(),
                    weight_order: VecDeque::new(),
                    weight_bytes: 0,
                    pool: HashMap::new(),
                    batch_retired: Vec::new(),
                    deferred: Vec::new(),
                    bias_scratch: Vec::new(),
                    acc_scratch: Vec::new(),
                    name,
                };
                // The dummy is what a shader's unused bindings point at.
                // It lives for the device's whole lifetime.
                gpu.dummy = gpu.make(4)?;
                Ok(gpu)
            }
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                Err(e)
            }
        }
    }

    /// The device, the queue, the frames and the one set layout.
    unsafe fn build_inner(
        instance: &ash::Instance,
        pd: vk::PhysicalDevice,
        family: u32,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
    ) -> Result<GpuRest, String> {
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let create = vk::DeviceCreateInfo::default().queue_create_infos(&queues);
        let device = unsafe { instance.create_device(pd, &create, None) }
            .map_err(|e| format!("cannot open the device: {e}"))?;
        let queue = unsafe { device.get_device_queue(family, 0) };

        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|e| format!("cannot create a command pool: {e}"))?;

        // Bindings 0..STORAGE_BINDINGS are storage buffers. Parameters are
        // push constants, so there is no uniform binding and no uniform
        // buffer to write on every dispatch.
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..STORAGE_BINDINGS)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(|e| format!("cannot create a descriptor set layout: {e}"))?;
        let layouts = [set_layout];
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(16)];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&push),
                None,
            )
        }
        .map_err(|e| format!("cannot create a pipeline layout: {e}"))?;

        // One pool for the device's whole life, sized for every set of every
        // frame. A batch records several dispatches into one command buffer;
        // each needs its own set because descriptors are read at execution.
        let total_sets = (FRAMES * SETS_PER_FRAME) as u32;
        let desc_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(total_sets)
                    .pool_sizes(&[vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(total_sets * STORAGE_BINDINGS)]),
                None,
            )
        }
        .map_err(|e| format!("cannot create a descriptor pool: {e}"))?;
        // ash 0.38 exposes `p_set_layouts` as a raw field: one layout per set.
        let layout_copies = vec![set_layout; total_sets as usize];
        let allocate = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            descriptor_pool: desc_pool,
            descriptor_set_count: total_sets,
            p_set_layouts: layout_copies.as_ptr(),
            _marker: std::marker::PhantomData,
        };
        let all_sets = unsafe { device.allocate_descriptor_sets(&allocate) }
            .map_err(|e| format!("cannot allocate descriptor sets: {e}"))?;

        // Make the zero-initialised `Frame`s first so a failure in the middle
        // unwinds through `GpuRest`'s Drop, which frees what was made so far.
        let mut rest = GpuRest {
            device: device.clone(),
            queue,
            cmd_pool,
            frames: Vec::with_capacity(FRAMES),
            set_layout,
            pipeline_layout,
            pipelines: HashMap::new(),
            dummy: Buf {
                buf: vk::Buffer::null(),
                mem: vk::DeviceMemory::null(),
                ptr: std::ptr::null_mut(),
                size: 0,
            },
            desc_pool,
        };
        for frame in 0..FRAMES {
            let cmd = unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
            }
            .map_err(|e| format!("cannot allocate a command buffer: {e}"))?[0];
            let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
                .map_err(|e| format!("cannot create a fence: {e}"))?;
            let sets = all_sets[frame * SETS_PER_FRAME..(frame + 1) * SETS_PER_FRAME].to_vec();
            rest.frames.push(Frame {
                cmd,
                fence,
                sets,
                pending: 0,
                busy: false,
            });
        }
        let _ = mem_props; // buffers are made after the struct owns the properties
        Ok(rest)
    }

    /// The device's name, as the driver reports it.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// A buffer of `bytes` f32 bytes, host visible, permanently mapped.
    fn make(&self, bytes: u64) -> Result<Buf, String> {
        unsafe {
            let size = bytes.max(4);
            let create = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buf = self
                .device
                .create_buffer(&create, None)
                .map_err(|e| format!("cannot create a buffer: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            // HOST_VISIBLE | HOST_COHERENT: see the module header. A device
            // with no such type cannot hold this backend's buffers, and that
            // is a refusal rather than a silent fallback to a type that
            // cannot be mapped.
            let wanted =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let index = (0..self.mem_props.memory_type_count).find(|i| {
                let ty = self.mem_props.memory_types[*i as usize];
                req.memory_type_bits & (1 << i) != 0 && ty.property_flags.contains(wanted)
            });
            let Some(index) = index else {
                self.device.destroy_buffer(buf, None);
                return Err("no host-visible coherent memory for a buffer".to_string());
            };
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(index);
            let mem = self.device.allocate_memory(&alloc, None).map_err(|e| {
                self.device.destroy_buffer(buf, None);
                format!("cannot allocate memory: {e}")
            })?;
            self.device
                .bind_buffer_memory(buf, mem, 0)
                .map_err(|e| format!("cannot bind memory: {e}"))?;
            let ptr = self
                .device
                .map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("cannot map a buffer: {e}"))? as *mut u8;
            Ok(Buf { buf, mem, ptr, size })
        }
    }

    /// Take a buffer of at least `bytes` from the pool, or make a new one.
    fn acquire(&mut self, bytes: u64) -> Result<Buf, String> {
        let key = bytes.max(4);
        if let Some(buf) = self.pool.get_mut(&key).and_then(Vec::pop) {
            return Ok(buf);
        }
        self.make(key)
    }

    /// Return a buffer for reuse.
    ///
    /// Inside a batch the release is deferred: a descriptor already recorded
    /// into the command buffer still names this buffer, and descriptors are
    /// read at EXECUTION time. Releasing it to the pool here would let the
    /// next op overwrite a buffer an earlier dispatch has yet to read.
    fn release(&mut self, buf: Buf) {
        if self.batching {
            self.batch_retired.push(buf);
        } else {
            self.pool.entry(buf.size).or_default().push(buf);
        }
    }

    /// Write f32 values into a buffer through its permanent mapping.
    fn write(&self, b: &Buf, values: &[f32]) -> Result<(), String> {
        if values.is_empty() {
            return Ok(());
        }
        if (values.len() * 4) as u64 > b.size {
            return Err(format!(
                "a {} f32 upload does not fit a {} byte device buffer",
                values.len(),
                b.size
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr() as *const u8,
                b.ptr,
                values.len() * 4,
            );
        }
        Ok(())
    }

    /// Read f32 values out of a buffer through its permanent mapping.
    fn read(&self, b: &Buf, len: usize) -> Result<Vec<f32>, String> {
        let mut out = vec![0.0f32; len];
        self.read_into(b, &mut out)?;
        Ok(out)
    }

    /// Read f32 values back directly into the caller's slice.
    fn read_into(&self, b: &Buf, out: &mut [f32]) -> Result<(), String> {
        if out.is_empty() {
            return Ok(());
        }
        if (out.len() * 4) as u64 > b.size {
            return Err(format!(
                "a {} f32 read does not fit a {} byte device buffer",
                out.len(),
                b.size
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(b.ptr, out.as_mut_ptr() as *mut u8, out.len() * 4);
        }
        Ok(())
    }

    /// Free a buffer and its memory, unmapping the permanent mapping first.
    fn free(&self, b: &Buf) {
        unsafe {
            if b.ptr != std::ptr::null_mut() {
                self.device.unmap_memory(b.mem);
            }
            if b.buf != vk::Buffer::null() {
                self.device.destroy_buffer(b.buf, None);
            }
            if b.mem != vk::DeviceMemory::null() {
                self.device.free_memory(b.mem, None);
            }
        }
    }

    /// The pipeline for a shader, created on first use.
    fn pipeline(&mut self, shader: &'static str) -> Result<vk::Pipeline, String> {
        if let Some(p) = self.pipelines.get(shader) {
            return Ok(*p);
        }
        let words = shaders::ALL
            .iter()
            .find(|(n, _)| *n == shader)
            .map(|(_, w)| *w)
            .ok_or_else(|| format!("no shader named {shader}"))?;
        unsafe {
            let module = self
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
                .map_err(|e| format!("cannot create the {shader} module: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let create = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(self.pipeline_layout);
            let made =
                self.device
                    .create_compute_pipelines(vk::PipelineCache::null(), &[create], None);
            self.device.destroy_shader_module(module, None);
            let pipeline =
                made.map_err(|(_, e)| format!("cannot create the {shader} pipeline: {e}"))?[0];
            self.pipelines.insert(shader, pipeline);
            Ok(pipeline)
        }
    }

    /// Wait for a frame's fence with a finite timeout, then make it reusable.
    fn wait_frame(&mut self, index: usize) -> Result<(), String> {
        let frame = &self.frames[index];
        if !frame.busy {
            return Ok(());
        }
        unsafe {
            self.device
                .wait_for_fences(&[frame.fence], true, WAIT_TIMEOUT_NS)
                .map_err(|e| {
                    if e == vk::Result::TIMEOUT {
                        "the GPU did not finish within 60 seconds".to_string()
                    } else {
                        format!("the GPU did not finish: {e}")
                    }
                })?;
            self.device
                .reset_fences(&[frame.fence])
                .map_err(|e| format!("cannot reset the fence: {e}"))?;
        }
        self.frames[index].busy = false;
        Ok(())
    }

    /// Submit a frame and leave its fence signalled.
    fn submit_frame(&mut self, index: usize) -> Result<(), String> {
        let frame = &self.frames[index];
        unsafe {
            self.device
                .end_command_buffer(frame.cmd)
                .map_err(|e| format!("cannot end the command buffer: {e}"))?;
            let cmds = [frame.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            self.device
                .queue_submit(self.queue, &[submit], frame.fence)
                .map_err(|e| format!("cannot submit: {e}"))?;
        }
        self.frames[index].busy = true;
        self.frames[index].pending = 0;
        Ok(())
    }

    /// Start recording into a frame, waiting for it if it is still in use.
    fn begin_frame(&mut self) -> Result<usize, String> {
        let index = self.frame;
        self.wait_frame(index)?;
        {
            let frame = &mut self.frames[index];
            unsafe {
                self.device
                    .reset_command_buffer(frame.cmd, vk::CommandBufferResetFlags::empty())
                    .map_err(|e| format!("cannot reset the command buffer: {e}"))?;
                // `run` begins the recording on its first dispatch, so that
                // the same code path works for a fresh frame and for a frame
                // that already has dispatches recorded in this batch.
            }
            frame.pending = 0;
        }
        self.frame = (index + 1) % FRAMES;
        Ok(index)
    }

    /// Group the following operations into a single submission.
    ///
    /// Results are not handed back to the host until `end_batch`. Host
    /// buffers passed as inputs must stay valid and unchanged until then.
    pub fn begin_batch(&mut self) -> Result<(), String> {
        if self.batching {
            return Err("a batch is already in progress".to_string());
        }
        self.batching = true;
        Ok(())
    }

    /// Submit a batch, wait for it, and hand every result back to the host.
    pub fn end_batch(&mut self) -> Result<(), String> {
        if !self.batching {
            return Err("no batch is in progress".to_string());
        }
        self.batching = false;
        let frame = (self.frame + FRAMES - 1) % FRAMES;
        if self.frames[frame].pending > 0 {
            self.submit_frame(frame)?;
            self.wait_frame(frame)?;
        }
        let deferred = std::mem::take(&mut self.deferred);
        let mut error = None;
        for d in deferred {
            // SAFETY: the caller promised the output buffers stay valid until
            // `end_batch`, and no Rust reference to them is alive here.
            unsafe {
                let out = std::slice::from_raw_parts_mut(d.out as *mut f32, d.len);
                if let Err(e) = self.read_into(&d.buf, out) {
                    error = Some(e);
                }
            }
            self.release(d.buf);
        }
        // Only now, with the fence waited on, may the batch's intermediate
        // buffers be handed to the next op.
        let retired = std::mem::take(&mut self.batch_retired);
        for b in retired {
            self.release(b);
        }
        error.map_or(Ok(()), Err)
    }

    /// Run one shader over `storage` buffers and 16 bytes of push constants.
    fn run(
        &mut self,
        shader: &'static str,
        storage: &[&Buf],
        params: [u32; 4],
        groups: (u32, u32, u32),
    ) -> Result<(), String> {
        let pipeline = self.pipeline(shader)?;
        if groups.0 == 0 || groups.1 == 0 || groups.2 == 0 {
            return Ok(());
        }
        // A batch that fills its descriptor sets is closed early: submitting
        // and waiting is always legal, and the next dispatch starts a fresh
        // command buffer.
        let mut index = (self.frame + FRAMES - 1) % FRAMES;
        if self.batching && self.frames[index].pending >= SETS_PER_FRAME {
            self.end_batch()?;
            self.begin_batch()?;
        }
        if self.frames[index].pending == 0 {
            // `begin_frame` returns the slot it just made ready; `self.frame`
            // has already advanced to the NEXT one.
            index = self.begin_frame()?;
        }

        let frame = &mut self.frames[index];
        let set = frame.sets[frame.pending];
        unsafe {
            // The descriptors have to be in place before the writes point at
            // them; infos owns the addresses the writes borrow.
            let mut infos: Vec<vk::DescriptorBufferInfo> =
                Vec::with_capacity(STORAGE_BINDINGS as usize);
            for binding in 0..STORAGE_BINDINGS {
                let buffer = storage
                    .get(binding as usize)
                    .map_or(&self.dummy, |b| *b)
                    .buf;
                infos.push(
                    vk::DescriptorBufferInfo::default()
                        .buffer(buffer)
                        .offset(0)
                        .range(vk::WHOLE_SIZE),
                );
            }
            let writes: Vec<vk::WriteDescriptorSet> = (0..STORAGE_BINDINGS)
                .map(|binding| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&infos[binding as usize]))
                })
                .collect();
            self.device.update_descriptor_sets(&writes, &[]);

            if frame.pending > 0 {
                // The previous dispatch in this submission produced the
                // inputs of this one: make its writes available and visible
                // to the compute shader that follows. There is deliberately
                // NO trailing barrier -- the fence wait makes the last
                // dispatch's writes available to the host on host-coherent
                // memory.
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                    );
                self.device.cmd_pipeline_barrier(
                    frame.cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }
            if frame.pending == 0 {
                self.device
                    .reset_command_buffer(frame.cmd, vk::CommandBufferResetFlags::empty())
                    .map_err(|e| format!("cannot reset the command buffer: {e}"))?;
                self.device
                    .begin_command_buffer(
                        frame.cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(|e| format!("cannot begin the command buffer: {e}"))?;
            }
            let bytes: &[u8] =
                std::slice::from_raw_parts(params.as_ptr() as *const u8, params.len() * 4);
            self.device.cmd_push_constants(
                frame.cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes,
            );
            self.device
                .cmd_bind_pipeline(frame.cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                frame.cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[set],
                &[],
            );
            self.device
                .cmd_dispatch(frame.cmd, groups.0, groups.1, groups.2);
            frame.pending += 1;
        }

        if !self.batching {
            self.submit_frame(index)?;
            self.wait_frame(index)?;
        }
        Ok(())
    }

    /// Look up a pending batch output by the host pointer it came from.
    fn deferred(&self, ptr: usize, len: usize) -> Option<&Buf> {
        self.deferred
            .iter()
            .find(|d| d.out == ptr && d.len == len)
            .map(|d| &d.buf)
    }

    /// Register a device buffer whose result must be handed back at batch end.
    ///
    /// A second write to the same host pointer replaces the old pending
    /// result: its buffer is no longer needed and goes back to the pool.
    fn defer(&mut self, out: usize, len: usize, buf: Buf) {
        if let Some(pos) = self.deferred.iter().position(|d| d.out == out) {
            let old = self.deferred.remove(pos);
            // An in-place op returns the very buffer it consumed: releasing
            // it here would put the same mapping into the pool twice.
            if old.buf.ptr != buf.ptr {
                self.release(old.buf);
            }
        }
        self.deferred.push(Deferred { out, len, buf });
    }

    /// The device buffer holding a host slice's input, if it is in this batch.
    fn input<'a>(&'a self, values: &[f32], len: usize) -> Option<&'a Buf> {
        self.deferred(values.as_ptr() as usize, len)
    }

    /// Take an op's input buffer, reusing a pending batch result when the
    /// host slice is one of this batch's outputs. The bool says whether the
    /// buffer is BORROWED from `deferred` (and must not be released here).
    fn take_input(&mut self, values: &[f32], len: usize) -> Result<(Buf, bool), String> {
        if let Some(buf) = self.input(values, len) {
            Ok((Self::cached(buf), true))
        } else {
            let b = self.acquire((len * 4) as u64)?;
            self.write(&b, values)?;
            Ok((b, false))
        }
    }

    /// Release a non-borrowed input buffer. A borrowed one is owned by
    /// `deferred`; `defer` releases it when the same pointer is overwritten.
    fn release_input(&mut self, _out_ptr: usize, buf: Buf, borrowed: bool) {
        if !borrowed {
            self.release(buf);
        }
    }

    /// A non-owning handle to a cache entry's buffer.
    ///
    /// `Buf` has no `Drop`: only `Gpu::free` releases its handles. Copying the
    /// four fields is therefore safe as long as the copy is never passed to
    /// `release` or `free` -- the cache owns the original.
    fn cached(buf: &Buf) -> Buf {
        Buf {
            buf: buf.buf,
            mem: buf.mem,
            ptr: buf.ptr,
            size: buf.size,
        }
    }

    /// The weight buffer for `values`, uploaded once and reused thereafter.
    ///
    /// The matrix keeps its `[k, n]` layout: the GEMM shader's threads map to
    /// output COLUMNS, so a row-major B is already coalesced on load. What
    /// needed fixing in the shader was the shared-memory bank pattern, not
    /// the global one -- see `gemm.wgsl`.
    fn weight(&mut self, values: &[f32]) -> Result<Buf, String> {
        let id = WeightId::new(values);
        if let Some(entry) = self.weights.get(&id) {
            if entry.len == values.len() && id.matches(values) {
                return Ok(Self::cached(&entry.buf));
            }
        }
        let bytes = (values.len() * 4) as u64;
        let b = self.acquire(bytes)?;
        self.write(&b, values)?;
        self.weight_bytes += bytes;
        self.weight_order.push_back(id);
        // A batch's recorded descriptors still name every cache entry, so
        // eviction waits until the batch is over.
        while !self.batching && self.weight_bytes > 256 * 1024 * 1024 {
            let Some(old) = self.weight_order.pop_front() else { break };
            if let Some(entry) = self.weights.remove(&old) {
                self.weight_bytes -= entry.buf.size;
                self.free(&entry.buf);
            }
        }
        self.weights.insert(
            id,
            WeightEntry {
                buf: Buf {
                    buf: b.buf,
                    mem: b.mem,
                    ptr: b.ptr,
                    size: b.size,
                },
                len: values.len(),
            },
        );
        Ok(b)
    }

    /// `c = a * b`, `a` is `[m, k]`, `b` is `[k, n]`, all row-major.
    pub fn gemm(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        self.gemm_with_bias(a, b, None, m, k, n)
    }

    /// `c = a * b + bias[col]`, written straight into the host slice `out`.
    ///
    /// In a batch the slice is registered as a pending result and filled in
    /// by `end_batch`; no host read happens here. The caller must keep the
    /// slice valid and untouched until then.
    pub fn gemm_with_bias_into(
        &mut self,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        m: usize,
        k: usize,
        n: usize,
        out: usize,
    ) -> Result<(), String> {
        self.gemm_full(a, b, bias, None, m, k, n, Some(out), true)?;
        Ok(())
    }

    /// `c = a * b + bias[col]`, with the bias added in the shader epilogue.
    pub fn gemm_with_bias(
        &mut self,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        self.gemm_full(a, b, bias, None, m, k, n, None, true)
    }

    /// `c = a * b + bias[col] + acc[row, col]` in ONE epilogue.
    ///
    /// This is the shape the tensor `Backend::gemm` needs when `acc` is
    /// present: previously it read the GEMM back to the host and issued a
    /// second `add` dispatch plus two more round-trips.
    pub fn gemm_with_acc(
        &mut self,
        a: &[f32],
        b: &[f32],
        acc: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        self.gemm_full(a, b, None, Some(acc), m, k, n, None, true)
    }

    /// The tensor `Backend`'s GEMM: operands come from reusable host scratch,
    /// so `b` is uploaded every time rather than looked up by pointer.
    pub fn gemm_tensor(
        &mut self,
        a: &[f32],
        b: &[f32],
        acc: Option<&[f32]>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        self.gemm_full(a, b, None, acc, m, k, n, None, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_full(
        &mut self,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        acc: Option<&[f32]>,
        m: usize,
        k: usize,
        n: usize,
        out: Option<usize>,
        cache_b: bool,
    ) -> Result<Vec<f32>, String> {
        // The shader reads exactly m*k, k*n and (with an accumulator) m*n
        // floats. A short operand would leave recycled pool bytes in the
        // unwritten tail; a long accumulator would overrun the buffer.
        if a.len() < m * k || b.len() < k * n || acc.is_some_and(|v| v.len() > m * n) {
            return Err(format!(
                "gemm: operands are {}, {} and accumulator {:?}, but the shape is [{m},{k}]x[{k},{n}]",
                a.len(),
                b.len(),
                acc.map(|v| v.len())
            ));
        }
        // A pending batch output can be consumed as an input without a
        // round-trip through the host. `take_input` says whether the buffer
        // was borrowed from the batch (and must not be released here).
        let (ab, a_borrowed) = self.take_input(a, a.len())?;
        let cb = self.acquire((m * n * 4) as u64)?;
        let bias_buf = bias.map(|_| self.acquire((n * 4) as u64)).transpose()?;
        let acc_buf = acc.map(|_| self.acquire((m * n * 4) as u64)).transpose()?;
        // A cache hit writes nothing; a miss uploads once for the model's
        // whole life (or until the bounded cache evicts it).
        let bb = if cache_b {
            self.weight(b)?
        } else {
            // The tensor `Backend` decodes into reusable scratch buffers:
            // same pointer, NEW contents. A pointer-keyed cache would be a
            // correctness bug there, so this path uploads every time.
            let buf = self.acquire((b.len() * 4) as u64)?;
            self.write(&buf, b)?;
            buf
        };
        if let (Some(values), Some(buf)) = (bias, bias_buf.as_ref()) {
            self.bias_scratch.resize(n, 0.0);
            self.bias_scratch.fill(0.0);
            self.bias_scratch[..values.len().min(n)]
                .copy_from_slice(&values[..values.len().min(n)]);
            self.write(buf, &self.bias_scratch)?;
        }
        if let (Some(values), Some(buf)) = (acc, acc_buf.as_ref()) {
            // The CPU reference treats a short accumulator as zero beyond its
            // length. The shader reads all m*n lanes, so the tail has to be
            // ZERO here -- a recycled pool buffer still holds another op's data.
            self.acc_scratch.resize(m * n, 0.0);
            self.acc_scratch.fill(0.0);
            self.acc_scratch[..values.len()].copy_from_slice(values);
            self.write(buf, &self.acc_scratch)?;
        }

        // Bindings are POSITIONAL: bias is 3 and acc is 4 even when one is
        // absent, so a missing bias must still occupy its slot or the
        // accumulator would land in binding 3 and read as zero.
        let zero = Self::cached(&self.dummy);
        let mut storage: Vec<&Buf> = vec![&ab, &bb, &cb];
        storage.push(bias_buf.as_ref().unwrap_or(&zero));
        storage.push(acc_buf.as_ref().unwrap_or(&zero));
        // Bit 0: per-column bias. Bit 2: per-element accumulator.
        let flags = u32::from(bias_buf.is_some()) | (u32::from(acc_buf.is_some()) << 2);
        let params = [m as u32, k as u32, n as u32, flags];
        let dispatched = self.run(
            "gemm",
            &storage,
            params,
            ((n as u32).div_ceil(32), (m as u32).div_ceil(8), 1),
        );

        // A cached `bb` is a non-owning copy; an uncached one was acquired
        // from the pool and has to go back.
        let result = if let Err(e) = dispatched {
            if !cache_b {
                self.release(bb);
            }
            self.release(ab);
            self.release(cb);
            if let Some(buf) = bias_buf {
                self.release(buf);
            }
            if let Some(buf) = acc_buf {
                self.release(buf);
            }
            return Err(e);
        } else if let Some(ptr) = out {
            self.release_input(ptr, ab, a_borrowed);
            self.defer(ptr, m * n, cb);
            Ok(Vec::new())
        } else {
            let values = self.read(&cb, m * n);
            self.release_input(usize::MAX, ab, a_borrowed);
            self.release(cb);
            values
        };
        if !cache_b {
            self.release(bb);
        }
        if let Some(buf) = bias_buf {
            self.release(buf);
        }
        if let Some(buf) = acc_buf {
            self.release(buf);
        }
        result
    }

    /// GELU, in place.
    pub fn gelu(&mut self, x: &mut [f32]) -> Result<(), String> {
        let ptr = x.as_ptr() as usize;
        let (xb, borrowed) = self.take_input(x, x.len())?;
        self.run(
            "gelu",
            &[&xb],
            [x.len() as u32, 0, 0, 0],
            ((x.len() as u32).div_ceil(64), 1, 1),
        )?;
        if self.batching {
            self.defer(ptr, x.len(), xb);
        } else {
            self.read_into(&xb, x)?;
            self.release_input(ptr, xb, borrowed);
        }
        Ok(())
    }

    /// Layer normalisation over rows of `width`.
    pub fn layernorm(
        &mut self,
        x: &mut [f32],
        rows: usize,
        width: usize,
        gamma: Option<&[f32]>,
        beta: Option<&[f32]>,
    ) -> Result<(), String> {
        self.layernorm_inner(x, rows, width, gamma, beta, true)
    }

    /// The tensor `Backend`'s LayerNorm: gamma/beta come from reusable
    /// scratch, so unlike the model weights they must be uploaded afresh.
    pub fn layernorm_fresh(
        &mut self,
        x: &mut [f32],
        rows: usize,
        width: usize,
        gamma: Option<&[f32]>,
        beta: Option<&[f32]>,
    ) -> Result<(), String> {
        self.layernorm_inner(x, rows, width, gamma, beta, false)
    }

    fn layernorm_inner(
        &mut self,
        x: &mut [f32],
        rows: usize,
        width: usize,
        gamma: Option<&[f32]>,
        beta: Option<&[f32]>,
        cache_params: bool,
    ) -> Result<(), String> {
        let ptr = x.as_ptr() as usize;
        let (xb, borrowed) = self.take_input(x, x.len())?;
        // gamma/beta are weights too: the model passes the same immutable
        // buffers on every pass, so they stay resident in the cache. The
        // tensor path opts out because its scratch buffers change contents.
        let (gb, bb) = if cache_params {
            (
                self.weight(gamma.unwrap_or(&[0.0]))?,
                self.weight(beta.unwrap_or(&[0.0]))?,
            )
        } else {
            let gb = self.acquire((gamma.map_or(1, |g| g.len()) * 4) as u64)?;
            let bb = self.acquire((beta.map_or(1, |b| b.len()) * 4) as u64)?;
            self.write(&gb, gamma.unwrap_or(&[0.0]))?;
            self.write(&bb, beta.unwrap_or(&[0.0]))?;
            (gb, bb)
        };
        self.run(
            "layernorm",
            &[&xb, &gb, &bb],
            [
                width as u32,
                rows as u32,
                u32::from(gamma.is_some()),
                u32::from(beta.is_some()),
            ],
            (rows as u32, 1, 1),
        )?;
        if !cache_params {
            // Cached handles are non-owning; freshly acquired ones go back
            // to the pool after the dispatch (or, in a batch, to retirement).
            self.release(gb);
            self.release(bb);
        }
        if self.batching {
            self.defer(ptr, x.len(), xb);
        } else {
            self.read_into(&xb, x)?;
            self.release_input(ptr, xb, borrowed);
        }
        Ok(())
    }

    /// Softmax over rows of `width`.
    pub fn softmax(&mut self, x: &mut [f32], rows: usize, width: usize) -> Result<(), String> {
        self.softmax_strided(x, rows, width, 0)
    }

    /// Softmax over rows of `width` at a row `stride` (0 means `width`).
    ///
    /// The strided form is what a padded score matrix needs: the softmax
    /// normalises the real `width` keys and never touches the padding.
    pub fn softmax_strided(
        &mut self,
        x: &mut [f32],
        rows: usize,
        width: usize,
        stride: usize,
    ) -> Result<(), String> {
        let ptr = x.as_ptr() as usize;
        let (xb, borrowed) = self.take_input(x, x.len())?;
        self.run(
            "softmax",
            &[&xb],
            [width as u32, rows as u32, stride as u32, 0],
            (rows as u32, 1, 1),
        )?;
        if self.batching {
            self.defer(ptr, x.len(), xb);
        } else {
            self.read_into(&xb, x)?;
            self.release_input(ptr, xb, borrowed);
        }
        Ok(())
    }

    /// `out = softmax(q k^T / sqrt(d)) v`, one row of `q` per invocation.
    pub fn attention(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        d: usize,
    ) -> Result<Vec<f32>, String> {
        self.attention_into(q, k, v, d, None)
    }

    fn attention_into(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        d: usize,
        out: Option<usize>,
    ) -> Result<Vec<f32>, String> {
        let q_len = q.len() / d;
        let kv_len = k.len() / d;
        let (qb, q_borrowed) = self.take_input(q, q.len())?;
        let (kb, k_borrowed) = self.take_input(k, k.len())?;
        let (vb, v_borrowed) = self.take_input(v, v.len())?;
        let ob = self.acquire((q.len() * 4) as u64)?;
        let dispatched = self.run(
            "attention",
            &[&qb, &kb, &vb, &ob],
            [q_len as u32, kv_len as u32, d as u32, 0],
            (q_len as u32, 1, 1),
        );
        if let Err(e) = dispatched {
            self.release(qb);
            self.release(kb);
            self.release(vb);
            self.release(ob);
            return Err(e);
        }
        let result = if let Some(ptr) = out {
            self.defer(ptr, q.len(), ob);
            Ok(Vec::new())
        } else {
            let values = self.read(&ob, q.len());
            self.release(ob);
            values
        };
        self.release_input(usize::MAX, qb, q_borrowed);
        self.release_input(usize::MAX, kb, k_borrowed);
        self.release_input(usize::MAX, vb, v_borrowed);
        result
    }

    /// `z = x + y`.
    pub fn add(&mut self, x: &[f32], y: &[f32]) -> Result<Vec<f32>, String> {
        self.add_into(x, y, None)
    }

    /// `x += y`, in place: the residual connection's shader path.
    pub fn add_inplace(&mut self, x: &mut [f32], y: &[f32]) -> Result<(), String> {
        let ptr = x.as_ptr() as usize;
        let out = if self.batching { Some(ptr) } else { None };
        let values = self.add_into(x, y, out)?;
        if !self.batching {
            x.copy_from_slice(&values);
        }
        Ok(())
    }

    fn add_into(&mut self, x: &[f32], y: &[f32], out: Option<usize>) -> Result<Vec<f32>, String> {
        let (xb, x_borrowed) = self.take_input(x, x.len())?;
        let (yb, y_borrowed) = self.take_input(y, y.len())?;
        let zb = self.acquire((x.len() * 4) as u64)?;
        let dispatched = self.run(
            "add",
            &[&xb, &yb, &zb],
            [x.len() as u32, 0, 0, 0],
            ((x.len() as u32).div_ceil(64), 1, 1),
        );
        if let Err(e) = dispatched {
            self.release(xb);
            self.release(yb);
            self.release(zb);
            return Err(e);
        }
        let result = if let Some(ptr) = out {
            self.defer(ptr, x.len(), zb);
            Ok(Vec::new())
        } else {
            let values = self.read(&zb, x.len());
            self.release(zb);
            values
        };
        self.release_input(out.unwrap_or(usize::MAX), xb, x_borrowed);
        self.release_input(out.unwrap_or(usize::MAX), yb, y_borrowed);
        result
    }
}

/// The handles `Gpu` owns once its instance is in place.
struct GpuRest {
    device: ash::Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    frames: Vec<Frame>,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipelines: HashMap<&'static str, vk::Pipeline>,
    dummy: Buf,
    desc_pool: vk::DescriptorPool,
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            for (_, p) in self.pipelines.drain() {
                self.device.destroy_pipeline(p, None);
            }
            self.free(&self.dummy);
            let weights: Vec<Buf> = self.weights.drain().map(|(_, e)| e.buf).collect();
            for b in &weights {
                self.free(b);
            }
            let deferred: Vec<Buf> = self.deferred.drain(..).map(|d| d.buf).collect();
            for b in &deferred {
                self.free(b);
            }
            let retired: Vec<Buf> = self.batch_retired.drain(..).collect();
            for b in &retired {
                self.free(b);
            }
            // Return every pooled allocation before the device goes away.
            let retired: Vec<Buf> = self.pool.drain().flat_map(|(_, v)| v).collect();
            for b in &retired {
                self.free(b);
            }
            for frame in &self.frames {
                if frame.busy {
                    let _ = self
                        .device
                        .wait_for_fences(&[frame.fence], true, WAIT_TIMEOUT_NS);
                }
                self.device.destroy_fence(frame.fence, None);
            }
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_descriptor_pool(self.desc_pool, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
