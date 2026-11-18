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
//! storage buffers, binding 5 is a 16-byte uniform. A shader may use fewer
//! bindings than the set declares -- Vulkan only requires the ones the
//! shader reads to be valid -- so one layout serves all seven shaders and a
//! dispatch is a list of buffers, not a reflection pass. The cost is five
//! descriptors per dispatch instead of two; the benefit is that adding a
//! shader cannot break the layout of another.
//!
//! # Host-visible memory, deliberately
//!
//! Every buffer here is `HOST_VISIBLE | HOST_COHERENT` storage. On the
//! integrated GPU this backend exists for, that IS device memory, so there
//! is no staging copy to save. On a discrete card it would be slower than
//! staging through device-local memory -- and slower is the right first
//! answer: a staging path is an optimisation, and it is measured before it
//! is added, not after.
//!
//! # A command buffer per dispatch
//!
//! Recorded, submitted and waited on per operation. That is the correctness
//! -first shape: no fences to reason about across operations, and a failed
//! dispatch names the operation it was in. Batching is the next measurable
//! improvement, not a correctness one.

use std::collections::HashMap;
use std::ffi::{CStr, c_char};

use ash::vk;

use crate::shaders;

/// The binding of the uniform block in every shader.
const UNIFORM_BINDING: u32 = 5;

/// The number of storage bindings every shader's set declares.
const STORAGE_BINDINGS: u32 = 5;

/// One device buffer and the memory behind it.
pub struct Buf {
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
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
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipelines: HashMap<&'static str, vk::Pipeline>,
    /// A valid one-word buffer for the bindings a shader does not use.
    dummy: Buf,
    name: String,
}

// SAFETY: every call below is a Vulkan call, and Vulkan has no Rust-side
// aliasing rules. The wrappers are `unsafe` because ash says so, not because
// two Rust references could overlap: a `Buf` is an opaque handle and the data
// behind it is never borrowed by Rust.
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
    ///
    /// Takes ownership of the instance and gives it back only on success, so
    /// a failed open cannot leave an instance behind.
    unsafe fn build(
        entry: ash::Entry,
        instance: ash::Instance,
        pd: vk::PhysicalDevice,
        family: u32,
    ) -> Result<Self, String> {
        let out = unsafe { Self::build_inner(&instance, pd, family) };
        match out {
            Ok(rest) => {
                let props = unsafe { instance.get_physical_device_properties(pd) };
                let name = unsafe {
                    CStr::from_ptr(props.device_name.as_ptr() as *const c_char)
                        .to_string_lossy()
                        .into_owned()
                };
                let mem_props = unsafe { instance.get_physical_device_memory_properties(pd) };
                let mut gpu = Self {
                    _entry: entry,
                    instance,
                    mem_props,
                    device: rest.device,
                    queue: rest.queue,
                    cmd_pool: rest.cmd_pool,
                    cmd: rest.cmd,
                    fence: rest.fence,
                    set_layout: rest.set_layout,
                    pipeline_layout: rest.pipeline_layout,
                    pipelines: rest.pipelines,
                    dummy: rest.dummy,
                    name,
                };
                // The dummy buffer is what a shader's unused bindings point
                // at. It is made here and not in `build_inner` because it
                // needs the memory properties, which need the whole struct.
                gpu.dummy = gpu.make(4)?;
                Ok(gpu)
            }
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                Err(e)
            }
        }
    }

    /// The device, the queue, the command buffer and the one set layout.
    unsafe fn build_inner(
        instance: &ash::Instance,
        pd: vk::PhysicalDevice,
        family: u32,
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

        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..=UNIFORM_BINDING)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(if b == UNIFORM_BINDING {
                        vk::DescriptorType::UNIFORM_BUFFER
                    } else {
                        vk::DescriptorType::STORAGE_BUFFER
                    })
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
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                None,
            )
        }
        .map_err(|e| format!("cannot create a pipeline layout: {e}"))?;

        Ok(GpuRest {
            device,
            queue,
            cmd_pool,
            cmd,
            fence,
            set_layout,
            pipeline_layout,
            pipelines: HashMap::new(),
            dummy: Buf {
                buf: vk::Buffer::null(),
                mem: vk::DeviceMemory::null(),
            },
        })
    }

    /// The device's name, as the driver reports it.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// A buffer of `len` f32, host visible.
    fn make(&self, bytes: u64) -> Result<Buf, String> {
        unsafe {
            let create = vk::BufferCreateInfo::default()
                .size(bytes.max(4))
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
            Ok(Buf { buf, mem })
        }
    }

    /// Write f32 values into a buffer.
    fn write(&self, b: &Buf, values: &[f32]) -> Result<(), String> {
        if values.is_empty() {
            return Ok(());
        }
        unsafe {
            let ptr = self
                .device
                .map_memory(b.mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("cannot map a buffer: {e}"))?;
            std::ptr::copy_nonoverlapping(
                values.as_ptr() as *const u8,
                ptr as *mut u8,
                values.len() * 4,
            );
            self.device.unmap_memory(b.mem);
        }
        Ok(())
    }

    /// Read f32 values back out of a buffer.
    fn read(&self, b: &Buf, len: usize) -> Result<Vec<f32>, String> {
        let mut out = vec![0.0f32; len];
        if len == 0 {
            return Ok(out);
        }
        unsafe {
            let ptr = self
                .device
                .map_memory(b.mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("cannot map a buffer: {e}"))?;
            std::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr() as *mut u8, len * 4);
            self.device.unmap_memory(b.mem);
        }
        Ok(out)
    }

    /// Free a buffer and its memory.
    fn free(&self, b: &Buf) {
        unsafe {
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

    /// Run one shader over `storage` buffers and a 16-byte uniform block.
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
        unsafe {
            let uniform = self.make(16)?;
            let words: [u32; 4] = params;
            let bytes: &[u8] = std::slice::from_raw_parts(words.as_ptr() as *const u8, 16);
            {
                let ptr = self
                    .device
                    .map_memory(uniform.mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .map_err(|e| format!("cannot map the uniform buffer: {e}"))?;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, 16);
                self.device.unmap_memory(uniform.mem);
            }

            let pool = self
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::STORAGE_BUFFER)
                                .descriptor_count(STORAGE_BINDINGS),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                                .descriptor_count(1),
                        ]),
                    None,
                )
                .map_err(|e| format!("cannot create a descriptor pool: {e}"))?;
            let layouts = [self.set_layout];
            let set = self
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
                .map_err(|e| format!("cannot allocate a descriptor set: {e}"))?[0];

            // Two passes: the `DescriptorBufferInfo`s have to be in place
            // (and not reallocated) before the writes point at them.
            let mut infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
            for binding in 0..=UNIFORM_BINDING {
                let buffer = if binding == UNIFORM_BINDING {
                    uniform.buf
                } else {
                    storage
                        .get(binding as usize)
                        .map_or(&self.dummy, |b| *b)
                        .buf
                };
                infos.push(
                    vk::DescriptorBufferInfo::default()
                        .buffer(buffer)
                        .offset(0)
                        .range(vk::WHOLE_SIZE),
                );
            }
            let writes: Vec<vk::WriteDescriptorSet> = (0..=UNIFORM_BINDING)
                .map(|binding| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding)
                        .descriptor_type(if binding == UNIFORM_BINDING {
                            vk::DescriptorType::UNIFORM_BUFFER
                        } else {
                            vk::DescriptorType::STORAGE_BUFFER
                        })
                        .buffer_info(std::slice::from_ref(&infos[binding as usize]))
                })
                .collect();
            self.device.update_descriptor_sets(&writes, &[]);

            self.device
                .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("cannot reset the command buffer: {e}"))?;
            self.device
                .begin_command_buffer(
                    self.cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| format!("cannot begin the command buffer: {e}"))?;
            self.device
                .cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[set],
                &[],
            );
            self.device
                .cmd_dispatch(self.cmd, groups.0, groups.1, groups.2);
            // The next dispatch reads what this one wrote, so the writes have
            // to be available AND visible. Both masks are the compute stage:
            // the host reads the result only after the fence, and a
            // host-coherent buffer needs no separate invalidation.
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            self.device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            self.device
                .end_command_buffer(self.cmd)
                .map_err(|e| format!("cannot end the command buffer: {e}"))?;

            let cmds = [self.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            self.device
                .queue_submit(self.queue, &[submit], self.fence)
                .map_err(|e| format!("cannot submit: {e}"))?;
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| format!("the GPU did not finish: {e}"))?;
            self.device
                .reset_fences(&[self.fence])
                .map_err(|e| format!("cannot reset the fence: {e}"))?;

            self.device.destroy_descriptor_pool(pool, None);
            self.free(&uniform);
        }
        Ok(())
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
        let ab = self.make((a.len() * 4) as u64)?;
        let bb = self.make((b.len() * 4) as u64)?;
        let cb = self.make((m * n * 4) as u64)?;
        self.write(&ab, a)?;
        self.write(&bb, b)?;
        let result = self
            .run(
                "gemm",
                &[&ab, &bb, &cb],
                [m as u32, k as u32, n as u32, 0],
                ((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1),
            )
            .and_then(|()| self.read(&cb, m * n));
        self.free(&ab);
        self.free(&bb);
        self.free(&cb);
        result
    }

    /// GELU, in place.
    pub fn gelu(&mut self, x: &mut [f32]) -> Result<(), String> {
        let xb = self.make((x.len() * 4) as u64)?;
        self.write(&xb, x)?;
        let result = self
            .run(
                "gelu",
                &[&xb],
                [x.len() as u32, 0, 0, 0],
                ((x.len() as u32).div_ceil(64), 1, 1),
            )
            .and_then(|()| self.read(&xb, x.len()));
        self.free(&xb);
        x.copy_from_slice(&result?);
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
        let xb = self.make((x.len() * 4) as u64)?;
        let gb = self.make((gamma.map_or(1, |g| g.len()) * 4) as u64)?;
        let bb = self.make((beta.map_or(1, |b| b.len()) * 4) as u64)?;
        self.write(&xb, x)?;
        self.write(&gb, gamma.unwrap_or(&[0.0]))?;
        self.write(&bb, beta.unwrap_or(&[0.0]))?;
        let result = self
            .run(
                "layernorm",
                &[&xb, &gb, &bb],
                [
                    width as u32,
                    rows as u32,
                    u32::from(gamma.is_some()),
                    u32::from(beta.is_some()),
                ],
                (rows as u32, 1, 1),
            )
            .and_then(|()| self.read(&xb, x.len()));
        self.free(&xb);
        self.free(&gb);
        self.free(&bb);
        x.copy_from_slice(&result?);
        Ok(())
    }

    /// Softmax over rows of `width`.
    pub fn softmax(&mut self, x: &mut [f32], rows: usize, width: usize) -> Result<(), String> {
        let xb = self.make((x.len() * 4) as u64)?;
        self.write(&xb, x)?;
        let result = self
            .run(
                "softmax",
                &[&xb],
                [width as u32, rows as u32, 0, 0],
                (rows as u32, 1, 1),
            )
            .and_then(|()| self.read(&xb, x.len()));
        self.free(&xb);
        x.copy_from_slice(&result?);
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
        let q_len = q.len() / d;
        let kv_len = k.len() / d;
        let qb = self.make((q.len() * 4) as u64)?;
        let kb = self.make((k.len() * 4) as u64)?;
        let vb = self.make((v.len() * 4) as u64)?;
        let ob = self.make((q.len() * 4) as u64)?;
        self.write(&qb, q)?;
        self.write(&kb, k)?;
        self.write(&vb, v)?;
        let result = self
            .run(
                "attention",
                &[&qb, &kb, &vb, &ob],
                [q_len as u32, kv_len as u32, d as u32, 0],
                ((q_len as u32).div_ceil(64), 1, 1),
            )
            .and_then(|()| self.read(&ob, q.len()));
        self.free(&qb);
        self.free(&kb);
        self.free(&vb);
        self.free(&ob);
        result
    }

    /// `z = x + y`.
    pub fn add(&mut self, x: &[f32], y: &[f32]) -> Result<Vec<f32>, String> {
        let xb = self.make((x.len() * 4) as u64)?;
        let yb = self.make((y.len() * 4) as u64)?;
        let zb = self.make((x.len() * 4) as u64)?;
        self.write(&xb, x)?;
        self.write(&yb, y)?;
        let result = self
            .run(
                "add",
                &[&xb, &yb, &zb],
                [x.len() as u32, 0, 0, 0],
                ((x.len() as u32).div_ceil(64), 1, 1),
            )
            .and_then(|()| self.read(&zb, x.len()));
        self.free(&xb);
        self.free(&yb);
        self.free(&zb);
        result
    }
}

/// The handles `Gpu` owns once its instance is in place.
struct GpuRest {
    device: ash::Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipelines: HashMap<&'static str, vk::Pipeline>,
    dummy: Buf,
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            for (_, p) in self.pipelines.drain() {
                self.device.destroy_pipeline(p, None);
            }
            self.free(&self.dummy);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
