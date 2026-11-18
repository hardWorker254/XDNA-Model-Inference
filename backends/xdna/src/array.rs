//! A design set, loaded onto the board.
//!
//! # WHAT A LOADED DESIGN IS
//!
//! One xclbin, its instruction streams, and its data buffers -- held by the
//! runtime for as long as the model runs. The reference runtime called this object
//! `npu::Design` and it is the same thing here: a resident design with the
//! buffers that belong to it.
//!
//! # THE GROUP IDS COME FROM THE KERNEL
//!
//! A buffer is not "for the kernel"; it is for a GROUP, and the group decides
//! which memory bank it lands in. The instruction stream is group 1 and the
//! data arguments are 3, 4, 5 -- the numbers the design's own kernel
//! signature uses -- and they are asked of the kernel rather than written
//! down here, because a hard-coded group id that disagrees with the design
//! is a dispatch that fails on a board it should have worked on.
//!
//! # THE DATA BUFFERS ARE ZEROED -- EXCEPT THE ONES THAT ARE FILLED FIRST
//!
//! A fresh buffer holds whatever the driver gave it. The C buffer is read
//! back after a dispatch, and a region the design did not write would then
//! read as stale bytes -- which looks exactly like "the design did not
//! compute this tile". So the design's own buffers are zeroed, as in the
//! Python design harness, for the same reason.
//!
//! The extra ping-pong slots are different. An A slot is written in full
//! before a dispatch reads it, and a C slot is read back only over the rows
//! the dispatch wrote; nothing observes the driver's initial bytes. Those
//! slots are allocated WITHOUT a zero-fill and without a sync -- see
//! [`ArrayDesign::add_slot_uninit`]. A caller that binds such a slot before
//! filling it has a bug, and the protocols that use them (the GEMM batch, the
//! convolution's chunk pipeline, the attention's panels) fill every slot
//! before its first dispatch.

use std::fmt;
use std::path::Path;

use npum_format::Container;

use crate::manifest::DesignInfo;
use crate::transport::{BufferFlags, Transport, TransportError};

/// The group the instruction streams live in.
const GROUP_INSTRUCTION: i32 = 1;
/// The group the first data argument lives in.
const GROUP_DATA: i32 = 3;

/// What went wrong while loading or running a design.
#[derive(Debug)]
pub enum ArrayError {
    /// The design set could not be read.
    Design(String),
    /// The board refused.
    Transport(TransportError),
}

impl fmt::Display for ArrayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArrayError::Design(message) => f.write_str(message),
            ArrayError::Transport(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ArrayError {}

impl From<TransportError> for ArrayError {
    fn from(error: TransportError) -> Self {
        ArrayError::Transport(error)
    }
}

/// One tensor's bytes in a container, borrowed rather than copied.
///
/// The container owns the whole file; every blob the design needs is a slice
/// of it. `load` only writes the bytes into buffers, so the slices only have
/// to live for that call -- the container is still open when it is made.
fn container_bytes<'a>(
    file: &'a Container,
    key: &str,
    container: &Path,
) -> Result<&'a [u8], ArrayError> {
    file.tensor_bytes(key)
        .map_err(|e| ArrayError::Design(format!("{container:?} has no {key}: {e}")))
}

/// A design set, loaded onto the board.
pub struct ArrayDesign<T: Transport> {
    transport: T,
    info: DesignInfo,
    /// The instruction streams, in slot order.
    instr: Vec<T::Buffer>,
    /// Each instruction stream's length in 32-bit words.
    ///
    /// The length is in WORDS and not in bytes because that is what the
    /// dispatch takes, and a byte count divided by four at the call site is
    /// a division somebody has to remember to write.
    instr_words: Vec<u32>,
    /// The data buffers' SLOTS.
    ///
    /// The first slot of each buffer is the design's own; the rest are what
    /// `stage` adds. The reason there are slots at all is that each
    /// convolution has its OWN weight panel, and the panels are big: one
    /// buffer per panel would be a buffer per convolution, and the design's
    /// memory budget is not that large. The reference runtime has the same slots,
    /// and the same rule that a dispatch uses the slot it was bound to.
    slots: Vec<Vec<T::Buffer>>,
    /// Which slot each data buffer is bound to.
    bound: Vec<usize>,
    /// Where it was loaded from, for the messages.
    label: String,
}

impl<T: Transport> ArrayDesign<T> {
    /// Load a design set from a directory.
    ///
    /// The transport is taken, not opened: the caller has already decided
    /// which board this is, and a design set loaded onto a different board
    /// than the one it was compiled for is a wrong answer rather than an
    /// error, so the choice belongs to the caller.
    pub fn open(dir: &Path, transport: T) -> Result<Self, ArrayError> {
        let info = DesignInfo::read(dir).map_err(ArrayError::Design)?;
        let xclbin = std::fs::read(dir.join("final.xclbin")).map_err(|e| {
            ArrayError::Design(format!("cannot read {}/final.xclbin: {e}", dir.display()))
        })?;
        let mut blobs = Vec::new();
        for stream in info.streams_in_slot_order() {
            let bytes = std::fs::read(dir.join(&stream.file)).map_err(|e| {
                ArrayError::Design(format!(
                    "cannot read {}/{}: {e}",
                    dir.display(),
                    stream.file
                ))
            })?;
            blobs.push(bytes);
        }
        Self::load(
            transport,
            info,
            &xclbin,
            blobs.iter().map(Vec::as_slice),
            dir.display().to_string(),
        )
    }

    /// Load a design set out of a container.
    ///
    /// A container can carry its own compiled design set, and a container that
    /// does needs nothing beside it: no directory, no MLIR-AIE on the reader's
    /// machine, no artifacts to install. That is why the packer embeds the
    /// set by default, and why the set's files live in the container under
    /// `design/<set>/`.
    ///
    /// THE CONTAINER IS THE PREFERRED SOURCE, and the directory is the
    /// fallback -- not the other way round, and not "preferred" in the sense
    /// of "changes nothing for anyone who has both": every existing layout,
    /// every gate and every refusal behaves exactly as it did.
    pub fn open_from_container(
        container: &Path,
        set: &str,
        transport: T,
    ) -> Result<Self, ArrayError> {
        let file = Container::open(container)
            .map_err(|e| ArrayError::Design(format!("cannot open {}: {e}", container.display())))?;
        let prefix = format!("design/{set}");
        // THE BLOBS ARE SLICES OF THE CONTAINER, NOT HOST COPIES.
        //
        // `load` only writes them into buffers before it returns; it never
        // keeps the bytes. The container stays open until that call is done,
        // so a `to_vec` per stream would be a copy of a large xclbin and of
        // every instruction stream for nothing.
        let key = format!("{prefix}/design.json");
        let manifest = container_bytes(&file, &key, container)?;
        let info: DesignInfo = serde_json::from_slice(manifest).map_err(|e| {
            ArrayError::Design(format!(
                "cannot parse the container's {prefix}/design.json: {e}"
            ))
        })?;
        let key = format!("{prefix}/final.xclbin");
        let xclbin = container_bytes(&file, &key, container)?;
        let mut blobs = Vec::new();
        for stream in info.streams_in_slot_order() {
            let key = format!("{prefix}/{}", stream.file);
            blobs.push(container_bytes(&file, &key, container)?);
        }
        Self::load(
            transport,
            info,
            xclbin,
            blobs,
            format!("the container's {prefix}"),
        )
    }

    /// The one body both constructors go through.
    ///
    /// Two constructors and one body, so the buffer-object dance that follows
    /// cannot be got right in one spelling and wrong in the other.
    fn load<'a>(
        transport: T,
        info: DesignInfo,
        xclbin: &[u8],
        instr_blobs: impl IntoIterator<Item = &'a [u8]>,
        label: String,
    ) -> Result<Self, ArrayError> {
        if info.buffers.len() < 2 {
            return Err(ArrayError::Design(format!(
                "{label}: fewer than two data buffers"
            )));
        }

        let mut transport = transport;
        transport.open_context(xclbin)?;

        // THE KERNEL BY NAME, AND THE NAME IS THE MANIFEST'S.
        //
        // The AIE flow's kernels are all called `MLIR_AIE...` and an xclbin
        // may carry several; the manifest names the one the design wants, and
        // the reference runtime looked for the first `MLIR_AIE` prefix match. The
        // manifest's name is used when it has one.
        transport.open_kernel(info.kernel_name())?;

        // THE INSTRUCTION STREAMS, IN SLOT ORDER.
        let mut instr = Vec::new();
        let mut instr_words = Vec::new();
        for bytes in instr_blobs {
            let group = transport.group_id(GROUP_INSTRUCTION)?;
            let buffer = transport.alloc(bytes.len() as u64, BufferFlags::Cacheable, group)?;
            T::write(&buffer, bytes, 0)?;
            T::sync_to_device(&buffer, bytes.len() as u64, 0)?;
            instr_words.push((bytes.len() / 4) as u32);
            instr.push(buffer);
        }

        // THE DATA BUFFERS, ZEROED.
        let mut slots: Vec<Vec<T::Buffer>> = Vec::new();
        let mut bound: Vec<usize> = Vec::new();
        for (index, bytes) in info.buffers.iter().enumerate() {
            let group = transport.group_id(GROUP_DATA + index as i32)?;
            let buffer = transport.alloc(*bytes, BufferFlags::HostOnly, group)?;
            let zeros = vec![0u8; *bytes as usize];
            T::write(&buffer, &zeros, 0)?;
            T::sync_to_device(&buffer, *bytes, 0)?;
            slots.push(vec![buffer]);
            bound.push(0);
        }

        Ok(Self {
            transport,
            info,
            instr,
            instr_words,
            slots,
            bound,
            label,
        })
    }

    /// The manifest this design was loaded from.
    pub fn info(&self) -> &DesignInfo {
        &self.info
    }

    /// The number of instruction slots.
    pub fn slots(&self) -> usize {
        self.instr.len()
    }

    /// The number of data buffers.
    pub fn buffers(&self) -> usize {
        self.slots.len()
    }

    /// The number of staged slots of one data buffer.
    pub fn slots_of(&self, index: usize) -> usize {
        self.slots.get(index).map(|s| s.len()).unwrap_or(0)
    }

    /// Bind a data buffer to one of its staged slots.
    ///
    /// The dispatch uses the slot each buffer is bound to, so a weight panel
    /// is staged once and bound before the call that uses it.
    pub fn bind(&mut self, index: usize, slot: usize) -> Result<(), ArrayError> {
        let slots = self
            .slots
            .get(index)
            .ok_or_else(|| ArrayError::Design(format!("{}: no data buffer {index}", self.label)))?;
        if slot >= slots.len() {
            return Err(ArrayError::Design(format!(
                "{}: data buffer {index} has no slot {slot}",
                self.label
            )));
        }
        self.bound[index] = slot;
        Ok(())
    }

    /// The slot the data buffer `index` is currently bound to.
    ///
    /// A dispatch captures all its bindings at `start_dispatch`, so a caller
    /// that pipelines GEMMs needs to remember which C slot a submitted run
    /// wrote into: the binding is rebound to the NEXT dispatch's slot before
    /// this one's output is read back. Reading the binding at submit time is
    /// how that slot is remembered without the caller having to thread it
    /// through every layer.
    pub fn bound(&self, index: usize) -> Result<usize, ArrayError> {
        self.bound
            .get(index)
            .copied()
            .ok_or_else(|| ArrayError::Design(format!("{}: no data buffer {index}", self.label)))
    }

    /// Put `data` into the data buffer `index` and make it visible to the
    /// device.
    ///
    /// The size is checked against the buffer's own size: a design computes
    /// a fixed shape, and staging fewer bytes than it will read is a stale
    /// tail, not a shorter answer.
    pub fn stage(&mut self, index: usize, data: &[u8]) -> Result<usize, ArrayError> {
        let capacity =
            self.info.buffers.get(index).copied().ok_or_else(|| {
                ArrayError::Design(format!("{}: no data buffer {index}", self.label))
            })?;
        if data.len() as u64 > capacity {
            return Err(ArrayError::Design(format!(
                "{}: {} bytes for data buffer {index}, and the design allows {capacity}",
                self.label,
                data.len()
            )));
        }
        let group = self.transport.group_id(GROUP_DATA + index as i32)?;
        let buffer = self
            .transport
            .alloc(data.len() as u64, BufferFlags::HostOnly, group)?;
        T::write(&buffer, data, 0)?;
        T::sync_to_device(&buffer, data.len() as u64, 0)?;
        let slot = self.slots[index].len();
        self.slots[index].push(buffer);
        Ok(slot)
    }

    /// Allocate an empty slot WITHOUT zeroing or synchronising it.
    ///
    /// For a slot whose whole used range is written with
    /// [`Self::write_bound`] / [`Self::write_bound_unsynced`] before the first
    /// dispatch reads it. The attention's K and V panels are staged this way:
    /// both ping-pong halves are filled before their first QK^T/softmax.V,
    /// so a zero pass over each panel at load would be work thrown away.
    ///
    /// Binding an uninitialised slot and dispatching against it is a bug: the
    /// driver does not promise that freshly allocated memory is zero.
    pub fn stage_uninit(&mut self, index: usize, bytes: u64) -> Result<usize, ArrayError> {
        let capacity =
            self.info.buffers.get(index).copied().ok_or_else(|| {
                ArrayError::Design(format!("{}: no data buffer {index}", self.label))
            })?;
        if bytes > capacity {
            return Err(ArrayError::Design(format!(
                "{}: {bytes} bytes for data buffer {index}, and the design allows {capacity}",
                self.label
            )));
        }
        let group = self.transport.group_id(GROUP_DATA + index as i32)?;
        let buffer = self.transport.alloc(bytes, BufferFlags::HostOnly, group)?;
        let slot = self.slots[index].len();
        self.slots[index].push(buffer);
        Ok(slot)
    }

    /// Write into the BOUND slot of a data buffer and make it visible to
    /// the device.
    ///
    /// The bound slot and not a new one: the A operand is written into the
    /// design's own A buffer, every time, and a slot per operand would leave
    /// the design's memory budget one operand short per dispatch.
    pub fn write_bound(&mut self, index: usize, data: &[u8]) -> Result<(), ArrayError> {
        self.write_bound_unsynced(index, data)?;
        self.sync_to_device_bound(index, data.len() as u64)
    }

    /// Write into the BOUND slot of a data buffer without synchronising it to
    /// the device.
    ///
    /// The caller is responsible for making the write visible with
    /// [`Self::sync_to_device_bound`] before the device reads it. Splitting the
    /// write from the sync lets a caller overlap host-side work with the
    /// transfer when the transport can do so.
    pub fn write_bound_unsynced(&mut self, index: usize, data: &[u8]) -> Result<(), ArrayError> {
        let capacity =
            self.info.buffers.get(index).copied().ok_or_else(|| {
                ArrayError::Design(format!("{}: no data buffer {index}", self.label))
            })?;
        if data.len() as u64 > capacity {
            return Err(ArrayError::Design(format!(
                "{}: {} bytes for data buffer {index}, and the design allows {capacity}",
                self.label,
                data.len()
            )));
        }
        let slot = self.bound[index];
        let buffer = &self.slots[index][slot];
        T::write(buffer, data, 0)?;
        Ok(())
    }

    /// Make the BOUND slot of a data buffer's contents visible to the device.
    pub fn sync_to_device_bound(&self, index: usize, bytes: u64) -> Result<(), ArrayError> {
        let buffer = &self.slots[index][self.bound[index]];
        let capacity = T::size(buffer);
        if bytes > capacity {
            return Err(ArrayError::Design(format!(
                "{}: sync_to_device {bytes} bytes of data buffer {index}, and the design allows {capacity}",
                self.label
            )));
        }
        T::sync_to_device(buffer, bytes, 0)?;
        Ok(())
    }

    /// Allocate an additional empty slot for a data buffer and return its slot
    /// index.
    ///
    /// The slot is zeroed and synchronised to the device, so it is safe to bind
    /// immediately. Extra slots enable ping-pong buffering: one slot can be in
    /// use by an in-flight dispatch while another is prepared for the next one.
    pub fn add_slot(&mut self, index: usize) -> Result<usize, ArrayError> {
        let slot = self.add_slot_uninit(index)?;
        // The design's own slot was zeroed at load for a reason: a region the
        // device does not write reads back as stale bytes. Keep that promise
        // for a caller that uses the general API and binds the slot before
        // filling it. The pipelined callers use `add_slot_uninit` instead.
        let capacity = self.info.buffers[index];
        let zeros = vec![0u8; capacity as usize];
        T::write(&self.slots[index][slot], &zeros, 0)?;
        T::sync_to_device(&self.slots[index][slot], capacity, 0)?;
        Ok(slot)
    }

    /// Allocate an extra slot WITHOUT zeroing or synchronising it.
    ///
    /// This is the slot for a protocol that guarantees a full write before
    /// the device reads it (A) and before the host reads it back (C). The GEMM
    /// batch and the convolution's chunk pipeline use it for their ping-pong
    /// pairs: every A slot is written in full with `write_bound`, and every C
    /// slot is read back only over the rows the device wrote. A zero pass
    /// over the whole capacity of those slots is work nothing ever observes.
    ///
    /// The driver does not promise that fresh memory is zero. Do not bind an
    /// uninitialised slot to a dispatch that may read or read back a region
    /// the protocol has not written.
    pub fn add_slot_uninit(&mut self, index: usize) -> Result<usize, ArrayError> {
        let capacity =
            self.info.buffers.get(index).copied().ok_or_else(|| {
                ArrayError::Design(format!("{}: no data buffer {index}", self.label))
            })?;
        let group = self.transport.group_id(GROUP_DATA + index as i32)?;
        let buffer = self
            .transport
            .alloc(capacity, BufferFlags::HostOnly, group)?;
        let slot = self.slots[index].len();
        self.slots[index].push(buffer);
        Ok(slot)
    }

    /// Run the instruction stream in `slot` and wait for it.
    ///
    /// A run that did not complete is an error and not a number: the design
    /// may have written half an output, and half an output read back is a
    /// plausible tensor.
    pub fn dispatch(&mut self, slot: usize) -> Result<(), ArrayError> {
        let run = self.start_dispatch(slot)?;
        Self::wait_dispatch(run)
    }

    /// Start the instruction stream in `slot` without waiting for it.
    ///
    /// The returned [`Run`] must be waited on with [`Self::wait_dispatch`]
    /// before any buffer written by the dispatch is read or reused for another
    /// dispatch. The data-buffer bindings at the time of this call are the ones
    /// the dispatch uses.
    pub fn start_dispatch(&mut self, slot: usize) -> Result<crate::transport::Run, ArrayError> {
        let instr = self.instr.get(slot).ok_or_else(|| {
            ArrayError::Design(format!("{}: no instruction slot {slot}", self.label))
        })?;
        let args: Vec<&T::Buffer> = self
            .slots
            .iter()
            .enumerate()
            .map(|(index, slots)| &slots[self.bound[index]])
            .collect();
        let words = self.instr_words[slot];
        let run = self.transport.start(instr, words, &args)?;
        Ok(run)
    }

    /// Wait for a previously started dispatch to finish.
    pub fn wait_dispatch(run: crate::transport::Run) -> Result<(), ArrayError> {
        T::wait(&run)?;
        Ok(())
    }

    /// Read the data buffer `index` back.
    ///
    /// The read is preceded by the device-to-host sync: a buffer that the
    /// device has written and the host has not read back holds whatever the
    /// host last put there.
    pub fn read(&self, index: usize, out: &mut [u8]) -> Result<(), ArrayError> {
        let buffer = &self.slots[index][self.bound[index]];
        let capacity = T::size(buffer);
        if out.len() as u64 > capacity {
            return Err(ArrayError::Design(format!(
                "{}: {} bytes out of data buffer {index}, and the design allows {capacity}",
                self.label,
                out.len()
            )));
        }
        T::sync_from_device(buffer, out.len() as u64, 0)?;
        T::read(buffer, out, 0)?;
        Ok(())
    }

    /// Make the device's writes to the BOUND slot of a data buffer visible to
    /// the host, without copying them out.
    pub fn sync_from_device_bound(&self, index: usize, bytes: u64) -> Result<(), ArrayError> {
        let buffer = &self.slots[index][self.bound[index]];
        let capacity = T::size(buffer);
        if bytes > capacity {
            return Err(ArrayError::Design(format!(
                "{}: sync_from_device {bytes} bytes of data buffer {index}, and the design allows {capacity}",
                self.label
            )));
        }
        T::sync_from_device(buffer, bytes, 0)?;
        Ok(())
    }

    /// Read from the BOUND slot of a data buffer without synchronising it from
    /// the device first.
    ///
    /// The caller must have made the device's writes visible with
    /// [`Self::sync_from_device_bound`] (or [`Self::read`]) before calling this.
    pub fn read_bound_unsynced(&self, index: usize, out: &mut [u8]) -> Result<(), ArrayError> {
        let buffer = &self.slots[index][self.bound[index]];
        let capacity = T::size(buffer);
        if out.len() as u64 > capacity {
            return Err(ArrayError::Design(format!(
                "{}: {} bytes out of data buffer {index}, and the design allows {capacity}",
                self.label,
                out.len()
            )));
        }
        T::read(buffer, out, 0)?;
        Ok(())
    }
}
