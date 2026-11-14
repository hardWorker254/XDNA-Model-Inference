//! A design set, loaded onto the board.
//!
//! # WHAT A LOADED DESIGN IS
//!
//! One xclbin, its instruction streams, and its data buffers -- held by the
//! runtime for as long as the model runs. The C++ runtime called this object
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
//! # THE DATA BUFFERS ARE ZEROED
//!
//! A fresh buffer holds whatever the driver gave it. The C buffer is read
//! back after a dispatch, and a region the design did not write would then
//! read as stale bytes -- which looks exactly like "the design did not
//! compute this tile". Zeroing costs one pass over the buffer and makes
//! "the element is still zero" mean what it says. The Python design harness
//! does the same thing, for the same reason.

use std::fmt;
use std::path::{Path, PathBuf};

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
    /// memory budget is not that large. The C++ runtime has the same slots,
    /// and the same rule that a dispatch uses the slot it was bound to.
    slots: Vec<Vec<T::Buffer>>,
    /// Which slot each data buffer is bound to.
    bound: Vec<usize>,
    /// Where it was loaded from, for the messages.
    dir: PathBuf,
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

        if info.buffers.len() < 2 {
            return Err(ArrayError::Design(format!(
                "{}: fewer than two data buffers",
                dir.display()
            )));
        }

        let xclbin = std::fs::read(dir.join("final.xclbin")).map_err(|e| {
            ArrayError::Design(format!("cannot read {}/final.xclbin: {e}", dir.display()))
        })?;

        let mut transport = transport;
        transport.open_context(&xclbin)?;

        // THE KERNEL BY NAME, AND THE NAME IS THE MANIFEST'S.
        //
        // The AIE flow's kernels are all called `MLIR_AIE...` and an xclbin
        // may carry several; the manifest names the one the design wants, and
        // the C++ runtime looked for the first `MLIR_AIE` prefix match. The
        // manifest's name is used when it has one.
        transport.open_kernel(info.kernel_name())?;

        // THE INSTRUCTION STREAMS, IN SLOT ORDER.
        let mut instr = Vec::new();
        let mut instr_words = Vec::new();
        for stream in info.streams_in_slot_order() {
            let bytes = std::fs::read(dir.join(&stream.file)).map_err(|e| {
                ArrayError::Design(format!(
                    "cannot read {}/{}: {e}",
                    dir.display(),
                    stream.file
                ))
            })?;
            let group = transport.group_id(GROUP_INSTRUCTION)?;
            let buffer = transport.alloc(
                bytes.len() as u64,
                BufferFlags::Cacheable,
                group,
            )?;
            T::write(&buffer, &bytes, 0)?;
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
            dir: dir.to_path_buf(),
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
        let slots = self.slots.get(index).ok_or_else(|| {
            ArrayError::Design(format!(
                "{}: no data buffer {index}",
                self.dir.display()
            ))
        })?;
        if slot >= slots.len() {
            return Err(ArrayError::Design(format!(
                "{}: data buffer {index} has no slot {slot}",
                self.dir.display()
            )));
        }
        self.bound[index] = slot;
        Ok(())
    }

    /// Put `data` into the data buffer `index` and make it visible to the
    /// device.
    ///
    /// The size is checked against the buffer's own size: a design computes
    /// a fixed shape, and staging fewer bytes than it will read is a stale
    /// tail, not a shorter answer.
    pub fn stage(&mut self, index: usize, data: &[u8]) -> Result<usize, ArrayError> {
        let capacity = self.info.buffers.get(index).copied().ok_or_else(|| {
            ArrayError::Design(format!(
                "{}: no data buffer {index}",
                self.dir.display()
            ))
        })?;
        if data.len() as u64 > capacity {
            return Err(ArrayError::Design(format!(
                "{}: {} bytes for data buffer {index}, and the design allows {capacity}",
                self.dir.display(),
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

    /// Write into the BOUND slot of a data buffer and make it visible to
    /// the device.
    ///
    /// The bound slot and not a new one: the A operand is written into the
    /// design's own A buffer, every time, and a slot per operand would leave
    /// the design's memory budget one operand short per dispatch.
    pub fn write_bound(&mut self, index: usize, data: &[u8]) -> Result<(), ArrayError> {
        let capacity = self.info.buffers.get(index).copied().ok_or_else(|| {
            ArrayError::Design(format!(
                "{}: no data buffer {index}",
                self.dir.display()
            ))
        })?;
        if data.len() as u64 > capacity {
            return Err(ArrayError::Design(format!(
                "{}: {} bytes for data buffer {index}, and the design allows {capacity}",
                self.dir.display(),
                data.len()
            )));
        }
        let slot = self.bound[index];
        let buffer = &self.slots[index][slot];
        T::write(buffer, data, 0)?;
        T::sync_to_device(buffer, data.len() as u64, 0)?;
        Ok(())
    }

    /// Run the instruction stream in `slot` and wait for it.
    ///
    /// A run that did not complete is an error and not a number: the design
    /// may have written half an output, and half an output read back is a
    /// plausible tensor.
    pub fn dispatch(&mut self, slot: usize) -> Result<(), ArrayError> {
        let instr = self.instr.get(slot).ok_or_else(|| {
            ArrayError::Design(format!(
                "{}: no instruction slot {slot}",
                self.dir.display()
            ))
        })?;
        let args: Vec<&T::Buffer> = self
            .slots
            .iter()
            .enumerate()
            .map(|(index, slots)| &slots[self.bound[index]])
            .collect();
        let words = self.instr_words[slot];
        let run = self.transport.start(instr, words, &args)?;
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
                self.dir.display(),
                out.len()
            )));
        }
        T::sync_from_device(buffer, out.len() as u64, 0)?;
        T::read(buffer, out, 0)?;
        Ok(())
    }
}
