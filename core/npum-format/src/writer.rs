//! The container writer.
//!
//! ONE writer. The format used to have two -- this crate and a Python packer
//! -- and the risk of two writers of one binary layout is not a theoretical
//! one: a disagreement between them means correctly-sized weights in the
//! wrong order, which no size check catches. The Python packer now calls this
//! crate through the cdylib; there is no second implementation of the layout.
//!
//! # Why tensors are staged
//!
//! The directory carries every offset, and the directory is written before
//! the first data byte. So the payloads cannot be streamed as they are
//! produced -- they are accumulated and written at the end. A 2.5 GB
//! container therefore needs its payload resident once, which is the price of
//! a format whose index comes first.

use crate::directory::{Config, Directory, TensorEntry};
use crate::header::{ContainerFormat, HEADER_SIZE, Header, VERSION};
use crate::layout::BLayout;
use std::path::Path;

/// The alignment every tensor's payload starts at.
pub const ALIGN: usize = 4096;

/// The role string a design set's files carry.
pub const DESIGN_ROLE: &str = "design";

/// The prefix a design set's file names carry.
///
/// `design/<set>/<file>`. The set name is part of the name rather than a
/// separate index: a self-sufficient container carries gemm_rtp AND gelu AND
/// layernorm AND softmax, the runtime picks between them by name and by
/// datapath, and a flat `final.xclbin` in a container that holds four sets
/// would be four files of one name.
pub const DESIGN_PREFIX: &str = "design/";

/// One tensor to stage.
pub struct Staged {
    /// The name.
    pub name: String,
    /// The role.
    pub role: String,
    /// The dtype tag.
    pub dtype: String,
    /// The logical shape.
    pub logical_shape: Vec<i64>,
    /// The padded shape.
    pub padded_shape: Vec<i64>,
    /// The bytes.
    pub blob: Vec<u8>,
    /// The layout descriptor, for a pre-tiled panel.
    pub layout: Option<BLayout>,
}

/// Accumulates tensors, then writes the container.
pub struct Writer {
    config: Config,
    arch: u32,
    flags: u32,
    entries: Vec<TensorEntry>,
    blobs: Vec<Vec<u8>>,
    offset: u64,
}

impl Writer {
    /// A writer for one container.
    ///
    /// `flags` carries `FLAG_PRETILED` -- bit 0 -- when the B panels are
    /// pre-tiled for the array. A container whose panels are plain row-major
    /// f32 does not set it, and the loader uses the flag to know which
    /// untile, if any, a panel needs.
    pub fn new(config: Config, arch: u32, flags: u32) -> Self {
        Self {
            config,
            arch,
            flags,
            entries: Vec::new(),
            blobs: Vec::new(),
            offset: 0,
        }
    }

    /// The config block.
    pub fn config_mut(&mut self) -> &mut Config {
        &mut self.config
    }

    /// Stage one tensor.
    pub fn add(&mut self, staged: Staged) -> &TensorEntry {
        let nbytes = staged.blob.len() as u64;
        let entry = TensorEntry {
            name: staged.name,
            role: staged.role,
            dtype: staged.dtype,
            logical_shape: staged.logical_shape,
            padded_shape: staged.padded_shape,
            offset: self.offset,
            nbytes,
            layout: staged
                .layout
                .as_ref()
                .map(|l| serde_json::to_value(l).expect("a layout is always serialisable")),
            layout_hash: staged.layout.as_ref().map(crate::layout::layout_hash),
        };
        self.entries.push(entry);
        self.blobs.push(staged.blob);
        // Pad AFTER each tensor so the next one starts aligned.
        self.offset += nbytes;
        let pad = (ALIGN as u64 - (self.offset % ALIGN as u64)) % ALIGN as u64;
        self.offset += pad;
        self.blobs.push(vec![0u8; pad as usize]);
        self.entries.last().expect("just pushed")
    }

    /// Stage one file of a compiled design set, as raw bytes.
    ///
    /// An xclbin is an ELF-ish container and an instruction stream is a flat
    /// array of 32-bit words; neither is an array of numbers, and there is no
    /// arithmetic over these bytes. It is stored as U8 with the byte count as
    /// the only shape, because pretending otherwise would invite someone to
    /// reshape one.
    pub fn add_design(&mut self, set_name: &str, file_name: &str, blob: Vec<u8>) -> &TensorEntry {
        let name = format!("{DESIGN_PREFIX}{set_name}/{file_name}");
        let nbytes = blob.len() as i64;
        self.add(Staged {
            name,
            role: DESIGN_ROLE.into(),
            dtype: "U8".into(),
            logical_shape: vec![nbytes],
            padded_shape: vec![nbytes],
            blob,
            layout: None,
        })
    }

    /// The directory this writer would write.
    pub fn directory(&self) -> Directory {
        Directory {
            config: self.config.clone(),
            tensors: self.entries.clone(),
        }
    }

    /// Write the container.
    ///
    /// The bytes are the ones the Python packer writes, and the two are
    /// compared byte for byte by the parity gate -- the Rust writer is not
    /// "equivalent" to the Python one, it is the same bytes.
    pub fn write(&self, path: &Path) -> std::io::Result<()> {
        let directory = self.directory();
        let js = serde_json::to_vec(&directory)?;

        let json_offset = HEADER_SIZE as u64;
        let data_offset = json_offset + js.len() as u64;
        let data_offset =
            data_offset + ((ALIGN as u64 - (data_offset % ALIGN as u64)) % ALIGN as u64);

        let header = Header {
            format: ContainerFormat::Current,
            version: VERSION,
            arch: self.arch,
            flags: self.flags,
            json_offset,
            json_length: js.len() as u64,
            data_offset,
            data_length: self.offset,
        };

        let mut out = Vec::with_capacity((data_offset + self.offset) as usize);
        out.extend_from_slice(&header.to_bytes());
        out.extend_from_slice(&js);
        out.resize(data_offset as usize, 0u8);
        for blob in &self.blobs {
            out.extend_from_slice(blob);
        }
        std::fs::write(path, out)
    }
}
