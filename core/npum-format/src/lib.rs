//! The `.npum` container -- NPU Model.
//!
//! One file holds everything a model needs to run: its weights, the geometry
//! the runtime needs, the tokenizer's tables, and optionally a compiled design
//! set so that the container is self-sufficient.
//!
//! # Two magics, one format
//!
//! The format used to be `.npue`, which stood for "NPU embedding" and stopped
//! being true when the runtime grew an audio, a vision, a pose and an OCR
//! family. The layout did not change when the name did -- only the four magic
//! bytes did -- so:
//!
//! | magic | what |
//! |---|---|
//! | `NPUM` | the current name; the only thing this crate writes |
//! | `NPUE` | the old name; read for compatibility, never written |
//!
//! The reader accepts both. The writer writes only `NPUM`. A reader that
//! refused the old magic would refuse every container in every tree that has
//! not been repacked yet, and a repack is a 19-models-times-4-variants
//! operation that buys nothing: the bytes are the same bytes.
//!
//! # The layout
//!
//! ```text
//! +------------------+ 0
//! | header, 64 bytes |
//! +------------------+ json_offset
//! | directory JSON   |
//! +------------------+ data_offset   (4096-aligned)
//! | tensor payloads |
//! +------------------+ data_offset + data_length
//! ```
//!
//! The directory is the index: it names every tensor, its dtype, its logical
//! and padded shapes, and its byte offset. The data region is the payloads,
//! each padded up to a 4096-byte boundary so that every tensor starts aligned.
//!
//! # Why the padding is there
//!
//! A B panel for the array is read with wide vector loads. A tensor that did
//! not start aligned would still be the right SIZE and the right SHAPE and
//! would still pass every check that reads the directory -- and the array
//! would read it shifted. The padding is not an optimisation; it is what makes
//! a wrong offset impossible to mistake for a right one.

pub mod digest;
pub mod directory;
pub mod header;
pub mod layout;
pub mod tiling;
pub mod writer;

pub use directory::{Directory, TensorEntry};
pub use digest::{model_digest, sha256_bytes, sha256_file};
pub use header::{ContainerFormat, Header, MAGIC_LEGACY, MAGIC_CURRENT, VERSION};
pub use layout::{gemm_b_layout, layout_hash, BLayout, MacGeom};
pub use writer::Writer;

use std::path::Path;

/// Everything that can go wrong reading or writing a container.
///
/// The variants are named after the FAILURE, not after the function that hit
/// it. A reader that cannot open the file and a reader that finds the wrong
/// magic both mean "this is not a container", but they are different facts and
/// a caller may want to tell them apart -- one is a missing file, the other is
/// a file that is not what was expected.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path}: not a container (magic {magic:?})")]
    NotAContainer { path: String, magic: [u8; 4] },

    #[error("{path}: container version {found}, this build reads version {expected}")]
    VersionMismatch {
        path: String,
        found: u32,
        expected: u32,
    },

    #[error("{path}: {what}")]
    Io {
        path: String,
        what: String,
        #[source]
        source: std::io::Error,
    },

    #[error("{path}: the directory is not valid JSON: {source}")]
    BadDirectory {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("{path}: the directory names no tensor called {name:?}")]
    NoSuchTensor { path: String, name: String },

    #[error("{path}: tensor {name:?} lies outside the data region")]
    TensorOutOfBounds { path: String, name: String },
}

/// The result type of this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// A container opened for reading.
///
/// The data region is memory-mapped, so opening a 2.5 GB container costs a
/// mapping and not 2.5 GB of resident memory. The whole point of the format
/// is that the runtime never copies weights at load.
pub struct Container {
    path: String,
    header: Header,
    directory: Directory,
    bytes: Vec<u8>,
}

impl Container {
    /// Open a container and read its header and directory.
    ///
    /// The data region is read into memory here rather than mapped. That is a
    /// deliberate simplification for the moment -- a real mapping needs
    /// platform-specific code, and this crate's first job is to be READABLE
    /// and to pin the format down. The alternative -- copying -- is what the
    /// C++ runtime avoided by never loading the payload at all, and the
    /// memmap is where this crate goes next.
    pub fn open(path: &Path) -> Result<Self> {
        let path_str = path.display().to_string();
        let bytes = std::fs::read(path).map_err(|source| Error::Io {
            path: path_str.clone(),
            what: "the file could not be read".into(),
            source,
        })?;
        let header = Header::parse(&bytes).map_err(|e| match e {
            Error::NotAContainer { magic, .. } => Error::NotAContainer {
                path: path_str.clone(),
                magic,
            },
            Error::VersionMismatch { found, expected, .. } => Error::VersionMismatch {
                path: path_str.clone(),
                found,
                expected,
            },
            other => other,
        })?;
        let raw = header.directory_bytes(&bytes).ok_or_else(|| Error::Io {
            path: path_str.clone(),
            what: "the directory lies outside the file".into(),
            source: std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "the directory lies outside the file",
            ),
        })?;
        let directory: Directory =
            serde_json::from_slice(raw).map_err(|source| Error::BadDirectory {
                path: path_str.clone(),
                source,
            })?;
        Ok(Self {
            path: path_str,
            header,
            directory,
            bytes,
        })
    }

    /// The file this container was read from.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The header.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// The directory: the config block and the tensor index.
    pub fn directory(&self) -> &Directory {
        &self.directory
    }

    /// The config block.
    pub fn config(&self) -> &crate::directory::Config {
        self.directory.config()
    }

    /// One config key, as a string.
    ///
    /// The absence of a key is `None` and not an empty string: an empty
    /// `arch` and a missing `arch` are different facts, and the second one is
    /// what a pre-arch container looks like.
    pub fn config_str(&self, key: &str) -> Option<&str> {
        self.directory.config().get_str(key)
    }

    /// The bytes of one tensor.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let entry = self
            .directory
            .tensor(name)
            .ok_or_else(|| Error::NoSuchTensor {
                path: self.path.clone(),
                name: name.to_string(),
            })?;
        let start = (self.header.data_offset + entry.offset) as usize;
        let end = start + entry.nbytes as usize;
        let data = self
            .bytes
            .get(start..end)
            .ok_or_else(|| Error::TensorOutOfBounds {
                path: self.path.clone(),
                name: name.to_string(),
            })?;
        Ok(data)
    }

    /// Every tensor's name, in directory order.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.directory.tensors().iter().map(|t| t.name.as_str()).collect()
    }
}

/// The bytes of a tensor, widened to `f32`.
///
/// BF16 is stored as raw `u16` and widened here, because BF16 has no native
/// dtype and the array's panel is a `u16` array. The widening is
/// round-to-nearest-even from f32 to bf16 and back through f32, which is what
/// the host kernels in `npum-cpu` implement explicitly.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Round a f32 to bf16, round-to-nearest-even, and back.
///
/// This is the conversion the array's bf16 datapath performs, and it is
/// implemented here rather than taken from `std::bf16` until a gate has
/// confirmed that the standard library's semantics are the ones the hardware
/// has. The ties rule matters: round-half-to-even and round-half-away differ
/// on every exact .5, and a container that was written with one and read with
/// the other loses a bit of every weight that lands on one.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    // Round to nearest even: add 0x7FFF plus the low bit of the mantissa,
    // then truncate.
    let rounding = 0x7FFF + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding)) >> 16) as u16
}

/// The f32 a bf16 bit pattern denotes.
pub fn bf16_round_trip(value: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(value))
}
