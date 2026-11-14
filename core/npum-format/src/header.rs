//! The 64-byte container header.
//!
//! Little-endian, fixed size, no padding:
//!
//! ```text
//! offset size field
//! 0      4    magic          "NPUM" (or the legacy "NPUE")
//! 4      4    version        1
//! 8      4    arch           the container's architecture id
//! 12     4    flags          bit 0: the B panels are pre-tiled
//! 16     8    json_offset    where the directory starts
//! 24     8    json_length    how many bytes the directory is
//! 32     8    data_offset    where the tensor payloads start (4096-aligned)
//! 40     8    data_length    how many bytes the payloads are
//! 48     16   reserved       zero
//! ```
//!
//! The offsets are `u64` and not `u32` on purpose: a container holds up to
//! 2.5 GB of weights today and the format is meant to outlive that.

use crate::{Error, Result};

/// The magic of a container written by this build.
pub const MAGIC_CURRENT: [u8; 4] = *b"NPUM";

/// The magic of a container written before the rename.
///
/// Still read. The layout is identical -- only these four bytes changed -- so
/// refusing them would refuse every container that has not been repacked, for
/// no gain.
pub const MAGIC_LEGACY: [u8; 4] = *b"NPUE";

/// The container version this build reads.
pub const VERSION: u32 = 1;

/// The header's size in bytes.
pub const HEADER_SIZE: usize = 64;

/// Which of the two names a container carries.
///
/// The distinction matters for exactly one question -- "should this file be
/// called `.npum` when it is next written out?" -- and for nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerFormat {
    /// `NPUM`: the current name.
    Current,
    /// `NPUE`: the name from before the rename. Read, never written.
    Legacy,
}

impl ContainerFormat {
    /// The magic bytes of this format.
    pub fn magic(self) -> [u8; 4] {
        match self {
            ContainerFormat::Current => MAGIC_CURRENT,
            ContainerFormat::Legacy => MAGIC_LEGACY,
        }
    }

    /// The file extension this format is written with.
    pub fn extension(self) -> &'static str {
        match self {
            ContainerFormat::Current => "npum",
            ContainerFormat::Legacy => "npue",
        }
    }
}

/// The parsed header.
#[derive(Debug, Clone)]
pub struct Header {
    /// Which name the container carries.
    pub format: ContainerFormat,
    /// The container version.
    pub version: u32,
    /// The architecture id.
    pub arch: u32,
    /// The flag bits.
    pub flags: u32,
    /// Where the directory starts.
    pub json_offset: u64,
    /// How many bytes the directory is.
    pub json_length: u64,
    /// Where the tensor payloads start.
    pub data_offset: u64,
    /// How many bytes the payloads are.
    pub data_length: u64,
}

fn rd_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn rd_u64(bytes: &[u8], at: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(b)
}

impl Header {
    /// Parse the header out of the first bytes of a file.
    ///
    /// `bytes` may be longer than the header; only the first 64 are read.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(Error::Io {
                path: String::new(),
                what: "the file is shorter than a header".into(),
                source: std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("{HEADER_SIZE} header bytes needed, {} present", bytes.len()),
                ),
            });
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[0..4]);
        let format = match magic {
            MAGIC_CURRENT => ContainerFormat::Current,
            MAGIC_LEGACY => ContainerFormat::Legacy,
            _ => {
                return Err(Error::NotAContainer {
                    path: String::new(),
                    magic,
                });
            }
        };
        let version = rd_u32(bytes, 4);
        if version != VERSION {
            return Err(Error::VersionMismatch {
                path: String::new(),
                found: version,
                expected: VERSION,
            });
        }
        Ok(Self {
            format,
            version,
            arch: rd_u32(bytes, 8),
            flags: rd_u32(bytes, 12),
            json_offset: rd_u64(bytes, 16),
            json_length: rd_u64(bytes, 24),
            data_offset: rd_u64(bytes, 32),
            data_length: rd_u64(bytes, 40),
        })
    }

    /// Serialise the header.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut out = [0u8; HEADER_SIZE];
        out[0..4].copy_from_slice(&self.format.magic());
        out[4..8].copy_from_slice(&self.version.to_le_bytes());
        out[8..12].copy_from_slice(&self.arch.to_le_bytes());
        out[12..16].copy_from_slice(&self.flags.to_le_bytes());
        out[16..24].copy_from_slice(&self.json_offset.to_le_bytes());
        out[24..32].copy_from_slice(&self.json_length.to_le_bytes());
        out[32..40].copy_from_slice(&self.data_offset.to_le_bytes());
        out[40..48].copy_from_slice(&self.data_length.to_le_bytes());
        // bytes 48..64 stay zero: the reserved field.
        out
    }

    /// The directory's bytes, if the header's offset and length fit in `file`.
    pub fn directory_bytes<'a>(&self, file: &'a [u8]) -> Option<&'a [u8]> {
        let start = self.json_offset as usize;
        let end = start.checked_add(self.json_length as usize)?;
        file.get(start..end)
    }
}
