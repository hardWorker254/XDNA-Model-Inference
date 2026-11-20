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

pub mod bf16;
pub mod digest;
pub mod directory;
pub mod header;
pub mod layout;
pub mod tiling;
pub mod writer;

pub use digest::{model_digest, sha256_bytes, sha256_file};
pub use directory::{Directory, TensorEntry};
pub use header::{ContainerFormat, Header, MAGIC_CURRENT, MAGIC_LEGACY, VERSION};
pub use layout::{BLayout, MacGeom, gemm_b_layout, layout_hash};
pub use tiling::tile_b_into;
pub use writer::Writer;

use memmap2::Mmap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

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

/// The shared, immutable part of a container.
///
/// One `.npum` can be opened a dozen times in a single run -- for its arch,
/// its layer count, its tokenizer, its design manifest, its panels. Holding
/// that state behind an `Arc` means all those opens share ONE mapping of the
/// file instead of copying (or mapping) its gigabytes over and over.
pub struct ContainerMeta {
    path: String,
    header: Header,
    directory: Directory,
    bytes: Mmap,
}

/// A container opened for reading.
///
/// The data region is memory-mapped, so opening a 2.5 GB container costs a
/// mapping and not 2.5 GB of resident memory. The whole point of the format
/// is that the runtime never copies weights at load. `Container` is a cheap
/// handle: clones share the same mapping, and `open()` hands back a handle to
/// a cached mapping whenever the file has not changed on disk.
#[derive(Clone)]
pub struct Container {
    meta: Arc<ContainerMeta>,
}

/// What makes one cached mapping distinct from another.
///
/// A container that is rewritten in place keeps its path, so the path alone
/// is not a cache key: the modification time and the length are part of it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    path: PathBuf,
    mtime: u128,
    len: u64,
}

static CACHE: OnceLock<Mutex<HashMap<CacheKey, Arc<ContainerMeta>>>> = OnceLock::new();

fn cache_key(path: &Path) -> std::io::Result<CacheKey> {
    let stat = std::fs::metadata(path)?;
    let mtime = stat
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    Ok(CacheKey {
        path: path.to_path_buf(),
        mtime,
        len: stat.len(),
    })
}

/// The cached mapping, keyed by (path, mtime, len).
///
/// Kept separate from `Container::open` so a caller that only needs metadata
/// can ask for the shared `Arc` without wrapping it in a handle first.
pub fn open_meta(path: &Path) -> Result<Arc<ContainerMeta>> {
    Ok(Container::open(path)?.meta)
}

impl Container {
    /// Open a container and read its header and directory.
    ///
    /// The file is MAPPED, not read: the first touch of a weight faults a page
    /// in from the page cache, and no part of the payload is ever copied into
    /// a Rust-owned buffer. Repeated opens of the same, unchanged file share
    /// one mapping -- see [`open_meta`] -- which is what keeps a CLI run from
    /// paying for the container a dozen times.
    pub fn open(path: &Path) -> Result<Self> {
        let key = cache_key(path).map_err(|source| Error::Io {
            path: path.display().to_string(),
            what: "the file could not be examined".into(),
            source,
        })?;

        // A poisoned mutex is still readable: the data behind it is immutable.
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        {
            let guard = cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(meta) = guard.get(&key) {
                return Ok(Self {
                    meta: Arc::clone(meta),
                });
            }
        }

        let meta = Arc::new(Self::map(path)?);
        let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        // Another thread may have mapped the same file while we did. Its
        // mapping is as good as ours, so keep the one already in the cache.
        let meta = guard.entry(key).or_insert(meta).clone();
        Ok(Self { meta })
    }

    /// Map the file and parse its header and directory. Called once per
    /// (path, mtime, len) by [`Self::open`].
    fn map(path: &Path) -> Result<ContainerMeta> {
        let path_str = path.display().to_string();
        let file = std::fs::File::open(path).map_err(|source| Error::Io {
            path: path_str.clone(),
            what: "the file could not be opened".into(),
            source,
        })?;
        // SAFETY: the file is opened read-only and is never written through
        // this mapping, so no other process can mutate the mapped bytes under
        // us. The directory is parsed before any tensor is handed out, and a
        // container that is rewritten in place gets a new key in `CACHE`.
        let bytes = unsafe { Mmap::map(&file) }.map_err(|source| Error::Io {
            path: path_str.clone(),
            what: "the file could not be memory-mapped".into(),
            source,
        })?;
        let header = Header::parse(&bytes).map_err(|e| match e {
            Error::NotAContainer { magic, .. } => Error::NotAContainer {
                path: path_str.clone(),
                magic,
            },
            Error::VersionMismatch {
                found, expected, ..
            } => Error::VersionMismatch {
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
        Ok(ContainerMeta {
            path: path_str,
            header,
            directory,
            bytes,
        })
    }
}

impl ContainerMeta {
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

    /// The bytes of one tensor, borrowed straight out of the mapping.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let entry = self
            .directory
            .tensor(name)
            .ok_or_else(|| Error::NoSuchTensor {
                path: self.path.clone(),
                name: name.to_string(),
            })?;
        let bad = || Error::TensorOutOfBounds {
            path: self.path.clone(),
            name: name.to_string(),
        };
        // A directory is data, and data can be corrupt. Neither the header's
        // offsets nor the entry's may wrap, on 32- or 64-bit hosts; and a
        // tensor must stay inside the header's declared payload, not just
        // inside the mapping (which would let it borrow the directory).
        let offset = usize::try_from(entry.offset).map_err(|_| bad())?;
        let nbytes = usize::try_from(entry.nbytes).map_err(|_| bad())?;
        let base = usize::try_from(self.header.data_offset).map_err(|_| bad())?;
        let start = base.checked_add(offset).ok_or_else(bad)?;
        let end = start.checked_add(nbytes).ok_or_else(bad)?;
        let payload_end = usize::try_from(self.header.data_length)
            .ok()
            .and_then(|len| base.checked_add(len))
            .unwrap_or(usize::MAX);
        if end > payload_end {
            return Err(bad());
        }
        self.bytes.get(start..end).ok_or_else(bad)
    }

    /// Every tensor's name, in directory order.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.directory
            .tensors()
            .iter()
            .map(|t| t.name.as_str())
            .collect()
    }
}

impl Container {
    /// The file this container was read from.
    pub fn path(&self) -> &str {
        self.meta.path()
    }

    /// The header.
    pub fn header(&self) -> &Header {
        self.meta.header()
    }

    /// The directory: the config block and the tensor index.
    pub fn directory(&self) -> &Directory {
        self.meta.directory()
    }

    /// The shared metadata behind this handle.
    pub fn meta(&self) -> &Arc<ContainerMeta> {
        &self.meta
    }

    /// The config block.
    pub fn config(&self) -> &crate::directory::Config {
        self.meta.config()
    }

    /// One config key, as a string.
    ///
    /// The absence of a key is `None` and not an empty string: an empty
    /// `arch` and a missing `arch` are different facts, and the second one is
    /// what a pre-arch container looks like.
    pub fn config_str(&self, key: &str) -> Option<&str> {
        self.meta.config_str(key)
    }

    /// The bytes of one tensor, borrowed straight out of the mapping.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        self.meta.tensor_bytes(name)
    }

    /// Every tensor's name, in directory order.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.meta.tensor_names()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory::Config;
    use crate::writer::Staged;

    fn tiny_container() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "npum-mmap-{}-{}.npum",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let config = Config::from_value(serde_json::json!({
            "arch": "test",
            "num_layers": 2,
            "max_seq_len": 8,
        }));
        let mut writer = Writer::new(config, 1, 0);
        writer.add(Staged {
            name: "tokenizer.vocab".into(),
            role: "tokenizer".into(),
            dtype: "U8".into(),
            logical_shape: vec![4],
            padded_shape: vec![4],
            blob: b"word".to_vec(),
            layout: None,
        });
        writer.write(&path).expect("the container writes");
        path
    }

    /// A container opened ten times maps the file once.
    #[test]
    fn repeated_opens_share_one_mapping() {
        let path = tiny_container();
        let a = Container::open(&path).expect("the container opens");
        let b = Container::open(&path).expect("the container opens again");
        assert!(
            Arc::ptr_eq(a.meta(), b.meta()),
            "two opens of an unchanged file must share one mapping"
        );
        assert_eq!(a.tensor_bytes("tokenizer.vocab").unwrap(), b"word");
        assert_eq!(a.config_str("arch"), Some("test"));
        assert_eq!(a.config().get_i64("num_layers"), Some(2));
        let meta = open_meta(&path).expect("the cached metadata opens");
        assert!(Arc::ptr_eq(&meta, a.meta()));
        drop((a, b, meta));
        std::fs::remove_file(&path).ok();
    }
}
