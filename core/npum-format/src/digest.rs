//! Digests: what a container says about where its weights came from.
//!
//! Two of them, and they are not interchangeable.
//!
//! * `sha256_file` is the digest of ONE file. It is what a downloader checks.
//! * `model_digest` is the digest of a SET of files: it frames each file's
//!   name and its own digest and hashes the frames.
//!
//! The framing is what stops a checkpoint's bytes being re-partitioned across
//! files to reproduce an unrelated pin. An ONNX export with external data is
//! several files that are one model; a pin that hashed their bytes together
//! without naming them could be reproduced by a different set of files with
//! the same content. The frames name each file, so the partition is part of
//! what is hashed.

use sha2::{Digest, Sha256};
use std::path::Path;

/// The SHA-256 of a byte slice, hex-encoded.
pub fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex(&hasher.finalize())
}

/// The SHA-256 of a file, read in chunks.
///
/// Streams: a 2.5 GB side file must not become 2.5 GB of resident memory
/// just to be hashed.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

/// The source digest of a set of checkpoint files.
///
/// `graphs` are the graph files -- the first is the graph, the rest are the
/// external data files it points at, in that order. Each contributes a frame:
///
/// ```text
/// basename NUL sha256-of-file LF
/// ```
///
/// and the frames are concatenated and hashed. The frames are hashed in the
/// order given, so the digest says not only WHAT the bytes were but HOW they
/// were partitioned.
pub fn model_digest(graphs: &[std::path::PathBuf]) -> std::io::Result<String> {
    let mut hasher = Sha256::new();
    for path in graphs {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let digest = sha256_file(path)?;
        hasher.update(name.as_bytes());
        hasher.update([0u8]);
        hasher.update(digest.as_bytes());
        hasher.update(b"\n");
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        out.push(char::from_digit((b & 0x0f) as u32, 16).unwrap());
    }
    out
}
