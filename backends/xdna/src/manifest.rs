//! The design set's manifest: what `design.json` says.
//!
//! # WHAT THIS FILE IS
//!
//! A compiled design set is a directory:
//!
//! ```text
//!   gemm_rtp/
//!     design.json          this file's subject
//!     final.xclbin         the compiled design
//!     insts.bin            the instruction stream of slot 0
//!     insts_qkv_b4.bin     the instruction stream of slot 1
//!     ...
//! ```
//!
//! `design.json` is the manifest: the shapes the design was compiled for, the
//! sizes of its buffers, the instruction streams' slots, and the B panel's
//! layout. Everything here is read FROM the file and never inferred: a
//! runtime that guessed a tile would compute the wrong thing plausibly.
//!
//! # WHY THE MANIFEST IS PARSED AND NOT ASSUMED
//!
//! The C++ runtime parsed `design.json` with its own small JSON reader and
//! read the same fields. The field names here are the file's, so a design
//! set that the C++ runtime could run is one this parser can read too -- and
//! a field that this parser does not know is a field that is ignored rather
//! than a design that is refused, because the exporter grows fields and a
//! runtime that refuses new ones is a runtime that stops working when the
//! exporter improves.
//!
//! # THE B LAYOUT IS THE DANGEROUS ONE
//!
//! The B panel's byte order depends on the board and on the operand dtype,
//! and a panel packed for one board and read by a core built for the other
//! has its columns permuted -- the same byte count, the same shapes, and
//! every product plausible. The manifest records the layout and the hash of
//! it, and the loader checks the two against each other and against the
//! device.

use std::path::Path;

use npum_format::layout::BLayout;
use serde::Deserialize;

/// The manifest of a design set.
#[derive(Debug, Clone, Deserialize)]
pub struct DesignInfo {
    /// The set's name -- `gemm_rtp`, `gelu`, `layernorm`, `softmax`.
    pub name: String,

    /// The kernel's name inside the xclbin.
    ///
    /// The AIE flow's kernels are all called `MLIR_AIE...`; the manifest
    /// records which one, and the loader opens that one rather than the first
    /// kernel it finds.
    #[serde(default)]
    pub kernel: String,

    /// The generation this design was compiled for -- 1 or 2.
    #[serde(default)]
    pub arch: i64,

    /// The generation this design was compiled for, by name -- `npu1`.
    #[serde(default)]
    pub device: String,

    /// The data buffers' sizes, in bytes: A, B, C.
    pub buffers: Vec<u64>,

    /// The C buffer's dtype -- `bf16` or `f32`.
    #[serde(default)]
    pub c_dtype: String,

    /// The A buffer's dtype -- `bf16`, `i8` or `f32`.
    #[serde(default)]
    pub a_dtype: String,

    /// The B panel's layout.
    ///
    /// Absent in an artifact written before the field existed, and ABSENT
    /// MEANS ABSENT: the loader does not substitute a default, because a
    /// default sub-tile is how an int8 design came to be read with bf16's.
    #[serde(default)]
    pub b_layout: Option<BLayout>,

    /// The hash of the layout above.
    #[serde(default)]
    pub b_layout_hash: String,

    /// The design's own tile.
    pub tile: Tile,

    /// The instruction stream of slot 0.
    #[serde(default)]
    pub slot0: Option<Stream>,

    /// The instruction streams, by slot.
    #[serde(default)]
    pub streams: Vec<Stream>,
}

/// The design's tile: the shape it was compiled for.
#[derive(Debug, Clone, Deserialize)]
pub struct Tile {
    /// The rows one dispatch computes.
    #[serde(default)]
    pub m: i64,
    /// The reduction extent.
    #[serde(default)]
    pub k: i64,
    /// The output columns.
    #[serde(default)]
    pub n: i64,
    /// The AIE rows per tile.
    #[serde(default)]
    pub rows: i64,
}

/// One instruction stream: a slot, a file, and the shape it computes.
#[derive(Debug, Clone, Deserialize)]
pub struct Stream {
    /// What the stream computes -- `qkv`, `attn_out`, `ffn_up`, ...
    #[serde(default)]
    pub op: String,

    /// The batch this stream was compiled for.
    #[serde(default)]
    pub batch: i64,

    /// The slot number.
    ///
    /// Slot 0 is the design's own entry point, and its record does not carry
    /// the number -- it IS slot 0 -- so the default is zero and the record
    /// and the number agree.
    #[serde(default)]
    pub slot: i64,

    /// The file, inside the set's directory.
    pub file: String,

    /// The rows.
    #[serde(rename = "M", default)]
    pub m: i64,
    /// The reduction extent.
    #[serde(rename = "K", default)]
    pub k: i64,
    /// The columns.
    #[serde(rename = "N", default)]
    pub n: i64,
}

impl DesignInfo {
    /// Read the manifest of a design set on disk.
    ///
    /// The refusal names the file it could not read, because "the design set
    /// is not there" and "the design set is there and its manifest is not
    /// JSON" are different problems with different fixes.
    pub fn read(dir: &Path) -> Result<Self, String> {
        let path = dir.join("design.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        serde_json::from_str(&text)
            .map_err(|e| format!("cannot parse {}: {e}", path.display()))
    }

    /// The instruction streams, in slot order: slot 0 first, then the rest.
    ///
    /// The order is the DISPATCH order, and it is not the order the file
    /// lists them in: slot 0 is the design's own entry point and the others
    /// are the per-op streams. The Python design harness walks them the same
    /// way, and the two must agree about which file is which slot -- a design
    /// dispatched with the wrong stream computes the wrong shape, correctly.
    pub fn streams_in_slot_order(&self) -> Vec<&Stream> {
        let mut out: Vec<&Stream> = Vec::new();
        if let Some(slot0) = &self.slot0 {
            out.push(slot0);
        }
        let mut rest: Vec<&Stream> = self.streams.iter().collect();
        rest.sort_by_key(|s| s.slot);
        out.extend(rest);
        out
    }

    /// The kernel's name inside the xclbin.
    ///
    /// The AIE flow names its kernels `MLIR_AIE...`; the manifest records
    /// the name when it was written and the fallback is the prefix's first
    /// match, which is what the C++ runtime looked for.
    pub fn kernel_name(&self) -> &str {
        if self.kernel.is_empty() {
            "MLIR_AIE"
        } else {
            &self.kernel
        }
    }
}
