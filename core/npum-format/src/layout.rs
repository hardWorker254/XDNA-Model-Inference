//! The pre-tiled B panel's layout, and the per-generation sub-tile.
//!
//! # Why this file is small and why it is here
//!
//! A B panel for the array is stored in the MMAC's own byte order inside one
//! (tile_k, tile_n) tile. That order depends on the BOARD and on the OPERAND
//! DTYPE, and the two answers differ:
//!
//! ```text
//!                npu1 / aie2              npu2 / aie2p
//!   bf16         (8, 4)                   (8, 8)
//!   i8           (8, 8)                   (8, 8)
//! ```
//!
//! A container packed with one pair and read by a core built for the other
//! has every panel's columns permuted. The byte count is right, the shapes
//! agree, BOTH SIDES derive the same `layout_hash` from the same wrong
//! constant, and every product is plausible. Nothing in the loader can see
//! it.
//!
//! That is not a hypothetical. A bge-small int8 design shipped with the bf16
//! pair on npu1, where the int8 MMAC's N sub-tile is 8 and bf16's is 4. The
//! exporter, the packer, the runtime's `layout_hash` comparison and the
//! numerics gate all agreed with each other and all disagreed with the
//! hardware, because all four read the same wrong constant. End to end it
//! read as 1-cos 8.6e-01 on a model whose bf16 path is 1e-05.
//!
//! So `mac_for` REQUIRES the dtype. The bf16 answer is the tempting one, and
//! a default is how the wrong one got shipped.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The MMAC sub-tile: `(s, t)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MacGeom {
    /// The `s` sub-tile.
    pub s: i64,
    /// The `t` sub-tile.
    pub t: i64,
}

/// The NPU generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Generation {
    /// XDNA1, `aie2`. The sub-tile is (8, 4) for bf16 and (8, 8) for int8.
    Npu1,
    /// XDNA2, `aie2p`. The sub-tile is (8, 8) for both.
    Npu2,
}

impl std::fmt::Display for Generation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Generation {
    /// The name this generation is spelled with on the command line.
    pub fn as_str(self) -> &'static str {
        match self {
            Generation::Npu1 => "npu1",
            Generation::Npu2 => "npu2",
        }
    }

    /// Parse a generation's name.
    ///
    /// A typo is an error and not a default. A misspelled `--dev npu3` that
    /// silently became npu1 would load a design for the wrong board and
    /// produce wrong numbers with every check green.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "npu1" => Some(Generation::Npu1),
            "npu2" => Some(Generation::Npu2),
            _ => None,
        }
    }
}

/// The operand dtype a layout is written for.
///
/// The dtype is part of the layout and not a separate question because on
/// npu1 it changes the sub-tile. See the module comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperandDtype {
    /// The array's bf16 datapath.
    Bf16,
    /// The array's int8 datapath, which an int4 (W4A8) container also uses:
    /// its B panel is widened to int8 before staging, so the bytes the core
    /// consumes are int8's.
    I8,
    /// Plain row-major f32 -- the host's datapath, and the CPU control
    /// container.
    F32,
}

impl OperandDtype {
    /// The dtype tag as it appears in a directory.
    pub fn tag(self) -> &'static str {
        match self {
            OperandDtype::Bf16 => "BF16",
            OperandDtype::I8 => "I8",
            OperandDtype::F32 => "F32",
        }
    }
}

/// The sub-tile for one generation and one dtype.
pub fn mac_for(generation: Generation, dtype: OperandDtype) -> MacGeom {
    match (generation, dtype) {
        (Generation::Npu1, OperandDtype::Bf16) => MacGeom { s: 8, t: 4 },
        (Generation::Npu1, OperandDtype::I8) => MacGeom { s: 8, t: 8 },
        (Generation::Npu2, OperandDtype::Bf16) => MacGeom { s: 8, t: 8 },
        (Generation::Npu2, OperandDtype::I8) => MacGeom { s: 8, t: 8 },
        // f32 is the host's operand and no array consumes it, so there is no
        // sub-tile to be wrong about.
        (_, OperandDtype::F32) => MacGeom { s: 8, t: 8 },
    }
}

/// The canonical B-layout descriptor.
///
/// Built HERE and never inlined. Every hand-written copy of this struct is a
/// chance for two sides to drift, and the drift is invisible: `layout_hash`
/// changes, the bytes do not, and the check that exists to catch wrong
/// layouts starts reporting a mismatch that is not one. That happened: the
/// exporter wrote the struct by hand and omitted `dtype`, so a correct file
/// failed the check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BLayout {
    /// Always `block_panel`.
    pub kind: String,
    /// The tile's K extent.
    pub tile_k: i64,
    /// The tile's N extent.
    pub tile_n: i64,
    /// The order of the blocks inside a tile.
    pub order: String,
    /// The order inside a block.
    pub inner: String,
    /// The MMAC's `s` sub-tile.
    pub mac_s: i64,
    /// The MMAC's `t` sub-tile.
    pub mac_t: i64,
    /// The operand dtype.
    pub dtype: String,
}

/// The canonical B layout descriptor for one tile and one sub-tile.
pub fn gemm_b_layout(
    tile_k: i64,
    tile_n: i64,
    mac_s: i64,
    mac_t: i64,
    dtype: OperandDtype,
) -> BLayout {
    BLayout {
        kind: "block_panel".into(),
        tile_k,
        tile_n,
        order: "k,n,kt,nt".into(),
        inner: "s,t".into(),
        mac_s,
        mac_t,
        dtype: dtype.tag().into(),
    }
}

/// The hash of everything that changes how bytes are laid out.
///
/// A file packed with different tile dimensions is not merely suboptimal for
/// a kernel expecting others -- it is wrong, silently. The loader compares
/// this and refuses rather than producing plausible garbage.
///
/// # The canonical form is KEY-SORTED
///
/// The Python packer hashes `json.dumps(layout, sort_keys=True,
/// separators=(",", ":"))`, and this function must produce the same bytes or
/// every container reads as a mismatch. That means the keys are sorted, not
/// in the order the struct declares them: the struct's order is
/// kind, tile_k, tile_n, order, inner, mac_s, mac_t, dtype, and the sorted
/// order is dtype, inner, kind, mac_s, mac_t, order, tile_k, tile_n.
///
/// The sort comes from `serde_json::Map`, which is a `BTreeMap` unless the
/// `preserve_order` feature is switched on. Nothing in this workspace
/// switches it on, and `layout_hash` is the one place where that would
/// matter -- so the dependency list is the thing to check when a hash starts
/// disagreeing with the Python packer's.
pub fn layout_hash(layout: &BLayout) -> String {
    let value = serde_json::to_value(layout).expect("a layout is always serialisable");
    let canonical = serde_json::to_string(&value).expect("a layout is always serialisable");
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    let out = hasher.finalize();
    let mut s = String::with_capacity(64);
    for b in out {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0x0f) as u32, 16).unwrap());
    }
    s
}
