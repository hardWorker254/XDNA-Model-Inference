//! The per-generation geometry: the numbers that differ between the boards.
//!
//! # The B panel's sub-tile, and why it is typed rather than configured
//!
//! The MMAC's sub-tile is (s, t), and it is NOT the same on both boards:
//!
//! ```text
//!   npu1 / aie2     bf16 (8, 4)   i8 (8, 8)
//!   npu2 / aie2p    bf16 (8, 8)   i8 (8, 8)
//! ```
//!
//! The C++ runtime carried these as STRINGS, in two places that had to agree:
//! the packer asked for "npu1" and got (8, 4), the exporter asked for the
//! same and got (8, 4), and both were written down again in the runtime's
//! own dispatch code. A container packed for one board and read by a design
//! built for the other is **not rejected anywhere**: the byte count agrees,
//! the shapes agree, and both sides derive the same `layout_hash` from the
//! same wrong constant. The only symptom is wrong numbers.
//!
//! So here they are CONSTANTS, in ONE place, reached only through
//! `Geometry::for_generation`. There is no way to ask for the wrong one, and
//! a typo is a compile error instead of a wrong answer.
//!
//! # The context budget is Option, not a number
//!
//! The C++ runtime's `context_budget()` returned a plain integer, and for
//! npu2 it returned a number nobody had measured. The comparison then read
//! as "the budget is zero" and the process refused to run -- or, worse, the
//! comparison was skipped and the contention went unnoticed.
//!
//! `Option<Budget>` with `None` meaning "this cannot be compared" is the
//! honest shape. A budget that has not been measured is a budget that has
//! not been measured, and a runtime that says so is a runtime that can be
//! trusted when it says the budget IS known.

use npum_format::layout::{Generation, MacGeom, OperandDtype};
use serde::{Deserialize, Serialize};

/// The geometry of one NPU generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Geometry {
    /// The generation this geometry is for.
    pub generation: Generation,
    /// The AIE column count the array presents.
    pub columns: i64,
    /// The BF16 operand's sub-tile.
    pub bf16: MacGeom,
    /// The int8 operand's sub-tile.
    pub i8: MacGeom,
    /// The hardware context budget, if it is known.
    pub context_budget: Option<Budget>,
}

/// The hardware context budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    /// How many hardware contexts the device offers.
    pub contexts: u32,
}

/// The AIE column counts, per generation.
///
/// A design's N must be a multiple of `tile_n * columns`, and that product
/// is what a B panel has to be padded UP to -- so a packer needs the column
/// count and cannot get it from the sub-tile, which knows about the MMAC and
/// not about the array's width.
const COLUMNS: [(Generation, i64); 2] = [
    (Generation::Npu1, 4),
    (Generation::Npu2, 8),
];

impl Geometry {
    /// The geometry of one generation.
    pub fn for_generation(generation: Generation) -> Self {
        let columns = COLUMNS
            .iter()
            .find(|(g, _)| *g == generation)
            .map(|(_, c)| *c)
            .unwrap_or(4);
        let bf16 = npum_format::layout::mac_for(generation, OperandDtype::Bf16);
        let i8 = npum_format::layout::mac_for(generation, OperandDtype::I8);
        Self {
            generation,
            columns,
            bf16,
            i8,
            context_budget: None,
        }
    }

    /// The sub-tile for one operand dtype.
    pub fn mac_for(&self, dtype: OperandDtype) -> MacGeom {
        match dtype {
            OperandDtype::Bf16 => self.bf16,
            OperandDtype::I8 => self.i8,
            OperandDtype::F32 => MacGeom { s: 8, t: 8 },
        }
    }

    /// The `s` sub-tile for the BF16 datapath.
    pub fn mac_s(&self) -> i64 {
        self.bf16.s
    }

    /// The `t` sub-tile for the BF16 datapath.
    pub fn mac_t(&self) -> i64 {
        self.bf16.t
    }

    /// Whether the context budget is known.
    ///
    /// `None` means "not measured", and the caller must NOT read it as
    /// "zero" or as "plenty".
    pub fn context_budget_is_known(&self) -> bool {
        self.context_budget.is_some()
    }
}
