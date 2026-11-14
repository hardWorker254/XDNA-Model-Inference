//! The two generations, side by side.
//!
//! # What this file is for
//!
//! The plan's section 6 is "compatibility with XDNA2", and the shape of that
//! work is a table: the same model, the same container, two boards whose
//! numbers differ in three places -- the sub-tile, the column count and the
//! design set's own generation stamp. A table that is only ever checked by
//! eye is a table that rots, so it is checked here, and the check is a TEST
//! rather than a comment because the failure it guards against is a wrong
//! number and not a crash.
//!
//! # What this file does NOT claim
//!
//! It does not claim that npu2 runs. There is no npu2 board here, no npu2
//! design set, and the exported design sets in this repository are all npu1.
//! What it claims is narrower and the only thing that can be claimed from a
//! npu1 machine: **the npu2 path compiles, and the numbers it would use are
//! the ones the plan says they are.** The geometry is reached only through
//! `Geometry::for_generation`, so there is no way to ask for the wrong one --
//! and the tests below would fail if someone added a second source of truth.

use npum_format::layout::{Generation, OperandDtype};
use npum_xdna::design::{design_fits, Design, DesignSet};
use npum_xdna::geometry::Geometry;

/// The names the CLI and the exporter use for the generations.
#[test]
fn the_generations_have_names_and_parse_back() {
    assert_eq!(Generation::Npu1.as_str(), "npu1");
    assert_eq!(Generation::Npu2.as_str(), "npu2");
    assert_eq!(Generation::parse("npu1"), Some(Generation::Npu1));
    assert_eq!(Generation::parse("npu2"), Some(Generation::Npu2));
    assert_eq!(Generation::parse("npu3"), None);
    assert_eq!(Generation::parse(""), None);
}

/// The sub-tile and the column count, for both boards.
///
/// The bf16 sub-tile is the one that DIFFERS: (8, 4) on npu1, (8, 8) on
/// npu2. The int8 one is (8, 8) on both, and the column count doubles.
#[test]
fn the_geometry_differs_where_the_plan_says_it_does() {
    let npu1 = Geometry::for_generation(Generation::Npu1);
    let npu2 = Geometry::for_generation(Generation::Npu2);

    assert_eq!((npu1.mac_s(), npu1.mac_t()), (8, 4), "npu1 bf16 is (8, 4)");
    assert_eq!((npu2.mac_s(), npu2.mac_t()), (8, 8), "npu2 bf16 is (8, 8)");

    assert_eq!(npu1.columns, 4, "npu1 presents four AIE columns");
    assert_eq!(npu2.columns, 8, "npu2 presents eight AIE columns");

    assert_eq!(
        (npu1.mac_for(OperandDtype::I8).s, npu1.mac_for(OperandDtype::I8).t),
        (8, 8),
        "int8 is (8, 8) on npu1"
    );
    assert_eq!(
        (npu2.mac_for(OperandDtype::I8).s, npu2.mac_for(OperandDtype::I8).t),
        (8, 8),
        "int8 is (8, 8) on npu2"
    );
}

/// The context budget is not measured on either board, and `None` means
/// exactly that.
///
/// A budget that has not been measured is a budget that has not been
/// measured. Reading `None` as zero would make the runtime refuse to run;
/// reading it as "plenty" would make it run without noticing contention.
#[test]
fn the_context_budget_is_unknown_on_both_boards() {
    for generation in [Generation::Npu1, Generation::Npu2] {
        let geometry = Geometry::for_generation(generation);
        assert!(!geometry.context_budget_is_known());
        assert!(geometry.context_budget.is_none());
    }
}

/// A design set built for the OTHER board is refused -- fail closed.
///
/// This is the check whose absence is invisible. A container packed for one
/// board and read by a design built for the other agrees on the byte count,
/// agrees on the shapes, and derives the same `layout_hash` from the same
/// wrong constant. The only symptom is wrong numbers.
#[test]
fn a_design_set_from_the_other_board_is_refused() {
    let npu1_design = Design {
        name: "gemm_rtp".to_string(),
        generation: Generation::Npu1,
        dtype: OperandDtype::Bf16,
        tile_k: 8,
        tile_n: 4,
        streams: vec![],
    };
    let npu2_design = Design {
        name: "gemm_rtp".to_string(),
        generation: Generation::Npu2,
        dtype: OperandDtype::Bf16,
        tile_k: 8,
        tile_n: 8,
        streams: vec![],
    };

    let npu1_set = DesignSet {
        designs: vec![npu1_design.clone()],
    };
    let npu2_set = DesignSet {
        designs: vec![npu2_design.clone()],
    };

    assert!(
        npu1_set.fits(Generation::Npu1, OperandDtype::Bf16).is_ok(),
        "an npu1 set fits an npu1 device"
    );
    assert!(
        npu2_set.fits(Generation::Npu2, OperandDtype::Bf16).is_ok(),
        "an npu2 set fits an npu2 device"
    );

    let wrong = npu1_set
        .fits(Generation::Npu2, OperandDtype::Bf16)
        .expect_err("an npu1 set on an npu2 device is refused");
    assert!(wrong.contains("npu1"), "the refusal names the design: {wrong}");
    assert!(wrong.contains("npu2"), "the refusal names the device: {wrong}");

    let wrong = npu2_set
        .fits(Generation::Npu1, OperandDtype::Bf16)
        .expect_err("an npu2 set on an npu1 device is refused");
    assert!(wrong.contains("npu2"), "the refusal names the design: {wrong}");
    assert!(wrong.contains("npu1"), "the refusal names the device: {wrong}");

    // The single-design check refuses the same way, and also refuses a set
    // whose GEOMETRY happens to agree: the generation stamp is what is
    // compared, not the numbers.
    assert!(
        design_fits(Generation::Npu2, OperandDtype::Bf16, &npu1_design, (8, 8)).is_err(),
        "the geometry agreeing does not make an npu1 design fit an npu2 device"
    );
    assert!(
        design_fits(Generation::Npu1, OperandDtype::Bf16, &npu2_design, (8, 4)).is_err(),
        "the geometry agreeing does not make an npu2 design fit an npu1 device"
    );
}
