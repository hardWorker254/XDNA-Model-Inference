//! The B panel's tiling, against the packer's own bytes.
//!
//! # WHY THE GOLDEN IS BYTES AND NOT A DESCRIPTION
//!
//! The tiling is a byte order, and a byte order cannot be checked by
//! describing it: two implementations that both "gather a tile and reorder
//! it inside the sub-tile" can disagree about the order and both be
//! described the same way. So the golden here is what the PYTHON PACKER
//! produces -- the implementation the containers in the tree were written by
//! -- and this test says so in its name.
//!
//! The operand is a [8, 8] f32 matrix whose element (i, j) is (i * 8 + j) / 7,
//! tiled into (4, 4) tiles with the (2, 2) sub-tile of npu1's bf16 MMAC.

use npum_format::tiling::{tile_b, untile_b, TileOrder};

/// The operand, row-major: element (i, j) is (i * 8 + j) / 7.
const INPUT: [u8; 256] = [
    0, 0, 0, 0, 37, 73, 18, 62, 37, 73, 146, 62, 183, 109, 219, 62, 37, 73, 18, 63, 110, 219, 54, 63,
    183, 109, 91, 63, 0, 0, 128, 63, 37, 73, 146, 63, 73, 146, 164, 63, 110, 219, 182, 63, 146, 36, 201,
    63, 183, 109, 219, 63, 219, 182, 237, 63, 0, 0, 0, 64, 146, 36, 9, 64, 37, 73, 18, 64, 183, 109, 27,
    64, 73, 146, 36, 64, 219, 182, 45, 64, 110, 219, 54, 64, 0, 0, 64, 64, 146, 36, 73, 64, 37, 73, 82,
    64, 183, 109, 91, 64, 73, 146, 100, 64, 219, 182, 109, 64, 110, 219, 118, 64, 0, 0, 128, 64, 73, 146,
    132, 64, 146, 36, 137, 64, 219, 182, 141, 64, 37, 73, 146, 64, 110, 219, 150, 64, 183, 109, 155, 64,
    0, 0, 160, 64, 73, 146, 164, 64, 146, 36, 169, 64, 219, 182, 173, 64, 37, 73, 178, 64, 110, 219, 182,
    64, 183, 109, 187, 64, 0, 0, 192, 64, 73, 146, 196, 64, 146, 36, 201, 64, 219, 182, 205, 64, 37, 73,
    210, 64, 110, 219, 214, 64, 183, 109, 219, 64, 0, 0, 224, 64, 73, 146, 228, 64, 146, 36, 233, 64, 219,
    182, 237, 64, 37, 73, 242, 64, 110, 219, 246, 64, 183, 109, 251, 64, 0, 0, 0, 65, 37, 73, 2, 65, 73,
    146, 4, 65, 110, 219, 6, 65, 146, 36, 9, 65, 183, 109, 11, 65, 219, 182, 13, 65, 0, 0, 16, 65,
];

/// What the Python packer produces for it.
const GOLDEN: [u8; 256] = [
    0, 0, 0, 0, 37, 73, 18, 62, 37, 73, 146, 63, 73, 146, 164, 63, 37, 73, 146, 62, 183, 109, 219, 62,
    110, 219, 182, 63, 146, 36, 201, 63, 37, 73, 18, 64, 183, 109, 27, 64, 183, 109, 91, 64, 73, 146, 100,
    64, 73, 146, 36, 64, 219, 182, 45, 64, 219, 182, 109, 64, 110, 219, 118, 64, 37, 73, 18, 63, 110, 219,
    54, 63, 183, 109, 219, 63, 219, 182, 237, 63, 183, 109, 91, 63, 0, 0, 128, 63, 0, 0, 0, 64, 146, 36, 9,
    64, 110, 219, 54, 64, 0, 0, 64, 64, 0, 0, 128, 64, 73, 146, 132, 64, 146, 36, 73, 64, 37, 73, 82, 64,
    146, 36, 137, 64, 219, 182, 141, 64, 37, 73, 146, 64, 110, 219, 150, 64, 110, 219, 182, 64, 183, 109,
    187, 64, 183, 109, 155, 64, 0, 0, 160, 64, 0, 0, 192, 64, 73, 146, 196, 64, 183, 109, 219, 64, 0, 0,
    224, 64, 0, 0, 0, 65, 37, 73, 2, 65, 73, 146, 228, 64, 146, 36, 233, 64, 73, 146, 4, 65, 110, 219, 6,
    65, 73, 146, 164, 64, 146, 36, 169, 64, 146, 36, 201, 64, 219, 182, 205, 64, 219, 182, 173, 64, 37, 73,
    178, 64, 37, 73, 210, 64, 110, 219, 214, 64, 219, 182, 237, 64, 37, 73, 242, 64, 146, 36, 9, 65, 183,
    109, 11, 65, 110, 219, 246, 64, 183, 109, 251, 64, 219, 182, 13, 65, 0, 0, 16, 65,
];

fn tiled() -> Vec<u8> {
    tile_b(&INPUT, 8, 8, 4, 4, 4, Some(2), Some(2), TileOrder::Kn)
        .expect("the tile divides the operand")
}

#[test]
fn the_tiling_is_the_packers_own_bytes() {
    let got = tiled();
    assert_eq!(got.len(), GOLDEN.len());
    if got != GOLDEN {
        let first = got
            .iter()
            .zip(GOLDEN.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "the tiling differs from the packer's at byte {first}: \
             ours {:#04x}, the packer's {:#04x}",
            got[first], GOLDEN[first]
        );
    }
}

#[test]
fn the_round_trip_is_bit_exact() {
    let back = untile_b(&tiled(), 8, 8, 4, 4, 4, Some(2), Some(2), TileOrder::Kn)
        .expect("the panel is the right size");
    assert_eq!(
        back,
        INPUT.to_vec(),
        "the un-tiling is not the exact inverse of the tiling"
    );
}

#[test]
fn a_tile_that_does_not_divide_is_refused() {
    let refused = tile_b(&INPUT, 8, 8, 3, 4, 4, Some(2), Some(2), TileOrder::Kn);
    assert!(refused.is_err(), "a tile of 3 rows must not be accepted");
}
