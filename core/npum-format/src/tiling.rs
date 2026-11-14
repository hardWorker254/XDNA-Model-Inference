//! The pre-tiled B panel: the order the DMA streams a weight operand in.
//!
//! # WHAT THE TILING DOES
//!
//! A B panel for the array is stored in the MMAC's own byte order inside one
//! (tile_k, tile_n) tile. It absorbs two re-layouts that the design used to
//! do at RUN time:
//!
//! 1. **L3 -> L2:** gathering a (tile_k, tile_n) tile out of a row-major
//!    [K, N] buffer. This is the strided access pattern whose dimension hits
//!    the 10-bit (max 1023) DMA BD size field.
//! 2. **L2 -> L1:** the intrinsic sub-tile order, which the design expressed
//!    as `dims_to_stream=[(k//s, s*n), (n//t, t), (s, n), (t, 1)]`.
//!
//! The output is a flat run of tiles in (kb, nb) order, each tile internally
//! in (s, t) order when the sub-tile is given.
//!
//! # WHY THIS LIVES IN THE FORMAT CRATE
//!
//! Because the bytes are the FORMAT. A container that carries a pre-tiled
//! panel has to agree with the array's reader about the order, and the one
//! place where that agreement can be checked is the crate both of them use.
//! The Python packer has its own copy of this function, and that copy is to
//! be replaced by a call into this crate -- the packer already loads this
//! crate's cdylib for the writer, and two implementations of one byte order
//! is the failure mode the writer's own comment is about.
//!
//! # THE SUB-TILE IS NOT A DEFAULT
//!
//! `s` and `t` are the MMAC's sub-tile, and they depend on the BOARD and on
//! the operand dtype: bf16 on npu1 is (8, 4) and int8 on npu1 is (8, 8). A
//! default here would be a wrong answer that is the right size, so the
//! caller passes them and the function refuses a tile they do not divide.

/// The order the tiles are laid out in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileOrder {
    /// The k-blocks are major: the tiles of one n-block are together.
    ///
    /// This is the order the format's layout descriptor records -- `k,n,kt,nt`
    /// -- and the one a container's panels are stored in.
    Kn,
    /// The n-blocks are major.
    ///
    /// A PERFORMANCE variant of the same memory: a core's inner loop walks
    /// all the k-blocks of one n-block, and with this order those tiles are
    /// one contiguous run.
    Nk,
}

/// The error a tiling cannot paper over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilingError {
    /// What could not be done, and why.
    pub message: String,
}

impl std::fmt::Display for TilingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TilingError {}

fn refuse(message: impl Into<String>) -> TilingError {
    TilingError {
        message: message.into(),
    }
}

/// Pre-tile a [K, N] operand of `elem` -byte elements into the array's order.
///
/// `b` is row-major: `k * n + j` is the element at row `k`, column `j`.
///
/// `mac_s` and `mac_t` are the MMAC's sub-tile, or `None` to leave the tile
/// in its plain (tile_k, tile_n) order.
pub fn tile_b(
    b: &[u8],
    k: usize,
    n: usize,
    tile_k: usize,
    tile_n: usize,
    elem: usize,
    mac_s: Option<usize>,
    mac_t: Option<usize>,
    order: TileOrder,
) -> Result<Vec<u8>, TilingError> {
    if elem == 0 {
        return Err(refuse("the element size is zero"));
    }
    if b.len() != k * n * elem {
        return Err(refuse(format!(
            "a [{k},{n}] operand of {elem}-byte elements is {} bytes, and the buffer is {}",
            k * n * elem,
            b.len()
        )));
    }
    if k % tile_k != 0 || n % tile_n != 0 {
        return Err(refuse(format!(
            "[{k},{n}] does not tile into ({tile_k},{tile_n})"
        )));
    }
    if (mac_s.is_some()) != (mac_t.is_some()) {
        return Err(refuse("the sub-tile is half given: s and t come together"));
    }
    let (s, t) = match (mac_s, mac_t) {
        (Some(s), Some(t)) => {
            if tile_k % s != 0 || tile_n % t != 0 {
                return Err(refuse(format!(
                    "the tile ({tile_k},{tile_n}) is not divisible by the sub-tile ({s},{t})"
                )));
            }
            (s, t)
        }
        (None, None) => (1, 1),
        _ => unreachable!("the half-given case was refused above"),
    };

    let kb = k / tile_k;
    let nb = n / tile_n;
    let mut out = Vec::with_capacity(b.len());

    // THE TILES, IN THE ORDER THE DESIGN WALKS THEM.
    //
    // The inner loop is the (s, t) order: within one tile, the element at
    // (kk, nn) sits at (kk * s + si, nn * t + ti) -- the sub-tile's own
    // order, which is what the DMA streams.
    let emit_tile = |kb_i: usize, nb_i: usize, out: &mut Vec<u8>| {
        for kk in 0..(tile_k / s) {
            for nn in 0..(tile_n / t) {
                for si in 0..s {
                    for ti in 0..t {
                        let src_row = kb_i * tile_k + kk * s + si;
                        let src_col = nb_i * tile_n + nn * t + ti;
                        let src = (src_row * n + src_col) * elem;
                        out.extend_from_slice(&b[src..src + elem]);
                    }
                }
            }
        }
    };
    match order {
        TileOrder::Kn => {
            for kb_i in 0..kb {
                for nb_i in 0..nb {
                    emit_tile(kb_i, nb_i, &mut out);
                }
            }
        }
        TileOrder::Nk => {
            for nb_i in 0..nb {
                for kb_i in 0..kb {
                    emit_tile(kb_i, nb_i, &mut out);
                }
            }
        }
    }
    Ok(out)
}

/// Un-tile a panel: the exact inverse of [`tile_b`].
///
/// The round trip is BIT-EXACT and not close, which is what makes it a test
/// of the tiling and not a test of a similarity.
pub fn untile_b(
    flat: &[u8],
    k: usize,
    n: usize,
    tile_k: usize,
    tile_n: usize,
    elem: usize,
    mac_s: Option<usize>,
    mac_t: Option<usize>,
    order: TileOrder,
) -> Result<Vec<u8>, TilingError> {
    if flat.len() != k * n * elem {
        return Err(refuse(format!(
            "a [{k},{n}] panel of {elem}-byte elements is {} bytes, and the buffer is {}",
            k * n * elem,
            flat.len()
        )));
    }
    if k % tile_k != 0 || n % tile_n != 0 {
        return Err(refuse(format!(
            "[{k},{n}] does not tile into ({tile_k},{tile_n})"
        )));
    }
    let (s, t) = match (mac_s, mac_t) {
        (Some(s), Some(t)) => (s, t),
        (None, None) => (1, 1),
        _ => return Err(refuse("the sub-tile is half given: s and t come together")),
    };

    let kb = k / tile_k;
    let nb = n / tile_n;
    let mut out = vec![0u8; flat.len()];
    let mut read = 0usize;
    let read_tile = |kb_i: usize, nb_i: usize, flat: &[u8], read: &mut usize, out: &mut Vec<u8>| {
        for kk in 0..(tile_k / s) {
            for nn in 0..(tile_n / t) {
                for si in 0..s {
                    for ti in 0..t {
                        let src_row = kb_i * tile_k + kk * s + si;
                        let src_col = nb_i * tile_n + nn * t + ti;
                        let dst = (src_row * n + src_col) * elem;
                        out[dst..dst + elem].copy_from_slice(&flat[*read..*read + elem]);
                        *read += elem;
                    }
                }
            }
        }
    };
    match order {
        TileOrder::Kn => {
            for kb_i in 0..kb {
                for nb_i in 0..nb {
                    read_tile(kb_i, nb_i, flat, &mut read, &mut out);
                }
            }
        }
        TileOrder::Nk => {
            for nb_i in 0..nb {
                for kb_i in 0..kb {
                    read_tile(kb_i, nb_i, flat, &mut read, &mut out);
                }
            }
        }
    }
    Ok(out)
}
