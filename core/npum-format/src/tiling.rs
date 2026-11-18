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
    let mut out = vec![0u8; b.len()];
    tile_b_into(&mut out, b, k, n, tile_k, tile_n, elem, mac_s, mac_t, order)?;
    Ok(out)
}

/// Pre-tile a [K, N] operand into a provided buffer.
///
/// This is [`tile_b`], but it writes into `out` instead of allocating it.
/// `out` must be exactly `k * n * elem` bytes long: the tiled panel is a
/// permutation of the same bytes, so the caller that reuses the buffer across
/// calls pays the write and not the allocation.
pub fn tile_b_into(
    out: &mut [u8],
    b: &[u8],
    k: usize,
    n: usize,
    tile_k: usize,
    tile_n: usize,
    elem: usize,
    mac_s: Option<usize>,
    mac_t: Option<usize>,
    order: TileOrder,
) -> Result<(), TilingError> {
    if elem == 0 {
        return Err(refuse("the element size is zero"));
    }
    let bytes = k * n * elem;
    if b.len() != bytes {
        return Err(refuse(format!(
            "a [{k},{n}] operand of {elem}-byte elements is {} bytes, and the buffer is {}",
            bytes,
            b.len()
        )));
    }
    if out.len() != bytes {
        return Err(refuse(format!(
            "the output holds {} bytes, and a [{k},{n}] tiled panel needs {bytes}",
            out.len()
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

    // THE ELEMENT SIZE IS NEARLY ALWAYS ONE OF THREE CONSTANTS.
    //
    // bf16 (elem == 2) is the array's operand and the hot case: the attention
    // rebuilds both panels for every head. Passing a LITERAL lets the emitter
    // fold the size into its address arithmetic (a shift, not a multiply) and
    // make each coalesced copy a fixed-width load/store. The fallback keeps
    // every other element size working, with the same bytes.
    match elem {
        1 => emit_tiles(out, b, n, tile_k, tile_n, kb, nb, s, t, 1, order),
        2 => emit_tiles(out, b, n, tile_k, tile_n, kb, nb, s, t, 2, order),
        4 => emit_tiles(out, b, n, tile_k, tile_n, kb, nb, s, t, 4, order),
        _ => emit_tiles(out, b, n, tile_k, tile_n, kb, nb, s, t, elem, order),
    }
    Ok(())
}

/// The emit loops, shared by the specializations above.
///
/// Marked `#[inline(always)]` so that the literal element size in each caller
/// is folded into the address math before the loops are code-generated.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn emit_tiles(
    out: &mut [u8],
    b: &[u8],
    n: usize,
    tile_k: usize,
    tile_n: usize,
    kb: usize,
    nb: usize,
    s: usize,
    t: usize,
    elem: usize,
    order: TileOrder,
) {
    // THE t ELEMENTS OF A SUB-TILE ROW ARE ONE CONTIGUOUS SOURCE RUN.
    //
    // The sub-tile's own order is (si, ti): within a fixed (kk, nn, si) the
    // `ti` elements are consecutive COLUMNS of one source row, hence
    // consecutive bytes of the source. Emitting them as one `t * elem` copy
    // replaces `t` indexed `elem`-byte copies -- four times fewer copies and
    // bounds checks for npu1's bf16 sub-tile (t=4), and one 64-bit load/store
    // per run for bf16. Neither the output order nor a byte changes: this is
    // the same sequence of source bytes, only fetched in larger pieces.
    let run = t * elem;
    let row_stride = n * elem;
    let mut write = 0usize;

    let mut emit_tile = |kb_i: usize, nb_i: usize| {
        for kk in 0..(tile_k / s) {
            // Byte offset of the first `si` row of this sub-tile row. Hoisted
            // out of both inner loops, as is the column base below: what was
            // a multiply and two adds per element is now one add per run.
            let row_base = (kb_i * tile_k + kk * s) * row_stride;
            for nn in 0..(tile_n / t) {
                let col_base = (nb_i * tile_n + nn * t) * elem;
                for si in 0..s {
                    let src = row_base + si * row_stride + col_base;
                    out[write..write + run].copy_from_slice(&b[src..src + run]);
                    write += run;
                }
            }
        }
    };
    match order {
        TileOrder::Kn => {
            for kb_i in 0..kb {
                for nb_i in 0..nb {
                    emit_tile(kb_i, nb_i);
                }
            }
        }
        TileOrder::Nk => {
            for nb_i in 0..nb {
                for kb_i in 0..kb {
                    emit_tile(kb_i, nb_i);
                }
            }
        }
    }
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
