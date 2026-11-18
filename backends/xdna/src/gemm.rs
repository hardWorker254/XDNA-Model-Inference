//! The array's GEMM: the op a model actually wants on the board.
//!
//! # WHAT ONE DISPATCH IS
//!
//! `out[0..n_real, n] = a[0..n_real, k] @ b[k, n] + bias[n]`, where `b` is
//! the weight panel the caller has staged and bound, and the design computes
//! `rows` rows whatever the caller has.
//!
//! The C++ runtime's `NpuGemm::run` is the same thing with the same names,
//! and the same three properties:
//!
//! * **THE PADDED TAIL IS ZEROED, NOT STALE.** The design computes all
//!   `rows` rows whatever the A buffer holds in the rest, so a stale row
//!   would put another call's activations into this one's output. The tail
//!   is a padded row, so zero is the only value that cannot be mistaken for
//!   a real position.
//!
//! * **THE BIAS IS ADDED ON THE HOST.** The MMAC multiplies; it does not
//!   add a bias, and the bias is in f32 while the array's output is bf16.
//!   Adding it after the read-back is what keeps the rounding count at one.
//!
//! * **THE ROWS THE DESIGN COMPUTES ARE NOT THE ROWS THE CALLER HAS.**
//!   `n_real` is the caller's, `rows` is the design's, and the design's
//!   number is the one the buffers are sized for.
//!
//! # THE DATAPATH IS CHECKED, NOT ASSUMED
//!
//! The array's MMAC multiplies bf16 and int8. A design whose A operand is
//! int8 needs the container's per-operand scales and the SmoothQuant
//! reciprocals, and the container is the only source of both -- so an int8
//! dispatch is refused here, by name, rather than computed with the scales
//! missing.

use crate::array::{ArrayDesign, ArrayError};
use crate::transport::Transport;

/// The design's datapath: the element sizes the dispatch works in.
///
/// This is the ONE place that decides what the array's operands are, and
/// both the projection and the convolution ask it -- a dispatch that
/// converted its A operand one way and read its C operand another would
/// produce a number that is wrong in a way no shape check would catch.
#[derive(Debug, Clone, Copy)]
pub struct Datapath {
    /// The A operand's element size, in bytes.
    pub a_bytes: usize,
    /// The C operand's element size, in bytes.
    pub c_bytes: usize,
}

impl Datapath {
    /// The design's datapath, checked.
    ///
    /// Only bf16 is dispatched. An int8 design needs the container's
    /// per-operand scales and the SmoothQuant reciprocals, and this runtime
    /// has no source for either yet -- so it says so by name instead of
    /// computing a number with the scales missing.
    pub fn of<T: Transport>(design: &ArrayDesign<T>) -> Result<Self, ArrayError> {
        let info = design.info();
        let a_bytes = elem_bytes(&info.a_dtype).ok_or_else(|| {
            ArrayError::Design(format!("the design's A dtype is {:?}", info.a_dtype))
        })?;
        let c_bytes = elem_bytes(&info.c_dtype).ok_or_else(|| {
            ArrayError::Design(format!("the design's C dtype is {:?}", info.c_dtype))
        })?;
        if a_bytes != 2 || c_bytes != 2 {
            return Err(ArrayError::Design(format!(
                "this runtime dispatches the bf16 datapath only, and the design's \
                 A and C are {} and {} bytes -- an int8 dispatch needs the \
                 container's per-operand scales and the SmoothQuant reciprocals, \
                 and this code has no source for either yet.",
                a_bytes, c_bytes
            )));
        }
        Ok(Self { a_bytes, c_bytes })
    }
}

/// The element size of an operand dtype, as the manifest spells it.
fn elem_bytes(dtype: &str) -> Option<usize> {
    match dtype {
        "bf16" => Some(2),
        "i8" | "int8" => Some(1),
        "f32" => Some(4),
        _ => None,
    }
}

/// Run one GEMM on the array.
///
/// `a` holds `n_real * k` floats; `b` is the weight panel, already staged
/// and bound by the caller; `out` receives `n_real * n` floats.
pub fn gemm<T: Transport>(
    design: &mut ArrayDesign<T>,
    instr: usize,
    a: &[f32],
    n_real: usize,
    rows: usize,
    k: usize,
    bias: Option<&[f32]>,
    n: usize,
    out: &mut [f32],
) -> Result<(), ArrayError> {
    // THE SHAPE, BEFORE THE HARDWARE.
    //
    // These checks are the same on the host path and on the array path, and
    // they come first: a shape bug is a shape bug whichever path the run
    // takes.
    check_shape(a.len(), n_real, rows, k, n, out.len(), bias)?;

    let dp = Datapath::of(design)?;

    // THE FIT, AGAINST THE DESIGN'S OWN BUFFERS.
    //
    // The design was exported for a shape; a shape that does not fit is not
    // a shape this runtime can serve, and saying so is better than a
    // dispatch that overruns a buffer.
    let info = design.info();
    if rows * k * dp.a_bytes > info.buffers[0] as usize
        || rows * n * dp.c_bytes > info.buffers[2] as usize
    {
        return Err(ArrayError::Design(format!(
            "{rows}x{k} A or {rows}x{n} C does not fit the design's buffers -- \
             the design was exported for a different shape (tile {}x{}x{})",
            info.tile.m, info.tile.k, info.tile.n
        )));
    }

    // THE A OPERAND, IN THE DESIGN'S OWN DTYPE.
    //
    // The real rows first, then the padded tail as zeros.
    let a_packed = pack_a(a, n_real, rows, k, dp.a_bytes);

    dispatch(design, instr, &a_packed, n_real, rows, dp, bias, n, out)
}

/// The A operand in the design's own dtype.
///
/// The real rows first, then the padded tail as zeros. The tail is a padded
/// ROW: the design computes all `rows` rows whatever the A buffer holds in
/// the rest, and zero is the only value that cannot be mistaken for a real
/// position.
pub(crate) fn pack_a(a: &[f32], n_real: usize, rows: usize, k: usize, a_bytes: usize) -> Vec<u8> {
    let mut packed = vec![0u8; rows * k * a_bytes];
    npum_format::bf16::f32_to_bf16_bytes(&a[..n_real * k], &mut packed);
    packed
}

/// The dispatch itself: the A operand is already in the design's dtype.
///
/// This is the shape both the projection and the convolution dispatch, and
/// it is one function so that the two cannot disagree about what a dispatch
/// is -- the convolution's padded rows and the projection's padded tail are
/// the same mechanism, and a second copy of it would be a second thing to be
/// wrong about.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch<T: Transport>(
    design: &mut ArrayDesign<T>,
    instr: usize,
    a_packed: &[u8],
    n_real: usize,
    rows: usize,
    dp: Datapath,
    bias: Option<&[f32]>,
    n: usize,
    out: &mut [f32],
) -> Result<(), ArrayError> {
    // THE DISPATCH WINDOW.
    //
    // Bind, sync, dispatch, sync back -- all of it before the host reads
    // the output. A buffer that was written through the host map and not
    // synced is still dirty in the host cache, and the device then reads
    // whatever the host last flushed, which looks exactly like a design that
    // computed nothing.
    design.write_bound(0, a_packed)?;
    design.dispatch(instr)?;

    // THE OUTPUT, BACK IN f32, WITH THE BIAS.
    //
    // ONE ROW AT A TIME, and the row is widened in blocks: the widening is
    // the other half of the boundary the A operand crossed, and at the
    // array's volume it is hundreds of millions of values.
    let mut c_packed = vec![0u8; rows * n * dp.c_bytes];
    design.read(2, &mut c_packed)?;
    let mut row = vec![0f32; n];
    for r in 0..n_real {
        let src = r * n * dp.c_bytes;
        npum_format::bf16::bf16_to_f32_bytes(&c_packed[src..src + n * dp.c_bytes], &mut row[..n]);
        match bias {
            Some(bias) => {
                for j in 0..n {
                    out[r * n + j] = row[j] + bias[j];
                }
            }
            None => {
                out[r * n..r * n + n].copy_from_slice(&row[..n]);
            }
        }
    }
    Ok(())
}

/// The shape a dispatch needs, checked before the hardware is touched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_shape(
    a_len: usize,
    n_real: usize,
    rows: usize,
    k: usize,
    n: usize,
    out_len: usize,
    bias: Option<&[f32]>,
) -> Result<(), ArrayError> {
    if n_real == 0 || n_real > rows {
        return Err(ArrayError::Design(format!(
            "{n_real} real rows into a {rows}-row dispatch"
        )));
    }
    if rows == 0 || k == 0 || n == 0 {
        return Err(ArrayError::Design(
            "a GEMM with a zero dimension".to_string(),
        ));
    }
    if a_len < n_real * k {
        return Err(ArrayError::Design(format!(
            "the A operand holds {a_len} floats, and {n_real} rows of {k} need {}",
            n_real * k
        )));
    }
    if out_len < n_real * n {
        return Err(ArrayError::Design(format!(
            "the output holds {out_len} floats, and {n_real} rows of {n} need {}",
            n_real * n
        )));
    }
    if let Some(bias) = bias {
        if bias.len() != n {
            return Err(ArrayError::Design(format!(
                "the bias holds {} values, and the output has {n} columns",
                bias.len()
            )));
        }
    }
    Ok(())
}
