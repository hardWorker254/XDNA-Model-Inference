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

/// f32 to bf16, round to nearest even.
///
/// The same rounding the C++ runtime's `bf16_fill` does: add 0x7fff and the
/// low bit of the mantissa's top half, then take the top half. Round-to-
/// nearest-EVEN, so a value exactly halfway between two bf16 neighbours
/// goes to the even one -- and a value that is exactly halfway is the case
/// that decides a number in the last digit.
fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounding = 0x7fffu64 + ((bits >> 16) & 1) as u64;
    ((bits as u64 + rounding) >> 16) as u16
}

/// bf16 to f32.
fn from_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
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
    // These three checks are the same on the host path and on the array
    // path, and they come first: a shape bug is a shape bug whichever path
    // the run takes.
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
    if a.len() < n_real * k {
        return Err(ArrayError::Design(format!(
            "the A operand holds {} floats, and {n_real} rows of {k} need {}",
            a.len(),
            n_real * k
        )));
    }
    if out.len() < n_real * n {
        return Err(ArrayError::Design(format!(
            "the output holds {} floats, and {n_real} rows of {n} need {}",
            out.len(),
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

    // THE DATAPATH.
    let info = design.info().clone();
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

    // THE FIT, AGAINST THE DESIGN'S OWN BUFFERS.
    //
    // The design was exported for a shape; a shape that does not fit is not
    // a shape this runtime can serve, and saying so is better than a
    // dispatch that overruns a buffer.
    if rows * k * a_bytes > info.buffers[0] as usize
        || rows * n * c_bytes > info.buffers[2] as usize
    {
        return Err(ArrayError::Design(format!(
            "{rows}x{k} A or {rows}x{n} C does not fit the design's buffers -- \
             the design was exported for a different shape (M {}, seq {})",
            info.tile.m, info.name
        )));
    }

    // THE A OPERAND, IN THE DESIGN'S OWN DTYPE.
    //
    // The real rows first, then the padded tail as zeros.
    let mut a_packed = vec![0u8; rows * k * a_bytes];
    for i in 0..(n_real * k) {
        a_packed[i * 2..i * 2 + 2].copy_from_slice(&to_bf16(a[i]).to_le_bytes());
    }
    design.write_bound(0, &a_packed)?;

    // THE DISPATCH WINDOW.
    //
    // Bind, sync, dispatch, sync back -- all of it before the host reads
    // the output. A buffer that was written through the host map and not
    // synced is still dirty in the host cache, and the device then reads
    // whatever the host last flushed, which looks exactly like a design that
    // computed nothing.
    design.dispatch(instr)?;

    // THE OUTPUT, BACK IN f32, WITH THE BIAS.
    let mut c_packed = vec![0u8; rows * n * c_bytes];
    design.read(2, &mut c_packed)?;
    for r in 0..n_real {
        for j in 0..n {
            let bits = u16::from_le_bytes([c_packed[(r * n + j) * 2], c_packed[(r * n + j) * 2 + 1]]);
            let mut value = from_bf16(bits);
            if let Some(bias) = bias {
                value += bias[j];
            }
            out[r * n + j] = value;
        }
    }
    Ok(())
}
