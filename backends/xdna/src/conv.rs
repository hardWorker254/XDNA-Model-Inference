//! The array's convolution: a dense convolution as a GEMM on the board.
//!
//! # WHAT A CONVOLUTION IS HERE
//!
//! A dense convolution is an im2col on the host and a GEMM on the array:
//! the host gathers the patches into rows of `K = cin * kh * kw` values, and
//! the array multiplies those rows by the weight panel. The reference runtime
//! splits it the same way (`conv::conv_array_dispatch`), and the split is
//! not a convenience: the array's MMAC multiplies matrices, and the im2col
//! is the only part of a convolution that is not a matrix multiplication.
//!
//! # THE TWO WIDTHS, AND WHY BOTH ARE NEEDED
//!
//! A convolution has its own shape -- `K = cin * kh * kw` and `cout`
//! channels -- and the DESIGN has a padded one: the compiled core runs at a
//! handful of (K, N) pairs, and a convolution is dispatched against the
//! smallest pair that holds it. So the A rows are padded to the design's K,
//! and the result is read back at the design's N and narrowed to the
//! convolution's own.
//!
//! The padding is not bookkeeping. The A operand's padded COLUMNS are
//! multiplied by the panel's padded rows, and a stale value there is a
//! wrong number rather than a wrong shape -- which is harder to see than a
//! wrong shape is. So the padded columns are zeroed, and they are zeroed
//! FIRST and copied SECOND, exactly as the reference runtime does it.
//!
//! # THE BIAS IS PADDED TOO
//!
//! The bias is the convolution's own -- `cout` values -- and the design's
//! output has `padded_n` columns. The tail of the padded bias is zero,
//! which is exact: those columns are channels the tensor does not have, and
//! the host narrows the result back to the real N on the way out.
//!
//! # THE ROWS ARE CHUNKED
//!
//! The design computes a fixed number of rows per dispatch, and a
//! convolution can have more rows than that. The rows go in chunks of the
//! design's own, and each chunk is one dispatch.
//!
//! # THE SCRATCH IS THE CALLER'S, AND IT IS REUSED
//!
//! The detector's stem convolution gathers 230 400 rows of 27 values and
//! dispatches them in 225 chunks of the design's 1024 rows. A scratch that
//! was allocated and zeroed per chunk would cost more than the device
//! does: the zeroing alone is megabytes per chunk, and the device's own
//! time for the whole detector is under half a second.
//!
//! So the scratch belongs to the caller, it grows when it has to, and only
//! the CHUNK'S OWN ROWS are touched per dispatch. The rows past the chunk
//! keep their previous values, and that is safe: the design's GEMM is
//! ROW-WISE, so a stale row can only reach its own output row, and the
//! output rows past the chunk are not read.

use crate::array::{ArrayDesign, ArrayError};
use crate::gemm::{Datapath, dispatch};
use crate::transport::Transport;

/// A convolution's own shape, and the design's padded one.
///
/// The caller fills it from two places: the convolution's own geometry, and
/// the design's stream -- the stream's name IS its shape (`conv64x128` ✓),
/// and its K and N are the padded widths.
#[derive(Debug, Clone, Copy)]
pub struct ConvShape {
    /// The rows one dispatch computes -- the design's own M.
    pub rows: usize,
    /// The convolution's own K: the im2col row's length.
    pub k: usize,
    /// The convolution's own output channels.
    pub n: usize,
    /// The design's padded K.
    pub padded_k: usize,
    /// The design's padded N.
    pub padded_n: usize,
}

impl ConvShape {
    /// The shape, checked against itself.
    ///
    /// The padded widths must HOLD the real ones: a design whose K is
    /// smaller than the convolution's would truncate the panel, and the
    /// dispatch would read past its end and return plausible products.
    pub fn checked(&self) -> Result<(), ArrayError> {
        if self.rows == 0 || self.k == 0 || self.n == 0 {
            return Err(ArrayError::Design(
                "a convolution with a zero dimension".to_string(),
            ));
        }
        if self.padded_k < self.k || self.padded_n < self.n {
            return Err(ArrayError::Design(format!(
                "the design's padded shape {}x{} is smaller than the \
                 convolution's own {}x{}, so the panel would be truncated. The \
                 design set belongs to another model, or its B-operand layout \
                 does not match this container's.",
                self.padded_k, self.padded_n, self.k, self.n
            )));
        }
        Ok(())
    }
}

/// The buffers one convolution's dispatch needs.
///
/// They are the caller's, and they are REUSED across the chunks and across
/// the convolutions: the sizes grow to the largest shape and stay there.
#[derive(Debug, Default)]
pub struct ConvScratch {
    /// The padded A operand, in the design's dtype.
    a_packed: Vec<u8>,
    /// The padded C operand, in f32.
    c: Vec<f32>,
    /// The conversion's own time, in seconds.
    ///
    /// It is a REPORT and not a control: the array's time is what the run
    /// costs, and this is the part of it that the host's own arithmetic
    /// spends turning the host's activations into the device's dtype.
    pub convert_seconds: f64,
}

impl ConvScratch {
    /// A scratch with room for `rows` rows of `k` columns, in f32.
    pub fn new(rows: usize, k: usize) -> Self {
        Self {
            a_packed: vec![0u8; rows * k * 2],
            c: Vec::new(),
            convert_seconds: 0.0,
        }
    }
}

/// Run one dense convolution on the array.
///
/// `a` holds the host's im2col rows -- `m` rows of `shape.k` values -- and
/// `out` receives `m` rows of `shape.n` values. The weight panel is staged
/// and bound by the caller, and it is the container's own: pre-tiled into
/// the MMAC's byte order and padded to the design's widths.
///
/// The number of dispatches is returned, because it is not the number of
/// convolutions: a convolution with more rows than the design computes is
/// several dispatches, and the count that says "the flag did what it said"
/// is the one the device saw.
pub fn conv<T: Transport>(
    design: &mut ArrayDesign<T>,
    instr: usize,
    shape: &ConvShape,
    bias: Option<&[f32]>,
    a: &[f32],
    out: &mut [f32],
    scratch: &mut ConvScratch,
) -> Result<usize, ArrayError> {
    shape.checked()?;
    if let Some(bias) = bias {
        if bias.len() != shape.n {
            return Err(ArrayError::Design(format!(
                "the bias holds {} values, and the convolution has {} output channels",
                bias.len(),
                shape.n
            )));
        }
    }
    let m = a.len() / shape.k;
    if m == 0 || a.len() % shape.k != 0 {
        return Err(ArrayError::Design(format!(
            "the im2col holds {} values, which is not a whole number of rows of {}",
            a.len(),
            shape.k
        )));
    }
    if out.len() < m * shape.n {
        return Err(ArrayError::Design(format!(
            "the output holds {} floats, and {} rows of {} need {}",
            out.len(),
            m,
            shape.n,
            m * shape.n
        )));
    }

    let dp = Datapath::of(design)?;

    // THE PADDED BIAS, ONCE.
    //
    // The design's output has `padded_n` columns and the convolution has
    // `n` channels; the tail is zero, which is exact.
    let padded_bias: Option<Vec<f32>> = bias.map(|bias| {
        let mut padded = vec![0f32; shape.padded_n];
        padded[..shape.n].copy_from_slice(bias);
        padded
    });

    // THE SCRATCH, GROWN TO THE DESIGN'S SHAPE.
    let a_need = shape.rows * shape.padded_k * dp.a_bytes;
    if scratch.a_packed.len() < a_need {
        scratch.a_packed.resize(a_need, 0);
    }
    let c_need = shape.rows * shape.padded_n;
    if scratch.c.len() < c_need {
        scratch.c.resize(c_need, 0.0);
    }

    // THE CHUNKS.
    //
    // The design computes `rows` rows per dispatch, and a convolution can
    // have more rows than that.
    let mut done = 0usize;
    let mut dispatches = 0usize;
    while done < m {
        let chunk = (m - done).min(shape.rows);

        // THE PADDED A, ZEROED FIRST AND COPIED SECOND.
        //
        // The padded COLUMNS are multiplied by the panel's padded rows, and
        // a stale value there is a wrong number rather than a wrong shape.
        // Only the CHUNK'S OWN ROWS are touched: the rows past the chunk
        // keep their old values, and the design's GEMM is row-wise, so they
        // can only reach their own output rows -- which are not read.
        //
        // THE ZEROING IS INSIDE THE TIMER, because it is the host's work and
        // not the device's: it is the padded columns being prepared for the
        // device, and a "time on the device" that carried it would report the
        // host's bookkeeping as the device's arithmetic.
        let row_bytes = shape.padded_k * dp.a_bytes;
        let t_convert = std::time::Instant::now();
        scratch.a_packed[..chunk * row_bytes].fill(0);
        for r in 0..chunk {
            let src = (done + r) * shape.k;
            let dst = r * row_bytes;
            // ONE ROW AT A TIME, because the row is the unit the padding
            // breaks: a row's values are contiguous on both sides, and the
            // columns past the convolution's own K are zeros.
            npum_format::bf16::f32_to_bf16_bytes(
                &a[src..src + shape.k],
                &mut scratch.a_packed[dst..dst + shape.k * 2],
            );
        }
        scratch.convert_seconds += t_convert.elapsed().as_secs_f64();

        // THE DISPATCH.
        //
        // The design's own K and N, the chunk's rows, and the padded bias.
        dispatch(
            design,
            instr,
            &scratch.a_packed,
            chunk,
            shape.rows,
            dp,
            padded_bias.as_deref(),
            shape.padded_n,
            &mut scratch.c,
        )?;

        // THE NARROWING BACK.
        //
        // The design's N columns, of which the convolution has its own.
        for r in 0..chunk {
            let src = r * shape.padded_n;
            let dst = (done + r) * shape.n;
            out[dst..dst + shape.n].copy_from_slice(&scratch.c[src..src + shape.n]);
        }

        done += chunk;
        dispatches += 1;
    }
    Ok(dispatches)
}

/// The A operand of a convolution, in the design's own dtype, without a
/// board.
///
/// The same RNE rounding and the same padding the dispatch does, so that a
/// tool can prepare an operand and a test can compare it with what the
/// board saw.
pub fn pack_im2col_rows(a: &[f32], rows: usize, k: usize, padded_k: usize) -> Vec<u8> {
    let mut packed = vec![0u8; rows * padded_k * 2];
    for r in 0..rows {
        npum_format::bf16::f32_to_bf16_bytes(
            &a[r * k..r * k + k],
            &mut packed[r * padded_k * 2..r * padded_k * 2 + k * 2],
        );
    }
    packed
}
