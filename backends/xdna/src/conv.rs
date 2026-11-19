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
//! wrong shape is. So the padded columns are zeroed. The zeroing happens in
//! the same pass as the conversion now -- each design row is written once,
//! real bytes then tail bytes -- which is the same result the reference
//! runtime got by zeroing first and copying second, in one visit instead of
//! two.
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
//! # THE CHUNKS ARE PIPELINED
//!
//! For a convolution with many chunks the three steps of a chunk -- pack A on
//! the host, run the GEMM on the device, read and narrow C -- used to be one
//! after another, with the host idle while the device worked and the device
//! idle while the host packed and read.
//!
//! They now OVERLAP. Buffer 0 (A) and buffer 2 (C) get a second slot each, and
//! the two slots alternate per chunk. While chunk N's GEMM is in flight, the
//! host packs and uploads A for chunk N+1 into the other slot; when chunk N
//! finishes, its C is read and narrowed from its own slot while chunk N+1 is
//! already running. A slot is never reused until the dispatch that owned it
//! has been waited on and, for C, read back. The bindings a dispatch captured
//! at `start_dispatch` are its own, so rebinding between the start and the
//! read-back is safe.
//!
//! This is the same shape of overlap `attention.rs` uses for its panel pairs.
//! It matters on the vision families, where one convolution is hundreds of
//! chunks: OCR's detector alone cuts a 960x960 map into 900.
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
use crate::gemm::{Datapath, GemmScratch, read_dispatched, submit_packed};
use crate::transport::{Run, Transport};

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
    /// The bias widened to the design's padded width, reused across calls.
    padded_bias: Vec<f32>,
    /// Read-back and widening buffers for each dispatch.
    gemm: GemmScratch,
    /// The conversion's own time, in seconds.
    ///
    /// It is a REPORT and not a control: the array's time is what the run
    /// costs, and this is the part of it that the host's own arithmetic
    /// spends turning the host's activations into the device's dtype.
    pub convert_seconds: f64,
}

impl ConvScratch {
    /// A scratch with room for `rows` rows of `k` columns, in bf16.
    ///
    /// This is sized to the convolution's OWN K. The device reads the design's
    /// PADDED widths, so [`Self::ensure`] grows it to those before a dispatch.
    pub fn new(rows: usize, k: usize) -> Self {
        Self {
            a_packed: vec![0u8; rows * k * 2],
            c: Vec::new(),
            padded_bias: Vec::new(),
            gemm: GemmScratch::new(),
            convert_seconds: 0.0,
        }
    }

    /// Grow the buffers to the design's PADDED shape.
    ///
    /// `new(rows, k)` is sized to the convolution's own K, but the A operand
    /// the device reads has `padded_k` columns per row and the C buffer holds
    /// `padded_n` per row -- both can be wider. Every dispatch calls this
    /// first, so a scratch reused across convolutions of different shapes is
    /// grown once and then reused, and never indexed past its end.
    pub fn ensure(&mut self, rows: usize, padded_k: usize, padded_n: usize, a_bytes: usize) {
        let a_need = rows * padded_k * a_bytes;
        if self.a_packed.len() < a_need {
            self.a_packed.resize(a_need, 0);
        }
        let c_need = rows * padded_n;
        if self.c.len() < c_need {
            self.c.resize(c_need, 0.0);
        }
    }
}

/// The one dispatch that is in flight between pipeline stages.
///
/// `c_slot` is the slot its C operand is written into. It is kept here because
/// the binding is rebound for the NEXT dispatch before this one's output is
/// read, and the read must go back to the slot this dispatch actually used.
struct InFlight {
    run: Run,
    c_slot: usize,
    chunk: usize,
    done: usize,
}

/// Pack one chunk's im2col rows into the host's A buffer, padded and bf16.
///
/// The padded COLUMNS are multiplied by the panel's padded rows, and a stale
/// value there is a wrong number rather than a wrong shape. The conversion
/// writes the real columns and the zero tail together, so no separate fill of
/// the chunk is needed. Only the CHUNK'S OWN ROWS are touched: the rows past
/// the chunk keep their old values, and the design's GEMM is row-wise, so they
/// can only reach their own output rows -- which are not read.
fn pack_chunk(
    a: &[f32],
    done: usize,
    chunk: usize,
    shape: &ConvShape,
    row_bytes: usize,
    scratch: &mut ConvScratch,
) {
    // INSIDE THE TIMER: this is the host's work, not the device's. It is the
    // padding being prepared for the device, and a "time on the device" that
    // carried it would report the host's bookkeeping as the device's
    // arithmetic.
    let t_convert = std::time::Instant::now();
    if shape.k == shape.padded_k {
        // NO TAIL: the rows are contiguous on both sides, so the whole chunk
        // is one conversion rather than one per row.
        npum_format::bf16::f32_to_bf16_bytes(
            &a[done * shape.k..(done + chunk) * shape.k],
            &mut scratch.a_packed[..chunk * row_bytes],
        );
    } else {
        // ONE ROW AT A TIME, because the row is the unit the padding breaks:
        // a row's values are contiguous on both sides, and the columns past
        // the convolution's own K are zeros.
        let pad_values = (row_bytes - shape.k * 2) / 2;
        crate::parallel::for_each_row(
            &mut scratch.a_packed[..chunk * row_bytes],
            row_bytes,
            |r, dst| {
                let src = (done + r) * shape.k;
                npum_format::bf16::f32_to_bf16_bytes_padded(
                    &a[src..src + shape.k],
                    dst,
                    pad_values,
                );
            },
        );
    }
    scratch.convert_seconds += t_convert.elapsed().as_secs_f64();
}

/// Narrow one chunk's widened C (the design's `padded_n` columns) into the
/// convolution's own `n` columns at the chunk's offset in `out`.
fn narrow(done: usize, chunk: usize, shape: &ConvShape, c: &[f32], out: &mut [f32]) {
    if shape.padded_n == shape.n {
        // NO TAIL: the widened rows are already contiguous at the
        // convolution's own width, so the chunk is one copy. The destination
        // is the CHUNK'S rows, not the start of the output: a second chunk
        // must not overwrite the first.
        out[done * shape.n..(done + chunk) * shape.n].copy_from_slice(&c[..chunk * shape.n]);
    } else {
        for r in 0..chunk {
            let src = r * shape.padded_n;
            let dst = (done + r) * shape.n;
            out[dst..dst + shape.n].copy_from_slice(&c[src..src + shape.n]);
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
    // THE PACKING BELOW WRITES BF16 BYTES. Fail closed on a datapath that
    // would give each element another width: the old code's per-row slice
    // assumed two bytes too, and a silent mismatch here would be a wrong
    // operand rather than a wrong shape.
    if dp.a_bytes != 2 {
        return Err(ArrayError::Design(format!(
            "the A operand is packed as bf16, and this datapath's A is {} bytes",
            dp.a_bytes
        )));
    }

    // THE PADDED BIAS, ONCE.
    //
    // The design's output has `padded_n` columns and the convolution has
    // `n` channels; the tail is zero, which is exact.
    // The padded bias buffer is taken from the reusable scratch instead of
    // allocated per convolution; it is handed back at the end of the run.
    let padded_bias: Option<Vec<f32>> = bias.map(|bias| {
        let mut padded = std::mem::take(&mut scratch.padded_bias);
        padded.resize(shape.padded_n, 0.0);
        padded.fill(0.0);
        padded[..shape.n].copy_from_slice(bias);
        padded
    });

    // THE SCRATCH, GROWN TO THE DESIGN'S PADDED SHAPE. A scratch made for a
    // narrower convolution, or reused across shapes, is grown here before any
    // packed row is written or any widened C is read back.
    scratch.ensure(shape.rows, shape.padded_k, shape.padded_n, dp.a_bytes);
    let row_bytes = shape.padded_k * dp.a_bytes;

    // THE CHUNKS, AND THE SLOTS THEY PING-PONG BETWEEN.
    //
    // The design computes `rows` rows per dispatch, and a convolution can
    // have more rows than that. With more than one chunk the A and C buffers
    // get a second slot each: while chunk N owns one pair, chunk N+1 is
    // prepared against the other. A single-chunk convolution has nothing to
    // overlap, so it pays for no extra slot.
    // The chunk size IS the design stream's M. Fewer, larger chunks would
    // mean streams compiled for a larger M (and A/C buffers that hold it), so
    // that lever belongs to the export/stream-selection side, not this loop.
    let chunks = m.div_ceil(shape.rows);
    if chunks > 1 {
        if design.slots_of(0) < 2 {
            design.add_slot_uninit(0)?;
        }
        if design.slots_of(2) < 2 {
            design.add_slot_uninit(2)?;
        }
    }
    // A previous convolution may have left these on the other slot of a
    // ping-pong pair; every pipeline starts from the design's own slot.
    design.bind(0, 0)?;
    design.bind(2, 0)?;

    // PROLOGUE: pack and submit chunk 0. It runs while chunk 1 is prepared.
    let first = m.min(shape.rows);
    pack_chunk(a, 0, first, shape, row_bytes, scratch);
    let run = submit_packed(design, instr, &scratch.a_packed[..first * row_bytes])?;
    let mut in_flight = Some(InFlight {
        run,
        c_slot: 0,
        chunk: first,
        done: 0,
    });
    let mut done = first;
    let mut dispatches = 1usize;

    while done < m {
        let chunk = (m - done).min(shape.rows);
        let slot = dispatches & 1;
        let prev = in_flight
            .take()
            .expect("a chunk is in flight until the last is read");

        // PACK chunk N+1 while chunk N is still computing on the device. The
        // host buffer is free: chunk N's A was already synced to its own
        // device slot by the submit above.
        pack_chunk(a, done, chunk, shape, row_bytes, scratch);

        // UPLOAD it into the OTHER A slot. Chunk N's dispatch still owns its
        // A slot, so writing the next one cannot disturb it.
        design.bind(0, slot)?;
        design.write_bound(0, &scratch.a_packed[..chunk * row_bytes])?;

        // WAIT for chunk N. Past this point its A slot is free to be reused
        // by the NEXT iteration, and its C slot is ready to read.
        ArrayDesign::<T>::wait_dispatch(prev.run)?;

        // SUBMIT chunk N+1 against the other C slot. The binding is captured
        // by start_dispatch, so rebinding C for the read-back below cannot
        // move this dispatch's output.
        design.bind(2, slot)?;
        let next_run = design.start_dispatch(instr)?;

        // READ and narrow chunk N from ITS C slot while chunk N+1 runs. The
        // widening and bias are the same as a plain dispatch's completion,
        // only the wait happened earlier so it could be moved out of the way.
        design.bind(2, prev.c_slot)?;
        if shape.padded_n == shape.n {
            // NO OUTPUT PADDING: the widened C is already at the convolution's
            // own width, so write it straight into its final rows. The plain
            // path would widen into `scratch.c` and then copy every value once
            // more in `narrow` -- a whole M x N pass for no reason. The bias is
            // the convolution's own here (the padded tail does not exist).
            let n = shape.n;
            read_dispatched(
                design,
                prev.chunk,
                dp,
                bias,
                n,
                &mut out[prev.done * n..(prev.done + prev.chunk) * n],
                &mut scratch.gemm.c_packed,
            )?;
        } else {
            read_dispatched(
                design,
                prev.chunk,
                dp,
                padded_bias.as_deref(),
                shape.padded_n,
                &mut scratch.c,
                &mut scratch.gemm.c_packed,
            )?;
            narrow(prev.done, prev.chunk, shape, &scratch.c, out);
        }

        in_flight = Some(InFlight {
            run: next_run,
            c_slot: slot,
            chunk,
            done,
        });
        done += chunk;
        dispatches += 1;
    }

    // EPILOGUE: the last dispatch has no successor to hide its read-back.
    let last = in_flight.expect("at least one chunk is dispatched");
    ArrayDesign::<T>::wait_dispatch(last.run)?;
    design.bind(2, last.c_slot)?;
    if shape.padded_n == shape.n {
        let n = shape.n;
        read_dispatched(
            design,
            last.chunk,
            dp,
            bias,
            n,
            &mut out[last.done * n..(last.done + last.chunk) * n],
            &mut scratch.gemm.c_packed,
        )?;
    } else {
        read_dispatched(
            design,
            last.chunk,
            dp,
            padded_bias.as_deref(),
            shape.padded_n,
            &mut scratch.c,
            &mut scratch.gemm.c_packed,
        )?;
        narrow(last.done, last.chunk, shape, &scratch.c, out);
    }

    // Leave both bindings on slot 0 for the next op.
    design.bind(0, 0)?;
    design.bind(2, 0)?;

    // Return the padded-bias buffer to the scratch for the next convolution.
    if let Some(padded) = padded_bias {
        scratch.padded_bias = padded;
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
    if k == padded_k {
        npum_format::bf16::f32_to_bf16_bytes(&a[..rows * k], &mut packed);
    } else {
        let pad_values = padded_k - k;
        for r in 0..rows {
            npum_format::bf16::f32_to_bf16_bytes_padded(
                &a[r * k..r * k + k],
                &mut packed[r * padded_k * 2..(r + 1) * padded_k * 2],
                pad_values,
            );
        }
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two widths, with and without padding, must give the same bytes as
    /// converting the real values and zeroing the tail separately -- the
    /// one-pass helper is only faster, never a different operand.
    #[test]
    fn packing_im2col_rows_matches_convert_then_zero() {
        let values: Vec<f32> = (0..24).map(|i| (i as f32 - 11.0) / 7.0).collect();
        for &(rows, k, padded_k) in &[(3usize, 8usize, 8usize), (3, 8, 16), (2, 7, 64)] {
            let got = pack_im2col_rows(&values, rows, k, padded_k);
            let mut want = vec![0u8; rows * padded_k * 2];
            for r in 0..rows {
                npum_format::bf16::f32_to_bf16_bytes(
                    &values[r * k..r * k + k],
                    &mut want[r * padded_k * 2..r * padded_k * 2 + k * 2],
                );
            }
            assert_eq!(got, want, "rows={rows} k={k} padded_k={padded_k}");
        }
    }

    /// A scratch sized to the convolution's own K grows to the design's
    /// padded widths and then keeps the allocation when reused by a smaller
    /// shape. Without this the packing/indexing below could overrun a
    /// scratch reused across convolutions of different shapes.
    #[test]
    fn scratch_ensure_covers_padded_widths_and_reuses_the_allocation() {
        let mut scratch = ConvScratch::new(2, 4);
        scratch.ensure(2, 16, 12, 2);
        assert!(scratch.a_packed.len() >= 2 * 16 * 2);
        assert!(scratch.c.len() >= 2 * 12);
        let a_cap = scratch.a_packed.capacity();
        let c_cap = scratch.c.capacity();

        scratch.ensure(2, 8, 6, 2);
        assert_eq!(scratch.a_packed.capacity(), a_cap);
        assert_eq!(scratch.c.capacity(), c_cap);
    }

    /// The narrowing of a chunk lands at the chunk's offset, not at the start
    /// of the output -- in BOTH branches of the width check. The no-tail
    /// branch used to write every chunk over the first one.
    #[test]
    fn narrow_writes_each_chunk_at_its_offset() {
        let plain = ConvShape {
            rows: 2,
            k: 3,
            n: 2,
            padded_k: 3,
            padded_n: 2,
        };
        let c: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let mut out = vec![0f32; 8];
        narrow(2, 2, &plain, &c, &mut out);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);

        let padded = ConvShape {
            rows: 2,
            k: 3,
            n: 2,
            padded_k: 4,
            padded_n: 4,
        };
        let c: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut out = vec![0f32; 8];
        narrow(2, 2, &padded, &c, &mut out);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 5.0]);
    }
}

/// The pipeline, end to end, on a transport that records every operation and
/// simulates the GEMM as "copy the first bf16 of A into every cell of C".
/// That marker makes each chunk's output identify the A slot its dispatch
/// actually read, so a slot mix-up is a wrong number rather than a race.
#[cfg(test)]
mod pipeline_tests {
    use super::*;
    use crate::array::ArrayDesign;
    use crate::transport::{BufferFlags, Transport, TransportError};
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    type Log = Arc<Mutex<Vec<String>>>;

    #[derive(Clone)]
    struct FakeBuffer {
        bytes: Arc<Mutex<Vec<u8>>>,
        log: Log,
        id: String,
    }

    struct FakeRun {
        a: FakeBuffer,
        c: FakeBuffer,
        log: Log,
    }

    unsafe extern "C" fn free_fake_run(ptr: *mut c_void) {
        if !ptr.is_null() {
            unsafe { drop(Box::from_raw(ptr as *mut FakeRun)) };
        }
    }

    struct FakeTransport {
        log: Log,
        /// How many buffers of each group have been handed out -- the slot
        /// number, so the log can name the exact slot an operation touched.
        counts: Mutex<HashMap<i32, usize>>,
    }

    impl FakeTransport {
        fn new(log: Log) -> Self {
            Self {
                log,
                counts: Mutex::new(HashMap::new()),
            }
        }
    }

    impl Transport for FakeTransport {
        type Buffer = FakeBuffer;

        fn open(_index: u32) -> Result<Self, TransportError> {
            Ok(Self::new(Arc::new(Mutex::new(Vec::new()))))
        }

        fn open_context(&mut self, _xclbin: &[u8]) -> Result<(), TransportError> {
            Ok(())
        }

        fn open_kernel(&mut self, _name: &str) -> Result<(), TransportError> {
            Ok(())
        }

        fn group_id(&self, argno: i32) -> Result<i32, TransportError> {
            Ok(argno)
        }

        fn alloc(
            &mut self,
            bytes: u64,
            _flags: BufferFlags,
            group: i32,
        ) -> Result<Self::Buffer, TransportError> {
            let mut counts = self.counts.lock().unwrap();
            let slot = counts.entry(group).or_insert(0);
            let id = format!("{group}:{slot}");
            *slot += 1;
            Ok(FakeBuffer {
                bytes: Arc::new(Mutex::new(vec![0u8; bytes as usize])),
                log: self.log.clone(),
                id,
            })
        }

        fn write(bo: &Self::Buffer, data: &[u8], offset: u64) -> Result<(), TransportError> {
            bo.log.lock().unwrap().push(format!("write({})", bo.id));
            let mut bytes = bo.bytes.lock().unwrap();
            let at = offset as usize;
            bytes[at..at + data.len()].copy_from_slice(data);
            Ok(())
        }

        fn size(bo: &Self::Buffer) -> u64 {
            bo.bytes.lock().unwrap().len() as u64
        }

        fn read(bo: &Self::Buffer, data: &mut [u8], offset: u64) -> Result<(), TransportError> {
            bo.log.lock().unwrap().push(format!("read({})", bo.id));
            let bytes = bo.bytes.lock().unwrap();
            let at = offset as usize;
            data.copy_from_slice(&bytes[at..at + data.len()]);
            Ok(())
        }

        fn sync_to_device(
            bo: &Self::Buffer,
            _bytes: u64,
            _offset: u64,
        ) -> Result<(), TransportError> {
            bo.log
                .lock()
                .unwrap()
                .push(format!("sync_to_device({})", bo.id));
            Ok(())
        }

        fn sync_from_device(
            bo: &Self::Buffer,
            _bytes: u64,
            _offset: u64,
        ) -> Result<(), TransportError> {
            bo.log
                .lock()
                .unwrap()
                .push(format!("sync_from_device({})", bo.id));
            Ok(())
        }

        fn start(
            &mut self,
            _instr: &Self::Buffer,
            _words: u32,
            args: &[&Self::Buffer],
        ) -> Result<Run, TransportError> {
            self.log.lock().unwrap().push("start".to_string());
            let run = Box::new(FakeRun {
                a: args[0].clone(),
                c: args[2].clone(),
                log: self.log.clone(),
            });
            Ok(Run::new(Box::into_raw(run) as *mut c_void, free_fake_run))
        }

        fn wait(run: &Run) -> Result<(), TransportError> {
            let r = unsafe { &*(run.handle() as *const FakeRun) };
            r.log.lock().unwrap().push("wait".to_string());
            // The simulated GEMM: every C element becomes this dispatch's A
            // marker. The A slot is captured at start, so a later rebind of
            // buffer 0 cannot change what this run reads.
            let a = r.a.bytes.lock().unwrap();
            let marker = [a[0], a[1]];
            let mut c = r.c.bytes.lock().unwrap();
            for cell in c.chunks_exact_mut(2) {
                cell.copy_from_slice(&marker);
            }
            Ok(())
        }
    }

    fn write_design(dir: &Path, a_cap: u64, c_cap: u64, rows: usize, k: usize, n: usize) {
        let manifest = format!(
            r#"{{
                "name": "fake",
                "kernel": "MLIR_AIE",
                "arch": 1,
                "device": "npu1",
                "buffers": [{a_cap}, 8, {c_cap}],
                "a_dtype": "bf16",
                "c_dtype": "bf16",
                "tile": {{ "m": {rows}, "k": {k}, "n": {n}, "rows": 1 }},
                "slot0": {{ "op": "", "batch": 0, "slot": 0, "file": "insts.bin",
                            "M": {rows}, "K": {k}, "N": {n} }}
            }}"#
        );
        std::fs::write(dir.join("design.json"), manifest).unwrap();
        std::fs::write(dir.join("final.xclbin"), b"xclbin").unwrap();
        std::fs::write(dir.join("insts.bin"), vec![0u8; 4]).unwrap();
    }

    /// Three chunks over a two-row design: chunks of 2, 2 and 1 rows.
    ///
    /// Checks both the numbers (each chunk's C came from its own A slot) and
    /// the ORDER of operations: the next chunk's A is uploaded BEFORE the wait
    /// for the current one, and the next dispatch is submitted BEFORE the
    /// current one's C is read back. That order IS the overlap.
    #[test]
    fn pipeline_ping_pongs_slots_and_overlaps_stages() {
        let (rows, k, padded_k, n, padded_n, m) = (2usize, 4usize, 8usize, 3usize, 4usize, 5usize);
        let dir = std::env::temp_dir().join(format!(
            "npum_conv_pipeline_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        write_design(
            &dir,
            (rows * padded_k * 2) as u64,
            (rows * padded_n * 2) as u64,
            rows,
            padded_k,
            padded_n,
        );

        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let mut design = ArrayDesign::open(&dir, FakeTransport::new(log.clone())).unwrap();

        // The first element of each row is a distinct, bf16-exact marker.
        let mut a = vec![0f32; m * k];
        for r in 0..m {
            a[r * k] = 10.0 + r as f32;
        }
        let bias = [1.0f32, 2.0, 3.0];
        let mut out = vec![0f32; m * n];
        let mut scratch = ConvScratch::default();
        let shape = ConvShape {
            rows,
            k,
            n,
            padded_k,
            padded_n,
        };

        let dispatches = conv(
            &mut design,
            0,
            &shape,
            Some(&bias),
            &a,
            &mut out,
            &mut scratch,
        )
        .expect("the pipelined convolution runs");
        assert_eq!(dispatches, 3);

        for r in 0..m {
            // The simulated GEMM writes one marker per DISPATCH -- the first
            // A value of the chunk -- into every row of that chunk's C.
            let marker = 10.0 + (r / rows * rows) as f32;
            for j in 0..n {
                assert_eq!(
                    out[r * n + j],
                    marker + bias[j],
                    "row {r} column {j} came from the wrong A slot"
                );
            }
        }

        let log = log.lock().unwrap();
        let first_start = log.iter().position(|e| e == "start").unwrap();
        assert_eq!(
            &log[first_start..],
            &[
                "start", // chunk 0 submitted
                "write(3:1)",
                "sync_to_device(3:1)", // chunk 1's A, while 0 runs
                "wait",                // chunk 0 done
                "start",               // chunk 1 submitted
                "sync_from_device(5:0)",
                "read(5:0)", // chunk 0's C, while 1 runs
                "write(3:0)",
                "sync_to_device(3:0)", // chunk 2's A, slot 0 again
                "wait",                // chunk 1 done
                "start",               // chunk 2 submitted
                "sync_from_device(5:1)",
                "read(5:1)", // chunk 1's C, while 2 runs
                "wait",      // chunk 2 done
                "sync_from_device(5:0)",
                "read(5:0)", // chunk 2's C
            ]
        );

        drop(log);
        drop(design);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// When the design's N is the convolution's own N, the read-back writes
    /// straight into `out` and `narrow` is skipped. The numbers must be the
    /// same as through the padded path -- including the chunk offsets and the
    /// bias -- because the two paths differ only by a copy that was removed.
    #[test]
    fn no_output_padding_writes_straight_into_out() {
        let (rows, k, padded_k, n, m) = (2usize, 4usize, 8usize, 3usize, 5usize);
        let dir = std::env::temp_dir().join(format!(
            "npum_conv_nopad_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        write_design(
            &dir,
            (rows * padded_k * 2) as u64,
            (rows * n * 2) as u64,
            rows,
            padded_k,
            n,
        );

        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let mut design = ArrayDesign::open(&dir, FakeTransport::new(log.clone())).unwrap();

        let mut a = vec![0f32; m * k];
        for r in 0..m {
            a[r * k] = 10.0 + r as f32;
        }
        let bias = [1.0f32, 2.0, 3.0];
        let mut out = vec![0f32; m * n];
        let mut scratch = ConvScratch::default();
        let shape = ConvShape {
            rows,
            k,
            n,
            padded_k,
            padded_n: n,
        };

        let dispatches = conv(
            &mut design,
            0,
            &shape,
            Some(&bias),
            &a,
            &mut out,
            &mut scratch,
        )
        .expect("the no-padding convolution runs");
        assert_eq!(dispatches, 3);
        for r in 0..m {
            let marker = 10.0 + (r / rows * rows) as f32;
            for j in 0..n {
                assert_eq!(
                    out[r * n + j],
                    marker + bias[j],
                    "row {r} column {j} came from the wrong A slot or lost its bias"
                );
            }
        }

        drop(log);
        drop(design);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
