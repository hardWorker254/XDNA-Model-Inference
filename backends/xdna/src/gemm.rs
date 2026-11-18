//! The array's GEMM: the op a model actually wants on the board.
//!
//! # WHAT ONE DISPATCH IS
//!
//! `out[0..n_real, n] = a[0..n_real, k] @ b[k, n] + bias[n]`, where `b` is
//! the weight panel the caller has staged and bound, and the design computes
//! `rows` rows whatever the caller has.
//!
//! The reference runtime's `NpuGemm::run` is the same thing with the same names,
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
//!
//! # THE SCRATCH IS THE CALLER'S, AND IT IS REUSED
//!
//! A transformer dispatches many projections per layer and per step, and a
//! scratch that was allocated per dispatch would spend its time in the
//! allocator rather than on the array. So the caller owns the scratch, it
//! grows when a larger tier or width appears, and the same buffers are
//! reused across dispatches.

use crate::array::{ArrayDesign, ArrayError};
use crate::transport::Transport;

/// The buffers one GEMM dispatch needs.
///
/// They are the caller's, and they are REUSED across dispatches: the sizes
/// grow to the largest shape seen and stay there.
#[derive(Debug, Default, Clone)]
pub struct GemmScratch {
    /// The A operand, in the design's dtype (`rows * k * a_bytes`).
    pub a_packed: Vec<u8>,
    /// The C operand read back from the device (`n_real * n * c_bytes`).
    pub c_packed: Vec<u8>,
    /// Reusable bias buffers for async submissions.
    ///
    /// A submitted dispatch owns one of these until it is completed, then the
    /// buffer goes back on the free list. A layer dispatches the same handful
    /// of biases thousands of times; with the pool the copy into the ticket
    /// stops allocating after the first few submissions.
    bias_slots: Vec<Vec<f32>>,
    /// Bias slot indices freed by a completed dispatch, ready for reuse.
    bias_free: Vec<usize>,
}

impl GemmScratch {
    /// An empty scratch; it grows on the first dispatch.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many bias buffers the async pool has allocated. For the test that
    /// checks repeated submissions reuse one buffer instead of growing it.
    #[cfg(test)]
    pub(crate) fn bias_pool_len(&self) -> usize {
        self.bias_slots.len()
    }
}

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
///
/// This allocates a one-shot scratch. Callers that dispatch many times
/// should prefer [`gemm_with_scratch`] and reuse a [`GemmScratch`].
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
    let mut scratch = GemmScratch::new();
    gemm_with_scratch(
        design,
        instr,
        a,
        n_real,
        rows,
        k,
        bias,
        n,
        out,
        &mut scratch,
    )
}

/// Run one GEMM on the array, reusing the caller's scratch buffers.
///
/// See [`gemm`] for the shape and buffer requirements. The scratch grows
/// when a larger tier or width appears and is then reused across dispatches.
#[allow(clippy::too_many_arguments)]
pub fn gemm_with_scratch<T: Transport>(
    design: &mut ArrayDesign<T>,
    instr: usize,
    a: &[f32],
    n_real: usize,
    rows: usize,
    k: usize,
    bias: Option<&[f32]>,
    n: usize,
    out: &mut [f32],
    scratch: &mut GemmScratch,
) -> Result<(), ArrayError> {
    // THE SHAPE, BEFORE THE HARDWARE.
    //
    // These checks are the same on the host path and on the array path, and
    // they come first: a shape bug is a shape bug whichever path the run
    // takes.
    check_shape(a.len(), n_real, rows, k, n, out.len(), bias)?;

    let dp = Datapath::of(design)?;

    // THE FIT, AGAINST THE DESIGN'S OWN BUFFERS.
    check_fit(design, rows, k, n, dp)?;

    // THE A OPERAND, IN THE DESIGN'S OWN DTYPE.
    //
    // The real rows first, then the padded tail as zeros. The scratch holds
    // the buffer across dispatches: it grows when it has to, and the tail
    // past the real rows is zeroed because it is a padded ROW and a stale
    // value there would be another call's activations.
    pack_a_into(a, n_real, rows, k, dp.a_bytes, &mut scratch.a_packed);
    let a_need = rows * k * dp.a_bytes;

    dispatch_with_scratch(
        design,
        instr,
        &scratch.a_packed[..a_need],
        n_real,
        dp,
        bias,
        n,
        out,
        &mut scratch.c_packed,
    )
}

/// The A operand in the design's own dtype.
///
/// The real rows first, then the padded tail as zeros. The tail is a padded
/// ROW: the design computes all `rows` rows whatever the A buffer holds in
/// the rest, and zero is the only value that cannot be mistaken for a real
/// position.
///
/// This allocates a buffer. Callers that dispatch many times should prefer
/// [`pack_a_into`] and reuse a buffer.
#[allow(dead_code)]
pub(crate) fn pack_a(a: &[f32], n_real: usize, rows: usize, k: usize, a_bytes: usize) -> Vec<u8> {
    let mut packed = vec![0u8; rows * k * a_bytes];
    pack_a_into(a, n_real, rows, k, a_bytes, &mut packed);
    packed
}

/// Pack the A operand into `packed`, reusing its allocation.
///
/// The real rows are converted first and the padded tail is zeroed, so a
/// stale row from a previous dispatch cannot be mistaken for a real
/// position. The buffer grows when it has to and is not shrunk.
pub(crate) fn pack_a_into(
    a: &[f32],
    n_real: usize,
    rows: usize,
    k: usize,
    a_bytes: usize,
    packed: &mut Vec<u8>,
) {
    let real_bytes = n_real * k * a_bytes;
    let total = rows * k * a_bytes;
    if packed.len() < total {
        packed.resize(total, 0);
    }
    // The real rows, in the design's dtype.
    npum_format::bf16::f32_to_bf16_bytes(&a[..n_real * k], &mut packed[..real_bytes]);
    // The padded tail is a padded ROW: zero it so a stale row from a
    // previous dispatch cannot be mistaken for a real position.
    if total > real_bytes {
        packed[real_bytes..total].fill(0);
    }
}

/// The dispatch itself: the A operand is already in the design's dtype.
///
/// This is the shape both the projection and the convolution dispatch, and
/// it is one function so that the two cannot disagree about what a dispatch
/// is -- the convolution's padded rows and the projection's padded tail are
/// the same mechanism, and a second copy of it would be a second thing to be
/// wrong about.
///
/// This allocates per-dispatch buffers. Callers that dispatch many times
/// should prefer [`dispatch_with_scratch`] and reuse buffers.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn dispatch<T: Transport>(
    design: &mut ArrayDesign<T>,
    instr: usize,
    a_packed: &[u8],
    n_real: usize,
    dp: Datapath,
    bias: Option<&[f32]>,
    n: usize,
    out: &mut [f32],
) -> Result<(), ArrayError> {
    let mut c_packed = Vec::new();
    dispatch_with_scratch(
        design,
        instr,
        a_packed,
        n_real,
        dp,
        bias,
        n,
        out,
        &mut c_packed,
    )
}

/// The dispatch itself, reusing the caller's read-back buffer.
///
/// `a_packed` must already hold the A operand for this dispatch; `c_packed` is
/// grown and reused for the read-back. Widening and the bias add are fused into
/// one pass straight into `out`, so no per-row scratch is needed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_with_scratch<T: Transport>(
    design: &mut ArrayDesign<T>,
    instr: usize,
    a_packed: &[u8],
    n_real: usize,
    dp: Datapath,
    bias: Option<&[f32]>,
    n: usize,
    out: &mut [f32],
    c_packed: &mut Vec<u8>,
) -> Result<(), ArrayError> {
    let run = submit_packed(design, instr, a_packed)?;
    complete_dispatched(design, run, n_real, dp, bias, n, out, c_packed)
}

/// Write the A operand, synchronise it to the device, and start a dispatch.
///
/// The returned [`crate::transport::Run`] must be completed with
/// [`complete_dispatched`] (or waited on and read back manually) before the C
/// buffer or the A slot is reused for another dispatch.
pub(crate) fn submit_packed<T: Transport>(
    design: &mut ArrayDesign<T>,
    instr: usize,
    a_packed: &[u8],
) -> Result<crate::transport::Run, ArrayError> {
    // THE DISPATCH WINDOW.
    //
    // Bind, sync, dispatch -- the wait and read-back are the caller's to
    // schedule, so host-side work can run while the device computes.
    design.write_bound(0, a_packed)?;
    design.start_dispatch(instr)
}

/// Wait for a dispatch, read back its output, widen it to f32 and add the bias.
///
/// This is the completion half of [`submit_packed`]. It reads from buffer 2 in
/// its currently bound slot. Only the real rows are synced and read back: the
/// output rows lie contiguously from row zero, so the design's padded tail is
/// never touched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn complete_dispatched<T: Transport>(
    design: &mut ArrayDesign<T>,
    run: crate::transport::Run,
    n_real: usize,
    dp: Datapath,
    bias: Option<&[f32]>,
    n: usize,
    out: &mut [f32],
    c_packed: &mut Vec<u8>,
) -> Result<(), ArrayError> {
    ArrayDesign::<T>::wait_dispatch(run)?;
    read_dispatched(design, n_real, dp, bias, n, out, c_packed)
}

/// Read a dispatch that has ALREADY been waited on, widen it, add the bias.
///
/// This is the read-back half of [`complete_dispatched`], split out for callers
/// that pipeline: they wait for one dispatch early so the next can be
/// submitted, then read the finished one's output while the next is in flight.
/// `design` is only read here -- the caller owns the slot binding and must have
/// bound buffer 2 to the slot the finished dispatch wrote into. Only the real
/// rows are synced and read back: the output rows lie contiguously from row
/// zero, so the design's padded tail is never touched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn read_dispatched<T: Transport>(
    design: &ArrayDesign<T>,
    n_real: usize,
    dp: Datapath,
    bias: Option<&[f32]>,
    n: usize,
    out: &mut [f32],
    c_packed: &mut Vec<u8>,
) -> Result<(), ArrayError> {
    // THE OUTPUT, BACK IN f32, WITH THE BIAS.
    //
    // ONE ROW AT A TIME, and the row is widened straight into its final place:
    // the widening is the other half of the boundary the A operand crossed,
    // and at the array's volume it is hundreds of millions of values. The
    // read-back buffer is the caller's scratch and grows when the tier or
    // width grows. Reading only `n_real` rows keeps both the sync and the
    // copy off the padded tail.
    let c_need = n_real * n * dp.c_bytes;
    if c_packed.len() < c_need {
        c_packed.resize(c_need, 0);
    }
    design.read(2, &mut c_packed[..c_need])?;

    match bias {
        None => {
            // NO BIAS: widen straight into the output's final place. The old
            // path decoded into the scratch row and then copied it into `out`
            // -- two passes over the row when one is enough. The widening
            // helper already writes f32 values in order, so handing it the
            // destination slice removes the intermediate copy entirely. The
            // `row` scratch is left untouched (and need not even be sized).
            for r in 0..n_real {
                let src = r * n * dp.c_bytes;
                npum_format::bf16::bf16_to_f32_bytes(
                    &c_packed[src..src + n * dp.c_bytes],
                    &mut out[r * n..r * n + n],
                );
            }
        }
        Some(bias) => {
            // BIAS: widen straight into the output with the bias added, in one
            // pass. The old path decoded into a scratch row and then read that
            // row again for the add -- two visits to every output value. The
            // fused helper writes the final f32 value in one visit, so the
            // scratch row is gone from this path entirely.
            for r in 0..n_real {
                let src = r * n * dp.c_bytes;
                npum_format::bf16::bf16_to_f32_add_bias_bytes(
                    &c_packed[src..src + n * dp.c_bytes],
                    &mut out[r * n..r * n + n],
                    bias,
                );
            }
        }
    }
    Ok(())
}

/// Check that a shape fits the design's own A and C buffers.
///
/// The design was exported for a shape; a shape that does not fit is not a
/// shape this runtime can serve, and saying so is better than a dispatch
/// that overruns a buffer. Both the synchronous path and the async one ask
/// this same question, so they cannot disagree about what fits.
pub fn check_fit<T: Transport>(
    design: &ArrayDesign<T>,
    rows: usize,
    k: usize,
    n: usize,
    dp: Datapath,
) -> Result<(), ArrayError> {
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
    Ok(())
}

/// One GEMM submitted to the array, waiting to be read back.
///
/// This is the whole point of splitting a dispatch in two: the device works
/// while the host does something else, and the bindings a dispatch captured
/// at `start_dispatch` belong to it alone. So the caller may rebind A, B and C
/// for another dispatch between [`submit_gemm`] and [`complete_gemm`] -- only
/// the C slot this run actually wrote into has to be remembered, and it is
/// kept here.
///
/// The bias is COPIED at submit into a slot of the submit scratch: the ticket
/// owns its data, so the caller is free to reuse the bias slice while the
/// device computes. The slot is returned to the scratch on completion, so the
/// copy allocates only while the pool is filling, not per dispatch. The same
/// scratch that submitted the run must complete it.
pub struct GemmDispatch {
    pub(crate) run: crate::transport::Run,
    pub(crate) n_real: usize,
    pub(crate) dp: Datapath,
    pub(crate) n: usize,
    /// The scratch's bias slot holding this dispatch's bias, when it has one.
    pub(crate) bias_slot: Option<usize>,
    /// The C slot bound when the dispatch was started. Rebound away for the
    /// next dispatch, and bound back here before the output is read.
    pub(crate) c_slot: usize,
}

/// Pack `a` into `scratch`, write it to the bound A slot, and START a GEMM
/// without waiting for it.
///
/// The caller owns the buffer bindings: before this call the weight buffer
/// must be bound to the panel, and A and C to a slot pair that no in-flight
/// dispatch still owns. The bindings are captured by `start_dispatch`, so
/// rebinding them after this call cannot move this run's operands.
///
/// Complete with [`complete_gemm`], which waits for the run, reads C back from
/// the slot recorded at submit, widens it to f32 and adds the bias.
#[allow(clippy::too_many_arguments)]
pub fn submit_gemm<T: Transport>(
    design: &mut ArrayDesign<T>,
    instr: usize,
    a: &[f32],
    n_real: usize,
    rows: usize,
    k: usize,
    dp: Datapath,
    bias: Option<&[f32]>,
    n: usize,
    scratch: &mut GemmScratch,
) -> Result<GemmDispatch, ArrayError> {
    check_shape(a.len(), n_real, rows, k, n, n_real * n, bias)?;
    check_fit(design, rows, k, n, dp)?;
    if dp.a_bytes != 2 {
        return Err(ArrayError::Design(format!(
            "the A operand is packed as bf16, and this datapath's A is {} bytes",
            dp.a_bytes
        )));
    }

    // The host half: pack A while no other dispatch needs this scratch's A
    // buffer. The device copy is written to its own slot by the submit below,
    // so the host buffer is free for the next dispatch as soon as it returns.
    pack_a_into(a, n_real, rows, k, dp.a_bytes, &mut scratch.a_packed);
    let a_need = rows * k * dp.a_bytes;
    let run = submit_packed(design, instr, &scratch.a_packed[..a_need])?;
    // Remember where THIS run writes. The binding may be moved for the next
    // dispatch before this one's C is read.
    let c_slot = design.bound(2)?;
    // Copy the bias into a reusable slot rather than a fresh `Vec` per
    // dispatch: the ticket must own its bias (the caller's slice may change
    // while the device works), but the allocation can outlive many runs.
    let bias_slot = bias.map(|b| {
        let index = match scratch.bias_free.pop() {
            Some(index) => index,
            None => {
                let index = scratch.bias_slots.len();
                scratch.bias_slots.push(Vec::new());
                index
            }
        };
        let slot = &mut scratch.bias_slots[index];
        slot.clear();
        slot.extend_from_slice(b);
        index
    });
    Ok(GemmDispatch {
        run,
        n_real,
        dp,
        n,
        bias_slot,
        c_slot,
    })
}

/// Wait for a submitted GEMM and read its output into `out`.
///
/// This is the completion half of [`submit_gemm`]. It waits for the run, binds
/// C back to the slot the run actually wrote into, and widens the bf16 output
/// to f32 with the bias added -- exactly what [`gemm_with_scratch`] does, only
/// scheduled so the host can work between the two halves.
pub fn complete_gemm<T: Transport>(
    design: &mut ArrayDesign<T>,
    dispatch: GemmDispatch,
    out: &mut [f32],
    scratch: &mut GemmScratch,
) -> Result<(), ArrayError> {
    if out.len() < dispatch.n_real * dispatch.n {
        return Err(ArrayError::Design(format!(
            "the output holds {} floats, and {} rows of {} need {}",
            out.len(),
            dispatch.n_real,
            dispatch.n,
            dispatch.n_real * dispatch.n
        )));
    }
    ArrayDesign::<T>::wait_dispatch(dispatch.run)?;
    design.bind(2, dispatch.c_slot)?;
    // The bias borrows one field of the scratch and the read-back buffer is
    // another, so the two can be live at once. The borrow ends with the call;
    // the slot is then free for the next submission.
    let bias = dispatch
        .bias_slot
        .map(|index| scratch.bias_slots[index].as_slice());
    let result = read_dispatched(
        design,
        dispatch.n_real,
        dispatch.dp,
        bias,
        dispatch.n,
        out,
        &mut scratch.c_packed,
    );
    if let Some(index) = dispatch.bias_slot {
        scratch.bias_free.push(index);
    }
    result
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
