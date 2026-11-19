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
//! The array's MMAC multiplies bf16 and int8. The bf16 path needs only the
//! panel; the int8 path needs the container's per-output-channel weight scale
//! and SmoothQuant factors, and the container is the only source of both.
//! [`Datapath::of`] names the pair it will dispatch (bf16/bf16 or i8/i32)
//! and refuses every other one -- a missing dequantisation step would give a
//! plausible number with the wrong scales, which is worse than an error.
//!
//! # THE SCRATCH IS THE CALLER'S, AND IT IS REUSED
//!
//! A transformer dispatches many projections per layer and per step, and a
//! scratch that was allocated per dispatch would spend its time in the
//! allocator rather than on the array. So the caller owns the scratch, it
//! grows when a larger tier or width appears, and the same buffers are
//! reused across dispatches.

use std::sync::Arc;

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
    /// Per-row activation scales for submitted int8 dispatches.
    ///
    /// An int8 A is quantised per row, and the row's scale is needed again when
    /// the int32 C is dequantised. Like the bias slots, these are pooled: the
    /// ticket owns one until completion, then returns it for the next run.
    a_scale_slots: Vec<Vec<f32>>,
    /// Row-scale slot indices freed by a completed int8 dispatch.
    a_scale_free: Vec<usize>,
    /// Row scales for the synchronous int8 path, reused across dispatches.
    pub(crate) a_scales: Vec<f32>,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatapathKind {
    /// The A and C operands are bf16 (the original GEMM path).
    Bf16,
    /// A is int8 and C is the raw int32 accumulator.
    ///
    /// The int32 output is what makes the scheme exact: the rank-1 dequant
    /// `(sa[row] * wscale[col])` and the bias add happen on the host, once,
    /// after the read-back -- not on a bf16-rounded accumulator.
    Int8I32,
}

#[derive(Debug, Clone, Copy)]
pub struct Datapath {
    /// The A operand's element size, in bytes.
    pub a_bytes: usize,
    /// The C operand's element size, in bytes.
    pub c_bytes: usize,
    /// Which datapath the design implements.
    pub kind: DatapathKind,
}

impl Datapath {
    /// The design's datapath, checked.
    ///
    /// Two datapaths are dispatched:
    ///
    /// * bf16 A and bf16 C -- the original path, which needs no container
    ///   metadata beyond the panel itself;
    /// * int8 A and int32 C -- the scheme whose per-output-channel weight
    ///   scale and per-input-channel SmoothQuant factors come from the
    ///   container. They are carried in [`QuantScales`], which the projection
    ///   backend reads from the container and hands to every dispatch.
    ///
    /// Anything else is refused here, by name, rather than computed with a
    /// dequantisation step missing -- a plausible number with the wrong scales
    /// is worse than a refusal that names the dtype pair.
    pub fn of<T: Transport>(design: &ArrayDesign<T>) -> Result<Self, ArrayError> {
        let info = design.info();
        let a_bytes = elem_bytes(&info.a_dtype).ok_or_else(|| {
            ArrayError::Design(format!("the design's A dtype is {:?}", info.a_dtype))
        })?;
        let c_bytes = elem_bytes(&info.c_dtype).ok_or_else(|| {
            ArrayError::Design(format!("the design's C dtype is {:?}", info.c_dtype))
        })?;
        let kind = match (
            info.a_dtype.as_str(),
            info.c_dtype.as_str(),
            a_bytes,
            c_bytes,
        ) {
            ("bf16", "bf16", 2, 2) => DatapathKind::Bf16,
            ("i8" | "int8", "i32", 1, 4) => DatapathKind::Int8I32,
            _ => {
                return Err(ArrayError::Design(format!(
                    "this runtime dispatches the bf16 (bf16/bf16) and int8 (i8/i32) \
                     datapaths, and the design's A and C are '{}' and '{}' ({} and {} bytes). \
                     An int8 design with a bf16 C is refused: its accumulator is rounded \
                     before the scale multiply, which is a different network.",
                    info.a_dtype, info.c_dtype, a_bytes, c_bytes
                )));
            }
        };
        Ok(Self {
            a_bytes,
            c_bytes,
            kind,
        })
    }
}

/// The element size of an operand dtype, as the manifest spells it.
fn elem_bytes(dtype: &str) -> Option<usize> {
    match dtype {
        "bf16" => Some(2),
        "i8" | "int8" => Some(1),
        "i32" => Some(4),
        "f32" => Some(4),
        _ => None,
    }
}

/// The container's scales for one int8 projection.
///
/// The pair is inseparable: `wscale[j]` is the per-output-channel weight scale
/// the packer computed from `W * asmooth`, and `recip[j] = 1 / asmooth[j]` is
/// what the activation is multiplied by before it is quantised. Applying one
/// without the other computes `X @ W * asmooth²`, a plausible and wrong answer.
#[derive(Debug, Clone)]
pub struct QuantScales {
    /// `1 / asmooth[j]`, one per input channel (the K of the projection).
    pub recip: Vec<f32>,
    /// Per-output-channel weight scale, one per output column (the N).
    pub wscale: Vec<f32>,
}

impl QuantScales {
    /// Build the scales from the container's `.asmooth` and `.wscale`.
    ///
    /// A non-positive or non-finite factor would make the reciprocal an inf or
    /// a NaN -- a number that would quietly poison every output -- so it is a
    /// refusal here rather than a NaN in the result.
    pub fn new(asmooth: &[f32], wscale: Vec<f32>) -> Result<Self, ArrayError> {
        if asmooth.is_empty() {
            return Err(ArrayError::Design(
                "an int8 projection has an empty SmoothQuant vector".to_string(),
            ));
        }
        if wscale.iter().any(|s| !s.is_finite() || *s <= 0.0) {
            return Err(ArrayError::Design(
                "an int8 projection has a non-positive or non-finite per-channel weight scale"
                    .to_string(),
            ));
        }
        let mut recip = Vec::with_capacity(asmooth.len());
        for s in asmooth {
            if !s.is_finite() || *s <= 0.0 {
                return Err(ArrayError::Design(
                    "an int8 projection has a non-positive or non-finite SmoothQuant factor"
                        .to_string(),
                ));
            }
            recip.push(1.0 / *s);
        }
        Ok(Self { recip, wscale })
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
    if dp.kind != DatapathKind::Bf16 {
        return Err(ArrayError::Design(
            "a bf16 GEMM was asked of an int8 design -- use gemm_quant_with_scratch with the \
             container's per-channel scales instead"
                .to_string(),
        ));
    }

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

/// Run one int8 GEMM, reusing the caller's scratch buffers.
///
/// The A operand is quantised per row against the container's SmoothQuant
/// reciprocals, and the int32 C is rescaled by the row scale and the container's
/// per-channel weight scale before the bias is added. The shape and fit checks
/// are the same as the bf16 path's -- only the buffers and the widening differ.
#[allow(clippy::too_many_arguments)]
pub fn gemm_quant_with_scratch<T: Transport>(
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
    quant: &QuantScales,
) -> Result<(), ArrayError> {
    check_shape(a.len(), n_real, rows, k, n, out.len(), bias)?;
    let dp = Datapath::of(design)?;
    if dp.kind != DatapathKind::Int8I32 {
        return Err(ArrayError::Design(
            "an int8 GEMM was asked of a design that does not dispatch the i8/i32 datapath"
                .to_string(),
        ));
    }
    check_fit(design, rows, k, n, dp)?;
    pack_a_quant_into(
        a,
        n_real,
        rows,
        k,
        quant,
        &mut scratch.a_packed,
        &mut scratch.a_scales,
    )?;
    let a_need = rows * k * dp.a_bytes;
    let run = submit_packed(design, instr, &scratch.a_packed[..a_need])?;
    ArrayDesign::<T>::wait_dispatch(run)?;
    let c_need = n_real * n * dp.c_bytes;
    if scratch.c_packed.len() < c_need {
        scratch.c_packed.resize(c_need, 0);
    }
    design.read(2, &mut scratch.c_packed[..c_need])?;
    dequant_quant_rows(
        &scratch.c_packed[..c_need],
        out,
        n_real,
        n,
        bias,
        quant,
        &scratch.a_scales,
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
    // The real rows, in the design's dtype. Row-local conversion, so the
    // pool splits it across cores for large operands; the bytes per row are
    // identical to one whole-buffer conversion. The bf16 helper always emits
    // two bytes per value, so the row-split path is only valid for a 2-byte A.
    if a_bytes == 2 {
        crate::parallel::for_each_row(&mut packed[..real_bytes], k * 2, |r, dst| {
            let src = r * k;
            npum_format::bf16::f32_to_bf16_bytes(&a[src..src + k], dst);
        });
    } else {
        npum_format::bf16::f32_to_bf16_bytes(&a[..n_real * k], &mut packed[..real_bytes]);
    }
    // The padded tail is a padded ROW: zero it so a stale row from a
    // previous dispatch cannot be mistaken for a real position.
    if total > real_bytes {
        packed[real_bytes..total].fill(0);
    }
}

/// `np.rint`: round to nearest, ties to even.
///
/// `f32::round` breaks a tie away from zero, which is a different integer from
/// `rint` on every exact `.5`. The packer quantised the weights with `rint`, so
/// the activations must use the same rule: one mismatched boundary value is a
/// one-LSB error that survives the exact int32 accumulator all the way to the
/// output.
#[inline]
fn rint_even(x: f32) -> f32 {
    let floor = x.floor();
    let frac = x - floor;
    if frac > 0.5 || (frac == 0.5 && (floor as i64) & 1 != 0) {
        floor + 1.0
    } else {
        floor
    }
}

/// Pack the A operand as int8, with the per-row scale the dequant needs later.
///
/// The rule is the one the packer's `quantise_a_int8` uses and the container's
/// `.asmooth` makes possible:
///
/// ```text
/// v[j]  = a[i, j] * recip[j]          # recip = 1 / asmooth[j]
/// sa[i] = max_j |v[j]| / 127          # 1 when the row is all zero
/// Aq[i,j] = clip(rint(v[j] / sa[i]), -127, 127)
/// ```
///
/// The padded tail is a padded ROW: its bytes and its scale are zero/one so a
/// stale row cannot contribute another call's activations. Both buffers are
/// reused and grow only when a larger tier appears.
pub(crate) fn pack_a_quant_into(
    a: &[f32],
    n_real: usize,
    rows: usize,
    k: usize,
    quant: &QuantScales,
    packed: &mut Vec<u8>,
    scales: &mut Vec<f32>,
) -> Result<(), ArrayError> {
    if quant.recip.len() != k {
        return Err(ArrayError::Design(format!(
            "the SmoothQuant vector holds {} factors and the projection reduces over {k}",
            quant.recip.len()
        )));
    }
    let total = rows * k;
    if packed.len() < total {
        packed.resize(total, 0);
    }
    if scales.len() < rows {
        scales.resize(rows, 0.0);
    }
    for r in 0..rows {
        let dst = &mut packed[r * k..(r + 1) * k];
        if r >= n_real {
            // A padded row: zero bytes, unit scale. The design computes it
            // whatever it holds, and only zero cannot stand for a real token.
            dst.fill(0);
            scales[r] = 1.0;
            continue;
        }
        let src = &a[r * k..(r + 1) * k];
        let mut absmax = 0.0f32;
        for (value, recip) in src.iter().zip(&quant.recip) {
            absmax = absmax.max((*value * *recip).abs());
        }
        let sa = if absmax > 0.0 && absmax.is_finite() {
            absmax / 127.0
        } else {
            1.0
        };
        scales[r] = sa;
        // The inner loop is row-local, so a coarse split across rows would
        // change neither the scale nor the bytes. It is deliberately NOT split
        // here: the work is small per row and the scale was just computed in
        // this same pass, which keeps the value/scale pairing local to a core.
        for (j, value) in src.iter().enumerate() {
            let v = (*value * quant.recip[j]) / sa;
            let q = rint_even(v).clamp(-127.0, 127.0) as i8;
            dst[j] = q as u8;
        }
    }
    Ok(())
}

/// Widen a raw int32 accumulator row into f32, rescale and add the bias.
///
/// The arithmetic order is the scheme's: `(acc as f32) * sa[row] * wscale[col]
/// + bias[col]`. Doing the row scale first is what a rank-1 dequant means, and
/// keeping that order makes this pass bit-identical to the reference's
/// `dequantise_c`.
pub(crate) fn dequant_quant_rows(
    c: &[u8],
    out: &mut [f32],
    n_real: usize,
    n: usize,
    bias: Option<&[f32]>,
    quant: &QuantScales,
    row_scales: &[f32],
) -> Result<(), ArrayError> {
    if quant.wscale.len() != n {
        return Err(ArrayError::Design(format!(
            "the per-channel weight scale holds {} values and the projection has {n} columns",
            quant.wscale.len()
        )));
    }
    if row_scales.len() < n_real {
        return Err(ArrayError::Design(
            "the activation scales are shorter than the dispatched rows".to_string(),
        ));
    }
    if c.len() < n_real * n * 4 {
        return Err(ArrayError::Design(
            "the int32 C buffer is shorter than the dispatched rows".to_string(),
        ));
    }
    let bytes_per_row = n * 4;
    crate::parallel::for_each_row(&mut out[..n_real * n], n, |r, dst| {
        let src = &c[r * bytes_per_row..(r + 1) * bytes_per_row];
        let sa = row_scales[r];
        for (col, (cell, value)) in src.chunks_exact(4).zip(dst.iter_mut()).enumerate() {
            let acc = i32::from_le_bytes([cell[0], cell[1], cell[2], cell[3]]);
            let scaled = (acc as f32) * sa * quant.wscale[col];
            *value = match bias {
                Some(bias) => scaled + bias[col],
                None => scaled,
            };
        }
    });
    Ok(())
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
    if dp.kind != DatapathKind::Bf16 {
        return Err(ArrayError::Design(
            "the bf16 read-back was asked of an int8 datapath".to_string(),
        ));
    }
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
    submit_packed_with(design, instr, a_packed, || {})
}

/// Like [`submit_packed`], but run `between` after the host write and before
/// the device sync.
///
/// The copy into the A slot and the sync that publishes it are separate
/// transport calls. On a transport whose sync waits for a DMA, host work
/// placed between the two overlaps with the transfer instead of sitting behind
/// it. The bias/row-scale copies that a ticket must own are exactly that work:
/// they have to happen before `start_dispatch` captures the bindings, but not
/// before the A bytes are handed to the driver.
pub(crate) fn submit_packed_with<T: Transport, F: FnOnce()>(
    design: &mut ArrayDesign<T>,
    instr: usize,
    a_packed: &[u8],
    between: F,
) -> Result<crate::transport::Run, ArrayError> {
    design.write_bound_unsynced(0, a_packed)?;
    between();
    design.sync_to_device_bound(0, a_packed.len() as u64)?;
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
    if dp.kind != DatapathKind::Bf16 {
        return Err(ArrayError::Design(
            "the bf16 widening was asked of an int8 datapath".to_string(),
        ));
    }
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

    let row_bytes = n * dp.c_bytes;
    let c = &c_packed[..c_need];
    match bias {
        None => {
            // NO BIAS: widen straight into the output's final place. The old
            // path decoded into the scratch row and then copied it into `out`
            // -- two passes over the row when one is enough. The widening
            // helper already writes f32 values in order, so handing it the
            // destination slice removes the intermediate copy entirely. The
            // `row` scratch is left untouched (and need not even be sized).
            crate::parallel::for_each_row(&mut out[..n_real * n], n, |r, dst| {
                let src = r * row_bytes;
                npum_format::bf16::bf16_to_f32_bytes(&c[src..src + row_bytes], dst);
            });
        }
        Some(bias) => {
            // BIAS: widen straight into the output with the bias added, in one
            // pass. The old path decoded into a scratch row and then read that
            // row again for the add -- two visits to every output value. The
            // fused helper writes the final f32 value in one visit, so the
            // scratch row is gone from this path entirely.
            crate::parallel::for_each_row(&mut out[..n_real * n], n, |r, dst| {
                let src = r * row_bytes;
                npum_format::bf16::bf16_to_f32_add_bias_bytes(&c[src..src + row_bytes], dst, bias);
            });
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
    /// The container scales this run was submitted with, when it is int8.
    ///
    /// They are owned by the ticket (an `Arc` into the backend's panel cache),
    /// so the projection set can be advanced while the device computes.
    pub(crate) quant: Option<Arc<QuantScales>>,
    /// The scratch's row-scale slot holding this run's per-row A scales.
    pub(crate) scale_slot: Option<usize>,
}

/// Take a reusable `Vec` slot from a free list, growing the list when empty.
#[inline]
fn take_slot<'a>(pool: &'a mut Vec<Vec<f32>>, free: &mut Vec<usize>) -> (usize, &'a mut Vec<f32>) {
    let index = match free.pop() {
        Some(index) => index,
        None => {
            let index = pool.len();
            pool.push(Vec::new());
            index
        }
    };
    (index, &mut pool[index])
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
    // Copy the bias into a reusable slot while the A bytes are on their way to
    // the device, so the copy is not on the critical path of the transfer. The
    // ticket must own its bias (the caller's slice may change while the device
    // works), but the allocation can outlive many runs.
    let mut bias_slot = None;
    let run = submit_packed_with(design, instr, &scratch.a_packed[..a_need], || {
        bias_slot = bias.map(|b| {
            let (index, slot) = take_slot(&mut scratch.bias_slots, &mut scratch.bias_free);
            slot.clear();
            slot.extend_from_slice(b);
            index
        });
    })?;
    // Remember where THIS run writes. The binding may be moved for the next
    // dispatch before this one's C is read.
    let c_slot = design.bound(2)?;
    Ok(GemmDispatch {
        run,
        n_real,
        dp,
        n,
        bias_slot,
        c_slot,
        quant: None,
        scale_slot: None,
    })
}

/// Pack `a` as int8, submit it, and START a GEMM without waiting for it.
///
/// The int8 counterpart of [`submit_gemm`]. The ticket owns a pooled copy of
/// the per-row activation scales (needed at read-back) and an `Arc` of the
/// container's scales, so the caller is free to move on to another projection
/// while this one computes. Complete it with [`complete_gemm`], which handles
/// both datapaths from the ticket alone.
#[allow(clippy::too_many_arguments)]
pub fn submit_gemm_quant<T: Transport>(
    design: &mut ArrayDesign<T>,
    instr: usize,
    a: &[f32],
    n_real: usize,
    rows: usize,
    k: usize,
    dp: Datapath,
    bias: Option<&[f32]>,
    n: usize,
    quant: Arc<QuantScales>,
    scratch: &mut GemmScratch,
) -> Result<GemmDispatch, ArrayError> {
    check_shape(a.len(), n_real, rows, k, n, n_real * n, bias)?;
    check_fit(design, rows, k, n, dp)?;
    if dp.kind != DatapathKind::Int8I32 || dp.a_bytes != 1 {
        return Err(ArrayError::Design(format!(
            "an int8 submission needs the i8/i32 datapath, and this design's A is {} bytes",
            dp.a_bytes
        )));
    }
    pack_a_quant_into(
        a,
        n_real,
        rows,
        k,
        &quant,
        &mut scratch.a_packed,
        &mut scratch.a_scales,
    )?;
    let a_need = rows * k * dp.a_bytes;
    // Both owned copies -- the bias and the per-row activation scales -- are
    // host work that belongs before `start_dispatch`. Doing them between the
    // A write and its sync hides them behind the transfer.
    let mut bias_slot = None;
    let mut scale_slot = None;
    let run = submit_packed_with(design, instr, &scratch.a_packed[..a_need], || {
        bias_slot = bias.map(|b| {
            let (index, slot) = take_slot(&mut scratch.bias_slots, &mut scratch.bias_free);
            slot.clear();
            slot.extend_from_slice(b);
            index
        });
        let (index, slot) = take_slot(&mut scratch.a_scale_slots, &mut scratch.a_scale_free);
        slot.clear();
        slot.extend_from_slice(&scratch.a_scales[..rows]);
        scale_slot = Some(index);
    })?;
    let c_slot = design.bound(2)?;
    Ok(GemmDispatch {
        run,
        n_real,
        dp,
        n,
        bias_slot,
        c_slot,
        quant: Some(quant),
        scale_slot,
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
    let run = dispatch.run;
    ArrayDesign::<T>::wait_dispatch(run)?;
    design.bind(2, dispatch.c_slot)?;
    // The bias borrows one field of the scratch and the read-back buffer is
    // another, so the two can be live at once. The borrow ends with the call;
    // the slot is then free for the next submission.
    let bias = dispatch
        .bias_slot
        .map(|index| scratch.bias_slots[index].as_slice());
    let result = match dispatch.dp.kind {
        DatapathKind::Bf16 => read_dispatched(
            design,
            dispatch.n_real,
            dispatch.dp,
            bias,
            dispatch.n,
            out,
            &mut scratch.c_packed,
        ),
        DatapathKind::Int8I32 => {
            let quant = dispatch.quant.as_deref().ok_or_else(|| {
                ArrayError::Design("an int8 ticket lost its container scales".to_string())
            })?;
            let row_scales = dispatch
                .scale_slot
                .map(|index| scratch.a_scale_slots[index].as_slice())
                .ok_or_else(|| {
                    ArrayError::Design("an int8 ticket lost its row scales".to_string())
                })?;
            let c_need = dispatch.n_real * dispatch.n * dispatch.dp.c_bytes;
            if scratch.c_packed.len() < c_need {
                scratch.c_packed.resize(c_need, 0);
            }
            design.read(2, &mut scratch.c_packed[..c_need])?;
            dequant_quant_rows(
                &scratch.c_packed[..c_need],
                out,
                dispatch.n_real,
                dispatch.n,
                bias,
                quant,
                row_scales,
            )
        }
    };
    if let Some(index) = dispatch.bias_slot {
        scratch.bias_free.push(index);
    }
    if let Some(index) = dispatch.scale_slot {
        scratch.a_scale_free.push(index);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rint_even_breaks_ties_to_even() {
        assert_eq!(rint_even(2.5), 2.0);
        assert_eq!(rint_even(3.5), 4.0);
        assert_eq!(rint_even(-2.5), -2.0);
        assert_eq!(rint_even(-3.5), -4.0);
        assert_eq!(rint_even(1.49), 1.0);
        assert_eq!(rint_even(1.51), 2.0);
    }

    #[test]
    fn quant_scales_refuse_bad_factors() {
        assert!(QuantScales::new(&[], vec![1.0]).is_err());
        assert!(QuantScales::new(&[1.0], vec![0.0]).is_err());
        assert!(QuantScales::new(&[0.0], vec![1.0]).is_err());
        assert!(QuantScales::new(&[f32::NAN], vec![1.0]).is_err());
        let q = QuantScales::new(&[2.0, 4.0], vec![0.5, 1.0]).unwrap();
        assert_eq!(q.recip, vec![0.5, 0.25]);
        assert_eq!(q.wscale, vec![0.5, 1.0]);
    }

    #[test]
    fn pack_a_quant_uses_row_scale_and_zero_tail() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let quant = QuantScales::new(&[1.0, 2.0, 4.0, 8.0], vec![1.0, 1.0]).unwrap();
        let mut packed = Vec::new();
        let mut scales = Vec::new();
        pack_a_quant_into(&a, 2, 3, 4, &quant, &mut packed, &mut scales).unwrap();
        assert_eq!(scales.len(), 3);
        assert!((scales[0] - 1.0 / 127.0).abs() < 1e-7);
        assert_eq!(scales[1], 1.0);
        assert_eq!(scales[2], 1.0);
        // v = [1, 1, 0.75, 0.5] -> /sa = [127, 127, 95.25, 63.5];
        // rint's tie-to-even makes the last one 64.
        assert_eq!(&packed[..4], &[127u8 as u8, 127, 95, 64]);
        assert_eq!(&packed[4..8], &[0, 0, 0, 0]);
        assert_eq!(&packed[8..12], &[0, 0, 0, 0]);
    }

    #[test]
    fn dequant_quant_rescales_before_the_bias() {
        let quant = QuantScales::new(&[1.0, 1.0], vec![2.0, 4.0]).unwrap();
        // Two int32 rows: [10, 20] and [5, 6], little-endian.
        let mut c = vec![0u8; 2 * 2 * 4];
        for (at, value) in [(0usize, 10i32), (1, 20), (2, 5), (3, 6)] {
            c[at * 4..at * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        let row_scales = [0.5f32, 0.25];
        let bias = [1.0f32, 2.0];
        let mut out = [0.0f32; 4];
        dequant_quant_rows(&c, &mut out, 2, 2, Some(&bias), &quant, &row_scales).unwrap();
        assert_eq!(out, [11.0, 42.0, 3.5, 8.0]);
    }
}
