//! The array's attention: two GEMMs and a softmax between them.
//!
//! # WHAT THIS IS
//!
//! The attention of a transformer is `QK^T`, a softmax, and `softmax.V` --
//! and the array can take the two GEMMs while the softmax stays on the host,
//! or the other way round. The two are separate decisions because the flag
//! names them separately, and this file is the array's half of both.
//!
//! It is the reference runtime's `NpuAttention`, and it is the same object: two
//! instruction streams of the projections' OWN set -- `attn_qk` and
//! `attn_av` -- and the panels they multiply by are ACTIVATIONS, not
//! weights. A weight panel is staged once and reused; these are rebuilt for
//! every head, because every head reads a different slice of the keys.
//!
//! # THE GEOMETRY, AND WHO OWNS IT
//!
//! The design owns the numbers and this file reads them:
//!
//! * `attn_qk`'s K is the padded head width -- one head, rounded UP to the
//!   design's tile. Never down to a head: a head that does not divide the K
//!   is not a head this design can compute.
//! * `attn_qk`'s N is the padded key count, and `attn_av`'s K is the same
//!   number: the score chunk travels from one to the other as the A
//!   operand, so the two are one width.
//! * `attn_av`'s N is the padded context width -- one head again, rounded up
//!   to the design's own N granularity.
//!
//! # THE PANELS ARE ZEROED BEFORE THEY ARE FILLED
//!
//! A panel has to tile evenly, and the part of it that is not a real key or
//! a real value is padding. That padding is ZERO and not stale: a stale
//! value there is another head's key taking part in this head's attention,
//! and the runtime masks the matching SCORE columns but cannot mask the
//! VALUE that a zeroed V is what makes the score masking sufficient.
//!
//! The rows past a head are zeroed for the same reason, and paired with the
//! zero tail of the gathered Q they contribute `0*0` -- which is what makes
//! a 32-wide head computable on a tile of 64.
//!
//! # THE SCRATCH IS REUSED
//!
//! A transformer runs attention per head, per chunk, per layer, and a buffer
//! allocated for every panel or every call would spend its time in the
//! allocator rather than on the array. So the buffers this run needs -- the
//! scores, the context rows, the gathered Q, the K and V panels in row-major
//! and in the MMAC's byte order -- live in the attention and grow when a
//! larger run appears.

use crate::array::{ArrayDesign, ArrayError};
use crate::transport::Transport;

/// The data buffer the panels live in.
///
/// The design's data buffers are A, B and C, in that order: the activations,
/// the operands, the result. The attention's panels are operands, so they
/// live in the B buffer -- the same one the projections' weight panels live
/// in, because it is the same design.
const BUFFER_OPERANDS: usize = 1;

/// How many keys the K-panel's transpose gathers at a time.
///
/// The gather from the Q|K|V block is contiguous per key, and the write into
/// the panel is contiguous per component. Doing the transpose a block of keys
/// at a time keeps both ends local: a block's worth of key slices stays in L1
/// while it is transposed into `BLOCK` float-long runs in the panel, instead
/// of writing one float per row with a stride of `n_kv`.
const K_PANEL_BLOCK: usize = 32;

/// One of the attention's instruction streams, and the tiers it has.
///
/// The tiers are the design's: the stream was compiled for batches of four,
/// eight and sixteen sequences, and each tier computes a fixed number of
/// rows. A run with fewer rows than that computes the rest of them anyway.
#[derive(Debug, Clone)]
pub struct AttnStream {
    /// The tiers: the rows one dispatch computes, and the instruction slot.
    pub tiers: Vec<(usize, usize)>,

    /// The reduction width.
    pub k: usize,

    /// The output width.
    pub n: usize,
}

impl AttnStream {
    /// The tier that holds `rows` rows, and the instruction slot.
    ///
    /// The smallest one that fits: a tier computes a fixed number of rows
    /// whatever the caller has, and the smallest tier that holds the run's
    /// rows is the one that wastes the least.
    pub fn tier(&self, rows: usize) -> Option<(usize, usize)> {
        self.tiers
            .iter()
            .find(|(tier, _)| *tier >= rows)
            .copied()
            .or_else(|| self.tiers.last().copied())
    }
}

/// The buffers one attention run needs.
///
/// They are the attention's, and they are REUSED across calls and across
/// heads: the sizes grow to the largest shape seen and stay there. The K
/// and V panels exist in three forms -- row-major f32, bf16 bytes, and the
/// MMAC's tiled byte order -- because the host builds them in the first and
/// the array reads them in the last.
#[derive(Debug, Default, Clone)]
struct AttentionScratch {
    /// The score rows, `[tier_rows, n_kv]`.
    scores: Vec<f32>,
    /// The context rows, `[tier_rows, ctx_cols]`.
    ctx_rows: Vec<f32>,
    /// The gathered Q operand, `[tier_rows, k_qk]`.
    a_q: Vec<f32>,
    /// A zero block for the padded batch's activations and mask.
    zeros: Vec<f32>,
    /// The K panel in row-major f32, `[k_qk, n_kv]`.
    k_panel: Vec<f32>,
    /// Staging for the K-panel's blocked transpose, `[K_PANEL_BLOCK, head_dim]`.
    k_stage: Vec<f32>,
    /// The V panel in row-major f32, `[n_kv, ctx_cols]`.
    v_panel: Vec<f32>,
    /// The K panel as bf16 bytes, before tiling.
    k_bf16: Vec<u8>,
    /// The V panel as bf16 bytes, before tiling.
    v_bf16: Vec<u8>,
    /// The K panel in the MMAC's byte order.
    k_tiled: Vec<u8>,
    /// The V panel in the MMAC's byte order.
    v_tiled: Vec<u8>,
    /// Packed A operand staging buffer (bf16), reused across dispatches.
    a_packed: Vec<u8>,
    /// Packed C operand staging buffer (bf16), reused across dispatches.
    c_packed: Vec<u8>,
}

impl AttentionScratch {
    /// Make the scratch large enough for one run.
    ///
    /// A run that is no larger than the previous one reuses the buffers as
    /// they are. The panel and zero buffers are cleared because the panels'
    /// padding and the padded batch's activations must be ZERO, and only the
    /// real parts of them are written over.
    fn ensure_run(
        &mut self,
        tier_rows: usize,
        n_kv: usize,
        ctx_cols: usize,
        k_qk: usize,
        seq: usize,
        hidden: usize,
        head_dim: usize,
    ) {
        ensure_f32(&mut self.scores, tier_rows * n_kv);
        ensure_f32(&mut self.ctx_rows, tier_rows * ctx_cols);
        ensure_f32(&mut self.a_q, tier_rows * k_qk);
        ensure_f32(&mut self.zeros, seq * 3 * hidden);
        ensure_f32(&mut self.k_panel, k_qk * n_kv);
        ensure_f32(&mut self.k_stage, K_PANEL_BLOCK * head_dim);
        ensure_f32(&mut self.v_panel, n_kv * ctx_cols);
        ensure_u8(&mut self.k_bf16, k_qk * n_kv * 2);
        ensure_u8(&mut self.v_bf16, n_kv * ctx_cols * 2);
        ensure_u8(&mut self.k_tiled, k_qk * n_kv * 2);
        ensure_u8(&mut self.v_tiled, n_kv * ctx_cols * 2);
        let a_max = tier_rows * k_qk.max(n_kv) * 2;
        let c_max = tier_rows * n_kv.max(ctx_cols) * 2;
        ensure_u8(&mut self.a_packed, a_max);
        ensure_u8(&mut self.c_packed, c_max);

        // THE PADDING IS ZERO, NOT STALE.
        //
        // The gathered Q is written per chunk but its columns past a head are
        // never touched, the panels only fill their real head/key/value parts,
        // and the padded batch reads a block of zeros -- so these stay zero
        // for the whole run. The scores and context rows are fully written
        // for the rows each chunk reads, so they need no clearing here.
        self.a_q.fill(0.0);
        self.zeros.fill(0.0);
    }
}

/// Grow a float buffer to hold `need` values, reusing its allocation.
fn ensure_f32(buf: &mut Vec<f32>, need: usize) {
    if buf.len() < need {
        buf.resize(need, 0.0);
    }
}

/// Grow a byte buffer to hold `need` bytes, reusing its allocation.
fn ensure_u8(buf: &mut Vec<u8>, need: usize) {
    if buf.len() < need {
        buf.resize(need, 0);
    }
}

/// The attention's geometry: the two streams, the panels' slots, and the
/// tiling the panels are laid out in.
#[derive(Debug, Clone)]
pub struct Attention {
    /// The QK^T stream.
    pub qk: AttnStream,

    /// The softmax.V stream.
    pub av: AttnStream,

    /// Two slot pairs for the operands' buffer: the K panel and the V panel.
    ///
    /// The panels are rewritten per head, and double-buffering lets the next
    /// head's panels be prepared and uploaded while the current head's last
    /// dispatch is still in flight.
    panel_slots: [(usize, usize); 2],

    /// The panel's tile, in K.
    pub tile_k: usize,

    /// The panel's tile, in N.
    pub tile_n: usize,

    /// The MMAC's `s` sub-tile.
    pub mac_s: usize,

    /// The MMAC's `t` sub-tile.
    pub mac_t: usize,

    /// How many dispatches have run.
    pub dispatches: usize,

    /// The QK^T half's time, in seconds.
    pub qk_seconds: f64,

    /// The softmax.V half's time, in seconds.
    pub av_seconds: f64,

    /// The buffers reused across calls, heads and chunks.
    scratch: AttentionScratch,
}

impl Attention {
    /// Find the attention's streams in a design and stage its panel slots.
    ///
    /// The design is passed in and not opened here: the caller has already
    /// decided where the design set comes from, and a design set loaded from
    /// the wrong place is a wrong answer rather than an error.
    ///
    /// THE PANEL SLOTS ARE STAGED HERE, BEFORE THE FIRST RUN. The panels are
    /// rewritten per head, but they are REWRITTEN IN PLACE: two slot pairs are
    /// staged once so the next head's panels can be uploaded while the current
    /// head's last dispatch is in flight.
    pub fn open<T: Transport>(design: &mut ArrayDesign<T>) -> Result<Self, ArrayError> {
        let qk = find_stream(design, "attn_qk")?;
        let av = find_stream(design, "attn_av")?;

        // THE TWO STREAMS' SHAPES AGREE, OR THE DESIGN IS NOT THIS ONE.
        //
        // The score chunk travels from qk's N to av's K as the A operand, so
        // the two are one number. A design whose two streams disagree is a
        // design for another model, and computing it anyway would produce a
        // plausible context of the wrong values.
        if qk.n != av.k {
            return Err(ArrayError::Design(format!(
                "{}: attn_qk's N is {} and attn_av's K is {}. The score chunk \
                 travels from one to the other as the A operand, so the two are \
                 the same padded key count.",
                design.info().name,
                qk.n,
                av.k
            )));
        }

        // THE PANEL'S BYTE ORDER IS THE DESIGN'S.
        let layout = design.info().b_layout.clone().ok_or_else(|| {
            ArrayError::Design(format!(
                "{}: the design set records no B panel layout, and the attention's \
                 panels are built on the host -- there is no container to read them \
                 from. A design without a layout cannot be dispatched.",
                design.info().name
            ))
        })?;
        if layout.order != "k,n,kt,nt" {
            return Err(ArrayError::Design(format!(
                "{}: the B panel layout's order is '{}', and this runtime lays the \
                 attention's panels out in 'k,n,kt,nt'.",
                design.info().name,
                layout.order
            )));
        }
        let tile_k = layout.tile_k as usize;
        let tile_n = layout.tile_n as usize;
        let mac_s = layout.mac_s as usize;
        let mac_t = layout.mac_t as usize;

        // THE PANELS' SIZES, AND THE DESIGN'S BUFFER.
        //
        // The K panel is the head width by the key count, and the V panel is
        // the key count by the context width. A design exported without the
        // attention streams has a B buffer sized for the set's own weights,
        // and writing past it is heap corruption -- so the sizes are checked
        // against the buffer, here, where the answer can still be a refusal.
        let k_bytes = qk.k * qk.n * 2;
        let v_bytes = av.k * av.n * 2;
        let capacity = design
            .info()
            .buffers
            .get(BUFFER_OPERANDS)
            .copied()
            .unwrap_or(0);
        if capacity < k_bytes as u64 || capacity < v_bytes as u64 {
            return Err(ArrayError::Design(format!(
                "{}: the B operands of attn_qk/attn_av are {k_bytes} and {v_bytes} \
                 bytes and the design's B buffer is {capacity}. This set was exported \
                 without the attention streams.",
                design.info().name
            )));
        }

        // Two slot pairs for ping-pong: one head can run while the next head's
        // panels are prepared and uploaded into the alternate pair.
        // Both halves of each pair are filled in full by `upload_panels_unsynced`
        // before their first dispatch, so zeroing them here and syncing the
        // zeros to the device would be work nothing ever reads.
        let k0 = design.stage_uninit(BUFFER_OPERANDS, k_bytes as u64)?;
        let v0 = design.stage_uninit(BUFFER_OPERANDS, v_bytes as u64)?;
        let k1 = design.stage_uninit(BUFFER_OPERANDS, k_bytes as u64)?;
        let v1 = design.stage_uninit(BUFFER_OPERANDS, v_bytes as u64)?;

        Ok(Self {
            qk,
            av,
            panel_slots: [(k0, v0), (k1, v1)],
            tile_k,
            tile_n,
            mac_s,
            mac_t,
            dispatches: 0,
            qk_seconds: 0.0,
            av_seconds: 0.0,
            scratch: AttentionScratch::default(),
        })
    }

    /// The width of a score row: the padded key count.
    pub fn score_cols(&self) -> usize {
        self.qk.n
    }
}

/// Find one of the attention's streams in a design.
fn find_stream<T: Transport>(design: &ArrayDesign<T>, op: &str) -> Result<AttnStream, ArrayError> {
    let info = design.info();
    let mut tiers: Vec<(usize, usize)> = Vec::new();
    let mut k = 0usize;
    let mut n = 0usize;
    let mut seen = false;
    for stream in info.streams.iter().filter(|s| s.op == op) {
        if seen && (stream.k as usize) != k {
            return Err(ArrayError::Design(format!(
                "{}: the {op} streams disagree about their K ({} and {}). A design \
                 set is compiled for one model.",
                info.name, k, stream.k
            )));
        }
        seen = true;
        k = stream.k as usize;
        n = stream.n as usize;
        tiers.push((stream.m as usize, stream.slot as usize));
    }
    if !seen {
        return Err(ArrayError::Design(format!(
            "{}: the design set has no {op} stream. A set exported without the \
             attention streams cannot run the attention on the array.",
            info.name
        )));
    }
    tiers.sort_by_key(|(rows, _)| *rows);
    tiers.dedup_by_key(|(rows, _)| *rows);
    Ok(AttnStream { tiers, k, n })
}

/// Run the attention on the array.
///
/// `qkv` is the batch's Q|K|V block and `mask` is the additive mask. The
/// context comes back in `ctx`.
#[allow(clippy::too_many_arguments)]
pub fn run<T: Transport>(
    design: &mut ArrayDesign<T>,
    attn: &mut Attention,
    qkv: &[f32],
    shape: npum_model::array::AttentionShape,
    mask: &[f32],
    mut softmax: Option<&mut (dyn npum_model::array::ArrayEltwise + '_)>,
    ctx: &mut [f32],
) -> Result<(), ArrayError> {
    let batch = shape.batch;
    let seq = shape.seq;
    let heads = shape.heads;
    let head_dim = shape.head_dim;
    let hidden = shape.hidden;

    // THE SHAPE, BEFORE THE HARDWARE.
    if heads == 0 || head_dim == 0 || hidden == 0 || seq == 0 || batch == 0 {
        return Err(ArrayError::Design(
            "the attention was given a zero extent".to_string(),
        ));
    }
    if heads * head_dim != hidden {
        return Err(ArrayError::Design(format!(
            "{heads} heads of {head_dim} are not a model of {hidden}."
        )));
    }
    if qkv.len() != batch * seq * 3 * hidden {
        return Err(ArrayError::Design(format!(
            "the Q|K|V block holds {} values, and {batch} sequences of {seq} rows of \
             three {hidden} need {}.",
            qkv.len(),
            batch * seq * 3 * hidden
        )));
    }
    if mask.len() != batch * seq {
        return Err(ArrayError::Design(format!(
            "the mask holds {} values, and {batch} sequences of {seq} need {}.",
            mask.len(),
            batch * seq
        )));
    }
    if ctx.len() < batch * seq * hidden {
        return Err(ArrayError::Design(format!(
            "the context holds {} floats, and {batch} sequences of {seq} rows of \
             {hidden} need {}.",
            ctx.len(),
            batch * seq * hidden
        )));
    }

    // THE HEAD IS THE K, AND THE K IS PADDED UP TO A HEAD.
    let k_qk = attn.qk.k;
    if k_qk < head_dim || k_qk % head_dim != 0 {
        return Err(ArrayError::Design(format!(
            "attn_qk's K is {k_qk} and this model's head is {head_dim} wide. The Q \
             operand of a score is one head, so K is the head width padded UP to \
             the design's tile -- never down to a head, and never a value a head \
             does not divide."
        )));
    }
    let n_kv = attn.qk.n;
    let ctx_cols = attn.av.n;
    if ctx_cols < head_dim {
        return Err(ArrayError::Design(format!(
            "attn_av's N is {ctx_cols} and a head is {head_dim} wide. A design's N \
             is padded up to the design's own N granularity, never down to a head."
        )));
    }
    if seq > n_kv {
        return Err(ArrayError::Design(format!(
            "{seq} keys, but the design's attn streams are built for {n_kv}. Re-export \
             the design set with this model's own window."
        )));
    }

    // THE TIER.
    //
    // One QK^T/softmax.V pair covers one head and one tier's worth of rows;
    // the head/chunk granularity here is the compiled streams' geometry. A
    // dispatch that covered more heads or rows would need streams exported for
    // that shape (and panels packed for it) -- it cannot be changed in this
    // loop alone.
    let rows = batch * seq;
    let (tier_rows, qk_instr) = attn.qk.tier(rows).ok_or_else(|| {
        ArrayError::Design(format!(
            "{}: the design set has no attn_qk stream for {rows} rows.",
            design.info().name
        ))
    })?;
    let (_, av_instr) = attn.av.tier(tier_rows).ok_or_else(|| {
        ArrayError::Design(format!(
            "{}: the design set has no attn_av stream for {tier_rows} rows.",
            design.info().name
        ))
    })?;

    // THE BATCH THE DESIGN HOLDS, AND WHY THE RUN PADS UP TO IT.
    let tier_batch = tier_rows / seq;
    if tier_batch < batch {
        return Err(ArrayError::Design(format!(
            "the design's tier computes {tier_rows} rows, which is {tier_batch} sequences of \
             {seq}, and the run has {batch}."
        )));
    }

    // THE SCORE BUFFER, THE CONTEXT ROWS, AND THE WORKSPACE.
    attn.scratch
        .ensure_run(tier_rows, n_kv, ctx_cols, k_qk, seq, hidden, head_dim);
    let tile_k = attn.tile_k;
    let tile_n = attn.tile_n;
    let mac_s = attn.mac_s;
    let mac_t = attn.mac_t;
    let dp = crate::gemm::Datapath::of(design)?;

    for b in 0..tier_batch {
        let real = b < batch;
        // The padded batch reads the zero block already prepared in
        // `ensure_run`; no per-iteration copy or allocation. Both the
        // activations and the (zero) mask are slices of that same block.
        let src: &[f32] = if real {
            &qkv[b * seq * 3 * hidden..(b + 1) * seq * 3 * hidden]
        } else {
            &attn.scratch.zeros[..seq * 3 * hidden]
        };
        let ctx_base = b * seq * hidden;
        let mk: &[f32] = if real {
            &mask[b * seq..(b + 1) * seq]
        } else {
            &attn.scratch.zeros[..seq]
        };

        if heads == 0 {
            continue;
        }

        // Two panel-slot pairs for ping-pong. Head zero's panels are built and
        // uploaded here; every later head's panels are built during the
        // current head's first QK^T dispatch, so the CPU work is off the
        // critical path between that head's two GEMMs.
        let k_bytes = k_qk * n_kv * 2;
        let v_bytes = n_kv * ctx_cols * 2;
        let mut cur = 0usize;
        build_panels(
            &mut attn.scratch.k_panel[..k_qk * n_kv],
            &mut attn.scratch.k_stage,
            &mut attn.scratch.v_panel[..n_kv * ctx_cols],
            &mut attn.scratch.k_tiled,
            &mut attn.scratch.k_bf16,
            &mut attn.scratch.v_tiled,
            &mut attn.scratch.v_bf16,
            src,
            seq,
            hidden,
            0,
            head_dim,
            k_qk,
            n_kv,
            ctx_cols,
            tile_k,
            tile_n,
            mac_s,
            mac_t,
        )?;
        let initial_slots = attn.panel_slots[cur];
        upload_panels_unsynced(
            design,
            &attn.scratch.k_tiled,
            &attn.scratch.v_tiled,
            initial_slots,
            k_bytes,
            v_bytes,
        )?;
        sync_panels(design, initial_slots, k_bytes, v_bytes)?;

        for head in 0..heads {
            let mut r0 = 0usize;
            let mut next_prepared = false;
            let mut next_synced = false;
            // The first chunk's Q is gathered before its QK submit. Every
            // later chunk's Q is gathered during the PREVIOUS chunk's AV
            // dispatch, so the gather is off the critical path between the
            // AV read-back and the next QK submit. A head boundary clears the
            // flag: the next head's first chunk always gathers fresh rows.
            let mut q_ready = false;
            while r0 < seq {
                let n = tier_rows.min(seq - r0);
                let next_r0 = r0 + tier_rows;
                let last_chunk = next_r0 >= seq;
                let k_slot = attn.panel_slots[cur].0;
                let v_slot = attn.panel_slots[cur].1;

                // Gather Q rows for this chunk. On later chunks this was
                // already done while the previous chunk's AV ran, so the
                // submit below can go straight out.
                if !q_ready {
                    gather_q(
                        &mut attn.scratch.a_q,
                        src,
                        r0,
                        n,
                        head,
                        head_dim,
                        hidden,
                        k_qk,
                    );
                }

                // QK^T.
                let t0 = std::time::Instant::now();
                design.bind(BUFFER_OPERANDS, k_slot)?;
                let run_qk = submit_gemm_packed(
                    design,
                    qk_instr,
                    &attn.scratch.a_q,
                    n,
                    tier_rows,
                    k_qk,
                    dp,
                    &mut attn.scratch.a_packed,
                )?;

                // Prepare the NEXT head's panels while this head's QK^T is in
                // flight, rather than only at the tail of the last chunk. The
                // current head's panels are already on the device, so its host
                // scratch is free, and the alternate slot pair is not read by
                // any dispatch still in flight.
                if !next_prepared && head + 1 < heads {
                    let next_slots = attn.panel_slots[1 - cur];
                    build_panels(
                        &mut attn.scratch.k_panel[..k_qk * n_kv],
                        &mut attn.scratch.k_stage,
                        &mut attn.scratch.v_panel[..n_kv * ctx_cols],
                        &mut attn.scratch.k_tiled,
                        &mut attn.scratch.k_bf16,
                        &mut attn.scratch.v_tiled,
                        &mut attn.scratch.v_bf16,
                        src,
                        seq,
                        hidden,
                        head + 1,
                        head_dim,
                        k_qk,
                        n_kv,
                        ctx_cols,
                        tile_k,
                        tile_n,
                        mac_s,
                        mac_t,
                    )?;
                    upload_panels_unsynced(
                        design,
                        &attn.scratch.k_tiled,
                        &attn.scratch.v_tiled,
                        next_slots,
                        k_bytes,
                        v_bytes,
                    )?;
                    next_prepared = true;
                }

                ArrayDesign::<T>::wait_dispatch(run_qk)?;
                read_gemm_f32(
                    design,
                    n,
                    n_kv,
                    dp,
                    &mut attn.scratch.c_packed,
                    &mut attn.scratch.scores,
                )?;

                // Mask and softmax. The two loops stay in the same order as
                // before: add the real keys' mask, then pin the padding cols.
                for i in 0..n {
                    let row = &mut attn.scratch.scores[i * n_kv..(i + 1) * n_kv];
                    for (score, m) in row[..seq].iter_mut().zip(&mk[..seq]) {
                        *score += *m;
                    }
                    row[seq..].fill(-1.0e30);
                }
                match softmax.as_deref_mut() {
                    Some(softmax) => softmax
                        .attention_softmax(&mut attn.scratch.scores[..n * n_kv], n)
                        .map_err(|e| ArrayError::Design(e.to_string()))?,
                    None => {
                        for i in 0..n {
                            host_softmax_row(&mut attn.scratch.scores[i * n_kv..(i + 1) * n_kv]);
                        }
                    }
                }
                attn.qk_seconds += t0.elapsed().as_secs_f64();
                attn.dispatches += 1;

                // softmax.V
                let t1 = std::time::Instant::now();
                design.bind(BUFFER_OPERANDS, v_slot)?;
                let run_av = submit_gemm_packed(
                    design,
                    av_instr,
                    &attn.scratch.scores,
                    n,
                    tier_rows,
                    n_kv,
                    dp,
                    &mut attn.scratch.a_packed,
                )?;

                // Publish the next head's panels now. The transfer overlaps
                // with this head's AV dispatch (a different BO), and the
                // panels are not read until the next head's first GEMM.
                if next_prepared && !next_synced {
                    sync_panels(design, attn.panel_slots[1 - cur], k_bytes, v_bytes)?;
                    next_synced = true;
                }

                // Gather the NEXT chunk's Q while this chunk's AV runs. The
                // QK submit already packed this chunk's Q into its own A slot,
                // so the host scratch is free; the next QK submit then only
                // has to pack these rows instead of waiting to gather them.
                if next_r0 < seq {
                    let next_n = tier_rows.min(seq - next_r0);
                    gather_q(
                        &mut attn.scratch.a_q,
                        src,
                        next_r0,
                        next_n,
                        head,
                        head_dim,
                        hidden,
                        k_qk,
                    );
                    q_ready = true;
                } else {
                    q_ready = false;
                }

                ArrayDesign::<T>::wait_dispatch(run_av)?;
                read_gemm_f32(
                    design,
                    n,
                    ctx_cols,
                    dp,
                    &mut attn.scratch.c_packed,
                    &mut attn.scratch.ctx_rows,
                )?;
                write_ctx_out(
                    attn, real, ctx, ctx_base, r0, n, head, head_dim, hidden, ctx_cols,
                );
                attn.av_seconds += t1.elapsed().as_secs_f64();
                attn.dispatches += 1;

                // Switch to the alternate pair only after the current head's
                // last chunk; earlier chunks still read the current pair.
                if last_chunk && head + 1 < heads {
                    cur = 1 - cur;
                }

                r0 = next_r0;
            }
        }
    }
    Ok(())
}

/// Pack A, write/sync it to buffer 0, and start a GEMM dispatch.
#[allow(clippy::too_many_arguments)]
fn submit_gemm_packed<T: Transport>(
    design: &mut ArrayDesign<T>,
    instr: usize,
    a: &[f32],
    n_real: usize,
    rows: usize,
    k: usize,
    dp: crate::gemm::Datapath,
    a_packed: &mut Vec<u8>,
) -> Result<crate::transport::Run, ArrayError> {
    crate::gemm::pack_a_into(a, n_real, rows, k, dp.a_bytes, a_packed);
    let a_need = rows * k * dp.a_bytes;
    crate::gemm::submit_packed(design, instr, &a_packed[..a_need])
}

/// Read GEMM output from buffer 2 and widen it straight into `out`.
///
/// Only the `n_real` real rows are synced and read back: the output rows lie
/// contiguously from row zero, so the design's padded tail is never touched.
/// The widening writes f32 values in order, so there is no intermediate row
/// to decode into and copy from -- the same single-pass completion the GEMM
/// path uses in `crate::gemm::read_dispatched`.
fn read_gemm_f32<T: Transport>(
    design: &mut ArrayDesign<T>,
    n_real: usize,
    n: usize,
    dp: crate::gemm::Datapath,
    c_packed: &mut Vec<u8>,
    out: &mut [f32],
) -> Result<(), ArrayError> {
    // Reading only `n_real` rows keeps both the sync and the copy off the
    // padded tail.
    let c_need = n_real * n * dp.c_bytes;
    if c_packed.len() < c_need {
        c_packed.resize(c_need, 0);
    }
    design.read(2, &mut c_packed[..c_need])?;
    for r in 0..n_real {
        let src = r * n * dp.c_bytes;
        npum_format::bf16::bf16_to_f32_bytes(
            &c_packed[src..src + n * dp.c_bytes],
            &mut out[r * n..r * n + n],
        );
    }
    Ok(())
}

/// Gather one chunk's Q rows into the A operand for a head.
///
/// Only the `[0..n] x [0..head_dim]` block is written. The columns past a head
/// and the rows past a chunk stay as they were -- zeroed once per run and never
/// touched by the gather, which is what makes the design's padded rows and
/// head-width padding compute as zeros.
#[allow(clippy::too_many_arguments)]
fn gather_q(
    a_q: &mut [f32],
    src: &[f32],
    r0: usize,
    n: usize,
    head: usize,
    head_dim: usize,
    hidden: usize,
    k_qk: usize,
) {
    for i in 0..n {
        let at = (r0 + i) * 3 * hidden + head * head_dim;
        let dst = i * k_qk;
        a_q[dst..dst + head_dim].copy_from_slice(&src[at..at + head_dim]);
    }
}

/// Copy the head's context rows into the output buffer for a real sequence.
#[allow(clippy::too_many_arguments)]
fn write_ctx_out(
    attn: &Attention,
    real: bool,
    ctx: &mut [f32],
    ctx_base: usize,
    r0: usize,
    n: usize,
    head: usize,
    head_dim: usize,
    hidden: usize,
    ctx_cols: usize,
) {
    if !real {
        return;
    }
    for i in 0..n {
        let from = &attn.scratch.ctx_rows[i * ctx_cols..i * ctx_cols + head_dim];
        let dst = ctx_base + (r0 + i) * hidden + head * head_dim;
        ctx[dst..dst + head_dim].copy_from_slice(from);
    }
}

/// Build and tile the K/V panels for one head into the host scratch buffers.
///
/// Split from the upload so the build can overlap with an in-flight dispatch:
/// once a head's panels have been uploaded, its scratch is free and the next
/// head's panels can be built while the current head's GEMM is still running.
///
/// Only the panel buffers are passed in, not the whole attention: `src` may be
/// a slice of the padding batch's zero block, so borrowing these fields
/// separately keeps that read-only borrow alive across the calls.
#[allow(clippy::too_many_arguments)]
fn build_panels(
    k_panel: &mut [f32],
    k_stage: &mut [f32],
    v_panel: &mut [f32],
    k_tiled: &mut Vec<u8>,
    k_bf16: &mut Vec<u8>,
    v_tiled: &mut Vec<u8>,
    v_bf16: &mut Vec<u8>,
    src: &[f32],
    seq: usize,
    hidden: usize,
    head: usize,
    head_dim: usize,
    k_qk: usize,
    n_kv: usize,
    ctx_cols: usize,
    tile_k: usize,
    tile_n: usize,
    mac_s: usize,
    mac_t: usize,
) -> Result<(), ArrayError> {
    fill_k_panel(k_panel, k_stage, src, seq, hidden, head, head_dim, n_kv);
    fill_v_panel(v_panel, src, seq, hidden, head, head_dim, ctx_cols);
    tile_into(
        k_tiled, k_bf16, k_panel, k_qk, n_kv, tile_k, tile_n, mac_s, mac_t,
    )?;
    tile_into(
        v_tiled, v_bf16, v_panel, n_kv, ctx_cols, tile_k, tile_n, mac_s, mac_t,
    )?;
    Ok(())
}

/// Copy the built panels into a slot pair WITHOUT making them visible to the
/// device yet. The writes are plain host copies; `sync_panels` publishes them.
///
/// Splitting the write from the sync lets a caller issue the sync at a point
/// where it overlaps with an in-flight dispatch on the other slot pair.
fn upload_panels_unsynced<T: Transport>(
    design: &mut ArrayDesign<T>,
    k_tiled: &[u8],
    v_tiled: &[u8],
    slots: (usize, usize),
    k_bytes: usize,
    v_bytes: usize,
) -> Result<(), ArrayError> {
    design.bind(BUFFER_OPERANDS, slots.0)?;
    design.write_bound_unsynced(BUFFER_OPERANDS, &k_tiled[..k_bytes])?;
    design.bind(BUFFER_OPERANDS, slots.1)?;
    design.write_bound_unsynced(BUFFER_OPERANDS, &v_tiled[..v_bytes])?;
    Ok(())
}

/// Publish a panel pair that was written with `upload_panels_unsynced`.
///
/// `sync_to_device_bound` acts on the slot currently bound, so the pair is
/// bound again here: a dispatch in between may have rebound the buffer.
fn sync_panels<T: Transport>(
    design: &mut ArrayDesign<T>,
    slots: (usize, usize),
    k_bytes: usize,
    v_bytes: usize,
) -> Result<(), ArrayError> {
    design.bind(BUFFER_OPERANDS, slots.0)?;
    design.sync_to_device_bound(BUFFER_OPERANDS, k_bytes as u64)?;
    design.bind(BUFFER_OPERANDS, slots.1)?;
    design.sync_to_device_bound(BUFFER_OPERANDS, v_bytes as u64)?;
    Ok(())
}

/// The K panel: `[k_qk, n_kv]`, row-major.
///
/// Row `t` of the panel is the `t`-th component of every key -- the panel is
/// the K half of the Q|K|V block, transposed into the MMAC's `[K, N]` operand.
/// That transpose is the reason a plain key-major loop scatters: writing
/// `m[t * n_kv + j]` for successive `t` jumps a whole panel row each time.
///
/// So the transpose is done in blocks of keys. A block's key slices are
/// gathered contiguously (the same read pattern `fill_v_panel` uses) into
/// `stage`, and then written out component by component: for a fixed `t` the
/// `cnt` writes to `m[t * n_kv + j]` are one contiguous run. Both ends are
/// local, and the staging buffer is reused across heads rather than allocated
/// per call.
///
/// Only the `[0..head_dim] x [0..seq]` block is overwritten below. The rest of
/// the panel is padding -- every column of the rows past a head, and the
/// key-tail columns of the head's own rows -- and a stale value there would be
/// another head's key taking part in this head's attention. Zero just that
/// padding instead of `fill(0.0)`-ing the whole panel and rewriting the real
/// block a second time. `stage` only covers the real keys/components, so it
/// needs no clearing.
fn fill_k_panel(
    m: &mut [f32],
    stage: &mut [f32],
    qkv: &[f32],
    seq: usize,
    hidden: usize,
    head: usize,
    head_dim: usize,
    n_kv: usize,
) {
    let mut rows = m.chunks_exact_mut(n_kv);
    for row in rows.by_ref().take(head_dim) {
        row[seq..].fill(0.0);
    }
    for row in rows {
        row.fill(0.0);
    }
    let k_base = hidden + head * head_dim;
    let mut j0 = 0;
    while j0 < seq {
        let j1 = (j0 + K_PANEL_BLOCK).min(seq);
        let cnt = j1 - j0;

        // Gather: each key's head slice is contiguous in the Q|K|V block.
        for (b, j) in (j0..j1).enumerate() {
            let src = j * 3 * hidden + k_base;
            stage[b * head_dim..(b + 1) * head_dim].copy_from_slice(&qkv[src..src + head_dim]);
        }

        // Transpose: each component's `cnt` values are contiguous in the panel.
        for t in 0..head_dim {
            let mut dst = t * n_kv + j0;
            for b in 0..cnt {
                m[dst] = stage[b * head_dim + t];
                dst += 1;
            }
        }

        j0 = j1;
    }
}

/// The V panel: `[n_kv, ctx_cols]`, row-major.
///
/// Row `j` of the panel is value `j`'s components -- the values are the
/// SECOND half of the K|V block, and the columns past a head stay zero.
///
/// Only the `[0..seq] x [0..head_dim]` block is overwritten below. The rest is
/// padding -- every column of the rows past the real keys, and the value-tail
/// columns of the real rows -- so zero just that, as in `fill_k_panel`.
fn fill_v_panel(
    m: &mut [f32],
    qkv: &[f32],
    seq: usize,
    hidden: usize,
    head: usize,
    head_dim: usize,
    ctx_cols: usize,
) {
    let mut rows = m.chunks_exact_mut(ctx_cols);
    for row in rows.by_ref().take(seq) {
        row[head_dim..].fill(0.0);
    }
    for row in rows {
        row.fill(0.0);
    }
    let v_base = 2 * hidden + head * head_dim;
    for j in 0..seq {
        let src = j * 3 * hidden + v_base;
        let dst = j * ctx_cols;
        m[dst..dst + head_dim].copy_from_slice(&qkv[src..src + head_dim]);
    }
}

/// The panel, in the MMAC's byte order, written into reused buffers.
///
/// The same order the container's own panels are stored in, and the same
/// one the reference's `tile_b_panel` produces: `[K,N] -> [K/tile_k][N/tile_n]
/// [tile_k/mac_s][tile_n/mac_t][s][t]`. The bf16 staging buffer and the tiled
/// output are the caller's, so a run that builds many panels reuses them
/// instead of allocating per head.
#[allow(clippy::too_many_arguments)]
fn tile_into(
    tiled: &mut Vec<u8>,
    bf16: &mut Vec<u8>,
    panel: &[f32],
    k: usize,
    n: usize,
    tile_k: usize,
    tile_n: usize,
    mac_s: usize,
    mac_t: usize,
) -> Result<(), ArrayError> {
    let bytes = panel.len() * 2;
    if bf16.len() < bytes {
        bf16.resize(bytes, 0);
    }
    if tiled.len() < bytes {
        tiled.resize(bytes, 0);
    }
    npum_format::bf16::f32_to_bf16_bytes(panel, &mut bf16[..bytes]);
    npum_format::tiling::tile_b_into(
        &mut tiled[..bytes],
        &bf16[..bytes],
        k,
        n,
        tile_k,
        tile_n,
        2,
        Some(mac_s),
        Some(mac_t),
        npum_format::tiling::TileOrder::Kn,
    )
    .map_err(|e| ArrayError::Design(format!("the attention's panel: {e}")))
}

/// The softmax over one score row, on the host.
///
/// THE reference's `NpuAttention::host_softmax`, and not the host path's own
/// softmax: the attention's is a plain loop over the row, while the host
/// path's is an eight-lane one, and the two sum the exponentials in
/// different orders. The row here is the PADDED one -- every column of it,
/// padding included -- and the padding's -1e30 underflows to a zero term.
#[inline]
fn host_softmax_row(row: &mut [f32]) {
    let mut mx = -3.4e38f32;
    for value in row.iter() {
        mx = mx.max(*value);
    }
    let mut sum = 0.0f32;
    for value in row.iter_mut() {
        *value = (*value - mx).exp();
        sum += *value;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for value in row.iter_mut() {
        *value *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The panel the original key-major scatter loop produced, as the oracle.
    fn scatter_k_panel(
        qkv: &[f32],
        seq: usize,
        hidden: usize,
        head: usize,
        head_dim: usize,
        n_kv: usize,
        k_qk: usize,
    ) -> Vec<f32> {
        let mut m = vec![0.0f32; k_qk * n_kv];
        let k_base = hidden + head * head_dim;
        for j in 0..seq {
            let src = j * 3 * hidden + k_base;
            for t in 0..head_dim {
                m[t * n_kv + j] = qkv[src + t];
            }
        }
        m
    }

    #[test]
    fn blocked_k_panel_matches_the_scatter_panel_and_zeroes_its_padding() {
        // A single short block, several full blocks plus a partial one, and a
        // shape that crosses the block boundary with a non-zero head offset.
        for &(seq, hidden, head, head_dim, n_kv, k_qk) in &[
            (5usize, 96usize, 1usize, 32usize, 64usize, 64usize),
            (70, 96, 1, 32, 128, 64),
            (33, 384, 5, 32, 384, 64),
            (K_PANEL_BLOCK * 2, 192, 2, 64, 256, 128),
        ] {
            let qkv: Vec<f32> = (0..seq * 3 * hidden)
                .map(|i| i as f32 * 0.25 - 7.0)
                .collect();
            let want = scatter_k_panel(&qkv, seq, hidden, head, head_dim, n_kv, k_qk);

            // Start from NaN so an unwritten byte of padding is caught as well.
            let mut got = vec![f32::NAN; k_qk * n_kv];
            let mut stage = vec![0.0f32; K_PANEL_BLOCK * head_dim];
            fill_k_panel(
                &mut got, &mut stage, &qkv, seq, hidden, head, head_dim, n_kv,
            );

            assert_eq!(got, want, "seq={seq} head={head} head_dim={head_dim}");
        }
    }

    /// The V panel's oracle: the real `[0..seq] x [0..head_dim]` block copied
    /// from the Q|K|V block, and every padding cell left zero.
    fn scatter_v_panel(
        qkv: &[f32],
        seq: usize,
        hidden: usize,
        head: usize,
        head_dim: usize,
        n_kv: usize,
        ctx_cols: usize,
    ) -> Vec<f32> {
        let mut m = vec![0.0f32; n_kv * ctx_cols];
        let v_base = 2 * hidden + head * head_dim;
        for j in 0..seq {
            let src = j * 3 * hidden + v_base;
            m[j * ctx_cols..j * ctx_cols + head_dim].copy_from_slice(&qkv[src..src + head_dim]);
        }
        m
    }

    #[test]
    fn v_panel_matches_the_reference_and_zeroes_its_padding() {
        // Short and long key windows, a non-zero head offset, and a context
        // width wider than the head so the value-tail padding is exercised.
        for &(seq, hidden, head, head_dim, n_kv, ctx_cols) in &[
            (5usize, 96usize, 1usize, 32usize, 64usize, 192usize),
            (70, 96, 1, 32, 128, 192),
            (33, 384, 5, 32, 384, 192),
            (128, 192, 2, 64, 256, 128),
        ] {
            let qkv: Vec<f32> = (0..seq * 3 * hidden)
                .map(|i| i as f32 * 0.25 - 7.0)
                .collect();
            let want = scatter_v_panel(&qkv, seq, hidden, head, head_dim, n_kv, ctx_cols);

            // Start from NaN so an unwritten cell of padding is caught as well.
            let mut got = vec![f32::NAN; n_kv * ctx_cols];
            fill_v_panel(&mut got, &qkv, seq, hidden, head, head_dim, ctx_cols);

            assert_eq!(got, want, "seq={seq} head={head} head_dim={head_dim}");
        }
    }

    /// The pipeline, end to end on a recording transport. The simulated GEMM
    /// writes zeros, so the numbers are not checked here -- the EVENT ORDER is:
    /// the next head's panels must be written while the current head's QK is
    /// still in flight, their sync must come later (overlapping the AV
    /// dispatch), and each head's two dispatches must use the right slot pair.
    mod pipeline_tests {
        use super::*;
        use crate::array::ArrayDesign;
        use crate::transport::{BufferFlags, Run, TransportError};
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
                // args[1] is the B buffer: the slot captured here is the panel
                // pair this dispatch actually reads.
                self.log
                    .lock()
                    .unwrap()
                    .push(format!("start(B={})", args[1].id));
                let run = Box::new(FakeRun {
                    c: args[2].clone(),
                    log: self.log.clone(),
                });
                Ok(Run::new(Box::into_raw(run) as *mut c_void, free_fake_run))
            }

            fn wait(run: &Run) -> Result<(), TransportError> {
                let r = unsafe { &*(run.handle() as *const FakeRun) };
                r.log.lock().unwrap().push("wait".to_string());
                let mut c = r.c.bytes.lock().unwrap();
                c.fill(0);
                Ok(())
            }
        }

        fn write_design(dir: &Path) {
            let manifest = r#"{
                "name": "fake-attn",
                "kernel": "MLIR_AIE",
                "arch": 1,
                "device": "npu1",
                "buffers": [80, 128, 80],
                "a_dtype": "bf16",
                "c_dtype": "bf16",
                "b_layout": {
                    "kind": "block_panel", "tile_k": 8, "tile_n": 8,
                    "order": "k,n,kt,nt", "inner": "s,t",
                    "mac_s": 4, "mac_t": 4, "dtype": "bf16"
                },
                "b_layout_hash": "fake",
                "tile": { "m": 5, "k": 8, "n": 8, "rows": 1 },
                "slot0": { "op": "", "slot": 0, "file": "insts.bin", "M": 5, "K": 8, "N": 8 },
                "streams": [
                    { "op": "attn_qk", "slot": 1, "file": "insts_qk.bin", "M": 5, "K": 8, "N": 8 },
                    { "op": "attn_av", "slot": 2, "file": "insts_av.bin", "M": 5, "K": 8, "N": 8 }
                ]
            }"#;
            std::fs::write(dir.join("design.json"), manifest).unwrap();
            std::fs::write(dir.join("final.xclbin"), b"xclbin").unwrap();
            for file in ["insts.bin", "insts_qk.bin", "insts_av.bin"] {
                std::fs::write(dir.join(file), vec![0u8; 4]).unwrap();
            }
        }

        #[test]
        fn the_pipeline_uploads_early_syncs_late_and_ping_pongs_slots() {
            let dir = std::env::temp_dir().join(format!(
                "npum_attn_pipeline_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            write_design(&dir);

            let log: Log = Arc::new(Mutex::new(Vec::new()));
            let mut design = ArrayDesign::open(&dir, FakeTransport::new(log.clone())).unwrap();
            let mut attn = Attention::open(&mut design).unwrap();
            log.lock().unwrap().clear(); // drop the design-loading operations

            let shape = npum_model::array::AttentionShape {
                batch: 1,
                seq: 5,
                heads: 3,
                head_dim: 8,
                hidden: 24,
            };
            let qkv = vec![0.25f32; shape.batch * shape.seq * 3 * shape.hidden];
            let mask = vec![0f32; shape.batch * shape.seq];
            let mut ctx = vec![0f32; shape.batch * shape.seq * shape.hidden];
            let dispatches = {
                run(&mut design, &mut attn, &qkv, shape, &mask, None, &mut ctx).unwrap();
                attn.dispatches
            };
            assert_eq!(dispatches, 2 * shape.heads, "two GEMMs per head");

            let log = log.lock().unwrap();
            let b_slots: Vec<&str> = log
                .iter()
                .filter_map(|event| {
                    event
                        .strip_prefix("start(B=")
                        .map(|rest| rest.trim_end_matches(')'))
                })
                .collect();
            // K then V of pair 0, then pair 1, then pair 0 again: head h's two
            // dispatches read its own head's panels, not the next head's.
            // Buffer 1's slot 0 is its base allocation; `Attention::open`
            // stages the two panel pairs as slots 1/2 and 3/4.
            assert_eq!(
                b_slots,
                ["4:1", "4:2", "4:3", "4:4", "4:1", "4:2"],
                "the panel slot pairs must ping-pong across heads"
            );

            let at = |event: &str| log.iter().position(|e| e == event).unwrap();
            // Head 1's K panel (slot 3) is written during head 0's QK dispatch,
            // BEFORE that dispatch is waited on -- the earlier overlap.
            assert!(
                at("write(4:3)") < at("wait"),
                "the next head's panels must be uploaded while the current QK runs"
            );
            // Its sync is deferred past that wait and past head 0's AV submit,
            // overlapping the AV dispatch rather than blocking the build.
            assert!(
                at("sync_to_device(4:3)") > at("wait")
                    && at("sync_to_device(4:3)") > at("start(B=4:2)"),
                "the panel sync must be deferred until after the QK wait and AV submit"
            );

            drop(log);
            drop(attn);
            drop(design);
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }
}
