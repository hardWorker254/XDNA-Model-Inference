//! The array's attention: two GEMMs and a softmax between them.
//!
//! # WHAT THIS IS
//!
//! The attention of a transformer is `QK^T`, a softmax, and `softmax.V` --
//! and the array can take the two GEMMs while the softmax stays on the host,
//! or the other way round. The two are separate decisions because the flag
//! names them separately, and this file is the array's half of both.
//!
//! It is the C++ runtime's `NpuAttention`, and it is the same object: two
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

use crate::array::{ArrayDesign, ArrayError};
use crate::gemm::gemm;
use crate::transport::Transport;

/// The data buffer the panels live in.
///
/// The design's data buffers are A, B and C, in that order: the activations,
/// the operands, the result. The attention's panels are operands, so they
/// live in the B buffer -- the same one the projections' weight panels live
/// in, because it is the same design.
const BUFFER_OPERANDS: usize = 1;

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

/// The attention's geometry: the two streams, the panels' slots, and the
/// tiling the panels are laid out in.
#[derive(Debug, Clone)]
pub struct Attention {
    /// The QK^T stream.
    pub qk: AttnStream,

    /// The softmax.V stream.
    pub av: AttnStream,

    /// The slot in the operands' buffer the K panel lives in.
    pub k_slot: usize,

    /// The slot in the operands' buffer the V panel lives in.
    pub v_slot: usize,

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
}

impl Attention {
    /// Find the attention's streams in a design and stage its panel slots.
    ///
    /// The design is passed in and not opened here: the caller has already
    /// decided where the design set comes from, and a design set loaded from
    /// the wrong place is a wrong answer rather than an error.
    ///
    /// THE PANEL SLOTS ARE STAGED HERE, BEFORE THE FIRST RUN. The panels are
    /// rewritten per head, but they are REWRITTEN IN PLACE: one slot each,
    /// staged once, and a slot per head per layer per window would allocate
    /// `heads * layers` buffers and never reuse one.
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
                design.info().name, qk.n, av.k
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
                design.info().name, layout.order
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
        let capacity = design.info().buffers.get(BUFFER_OPERANDS).copied().unwrap_or(0);
        if capacity < k_bytes as u64 || capacity < v_bytes as u64 {
            return Err(ArrayError::Design(format!(
                "{}: the B operands of attn_qk/attn_av are {k_bytes} and {v_bytes} \
                 bytes and the design's B buffer is {capacity}. This set was exported \
                 without the attention streams.",
                design.info().name
            )));
        }

        // The slots: one for each panel, sized for the panel.
        let k_slot = design.stage(BUFFER_OPERANDS, &vec![0u8; k_bytes])?;
        let v_slot = design.stage(BUFFER_OPERANDS, &vec![0u8; v_bytes])?;

        Ok(Self {
            qk,
            av,
            k_slot,
            v_slot,
            tile_k,
            tile_n,
            mac_s,
            mac_t,
            dispatches: 0,
            qk_seconds: 0.0,
            av_seconds: 0.0,
        })
    }

    /// The width of a score row: the padded key count.
    pub fn score_cols(&self) -> usize {
        self.qk.n
    }
}

/// Find one of the attention's streams in a design.
fn find_stream<T: Transport>(
    design: &ArrayDesign<T>,
    op: &str,
) -> Result<AttnStream, ArrayError> {
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
    //
    // The Q operand of a score is one head, so the design's K is the head
    // width rounded UP to the design's tile -- never down to a head, and
    // never a width a head does not divide. A design whose K is 32 and a
    // head that is 48 has no mapping at all.
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
    // The design was compiled for batches of four, eight and sixteen
    // sequences, and the tier is the one the RUN's rows call for -- the same
    // rule the projections use, because it is the same design.
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

    // THE SCORE BUFFER, AND THE CONTEXT ROWS.
    //
    // The scores are the design's own row width -- the PADDED key count --
    // because the row is what the softmax will read, and a softmax that
    // normalises a different number of keys than the row holds is a softmax
    // over the wrong row.
    let mut scores = vec![0.0f32; tier_rows * n_kv];
    let mut ctx_rows = vec![0.0f32; tier_rows * ctx_cols];
    let mut a_q = vec![0.0f32; tier_rows * k_qk];

    // THE BATCH THE DESIGN HOLDS, AND WHY THE RUN PADS UP TO IT.
    //
    // The attention's dispatch is ONE SEQUENCE at a time, and that is forced
    // by the hardware rather than chosen: attn_qk's B operand is one
    // sequence's K panel, shared by every query row the dispatch computes,
    // and a panel that held two sequences' keys would be a panel of the
    // wrong width.
    //
    // So a run that has fewer sequences than the tier PADS: it computes the
    // tier's worth and throws the padding away. The C++'s encoder does the
    // same -- its batch IS the tier's -- and the dispatch count is the
    // number that says what the array did, so a run that computed fewer
    // sequences would be a run that dispatched a different number of times.
    //
    // The padding is inert: every sequence's attention is its own, and the
    // padded sequences' contexts are never read.
    let tier_batch = tier_rows / seq;
    if tier_batch < batch {
        return Err(ArrayError::Design(format!(
            "the design's tier computes {tier_rows} rows, which is {tier_batch} sequences of \
             {seq}, and the run has {batch}."
        )));
    }

    // THE PADDING'S ACTIVATIONS.
    //
    // Zeros, and not the projections' output for the padded rows: the
    // padded sequences' contexts are discarded, so their numbers are
    // numbers nobody reads, and the dispatch count -- which IS read -- does
    // not depend on them.
    let zeros = vec![0.0f32; seq * 3 * hidden];

    for b in 0..tier_batch {
        let real = b < batch;
        let src = if real {
            &qkv[b * seq * 3 * hidden..(b + 1) * seq * 3 * hidden]
        } else {
            &zeros[..]
        };
        let ctx_base = b * seq * hidden;
        let mk: &[f32] = if real {
            &mask[b * seq..(b + 1) * seq]
        } else {
            &zeros[..seq]
        };

        for head in 0..heads {
            // THE PANELS.
            //
            // They depend on the head and on nothing else -- not on the
            // query chunk -- so they are built once and the chunks run
            // inside them.
            let k_panel = build_k_panel(src, seq, hidden, head, head_dim, k_qk, n_kv);
            let v_panel = build_v_panel(src, seq, hidden, head, head_dim, n_kv, ctx_cols);
            let k_tiled = tile(&k_panel, k_qk, n_kv, attn)?;
            let v_tiled = tile(&v_panel, n_kv, ctx_cols, attn)?;

            // Written through the NAMED slot, not through "whichever B is
            // bound": the two panels are two different slots and the last
            // dispatch left one of them live.
            design.bind(BUFFER_OPERANDS, attn.k_slot)?;
            design.write_bound(BUFFER_OPERANDS, &k_tiled)?;
            design.bind(BUFFER_OPERANDS, attn.v_slot)?;
            design.write_bound(BUFFER_OPERANDS, &v_tiled)?;

            let mut r0 = 0usize;
            while r0 < seq {
                let n = tier_rows.min(seq - r0);

                // THE Q ROWS: one head's slice, contiguous, at the PADDED
                // row stride. The columns past a head are zero, and they
                // contribute `0*0` to every score -- which is what makes a
                // head narrower than the tile computable.
                //
                // The buffer is zeroed ONCE and never again: the gather
                // writes the first head_dim of each real row and nothing
                // else, so the padding is the zeros it started with.
                for i in 0..n {
                    let at = (r0 + i) * 3 * hidden + head * head_dim;
                    a_q[i * k_qk..i * k_qk + head_dim].copy_from_slice(&src[at..at + head_dim]);
                }

                // THE QK^T.
                let t0 = std::time::Instant::now();
                design.bind(BUFFER_OPERANDS, attn.k_slot)?;
                gemm(
                    design,
                    qk_instr,
                    &a_q,
                    n,
                    tier_rows,
                    k_qk,
                    None,
                    n_kv,
                    &mut scores,
                )?;

                // THE MASK, AND THE PADDED KEYS.
                //
                // The mask is added to the row before the softmax, because
                // the softmax's maximum has to see it -- and it covers the
                // REAL keys only, so the padding columns keep the -1e30
                // they are given here.
                //
                // The scale is ONE, and it is applied and not assumed: the
                // packer folded 1/sqrt(head_dim) into Q's weight and bias,
                // which is why the host path applies no scale either.
                for i in 0..n {
                    let row = &mut scores[i * n_kv..(i + 1) * n_kv];
                    for j in 0..seq {
                        row[j] = row[j] * 1.0 + mk[j];
                    }
                    for slot in row.iter_mut().take(n_kv).skip(seq) {
                        *slot = -1.0e30;
                    }
                }

                // THE SOFTMAX, ON THE ARRAY OR ON THE HOST.
                match softmax.as_deref_mut() {
                    Some(softmax) => softmax
                        .attention_softmax(&mut scores[..n * n_kv], n)
                        .map_err(|e| ArrayError::Design(e.to_string()))?,
                    None => {
                        for i in 0..n {
                            host_softmax_row(&mut scores[i * n_kv..(i + 1) * n_kv]);
                        }
                    }
                }
                attn.qk_seconds += t0.elapsed().as_secs_f64();

                // THE softmax.V.
                let t1 = std::time::Instant::now();
                design.bind(BUFFER_OPERANDS, attn.v_slot)?;
                gemm(
                    design,
                    av_instr,
                    &scores,
                    n,
                    tier_rows,
                    n_kv,
                    None,
                    ctx_cols,
                    &mut ctx_rows,
                )?;

                // Row i of the C buffer starts at i * ctx_cols, NOT at
                // i * head_dim: the design's N is padded up to its own
                // granularity, so a context row is wider than a head and
                // holds one head's worth of answer. Reading it with a
                // head_dim stride takes the first head_dim floats of row 0,
                // then the SECOND head_dim floats of row 0 -- which are the
                // padding zeros -- and every other row's output lands one
                // row early. It is a plausible-looking context of the wrong
                // values, which is the worst shape a bug in this file can
                // take.
                //
                // The PADDING SEQUENCES' CONTEXTS ARE NOT WRITTEN: they are
                // computed and thrown away, and the family's buffer holds
                // the real sequences only.
                if real {
                    for i in 0..n {
                        let from = &ctx_rows[i * ctx_cols..i * ctx_cols + head_dim];
                        let dst = ctx_base + (r0 + i) * hidden + head * head_dim;
                        ctx[dst..dst + head_dim].copy_from_slice(from);
                    }
                }
                attn.av_seconds += t1.elapsed().as_secs_f64();
                attn.dispatches += 2;

                r0 += tier_rows;
            }
        }
    }
    Ok(())
}

/// The K panel: `[k_qk, n_kv]`, row-major.
///
/// Row `t` of the panel is key `t`'s components -- the panel is the K half
/// of the Q|K|V block, transposed into the MMAC's `[K, N]` operand.
fn build_k_panel(
    qkv: &[f32],
    seq: usize,
    hidden: usize,
    head: usize,
    head_dim: usize,
    k_qk: usize,
    n_kv: usize,
) -> Vec<f32> {
    let mut m = vec![0.0f32; k_qk * n_kv];
    for t in 0..head_dim {
        for j in 0..seq {
            m[t * n_kv + j] = qkv[j * 3 * hidden + hidden + head * head_dim + t];
        }
    }
    m
}

/// The V panel: `[n_kv, ctx_cols]`, row-major.
///
/// Row `j` of the panel is value `j`'s components -- the values are the
/// SECOND half of the K|V block, and the columns past a head stay zero.
fn build_v_panel(
    qkv: &[f32],
    seq: usize,
    hidden: usize,
    head: usize,
    head_dim: usize,
    n_kv: usize,
    ctx_cols: usize,
) -> Vec<f32> {
    let mut m = vec![0.0f32; n_kv * ctx_cols];
    for j in 0..seq {
        for t in 0..head_dim {
            m[j * ctx_cols + t] = qkv[j * 3 * hidden + 2 * hidden + head * head_dim + t];
        }
    }
    m
}

/// The panel, in the MMAC's byte order.
///
/// The same order the container's own panels are stored in, and the same
/// one the C++'s `tile_b_panel` produces: `[K,N] -> [K/tile_k][N/tile_n]
/// [tile_k/mac_s][tile_n/mac_t][s][t]`.
fn tile(panel: &[f32], k: usize, n: usize, attn: &Attention) -> Result<Vec<u8>, ArrayError> {
    let mut bytes = vec![0u8; panel.len() * 2];
    npum_format::bf16::f32_to_bf16_bytes(panel, &mut bytes);
    npum_format::tiling::tile_b(
        &bytes,
        k,
        n,
        attn.tile_k,
        attn.tile_n,
        2,
        Some(attn.mac_s),
        Some(attn.mac_t),
        npum_format::tiling::TileOrder::Kn,
    )
    .map_err(|e| ArrayError::Design(format!("the attention's panel: {e}")))
}

/// The softmax over one score row, on the host.
///
/// THE C++'s `NpuAttention::host_softmax`, and not the host path's own
/// softmax: the attention's is a plain loop over the row, while the host
/// path's is an eight-lane one, and the two sum the exponentials in
/// different orders. The row here is the PADDED one -- every column of it,
/// padding included -- and the padding's -1e30 underflows to a zero term.
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
