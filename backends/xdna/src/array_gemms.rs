//! The array's projections, for a family that dispatches them to the board.
//!
//! # WHAT THIS IS
//!
//! The seam's other end, for the text family. A BERT layer is four
//! projections -- the fused QKV, the attention output, and the two FFN
//! halves -- and this is the object that runs them on the array.
//!
//! # THE ATTENTION IS HERE TOO, AND NOT BY ACCIDENT
//!
//! The attention's two GEMMs are streams INSIDE this design set -- `attn_qk`
//! and `attn_av` -- and not a set of their own. So they are not a second
//! backend and not a second context: they are two more instruction streams
//! of the design this object already holds, and they dispatch on it.
//!
//! # THE TIERS ARE THE DESIGN'S
//!
//! The design was compiled for batches of four, eight and sixteen
//! sequences, and each tier is a different instruction stream: the stream
//! computes a fixed number of rows, and a run with fewer rows than that
//! computes the rest of them anyway.
//!
//! So the tier is chosen by the RUN -- the smallest tier that holds the
//! run's rows -- and not by the family: a family that asked for a tier
//! would be asking about the design's compilation, which is not its
//! business.
//!
//! # THE PANELS COME FROM THE CONTAINER
//!
//! The container stores each projection's weights twice: as the host's fp32
//! filter, and as a bf16 panel already padded to the design's widths and
//! already tiled into the MMAC's byte order. The panel is what the array
//! multiplies by, and it is the SAME numbers as the host's filter rounded to
//! bf16 -- so a run that dispatches to the array is the same network, not a
//! second network.
//!
//! The attention's panels are the exception: they are ACTIVATIONS, and they
//! are built on the host, per head, because every head reads a different
//! slice of the keys.
//!
//! # A DISPATCH CAN BE SPLIT IN TWO
//!
//! The trait's `gemm` is synchronous: pack A, submit, wait, read C, one after
//! another. A caller with INDEPENDENT projections does not have to pay for
//! that. [`ArrayGemms::gemm_submit`] packs and starts a dispatch and returns a
//! ticket; [`ArrayGemms::gemm_complete`] waits and reads it back. Between the
//! two the host is free to work, and a second dispatch can run against the
//! other slot of a ping-pong pair. [`ArrayGemms::gemm_batch`] drives that
//! overlap for a whole set of independent projections, the same way the
//! convolution's chunk pipeline does.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use npum_format::Container;
use npum_model::array::PipelinedGemm;

use crate::array::{ArrayDesign, ArrayError as DesignError};
use crate::attention::Attention;
use crate::gemm::{
    Datapath, DatapathKind, GemmDispatch, GemmScratch, QuantScales, check_fit, complete_gemm,
    gemm_quant_with_scratch, gemm_with_scratch, submit_gemm, submit_gemm_quant,
};
use crate::transport::Transport;

/// The data buffer the weight panels live in.
///
/// The design's data buffers are A, B and C, in that order: the activations,
/// the weights, the result.
const BUFFER_WEIGHTS: usize = 1;

/// Read one container tensor as little-endian f32 values.
///
/// The dequantisation scales are metadata, not weights: they are small, and
/// reading them once at `open` is what keeps them off the dispatch path. A
/// wrong dtype or length is refused here, before any panel is staged.
fn read_f32_tensor(
    file: &Container,
    name: &str,
    container: &Path,
) -> Result<Vec<f32>, DesignError> {
    let entry = file.directory().tensor(name).ok_or_else(|| {
        DesignError::Design(format!(
            "{}: the container has no {name}, which an int8 projection needs to dequantise",
            container.display()
        ))
    })?;
    if entry.dtype != "F32" {
        return Err(DesignError::Design(format!(
            "{}: {name} is {} and not F32 -- the int8 scale must be a float vector",
            container.display(),
            entry.dtype
        )));
    }
    let bytes = file.tensor_bytes(name).map_err(|e| {
        DesignError::Design(format!("{}: cannot read {name}: {e}", container.display()))
    })?;
    if bytes.len() % 4 != 0 {
        return Err(DesignError::Design(format!(
            "{}: {name} is {} bytes, not a whole number of f32 values",
            container.display(),
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// One projection the array will run.
#[derive(Debug, Clone)]
pub struct GemmSpec {
    /// The weight's name in the container -- "layer.0.qkv".
    ///
    /// It is the identity: the panel is the weight's own, and two
    /// projections of the same shape are two different weights.
    pub weight: String,

    /// The op the design's streams name it by -- "qkv", "attn_out",
    /// "ffn_up", "ffn_down".
    pub op: String,

    /// The reduction width.
    pub k: usize,

    /// The output width.
    pub n: usize,
}

/// One staged panel.
struct Panel {
    /// The slot in the weights' buffer.
    slot: usize,
    /// The instruction slots, by the tier's rows, ascending.
    ///
    /// The rows are the tier's own: the stream computes that many rows
    /// whatever the caller has, and the caller's rows are fewer.
    tiers: Vec<(usize, usize)>,
    /// The projection's own reduction width.
    k: usize,
    /// The projection's own output width.
    n: usize,
    /// The int8 scales for this panel, by index in `ArrayGemms::quant`.
    ///
    /// `None` on a bf16 design, whose panel is the whole story. An index rather
    /// than an `Arc` here keeps `ResolvedGemm` `Copy`, which the two-slot
    /// pipeline relies on.
    quant: Option<usize>,
}

/// A projection submitted with [`ArrayGemms::gemm_submit`] and not yet read.
///
/// An index into the backend's outstanding runs, not a pointer or a borrow:
/// the caller can hold it across other calls and hand it back to
/// [`ArrayGemms::gemm_complete`] whenever the host work between the two halves
/// is done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmTicket(pub usize);

/// The key of a validated projection in the resolve cache.
///
/// Everything the design's resolution reads is either fixed after `open` or
/// named here, so a hit needs no re-validation.
type ResolvedKey = (String, usize, usize, usize, usize, usize, usize);

/// A submitted run and the A slot it owns until it is completed.
struct Outstanding {
    dispatch: GemmDispatch,
    a_slot: usize,
}

/// A projection after validation, with the design's own numbers resolved.
#[derive(Debug, Clone, Copy)]
struct ResolvedGemm {
    weight_slot: usize,
    instr: usize,
    tier_rows: usize,
    dp: Datapath,
    rows: usize,
    k: usize,
    n: usize,
    /// The int8 scales for this projection, by index in `ArrayGemms::quant`.
    quant: Option<usize>,
}

/// The array's projections.
pub struct ArrayGemms<T: Transport> {
    design: ArrayDesign<T>,
    /// The panels, by the weight's name.
    panels: HashMap<String, Panel>,
    /// The attention, when the run asked for it.
    ///
    /// `None` is a run that asked for the projections only, and then the
    /// attention is the host's.
    attention: Option<Attention>,
    /// The GEMM's scratch, reused across dispatches.
    ///
    /// A transformer dispatches many projections per layer and per step, and
    /// reusing these buffers avoids an allocation per projection.
    scratch: GemmScratch,
    /// How many projections have been dispatched.
    dispatches: usize,
    /// How many bytes of panels are staged.
    staged: usize,
    /// Runs submitted but not yet read back, by ticket.
    ///
    /// Completed entries are set to `None`; the index is never reused, so an
    /// old ticket cannot accidentally complete a different run.
    outstanding: Vec<Option<Outstanding>>,
    /// Which ping-pong slot pair is held by an in-flight run.
    slot_busy: [bool; 2],
    /// Reused staging for the validated ops of one `gemm_batch`.
    ///
    /// A model repeats the same projection sets every layer and every step;
    /// resolving them into a fresh `Vec` each call would allocate in the
    /// inference loop. The buffer is cleared and refilled in place.
    resolved_buf: Vec<ResolvedGemm>,
    /// Validated projections, keyed by everything that can affect the result.
    ///
    /// The panels and the design are fixed after `open`, so `resolve` is a
    /// pure function of its arguments. A transformer resolves the same few
    /// projections every layer and every step; remembering the first result
    /// takes the validation, tier search and `check_fit` off the steady-state
    /// path. Errors are never cached -- only successful resolutions are.
    resolved_cache: HashMap<ResolvedKey, ResolvedGemm>,
    /// The container's int8 scales, one entry per projection that needs them.
    ///
    /// Read once at `open`, shared with each submitted ticket as an `Arc`, and
    /// never looked up again in the dispatch loop.
    quant: Vec<Arc<QuantScales>>,
}

impl<T: Transport> ArrayGemms<T> {
    /// Open the array's projections for a family.
    ///
    /// The design is passed in and not opened here: the caller has already
    /// decided where the design set comes from -- the container's own copy
    /// or a directory beside it -- and a design set loaded from the wrong
    /// place is a wrong answer rather than an error.
    ///
    /// EVERY PROJECTION IS STAGED HERE, BEFORE THE FIRST RUN. The flag says
    /// the projections move to the array, and a container packed without the
    /// panels has to be refused while there is still nothing to unwind.
    ///
    /// `attention` asks for the attention's streams as well. They are
    /// streams of the SAME design, so asking for them costs no context and
    /// no second design set -- and not asking for them leaves the attention
    /// on the host, which is what a run that did not name `attn` wants.
    pub fn open(
        container: &Path,
        mut design: ArrayDesign<T>,
        specs: &[GemmSpec],
        attention: bool,
    ) -> Result<Self, DesignError> {
        let file = Container::open(container).map_err(|e| {
            DesignError::Design(format!("cannot open {}: {e}", container.display()))
        })?;
        // The design set is fixed for all panels of one backend, so the
        // datapath is decided once here. An int8 set needs the container's
        // per-channel weight scale and SmoothQuant factors for EVERY panel;
        // reading them here means a missing one is refused before the first
        // dispatch rather than at the first projection that uses it.
        let dp = Datapath::of(&design)?;
        let mut quant: Vec<Arc<QuantScales>> = Vec::new();

        let mut panels: HashMap<String, Panel> = HashMap::new();
        let mut staged = 0usize;
        for spec in specs {
            // THE PANEL: the weight itself, already rounded to bf16 and
            // already tiled into the MMAC's byte order.
            let bytes = file.tensor_bytes(&spec.weight).map_err(|e| {
                DesignError::Design(format!(
                    "{}: the container has no {}, which the array multiplies by. A container \
                     packed for the host keeps its weights in fp32 and has no panel; repack \
                     it with the packer's --dtype bf16. ({e})",
                    container.display(),
                    spec.weight
                ))
            })?;

            // THE SHAPE, FROM THE CONTAINER.
            let entry = file.directory().tensor(&spec.weight).ok_or_else(|| {
                DesignError::Design(format!(
                    "{}: the container's directory does not list {}.",
                    container.display(),
                    spec.weight
                ))
            })?;
            let shape: Vec<i64> = entry.logical_shape.clone();
            if shape.len() != 2 || shape[0] as usize != spec.k || shape[1] as usize != spec.n {
                return Err(DesignError::Design(format!(
                    "{}: {} is {} but the projection is {}x{}. The panel and the graph have to \
                     agree on the shape.",
                    container.display(),
                    spec.weight,
                    shape
                        .iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<String>>()
                        .join("x"),
                    spec.k,
                    spec.n
                )));
            }

            // THE TIERS, THE SMALLEST FIRST.
            //
            // The design's streams are named after the op and the batch they
            // were compiled for, and the tier that holds the run's rows is
            // the one to dispatch against.
            let mut tiers: Vec<(usize, usize)> = Vec::new();
            for stream in design.info().streams.iter().filter(|s| s.op == spec.op) {
                if stream.k as usize != spec.k || stream.n as usize != spec.n {
                    continue;
                }
                tiers.push((stream.m as usize, stream.slot as usize));
            }
            tiers.sort_by_key(|(rows, _)| *rows);
            tiers.dedup_by_key(|(rows, _)| *rows);
            if tiers.is_empty() {
                return Err(DesignError::Design(format!(
                    "{}: the design set has no stream for '{}' at {}x{}. The design set was \
                     exported for a different model.",
                    container.display(),
                    spec.op,
                    spec.k,
                    spec.n
                )));
            }

            // THE INT8 SCALES, WHEN THE DESIGN'S DATAPATH NEEDS THEM.
            //
            // `.wscale` is one float per output column and `.asmooth` one per
            // input channel. Both ship with every I8 panel, even when the
            // smoothing was calibrated as all ones, so there is one code path
            // and no container whose smoothing is silently skipped.
            let quant_index = if dp.kind == DatapathKind::Int8I32 {
                let wscale = read_f32_tensor(&file, &format!("{}.wscale", spec.weight), container)?;
                let asmooth =
                    read_f32_tensor(&file, &format!("{}.asmooth", spec.weight), container)?;
                if wscale.len() != spec.n || asmooth.len() != spec.k {
                    return Err(DesignError::Design(format!(
                        "{}: {} has {} weight scales and {} smoothing factors, and the \
                         projection is {}x{}. Int8 scales must follow the panel's shape.",
                        container.display(),
                        spec.weight,
                        wscale.len(),
                        asmooth.len(),
                        spec.k,
                        spec.n
                    )));
                }
                let scales = QuantScales::new(&asmooth, wscale)?;
                let index = quant.len();
                quant.push(Arc::new(scales));
                Some(index)
            } else {
                None
            };

            let slot = design.stage(BUFFER_WEIGHTS, bytes)?;
            staged += bytes.len();
            panels.insert(
                spec.weight.clone(),
                Panel {
                    slot,
                    tiers,
                    k: spec.k,
                    n: spec.n,
                    quant: quant_index,
                },
            );
        }

        if panels.is_empty() && !attention {
            return Err(DesignError::Design(format!(
                "{}: the array was asked for and no projection could be staged.",
                container.display()
            )));
        }

        // THE ATTENTION'S STREAMS.
        //
        // They are streams of THIS design, so they cost no context and no
        // second design set -- and they are found here, while there is still
        // nothing to unwind: a set exported without them is refused by
        // name, and not discovered at the first attention.
        let attention = if attention {
            Some(Attention::open(&mut design)?)
        } else {
            None
        };

        Ok(Self {
            design,
            panels,
            attention,
            scratch: GemmScratch::new(),
            dispatches: 0,
            staged,
            outstanding: Vec::new(),
            slot_busy: [false; 2],
            resolved_buf: Vec::new(),
            resolved_cache: HashMap::new(),
            quant,
        })
    }

    /// The design this backend runs on.
    pub fn design(&self) -> &ArrayDesign<T> {
        &self.design
    }

    /// Validate one projection and resolve the design's own numbers.
    ///
    /// Everything that can be checked without the board is checked here, BEFORE
    /// a dispatch is started: a shape bug in the last op of a batch is still a
    /// shape bug, and must not leave earlier runs in flight. The result holds
    /// only owned, Copy data, so no borrow of `self` survives into the
    /// submission.
    #[allow(clippy::too_many_arguments)]
    fn resolve(
        &mut self,
        weight: &str,
        a_len: usize,
        rows: usize,
        k: usize,
        n: usize,
        bias: Option<&[f32]>,
        out_len: usize,
    ) -> Result<ResolvedGemm, DesignError> {
        // Everything this function reads is fixed after `open` (the panels and
        // the design) or is in the key, so a successful resolution is reusable.
        let bias_len = bias.map_or(0, |b| b.len());
        let key = (weight.to_owned(), a_len, rows, k, n, bias_len, out_len);
        if let Some(hit) = self.resolved_cache.get(&key) {
            return Ok(*hit);
        }
        let panel = self.panels.get(weight).ok_or_else(|| {
            DesignError::Design(format!(
                "the array has no panel for {weight}. The projection was not staged, and a \
                 dispatch against a missing panel is a wrong answer rather than an error."
            ))
        })?;
        if k != panel.k || n != panel.n {
            return Err(DesignError::Design(format!(
                "{weight} is {k}x{n}, and the array's panel for it is {}x{}.",
                panel.k, panel.n
            )));
        }
        if a_len != rows * k {
            return Err(DesignError::Design(format!(
                "the activations hold {a_len} values, and {rows} rows of {k} need {}.",
                rows * k
            )));
        }
        if out_len < rows * n {
            return Err(DesignError::Design(format!(
                "the output holds {out_len} floats, and {rows} rows of {n} need {}.",
                rows * n
            )));
        }
        if let Some(bias) = bias {
            if bias.len() != n {
                return Err(DesignError::Design(format!(
                    "the bias holds {} values, and the projection has {n} output columns.",
                    bias.len()
                )));
            }
        }

        // THE TIER: the smallest one that holds the run's rows.
        let Some((tier_rows, instr)) = panel.tiers.iter().find(|(tier, _)| *tier >= rows).copied()
        else {
            return Err(DesignError::Design(format!(
                "{weight}: the design's smallest tier computes {} rows, and the run has {rows}. \
                 The design set was exported for a different batch.",
                panel.tiers.first().map(|(tier, _)| *tier).unwrap_or(0)
            )));
        };

        let dp = Datapath::of(&self.design)?;
        check_fit(&self.design, tier_rows, k, n, dp)?;
        let resolved = ResolvedGemm {
            weight_slot: panel.slot,
            instr,
            tier_rows,
            dp,
            rows,
            k,
            n,
            quant: panel.quant,
        };
        self.resolved_cache.insert(key, resolved);
        Ok(resolved)
    }

    /// Submit one projection to the array WITHOUT waiting for it.
    ///
    /// The returned ticket is handed to [`Self::gemm_complete`] once the host
    /// work between the two halves is done. While the device computes, the
    /// caller may submit another projection -- the bindings a dispatch
    /// captured at `start_dispatch` are its own, so rebinding does not move an
    /// in-flight run's operands. At most two runs may be in flight, one per
    /// ping-pong slot pair; a third is refused until one is completed.
    pub fn gemm_submit(
        &mut self,
        weight: &str,
        a: &[f32],
        rows: usize,
        k: usize,
        n: usize,
        bias: Option<&[f32]>,
    ) -> Result<GemmTicket, DesignError> {
        let resolved = self.resolve(weight, a.len(), rows, k, n, bias, rows * n)?;
        self.submit_resolved(&resolved, a, bias)
    }

    /// Submit a validated projection into a free ping-pong slot pair.
    fn submit_resolved(
        &mut self,
        resolved: &ResolvedGemm,
        a: &[f32],
        bias: Option<&[f32]>,
    ) -> Result<GemmTicket, DesignError> {
        if self.slot_busy[0] && self.slot_busy[1] {
            return Err(DesignError::Design(
                "both GEMM ping-pong slots are busy -- complete a submitted projection \
                 before submitting a third"
                    .to_string(),
            ));
        }
        let slot = usize::from(self.slot_busy[0]);
        if slot == 1 {
            // A second run needs its own A and C slot pair. Allocate it on the
            // first overlap, so a plain submit/complete pair pays for no extra
            // buffer and a batch does not have to know how slots are managed.
            if self.design.slots_of(0) < 2 {
                self.design.add_slot_uninit(0)?;
            }
            if self.design.slots_of(2) < 2 {
                self.design.add_slot_uninit(2)?;
            }
        }

        // The bindings belong to THIS dispatch: start_dispatch captures them,
        // so rebinding for the next run cannot move this one's operands.
        self.design.bind(BUFFER_WEIGHTS, resolved.weight_slot)?;
        self.design.bind(0, slot)?;
        self.design.bind(2, slot)?;
        let dispatch = if resolved.dp.kind == DatapathKind::Int8I32 {
            let quant = resolved
                .quant
                .and_then(|index| self.quant.get(index))
                .ok_or_else(|| {
                    DesignError::Design(format!(
                        "the int8 projection {} lost its container scales",
                        resolved.weight_slot
                    ))
                })?;
            submit_gemm_quant(
                &mut self.design,
                resolved.instr,
                a,
                resolved.rows,
                resolved.tier_rows,
                resolved.k,
                resolved.dp,
                bias,
                resolved.n,
                Arc::clone(quant),
                &mut self.scratch,
            )?
        } else {
            submit_gemm(
                &mut self.design,
                resolved.instr,
                a,
                resolved.rows,
                resolved.tier_rows,
                resolved.k,
                resolved.dp,
                bias,
                resolved.n,
                &mut self.scratch,
            )?
        };

        self.slot_busy[slot] = true;
        self.dispatches += 1;
        let ticket = GemmTicket(self.outstanding.len());
        self.outstanding.push(Some(Outstanding {
            dispatch,
            a_slot: slot,
        }));
        Ok(ticket)
    }

    /// Wait for a submitted projection, read its output and add the bias.
    ///
    /// Between [`Self::gemm_submit`] and this call the host is free to do any
    /// work that does not depend on this projection's output -- pack another
    /// projection, run an independent LayerNorm, and so on. The read-back goes
    /// to the C slot captured at submit, not to whatever slot is bound now.
    pub fn gemm_complete(
        &mut self,
        ticket: GemmTicket,
        out: &mut [f32],
    ) -> Result<(), DesignError> {
        // Validate before consuming the ticket: a too-short output must not
        // leave a run in flight with its slot mistakenly marked free.
        let needed = self
            .outstanding
            .get(ticket.0)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                DesignError::Design(format!(
                    "unknown or already completed GEMM ticket {}",
                    ticket.0
                ))
            })?;
        let needed = needed.dispatch.n_real * needed.dispatch.n;
        if out.len() < needed {
            return Err(DesignError::Design(format!(
                "the output holds {} floats, and the submitted projection needs {}",
                out.len(),
                needed
            )));
        }

        let entry = self.outstanding[ticket.0]
            .take()
            .expect("the ticket was checked above");
        let result = complete_gemm(&mut self.design, entry.dispatch, out, &mut self.scratch);
        self.slot_busy[entry.a_slot] = false;
        // Leave both buffers on slot 0 so the plain synchronous path stays
        // predictable. Other in-flight runs captured their own bindings.
        self.design.bind(0, 0)?;
        self.design.bind(2, 0)?;
        result
    }

    /// Run a batch of INDEPENDENT projections with the two halves overlapped.
    ///
    /// Each op's output goes to the matching `outs` entry. While op N's GEMM is
    /// on the device, op N+1's A is packed and uploaded; once N is waited on,
    /// its C is read and widened while N+1 is already running. This is the same
    /// two-slot ping-pong that the convolution's chunk pipeline uses. A slot is
    /// never reused until the run that owned it has been waited on and read.
    ///
    /// Ops that depend on one another do NOT belong in one batch: there is no
    /// overlap to gain and the result would be a race on the host buffers.
    pub fn gemm_batch(
        &mut self,
        ops: &[PipelinedGemm<'_>],
        outs: &mut [&mut [f32]],
    ) -> Result<(), DesignError> {
        if ops.len() != outs.len() {
            return Err(DesignError::Design(format!(
                "{} projections were submitted with {} output buffers",
                ops.len(),
                outs.len()
            )));
        }
        if ops.is_empty() {
            return Ok(());
        }

        // Resolve and validate ALL of them first. A bad shape in the last op
        // must be reported before any run is started. The staging buffer is
        // the backend's and is refilled in place: a model repeats its
        // projection sets every layer and every step, and a fresh `Vec` here
        // would be an allocation in the middle of the inference loop.
        self.resolved_buf.clear();
        for (op, out) in ops.iter().zip(outs.iter()) {
            let resolved = self.resolve(
                op.weight,
                op.a.len(),
                op.rows,
                op.k,
                op.n,
                op.bias,
                out.len(),
            )?;
            self.resolved_buf.push(resolved);
        }

        // More than one op needs the ping-pong pairs. A single op has nothing
        // to overlap and pays for no extra slot.
        if ops.len() > 1 {
            if self.design.slots_of(0) < 2 {
                self.design.add_slot_uninit(0)?;
            }
            if self.design.slots_of(2) < 2 {
                self.design.add_slot_uninit(2)?;
            }
        }

        // At most two runs are in flight -- one per ping-pong pair -- so this
        // is a fixed two-slot ring with no heap allocation per batch.
        let mut in_flight: [Option<(GemmTicket, &mut [f32])>; 2] = [None, None];
        let mut head = 0usize;
        let mut live = 0usize;
        for (index, out) in outs.iter_mut().enumerate() {
            // Keep at most two runs in flight. Completing the oldest frees its
            // slot pair for the run about to be submitted; its read-back still
            // overlaps with the younger run's device compute.
            if live == 2 {
                let (ticket, done_out) = in_flight[head].take().expect("two runs are in flight");
                head = (head + 1) & 1;
                live -= 1;
                self.gemm_complete(ticket, done_out)?;
            }
            // `ResolvedGemm` is Copy, so the borrow of the reuse buffer ends
            // here and the submit can borrow the backend mutably.
            let resolved = self.resolved_buf[index];
            let op = &ops[index];
            let ticket = self.submit_resolved(&resolved, op.a, op.bias)?;
            let tail = (head + live) & 1;
            in_flight[tail] = Some((ticket, &mut **out));
            live += 1;
        }
        // EPILOGUE: the last runs have no successors to hide their read-back.
        while live > 0 {
            let (ticket, out) = in_flight[head].take().expect("a run is in flight");
            head = (head + 1) & 1;
            live -= 1;
            self.gemm_complete(ticket, out)?;
        }
        Ok(())
    }
}

impl<T: Transport> npum_model::array::ArrayGemm for ArrayGemms<T> {
    fn name(&self) -> &'static str {
        "xrt"
    }

    fn gemm_on_array(&self) -> bool {
        !self.panels.is_empty()
    }

    fn dispatches(&self) -> usize {
        self.dispatches
    }

    fn staged_bytes(&self) -> usize {
        self.staged
    }

    fn gemm(
        &mut self,
        weight: &str,
        a: &[f32],
        rows: usize,
        k: usize,
        n: usize,
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> npum_model::array::Result<()> {
        let resolved = self
            .resolve(weight, a.len(), rows, k, n, bias, out.len())
            .map_err(|e| npum_model::array::ArrayError(e.to_string()))?;

        // An in-flight async run owns its A/C slot pair even though its
        // bindings were captured at submit. Use a free pair here rather than
        // writing into the slot it is still reading from.
        let slot = usize::from(self.slot_busy[0]);
        if slot == 1 {
            if self.slot_busy[1] {
                return Err(npum_model::array::ArrayError(
                    "both GEMM slots are busy -- complete the submitted projections first"
                        .to_string(),
                ));
            }
            if self.design.slots_of(0) < 2 {
                self.design
                    .add_slot_uninit(0)
                    .map_err(|e| npum_model::array::ArrayError(e.to_string()))?;
            }
            if self.design.slots_of(2) < 2 {
                self.design
                    .add_slot_uninit(2)
                    .map_err(|e| npum_model::array::ArrayError(e.to_string()))?;
            }
        }
        self.design
            .bind(BUFFER_WEIGHTS, resolved.weight_slot)
            .map_err(|e| npum_model::array::ArrayError(e.to_string()))?;
        self.design
            .bind(0, slot)
            .map_err(|e| npum_model::array::ArrayError(e.to_string()))?;
        self.design
            .bind(2, slot)
            .map_err(|e| npum_model::array::ArrayError(e.to_string()))?;
        if resolved.dp.kind == DatapathKind::Int8I32 {
            let quant = resolved
                .quant
                .and_then(|index| self.quant.get(index))
                .ok_or_else(|| {
                    npum_model::array::ArrayError(
                        "the int8 projection lost its container scales".to_string(),
                    )
                })?;
            gemm_quant_with_scratch(
                &mut self.design,
                resolved.instr,
                a,
                rows,
                resolved.tier_rows,
                k,
                bias,
                n,
                out,
                &mut self.scratch,
                quant,
            )
            .map_err(|e| npum_model::array::ArrayError(e.to_string()))?;
        } else {
            gemm_with_scratch(
                &mut self.design,
                resolved.instr,
                a,
                rows,
                resolved.tier_rows,
                k,
                bias,
                n,
                out,
                &mut self.scratch,
            )
            .map_err(|e| npum_model::array::ArrayError(e.to_string()))?;
        }
        self.dispatches += 1;
        Ok(())
    }

    fn gemm_batch(
        &mut self,
        ops: &[PipelinedGemm<'_>],
        outs: &mut [&mut [f32]],
    ) -> npum_model::array::Result<()> {
        // The inherent method is the pipelined one; the trait's default would
        // just run the ops in order. Reaching the seam from a family must not
        // quietly lose the overlap.
        ArrayGemms::gemm_batch(self, ops, outs)
            .map_err(|e| npum_model::array::ArrayError(e.to_string()))
    }
}

impl<T: Transport> npum_model::array::ArrayAttention for ArrayGemms<T> {
    fn name(&self) -> &'static str {
        "xrt"
    }

    fn on_array(&self) -> bool {
        self.attention.is_some()
    }

    fn score_cols(&self) -> Option<usize> {
        self.attention.as_ref().map(|a| a.score_cols())
    }

    fn attn_dispatches(&self) -> usize {
        self.attention.as_ref().map(|a| a.dispatches).unwrap_or(0)
    }

    fn attn_seconds(&self) -> (f64, f64) {
        match &self.attention {
            Some(a) => (a.qk_seconds, a.av_seconds),
            None => (0.0, 0.0),
        }
    }

    fn attention(
        &mut self,
        qkv: &[f32],
        shape: npum_model::array::AttentionShape,
        mask: &[f32],
        softmax: Option<&mut (dyn npum_model::array::ArrayEltwise + '_)>,
        ctx: &mut [f32],
    ) -> npum_model::array::Result<()> {
        let attention = self.attention.as_mut().ok_or_else(|| {
            npum_model::array::ArrayError(
                "this run did not ask for the attention to go on the array".to_string(),
            )
        })?;
        crate::attention::run(&mut self.design, attention, qkv, shape, mask, softmax, ctx)
            .map_err(|e| npum_model::array::ArrayError(e.to_string()))
    }
}

impl<T: Transport> npum_model::array::ArrayText for ArrayGemms<T> {}

#[cfg(test)]
mod int8_tests {
    use super::*;
    use crate::array::ArrayDesign;
    use crate::transport::{BufferFlags, TransportError};
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct Buf {
        bytes: Arc<Mutex<Vec<u8>>>,
    }

    struct Int8Run {
        a: Buf,
        b: Buf,
        c: Buf,
    }

    unsafe extern "C" fn free_run(ptr: *mut c_void) {
        if !ptr.is_null() {
            unsafe { drop(Box::from_raw(ptr as *mut Int8Run)) };
        }
    }

    struct Int8Transport {
        counts: Mutex<HashMap<i32, usize>>,
    }

    impl Transport for Int8Transport {
        type Buffer = Buf;

        fn open(_index: u32) -> Result<Self, TransportError> {
            Ok(Self {
                counts: Mutex::new(HashMap::new()),
            })
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
            *slot += 1;
            Ok(Buf {
                bytes: Arc::new(Mutex::new(vec![0u8; bytes as usize])),
            })
        }
        fn write(bo: &Self::Buffer, data: &[u8], offset: u64) -> Result<(), TransportError> {
            bo.bytes.lock().unwrap()[offset as usize..offset as usize + data.len()]
                .copy_from_slice(data);
            Ok(())
        }
        fn size(bo: &Self::Buffer) -> u64 {
            bo.bytes.lock().unwrap().len() as u64
        }
        fn read(bo: &Self::Buffer, data: &mut [u8], offset: u64) -> Result<(), TransportError> {
            data.copy_from_slice(
                &bo.bytes.lock().unwrap()[offset as usize..offset as usize + data.len()],
            );
            Ok(())
        }
        fn sync_to_device(
            _bo: &Self::Buffer,
            _bytes: u64,
            _offset: u64,
        ) -> Result<(), TransportError> {
            Ok(())
        }
        fn sync_from_device(
            _bo: &Self::Buffer,
            _bytes: u64,
            _offset: u64,
        ) -> Result<(), TransportError> {
            Ok(())
        }
        fn start(
            &mut self,
            _instr: &Self::Buffer,
            _words: u32,
            args: &[&Self::Buffer],
        ) -> Result<crate::transport::Run, TransportError> {
            Ok(crate::transport::Run::new(
                Box::into_raw(Box::new(Int8Run {
                    a: args[0].clone(),
                    b: args[1].clone(),
                    c: args[2].clone(),
                })) as *mut c_void,
                free_run,
            ))
        }
        fn wait(run: &crate::transport::Run) -> Result<(), TransportError> {
            let r = unsafe { &*(run.handle() as *const Int8Run) };
            let a = r.a.bytes.lock().unwrap();
            let b = r.b.bytes.lock().unwrap();
            let mut c = r.c.bytes.lock().unwrap();
            // An exact int8 x int8 -> int32 dot: no rounding in the reduction.
            let k = 4usize;
            let n = 2usize;
            let rows = a.len() / k;
            for row in 0..rows {
                for col in 0..n {
                    let mut acc = 0i32;
                    for kk in 0..k {
                        acc += (a[row * k + kk] as i8 as i32) * (b[kk * n + col] as i8 as i32);
                    }
                    let at = (row * n + col) * 4;
                    c[at..at + 4].copy_from_slice(&acc.to_le_bytes());
                }
            }
            Ok(())
        }
    }

    /// An int8 design, panel and scales: the whole int8 dispatch in one test.
    #[test]
    fn int8_projection_uses_the_container_scales_end_to_end() {
        let dir = std::env::temp_dir().join(format!(
            "npum_int8_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let design = dir.join("design");
        std::fs::create_dir_all(&design).unwrap();
        let manifest = r#"{
            "name": "fake-i8", "kernel": "MLIR_AIE", "arch": 1, "device": "npu1",
            "buffers": [8, 8, 16], "a_dtype": "i8", "c_dtype": "i32",
            "tile": { "m": 2, "k": 4, "n": 2, "rows": 1 },
            "slot0": { "op": "qkv", "batch": 0, "slot": 0, "file": "insts.bin",
                       "M": 2, "K": 4, "N": 2 },
            "streams": [ { "op": "qkv", "batch": 0, "slot": 0, "file": "insts.bin",
                           "M": 2, "K": 4, "N": 2 } ]
        }"#;
        std::fs::write(design.join("design.json"), manifest).unwrap();
        std::fs::write(design.join("final.xclbin"), b"xclbin").unwrap();
        std::fs::write(design.join("insts.bin"), vec![0u8; 4]).unwrap();

        let mut writer = npum_format::Writer::new(
            npum_format::directory::Config::from_value(serde_json::json!({})),
            1,
            0,
        );
        let panel = vec![1u8; 8]; // all weights quantise to 1
        writer.add(npum_format::writer::Staged {
            name: "w".into(),
            role: "gemm_b".into(),
            dtype: "I8".into(),
            logical_shape: vec![4, 2],
            padded_shape: vec![4, 2],
            blob: panel,
            layout: None,
        });
        writer.add(npum_format::writer::Staged {
            name: "w.wscale".into(),
            role: "quant_scale".into(),
            dtype: "F32".into(),
            logical_shape: vec![2],
            padded_shape: vec![2],
            blob: [2.0f32, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
            layout: None,
        });
        writer.add(npum_format::writer::Staged {
            name: "w.asmooth".into(),
            role: "quant_smooth".into(),
            dtype: "F32".into(),
            logical_shape: vec![4],
            padded_shape: vec![4],
            blob: [1.0f32, 2.0, 4.0, 8.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            layout: None,
        });
        let container = dir.join("model.npum");
        writer.write(&container).unwrap();

        let design = ArrayDesign::open(
            &design,
            Int8Transport {
                counts: Mutex::new(HashMap::new()),
            },
        )
        .unwrap();
        let mut gemms = ArrayGemms::open(
            &container,
            design,
            &[GemmSpec {
                weight: "w".into(),
                op: "qkv".into(),
                k: 4,
                n: 2,
            }],
            false,
        )
        .unwrap();

        let a = [1.0f32, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let bias = [1.0f32, 2.0];
        let mut out = [0.0f32; 4];
        npum_model::array::ArrayGemm::gemm(&mut gemms, "w", &a, 2, 4, 2, Some(&bias), &mut out)
            .unwrap();

        // Row 0 quantises to [127,127,95,64], so the exact dot is 413; the
        // container's wscale is [2,4], and the row scale is 1/127.
        let sa = 1.0f32 / 127.0;
        let expected = [413.0 * sa * 2.0 + 1.0, 413.0 * sa * 4.0 + 2.0, 1.0, 2.0];
        for (got, want) in out.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }
}

#[cfg(test)]
mod async_tests {
    use super::*;
    use crate::array::ArrayDesign;
    use crate::transport::{BufferFlags, TransportError};
    use std::collections::HashMap;
    use std::ffi::c_void;
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
        ) -> Result<crate::transport::Run, TransportError> {
            self.log.lock().unwrap().push("start".to_string());
            let run = Box::new(FakeRun {
                a: args[0].clone(),
                c: args[2].clone(),
                log: self.log.clone(),
            });
            Ok(crate::transport::Run::new(
                Box::into_raw(run) as *mut c_void,
                free_fake_run,
            ))
        }
        fn wait(run: &crate::transport::Run) -> Result<(), TransportError> {
            let r = unsafe { &*(run.handle() as *const FakeRun) };
            r.log.lock().unwrap().push("wait".to_string());
            // Simulated GEMM: every C element becomes this dispatch's A marker.
            // The A slot is captured at start, so a later rebind cannot change
            // what this run read -- a slot mix-up is a wrong number, not a race.
            let a = r.a.bytes.lock().unwrap();
            let marker = [a[0], a[1]];
            let mut c = r.c.bytes.lock().unwrap();
            for cell in c.chunks_exact_mut(2) {
                cell.copy_from_slice(&marker);
            }
            Ok(())
        }
    }

    /// A design with one stream, and an `ArrayGemms` whose single panel is
    /// staged in slot 0. Built directly so the test needs no container.
    fn backend(rows: usize, k: usize, n: usize) -> (ArrayGemms<FakeTransport>, Log) {
        let dir = std::env::temp_dir().join(format!(
            "npum_gemm_async_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = format!(
            r#"{{
                "name": "fake",
                "kernel": "MLIR_AIE",
                "arch": 1,
                "device": "npu1",
                "buffers": [{}, 8, {}],
                "a_dtype": "bf16",
                "c_dtype": "bf16",
                "tile": {{ "m": {rows}, "k": {k}, "n": {n}, "rows": 1 }},
                "slot0": {{ "op": "qkv", "batch": 0, "slot": 0, "file": "insts.bin",
                            "M": {rows}, "K": {k}, "N": {n} }}
            }}"#,
            rows * k * 2,
            rows * n * 2
        );
        std::fs::write(dir.join("design.json"), manifest).unwrap();
        std::fs::write(dir.join("final.xclbin"), b"xclbin").unwrap();
        std::fs::write(dir.join("insts.bin"), vec![0u8; 4]).unwrap();

        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let design = ArrayDesign::open(&dir, FakeTransport::new(log.clone())).unwrap();
        let mut panels = HashMap::new();
        panels.insert(
            "w".to_string(),
            Panel {
                slot: 0,
                tiers: vec![(rows, 0)],
                k,
                n,
                quant: None,
            },
        );
        (
            ArrayGemms {
                design,
                panels,
                attention: None,
                scratch: GemmScratch::new(),
                dispatches: 0,
                staged: 0,
                outstanding: Vec::new(),
                slot_busy: [false; 2],
                resolved_buf: Vec::new(),
                resolved_cache: HashMap::new(),
                quant: Vec::new(),
            },
            log,
        )
    }

    /// bf16-exact marker values, one per op: each dispatch's output identifies
    /// the A slot it actually read.
    fn activation(marker: f32, rows: usize, k: usize) -> Vec<f32> {
        let mut a = vec![0f32; rows * k];
        a[0] = marker;
        a
    }

    /// Three independent projections through the pipelined batch. Checks both
    /// the numbers and the ORDER of operations: the next op's A is written
    /// BEFORE the wait for the current one, and the current one's C is read
    /// AFTER the next dispatch has started. That order IS the overlap.
    #[test]
    fn batch_ping_pongs_slots_and_overlaps_stages() {
        let (rows, k, n) = (2usize, 4usize, 3usize);
        let (mut gemms, log) = backend(rows, k, n);
        let a0 = activation(10.0, rows, k);
        let a1 = activation(11.0, rows, k);
        let a2 = activation(12.0, rows, k);
        let bias = [1.0f32, 2.0, 3.0];
        let mut o0 = vec![0f32; rows * n];
        let mut o1 = vec![0f32; rows * n];
        let mut o2 = vec![0f32; rows * n];
        let ops = [
            PipelinedGemm {
                weight: "w",
                a: &a0,
                rows,
                k,
                n,
                bias: Some(&bias),
            },
            PipelinedGemm {
                weight: "w",
                a: &a1,
                rows,
                k,
                n,
                bias: Some(&bias),
            },
            PipelinedGemm {
                weight: "w",
                a: &a2,
                rows,
                k,
                n,
                bias: Some(&bias),
            },
        ];
        let mut outs: [&mut [f32]; 3] = [&mut o0, &mut o1, &mut o2];
        gemms.gemm_batch(&ops, &mut outs).unwrap();

        for (out, marker) in [(&o0, 10.0), (&o1, 11.0), (&o2, 12.0)] {
            for r in 0..rows {
                for j in 0..n {
                    assert_eq!(
                        out[r * n + j],
                        marker + bias[j],
                        "row {r} column {j} came from the wrong A slot"
                    );
                }
            }
        }

        let log = log.lock().unwrap();
        // The design's load zeroes its instruction and data buffers first.
        // The pipeline starts at the first dispatch.
        let first_start = log.iter().position(|e| e == "start").unwrap();
        let expected = [
            "start",
            "write(3:1)",
            "sync_to_device(3:1)",
            "start",
            "wait",
            "sync_from_device(5:0)",
            "read(5:0)",
            "write(3:0)",
            "sync_to_device(3:0)",
            "start",
            "wait",
            "sync_from_device(5:1)",
            "read(5:1)",
            "wait",
            "sync_from_device(5:0)",
            "read(5:0)",
        ];
        assert_eq!(
            &log[first_start..],
            &expected[..],
            "the pipeline stages did not overlap"
        );
    }

    /// The second ping-pong pair is allocated WITHOUT a zero-fill: the A half
    /// is written in full by the submit, and the C half is read back only over
    /// the rows the device wrote. Zeroing either would be work nobody reads.
    #[test]
    fn pipelined_slots_are_not_zero_filled() {
        let (rows, k, n) = (2usize, 4usize, 3usize);
        let (mut gemms, log) = backend(rows, k, n);
        let a = activation(10.0, rows, k);

        // The first submit reuses the design's own slot pair; the SECOND one
        // allocates the ping-pong pair, which is where the zero-fill used to
        // be paid.
        let t0 = gemms.gemm_submit("w", &a, rows, k, n, None).unwrap();
        let t1 = gemms.gemm_submit("w", &a, rows, k, n, None).unwrap();
        let mut out0 = vec![0f32; rows * n];
        let mut out1 = vec![0f32; rows * n];
        gemms.gemm_complete(t0, &mut out0).unwrap();
        gemms.gemm_complete(t1, &mut out1).unwrap();

        let log = log.lock().unwrap();
        let starts: Vec<usize> = log
            .iter()
            .enumerate()
            .filter_map(|(i, e)| (e == "start").then_some(i))
            .collect();
        let before_second = &log[..starts[1]];
        assert_eq!(
            before_second.iter().filter(|e| **e == "write(3:1)").count(),
            1,
            "the second A slot is written once, by the submit, not zeroed first: {before_second:?}"
        );
        assert!(
            !before_second
                .iter()
                .any(|e| e == "write(5:1)" || e == "sync_to_device(5:1)"),
            "the second C slot must not be zeroed or synced before the device writes it: {before_second:?}"
        );
    }

    /// The pipelined batch is reachable through the family's trait seam: a
    /// family holding `&mut dyn ArrayText` gets the overlap, not the trait's
    /// sequential default.
    #[test]
    fn trait_batch_delegates_to_the_pipeline() {
        let (rows, k, n) = (2usize, 4usize, 3usize);
        let (mut gemms, log) = backend(rows, k, n);
        let a0 = activation(10.0, rows, k);
        let a1 = activation(11.0, rows, k);
        let bias = [1.0f32, 2.0, 3.0];
        let ops = [
            PipelinedGemm {
                weight: "w",
                a: &a0,
                rows,
                k,
                n,
                bias: Some(&bias),
            },
            PipelinedGemm {
                weight: "w",
                a: &a1,
                rows,
                k,
                n,
                bias: Some(&bias),
            },
        ];
        let mut o0 = vec![0f32; rows * n];
        let mut o1 = vec![0f32; rows * n];
        let mut outs: [&mut [f32]; 2] = [&mut o0, &mut o1];

        npum_model::array::ArrayGemm::gemm_batch(&mut gemms, &ops, &mut outs).unwrap();

        assert_eq!(o0[0], 10.0 + bias[0]);
        assert_eq!(o1[0], 11.0 + bias[0]);
        assert_eq!(gemms.dispatches, 2);

        // Overlap's signature: op 2's A is uploaded before op 1 is waited on.
        let log = log.lock().unwrap();
        let first_wait = log.iter().position(|e| e == "wait").unwrap();
        assert!(
            log[..first_wait].iter().any(|e| e == "write(3:1)"),
            "the trait path ran the ops in order instead of overlapping them: {log:?}"
        );
    }

    /// Repeated biased submissions reuse ONE bias buffer: the async ticket
    /// must own its bias while the device works, but it need not allocate a
    /// fresh `Vec` for every dispatch of a long inference loop.
    #[test]
    fn bias_buffer_is_reused_across_submissions() {
        let (rows, k, n) = (2usize, 4usize, 3usize);
        let (mut gemms, _log) = backend(rows, k, n);
        let bias = [1.0f32, 2.0, 3.0];
        let a = activation(10.0, rows, k);
        let before = gemms.scratch.bias_pool_len();

        for _ in 0..8 {
            let ticket = gemms.gemm_submit("w", &a, rows, k, n, Some(&bias)).unwrap();
            let mut out = vec![0f32; rows * n];
            gemms.gemm_complete(ticket, &mut out).unwrap();
            assert_eq!(out[0], 10.0 + bias[0]);
        }

        assert_eq!(
            gemms.scratch.bias_pool_len(),
            before + 1,
            "the bias pool must reuse its buffer, not grow per dispatch"
        );
    }

    /// The submit/complete pair leaves the host free between the halves, and
    /// a ticket cannot be completed twice.
    #[test]
    fn submit_complete_tickets_are_one_shot() {
        let (rows, k, n) = (2usize, 4usize, 3usize);
        let (mut gemms, _log) = backend(rows, k, n);
        let a0 = activation(10.0, rows, k);
        let a1 = activation(11.0, rows, k);
        let bias = [1.0f32, 2.0, 3.0];

        let t0 = gemms
            .gemm_submit("w", &a0, rows, k, n, Some(&bias))
            .unwrap();
        // Host work between the two halves has no effect on the outcome.
        let _host_side = 1 + 1;
        let t1 = gemms
            .gemm_submit("w", &a1, rows, k, n, Some(&bias))
            .unwrap();

        let mut o0 = vec![0f32; rows * n];
        let mut o1 = vec![0f32; rows * n];
        gemms.gemm_complete(t0, &mut o0).unwrap();
        gemms.gemm_complete(t1, &mut o1).unwrap();
        assert_eq!(o0[0], 10.0 + bias[0]);
        assert_eq!(o1[0], 11.0 + bias[0]);

        assert!(gemms.gemm_complete(t0, &mut o0).is_err());
        assert_eq!(gemms.dispatches, 2);
    }
}
