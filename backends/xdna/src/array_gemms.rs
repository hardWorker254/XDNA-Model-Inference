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

use std::collections::HashMap;
use std::path::Path;

use npum_format::Container;

use crate::array::{ArrayDesign, ArrayError as DesignError};
use crate::attention::Attention;
use crate::gemm::gemm;
use crate::transport::Transport;

/// The data buffer the weight panels live in.
///
/// The design's data buffers are A, B and C, in that order: the activations,
/// the weights, the result.
const BUFFER_WEIGHTS: usize = 1;

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
    /// How many projections have been dispatched.
    dispatches: usize,
    /// How many bytes of panels are staged.
    staged: usize,
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

            let slot = design.stage(BUFFER_WEIGHTS, bytes)?;
            staged += bytes.len();
            panels.insert(
                spec.weight.clone(),
                Panel {
                    slot,
                    tiers,
                    k: spec.k,
                    n: spec.n,
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
            dispatches: 0,
            staged,
        })
    }

    /// The design this backend runs on.
    pub fn design(&self) -> &ArrayDesign<T> {
        &self.design
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
        let panel = self.panels.get(weight).ok_or_else(|| {
            npum_model::array::ArrayError(format!(
                "the array has no panel for {weight}. The projection was not staged, and a \
                 dispatch against a missing panel is a wrong answer rather than an error."
            ))
        })?;
        if k != panel.k || n != panel.n {
            return Err(npum_model::array::ArrayError(format!(
                "{weight} is {k}x{n}, and the array's panel for it is {}x{}.",
                panel.k, panel.n
            )));
        }
        if a.len() != rows * k {
            return Err(npum_model::array::ArrayError(format!(
                "the activations hold {} values, and {rows} rows of {k} need {}.",
                a.len(),
                rows * k
            )));
        }
        if out.len() < rows * n {
            return Err(npum_model::array::ArrayError(format!(
                "the output holds {} floats, and {rows} rows of {n} need {}.",
                out.len(),
                rows * n
            )));
        }
        if let Some(bias) = bias {
            if bias.len() != n {
                return Err(npum_model::array::ArrayError(format!(
                    "the bias holds {} values, and the projection has {n} output columns.",
                    bias.len()
                )));
            }
        }

        // THE TIER: the smallest one that holds the run's rows.
        let Some((tier_rows, instr)) = panel.tiers.iter().find(|(tier, _)| *tier >= rows).copied()
        else {
            return Err(npum_model::array::ArrayError(format!(
                "{weight}: the design's smallest tier computes {} rows, and the run has {rows}. \
                 The design set was exported for a different batch.",
                panel.tiers.first().map(|(tier, _)| *tier).unwrap_or(0)
            )));
        };

        self.design
            .bind(BUFFER_WEIGHTS, panel.slot)
            .map_err(|e| npum_model::array::ArrayError(e.to_string()))?;
        gemm(&mut self.design, instr, a, rows, tier_rows, k, bias, n, out)
            .map_err(|e| npum_model::array::ArrayError(e.to_string()))?;
        self.dispatches += 1;
        Ok(())
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
