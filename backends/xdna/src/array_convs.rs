//! The array's convolutions, for a family that dispatches its dense
//! convolutions to the board.
//!
//! # WHAT THIS IS
//!
//! The seam's other end. A family -- the OCR family, the pose families --
//! has a graph of dense convolutions, and this is the object that runs them
//! on the array: the design, the panels, and the dispatch.
//!
//! # ONE PANEL PER CONVOLUTION, NEVER PER STREAM
//!
//! This is the rule that matters most in this file. The design's streams
//! are named after the SHAPE they compute -- "conv64x128" -- and a family's
//! convolutions collapse onto a handful of shapes: the OCR family's 81
//! dense convolutions are 16 different padded shapes. A panel table keyed by
//! the stream name would hand the first convolution's panel to the other 39
//! that share its shape, and nothing would report it: the panel would be the
//! right size, the layout would match, and the products would be
//! arithmetic.
//!
//! So the panels are keyed by the CONVOLUTION, and the stream is only ever
//! the shape the convolution is dispatched against.
//!
//! # THE PANELS COME FROM THE CONTAINER
//!
//! The container stores each dense convolution's weights twice: as the host's
//! fp32 filter, and as a bf16 panel already padded to the design's widths
//! and already tiled into the MMAC's byte order. The panel is what the array
//! multiplies by, and it is the SAME numbers as the host's filter rounded to
//! bf16 -- so a run that dispatches to the array is the same network, not a
//! second network.
//!
//! # THE BIAS TRAVELS WITH THE PANEL
//!
//! The bias is the convolution's own, padded with zeros to the design's
//! width. The tail is zero, which is exact: those columns are channels the
//! tensor does not have, and the host narrows the result back to the real
//! width on the way out.

use std::collections::HashMap;
use std::path::Path;

use npum_format::Container;
use npum_model::array::ConvSpec;

use crate::array::{ArrayDesign, ArrayError as DesignError};
use crate::conv::{ConvScratch, ConvShape, conv};
use crate::transport::Transport;

/// The data buffer the weight panels live in.
///
/// The design's data buffers are A, B and C, in that order: the activations,
/// the weights, the result.
const BUFFER_WEIGHTS: usize = 1;

/// One staged panel.
struct Panel {
    /// The slot in the weights' buffer.
    slot: usize,
    /// The instruction slot.
    instr: usize,
    /// The design's padded reduction width.
    k: usize,
    /// The design's padded output width.
    n: usize,
    /// The rows one dispatch computes.
    rows: usize,
    /// The convolution's own bias.
    bias: Option<Vec<f32>>,
}

/// The array's convolutions.
pub struct ArrayConvs<T: Transport> {
    design: ArrayDesign<T>,
    /// The panels, by the weight's name.
    panels: HashMap<String, Panel>,
    /// How many convolutions have been dispatched.
    dispatches: usize,
    /// How many bytes of panels are staged.
    staged: usize,
    /// The dispatch's scratch, reused across the convolutions.
    scratch: ConvScratch,
}

impl<T: Transport> ArrayConvs<T> {
    /// Open the array's convolutions for a family.
    ///
    /// The design is passed in and not opened here: the caller has already
    /// decided where the design set comes from -- the container's own copy
    /// or a directory beside it -- and a design set loaded from the wrong
    /// place is a wrong answer rather than an error.
    ///
    /// EVERY CONVOLUTION IS STAGED HERE, BEFORE THE FIRST FRAME. The flag
    /// says the convolutions move to the array, and a container packed
    /// without the panels has to be refused while there is still nothing
    /// to unwind -- not half way through the first frame, with the boxes
    /// already drawn.
    pub fn open(
        container: &Path,
        mut design: ArrayDesign<T>,
        convs: &[ConvSpec],
    ) -> Result<Self, DesignError> {
        let file = Container::open(container).map_err(|e| {
            DesignError::Design(format!("cannot open {}: {e}", container.display()))
        })?;

        let mut panels: HashMap<String, Panel> = HashMap::new();
        let mut staged = 0usize;
        for spec in convs {
            let panel_name = panel_name(&spec.weight)?;
            let bytes = file.tensor_bytes(&panel_name).map_err(|e| {
                DesignError::Design(format!(
                    "{}: the container has no {panel_name}, which the graph's stream '{}' \
                     needs. Re-pack the container with the packer's --npu so the pre-tiled \
                     panel is in it. ({e})",
                    container.display(),
                    spec.stream
                ))
            })?;

            // THE STREAM, BY NAME.
            //
            // The name is the shape, and the shape is what the convolution
            // is dispatched against: a design set exported for another model
            // has streams of other names, and a convolution dispatched
            // against a missing slot is a wrong answer rather than an error.
            let stream = design
                .info()
                .streams
                .iter()
                .find(|s| s.op == spec.stream)
                .ok_or_else(|| {
                    DesignError::Design(format!(
                        "{}: the design set has no stream named '{}'. The design set was \
                         exported for a different model, or the container names a stream its \
                         design does not carry.",
                        container.display(),
                        spec.stream
                    ))
                })?;
            let (instr, rows, padded_k, padded_n) = (
                stream.slot as usize,
                stream.m as usize,
                stream.k as usize,
                stream.n as usize,
            );
            if rows == 0 || padded_k == 0 || padded_n == 0 {
                return Err(DesignError::Design(format!(
                    "{}: the design's stream '{}' reports no shape (M {rows}, K {padded_k}, \
                     N {padded_n}).",
                    container.display(),
                    spec.stream
                )));
            }

            // THE WIDTHS, CROSS-CHECKED.
            //
            // The real widths come from the convolution, and the padded ones
            // from the design, and the two have to hold: a design whose K is
            // smaller than the convolution's would truncate the panel, and
            // the dispatch would read past its end and return plausible
            // products.
            if padded_k < spec.k || padded_n < spec.n {
                return Err(DesignError::Design(format!(
                    "{}: {} needs {}x{} but the design's stream '{}' is {}x{}. The design set \
                     was exported for a different model.",
                    container.display(),
                    spec.weight,
                    spec.k,
                    spec.n,
                    spec.stream,
                    padded_k,
                    padded_n
                )));
            }

            // THE PANEL'S OWN SHAPE, AGAINST THE STREAM'S.
            //
            // The panel is the convolution's weights padded to the design's
            // widths, and the stream is the shape the design computes. A
            // panel of a different shape would be read past its end, so the
            // two are compared rather than trusted.
            let entry = file.directory().tensor(&panel_name).ok_or_else(|| {
                DesignError::Design(format!(
                    "{}: the container's directory does not list {panel_name}.",
                    container.display()
                ))
            })?;
            let shape: Vec<i64> = entry.logical_shape.clone();
            if shape.len() != 2 || shape[0] as usize != padded_k || shape[1] as usize != padded_n {
                return Err(DesignError::Design(format!(
                    "{}: {panel_name} is {} but the design's stream '{}' computes {}x{}. The \
                     panel and the design have to agree on the shape.",
                    container.display(),
                    shape
                        .iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<String>>()
                        .join("x"),
                    spec.stream,
                    padded_k,
                    padded_n
                )));
            }

            // THE BIAS, CHECKED AGAINST THE CONVOLUTION.
            if let Some(bias) = &spec.bias {
                if bias.len() != spec.n {
                    return Err(DesignError::Design(format!(
                        "{}: the bias of {} holds {} values and the convolution has {} output \
                         channels.",
                        container.display(),
                        spec.weight,
                        bias.len(),
                        spec.n
                    )));
                }
            }

            let slot = design.stage(BUFFER_WEIGHTS, bytes)?;
            staged += bytes.len();
            panels.insert(
                spec.weight.clone(),
                Panel {
                    slot,
                    instr,
                    k: padded_k,
                    n: padded_n,
                    rows,
                    bias: spec.bias.clone(),
                },
            );
        }

        if panels.is_empty() {
            return Err(DesignError::Design(format!(
                "{}: the array was asked for and no convolution could be staged. A design set \
                 with no panels behind it is not a design set.",
                container.display()
            )));
        }

        Ok(Self {
            design,
            panels,
            dispatches: 0,
            staged,
            scratch: ConvScratch::default(),
        })
    }

    /// The design this backend runs on.
    pub fn design(&self) -> &ArrayDesign<T> {
        &self.design
    }
}

/// The panel's name, from the weight's name.
///
/// The packer writes the panel beside the weight it belongs to, with
/// `.btile` where the weight has `.w`. The rule is the packer's, and it is
/// written down here rather than spread over the callers: a panel that is
/// looked for under the wrong name is a container that "has no panel", and
/// the message would blame the container for the runtime's spelling.
fn panel_name(weight: &str) -> Result<String, DesignError> {
    match weight.strip_suffix(".w") {
        Some(base) => Ok(format!("{base}.btile")),
        None => Err(DesignError::Design(format!(
            "the weight's name '{weight}' does not end in '.w', so the panel's name cannot be \
             derived from it."
        ))),
    }
}

impl<T: Transport> npum_model::ArrayConvs for ArrayConvs<T> {
    fn name(&self) -> &'static str {
        "xrt"
    }

    fn dispatches(&self) -> usize {
        self.dispatches
    }

    fn staged_bytes(&self) -> usize {
        self.staged
    }

    fn convert_seconds(&self) -> f64 {
        self.scratch.convert_seconds
    }

    fn rows_per_dispatch(&self) -> usize {
        // THE DESIGN'S OWN ROWS, and the largest of them.
        //
        // Every stream of a conv-only design set computes the same number of
        // rows -- it is the design's single dispatch chunk -- so the first
        // stream's M is the design's M. A design with no stream has no rows,
        // and zero is the honest answer.
        self.design
            .info()
            .streams
            .iter()
            .map(|s| s.m as usize)
            .max()
            .unwrap_or(0)
    }

    fn stream_count(&self) -> usize {
        self.design.info().streams.len()
    }

    fn conv(
        &mut self,
        weight: &str,
        a: &[f32],
        rows: usize,
        k: usize,
        n: usize,
        out: &mut [f32],
    ) -> npum_model::array::Result<()> {
        let panel = self.panels.get(weight).ok_or_else(|| {
            npum_model::array::ArrayError(format!(
                "the array has no panel for {weight}. The convolution was not staged, and a \
                 dispatch against a missing panel is a wrong answer rather than an error."
            ))
        })?;
        if k > panel.k || n > panel.n {
            return Err(npum_model::array::ArrayError(format!(
                "{weight} is {}x{}, and the array's design for it is {}x{}.",
                k, n, panel.k, panel.n
            )));
        }
        if a.len() != rows * k {
            return Err(npum_model::array::ArrayError(format!(
                "the im2col holds {} values, and {rows} rows of {k} need {}.",
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

        let shape = ConvShape {
            rows: panel.rows,
            k,
            n,
            padded_k: panel.k,
            padded_n: panel.n,
        };
        self.design
            .bind(BUFFER_WEIGHTS, panel.slot)
            .map_err(|e| npum_model::array::ArrayError(e.to_string()))?;
        let dispatched = conv(
            &mut self.design,
            panel.instr,
            &shape,
            panel.bias.as_deref(),
            a,
            out,
            &mut self.scratch,
        )
        .map_err(|e| npum_model::array::ArrayError(e.to_string()))?;
        self.dispatches += dispatched;
        Ok(())
    }
}
