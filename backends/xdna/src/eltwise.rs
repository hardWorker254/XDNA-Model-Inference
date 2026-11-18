//! The array's elementwise passes: a LayerNorm, a softmax and a GELU.
//!
//! # WHAT THEY ARE
//!
//! Three designs, three shapes, one mechanism: the array computes a block
//! of rows, the host fills it, the device runs the core, the host reads the
//! block back. The C++ runtime called the object `NpuEltwise` and it is the
//! same thing here.
//!
//! # THREE DESIGNS AND THREE CONTEXTS
//!
//! A LayerNorm is its own program, a softmax is its own program, and a
//! GELU is its own program. They do not share a context, they do not share
//! a design set, and they do not share a buffer budget -- so they are
//! three handles here and not one handle with a mode.
//!
//! # THE THREE DESIGNS DISAGREE ABOUT WHAT A ROW IS
//!
//! And that is why the work is in ELEMENTS and not in rows:
//!
//! * a LayerNorm's row is the hidden dimension wide -- 384 for this
//!   family;
//! * a softmax's row is a score row -- the sequence length wide;
//! * a GELU's "row" is the WHOLE flat span of activations, and its row
//!   capacity is one.
//!
//! So each pass carries its own `cols` and its own row capacity, and the
//! pass that fills the buffer does not care which it is: the capacity is
//! `rows * cols` either way.
//!
//! # THE TAIL IS ZEROED, NOT STALE
//!
//! The buffer is the design's own, and it is filled WHOLE on every call,
//! so a row still holding the previous call's activations would be read as
//! this call's data. The result of such a row is discarded -- but it has to
//! be a number, and a LayerNorm's tail would be `beta`.

use crate::array::{ArrayDesign, ArrayError};
use crate::transport::Transport;
use npum_model::array::Result as SeamResult;

/// The data buffer the LayerNorm's parameters live in.
///
/// The design's data buffers are A, the parameters and C, in that order:
/// the activations, the gamma and beta, the result.
const BUFFER_PARAMS: usize = 1;

/// Which pass a design computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// The LayerNorm: the row's mean and variance, then the gain and the
    /// offset.
    LayerNorm,
    /// The softmax: the row's maximum, then the exponentials, then their
    /// sum.
    Softmax,
    /// The GELU: the activation, element by element.
    Gelu,
    /// The RMSNorm: root mean square normalisation with a weight (no mean/beta).
    RmsNorm,
}

impl Kind {
    /// The kind the manifest records.
    ///
    /// The GELU's design says `eltwise` -- it is the pass that has no
    /// shape of its own -- and the name is what tells it apart from a
    /// design that computes something else entirely.
    fn of(name: &str) -> Option<Kind> {
        match name {
            "layernorm" => Some(Kind::LayerNorm),
            "softmax" => Some(Kind::Softmax),
            "eltwise" | "gelu" => Some(Kind::Gelu),
            "rmsnorm" => Some(Kind::RmsNorm),
            _ => None,
        }
    }

    /// The kind's own name, for the messages.
    fn label(self) -> &'static str {
        match self {
            Kind::LayerNorm => "a LayerNorm",
            Kind::Softmax => "a softmax",
            Kind::Gelu => "a GELU",
            Kind::RmsNorm => "an RMSNorm",
        }
    }
}

/// One elementwise design, loaded onto the board.
///
/// A design is a program and a set of buffers, and it computes ONE pass:
/// a LayerNorm design cannot compute a softmax, and the check is here
/// rather than in the family, because the family is the one that would get
/// it wrong.
pub struct EltwiseDesign<T: Transport> {
    design: ArrayDesign<T>,
    kind: Kind,
    /// The row width.
    cols: usize,
    /// How many rows one dispatch covers.
    rows: usize,
    /// How many elements one dispatch covers: `rows * cols`.
    capacity: usize,
    /// How many dispatches have run.
    dispatches: usize,
    /// THE LAYERNORM'S PARAMETERS, STAGED ONCE PER SITE.
    ///
    /// A LayerNorm has one pair of parameters per site -- the embedding
    /// and two per layer -- and a site is a different pair every time.
    /// The parameters are STAGED, not written into the design's own
    /// buffer: a staged slot is uploaded once and bound, and the design
    /// reads them out of buffer 1 whichever slot that buffer is bound
    /// to. Writing them per call would re-upload them per call, and the
    /// status block's line says they are staged once.
    staged_params: Vec<(Vec<f32>, usize)>,
}

impl<T: Transport> EltwiseDesign<T> {
    /// Open an elementwise design.
    ///
    /// The design's own manifest says what it computes and how wide its
    /// rows are, and both are checked here: a design whose kind is not one
    /// of the supported eltwise kinds is refused, and so is one with no
    /// width -- because a pass with no width would read the buffer as one
    /// long row.
    pub fn open(design: ArrayDesign<T>) -> Result<Self, ArrayError> {
        let info = design.info();
        let kind = Kind::of(&info.kind).or_else(|| Kind::of(&info.name)).ok_or_else(|| {
            ArrayError::Design(format!(
                "{}: this design computes '{}', which is not a LayerNorm, an RMSNorm, a softmax or a GELU.",
                info.name, info.kind
            ))
        })?;
        let cols = info.cols as usize;
        let rows = info.row_capacity as usize;
        if cols == 0 || rows == 0 {
            return Err(ArrayError::Design(format!(
                "{}: the design records no row width or row capacity (cols {}, row_capacity {}).",
                info.name, cols, rows
            )));
        }
        let capacity = rows * cols;
        // THE BUFFER, CHECKED AGAINST THE DESIGN'S OWN.
        //
        // The design was exported for a shape; a shape that does not fit is
        // not a shape this runtime can serve, and saying so is better than
        // a dispatch that overruns a buffer.
        let buffers = &info.buffers;
        if buffers.len() < 2
            || buffers[0] < capacity as u64 * 2
            || buffers[buffers.len() - 1] < capacity as u64 * 2
        {
            return Err(ArrayError::Design(format!(
                "{}: {} rows of {} do not fit the design's own buffers ({:?}).",
                info.name, rows, cols, buffers
            )));
        }
        Ok(Self {
            design,
            kind,
            cols,
            rows,
            capacity,
            dispatches: 0,
            staged_params: Vec::new(),
        })
    }

    /// Which pass this design computes.
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// The row width.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// How many rows one dispatch covers.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// How many dispatches have run.
    pub fn dispatches(&self) -> usize {
        self.dispatches
    }

    /// The LayerNorm, over `n_rows` rows.
    ///
    /// `params` is the gain and the offset, f32 and `2 * cols` values
    /// long -- gamma first, then beta, which is the order the design's
    /// parameter buffer reads them in.
    pub fn layernorm(
        &mut self,
        params: &[f32],
        x: &mut [f32],
        n_rows: usize,
    ) -> Result<(), ArrayError> {
        self.require(Kind::LayerNorm)?;
        if params.len() != 2 * self.cols {
            return Err(ArrayError::Design(format!(
                "{}: gamma|beta is {} floats, this design's row is {} wide and wants 2*{}. The \
                 model's hidden size is not the one this design was exported for.",
                self.design.info().name,
                params.len(),
                self.cols,
                self.cols
            )));
        }
        // THE PARAMETERS, IN BYTES.
        //
        // The design's parameter buffer holds them as f32 -- gamma then
        // beta -- and the buffer is BOUND to a staged slot: a site is
        // staged once and bound, and the same site is bound again the
        // next time it is used.
        let mut bytes = Vec::with_capacity(params.len() * 4);
        for value in params {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let slot = match self.staged_params.iter().find(|(p, _)| p == params) {
            Some((_, slot)) => *slot,
            None => {
                let slot = self.design.stage(BUFFER_PARAMS, &bytes)?;
                self.staged_params.push((params.to_vec(), slot));
                slot
            }
        };
        self.design.bind(BUFFER_PARAMS, slot)?;

        let mut done = 0usize;
        while done < n_rows {
            let chunk = (n_rows - done).min(self.rows);
            let n = chunk * self.cols;
            self.fill(&x[done * self.cols..done * self.cols + n])?;
            self.dispatch()?;
            self.read_back(&mut x[done * self.cols..done * self.cols + n])?;
            done += chunk;
        }
        Ok(())
    }

    /// The RMSNorm, over `n_rows` rows.
    ///
    /// `weight` is the gain, f32 and `cols` values long.
    pub fn rmsnorm(
        &mut self,
        weight: &[f32],
        x: &mut [f32],
        n_rows: usize,
    ) -> Result<(), ArrayError> {
        self.require(Kind::RmsNorm)?;
        if weight.len() != self.cols {
            return Err(ArrayError::Design(format!(
                "{}: RMSNorm weight is {} floats, this design's row is {} wide and wants {}. The \
                 model's hidden size is not the one this design was exported for.",
                self.design.info().name,
                weight.len(),
                self.cols,
                self.cols
            )));
        }
        let mut bytes = Vec::with_capacity(weight.len() * 4);
        for value in weight {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let slot = match self.staged_params.iter().find(|(p, _)| p == weight) {
            Some((_, slot)) => *slot,
            None => {
                let slot = self.design.stage(BUFFER_PARAMS, &bytes)?;
                self.staged_params.push((weight.to_vec(), slot));
                slot
            }
        };
        self.design.bind(BUFFER_PARAMS, slot)?;

        let mut done = 0usize;
        while done < n_rows {
            let chunk = (n_rows - done).min(self.rows);
            let n = chunk * self.cols;
            self.fill(&x[done * self.cols..done * self.cols + n])?;
            self.dispatch()?;
            self.read_back(&mut x[done * self.cols..done * self.cols + n])?;
            done += chunk;
        }
        Ok(())
    }

    /// The softmax, over `n_rows` rows.
    ///
    /// The row is a score row: the softmax is along it, and the row's
    /// width is the design's own.
    pub fn softmax(&mut self, x: &mut [f32], n_rows: usize) -> Result<(), ArrayError> {
        self.require(Kind::Softmax)?;
        let mut done = 0usize;
        while done < n_rows {
            let chunk = (n_rows - done).min(self.rows);
            let n = chunk * self.cols;
            self.fill(&x[done * self.cols..done * self.cols + n])?;
            self.dispatch()?;
            self.read_back(&mut x[done * self.cols..done * self.cols + n])?;
            done += chunk;
        }
        Ok(())
    }

    /// The softmax, over `n_rows` rows, WITHOUT counting the dispatch.
    ///
    /// The count is the number of times the FAMILY asked the array to do
    /// something, and the attention's own softmax is not the family's ask:
    /// the C++'s encoder counts the attention's two GEMMs and not the
    /// softmax between them, and a count that disagreed with the C++'s on
    /// a run that did exactly the same work would be a count of something
    /// else.
    pub fn softmax_uncounted(&mut self, x: &mut [f32], n_rows: usize) -> Result<(), ArrayError> {
        let before = self.dispatches;
        self.softmax(x, n_rows)?;
        self.dispatches = before;
        Ok(())
    }

    /// The GELU, over `n` values.
    ///
    /// The GELU's "row" is the whole flat span of activations, so the
    /// count is in elements and not in rows.
    pub fn gelu(&mut self, x: &mut [f32]) -> Result<(), ArrayError> {
        self.require(Kind::Gelu)?;
        let n = x.len();
        let mut done = 0usize;
        while done < n {
            let chunk = (n - done).min(self.capacity);
            self.fill(&x[done..done + chunk])?;
            self.dispatch()?;
            self.read_back(&mut x[done..done + chunk])?;
            done += chunk;
        }
        Ok(())
    }

    /// THE INPUT BUFFER, FILLED WHOLE.
    ///
    /// The real values first, then the tail as zeros: the buffer is the
    /// design's own, so a tail still holding the previous call's
    /// activations would be read as this call's data.
    fn fill(&mut self, values: &[f32]) -> Result<(), ArrayError> {
        let mut bytes = vec![0u8; self.capacity * 2];
        npum_format::bf16::f32_to_bf16_bytes(values, &mut bytes);
        self.design.write_bound(0, &bytes)
    }

    /// THE DISPATCH.
    fn dispatch(&mut self) -> Result<(), ArrayError> {
        self.design.dispatch(0)?;
        self.dispatches += 1;
        Ok(())
    }

    /// THE OUTPUT BUFFER, BACK IN f32.
    ///
    /// The C buffer is the LAST one: the design's buffers are A, the
    /// parameters and C.
    fn read_back(&self, values: &mut [f32]) -> Result<(), ArrayError> {
        let mut out = vec![0u8; values.len() * 2];
        let c = self.design.info().buffers.len() - 1;
        self.design.read(c, &mut out)?;
        npum_format::bf16::bf16_to_f32_bytes(&out, values);
        Ok(())
    }

    /// THE CHECK THAT A DESIGN COMPUTES WHAT IT IS ASKED FOR.
    fn require(&self, kind: Kind) -> Result<(), ArrayError> {
        if self.kind != kind {
            return Err(ArrayError::Design(format!(
                "{}: this design computes {}, not {}.",
                self.design.info().name,
                self.kind.label(),
                kind.label()
            )));
        }
        Ok(())
    }
}

/// The array's elementwise passes, for a family that dispatches them.
///
/// Each pass is its own design and its own context, and a pass whose
/// design was not opened is a pass that stays on the host.
pub struct Eltwise<T: Transport> {
    layernorm: Option<EltwiseDesign<T>>,
    gelu: Option<EltwiseDesign<T>>,
    softmax: Option<EltwiseDesign<T>>,
    rmsnorm: Option<EltwiseDesign<T>>,
}

impl<T: Transport> Eltwise<T> {
    /// The passes, each one optional.
    ///
    /// The designs are checked here and not left to the call: a design
    /// handed to the wrong slot is a design that computes the wrong thing,
    /// and it is better to refuse it at load than to dispatch it and read
    /// back a plausible number.
    pub fn new(
        layernorm: Option<EltwiseDesign<T>>,
        gelu: Option<EltwiseDesign<T>>,
        softmax: Option<EltwiseDesign<T>>,
    ) -> Result<Self, ArrayError> {
        Self::with_rmsnorm(layernorm, gelu, softmax, None)
    }

    /// The passes including RMSNorm, each one optional.
    pub fn with_rmsnorm(
        layernorm: Option<EltwiseDesign<T>>,
        gelu: Option<EltwiseDesign<T>>,
        softmax: Option<EltwiseDesign<T>>,
        rmsnorm: Option<EltwiseDesign<T>>,
    ) -> Result<Self, ArrayError> {
        if let Some(d) = &layernorm {
            d.require(Kind::LayerNorm)?;
        }
        if let Some(d) = &gelu {
            d.require(Kind::Gelu)?;
        }
        if let Some(d) = &softmax {
            d.require(Kind::Softmax)?;
        }
        if let Some(d) = &rmsnorm {
            d.require(Kind::RmsNorm)?;
        }
        Ok(Self {
            layernorm,
            gelu,
            softmax,
            rmsnorm,
        })
    }

    /// How many dispatches the passes have run between them.
    pub fn dispatches(&self) -> usize {
        self.layernorm.as_ref().map_or(0, |d| d.dispatches())
            + self.gelu.as_ref().map_or(0, |d| d.dispatches())
            + self.softmax.as_ref().map_or(0, |d| d.dispatches())
            + self.rmsnorm.as_ref().map_or(0, |d| d.dispatches())
    }
}

impl<T: Transport> npum_model::array::ArrayEltwise for Eltwise<T> {
    fn on_array(&self, op: npum_model::array::EltOp) -> bool {
        match op {
            npum_model::array::EltOp::LayerNorm => self.layernorm.is_some(),
            npum_model::array::EltOp::Gelu => self.gelu.is_some(),
            npum_model::array::EltOp::Softmax => self.softmax.is_some(),
            npum_model::array::EltOp::RmsNorm => self.rmsnorm.is_some(),
        }
    }

    fn layernorm(&mut self, params: &[f32], x: &mut [f32], n_rows: usize) -> SeamResult<()> {
        let design = self.layernorm.as_mut().ok_or_else(|| {
            npum_model::array::ArrayError(
                "this run did not ask for the LayerNorm to go on the array".to_string(),
            )
        })?;
        design
            .layernorm(params, x, n_rows)
            .map_err(|e| npum_model::array::ArrayError(e.to_string()))
    }

    fn rmsnorm(&mut self, weight: &[f32], x: &mut [f32], n_rows: usize) -> SeamResult<()> {
        let design = self.rmsnorm.as_mut().ok_or_else(|| {
            npum_model::array::ArrayError(
                "this run did not ask for RMSNorm to go on the array".to_string(),
            )
        })?;
        design
            .rmsnorm(weight, x, n_rows)
            .map_err(|e| npum_model::array::ArrayError(e.to_string()))
    }

    fn gelu(&mut self, x: &mut [f32]) -> SeamResult<()> {
        let design = self.gelu.as_mut().ok_or_else(|| {
            npum_model::array::ArrayError(
                "this run did not ask for the GELU to go on the array".to_string(),
            )
        })?;
        design
            .gelu(x)
            .map_err(|e| npum_model::array::ArrayError(e.to_string()))
    }

    fn softmax(&mut self, x: &mut [f32], n_rows: usize) -> SeamResult<()> {
        let design = self.softmax.as_mut().ok_or_else(|| {
            npum_model::array::ArrayError(
                "this run did not ask for the softmax to go on the array".to_string(),
            )
        })?;
        design
            .softmax(x, n_rows)
            .map_err(|e| npum_model::array::ArrayError(e.to_string()))
    }

    fn attention_softmax(&mut self, x: &mut [f32], n_rows: usize) -> SeamResult<()> {
        let design = self.softmax.as_mut().ok_or_else(|| {
            npum_model::array::ArrayError(
                "this run did not ask for the softmax to go on the array".to_string(),
            )
        })?;
        design
            .softmax_uncounted(x, n_rows)
            .map_err(|e| npum_model::array::ArrayError(e.to_string()))
    }

    fn softmax_cols(&self) -> Option<usize> {
        self.softmax.as_ref().map(|d| d.cols())
    }

    fn dispatches(&self) -> usize {
        self.dispatches()
    }
}
