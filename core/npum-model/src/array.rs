//! The seam between a family's host path and the array.
//!
//! # WHY A TRAIT AND NOT A FLAG
//!
//! A family's path is a straight line of host code -- an im2col, a GEMM, a
//! transpose -- and the array's path is the same line with one call
//! different. The difference has to be made ONCE, in one place, and it has
//! to be made by the CALLER: the family knows which convolution it is
//! running and what its own shape is, and only the family can say that.
//!
//! So the family calls a trait, and the trait has two implementations that
//! matter: the array's, and the host's. The host's implementation is not a
//! fallback bolted on afterwards -- it is the path every run took before the
//! array existed, and it stays bit for bit the same, because a port that
//! moves the numbers is not a port.
//!
//! # WHY IT IS IN THIS CRATE
//!
//! The same reason the `Backend` trait is: the family calls it and the
//! backend implements it, and a trait that only one side can name is a trait
//! the other side cannot use. `npum-cpu` and `npum-xdna` both depend on
//! this crate and neither on the other, which is what keeps the CPU
//! reference comparable against anything.
//!
//! # WHAT THE ARRAY'S SIDE IS TOLD
//!
//! The family's own shape -- the rows, the reduction width, the output
//! width -- and the convolution's identity. The identity is the WEIGHT's
//! name in the container ("det.conv.0.w"), because that is the name the
//! family's graph already carries, and because the panel the array
//! multiplies by is the weight's own: the same numbers, rounded to bf16,
//! padded to the design's widths and tiled into the MMAC's byte order.
//!
//! The array's side knows the widths IT wants -- the design's padded ones --
//! and it does the padding. The family must not: a family that padded its
//! own rows would have to know the design's shape, and the design's shape
//! is the design's business.

use std::fmt;

/// What went wrong while a family dispatched work to the array.
///
/// The message is the whole of it: the array's failures are reported to
/// the person who asked for the array, and a refusal that names the
/// convolution and the shape is worth ten that name the module.
#[derive(Debug, Clone)]
pub struct ArrayError(pub String);

impl fmt::Display for ArrayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ArrayError {}

/// The result of an array operation.
pub type Result<T> = std::result::Result<T, ArrayError>;

/// One convolution the array will run.
///
/// This is the family's description of a convolution, and it is here
/// because both sides need it: the family knows which convolutions it runs
/// and what their own shapes are, and the array's side knows what it can
/// dispatch. The shape in here is the CONVOLUTION's own -- the reduction
/// width and the output width -- and the design's padded widths are the
/// array's business.
#[derive(Debug, Clone)]
pub struct ConvSpec {
    /// The weight's name in the container -- "det.conv.0.w".
    ///
    /// It is the identity: the panel is the weight's own, and two
    /// convolutions of the same shape are two different weights.
    pub weight: String,

    /// The design's stream for it -- "conv64x128" -- as the packer recorded
    /// it in the graph's op list.
    ///
    /// It is the SHAPE the convolution is dispatched against, and two
    /// convolutions of the same shape share it -- which is exactly why it
    /// cannot be the panel's key.
    pub stream: String,

    /// The convolution's own reduction width: `cin * kh * kw`.
    pub k: usize,

    /// The convolution's own output width: its output channels.
    pub n: usize,

    /// The convolution's own bias, if it has one.
    pub bias: Option<Vec<f32>>,
}

/// A borrowed handle on the array's convolutions.
///
/// THE TWO LIFETIMES ARE SEPARATE ON PURPOSE. A `&mut` is invariant in the
/// lifetime of the thing it points at, so a handle that tied the trait
/// object's lifetime to the borrow's would refuse every REBORROW -- and a
/// reborrow is exactly what a walk over a graph does, once per op, with the
/// same handle.
pub type ArrayRef<'a, 'b> = Option<&'a mut (dyn ArrayConvs + 'b)>;

/// A borrowed handle on the array's projections.
///
/// The two lifetimes are separate for the same reason the convolution
/// handle's are: a `&mut` is invariant in what it points at, and a
/// reborrow is what a walk over a graph does.
pub type ArrayGemmRef<'a, 'b> = Option<&'a mut (dyn ArrayGemm + 'b)>;

/// A borrowed handle on the array's attention.
///
/// The two lifetimes are separate for the same reason the others are: a
/// `&mut` is invariant in what it points at, and a reborrow is what a walk
/// over a graph does.
pub type ArrayAttentionRef<'a, 'b> = Option<&'a mut (dyn ArrayAttention + 'b)>;

/// A borrowed handle on the array's dense work: the projections and the
/// attention, which are one design on the board.
///
/// The two lifetimes are separate for the same reason the others are.
pub type ArrayTextRef<'a, 'b> = Option<&'a mut (dyn ArrayText + 'b)>;

/// Which elementwise pass a question is about.
///
/// The passes are separate designs and separate programs, so a run can put
/// one of them on the array and leave the others on the host -- and the
/// family has to ask WHICH, once per pass, before it calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EltOp {
    /// The LayerNorm.
    LayerNorm,
    /// The GELU.
    Gelu,
    /// The softmax.
    Softmax,
    /// The RMSNorm (weight only, no mean/beta).
    RmsNorm,
}

/// The array's side of a family's elementwise pass.
///
/// A LayerNorm, a softmax and a GELU are three designs and one mechanism:
/// the array computes a block of rows and the host fills and reads it. The
/// trait is the seam, and it is the same seam as the projection's one.
pub trait ArrayEltwise {
    /// Whether one pass is on the array.
    ///
    /// The flag names the passes one by one, and a run can name one of
    /// them and not the others -- so the family asks, once per pass, and
    /// sends the rest to the host.
    fn on_array(&self, op: EltOp) -> bool;

    /// The LayerNorm, over `n_rows` rows of the design's own width.
    ///
    /// `params` is the gain and the offset -- gamma first, then beta, f32 --
    /// and it is `2 * cols` values long.
    fn layernorm(&mut self, params: &[f32], x: &mut [f32], n_rows: usize) -> Result<()>;

    /// The RMSNorm, over `n_rows` rows of the design's own width.
    ///
    /// `weight` is the gain (gamma), f32, and it is `cols` values long.
    /// Models that use RMSNorm (e.g. Gemma) pass no mean subtraction and no
    /// beta; the epsilon is compiled into the design.
    fn rmsnorm(&mut self, _weight: &[f32], _x: &mut [f32], _n_rows: usize) -> Result<()> {
        Err(ArrayError(
            "this run did not ask for RMSNorm to go on the array".to_string(),
        ))
    }

    /// The GELU, over the whole span.
    ///
    /// The GELU's "row" is the whole flat span of activations, so the count
    /// is in elements and not in rows.
    fn gelu(&mut self, x: &mut [f32]) -> Result<()>;

    /// The softmax, over `n_rows` score rows.
    fn softmax(&mut self, x: &mut [f32], n_rows: usize) -> Result<()>;

    /// The softmax the ATTENTION uses.
    ///
    /// A separate method and not the family's `softmax`, because the
    /// dispatch count means ONE thing: the number of times the FAMILY
    /// asked the array to do something. The attention's own softmax is the
    /// attention's business -- the reference's encoder counts the attention's
    /// two GEMMs and not the softmax between them -- and counting it here
    /// would make this runtime's number disagree with the reference's on a run
    /// that did exactly the same work.
    ///
    /// The default forwards to `softmax`, and an implementation that has
    /// no count of its own may keep it.
    fn attention_softmax(&mut self, x: &mut [f32], n_rows: usize) -> Result<()> {
        self.softmax(x, n_rows)
    }

    /// The width of a score row, when the softmax is on the array.
    ///
    /// `None` when the softmax stays on the host, and then the family's own
    /// width is the right one.
    ///
    /// THE ARRAY'S WIDTH IS THE DESIGN'S, and the family's is the sequence
    /// length. They are the same number only when the design was exported
    /// for this model's sequence, and a softmax that normalises a
    /// different number of keys than the family laid out is a softmax over
    /// the wrong row -- so the family asks, and lays the scores out at the
    /// width it is given.
    fn softmax_cols(&self) -> Option<usize> {
        None
    }

    /// How many dispatches this backend has run.
    ///
    /// The count is the RUN'S, and it is the number that says whether the
    /// flag did what it said.
    fn dispatches(&self) -> usize;
}

/// The array's side of a family's projection.
///
/// A projection is a matrix multiplication and nothing else, and this is
/// the same seam as the convolution's one: the family calls the trait, the
/// array's side implements it, and the difference between the two paths is
/// the weights' width -- the array multiplies them in bf16 and the host in
/// f32 -- and not the shape.
pub trait ArrayGemm {
    /// The backend's name, as it appears in the status block.
    fn name(&self) -> &'static str;

    /// Whether the projections are on the array.
    ///
    /// THE HANDLE'S PRESENCE IS NOT THE ANSWER: the text family's handle
    /// covers the projections AND the attention, and a run that asked only
    /// for the attention has a handle and no projection panels. So the
    /// family asks, once, and sends the rest to the host.
    fn gemm_on_array(&self) -> bool;

    /// How many projections this backend has dispatched.
    fn dispatches(&self) -> usize;

    /// How many bytes of weight panels are staged on the board.
    fn staged_bytes(&self) -> usize;

    /// Run one projection on the array.
    ///
    /// `weight` is the projection's weight name in the container --
    /// "layer.0.qkv" -- and the array's side finds the panel that belongs
    /// to it. `a` holds the host's activations, `rows` of `k` values, and
    /// `out` receives `rows` of `n` values.
    ///
    /// The bias is an argument and not part of the panel, because the
    /// panel's own width is the DESIGN's and the bias's is the PROJECTION's
    /// -- and they are the same number only because the design was exported
    /// for this model.
    fn gemm(
        &mut self,
        weight: &str,
        a: &[f32],
        rows: usize,
        k: usize,
        n: usize,
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()>;

    /// Run a batch of INDEPENDENT projections.
    ///
    /// `outs[i]` receives op `i`'s output. An array backend overlaps the host
    /// side of op N+1 with the device compute of op N; a backend with nothing
    /// to overlap runs the ops in order through [`Self::gemm`], which is what
    /// the default does and what every caller sees when the projections stay
    /// on the host.
    ///
    /// Dependent ops do NOT belong in one batch: the overlap is only safe when
    /// an op does not read another op's output. The four projections of a
    /// transformer layer are a chain, not a batch -- QKV feeds the attention,
    /// whose output feeds the output projection, and so on -- so a layer's own
    /// GEMMs stay sequential. A batch is for projections that really are
    /// independent, such as a decoder's self-attention and cross-attention
    /// inputs, or a head's parallel key and value panels.
    fn gemm_batch(&mut self, ops: &[PipelinedGemm<'_>], outs: &mut [&mut [f32]]) -> Result<()> {
        if ops.len() != outs.len() {
            return Err(ArrayError(format!(
                "{} projections were submitted with {} output buffers",
                ops.len(),
                outs.len()
            )));
        }
        for (op, out) in ops.iter().zip(outs.iter_mut()) {
            self.gemm(op.weight, op.a, op.rows, op.k, op.n, op.bias, out)?;
        }
        Ok(())
    }
}

/// One independent projection in a pipelined batch.
///
/// The same arguments as [`ArrayGemm::gemm`], gathered so a caller can hand
/// several projections to [`ArrayGemm::gemm_batch`] at once. The ops must NOT
/// depend on one another: an array implementation overlaps op N+1's host
/// packing and upload with op N's device compute, which is only safe when
/// N+1 does not read N's output.
#[derive(Debug, Clone, Copy)]
pub struct PipelinedGemm<'a> {
    /// The weight's name in the container -- "layer.0.qkv".
    pub weight: &'a str,
    /// The activations: `rows` rows of `k` f32 values.
    pub a: &'a [f32],
    /// The number of rows in `a` and in the output.
    pub rows: usize,
    /// The reduction width.
    pub k: usize,
    /// The output width.
    pub n: usize,
    /// The bias of `n` values, when the projection has one.
    pub bias: Option<&'a [f32]>,
}

/// The array's side of a family's dense convolution.
///
/// A family's dense convolution is an im2col on the host and a matrix
/// multiplication on the array, and this trait is the second half. The
/// transposed convolutions and projections stay on the host: a transposed
/// convolution's output is a sum of shifted contributions, not a matrix
/// product. 3x3 depthwise has its own design and `ArrayConvs::dwconv`; it
/// runs on the host unless that op is named and the handle carries the set.
pub trait ArrayConvs {
    /// The backend's name, as it appears in the status block.
    fn name(&self) -> &'static str;

    /// How many convolutions this backend has dispatched.
    ///
    /// The count is the RUN'S, and it is the number that says whether the
    /// flag did what it said. A run that asked for the array and dispatched
    /// nothing is a run that lied, and this is where the lie shows.
    fn dispatches(&self) -> usize;

    /// How many bytes of weight panels are staged on the board.
    ///
    /// It is the load's number and not the run's: the panels are staged
    /// once, at load, and every run of the model reuses them.
    fn staged_bytes(&self) -> usize;

    /// How long the host spent turning its activations into the device's
    /// dtype, in seconds.
    ///
    /// It is a REPORT and not a cost: the array's time is what the run
    /// costs, and this is the part of it that the host spends on its own
    /// side of the boundary.
    fn convert_seconds(&self) -> f64;

    /// How many rows one dispatch computes.
    ///
    /// It is the DESIGN's number and not the family's: the design was
    /// compiled for a fixed number of rows per dispatch, and a convolution
    /// with more rows than that is cut into chunks of it. The status block
    /// prints it, because "how many rows at a time" is what says whether
    /// the array's number and the host's number are about the same work.
    fn rows_per_dispatch(&self) -> usize;

    /// How many streams the design set carries.
    ///
    /// It is the design's own count of the shapes it can compute, and it is
    /// printed because a design with fourteen streams and a family with
    /// fourteen shapes is a coincidence worth seeing.
    fn stream_count(&self) -> usize;

    /// Run one dense convolution on the array.
    ///
    /// `weight` is the convolution's weight name in the container --
    /// "det.conv.0.w" -- and the array's side finds the panel that belongs
    /// to it. `a` holds the host's im2col rows, `rows` of `k` values;
    /// `out` receives `rows` of `n` values.
    ///
    /// The bias is NOT an argument: the panel carries the convolution's own
    /// bias, padded with zeros to the design's width, and adding it here
    /// would add it twice.
    fn conv(
        &mut self,
        weight: &str,
        a: &[f32],
        rows: usize,
        k: usize,
        n: usize,
        out: &mut [f32],
    ) -> Result<()>;

    /// Whether this handle can dispatch 3x3 depthwise convolutions.
    ///
    /// False by default: a backend opened for `conv` alone must leave
    /// depthwise on the host rather than refuse a layer the user never
    /// asked to move.
    fn dwconv_on_array(&self) -> bool {
        false
    }

    /// Dispatch a 3x3 depthwise convolution on the array.
    ///
    /// `weight` names the container tensor (`<net>.conv.<i>.w`) whose
    /// `[C,1,3,3]` weights and `[C]` bias were staged at open. `a` holds, for
    /// every channel, three contiguous padded rows of `stride` f32 each --
    /// the caller's gather, with the horizontal pad already applied and the
    /// output row already chosen. `outputs` is the real width of one output
    /// row; `out` receives `[C, outputs]` pre-activation f32, and the caller
    /// applies the fused activation exactly as it does on the host path.
    ///
    /// Default: fail closed. A backend with no depthwise design set must
    /// refuse by name, never silently compute this on the host.
    fn dwconv(
        &mut self,
        _weight: &str,
        _a: &[f32],
        _channels: usize,
        _outputs: usize,
        _stride: usize,
        _out: &mut [f32],
    ) -> Result<()> {
        Err(ArrayError(
            "this array backend has no depthwise design set; --npu-ops dwconv \
             cannot be honoured"
                .to_string(),
        ))
    }
}

/// The Whisper front end's two passes that are matrix products on the array.
///
/// The 400-point DFT IS a matrix of twiddles and the Slaney mel bank IS a
/// `[201, n_mels]` matrix. Both ship BAKED into the exported xclbin -- there is
/// no weight panel to stage -- so they are not `ArrayGemm` calls against a
/// container operand. They live in the encoder's design set, behind the same
/// handle and the same hardware context. `--npu-ops fft` / `mproj` must refuse
/// by name when the set does not carry their streams.
///
/// The defaults fail: a backend that has not ported these operations must not
/// compute them on the host and report the flag as honoured.
pub trait ArrayFrontend {
    /// The backend's name, as it appears in the status block.
    fn name(&self) -> &'static str {
        "unavailable"
    }

    /// Whether this backend really dispatches the front end.
    fn frontend_on_array(&self) -> bool {
        false
    }

    /// Whether the 400-point transform has a stream on this array.
    fn dft_on_array(&self) -> bool {
        false
    }

    /// Whether the mel projection has a stream on this array.
    fn mproj_on_array(&self) -> bool {
        false
    }

    /// How many front-end streams this backend has dispatched.
    fn frontend_dispatches(&self) -> usize {
        0
    }

    /// 400-point DFT of `frames` windowed rows of 400 f32 samples.
    ///
    /// `power` receives the magnitude-squared spectrum in the front end's
    /// channels-first layout, `[n_bins, frames]`. Squaring is done here because
    /// the design has no epilogue for it -- it is not a matrix product.
    fn dft400(
        &mut self,
        _frames: &[f32],
        _n_frames: usize,
        _n_bins: usize,
        _power: &mut [f32],
    ) -> Result<()> {
        Err(ArrayError(
            "this array backend cannot dispatch dft400".to_string(),
        ))
    }

    /// Slaney mel projection of a channels-first power spectrum.
    ///
    /// `power` is `[n_bins, frames]`; `mel` receives `[n_mels, frames]`.
    fn mproj(
        &mut self,
        _power: &[f32],
        _n_frames: usize,
        _n_bins: usize,
        _n_mels: usize,
        _mel: &mut [f32],
    ) -> Result<()> {
        Err(ArrayError(
            "this array backend cannot dispatch mel_proj".to_string(),
        ))
    }
}

/// The Whisper decoder's tied-embedding logit head.
///
/// The eight `logits_*` streams live in the decoder design set and compute
/// eight vocabulary slices of one step. They are not `ArrayGemm` calls: the
/// B operands are the tied embedding table, staged as host-generated panels,
/// and the caller wants one flat `[vocab]` vector back.
///
/// The defaults fail closed: a backend that has not staged these panels must
/// refuse the operation, not silently answer from the host and report the
/// flag as honoured.
pub trait ArrayLogits {
    /// Whether this backend really dispatches the tied logit head.
    fn logit_on_array(&self) -> bool {
        false
    }

    /// How many logit chunks have been dispatched.
    fn logit_dispatches(&self) -> usize {
        0
    }

    /// `logits = h @ embed_tokens.T`, for one decoder step.
    fn logits(&mut self, _h: &[f32], _out: &mut [f32]) -> Result<()> {
        Err(ArrayError(
            "this array backend cannot dispatch the tied logit head".to_string(),
        ))
    }
}

/// The ViT classifier head: one CLS row times a `[hidden, num_labels]` weight.
///
/// The labels are not a legal B-panel width (1000 is not a multiple of
/// tile_n*cols), but the ViT set's `qkv` stream is `[hidden, 2304]`, so the
/// backend can stage a zero-padded panel from `classifier.weight` and read
/// only the real labels back. Like `ArrayLogits`, this is a supertrait of
/// `ArrayText` with fail-closed defaults: an unported backend must refuse.
pub trait ArrayHead {
    fn head_on_array(&self) -> bool {
        false
    }

    fn head_dispatches(&self) -> usize {
        0
    }

    fn head(&mut self, _cls: &[f32], _logits: &mut [f32]) -> Result<()> {
        Err(ArrayError(
            "this array backend cannot dispatch the ViT classifier head".to_string(),
        ))
    }
}

/// Whisper's two audio convolutions, expressed as GEMMs on the encoder set's
/// existing streams.
///
/// conv1 is `[frames, 3*n_mels] x [3*n_mels, hidden]` and conv2 is
/// `[frames, 3*hidden] x [3*hidden, hidden]`; both B panels are generated
/// from the Conv1d weights and staged like the front end's. The host gathers
/// the windowed A rows and applies bias + GELU -- the design has no epilogue.
/// Defaults fail closed.
pub trait ArrayAudioConv {
    fn conv_on_array(&self) -> bool {
        false
    }

    fn conv_dispatches(&self) -> usize {
        0
    }

    /// The reduction width the staged conv panels were compiled for.
    fn conv1_k(&self) -> usize {
        0
    }

    /// The reduction width of conv2's panel.
    fn conv2_k(&self) -> usize {
        0
    }

    fn conv1_gemm(&mut self, _a: &[f32], _frames: usize, _out: &mut [f32]) -> Result<()> {
        Err(ArrayError(
            "this array backend cannot dispatch Whisper's conv1".to_string(),
        ))
    }

    fn conv2_gemm(&mut self, _a: &[f32], _frames: usize, _out: &mut [f32]) -> Result<()> {
        Err(ArrayError(
            "this array backend cannot dispatch Whisper's conv2".to_string(),
        ))
    }
}

/// The attention's own shape.
///
/// The family knows it and the array's side needs it, and it is one struct
/// because it is one fact: a head is `head_dim` wide, there are `heads` of
/// them, and they make a row of `hidden`.
#[derive(Debug, Clone, Copy)]
pub struct AttentionShape {
    /// How many sequences.
    pub batch: usize,

    /// How many positions per sequence.
    pub seq: usize,

    /// How many heads.
    pub heads: usize,

    /// One head's width.
    pub head_dim: usize,

    /// The model's width -- the hidden size, and `heads * head_dim`.
    pub hidden: usize,
}

/// The array's side of a family's attention.
///
/// The attention is three things -- QK^T, a softmax and softmax.V -- and the
/// array can take the two GEMMs while the softmax stays on the host, or the
/// other way round. The two are SEPARATE decisions because the flag names
/// them separately: `attn` moves the GEMMs and `softm` moves the softmax,
/// and a run that names one has not named the other.
///
/// The trait is the whole block: the family hands over the activations and
/// the mask, and the context comes back. The family does not walk the heads
/// and the query chunks itself, because the array's side is the only one
/// that knows how many rows one dispatch computes.
pub trait ArrayAttention {
    /// The backend's name, as it appears in the status block.
    fn name(&self) -> &'static str;

    /// Whether the attention's GEMMs are on the array.
    fn on_array(&self) -> bool;

    /// The width of a score row, when the attention is on the array.
    ///
    /// `None` when the attention stays on the host, and then the family's own
    /// width is the right one.
    ///
    /// THE ARRAY'S WIDTH IS THE DESIGN'S, and it is the PADDED key count: the
    /// array's score row has a column for every key the design was compiled
    /// for, and the columns past the sequence are padding. A softmax that
    /// normalises a different number of keys than the row holds is a softmax
    /// over the wrong row.
    fn score_cols(&self) -> Option<usize> {
        None
    }

    /// How many attention dispatches this backend has run.
    ///
    /// The count is the RUN'S, and it is the number that says whether the
    /// flag did what it said: two dispatches per head per query chunk.
    fn attn_dispatches(&self) -> usize;

    /// The attention's own QK^T time and its softmax.V time, in seconds.
    ///
    /// They are the attention's sub-items and not the block's: the block's
    /// own time also covers the panels, which are built on the host once per
    /// head. The two are reported separately so that a reader can see that
    /// the sub-items add up to less than the block, which is what they do
    /// on the array's path too.
    fn attn_seconds(&self) -> (f64, f64);

    /// Run the attention over the batch.
    ///
    /// `qkv` is the batch's Q|K|V block -- `batch * seq` rows of three
    /// `hidden` values -- and `mask` is the additive mask, `batch * seq`
    /// values, one per query position.
    ///
    /// `softmax` is the array's softmax when the run put it there, and
    /// `None` when the softmax is on the host. Either way the softmax runs
    /// at the PADDED width, because the score rows are the array's rows.
    ///
    /// `ctx` receives `batch * seq` rows of `hidden` values: the context.
    fn attention(
        &mut self,
        qkv: &[f32],
        shape: AttentionShape,
        mask: &[f32],
        softmax: Option<&mut (dyn ArrayEltwise + '_)>,
        ctx: &mut [f32],
    ) -> Result<()>;
}

/// The text family's handle on the array.
///
/// THE PROJECTIONS AND THE ATTENTION ARE ONE DESIGN on the board: the same
/// xclbin, the same instruction streams, the same buffers, one hardware
/// context. The attention's two GEMMs are streams INSIDE the projections'
/// set -- `attn_qk` and `attn_av` -- and not a set of their own, so a
/// family that holds one handle for each would be holding two handles on
/// one thing, and the borrow checker is right to refuse it.
///
/// So the family holds ONE handle, and this trait is it: the two traits'
/// operations, on one object.
pub trait ArrayText: ArrayGemm + ArrayAttention + ArrayFrontend + ArrayLogits + ArrayHead + ArrayAudioConv {}
