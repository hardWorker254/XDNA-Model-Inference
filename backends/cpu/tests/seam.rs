//! The family's SEAM: which operation goes to the array and which stays on
//! the host.
//!
//! # What this test is
//!
//! The `--npu-ops` flag names the operations one by one, and the family's
//! call sites have to honour that: a run that asked for the LayerNorm on
//! the array and left the GELU on the host must do exactly that, and a
//! call site that sent everything to the array would move work the reader
//! did not ask to move -- and one that sent nothing would make the command
//! line lie.
//!
//! So the test puts a RECORDER where the array goes and checks what the
//! family asked of it. The recorder does not compute anything: it counts
//! the calls, and the vector it produces is not the point.
//!
//! # What it is NOT
//!
//! It is not a numerics gate. The numbers the array produces are checked
//! against the oracle's on the board, by
//! `tools/verify/verify_array_embed.py`; this is about the ROUTING.

use std::path::{Path, PathBuf};

use npum_model::array::{ArrayEltwise, ArrayGemm, ArrayLogits, ArrayText, AttentionShape, EltOp, };

/// The repository root, from this crate's manifest directory.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("the repository root exists")
}

/// The container, if it is installed.
fn model_path() -> Option<PathBuf> {
    let path = repo_root().join("models/all-MiniLM-L6-v2.npue");
    if path.is_file() { Some(path) } else { None }
}

/// The sequence length the oracle's design set was built for.
const SEQ: usize = 64;

/// What the family asked of the array.
#[derive(Default)]
struct Recorder {
    /// Which passes are on the array.
    on: Vec<EltOp>,
    /// How many times the LayerNorm was called.
    layernorm: usize,
    /// How many times the GELU was called.
    gelu: usize,
    /// How many times the softmax was called.
    softmax: usize,
    /// The parameters' length at each LayerNorm call.
    params: Vec<usize>,
    /// The row count at each softmax call.
    rows: Vec<usize>,
    /// The width of the score matrix the softmax was handed.
    width: usize,
}

impl Recorder {
    /// A recorder with the given passes on the array.
    fn with(ops: &[EltOp]) -> Self {
        Self {
            on: ops.to_vec(),
            ..Self::default()
        }
    }
}

impl ArrayEltwise for Recorder {
    fn on_array(&self, op: EltOp) -> bool {
        self.on.contains(&op)
    }

    fn layernorm(
        &mut self,
        params: &[f32],
        _x: &mut [f32],
        _n_rows: usize,
    ) -> npum_model::array::Result<()> {
        self.layernorm += 1;
        self.params.push(params.len());
        Ok(())
    }

    fn gelu(&mut self, _x: &mut [f32]) -> npum_model::array::Result<()> {
        self.gelu += 1;
        Ok(())
    }

    fn softmax(&mut self, x: &mut [f32], n_rows: usize) -> npum_model::array::Result<()> {
        self.softmax += 1;
        self.rows.push(n_rows);
        self.width = x.len() / n_rows.max(1);
        Ok(())
    }

    fn softmax_cols(&self) -> Option<usize> {
        if self.on.contains(&EltOp::Softmax) {
            Some(384)
        } else {
            None
        }
    }

    fn dispatches(&self) -> usize {
        self.layernorm + self.gelu + self.softmax
    }
}

/// Run the model with a recorder where the array goes.
fn run(path: &Path, model: &npum_cpu::bert::BertModel, recorder: &mut Recorder) {
    let container = npum_format::Container::open(path).expect("the container opens");
    let blob = container
        .tensor_bytes("tokenizer.vocab")
        .expect("the container holds a vocabulary");
    let tokenizer =
        npum_tokenizers::wordpiece::WordPiece::from_lines(&String::from_utf8_lossy(blob))
            .expect("the vocabulary is usable");
    let (encoding, truncated) = tokenizer
        .encode_padded("a golden transcript is a contract", SEQ)
        .expect("the text encodes");
    assert!(!truncated, "the text fits");
    model
        .embed(
            &encoding.ids,
            &encoding.mask,
            1,
            SEQ,
            &mut npum_cpu::ops::CpuOps::best(),
            None,
            Some(recorder),
        )
        .expect("the model runs");
}

#[test]
fn every_pass_named_reaches_the_array_and_the_others_do_not() {
    let Some(path) = model_path() else {
        eprintln!("skipped: the container is not installed");
        return;
    };
    let model = npum_cpu::bert::BertModel::open(&path).expect("the container opens");

    // ALL THREE, and all three are called.
    let mut all = Recorder::with(&[EltOp::LayerNorm, EltOp::Gelu, EltOp::Softmax]);
    run(&path, &model, &mut all);
    // One LayerNorm for the embedding and two per layer, for six layers.
    assert_eq!(all.layernorm, 1 + 2 * 6, "the LayerNorm sites");
    assert_eq!(all.gelu, 6, "the GELU is one pass per layer");
    assert_eq!(all.softmax, 6, "the softmax is one pass per layer");
    // The parameters are the gain and the offset, f32, two of them per row.
    assert!(
        all.params.iter().all(|n| *n == 2 * 384),
        "every site stages gamma and beta for a 384-wide row: {:?}",
        all.params
    );
    // The score matrix is one row per (sequence, head, query position),
    // and its width is the DESIGN's width -- the padded key count -- and
    // not the sequence length.
    assert!(
        all.rows.iter().all(|n| *n == 12 * 64),
        "the softmax's rows: {:?}",
        all.rows
    );
    assert_eq!(all.width, 384, "the score matrix's row width");

    // ONLY THE LAYERNORM, and only the LayerNorm is called.
    let mut ln = Recorder::with(&[EltOp::LayerNorm]);
    run(&path, &model, &mut ln);
    assert_eq!(ln.layernorm, 1 + 2 * 6, "the LayerNorm sites");
    assert_eq!(ln.gelu, 0, "the GELU stays on the host");
    assert_eq!(ln.softmax, 0, "the softmax stays on the host");

    // ONLY THE SOFTMAX, and only the softmax is called.
    let mut sm = Recorder::with(&[EltOp::Softmax]);
    run(&path, &model, &mut sm);
    assert_eq!(sm.layernorm, 0, "the LayerNorm stays on the host");
    assert_eq!(sm.gelu, 0, "the GELU stays on the host");
    assert_eq!(sm.softmax, 6, "the softmax is one pass per layer");
}

/// What the family asked of the array's DENSE side.
///
/// The projections and the attention are one handle -- one design, one
/// hardware context -- so the recorder is one object too.
#[derive(Default)]
struct Dense {
    /// Whether the projections are on the array.
    gemm_on: bool,
    /// Whether the attention is on the array.
    attn_on: bool,
    /// How many projections were dispatched.
    gemms: usize,
    /// How many times the attention was called.
    attns: usize,
    /// The shape at each attention call.
    shapes: Vec<AttentionShape>,
    /// The mask's length at each attention call.
    masks: Vec<usize>,
    /// Whether the attention was handed the array's softmax.
    softmaxes: Vec<bool>,
}

impl ArrayGemm for Dense {
    fn name(&self) -> &'static str {
        "recorder"
    }

    fn gemm_on_array(&self) -> bool {
        self.gemm_on
    }

    fn dispatches(&self) -> usize {
        self.gemms
    }

    fn staged_bytes(&self) -> usize {
        0
    }

    fn gemm(
        &mut self,
        _weight: &str,
        _a: &[f32],
        _rows: usize,
        _k: usize,
        _n: usize,
        _bias: Option<&[f32]>,
        _out: &mut [f32],
    ) -> npum_model::array::Result<()> {
        self.gemms += 1;
        Ok(())
    }
}

impl npum_model::array::ArrayAttention for Dense {
    fn name(&self) -> &'static str {
        "recorder"
    }

    fn on_array(&self) -> bool {
        self.attn_on
    }

    fn score_cols(&self) -> Option<usize> {
        if self.attn_on { Some(384) } else { None }
    }

    fn attn_dispatches(&self) -> usize {
        0
    }

    fn attn_seconds(&self) -> (f64, f64) {
        (0.0, 0.0)
    }

    fn attention(
        &mut self,
        _qkv: &[f32],
        shape: AttentionShape,
        mask: &[f32],
        softmax: Option<&mut (dyn ArrayEltwise + '_)>,
        ctx: &mut [f32],
    ) -> npum_model::array::Result<()> {
        self.attns += 1;
        self.shapes.push(shape);
        self.masks.push(mask.len());
        self.softmaxes.push(softmax.is_some());
        // The recorder computes nothing, so the context is zeros -- and the
        // family carries on with them, which is the point: the ROUTING is
        // what this test is about.
        for value in ctx.iter_mut() {
            *value = 0.0;
        }
        Ok(())
    }
}

impl ArrayText for Dense {}
impl ArrayLogits for Dense {}
use npum_model::array::ArrayFrontend;
impl ArrayFrontend for Dense {}

/// Run the model with a recorder where the array's dense side goes.
fn run_dense(path: &Path, model: &npum_cpu::bert::BertModel, dense: &mut Dense) {
    let container = npum_format::Container::open(path).expect("the container opens");
    let blob = container
        .tensor_bytes("tokenizer.vocab")
        .expect("the container holds a vocabulary");
    let tokenizer =
        npum_tokenizers::wordpiece::WordPiece::from_lines(&String::from_utf8_lossy(blob))
            .expect("the vocabulary is usable");
    let (encoding, truncated) = tokenizer
        .encode_padded("a golden transcript is a contract", SEQ)
        .expect("the text encodes");
    assert!(!truncated, "the text fits");
    model
        .embed(
            &encoding.ids,
            &encoding.mask,
            1,
            SEQ,
            &mut npum_cpu::ops::CpuOps::best(),
            Some(dense),
            None,
        )
        .expect("the model runs");
}

#[test]
fn the_attention_is_one_call_per_layer_and_the_projections_stay_put() {
    let Some(path) = model_path() else {
        eprintln!("skipped: the container is not installed");
        return;
    };
    let model = npum_cpu::bert::BertModel::open(&path).expect("the container opens");

    // THE ATTENTION ON THE ARRAY, THE PROJECTIONS ON THE HOST.
    //
    // The flag names the ops one by one, and `attn` names the attention
    // and not the projections: a run that asked for the attention and got
    // the projections moved as well would have done work the reader did
    // not ask for.
    let mut dense = Dense {
        attn_on: true,
        ..Dense::default()
    };
    run_dense(&path, &model, &mut dense);
    assert_eq!(dense.attns, 6, "the attention is one block per layer");
    assert_eq!(dense.gemms, 0, "the projections stay on the host");
    // The shape is the model's own: one sequence of 64 positions, twelve
    // heads of 32, and a model 384 wide.
    for shape in &dense.shapes {
        assert_eq!(shape.batch, 1);
        assert_eq!(shape.seq, SEQ);
        assert_eq!(shape.heads, 12);
        assert_eq!(shape.head_dim, 32);
        assert_eq!(shape.hidden, 384);
    }
    // The mask is one row per query position.
    assert!(dense.masks.iter().all(|n| *n == SEQ));
    // The softmax is a SEPARATE decision, and the attention was not handed
    // one: a run that asked only for `attn` has no softmax design.
    assert!(dense.softmaxes.iter().all(|b| !b));

    // THE PROJECTIONS ON THE ARRAY, THE ATTENTION ON THE HOST.
    let mut dense = Dense {
        gemm_on: true,
        ..Dense::default()
    };
    run_dense(&path, &model, &mut dense);
    assert_eq!(dense.attns, 0, "the attention stays on the host");
    assert_eq!(dense.gemms, 4 * 6, "four projections per layer");
}
