//! The arch=9 graph walk: twenty-four ops, two graphs.
//!
//! # WHAT IS IN THIS CONTAINER
//!
//! One `.npum` holds a TEXT DETECTOR, a TEXT RECOGNISER and the CHARACTER
//! DICTIONARY, plus both front ends' geometry. The three are one model: a
//! detector finds text and cannot read it, and a recogniser reads a crop and
//! cannot find one, so a container holding only one of them answers "no text
//! here" to every image.
//!
//! * `det` -- PaddleOCR's DB detector. A probability map over the whole frame
//!   at the input's own resolution, thresholded, turned into boxes.
//! * `rec` -- PaddleOCR's SVTR recogniser. One crop in, a CTC sequence out.
//! * `dict` -- the characters the recogniser's classes stand for. 436 lines for
//!   English, 438 classes once the blank and the space are added.
//!
//! # WHY THE VOCABULARY IS TWENTY-FOUR OPS AND NOT SIX
//!
//! arch=6 walks conv, concat, slice, add, maxpool, upsample and detect.
//! arch=7 walks conv, dwconv, add, maxpool, pad_c and resize. arch=8 walks
//! conv, dwconv, add, maxpool, resize and d2s. This one walks everything the
//! two PaddleOCR graphs contain, and the list is longer because the
//! RECOGNISER is a different kind of network from the other three:
//!
//! * the RECOGNISER's head is a LayerNorm and an attention block: a
//!   ReduceMean, a Sub, a Sqrt, a Div, a MatMul and a Softmax. None of those
//!   is a convolution or an elementwise pass over one, and a runtime without
//!   them could not read a crop at all.
//! * the tensors are not all `[C, H, W]`. The CTC head's output is rank 3
//!   (`[1, 40, 438]`) and the attention block's are rank 4 and rank 5, and a
//!   runtime that only held four-dimensional tensors would have to fake the
//!   rest.
//! * there is a CONVTRANSPOSE -- the DB head's two upsampling convolutions
//!   -- whose output is a sum of SHIFTED contributions and not a GEMM, so it
//!   stays on the host whatever the array is asked to do.
//!
//! # EVERY OUTPUT EXTENT IS CHECKED
//!
//! The packer resolved every shape at pack time and recorded each op's output
//! extent in the container. The walk computes its own and compares. A
//! disagreement is a bug in the packer or in here, and it is caught AT THE OP
//! rather than at the end of the graph -- which is the difference between
//! "the recogniser disagrees" and "the resize indexes its source row
//! wrongly".
//!
//! # THE ACTIVATIONS ARE THE MODEL'S
//!
//! relu, sigmoid, hardswish and hardsigmoid, and the last two carry their own
//! alpha and beta: the DETECTOR's HardSigmoid has alpha 0.2 and the
//! RECOGNISER's has 1/6. They are NOT fused into the convolution's epilogue,
//! and the reason is arithmetic rather than taste: the host convolution's
//! epilogue lives in `kernels.rs` and computes the OTHER architectures'
//! activations, and this one's carry a parameter that an enum cannot hold.
//! So they are a separate pass over the finished tensor.
//!
//! # THE ARITHMETIC IS THE ORACLE'S, AND THE FUSION IS CHECKED
//!
//! Three of the loops below accumulate `acc += a * b`, and they do NOT all
//! fuse. Read off the oracle's object file (`objdump -d .../ocr/net.cpp.o`):
//!
//! | loop | instruction | so |
//! |---|---|---|
//! | `dwconv` | `vfmadd231ss` | FUSED |
//! | `matmul` | `vfmadd231ss` | FUSED |
//! | `convtranspose` | `vmulss` then `vaddss` | NOT fused |
//!
//! The convtranspose is the odd one out, and it is the DB head's two
//! upsampling convolutions -- so the two halves of the same head disagree
//! about their rounding by one ulp per tap. `mul_add` where the oracle
//! fuses, and a plain `+` where it does not: the other way round is a
//! detector whose probability map is off by a quantization step.

use crate::kernels;
use crate::parallel;
use npum_format::Container;
use npum_model::array::{ArrayRef, ConvSpec};
use serde_json::Value;
use std::collections::BTreeMap;

/// The arch=9 container's `arch` string. Refused by name rather than run with
/// the wrong front end, which is the whole point of carrying one.
pub const ARCH: &str = "paddleocr_db_svtr_ctc";

// ---------------------------------------------------------------------------
// The tensor
// ---------------------------------------------------------------------------

/// A tensor of any rank: a shape and the values, row-major.
///
/// NOT NCHW-ONLY, and that is the difference from arch=6's and arch=8's
/// tensors. The recogniser's graph has rank-3 tensors (the CTC head's
/// `[1, 40, 438]`) and rank-5 ones (the attention block's projections), and a
/// runtime that only held `[C, H, W]` would have to fake the rest -- which is
/// how a recogniser ends up reading a transposed matrix and transcribing
/// confidently wrong characters.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Tensor {
    /// The shape, outermost first.
    pub shape: Vec<i64>,
    /// The values, row-major.
    pub d: Vec<f32>,
}

impl Tensor {
    /// The tensor's rank.
    pub fn rank(&self) -> i64 {
        self.shape.len() as i64
    }

    /// How many elements the shape holds.
    pub fn numel(&self) -> i64 {
        self.shape.iter().product::<i64>()
    }

    /// The NCHW view, for the ops that are NCHW.
    ///
    /// A tensor that is not rank 4 is refused at the call site with the op's
    /// name, not here.
    pub fn channels(&self) -> i64 {
        if self.shape.len() == 4 {
            self.shape[1]
        } else {
            0
        }
    }

    /// The NCHW height.
    pub fn height(&self) -> i64 {
        if self.shape.len() == 4 {
            self.shape[2]
        } else {
            0
        }
    }

    /// The NCHW width.
    pub fn width(&self) -> i64 {
        if self.shape.len() == 4 {
            self.shape[3]
        } else {
            0
        }
    }

    /// One channel's plane of an NCHW tensor.
    pub fn chw(&self, c: i64) -> &[f32] {
        let plane = (self.shape[2] * self.shape[3]) as usize;
        let at = (c as usize) * plane;
        &self.d[at..at + plane]
    }

    /// One channel's plane, mutably.
    pub fn chw_mut(&mut self, c: i64) -> &mut [f32] {
        let plane = (self.shape[2] * self.shape[3]) as usize;
        let at = (c as usize) * plane;
        &mut self.d[at..at + plane]
    }
}

// ---------------------------------------------------------------------------
// The ops
// ---------------------------------------------------------------------------

/// The op vocabulary of a packed graph. The runtime walks exactly these and
/// refuses anything else BY NAME: a dropped op is a network that computes
/// something else and returns a plausible transcription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Op {
    #[default]
    Conv,
    ConvTranspose,
    Relu,
    Sigmoid,
    HardSwish,
    HardSigmoid,
    Add,
    Sub,
    Mul,
    Div,
    Sqrt,
    Concat,
    MaxPool,
    AvgPool,
    GAveragePool,
    Resize,
    Reshape,
    Transpose,
    Slice,
    Squeeze,
    Unsqueeze,
    MatMul,
    Softmax,
    ReduceMean,
}

impl Op {
    /// The op's name, as the container spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            Op::Conv => "conv",
            Op::ConvTranspose => "convtranspose",
            Op::Relu => "relu",
            Op::Sigmoid => "sigmoid",
            Op::HardSwish => "hardswish",
            Op::HardSigmoid => "hardsigmoid",
            Op::Add => "add",
            Op::Sub => "sub",
            Op::Mul => "mul",
            Op::Div => "div",
            Op::Sqrt => "sqrt",
            Op::Concat => "concat",
            Op::MaxPool => "maxpool",
            Op::AvgPool => "avgpool",
            Op::GAveragePool => "gapool",
            Op::Resize => "resize",
            Op::Reshape => "reshape",
            Op::Transpose => "transpose",
            Op::Slice => "slice",
            Op::Squeeze => "squeeze",
            Op::Unsqueeze => "unsqueeze",
            Op::MatMul => "matmul",
            Op::Softmax => "softmax",
            Op::ReduceMean => "reducemean",
        }
    }

    /// The op, from the container's spelling.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "conv" => Op::Conv,
            "convtranspose" => Op::ConvTranspose,
            "relu" => Op::Relu,
            "sigmoid" => Op::Sigmoid,
            "hardswish" => Op::HardSwish,
            "hardsigmoid" => Op::HardSigmoid,
            "add" => Op::Add,
            "sub" => Op::Sub,
            "mul" => Op::Mul,
            "div" => Op::Div,
            "sqrt" => Op::Sqrt,
            "concat" => Op::Concat,
            "maxpool" => Op::MaxPool,
            "avgpool" => Op::AvgPool,
            "gapool" => Op::GAveragePool,
            "resize" => Op::Resize,
            "reshape" => Op::Reshape,
            "transpose" => Op::Transpose,
            "slice" => Op::Slice,
            "squeeze" => Op::Squeeze,
            "unsqueeze" => Op::Unsqueeze,
            "matmul" => Op::MatMul,
            "softmax" => Op::Softmax,
            "reducemean" => Op::ReduceMean,
            _ => return None,
        })
    }
}

/// The four activations the graphs carry, and the two that carry a parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    Relu,
    Sigmoid,
    HardSwish,
    HardSigmoid,
}

impl Act {
    /// The activation's name, as the container spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            Act::Relu => "relu",
            Act::Sigmoid => "sigmoid",
            Act::HardSwish => "hardswish",
            Act::HardSigmoid => "hardsigmoid",
        }
    }

    /// The activation, from the container's spelling.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "relu" => Act::Relu,
            "sigmoid" => Act::Sigmoid,
            "hardswish" => Act::HardSwish,
            "hardsigmoid" => Act::HardSigmoid,
            _ => return None,
        })
    }
}

/// One op, as the container records it.
#[derive(Debug, Clone, Default)]
pub struct Layer {
    /// Which op.
    pub op: Op,
    /// Which convolution, into the graph's convolution table.
    pub conv: i64,
    /// The FUSED activation, for the convolutions only.
    pub act: Option<Act>,
    /// The activation's OWN parameters. They are not the defaults: the
    /// DETECTOR's HardSigmoid has alpha 0.2 and the RECOGNISER's has 1/6, and
    /// an activation that lost its alpha is a different function in the middle
    /// of the network.
    pub alpha: f64,
    /// The activation's beta.
    pub beta: f64,
    /// The stride, per axis.
    pub stride: [i64; 2],
    /// The padding: top, left, bottom, right.
    pub pad: [i64; 4],
    /// The dilation, per axis.
    pub dilation: [i64; 2],
    /// The kernel, per axis.
    pub kernel: [i64; 2],
    /// The transposed convolution's output padding.
    pub outpad: [i64; 2],
    /// The convolution's group count.
    pub group: i64,
    /// The resize's vertical scale.
    pub scale_h: f64,
    /// The resize's horizontal scale.
    pub scale_w: f64,
    /// The concat's or the softmax's axis.
    pub axis: i64,
    /// The reducemean's keepdims.
    pub keepdims: i64,
    /// The reshape's target shape.
    pub shape: Vec<i64>,
    /// The transpose's permutation.
    pub perm: Vec<i64>,
    /// The slice's, squeeze's, unsqueeze's or reducemean's axes.
    pub axes: Vec<i64>,
    /// The slice's starts.
    pub starts: Vec<i64>,
    /// The slice's ends.
    pub ends: Vec<i64>,
    /// The slice's steps.
    pub steps: Vec<i64>,
    /// The container tensor names: a convolution's weight and bias, a
    /// projection's weight and bias, an elementwise op's constant.
    pub w: String,
    /// The bias's name.
    pub b: String,
    /// The constant's name.
    pub c: String,
    /// The operands: an index into the walk's output list, or -1 for the image.
    pub inputs: Vec<i64>,
    /// The op's declared output extent, which the walk CHECKS against its own
    /// arithmetic.
    pub oshape: Vec<i64>,
    /// The array stream this convolution runs on, when the container was packed
    /// with the panels.
    pub stream: String,
}

/// One convolution's weights, as read from the container.
///
/// `group` is read from the OP LIST and not inferred from the weight's shape,
/// because the weight cannot tell a depthwise convolution from a dense one
/// that happens to have one input channel -- and a dense one dispatched
/// against a depthwise panel is a different network rather than an error.
#[derive(Debug, Clone, Default)]
pub struct ConvW {
    /// The convolution's index in its graph's table.
    pub index: i64,
    /// The output channels.
    pub cout: i64,
    /// The input channels.
    pub cin: i64,
    /// The kernel's height.
    pub kh: i64,
    /// The kernel's width.
    pub kw: i64,
    /// The group count.
    pub group: i64,
    /// The weights, `[cout, cin, kh, kw]` -- or `[cin, cout, kh, kw]` for a
    /// TRANSPOSED convolution, which is ONNX's own rule and the opposite of
    /// the other one.
    pub w: Vec<f32>,
    /// The bias, one per output channel, or none.
    pub b: Option<Vec<f32>>,
    /// Whether this is a transposed convolution.
    pub transposed: bool,
}

impl ConvW {
    /// Whether this is a depthwise convolution.
    pub fn depthwise(&self) -> bool {
        self.group > 1
    }
}

/// A named tensor the graph reads: an elementwise constant, a projection's
/// weight or bias. Read by NAME, because the op list names them.
#[derive(Debug, Clone, Default)]
pub struct Named {
    /// The tensor's shape.
    pub shape: Vec<i64>,
    /// The values.
    pub d: Vec<f32>,
}

// ---------------------------------------------------------------------------
// The geometry
// ---------------------------------------------------------------------------

/// Everything the container says about the model.
#[derive(Debug, Clone, Default)]
pub struct Geometry {
    // -- the two front ends
    /// The detector's input side, in pixels.
    pub det_input_size: i64,
    /// The recogniser's input height.
    pub rec_input_height: i64,
    /// The recogniser's input width.
    pub rec_input_width: i64,
    /// The detector's per-channel mean.
    pub det_mean: Vec<f32>,
    /// The detector's per-channel standard deviation.
    pub det_std: Vec<f32>,
    /// The recogniser's per-channel mean.
    pub rec_mean: Vec<f32>,
    /// The recogniser's per-channel standard deviation.
    pub rec_std: Vec<f32>,
    /// The pad values, in the space the pad happens in. The detector's is a
    /// uint8 raster value (0 = black) and the recogniser's is a NORMALISED
    /// value (0.0 = mid-grey), and they are not interchangeable.
    pub det_pad_value: f64,
    /// The recogniser's pad value.
    pub rec_pad_value: f64,

    // -- the detector's post-process
    /// The probability above which a pixel is text.
    pub det_thresh: f64,
    /// The mean probability a box must reach.
    pub det_box_thresh: f64,
    /// How far the box grows.
    pub det_unclip_ratio: f64,
    /// How many blobs are considered.
    pub det_max_candidates: i64,
    /// The smallest side a box may have.
    pub det_min_size: i64,

    // -- the recogniser's head
    /// How many classes the recogniser emits.
    pub rec_num_classes: i64,
    /// How many characters the dictionary holds.
    pub rec_dict_size: i64,
    /// The character table, in class order: 'blank' FIRST, the dictionary's
    /// characters in the file's order, and ' ' LAST. The blank is the CTC null
    /// class and the space is outside the file, and a decoder that read the
    /// file alone would map every class index one too low and turn every digit
    /// into the character before it.
    pub dict: Vec<String>,

    // -- the two graphs
    /// The detector's ops.
    pub det_graph: Vec<Layer>,
    /// The recogniser's ops.
    pub rec_graph: Vec<Layer>,
    /// The detector's convolutions.
    pub det_convs: Vec<ConvW>,
    /// The recogniser's convolutions.
    pub rec_convs: Vec<ConvW>,
    /// The detector's node names.
    pub det_nodes: Vec<String>,
    /// The recogniser's node names.
    pub rec_nodes: Vec<String>,
    /// The named tensors the graphs read, by name.
    pub consts: BTreeMap<String, Named>,
}

impl Geometry {
    /// How many of the detector's convolutions are DENSE.
    pub fn det_dense(&self) -> i64 {
        self.det_convs
            .iter()
            .filter(|c| !c.transposed && !c.depthwise())
            .count() as i64
    }

    /// How many of the detector's convolutions are DEPTHWISE.
    pub fn det_depthwise(&self) -> i64 {
        self.det_convs
            .iter()
            .filter(|c| !c.transposed && c.depthwise())
            .count() as i64
    }

    /// How many of the detector's convolutions are TRANSPOSED.
    pub fn det_transposed(&self) -> i64 {
        self.det_convs.iter().filter(|c| c.transposed).count() as i64
    }

    /// How many of the recogniser's convolutions are DENSE.
    pub fn rec_dense(&self) -> i64 {
        self.rec_convs
            .iter()
            .filter(|c| !c.transposed && !c.depthwise())
            .count() as i64
    }

    /// How many of the recogniser's convolutions are DEPTHWISE.
    pub fn rec_depthwise(&self) -> i64 {
        self.rec_convs
            .iter()
            .filter(|c| !c.transposed && c.depthwise())
            .count() as i64
    }

    /// How many of the recogniser's convolutions are TRANSPOSED.
    pub fn rec_transposed(&self) -> i64 {
        self.rec_convs.iter().filter(|c| c.transposed).count() as i64
    }
}

/// The dense convolutions of both graphs, as the array's list.
///
/// THE LIST IS THE FAMILY'S OWN, and it is built from the graphs rather
/// than from the container's directory: a panel exists for a convolution
/// the graph RUNS, and a panel that no op calls is a panel nobody stages.
///
/// Three kinds of convolution are left out, and each for its own reason:
///
/// * the DEPTHWISE ones, because a depthwise filter reduces WITHIN one
///   channel and has no `[M, N]` product to dispatch;
/// * the TRANSPOSED ones, because a transposed convolution's output is a
///   sum of shifted contributions and not a matrix product;
/// * the ones whose graph names no stream, which is what a container
///   packed WITHOUT the array panels looks like.
pub fn array_specs(geometry: &Geometry) -> Vec<ConvSpec> {
    let mut specs = Vec::new();
    for (graph, convs) in [
        (&geometry.det_graph, &geometry.det_convs),
        (&geometry.rec_graph, &geometry.rec_convs),
    ] {
        for l in graph {
            if l.op != Op::Conv || l.group != 1 || l.stream.is_empty() {
                continue;
            }
            let Some(cw) = convs.get(l.conv as usize) else {
                continue;
            };
            if cw.transposed || cw.depthwise() || cw.w.is_empty() {
                continue;
            }
            specs.push(ConvSpec {
                weight: l.w.clone(),
                stream: l.stream.clone(),
                k: (cw.cin * cw.kh * cw.kw) as usize,
                n: cw.cout as usize,
                bias: cw.b.clone(),
            });
        }
    }
    specs
}

// ---------------------------------------------------------------------------
// Reading the container
// ---------------------------------------------------------------------------

/// A refusal, in the oracle's words.
type Refusal = String;

fn need_i64(container: &Container, key: &str, label: &str) -> Result<i64, Refusal> {
    container
        .config()
        .get_i64(key)
        .ok_or_else(|| format!("{label}: '{key}' is not in the container"))
}

fn need_f64(container: &Container, key: &str, label: &str) -> Result<f64, Refusal> {
    container
        .config()
        .get_f64(key)
        .ok_or_else(|| format!("{label}: '{key}' is not in the container"))
}

fn need_doubles(
    container: &Container,
    key: &str,
    want: usize,
    label: &str,
) -> Result<Vec<f64>, Refusal> {
    let value = container
        .config()
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{label}: '{key}' is not an array"))?;
    let mut out = Vec::with_capacity(value.len());
    for e in value {
        out.push(
            e.as_f64()
                .ok_or_else(|| format!("{label}: '{key}' holds a non-number"))?,
        );
    }
    if want > 0 && out.len() != want {
        return Err(format!(
            "{label}: '{key}' has {} entries, expected {}",
            out.len(),
            want
        ));
    }
    Ok(out)
}

/// The container's config value, parsed as JSON.
///
/// The graph lists are stored as JSON STRINGS -- the packer writes them as
/// strings so that the config block stays a flat map of scalars -- so they
/// are parsed here rather than read as arrays.
fn need_json(container: &Container, key: &str, label: &str) -> Result<Value, Refusal> {
    let raw = container
        .config()
        .get_str(key)
        .ok_or_else(|| format!("{label}: '{key}' is not in the container"))?;
    serde_json::from_str(raw)
        .map_err(|e| format!("{label}: '{key}' is not readable JSON ({e}): {raw}"))
}

fn jint(node: &Value, key: &str, label: &str) -> Result<i64, Refusal> {
    node.get(key)
        .and_then(|v| v.as_i64())
        .ok_or_else(|| format!("{label}: no '{key}'"))
}

fn jdouble_or(node: &Value, key: &str, dflt: f64) -> f64 {
    node.get(key).and_then(|v| v.as_f64()).unwrap_or(dflt)
}

fn jstr(node: &Value, key: &str, label: &str) -> Result<String, Refusal> {
    node.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| format!("{label}: no '{key}'"))
}

fn jopt_str(node: &Value, key: &str, dflt: &str) -> String {
    node.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or(dflt)
        .to_string()
}

fn jpair(node: &Value, key: &str, label: &str) -> Result<[i64; 2], Refusal> {
    let value = node
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{label}: '{key}' is not a pair"))?;
    if value.len() != 2 {
        return Err(format!(
            "{label}: '{key}' is not a pair. Every stride, kernel and scale in \
             these graphs is per axis and both axes have to be stated."
        ));
    }
    Ok([
        value[0].as_i64().unwrap_or(0),
        value[1].as_i64().unwrap_or(0),
    ])
}

fn jpad4(node: &Value, label: &str) -> Result<[i64; 4], Refusal> {
    let value = node.get("pad").and_then(|v| v.as_array()).ok_or_else(|| {
        format!(
            "{label}: 'pad' is not four terms [top, left, bottom, right]. It is \
                 four and not two because the v5 detector spells two of its stem \
                 convolutions SAME_UPPER, which puts the extra row and column at the \
                 END -- a symmetric transcription of that is a 479-wide map where the \
                 checkpoint has a 480-wide one, and the stem's two branches then \
                 cannot be joined."
        )
    })?;
    if value.len() != 4 {
        return Err(format!(
            "{label}: 'pad' is not four terms [top, left, bottom, right]."
        ));
    }
    Ok([
        value[0].as_i64().unwrap_or(0),
        value[1].as_i64().unwrap_or(0),
        value[2].as_i64().unwrap_or(0),
        value[3].as_i64().unwrap_or(0),
    ])
}

fn jints(node: &Value, key: &str, label: &str) -> Result<Vec<i64>, Refusal> {
    let value = node
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{label}: '{key}' is not an array"))?;
    Ok(value.iter().map(|e| e.as_i64().unwrap_or(0)).collect())
}

/// The bytes of one tensor, widened to `f32`.
fn f32s(container: &Container, name: &str, label: &str) -> Result<Vec<f32>, Refusal> {
    let bytes = container
        .tensor_bytes(name)
        .map_err(|_| format!("{label}: the container holds no tensor called '{name}'"))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{label}: '{name}' is {} bytes, which is not a whole number of fp32 values",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// One convolution's weights, as read from the container.
///
/// `wdtype` is the container's `conv_weight_dtype`. Whatever it says, the byte
/// count is checked against what that dtype implies for this shape BEFORE
/// anything is read: a payload that contradicts its declared precision is not a
/// weight at a different precision, it is a number of the wrong size.
fn read_conv(
    container: &Container,
    pfx: &str,
    i: i64,
    wdtype: &str,
    i4_group: i64,
    transposed: bool,
    label: &str,
) -> Result<ConvW, Refusal> {
    let base = format!("{pfx}.conv.{i}");
    let entry = container
        .directory()
        .tensor(&format!("{base}.w"))
        .ok_or_else(|| format!("{label}: the container holds no '{base}.w'"))?;
    if entry.logical_shape.len() != 4 {
        return Err(format!("{label}: {base}.w is not a 4-D filter"));
    }
    // THE TWO LAYOUTS, AND THEY ARE OPPOSITE.
    //
    // A CONVOLUTION's weight is `[Cout, Cin, kh, kw]` and a TRANSPOSED
    // CONVOLUTION's is `[Cin, Cout, kh, kw]` -- the INPUT channels first,
    // which is ONNX's own rule and the opposite of the other one. A reader
    // that took the first extent for the output count in both cases builds the
    // DB head's two upsampling convolutions with 24 output channels where the
    // checkpoint has one, and the probability map comes out 24 times too wide.
    let mut c = ConvW {
        index: i,
        transposed,
        ..ConvW::default()
    };
    if transposed {
        c.cin = entry.logical_shape[0];
        c.cout = entry.logical_shape[1];
    } else {
        c.cout = entry.logical_shape[0];
        c.cin = entry.logical_shape[1];
    }
    c.kh = entry.logical_shape[2];
    c.kw = entry.logical_shape[3];
    let want = (c.cout * c.cin * c.kh * c.kw) as usize;
    let n = c.cout as usize;
    let k = (c.cin * c.kh * c.kw) as usize;
    let bytes = container
        .tensor_bytes(&format!("{base}.w"))
        .map_err(|_| format!("{label}: the container holds no '{base}.w'"))?;
    match wdtype {
        "f32" => {
            if bytes.len() != want * 4 {
                return Err(format!(
                    "{label}: {base}.w is {} bytes, which is not {} fp32 values for a \
                     [{}, {}, {}, {}] weight.",
                    bytes.len(),
                    want * 4,
                    c.cout,
                    c.cin,
                    c.kh,
                    c.kw
                ));
            }
            c.w = f32s(container, &format!("{base}.w"), label)?;
        }
        "bf16" => {
            if bytes.len() != want * 2 {
                return Err(format!(
                    "{label}: {base}.w is {} bytes, which is not {} for {} bf16 values.",
                    bytes.len(),
                    want * 2,
                    want
                ));
            }
            c.w = bytes
                .chunks_exact(2)
                .map(|b| npum_format::bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect();
        }
        _ => {
            // int8: one byte per weight. int4: one byte per TWO weights, and K
            // may be ODD -- these kernels are 3x3xC -- so the payload is
            // ceil(K/2) bytes per row with a zero pad nibble at the end. ceil,
            // not K/2: computing it the other way makes the last nibble of the
            // last row a phantom weight.
            let packed = if wdtype == "i8" { k } else { (k + 1) / 2 };
            if bytes.len() != n * packed {
                return Err(format!(
                    "{label}: {base}.w is {} bytes, which is not {} for a [{}, {}] {} weight.",
                    bytes.len(),
                    n * packed,
                    n,
                    k,
                    wdtype
                ));
            }
            let scales = f32s(container, &format!("{base}.wscale"), label).map_err(|_| {
                format!(
                    "{label}: {base}.wscale is not in the container, and {base}.w says its \
                     weights are {wdtype}. An integer payload without its scale is bytes \
                     whose meaning is one multiply away and not present in the file."
                )
            })?;
            if scales.len() != n {
                return Err(format!(
                    "{label}: {base}.wscale is for {} output channels and the filter has {} \
                     rows. The scale is per output channel -- one per row of this weight \
                     -- and a scale count that disagrees with the row count puts every \
                     channel after the last one on another channel's factor.",
                    scales.len(),
                    n
                ));
            }
            let mut w = vec![0.0f32; want];
            if wdtype == "i8" {
                for row in 0..n {
                    for col in 0..k {
                        w[row * k + col] = bytes[row * k + col] as i8 as f32 * scales[row];
                    }
                }
            } else {
                let g = if i4_group > 0 {
                    (k as i64 + i4_group - 1) / i4_group
                } else {
                    1
                } as usize;
                let group_scales =
                    f32s(container, &format!("{base}.gscale"), label).map_err(|_| {
                        format!(
                            "{label}: {base}.gscale is not in the container, and {base}.w says \
                             its weights are i4 with a group of {i4_group}."
                        )
                    })?;
                if group_scales.len() != g * n {
                    return Err(format!(
                        "{label}: {base}.gscale is for {} groups and the filter needs {}.",
                        group_scales.len(),
                        g * n
                    ));
                }
                for row in 0..n {
                    for col in 0..k {
                        // Four bits, low nibble first, SIGN-EXTENDED. A nibble of
                        // 0xF is -1 and not 15: reading them unsigned maps the most
                        // negative weight onto the most positive one, in every group.
                        let byte = bytes[row * ((k + 1) / 2) + col / 2];
                        let mut v = if col & 1 != 0 {
                            ((byte >> 4) & 0x0F) as i32
                        } else {
                            (byte & 0x0F) as i32
                        };
                        if v >= 8 {
                            v -= 16;
                        }
                        let gsel = if i4_group > 0 {
                            col / i4_group as usize
                        } else {
                            0
                        };
                        w[row * k + col] = v as f32 * scales[row] * group_scales[gsel * n + row];
                    }
                }
            }
            c.w = w;
        }
    }
    if container.directory().tensor(&format!("{base}.b")).is_some() {
        c.b = Some(f32s(container, &format!("{base}.b"), label)?);
    }
    Ok(c)
}

/// One graph, read out of the container.
struct GraphRead {
    layers: Vec<Layer>,
    convs: Vec<ConvW>,
    nodes: Vec<String>,
}

/// Read one graph.
///
/// `pfx` namespaces its tensors, and it is not decoration: both graphs number
/// their convolutions from zero, so an unprefixed lookup reads the OTHER
/// graph's weights.
fn read_one_graph(
    container: &Container,
    pfx: &str,
    graph_key: &str,
    nodes_key: &str,
    num_convs: i64,
    wdtype: &str,
    i4_group: i64,
    label: &str,
) -> Result<GraphRead, Refusal> {
    let ops = need_json(container, graph_key, label)?;
    let names = need_json(container, nodes_key, label)?;
    let ops = ops
        .as_array()
        .ok_or_else(|| format!("{label}: {graph_key} is not an array"))?;
    let names = names
        .as_array()
        .ok_or_else(|| format!("{label}: {nodes_key} is not an array"))?;
    if ops.len() != names.len() {
        return Err(format!(
            "{label}: {graph_key} has {} nodes and {nodes_key} has {}. The second list is \
             what a refusal quotes by name, and a refusal that quotes node 7 when node 7 \
             does not exist is worse than none.",
            ops.len(),
            names.len()
        ));
    }

    let mut layers: Vec<Layer> = Vec::with_capacity(ops.len());
    let mut node_names: Vec<String> = Vec::with_capacity(ops.len());
    let mut max_conv: i64 = -1;
    for (n, o) in ops.iter().enumerate() {
        let nl = format!(
            "{label}/{graph_key}: node '{}' [{n}]",
            names[n].as_str().unwrap_or("?")
        );
        let mut l = Layer {
            op: Op::from_name(&jstr(o, "op", &nl)?).ok_or_else(|| {
                format!(
                    "{nl}: unknown op '{}' in the packed graph. This build walks conv, \
                     convtranspose, relu, sigmoid, hardswish, hardsigmoid, add, sub, mul, \
                     div, sqrt, concat, maxpool, avgpool, gapool, resize, reshape, \
                     transpose, slice, squeeze, unsqueeze, matmul, softmax and reducemean. \
                     An op that is not in that list is a network this runtime cannot \
                     compute, and a transcription that comes out of it would be a plausible \
                     wrong answer.",
                    jstr(o, "op", &nl).unwrap_or_default()
                )
            })?,
            ..Layer::default()
        };
        l.alpha = jdouble_or(o, "alpha", 1.0 / 6.0);
        l.beta = jdouble_or(o, "beta", 0.5);
        let act = jopt_str(o, "act", "");
        // "none" IS the absence of an activation, and the packer writes it as a
        // word rather than leaving the key out.
        if !act.is_empty() && act != "none" {
            l.act = Some(Act::from_name(&act).ok_or_else(|| {
                format!(
                    "{nl}: the fused activation is '{act}'. This runtime computes relu, \
                     sigmoid, hardswish and hardsigmoid, and an activation it does not have \
                     is a network it cannot run."
                )
            })?);
        }
        match l.op {
            Op::Conv | Op::ConvTranspose => {
                l.conv = jint(o, "conv", &nl)?;
                if l.conv < 0 {
                    return Err(format!("{nl}: no 'conv' index"));
                }
                if l.conv >= num_convs {
                    return Err(format!(
                        "{nl}: conv {} is past the container's {num_convs} convolutions for \
                         this graph",
                        l.conv
                    ));
                }
                max_conv = max_conv.max(l.conv);
                l.pad = jpad4(o, &nl)?;
                l.stride = jpair(o, "stride", &nl)?;
                if let Some(dil) = o.get("dilation").and_then(|v| v.as_array()) {
                    if dil.len() == 2 {
                        l.dilation = [dil[0].as_i64().unwrap_or(1), dil[1].as_i64().unwrap_or(1)];
                    }
                }
                if let Some(group) = o.get("group").and_then(|v| v.as_i64()) {
                    l.group = group;
                }
                if let Some(op) = o.get("outpad").and_then(|v| v.as_array()) {
                    if op.len() == 2 {
                        l.outpad = [op[0].as_i64().unwrap_or(0), op[1].as_i64().unwrap_or(0)];
                    }
                }
                l.w = jopt_str(o, "w", "");
                l.b = jopt_str(o, "b", "");
                l.stream = jopt_str(o, "stream", "");
            }
            Op::MaxPool | Op::AvgPool => {
                l.kernel = jpair(o, "kernel", &nl)?;
                l.stride = jpair(o, "stride", &nl)?;
                l.pad = jpad4(o, &nl)?;
            }
            Op::Resize => {
                let s = o.get("scale_h").and_then(|v| v.as_f64());
                let t = o.get("scale_w").and_then(|v| v.as_f64());
                let (s, t) = match (s, t) {
                    (Some(s), Some(t)) => (s, t),
                    _ => return Err(format!("{nl}: no 'scale_h'/'scale_w'")),
                };
                if s <= 0.0 || t <= 0.0 {
                    return Err(format!("{nl}: resize has a non-positive scale"));
                }
                l.scale_h = s;
                l.scale_w = t;
            }
            Op::Concat | Op::Softmax => {
                l.axis = jint(o, "axis", &nl)?;
            }
            Op::Reshape => {
                l.shape = jints(o, "shape", &nl)?;
            }
            Op::Transpose => {
                l.perm = jints(o, "perm", &nl)?;
            }
            Op::Slice => {
                l.axes = jints(o, "axes", &nl)?;
                l.starts = jints(o, "starts", &nl)?;
                l.ends = jints(o, "ends", &nl)?;
                l.steps = jints(o, "steps", &nl)?;
            }
            Op::Squeeze | Op::Unsqueeze => {
                l.axes = jints(o, "axes", &nl)?;
            }
            Op::ReduceMean => {
                l.axes = jints(o, "axes", &nl)?;
                if let Some(kd) = o.get("keepdims").and_then(|v| v.as_i64()) {
                    l.keepdims = kd;
                }
            }
            Op::Add | Op::Sub | Op::Mul | Op::Div => {
                l.c = jopt_str(o, "c", "");
            }
            Op::MatMul => {
                l.w = jopt_str(o, "w", "");
                l.b = jopt_str(o, "b", "");
            }
            _ => {}
        }
        // The op's declared output extent, which the walk CHECKS.
        let out = o.get("oshape").and_then(|v| v.as_array()).ok_or_else(|| {
            format!(
                "{nl}: no 'oshape'. The packer resolved every extent at pack time, and a \
                 graph without them is a graph whose arithmetic this runtime would have to \
                 guess."
            )
        })?;
        l.oshape = out.iter().map(|e| e.as_i64().unwrap_or(0)).collect();
        let inputs = o
            .get("inputs")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("{nl}: no 'inputs'"))?;
        l.inputs = inputs.iter().map(|e| e.as_i64().unwrap_or(-1)).collect();
        if l.inputs.is_empty() {
            return Err(format!("{nl}: no 'inputs'"));
        }
        for input in &l.inputs {
            if *input != -1 && (*input as usize) >= n {
                return Err(format!(
                    "{nl}: takes operand {input}, which is not an earlier op and not the \
                     image (-1). This graph is evaluated in order, so a forward reference is \
                     not a cycle the runtime could resolve -- it is a graph whose outputs it \
                     cannot compute."
                ));
            }
        }
        layers.push(l);
        node_names.push(names[n].as_str().unwrap_or("?").to_string());
    }

    // WHICH CONVOLUTIONS ARE TRANSPOSED, taken from the op list. The weight's
    // own shape cannot say: a transposed convolution's `[Cin, Cout, kh, kw]`
    // and a dense one's `[Cout, Cin, kh, kw]` are the same four numbers in a
    // different order, and reading them the same way is a head of the wrong
    // width.
    let n_conv = num_convs.max(max_conv + 1).max(0) as usize;
    let mut transposed = vec![false; n_conv];
    for l in &layers {
        if l.op != Op::Conv && l.op != Op::ConvTranspose {
            continue;
        }
        if l.conv >= 0 && (l.conv as usize) < transposed.len() {
            transposed[l.conv as usize] = l.op == Op::ConvTranspose;
        }
    }

    let mut convs: Vec<ConvW> = vec![ConvW::default(); n_conv];
    for i in 0..n_conv {
        let base = format!("{pfx}.conv.{i}");
        if container.directory().tensor(&format!("{base}.w")).is_none() {
            continue; // a sparse index
        }
        convs[i] = read_conv(
            container,
            pfx,
            i as i64,
            wdtype,
            i4_group,
            transposed[i],
            label,
        )?;
    }
    // THE GROUP, TAKEN FROM THE OP AND NOT FROM THE WEIGHT.
    //
    // A depthwise convolution is group == Cin with one input channel per
    // group, and its weight reads `[Cin, 1, kh, kw]` -- which is the same
    // shape as a DENSE convolution with one input channel. The weight cannot
    // tell the two apart, and the difference is not a detail: a depthwise
    // filter reduces WITHIN one channel and has no `[M, N]` GEMM in it.
    let mut seen = vec![false; convs.len()];
    for l in &layers {
        if l.op != Op::Conv && l.op != Op::ConvTranspose {
            continue;
        }
        let cw = &mut convs[l.conv as usize];
        if cw.w.is_empty() {
            return Err(format!(
                "{label}: the graph names conv {} and the container has no weights for it.",
                l.conv
            ));
        }
        if seen[l.conv as usize] {
            return Err(format!(
                "{label}: conv {} is called by two graph nodes. The array path keys panels \
                 by which node calls a convolution, and two nodes sharing one would stage \
                 one panel for two different layers.",
                l.conv
            ));
        }
        seen[l.conv as usize] = true;
        cw.group = l.group;
    }
    Ok(GraphRead {
        layers,
        convs,
        nodes: node_names,
    })
}

/// Read both graphs, their weights, the constants and the dictionary.
///
/// `container` must outlive the returned Geometry, whose weights are read
/// into it rather than pointed at.
pub fn read_geometry(container: &Container, label: &str) -> Result<Geometry, Refusal> {
    let arch = container.config_str("arch").unwrap_or_default();
    if arch != ARCH {
        return Err(format!(
            "{label}: this is an arch={arch} container and this build reads arch={ARCH}. \
             The OCR family holds a text DETECTOR, a text RECOGNISER and the character \
             dictionary between them, and none of those three is what another \
             architecture's container holds."
        ));
    }
    let mut g = Geometry::default();
    g.det_input_size = need_i64(container, "det_input_size", label)?;
    g.rec_input_height = need_i64(container, "rec_input_height", label)?;
    g.rec_input_width = need_i64(container, "rec_input_width", label)?;
    if g.det_input_size <= 0 || g.rec_input_height <= 0 || g.rec_input_width <= 0 {
        return Err(format!(
            "{label}: the container declares a front end of {}px and {}x{}, and a size of \
             zero is not a size.",
            g.det_input_size, g.rec_input_height, g.rec_input_width
        ));
    }

    // -- the two normalisations, and the two pad values
    {
        let norm = |mkey: &str, skey: &str, who: &str| -> Result<(Vec<f32>, Vec<f32>), Refusal> {
            let m = need_doubles(container, mkey, 3, label)?;
            let s = need_doubles(container, skey, 3, label)?;
            if s.iter().any(|v| *v == 0.0) {
                return Err(format!(
                    "{label}: {skey} has a zero. Both front ends normalise by it, so that is \
                     a division by zero on every pixel of every frame. ({who})"
                ));
            }
            Ok((
                m.iter().map(|v| *v as f32).collect(),
                s.iter().map(|v| *v as f32).collect(),
            ))
        };
        let (m, s) = norm("det_image_mean", "det_image_std", "detector")?;
        g.det_mean = m;
        g.det_std = s;
        let (m, s) = norm("rec_image_mean", "rec_image_std", "recogniser")?;
        g.rec_mean = m;
        g.rec_std = s;
    }
    g.det_pad_value = need_f64(container, "det_pad_value", label)?;
    g.rec_pad_value = need_f64(container, "rec_pad_value", label)?;

    // -- the detector's post-process constants
    g.det_thresh = need_f64(container, "det_thresh", label)?;
    g.det_box_thresh = need_f64(container, "det_box_thresh", label)?;
    g.det_unclip_ratio = need_f64(container, "det_unclip_ratio", label)?;
    g.det_max_candidates = need_i64(container, "det_max_candidates", label)?;
    g.det_min_size = need_i64(container, "det_min_size", label)?;
    if g.det_thresh < 0.0 || g.det_thresh > 1.0 || g.det_box_thresh < 0.0 || g.det_box_thresh > 1.0
    {
        return Err(format!(
            "{label}: the detector's thresholds are {} and {}, and a threshold outside 0..1 is \
             not one.",
            g.det_thresh, g.det_box_thresh
        ));
    }

    // -- the recogniser's head
    g.rec_num_classes = need_i64(container, "rec_num_classes", label)?;
    g.rec_dict_size = need_i64(container, "rec_dict_size", label)?;
    if g.rec_num_classes != g.rec_dict_size + 2 {
        return Err(format!(
            "{label}: the container declares {} classes over a dictionary of {} characters. \
             The CTC table is the dictionary's characters plus the blank and the space, so \
             the two must differ by exactly two: a decoder that reads one table and a \
             network that learned another produce text that is confidently wrong.",
            g.rec_num_classes, g.rec_dict_size
        ));
    }

    // -- the character table
    //
    // The container stores the dictionary's BYTES -- the characters joined by
    // newlines -- and the blank and the space are added HERE, by the decoder,
    // because they are not in the file and a dictionary that carried its own
    // blank would be a second place to keep the two in step.
    let dict_bytes = container
        .tensor_bytes("rec_dict")
        .map_err(|_| {
            format!(
                "{label}: rec_dict is not in the container. The recogniser emits one logit \
                 per CLASS and the classes are the dictionary's characters; a recogniser \
                 without them transcribes noise."
            )
        })?
        .to_vec();
    let text = String::from_utf8_lossy(&dict_bytes).to_string();
    let mut lines: Vec<String> = Vec::new();
    for line in text.split('\n') {
        lines.push(line.strip_suffix('\r').unwrap_or(line).to_string());
    }
    // The oracle's split does not produce a trailing empty line, and neither
    // does this one: the file ends with a newline.
    if lines.last().map(|l| l.is_empty()).unwrap_or(false) {
        lines.pop();
    }
    g.dict.reserve(lines.len() + 2);
    g.dict.push("blank".to_string()); // the CTC null class, FIRST
    for line in lines {
        g.dict.push(line);
    }
    g.dict.push(" ".to_string()); // the space, LAST
    if g.dict.len() as i64 != g.rec_num_classes {
        return Err(format!(
            "{label}: the dictionary holds {} characters, so the CTC table is {} classes with \
             the blank and the space, and the container declares {}.",
            g.dict.len() - 2,
            g.dict.len(),
            g.rec_num_classes
        ));
    }

    // -- the weight precision, and the int4 group
    let wdtype = container
        .config_str("conv_weight_dtype")
        .unwrap_or("f32")
        .to_string();
    if wdtype != "f32" && wdtype != "bf16" && wdtype != "i8" && wdtype != "i4" {
        return Err(format!(
            "{label}: conv_weight_dtype is '{wdtype}', and this runtime reads f32, bf16, i8 \
             and i4."
        ));
    }
    let i4_group = container.config().get_i64("conv_int4_group").unwrap_or(0);
    if i4_group > 0 && wdtype != "i4" {
        return Err(format!(
            "{label}: conv_int4_group is {i4_group} and conv_weight_dtype is '{wdtype}'."
        ));
    }

    // -- the two graphs and their weights
    {
        let det_num = need_i64(container, "det_num_convs", label)?;
        let rec_num = need_i64(container, "rec_num_convs", label)?;
        let det = read_one_graph(
            container,
            "det",
            "det_graph",
            "det_graph_nodes",
            det_num,
            &wdtype,
            i4_group,
            &format!("{label}/det"),
        )?;
        let rec = read_one_graph(
            container,
            "rec",
            "rec_graph",
            "rec_graph_nodes",
            rec_num,
            &wdtype,
            i4_group,
            &format!("{label}/rec"),
        )?;
        g.det_graph = det.layers;
        g.det_nodes = det.nodes;
        g.det_convs = det.convs;
        g.rec_graph = rec.layers;
        g.rec_nodes = rec.nodes;
        g.rec_convs = rec.convs;
    }

    // -- the named tensors: the elementwise constants and the projections
    //
    // They are read by NAME, because the op list names them, and the names are
    // the packer's own and ALREADY carry the graph's prefix: "det.const.0" for
    // a constant and "rec.linear.0.w" for a projection. The prefix is not
    // decoration -- the detector's constant 0 and the recogniser's constant 0
    // are different numbers.
    {
        let mut want: Vec<String> = Vec::new();
        for l in g.det_graph.iter().chain(g.rec_graph.iter()) {
            if !l.c.is_empty() && !want.contains(&l.c) {
                want.push(l.c.clone());
            }
            if l.op == Op::MatMul {
                if !l.w.is_empty() && !want.contains(&l.w) {
                    want.push(l.w.clone());
                }
                if !l.b.is_empty() && !want.contains(&l.b) {
                    want.push(l.b.clone());
                }
            }
        }
        want.sort();
        for name in want {
            let bytes = container.tensor_bytes(&name).map_err(|_| {
                format!(
                    "{label}: the graph names the tensor '{name}', which is not in the container."
                )
            })?;
            if bytes.len() % 4 != 0 {
                return Err(format!(
                    "{label}: {name} is {} bytes, which is not a whole number of fp32 values.",
                    bytes.len()
                ));
            }
            let shape = container
                .directory()
                .tensor(&name)
                .map(|t| t.logical_shape.clone())
                .unwrap_or_default();
            g.consts.insert(
                name,
                Named {
                    shape,
                    d: bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                },
            );
        }
    }

    Ok(g)
}

// ---------------------------------------------------------------------------
// The walk
// ---------------------------------------------------------------------------

/// What one run cost.
#[derive(Debug, Clone, Default)]
pub struct Cost {
    /// The dense convolutions that ran on the host.
    pub convs_host: i64,
    /// The dense convolutions that ran on the array.
    pub convs_array: i64,
    /// The depthwise convolutions that ran on the host.
    pub dwconvs_host: i64,
    /// The transposed convolutions that ran on the host.
    pub transposed_convs: i64,
    /// The elementwise passes.
    pub elementwise_ops: i64,
    /// The host's share of the dense convolutions.
    pub t_host: f64,
    /// The array's share of the dense convolutions.
    ///
    /// The array's own time, measured around the call: the im2col and the
    /// transpose back to NCHW stay on the host either way, and they are
    /// counted as the host's, because that is where they happen.
    pub t_array: f64,
    /// The depthwise convolutions' own time.
    pub t_dwconv: f64,
    /// The im2col's own time.
    pub t_im2col: f64,
    /// The GEMMs' own time.
    pub t_gemm: f64,
    /// The transpose back to NCHW.
    pub t_transpose: f64,
    /// The transposed convolutions' own time.
    pub t_transposed: f64,
    /// The elementwise passes' own time.
    pub t_elementwise: f64,
}

impl Cost {
    /// Forget everything.
    pub fn reset(&mut self) {
        *self = Cost::default();
    }
}

/// One activation, applied to one element.
///
/// The arithmetic is the oracle's, and the PROMOTION is part of it: the
/// oracle computes `v * alpha + beta` in DOUBLE and converts once at the end,
/// so the intermediate is not rounded to f32.
fn act1(act: Act, alpha: f64, beta: f64, v: f32) -> f32 {
    match act {
        Act::Relu => {
            if v > 0.0 {
                v
            } else {
                0.0
            }
        }
        Act::Sigmoid => {
            // The oracle's `std::exp` on a float argument is the SINGLE
            // precision one, and the `1.0 +` and the division around it are
            // DOUBLE. So the exponent is an f32 and the arithmetic around it
            // is an f64 -- and the result is converted once.
            let e = (-v).exp();
            (1.0f64 / (1.0f64 + e as f64)) as f32
        }
        Act::HardSwish => {
            // `v * alpha + beta` IS A FUSED MULTIPLY-ADD IN THE ORACLE, and
            // that is not a detail: the compiler contracts it into one
            // rounding (the object file says `vfmadd132sd`), and this is the
            // one place where the rounding COUNT decides an answer.
            //
            // The detector's stem hardswish lands on a value whose product
            // `v * c` is EXACTLY halfway between two f32 neighbours -- 25
            // bits of mantissa in a 24-bit field -- and the two roundings
            // differ by 2.8e-17, which is enough to pick the other
            // neighbour. One element of 1 843 200 changed, and the box it
            // fed changed with it: a confidence that came out 0.986 where
            // the oracle's said 0.973.
            let t = (v as f64).mul_add(alpha, beta);
            let c = if t < 0.0 {
                0.0
            } else if t > 1.0 {
                1.0
            } else {
                t
            };
            (v as f64 * c) as f32
        }
        Act::HardSigmoid => {
            // The same contraction, for the same reason.
            let t = (v as f64).mul_add(alpha, beta);
            (if t < 0.0 {
                0.0
            } else if t > 1.0 {
                1.0
            } else {
                t
            }) as f32
        }
    }
}

/// Apply a layer's activation to a whole tensor.
fn apply_act(t: &mut Tensor, l: &Layer) {
    let Some(act) = l.act else { return };
    for v in t.d.iter_mut() {
        *v = act1(act, l.alpha, l.beta, *v);
    }
}

/// numpy's broadcasting: the shapes are right-aligned and an axis of extent 1
/// stretches. The output extent is the larger of each pair.
fn broadcast_shape(a: &[i64], b: &[i64]) -> Result<Vec<i64>, Refusal> {
    let ra = a.len();
    let rb = b.len();
    let r = ra.max(rb);
    let mut out = vec![1i64; r];
    for i in 0..r {
        let av = if i < r - ra { 1 } else { a[i - (r - ra)] };
        let bv = if i < r - rb { 1 } else { b[i - (r - rb)] };
        if av != bv && av != 1 && bv != 1 {
            return Err(format!(
                "cannot broadcast [{}] against [{}]",
                a.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                b.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        out[i] = av.max(bv);
    }
    Ok(out)
}

/// The strides of ONE operand against an output shape: 0 on an axis the
/// operand does not have or whose extent is 1, and the operand's own row
/// stride otherwise.
///
/// THE AXES AND THE STRIDES ADVANCE TOGETHER, from the last one backwards:
/// the last axis of the operand has stride 1, the one before it has the
/// stride of the last axis's extent, and so on.
///
/// They are two different counters and pairing them wrongly -- taking the
/// stride of the LAST axis and giving it to the FIRST -- produces a tensor
/// of the right shape whose elements are read from the wrong offsets.
fn bstride(out: &[i64], input: &[i64]) -> Vec<i64> {
    let r = out.len();
    let ri = input.len();
    let mut s = vec![0i64; r];
    let mut acc = 1i64;
    for i in (0..ri).rev() {
        let axis = r - ri + i;
        if input[i] != 1 {
            s[axis] = acc;
            acc *= input[i];
        }
    }
    s
}

/// One elementwise pass over two operands, broadcast against each other.
///
/// THE ODOMETER, NOT A MULTI-INDEX PER ELEMENT. The strides are worked out
/// once and the offsets are advanced as the output's index runs, so the inner
/// loop is two adds and a compare.
fn broadcast_apply(
    oshape: &[i64],
    ashape: &[i64],
    a: &[f32],
    bshape: &[i64],
    b: &[f32],
    op: Op,
) -> Vec<f32> {
    // The op is a RUNTIME value, and a match on it inside the element loop
    // is a branch per element -- a branch the predictor gets right and the
    // compiler cannot hoist. Four monomorphised copies of the loop do not
    // have that branch at all, and the inner loop becomes a plain add or
    // multiply over contiguous memory.
    match op {
        Op::Add => broadcast_apply_with(oshape, ashape, a, bshape, b, |x, y| x + y),
        Op::Sub => broadcast_apply_with(oshape, ashape, a, bshape, b, |x, y| x - y),
        Op::Mul => broadcast_apply_with(oshape, ashape, a, bshape, b, |x, y| x * y),
        _ => broadcast_apply_with(oshape, ashape, a, bshape, b, |x, y| x / y),
    }
}

/// One elementwise pass over two operands, with the operation FIXED.
fn broadcast_apply_with<F: Fn(f32, f32) -> f32>(
    oshape: &[i64],
    ashape: &[i64],
    a: &[f32],
    bshape: &[i64],
    b: &[f32],
    f: F,
) -> Vec<f32> {
    let as_ = bstride(oshape, ashape);
    let bs = bstride(oshape, bshape);
    let r = oshape.len();
    let total: i64 = oshape.iter().product();
    let mut out = vec![0.0f32; total as usize];
    if r == 0 || total == 0 {
        return out;
    }
    // THE FAST PATH: the right-hand side is CONSTANT over a run.
    //
    // The odometer below is the general case, and it costs a compare and two
    // adds per ELEMENT. The shapes that actually occur are simpler: the
    // constant is a per-CHANNEL bias ([1, C, 1, 1]), a scalar, or the
    // operand's own shape -- and in all three the right-hand side moves on at
    // most one axis, so the output's last axis is a RUN of elements that
    // share it. The odometer then costs a compare per RUN instead of per
    // element, and the inner loop is a plain add or multiply over
    // contiguous memory.
    //
    // The detector's probability map is five and a half million elements and
    // there are twenty of these passes over it, so this is the difference
    // between the map being computed in thirty milliseconds and in three.
    //
    // The ARITHMETIC is one add or one multiply per element either way, so
    // this changes the speed and not the bits.
    let last = r - 1;
    let run = oshape[last];
    let moving = (0..r).filter(|k| bs[*k] != 0).count();
    if run > 0 && moving <= 1 && total % run == 0 {
        let runs = total / run;
        let a_last = as_[last];
        let b_last = bs[last];
        let mut idx = vec![0i64; r];
        let mut aoff = 0i64;
        let mut boff = 0i64;
        for run_index in 0..runs {
            let base = (run_index * run) as usize;
            if a_last == 1 && b_last == 0 {
                // THE COMMON CASE: a contiguous run of the operand and ONE
                // constant on the right -- a per-channel bias or a scalar.
                let y = b[boff as usize];
                let src = &a[aoff as usize..aoff as usize + run as usize];
                let dst = &mut out[base..base + run as usize];
                for i in 0..run as usize {
                    dst[i] = f(src[i], y);
                }
            } else if a_last == 1 && b_last == 1 {
                // Both sides contiguous: a straight elementwise pass.
                let src = &a[aoff as usize..aoff as usize + run as usize];
                let rhs = &b[boff as usize..boff as usize + run as usize];
                let dst = &mut out[base..base + run as usize];
                for i in 0..run as usize {
                    dst[i] = f(src[i], rhs[i]);
                }
            } else {
                for i in 0..run {
                    let x = a[(aoff + i * a_last) as usize];
                    let y = b[(boff + i * b_last) as usize];
                    out[base + i as usize] = f(x, y);
                }
            }
            // The odometer, over the OUTER axes only: the last axis is the
            // run, and it starts again at every run.
            let mut k = r - 1;
            while k > 0 {
                k -= 1;
                idx[k] += 1;
                aoff += as_[k];
                boff += bs[k];
                if idx[k] < oshape[k] {
                    break;
                }
                aoff -= as_[k] * oshape[k];
                boff -= bs[k] * oshape[k];
                idx[k] = 0;
            }
        }
        return out;
    }
    let mut idx = vec![0i64; r];
    let mut aoff = 0i64;
    let mut boff = 0i64;
    for n in 0..total {
        let (x, y) = (a[aoff as usize], b[boff as usize]);
        out[n as usize] = f(x, y);
        let mut k = r;
        while k > 0 {
            k -= 1;
            idx[k] += 1;
            aoff += as_[k];
            boff += bs[k];
            if idx[k] < oshape[k] {
                break;
            }
            aoff -= as_[k] * oshape[k];
            boff -= bs[k] * oshape[k];
            idx[k] = 0;
        }
    }
    out
}

/// The stride of one axis of a row-major shape.
fn row_stride(shape: &[i64], axis: usize) -> i64 {
    let mut acc = 1i64;
    for i in (axis + 1)..shape.len() {
        acc *= shape[i];
    }
    acc
}

/// The shape, as the oracle's `shape_str` writes it.
fn shape_str(shape: &[i64]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// The convolution scratch, split out of `Network` for one walk.
///
/// Keeping it separate lets `step` borrow the WEIGHTS by reference while a
/// convolution mutates the im2col/GEMM buffers. Previously the borrow checker
/// forced a full clone of the weight vector and of the `[K, N]` table on
/// every convolution -- megabytes copied per layer of every image. The
/// arithmetic is untouched; only the borrows change.
struct ConvScratch {
    a: Vec<f32>,
    c: Vec<f32>,
    cost: Cost,
    threads: usize,
}

impl ConvScratch {
    /// Merge the convolution timings back into the network's public cost.
    fn merge_cost(&self, cost: &mut Cost) {
        cost.convs_host += self.cost.convs_host;
        cost.convs_array += self.cost.convs_array;
        cost.dwconvs_host += self.cost.dwconvs_host;
        cost.transposed_convs += self.cost.transposed_convs;
        cost.t_host += self.cost.t_host;
        cost.t_array += self.cost.t_array;
        cost.t_dwconv += self.cost.t_dwconv;
        cost.t_im2col += self.cost.t_im2col;
        cost.t_gemm += self.cost.t_gemm;
        cost.t_transpose += self.cost.t_transpose;
        cost.t_transposed += self.cost.t_transposed;
        cost.t_elementwise += self.cost.t_elementwise;
        cost.elementwise_ops += self.cost.elementwise_ops;
    }
}

/// One packed graph, ready to walk.
pub struct Network {
    /// The graph's ops.
    graph: Vec<Layer>,
    /// The graph's convolutions.
    convs: Vec<ConvW>,
    /// The `[K, N]` weight tables, one per DENSE convolution.
    ///
    /// `gemm_nt` reads `B[k*N + n]`, so the filter has to be `[K, N]` and the
    /// container stores it as `[Cout, Cin, kh, kw] = [N, K]`. The transpose
    /// happens ONCE, at load: the filter is the same on every frame.
    wmat: Vec<Option<Vec<f32>>>,
    /// The named tensors the graph reads: the elementwise constants and the
    /// projections' weights and biases.
    consts: BTreeMap<String, Named>,
    /// How many threads the host convolutions are split over.
    ///
    /// The oracle's OCR takes `--threads` and defaults to sixteen, and the
    /// work is split the same way: the GEMM over rows, the depthwise
    /// convolutions over channels, the im2col over output rows. None of those
    /// splits crosses a reduction, so the threads are free and the bits do
    /// not move.
    threads: usize,
    /// What the run cost.
    pub cost: Cost,
    /// The im2col buffer, reused across convolutions.
    a: Vec<f32>,
    /// The `[M, N]` GEMM output, reused.
    c: Vec<f32>,
}

impl Network {
    /// Build a network over one graph.
    pub fn new(
        graph: Vec<Layer>,
        convs: Vec<ConvW>,
        consts: BTreeMap<String, Named>,
        threads: usize,
    ) -> Network {
        let mut wmat: Vec<Option<Vec<f32>>> = Vec::with_capacity(convs.len());
        for cw in &convs {
            if cw.depthwise() || cw.w.is_empty() {
                wmat.push(None);
                continue;
            }
            let k = (cw.cin * cw.kh * cw.kw) as usize;
            let n = cw.cout as usize;
            // transpose_nk_to_kn: dst[k*N + n] = src[n*K + k].
            let mut table = vec![0.0f32; k * n];
            for kk in 0..k {
                for nn in 0..n {
                    table[kk * n + nn] = cw.w[nn * k + kk];
                }
            }
            wmat.push(Some(table));
        }
        Network {
            graph,
            convs,
            wmat,
            consts,
            threads: threads.max(1),
            cost: Cost::default(),
            a: Vec::new(),
            c: Vec::new(),
        }
    }

    /// The graph's ops.
    pub fn graph(&self) -> &[Layer] {
        &self.graph
    }

    /// Forget what the last run cost.
    pub fn reset_cost(&mut self) {
        self.cost.reset();
    }

    /// One image in, every op's output out.
    ///
    /// Operand -1 in an op's `inputs` is the image, so the returned vector is
    /// one longer than the graph and op i is at index i + 1 -- index 0 is the
    /// image. Shifting by one rather than keeping a side map because an op's
    /// operand list is read by array index at a dozen call sites and a
    /// special case at each of them is a dozen chances to forget it.
    ///
    /// `on_node` is called once per op, in graph order, with the op's index
    /// and its output. The index is -1 for the network's INPUT, which is
    /// reported before the first op: a gate that compares this walk against
    /// an independent reading of the ONNX graph needs the same input on both
    /// sides, and the input is the one thing the two sides cannot each derive
    /// and agree on -- it is a resample of a photograph, and a resampler that
    /// rounds differently puts the two sides a whole quantization step apart
    /// before the first convolution.
    pub fn run(
        &mut self,
        input: &Tensor,
        on_node: &mut dyn FnMut(i64, &Tensor),
        mut array: ArrayRef<'_, '_>,
    ) -> Result<Vec<Tensor>, Refusal> {
        // THE BODY IS A LOCAL, not a field, and that is why the operands can
        // be BORROWED.
        //
        // A field would make every operand a borrow of `self`, and every op
        // that writes to `self` -- the convolution's scratch buffers, the
        // cost counters -- would have to wait for that borrow to end. The
        // only way out of that within one expression is to CLONE the
        // operand, and the detector's probability map is five and a half
        // million floats: twenty passes over it, twenty copies of 22 MB,
        // and a hundred milliseconds of copying a number that was not
        // changed.
        let mut body: Vec<Tensor> = vec![Tensor::default(); self.graph.len() + 1];
        body[0] = input.clone();
        on_node(-1, &body[0]);
        // Take the reusable convolution scratch out of the network for the
        // whole walk. The buffers go back at the end, so the NEXT image keeps
        // the allocations; only the borrows differ from the old code.
        let mut scratch = ConvScratch {
            a: std::mem::take(&mut self.a),
            c: std::mem::take(&mut self.c),
            cost: Cost::default(),
            threads: self.threads,
        };
        for i in 0..self.graph.len() {
            let l = self.graph[i].clone();
            let nl = format!("node {i}");
            let out = self.step(&body, i, &l, &nl, array.as_deref_mut(), &mut scratch)?;
            // THE CHECK, AND WHY IT IS HERE.
            //
            // Every op's output extent was resolved at PACK time and recorded
            // in the container, and the walk computes its own. The two must
            // agree, and where they do not, one of the two is wrong.
            //
            // A disagreement is caught AT THE OP, and that is the whole point:
            // "the recogniser disagrees with the checkpoint" and "the resize
            // indexes its source row wrongly" are different sentences, and
            // only the second one is actionable.
            if !l.oshape.is_empty() && out.shape != l.oshape {
                return Err(format!(
                    "{nl}: op '{}' produced {} and the container declares {}. The packer \
                     resolved every extent at pack time and recorded it, so one of the two is \
                     wrong -- and this is the op where they part company.",
                    l.op.as_str(),
                    shape_str(&out.shape),
                    shape_str(&l.oshape)
                ));
            }
            body[i + 1] = out;
            on_node(i as i64, &body[i + 1]);
        }
        self.a = std::mem::take(&mut scratch.a);
        self.c = std::mem::take(&mut scratch.c);
        scratch.merge_cost(&mut self.cost);
        Ok(body)
    }

    /// One op's output.
    fn step(
        &mut self,
        body: &[Tensor],
        index: usize,
        l: &Layer,
        nl: &str,
        array: ArrayRef<'_, '_>,
        scratch: &mut ConvScratch,
    ) -> Result<Tensor, Refusal> {
        // The operands. -1 is the image, and every other index is an EARLIER
        // op's output -- the graph is evaluated in order, which the reader
        // checked.
        let operand = |k: usize| -> Result<&Tensor, Refusal> {
            let j = l.inputs[k];
            if j == -1 {
                return Ok(&body[0]);
            }
            if j < 0 || j as usize >= index {
                return Err(format!(
                    "{nl}: operand {j} is not an earlier op and not the image (-1)."
                ));
            }
            Ok(&body[j as usize + 1])
        };
        match l.op {
            Op::Conv => {
                // BORROW the weight and the transposed table: the old
                // `.clone()` copied both on every single convolution. With
                // the scratch split out of `self`, the immutable borrow of
                // `self.convs`/`self.wmat` and the mutable scratch coexist.
                let w = self.conv_of(l, nl)?;
                let inp = operand(0)?;
                if w.depthwise() {
                    self.dwconv(inp, l, w, nl, scratch)
                } else {
                    let table = self.wmat[l.conv as usize].as_deref();
                    self.conv(inp, l, w, table, nl, array, scratch)
                }
            }
            Op::ConvTranspose => {
                let w = self.conv_of(l, nl)?;
                let inp = operand(0)?;
                self.convtranspose(inp, l, w, nl, scratch)
            }
            Op::Relu | Op::Sigmoid | Op::HardSwish | Op::HardSigmoid => {
                // A STANDALONE activation. The container records its
                // parameters on the op, and they are not the defaults -- the
                // detector's HardSigmoid has alpha 0.2 -- so they are read here
                // rather than assumed.
                let mut out = operand(0)?.clone();
                let mut a = l.clone();
                a.act = Some(match l.op {
                    Op::Relu => Act::Relu,
                    Op::Sigmoid => Act::Sigmoid,
                    Op::HardSwish => Act::HardSwish,
                    _ => Act::HardSigmoid,
                });
                apply_act(&mut out, &a);
                Ok(out)
            }
            Op::Add | Op::Sub | Op::Mul | Op::Div => {
                if !l.c.is_empty() {
                    let c = self.named(&l.c, nl)?.clone();
                    let inp = operand(0)?;
                    Ok(self.binary(inp, l, &c, nl))
                } else {
                    // TWO ACTIVATIONS. The recogniser's attention block
                    // multiplies (1,8,40,15) by (1,8,15,40), and the batch axes
                    // BROADCAST: the two operands are not the same shape and
                    // are not required to be.
                    let a = operand(0)?;
                    let b = operand(1)?;
                    let os = broadcast_shape(&a.shape, &b.shape)?;
                    let r = broadcast_apply(&os, &a.shape, &a.d, &b.shape, &b.d, l.op);
                    Ok(Tensor { shape: os, d: r })
                }
            }
            Op::Sqrt => {
                let mut out = operand(0)?.clone();
                for v in out.d.iter_mut() {
                    *v = v.sqrt();
                }
                Ok(out)
            }
            Op::Concat => self.concat(body, l, nl),
            Op::MaxPool | Op::AvgPool => {
                let inp = operand(0)?;
                self.pool(inp, l, nl)
            }
            Op::GAveragePool => {
                let inp = operand(0)?;
                self.gapool(inp, l, nl)
            }
            Op::Resize => {
                let inp = operand(0)?;
                self.resize(inp, l, nl)
            }
            Op::Reshape => {
                let inp = operand(0)?;
                Ok(self.reshape(inp, l, nl)?)
            }
            Op::Transpose => {
                let inp = operand(0)?;
                Ok(self.transpose(inp, l, nl)?)
            }
            Op::Slice => {
                let inp = operand(0)?;
                Ok(self.slice(inp, l, nl)?)
            }
            Op::Squeeze => {
                let inp = operand(0)?;
                Ok(self.squeeze(inp, l, false, nl)?)
            }
            Op::Unsqueeze => {
                let inp = operand(0)?;
                Ok(self.squeeze(inp, l, true, nl)?)
            }
            Op::MatMul => {
                let w = if l.w.is_empty() {
                    None
                } else {
                    Some(self.named(&l.w, nl)?.clone())
                };
                let b = if l.b.is_empty() {
                    None
                } else {
                    Some(self.named(&l.b, nl)?.clone())
                };
                let a = operand(0)?;
                let second = if l.inputs.len() > 1 {
                    Some(operand(1)?)
                } else {
                    None
                };
                Ok(self.matmul(a, l, w.as_ref(), b.as_ref(), second, nl)?)
            }
            Op::Softmax => {
                let inp = operand(0)?;
                Ok(self.softmax(inp, l, nl)?)
            }
            Op::ReduceMean => {
                let inp = operand(0)?;
                Ok(self.reducemean(inp, l, nl)?)
            }
        }
    }

    /// The convolution an op names.
    fn conv_of(&self, l: &Layer, nl: &str) -> Result<&ConvW, Refusal> {
        if l.conv < 0 || l.conv as usize >= self.convs.len() {
            return Err(format!(
                "{nl}: op '{}' names conv {}, which the container has no weights for.",
                l.op.as_str(),
                l.conv
            ));
        }
        let cw = &self.convs[l.conv as usize];
        if cw.w.is_empty() {
            return Err(format!(
                "{nl}: op '{}' names conv {}, which the container has no weights for. Both \
                 lists are written by the same packer pass, so a disagreement here means the \
                 container was edited or half-written -- and a convolution with no weights is \
                 a graph that would run on whatever was in the buffer.",
                l.op.as_str(),
                l.conv
            ));
        }
        Ok(cw)
    }

    /// A named tensor the graph reads.
    fn named(&self, name: &str, nl: &str) -> Result<&Named, Refusal> {
        self.consts.get(name).ok_or_else(|| {
            format!(
                "{nl}: the graph names the tensor '{name}', which is not in the container. The \
                 names come from the same packer pass that wrote the op list, so this is a \
                 container that was edited."
            )
        })
    }

    // -- the dense convolution ------------------------------------------------

    fn conv(
        &self,
        input: &Tensor,
        l: &Layer,
        w: &ConvW,
        table: Option<&[f32]>,
        nl: &str,
        array: ArrayRef<'_, '_>,
        scratch: &mut ConvScratch,
    ) -> Result<Tensor, Refusal> {
        if input.rank() != 4 {
            return Err(format!(
                "{nl}: conv {} reads a rank-{} tensor. The convolutions are NCHW.",
                w.index,
                input.rank()
            ));
        }
        if l.dilation[0] != 1 || l.dilation[1] != 1 {
            return Err(format!(
                "{nl}: conv {} is dilated ({}, {}). The shared im2col gathers the kernel's \
                 own positions and has no dilation term; these checkpoints do not use one.",
                w.index, l.dilation[0], l.dilation[1]
            ));
        }
        if w.cin != input.shape[1] {
            return Err(format!(
                "{nl}: conv {} takes {} channels and the tensor has {}. Reading the other \
                 one would index past the end of the im2col row.",
                w.index, w.cin, input.shape[1]
            ));
        }
        let h = input.shape[2];
        let width = input.shape[3];
        let eh = (w.kh - 1) * l.dilation[0] + 1;
        let ew = (w.kw - 1) * l.dilation[1] + 1;
        let oh = (h + l.pad[0] + l.pad[2] - eh) / l.stride[0] + 1;
        let ow = (width + l.pad[1] + l.pad[3] - ew) / l.stride[1] + 1;
        if oh <= 0 || ow <= 0 {
            return Err(format!(
                "{nl}: conv {} produces a {oh}x{ow} output from a {h}x{width} input.",
                w.index
            ));
        }
        let Some(wkn) = table else {
            return Err(format!(
                "{nl}: conv {} has no [K,N] weight table. The transpose is built in the \
                 constructor for every DENSE filter, so a missing one means this convolution \
                 is depthwise and was routed here by name -- which would compute a different \
                 network rather than fail.",
                w.index
            ));
        };
        let mut out = Tensor {
            shape: vec![1, w.cout, oh, ow],
            d: vec![0.0f32; (w.cout * oh * ow) as usize],
        };
        let m = oh * ow;
        let k = w.cin * w.kh * w.kw;
        let n = w.cout;

        let t0 = std::time::Instant::now();
        im2col(
            &input.d,
            w.cin,
            h,
            width,
            w.kh,
            w.kw,
            l.pad[0],
            l.pad[1],
            l.stride[0],
            l.stride[1],
            l.pad[2],
            l.pad[3],
            scratch.threads,
            &mut scratch.a,
        );
        debug_assert_eq!(scratch.a.len(), (oh * ow * k) as usize);
        scratch.cost.t_im2col += t0.elapsed().as_secs_f64();

        // THE BIAS, OR A ROW OF ZEROS IN ITS PLACE. `gemm_nt` adds bias[n] to
        // every output column, so a null bias is a null dereference and not a
        // zero contribution -- and the detector's stem convolutions have no
        // bias of their own.
        let bias: Vec<f32> = match &w.b {
            Some(b) => b.clone(),
            None => vec![0.0f32; n as usize],
        };
        let t1 = std::time::Instant::now();
        scratch.c.resize((m * n) as usize, 0.0);
        // THE ARRAY, OR THE HOST.
        //
        // This is the SEAM, and it is one branch in one place. The im2col
        // above is the host's either way -- it is a gather, and the array's
        // MMAC multiplies matrices -- and so is the transpose below.
        //
        // The two paths are NOT the same numbers: the array multiplies the
        // weights rounded to bf16, and the host multiplies them in f32. So a
        // run on the array is the same network with a coarser weight, and the
        // difference is the width of bf16 and not a bug.
        let on_array = array.is_some();
        match array {
            Some(array) => {
                let t = std::time::Instant::now();
                array
                    .conv(
                        &l.w,
                        &scratch.a,
                        m as usize,
                        k as usize,
                        n as usize,
                        &mut scratch.c,
                    )
                    .map_err(|e| format!("{nl}: {e}"))?;
                scratch.cost.t_array += t.elapsed().as_secs_f64();
                scratch.cost.convs_array += 1;
            }
            None => {
                // OVER ROWS, like the oracle: a row's reduction is over `k` in
                // order, so which thread computes a row changes the thread and not
                // the number.
                kernels::gemm_nt_rows(
                    &scratch.a,
                    m as usize,
                    wkn,
                    k as usize,
                    n as usize,
                    &bias,
                    &mut scratch.c,
                    scratch.threads,
                );
                scratch.cost.t_gemm += t1.elapsed().as_secs_f64();
                scratch.cost.convs_host += 1;
            }
        }
        let t2 = std::time::Instant::now();
        transpose_mn_to_nchw(&scratch.c, m, n, &mut out.d, scratch.threads);
        scratch.cost.t_transpose += t2.elapsed().as_secs_f64();
        if !on_array {
            // The host's share of a dense convolution is the GEMM AND the
            // transpose back to NCHW, and it is measured HERE so that the
            // transpose is in it. An array run has no host share of the
            // multiplication: the transpose is the host's, and the GEMM is
            // the array's.
            scratch.cost.t_host += t1.elapsed().as_secs_f64();
        }
        // The activation is a SEPARATE PASS, and the reason is arithmetic: the
        // host convolution's epilogue computes the OTHER architectures'
        // activations, and this one's hardswish and hardsigmoid carry a
        // parameter that an enum cannot hold.
        let t3 = std::time::Instant::now();
        apply_act(&mut out, l);
        scratch.cost.t_elementwise += t3.elapsed().as_secs_f64();
        Ok(out)
    }

    // -- the depthwise convolution --------------------------------------------

    fn dwconv(
        &self,
        input: &Tensor,
        l: &Layer,
        w: &ConvW,
        nl: &str,
        scratch: &mut ConvScratch,
    ) -> Result<Tensor, Refusal> {
        if input.rank() != 4 {
            return Err(format!(
                "{nl}: dwconv {} reads a rank-{} tensor.",
                w.index,
                input.rank()
            ));
        }
        let h = input.shape[2];
        let width = input.shape[3];
        let kh = w.kh;
        let kw = w.kw;
        let pt = l.pad[0];
        let pl = l.pad[1];
        let oh = (h + l.pad[0] + l.pad[2] - kh) / l.stride[0] + 1;
        let ow = (width + l.pad[1] + l.pad[3] - kw) / l.stride[1] + 1;
        if w.cout != input.shape[1] {
            return Err(format!(
                "{nl}: dwconv {} writes {} channels and the tensor has {}. A depthwise \
                 convolution reads channel c and writes channel c, and the two counts have \
                 to be the same number or the kernel below reads past the end of the input.",
                w.index, w.cout, input.shape[1]
            ));
        }
        let mut out = Tensor {
            shape: vec![1, w.cout, oh, ow],
            d: vec![0.0f32; (w.cout * oh * ow) as usize],
        };
        let t0 = std::time::Instant::now();
        // One filter per channel. The weight is `[Cout, 1, kh, kw]` and the
        // input channel of filter co IS channel co -- a depthwise convolution
        // reduces WITHIN one channel, which is why the array cannot take it.
        //
        // OVER CHANNELS, like the oracle: a channel's output does not depend
        // on any other channel's, so the split crosses no reduction.
        let plane_in = (h * width) as usize;
        let plane_out = (oh * ow) as usize;
        let cout = w.cout as usize;
        let threads = scratch.threads.min(cout.max(1));
        let work = (w.cout * oh * ow) as i64;
        let Some(chunk) = parallel::chunk_rows(cout, work, threads) else {
            self.dwconv_serial(input, l, w, &mut out, oh, ow, h, width, pt, pl);
            scratch.cost.t_dwconv += t0.elapsed().as_secs_f64();
            scratch.cost.dwconvs_host += 1;
            return Ok(out);
        };
        let ins: Vec<&[f32]> = input.d.chunks(plane_in.max(1)).collect();
        let mut outs: Vec<&mut [f32]> = out.d.chunks_mut(plane_out.max(1)).collect();
        parallel::for_each_chunk_mut(&mut outs, chunk.max(1), |gi, group| {
            let base = gi * chunk;
            let inputs = &ins[base..base + group.len()];
            for (i, dst) in group.iter_mut().enumerate() {
                let co = base + i;
                let wf = &w.w[((co as i64) * kh * kw) as usize..][..(kh * kw) as usize];
                let src = inputs[i];
                let b = w.b.as_ref().map(|v| v[co]).unwrap_or(0.0);
                for y in 0..oh {
                    for x in 0..ow {
                        // FUSED, and the object file says so: the
                        // oracle's dwconv is `vfmadd231ss`. A plain
                        // `acc += a * b` here would round twice where
                        // the oracle rounds once.
                        let mut acc = b;
                        for i in 0..kh {
                            let sy = y * l.stride[0] + i - pt;
                            if sy < 0 || sy >= h {
                                continue;
                            }
                            for j in 0..kw {
                                let sx = x * l.stride[1] + j - pl;
                                if sx < 0 || sx >= width {
                                    continue;
                                }
                                acc = wf[(i * kw + j) as usize]
                                    .mul_add(src[(sy * width + sx) as usize], acc);
                            }
                        }
                        dst[(y * ow + x) as usize] = match l.act {
                            Some(act) => act1(act, l.alpha, l.beta, acc),
                            None => acc,
                        };
                    }
                }
            }
        });
        scratch.cost.t_dwconv += t0.elapsed().as_secs_f64();
        scratch.cost.dwconvs_host += 1;
        Ok(out)
    }

    /// The depthwise convolution, one channel at a time.
    ///
    /// The same arithmetic as the parallel path, and the same bits: a
    /// channel's output does not depend on any other channel's, so the split
    /// is a choice of who does what and never of what the answer is. This
    /// is the path for the convolutions too small to be worth a thread.
    #[allow(clippy::too_many_arguments)]
    fn dwconv_serial(
        &self,
        input: &Tensor,
        l: &Layer,
        w: &ConvW,
        out: &mut Tensor,
        oh: i64,
        ow: i64,
        h: i64,
        width: i64,
        pt: i64,
        pl: i64,
    ) {
        let kh = w.kh;
        let kw = w.kw;
        let plane_out = (oh * ow) as usize;
        let mut outs = out.d.chunks_mut(plane_out.max(1));
        for co in 0..w.cout {
            let wf = &w.w[((co as i64) * kh * kw) as usize..][..(kh * kw) as usize];
            let src = input.chw(co);
            let b = w.b.as_ref().map(|v| v[co as usize]).unwrap_or(0.0);
            let dst = outs.next().unwrap_or(&mut []);
            for y in 0..oh {
                for x in 0..ow {
                    let mut acc = b;
                    for i in 0..kh {
                        let sy = y * l.stride[0] + i - pt;
                        if sy < 0 || sy >= h {
                            continue;
                        }
                        for j in 0..kw {
                            let sx = x * l.stride[1] + j - pl;
                            if sx < 0 || sx >= width {
                                continue;
                            }
                            acc = wf[(i * kw + j) as usize]
                                .mul_add(src[(sy * width + sx) as usize], acc);
                        }
                    }
                    dst[(y * ow + x) as usize] = match l.act {
                        Some(act) => act1(act, l.alpha, l.beta, acc),
                        None => acc,
                    };
                }
            }
        }
    }

    // -- the transposed convolution -------------------------------------------

    fn convtranspose(
        &self,
        input: &Tensor,
        l: &Layer,
        w: &ConvW,
        nl: &str,
        scratch: &mut ConvScratch,
    ) -> Result<Tensor, Refusal> {
        if input.rank() != 4 {
            return Err(format!(
                "{nl}: convtranspose reads a rank-{} tensor.",
                input.rank()
            ));
        }
        let h = input.shape[2];
        let width = input.shape[3];
        let eh = (w.kh - 1) * l.dilation[0] + 1;
        let ew = (w.kw - 1) * l.dilation[1] + 1;
        let oh = l.stride[0] * (h - 1) + l.outpad[0] + eh - l.pad[0] - l.pad[2];
        let ow = l.stride[1] * (width - 1) + l.outpad[1] + ew - l.pad[1] - l.pad[3];
        let mut out = Tensor {
            shape: vec![1, w.cout, oh, ow],
            d: vec![0.0f32; (w.cout * oh * ow) as usize],
        };
        let t0 = std::time::Instant::now();
        // The weight is `[Cin, Cout, kh, kw]` -- the SECOND extent is the
        // OUTPUT channel count, which is the opposite of a convolution's.
        //
        // NOT FUSED, and the object file says so: the oracle's convtranspose
        // is `vmulss` then `vaddss`, while its dwconv and its matmul are
        // `vfmadd231ss`. The DB head's two upsampling convolutions therefore
        // round TWICE per tap, and a `mul_add` here would move the
        // probability map by an ulp per tap.
        let cout = w.cout as usize;
        let khkw = (w.kh * w.kw) as usize;
        let tap = |i: i64, j: i64| (i * w.kw + j) as usize;
        let plane_out = (oh * ow) as usize;
        // OVER OUTPUT CHANNELS, in groups. A channel's plane is written by
        // one worker and read by no other, so the split crosses no reduction.
        // The input pixel is still gathered ONCE per group rather than once
        // per (channel, output channel) pair -- the old inner loop did 24
        // times more slice builds than values it read.
        let run_group = |first: usize, group: &mut [&mut [f32]]| {
            let mut pixel = vec![0.0f32; w.cin as usize];
            for i in 0..w.kh {
                for j in 0..w.kw {
                    let t = tap(i, j);
                    for iy in 0..h {
                        let oy = iy * l.stride[0] - l.pad[0] + i * l.dilation[0];
                        if oy < 0 || oy >= oh {
                            continue;
                        }
                        for ix in 0..width {
                            let ox = ix * l.stride[1] - l.pad[1] + j * l.dilation[1];
                            if ox < 0 || ox >= ow {
                                continue;
                            }
                            for (ci, v) in pixel.iter_mut().enumerate() {
                                *v = input.chw(ci as i64)[(iy * width + ix) as usize];
                            }
                            for (local, dst) in group.iter_mut().enumerate() {
                                let co = first + local;
                                let mut acc = 0.0f32;
                                for ci in 0..w.cin as usize {
                                    // NOT FUSED, like the oracle: vmulss then
                                    // vaddss. A `mul_add` would move the
                                    // probability map by an ulp per tap.
                                    acc += pixel[ci] * w.w[(ci * cout + co) * khkw + t];
                                }
                                dst[(oy * ow + ox) as usize] += acc;
                            }
                        }
                    }
                }
            }
            if let Some(b) = &w.b {
                for (local, dst) in group.iter_mut().enumerate() {
                    let bias = b[(first + local) as usize];
                    for v in dst.iter_mut() {
                        *v += bias;
                    }
                }
            }
        };
        let mut outs: Vec<&mut [f32]> = out.d.chunks_mut(plane_out.max(1)).collect();
        let work = (cout as i64)
            * (w.cin as i64)
            * (khkw as i64)
            * (h as i64)
            * (width as i64);
        match parallel::chunk_rows(cout, work, parallel::threads()) {
            Some(chunk) => {
                parallel::for_each_chunk_mut(&mut outs, chunk, |gi, group| {
                    run_group(gi * chunk, group)
                });
            }
            None => run_group(0, &mut outs),
        }
        scratch.cost.t_transposed += t0.elapsed().as_secs_f64();
        scratch.cost.transposed_convs += 1;
        Ok(out)
    }

    // -- the elementwise pass with a constant ----------------------------------

    fn binary(&mut self, input: &Tensor, l: &Layer, c: &Named, nl: &str) -> Tensor {
        // THE CONSTANT BROADCASTS. It is usually a `[1, C, 1, 1]` bias --
        // paddle2onnx expresses a convolution's bias that way -- and sometimes
        // a scalar.
        let os = broadcast_shape(&input.shape, &c.shape).unwrap_or_else(|e| panic!("{e}: {nl}"));
        let r = broadcast_apply(&os, &input.shape, &input.d, &c.shape, &c.d, l.op);
        self.cost.elementwise_ops += 1;
        Tensor { shape: os, d: r }
    }

    // -- the concat -------------------------------------------------------------

    fn concat(&mut self, body: &[Tensor], l: &Layer, nl: &str) -> Result<Tensor, Refusal> {
        // The first operand carries the shape; the others are joined along the
        // axis. The packer resolved the axis at pack time, and the walk checks
        // the result against the declared extent.
        let first = &body[l.inputs[0] as usize + 1];
        let axis = l.axis;
        let rank = first.rank();
        if axis < 0 || axis >= rank {
            return Err(format!(
                "{nl}: concat's axis is {axis} and its operand has rank {rank}."
            ));
        }
        let mut os = first.shape.clone();
        let mut total = 0i64;
        for k in &l.inputs {
            let t = &body[*k as usize + 1];
            if t.rank() != rank {
                return Err(format!(
                    "{nl}: concat joins a rank-{} tensor with a rank-{rank} one.",
                    t.rank()
                ));
            }
            total += t.shape[axis as usize];
        }
        os[axis as usize] = total;
        let mut out = Tensor {
            shape: os.clone(),
            d: vec![0.0f32; (os.iter().product::<i64>()) as usize],
        };
        // The strides of the output's axes, and of each operand's.
        let mut ostride = vec![1i64; rank as usize];
        for k in (0..rank).rev() {
            ostride[k as usize] = if k == rank - 1 {
                1
            } else {
                ostride[(k + 1) as usize] * os[(k + 1) as usize]
            };
        }
        let mut offset = 0i64;
        for k in &l.inputs {
            let t = &body[*k as usize + 1];
            let mut istride = vec![1i64; rank as usize];
            for a in (0..rank).rev() {
                istride[a as usize] = if a == rank - 1 {
                    1
                } else {
                    istride[(a + 1) as usize] * t.shape[(a + 1) as usize]
                };
            }
            let mut idx = vec![0i64; rank as usize];
            let total_elems: i64 = t.shape.iter().product();
            for n in 0..total_elems {
                let mut o = offset;
                for a in 0..rank as usize {
                    o += idx[a] * ostride[a];
                }
                out.d[o as usize] = t.d[n as usize];
                let mut a = rank;
                while a > 0 {
                    a -= 1;
                    idx[a as usize] += 1;
                    if idx[a as usize] < t.shape[a as usize] {
                        break;
                    }
                    idx[a as usize] = 0;
                }
            }
            offset += t.shape[axis as usize] * ostride[axis as usize];
        }
        self.cost.elementwise_ops += 1;
        Ok(out)
    }

    // -- the pooling -------------------------------------------------------------

    fn pool(&mut self, input: &Tensor, l: &Layer, nl: &str) -> Result<Tensor, Refusal> {
        if input.rank() != 4 {
            return Err(format!(
                "{nl}: pooling reads a rank-{} tensor.",
                input.rank()
            ));
        }
        let h = input.shape[2];
        let width = input.shape[3];
        let kh = l.kernel[0];
        let kw = l.kernel[1];
        let pt = l.pad[0];
        let pl = l.pad[1];
        let oh = (h + l.pad[0] + l.pad[2] - kh) / l.stride[0] + 1;
        let ow = (width + l.pad[1] + l.pad[3] - kw) / l.stride[1] + 1;
        let channels = input.shape[1];
        let mut out = Tensor {
            shape: vec![1, channels, oh, ow],
            d: vec![0.0f32; (channels * oh * ow) as usize],
        };
        let is_max = l.op == Op::MaxPool;
        let t0 = std::time::Instant::now();
        for c in 0..channels {
            let src = input.chw(c);
            for y in 0..oh {
                for x in 0..ow {
                    let mut best = f32::NEG_INFINITY;
                    let mut sum = 0.0f32;
                    let mut n = 0i64;
                    for i in 0..kh {
                        let sy = y * l.stride[0] + i - pt;
                        if sy < 0 || sy >= h {
                            continue;
                        }
                        for j in 0..kw {
                            let sx = x * l.stride[1] + j - pl;
                            if sx < 0 || sx >= width {
                                continue;
                            }
                            let v = src[(sy * width + sx) as usize];
                            if is_max {
                                best = best.max(v);
                            }
                            sum += v;
                            n += 1;
                        }
                    }
                    // The AVERAGE divides by the elements that were IN the
                    // window, not by the window's size: ONNX's
                    // count_include_pad is 0, which is what PaddleOCR's own
                    // average pool carries, and these graphs' pools have no
                    // padding at all.
                    out.chw_mut(c)[(y * ow + x) as usize] = if is_max {
                        best
                    } else if n > 0 {
                        sum / n as f32
                    } else {
                        0.0
                    };
                }
            }
        }
        self.cost.t_elementwise += t0.elapsed().as_secs_f64();
        self.cost.elementwise_ops += 1;
        Ok(out)
    }

    fn gapool(&mut self, input: &Tensor, _l: &Layer, nl: &str) -> Result<Tensor, Refusal> {
        if input.rank() != 4 {
            return Err(format!(
                "{nl}: the global average pool reads a rank-{} tensor.",
                input.rank()
            ));
        }
        let channels = input.shape[1];
        let h = input.shape[2];
        let width = input.shape[3];
        let mut out = Tensor {
            shape: vec![1, channels, 1, 1],
            d: vec![0.0f32; channels as usize],
        };
        let t0 = std::time::Instant::now();
        for c in 0..channels {
            let src = input.chw(c);
            // The accumulator is DOUBLE, and that is the oracle's: a mean over
            // a plane of a few thousand elements in f32 would lose the last
            // digits of the mean.
            let mut acc = 0.0f64;
            for v in src {
                acc += *v as f64;
            }
            out.d[c as usize] = (acc / ((h * width) as f64)) as f32;
        }
        self.cost.t_elementwise += t0.elapsed().as_secs_f64();
        self.cost.elementwise_ops += 1;
        Ok(out)
    }

    // -- the resize ---------------------------------------------------------------

    /// ONNX Resize, nearest, "asymmetric" + "floor".
    ///
    /// THE SOURCE INDEX IS floor(dst * in / out) -- NOT dst * scale and NOT
    /// the half-pixel grid. The container records the SCALE and the walk
    /// derives the output extent from the input and the scale, which is the
    /// same number the packer used, and the source index comes out of the two.
    fn resize(&mut self, input: &Tensor, l: &Layer, nl: &str) -> Result<Tensor, Refusal> {
        if input.rank() != 4 {
            return Err(format!(
                "{nl}: resize reads a rank-{} tensor.",
                input.rank()
            ));
        }
        let channels = input.shape[1];
        let h = input.shape[2];
        let width = input.shape[3];
        let oh = (h as f64 * l.scale_h).floor() as i64;
        let ow = (width as f64 * l.scale_w).floor() as i64;
        let mut out = Tensor {
            shape: vec![1, channels, oh, ow],
            d: vec![0.0f32; (channels * oh * ow) as usize],
        };
        let ys: Vec<i64> = (0..oh)
            .map(|d| if oh > 0 { ((d * h) / oh).min(h - 1) } else { 0 })
            .collect();
        let xs: Vec<i64> = (0..ow)
            .map(|d| {
                if ow > 0 {
                    ((d * width) / ow).min(width - 1)
                } else {
                    0
                }
            })
            .collect();
        let t0 = std::time::Instant::now();
        for c in 0..channels {
            let src = input.chw(c);
            for y in 0..oh {
                let srow = &src[(ys[y as usize] * width) as usize..];
                let drow = out.chw_mut(c);
                let base = (y * ow) as usize;
                for x in 0..ow {
                    drow[base + x as usize] = srow[xs[x as usize] as usize];
                }
            }
        }
        self.cost.t_elementwise += t0.elapsed().as_secs_f64();
        self.cost.elementwise_ops += 1;
        Ok(out)
    }

    // -- the shape ops -------------------------------------------------------------

    fn reshape(&mut self, input: &Tensor, l: &Layer, nl: &str) -> Result<Tensor, Refusal> {
        let n: i64 = l.shape.iter().product();
        if n != input.d.len() as i64 {
            return Err(format!(
                "{nl}: reshape to {} holds {n} values and the input holds {}. A reshape \
                 preserves the element count.",
                shape_str(&l.shape),
                input.d.len()
            ));
        }
        Ok(Tensor {
            shape: l.shape.clone(),
            d: input.d.clone(),
        })
    }

    fn transpose(&mut self, input: &Tensor, l: &Layer, nl: &str) -> Result<Tensor, Refusal> {
        let rank = input.rank();
        let perm = &l.perm;
        if perm.len() as i64 != rank {
            return Err(format!(
                "{nl}: transpose's perm has {} entries and the tensor has rank {rank}.",
                perm.len()
            ));
        }
        let mut os = vec![1i64; rank as usize];
        for i in 0..rank as usize {
            os[i] = input.shape[perm[i] as usize];
        }
        let mut out = Tensor {
            shape: os.clone(),
            d: vec![0.0f32; (os.iter().product::<i64>()) as usize],
        };
        // The output's strides, and the input's strides under the permutation.
        let mut ostride = vec![1i64; rank as usize];
        let mut istride = vec![1i64; rank as usize];
        for i in (0..rank).rev() {
            ostride[i as usize] = if i == rank - 1 {
                1
            } else {
                ostride[(i + 1) as usize] * os[(i + 1) as usize]
            };
            istride[i as usize] = if i == rank - 1 {
                1
            } else {
                istride[(i + 1) as usize] * input.shape[(i + 1) as usize]
            };
        }
        let mut idx = vec![0i64; rank as usize];
        let total: i64 = os.iter().product();
        for n in 0..total {
            let mut src = 0i64;
            for i in 0..rank as usize {
                src += idx[i] * istride[perm[i] as usize];
            }
            out.d[n as usize] = input.d[src as usize];
            let mut i = rank;
            while i > 0 {
                i -= 1;
                idx[i as usize] += 1;
                if idx[i as usize] < os[i as usize] {
                    break;
                }
                idx[i as usize] = 0;
            }
        }
        self.cost.elementwise_ops += 1;
        Ok(out)
    }

    fn slice(&mut self, input: &Tensor, l: &Layer, nl: &str) -> Result<Tensor, Refusal> {
        let rank = input.rank();
        // The starts, ends and steps are PER AXIS, and the axes they apply to
        // are named -- ONNX's Slice takes one start, end and step per axis and
        // says which axis each is for.
        let mut begin = vec![0i64; rank as usize];
        let mut stop = input.shape.clone();
        let mut step = vec![1i64; rank as usize];
        for k in 0..l.axes.len() {
            let a = l.axes[k];
            if a < 0 || a >= rank {
                return Err(format!(
                    "{nl}: slice names axis {a} and the tensor has rank {rank}."
                ));
            }
            let dim = input.shape[a as usize];
            let mut s = l.starts.get(k).copied().unwrap_or(0);
            let mut e = l.ends.get(k).copied().unwrap_or(input.shape[a as usize]);
            let sp = l.steps.get(k).copied().unwrap_or(1);
            // numpy's rules: a negative start counts from the end, a negative
            // end counts from the end, and both are clipped into the axis.
            if s < 0 {
                s += dim;
            }
            if e < 0 {
                e += dim;
            }
            s = s.clamp(0, dim);
            e = e.clamp(0, dim);
            if sp > 0 {
                if e < s {
                    e = s;
                }
            } else {
                s = s.clamp(-1, dim - 1);
                e = e.clamp(-1, dim - 1);
                if e > s {
                    e = s;
                }
            }
            begin[a as usize] = s;
            stop[a as usize] = e;
            step[a as usize] = sp;
        }
        let mut os = vec![1i64; rank as usize];
        for a in 0..rank as usize {
            let n = if step[a] > 0 {
                (stop[a] - begin[a] + step[a] - 1) / step[a]
            } else {
                (begin[a] - stop[a] + (-step[a]) - 1) / (-step[a])
            };
            os[a] = n.max(0);
        }
        let mut out = Tensor {
            shape: os.clone(),
            d: vec![0.0f32; (os.iter().product::<i64>()) as usize],
        };
        let mut idx = vec![0i64; rank as usize];
        let total: i64 = os.iter().product();
        for n in 0..total {
            let mut src = 0i64;
            for a in 0..rank as usize {
                src += (begin[a] + idx[a] * step[a]) * row_stride(&input.shape, a);
            }
            out.d[n as usize] = input.d[src as usize];
            let mut a = rank;
            while a > 0 {
                a -= 1;
                idx[a as usize] += 1;
                if idx[a as usize] < os[a as usize] {
                    break;
                }
                idx[a as usize] = 0;
            }
        }
        self.cost.elementwise_ops += 1;
        Ok(out)
    }

    fn squeeze(
        &mut self,
        input: &Tensor,
        l: &Layer,
        unsqueeze: bool,
        nl: &str,
    ) -> Result<Tensor, Refusal> {
        if unsqueeze {
            // AN UNSQUEEZE INSERTS a 1 at each of the axes it names, and the
            // axes are positions in the OUTPUT's rank -- not in the input's.
            let rank = input.rank() + l.axes.len() as i64;
            let mut sorted = l.axes.clone();
            sorted.sort_unstable();
            let mut os: Vec<i64> = Vec::with_capacity(rank as usize);
            let mut next = 0usize;
            for a in 0..rank {
                if sorted.binary_search(&(a as i64)).is_ok() {
                    os.push(1);
                } else {
                    os.push(input.shape[next]);
                    next += 1;
                }
            }
            if next != input.shape.len() {
                return Err(format!(
                    "{nl}: unsqueeze names {} axes and the tensor has rank {}.",
                    l.axes.len(),
                    input.rank()
                ));
            }
            return Ok(Tensor {
                shape: os,
                d: input.d.clone(),
            });
        }
        // A SQUEEZE DROPS THE AXES IT NAMES, and the others keep their order.
        let mut os: Vec<i64> = Vec::with_capacity(input.rank() as usize);
        for a in 0..input.rank() {
            if !l.axes.contains(&a) {
                os.push(input.shape[a as usize]);
            }
        }
        self.cost.elementwise_ops += 1;
        Ok(Tensor {
            shape: os,
            d: input.d.clone(),
        })
    }

    // -- the matmul ----------------------------------------------------------------

    /// ONNX's MatMul, in ONNX's general sense.
    ///
    /// Three spellings occur in these graphs:
    ///
    /// * `x @ W` -- a projection onto a matrix from the container. W is rank 2.
    /// * `A @ B` -- two activations, batched. The recogniser's attention block
    ///   multiplies (1,8,40,15) by (1,8,15,40) and gets (1,8,40,40) -- the
    ///   leading axes are BATCH axes and they broadcast.
    /// * `W @ x` -- a rank-2 matrix times a rank-1 activation.
    fn matmul(
        &mut self,
        input: &Tensor,
        _l: &Layer,
        w: Option<&Named>,
        b: Option<&Named>,
        second: Option<&Tensor>,
        nl: &str,
    ) -> Result<Tensor, Refusal> {
        let mut ashape = input.shape.clone();
        let (bshape, bdata): (Vec<i64>, Vec<f32>) = match w {
            Some(w) => (w.shape.clone(), w.d.clone()),
            None => {
                // A @ B, both activations. The second operand is the graph's
                // other operand.
                let rhs = second.ok_or_else(|| {
                    format!("{nl}: matmul has neither a named weight nor a second operand.")
                })?;
                (rhs.shape.clone(), rhs.d.clone())
            }
        };
        if input.rank() == 1 {
            ashape.insert(0, 1);
        }
        let mut bshape = bshape;
        if bshape.len() == 1 {
            bshape.push(1);
        }
        let m = ashape[ashape.len() - 2];
        let k = ashape[ashape.len() - 1];
        let n = bshape[bshape.len() - 1];
        if bshape[bshape.len() - 2] != k {
            return Err(format!(
                "{nl}: matmul's operands are {} and {}, and the inner extents do not agree.",
                shape_str(&ashape),
                shape_str(&bshape)
            ));
        }
        // The batch axes, broadcast.
        let ab = ashape.len() - 2;
        let bb = bshape.len() - 2;
        let nb = ab.max(bb);
        let mut batch = vec![1i64; nb];
        for i in 0..nb {
            let av = if i < nb - ab {
                1
            } else {
                ashape[i - (nb - ab)]
            };
            let bv = if i < nb - bb {
                1
            } else {
                bshape[i - (nb - bb)]
            };
            if av != bv && av != 1 && bv != 1 {
                return Err(format!(
                    "{nl}: matmul's batch axes do not broadcast: {} and {}.",
                    shape_str(&ashape),
                    shape_str(&bshape)
                ));
            }
            batch[i] = av.max(bv);
        }
        let mut os = batch.clone();
        os.push(m);
        os.push(n);
        let mut out = Tensor {
            shape: os.clone(),
            d: vec![0.0f32; (os.iter().product::<i64>()) as usize],
        };
        // The strides of the operands' batch axes, in units of ONE BATCH
        // ELEMENT (M*K for the left operand and K*N for the right one).
        //
        // The axes and the strides advance TOGETHER, from the last one
        // backwards: the last batch axis has stride 1 and each earlier one is
        // multiplied by the extents of the ones after it. Pairing them the
        // other way round leaves every batch reading the FIRST batch's slice of
        // the operand, and the attention's eight heads then all read the same
        // keys.
        let batch_stride = |shape: &[i64], nb: usize| -> Vec<i64> {
            let nbatch = shape.len() - 2;
            let mut s = vec![0i64; nb];
            let mut acc = 1i64;
            for k in 0..nbatch {
                let i = nbatch - 1 - k;
                let out_axis = nb - nbatch + i;
                if shape[i] != 1 {
                    s[out_axis] = acc;
                    acc *= shape[i];
                }
            }
            s
        };
        let astr = batch_stride(&ashape, nb);
        let bstr = batch_stride(&bshape, nb);
        let mut nbatches = 1i64;
        for v in &batch {
            nbatches *= v;
        }
        let t0 = std::time::Instant::now();
        // One pass over the batch, and one GEMM per batch element.
        let mut bidx = vec![0i64; nb];
        for bi in 0..nbatches {
            let mut aoff = 0i64;
            let mut boff = 0i64;
            for i in 0..nb {
                aoff += bidx[i] * astr[i];
                boff += bidx[i] * bstr[i];
            }
            let a = &input.d[(aoff * m * k) as usize..];
            let bb_ = &bdata[(boff * k * n) as usize..];
            let c = &mut out.d[(bi * m * n) as usize..];
            // TWO ROUNDINGS IN THE BODY, ONE IN THE TAIL.
            //
            // The oracle's matmul is the plain triple loop, and the compiler
            // rewrote it: the body of the k loop runs FOUR ELEMENTS AT A TIME
            // (`vmulps` then four `vaddss`, in the source's order), and what
            // the vectoriser leaves over -- the last K mod 4 of them -- stays
            // scalar, where `acc += a * b` is CONTRACTED into `vfmadd231ss`.
            //
            // So one product is rounded twice in the body and once in the
            // tail, and the two roundings differ by one ulp. The recogniser's
            // attention has K = 15, so its last three products are fused and
            // its first twelve are not, and the difference is visible: the
            // oracle's attention scores and this runtime's differed in the
            // sixth decimal, which is exactly what the golden transcript
            // prints.
            let kbody = k - k % 4;
            for mi in 0..m {
                for ni in 0..n {
                    let mut acc = 0.0f32;
                    for ki in 0..kbody {
                        acc += a[(mi * k + ki) as usize] * bb_[(ki * n + ni) as usize];
                    }
                    for ki in kbody..k {
                        acc = a[(mi * k + ki) as usize].mul_add(bb_[(ki * n + ni) as usize], acc);
                    }
                    c[(mi * n + ni) as usize] = acc;
                }
            }
            let mut i = nb;
            while i > 0 {
                i -= 1;
                bidx[i] += 1;
                if bidx[i] < batch[i] {
                    break;
                }
                bidx[i] = 0;
            }
        }
        self.cost.t_gemm += t0.elapsed().as_secs_f64();
        // The bias, if the graph adds one: it was FOLDED into the projection by
        // the packer, so it is added here rather than being a second op.
        if let Some(b) = b {
            let count = b.d.len();
            // THE COLUMN, NOT THE WIDTH. The bias's own index picks the
            // COLUMN and the output's WIDTH picks the stride between the
            // rows: a loop variable that shadows the width writes the bias
            // into the wrong place, and the wrong place is a different
            // element of the projection -- not a different rounding of the
            // same one.
            for col in 0..count {
                let bias = b.d[col];
                for bi in 0..nbatches {
                    for mi in 0..m {
                        out.d[((bi * m + mi) * n + col as i64) as usize] += bias;
                    }
                }
            }
        }
        self.cost.elementwise_ops += 1;
        Ok(out)
    }

    // -- the softmax and the reduction ---------------------------------------------

    fn softmax(&mut self, input: &Tensor, l: &Layer, nl: &str) -> Result<Tensor, Refusal> {
        let mut out = input.clone();
        let rank = input.rank();
        let mut axis = l.axis;
        if axis < 0 {
            axis += rank;
        }
        if axis < 0 || axis >= rank {
            return Err(format!(
                "{nl}: softmax's axis is {} and the tensor has rank {rank}.",
                l.axis
            ));
        }
        // The axis's extent, and the strides of the other axes.
        let dim = input.shape[axis as usize];
        let mut outer = 1i64;
        let mut inner = 1i64;
        for i in 0..axis {
            outer *= input.shape[i as usize];
        }
        for i in (axis + 1)..rank {
            inner *= input.shape[i as usize];
        }
        let t0 = std::time::Instant::now();
        for o in 0..outer {
            for i in 0..inner {
                let base = ((o * dim) * inner + i) as usize;
                let mut m = f32::NEG_INFINITY;
                for k in 0..dim {
                    m = m.max(out.d[base + (k * inner) as usize]);
                }
                let mut sum = 0.0f32;
                for k in 0..dim {
                    // The oracle's `std::exp` on a float argument is the
                    // SINGLE precision one.
                    let e = (out.d[base + (k * inner) as usize] - m).exp();
                    out.d[base + (k * inner) as usize] = e;
                    sum += e;
                }
                for k in 0..dim {
                    out.d[base + (k * inner) as usize] /= sum;
                }
            }
        }
        self.cost.t_elementwise += t0.elapsed().as_secs_f64();
        self.cost.elementwise_ops += 1;
        Ok(out)
    }

    fn reducemean(&mut self, input: &Tensor, l: &Layer, nl: &str) -> Result<Tensor, Refusal> {
        let rank = input.rank();
        let mut reduced = vec![false; rank as usize];
        for a in &l.axes {
            let ax = if *a < 0 { a + rank } else { *a };
            if ax < 0 || ax >= rank {
                return Err(format!(
                    "{nl}: reducemean names axis {a} and the tensor has rank {rank}."
                ));
            }
            reduced[ax as usize] = true;
        }
        let mut os: Vec<i64> = Vec::with_capacity(rank as usize);
        for a in 0..rank as usize {
            if l.keepdims != 0 {
                os.push(if reduced[a] { 1 } else { input.shape[a] });
            } else if !reduced[a] {
                os.push(input.shape[a]);
            }
        }
        let mut out = Tensor {
            shape: os.clone(),
            d: vec![0.0f32; (os.iter().product::<i64>()) as usize],
        };
        // THE INPUT'S OWN STRIDES, and not the output's sizes: a reduced axis
        // still separates the elements on either side of it.
        let mut ostride = vec![1i64; rank as usize];
        {
            let mut acc = 1i64;
            for a in (0..rank).rev() {
                ostride[a as usize] = acc;
                acc *= input.shape[a as usize];
            }
        }
        let n_red = reduced.iter().filter(|r| **r).count();
        let mut count = 1i64;
        for a in 0..rank as usize {
            if reduced[a] {
                count *= input.shape[a];
            }
        }
        // The output's own strides over the kept axes.
        let mut out_stride = vec![0i64; rank as usize];
        {
            let mut acc = 1i64;
            for a in (0..rank).rev() {
                if !reduced[a as usize] {
                    out_stride[a as usize] = acc;
                    acc *= input.shape[a as usize];
                }
            }
        }
        // The reduced axes' extents, in the order the reduced index walks them.
        let red_dims: Vec<i64> = (0..rank as usize)
            .filter(|a| reduced[*a])
            .map(|a| input.shape[a])
            .collect();
        let t0 = std::time::Instant::now();
        let mut idx = vec![0i64; rank as usize];
        let total: i64 = os.iter().product();
        for _ in 0..total {
            // The accumulator is DOUBLE, and that is the oracle's.
            let mut acc = 0.0f64;
            // Every combination of the reduced axes.
            let mut ridx = vec![0i64; n_red];
            for _ in 0..count {
                let mut src = 0i64;
                let mut r = 0usize;
                for a in 0..rank as usize {
                    let v = if reduced[a] {
                        let v = ridx[r];
                        r += 1;
                        v
                    } else {
                        idx[a]
                    };
                    src += v * ostride[a];
                }
                acc += input.d[src as usize] as f64;
                let mut a = n_red;
                while a > 0 {
                    a -= 1;
                    ridx[a] += 1;
                    if ridx[a] < red_dims[a] {
                        break;
                    }
                    ridx[a] = 0;
                }
            }
            let mut dst = 0i64;
            for a in 0..rank as usize {
                if !reduced[a] {
                    dst += idx[a] * out_stride[a];
                }
            }
            out.d[dst as usize] = (acc / count as f64) as f32;
            let mut a = rank;
            while a > 0 {
                a -= 1;
                if reduced[a as usize] {
                    continue;
                }
                idx[a as usize] += 1;
                if idx[a as usize] < input.shape[a as usize] {
                    break;
                }
                idx[a as usize] = 0;
            }
        }
        self.cost.t_elementwise += t0.elapsed().as_secs_f64();
        self.cost.elementwise_ops += 1;
        Ok(out)
    }
}

/// The im2col for a stride/padding generalisation of the same case.
///
/// The k index is `(ci*kh + i)*kw + j` and the row index is `y*W + x`, which
/// is the order the weights are read in too, so the GEMM's K reduction walks
/// the kernel exactly as the checkpoint stores it.
///
/// The output extent is `floor((H + ph + pb - kh)/sh) + 1` and each row
/// gathers the kh x kw window at `(y*sh + i - ph, x*sw + j - pw)`, reading
/// zero outside. Zero rather than clamp: a clamped edge replicates the border
/// pixel, which is a different network.
///
/// PADDING IS FOUR TERMS HERE, NOT TWO: the v5 detector spells two of its stem
/// convolutions SAME_UPPER, which is asymmetric.
#[allow(clippy::too_many_arguments)]
fn im2col(
    src: &[f32],
    c: i64,
    h: i64,
    width: i64,
    kh: i64,
    kw: i64,
    ph: i64,
    pw: i64,
    sh: i64,
    sw: i64,
    pb: i64,
    pr: i64,
    threads: usize,
    a: &mut Vec<f32>,
) {
    // THE OUTPUT EXTENT, and the FOUR pad terms are what make it.
    //
    // The v5 detector spells two of its stem convolutions SAME_UPPER, which
    // is ASYMMETRIC padding: the extra row and column go at the end. A
    // symmetric transcription of it is a 479-wide map where the checkpoint
    // has a 480-wide one.
    let oh = (h + ph + pb - kh) / sh + 1;
    let ow = (width + pw + pr - kw) / sw + 1;
    let m = oh * ow;
    let k = c * kh * kw;
    if m <= 0 || k <= 0 {
        a.clear();
        return;
    }
    // The caller owns the buffer and reuses it across convolutions. Resize
    // grows it to the largest shape; clearing first zeroes the out-of-window
    // columns that the gather below leaves untouched.
    let len = (m * k) as usize;
    a.resize(len, 0.0);
    a.fill(0.0);
    // OVER OUTPUT ROWS, like the oracle: a row's im2col reads the source and
    // writes its own run of A, and no row reads another's. The split crosses
    // no reduction, so the threads are free.
    let row_floats = (ow * k) as usize;
    let Some(rows_per) = parallel::chunk_rows(oh as usize, m * k, threads as usize) else {
        // Too small to be worth a worker: the barrier costs more than the
        // gather.
        for y in 0..oh {
            for x in 0..ow {
                let row = &mut a[((y * ow + x) * k) as usize..][..k as usize];
                for i in 0..kh {
                    let sy = y * sh + i - ph;
                    if sy < 0 || sy >= h {
                        continue;
                    }
                    for j in 0..kw {
                        let sx = x * sw + j - pw;
                        if sx < 0 || sx >= width {
                            continue;
                        }
                        for ch in 0..c {
                            row[((ch * kh + i) * kw + j) as usize] =
                                src[((ch * h + sy) * width + sx) as usize];
                        }
                    }
                }
            }
        }
        return;
    };
    parallel::for_each_chunk_mut(&mut a[..], rows_per * row_floats, |ci, chunk| {
        let first = ci as i64 * rows_per as i64;
        let rows = (chunk.len() / row_floats) as i64;
        for y in first..first + rows {
            for x in 0..ow {
                let row =
                    &mut chunk[(((y - first) * ow + x) as usize) * k as usize..][..k as usize];
                for i in 0..kh {
                    let sy = y * sh + i - ph;
                    if sy < 0 || sy >= h {
                        continue;
                    }
                    for j in 0..kw {
                        let sx = x * sw + j - pw;
                        if sx < 0 || sx >= width {
                            continue;
                        }
                        for ch in 0..c {
                            row[((ch * kh + i) * kw + j) as usize] =
                                src[((ch * h + sy) * width + sx) as usize];
                        }
                    }
                }
            }
        }
    });
}

/// The `[M, N]` to NCHW transpose.
///
/// Tiled 32 rows at a time: a naive n-outer / m-inner copy writes M runs of
/// one float each for every channel and streams the destination as N
/// simultaneous write pointers.
///
/// OVER CHANNELS: a channel's plane is written by one thread and read from
/// no other, so the split crosses no reduction.
fn transpose_mn_to_nchw(src: &[f32], m: i64, n: i64, dst: &mut [f32], threads: usize) {
    const TILE: i64 = 32;
    let Some(per) = parallel::chunk_rows(n as usize, m * n, threads) else {
        // Too small to be worth a worker.
        let mut m0 = 0i64;
        while m0 < m {
            let m1 = (m0 + TILE).min(m);
            for ch in 0..n {
                let d = &mut dst[(ch * m) as usize..][..m as usize];
                for mm in m0..m1 {
                    d[mm as usize] = src[(mm * n + ch) as usize];
                }
            }
            m0 = m1;
        }
        return;
    };
    let mut channels: Vec<&mut [f32]> = dst.chunks_mut((m as usize).max(1)).collect();
    parallel::for_each_chunk_mut(&mut channels, per.max(1), |gi, group| {
        let base = gi as i64 * per as i64;
        for (i, d) in group.iter_mut().enumerate() {
            let ch = base + i as i64;
            let mut m0 = 0i64;
            while m0 < m {
                let m1 = (m0 + TILE).min(m);
                for mm in m0..m1 {
                    d[mm as usize] = src[(mm * n + ch) as usize];
                }
                m0 = m1;
            }
        }
    });
}
