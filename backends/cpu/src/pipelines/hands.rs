//! The MediaPipe hands networks: two packed graphs, one container, two heads.
//!
//! # WHAT IS IN THE CONTAINER
//!
//! One container holds BOTH networks and the front end's constants:
//!
//! * **palm** -- MediaPipe's palm detector, 192x192, 87 graph nodes, 30 dense
//!   convolutions and 23 depthwise ones, and a detection head that reshapes
//!   four NCHW feature maps into 2016 rows of 18 numbers plus 2016 scores;
//! * **lm** -- MediaPipe's hand-landmark network, 224x224, 58 nodes, 31 dense
//!   and 16 depthwise convolutions, and a head that global-average-pools one
//!   672x7x7 map and projects it to 63 + 1 + 1 + 63.
//!
//! They are one file because a hand is not a palm and a palm is not a hand:
//! the second consumes a crop rotated by an angle the FIRST predicted. The
//! dependency cannot live between two containers, and a runtime handed two
//! files would have to trust the caller to have paired a detector with a
//! landmark net trained for its output.
//!
//! # THE VOCABULARY IS SIX OPS, AND IT IS NOT POSE'S SEVEN
//!
//! `conv, dwconv, add, maxpool, pad_c, resize`. Three of these do not appear
//! in the pose graph and three of pose's do not appear here, which is why the
//! op list is parsed rather than assumed:
//!
//! * **dwconv** -- ONE filter per channel. 39 of this container's 100
//!   convolutions are depthwise, and each is kept as its own op rather than
//!   quietly folded into a dense convolution, because a gate that cannot tell
//!   the two apart has stopped measuring anything;
//! * **pad_c** -- append N ZERO CHANNELS. It is a channel-wise concat with a
//!   constant, not a pixel pad: reading it as a pixel pad gives the right
//!   channel count and a wrong tensor;
//! * **resize** -- ONNX Resize, linear, half_pixel. The other three coordinate
//!   transforms are each off by half a source pixel, which produces a detector
//!   that finds a hand a fraction of a pixel out at every level.
//!
//! # THE ARITHMETIC IS TRANSCRIBED FROM THE ORACLE'S OBJECT FILE
//!
//! Where the reference source leaves a multiply and an add adjacent, the compiler
//! fused some of them and not others, and the difference is one rounding. The
//! four places it matters here were read off the oracle's disassembly
//! (`objdump -d runtime/build/libnpue_core.a`, `vfmadd`/`vmulss`):
//!
//! | site | the oracle's instruction | this port |
//! |---|---|---|
//! | dense conv | `gemm_nt`, bias added after the K reduction | `kernels::gemm_nt`, shared |
//! | dwconv | `vfmadd231ss` in the tap loop | `w.mul_add(src, acc)` |
//! | resize | `vmulss` then `vfmadd231ss`, both passes | the same split, both passes |
//! | add | `vaddss` only | a plain `+` |
//!
//! The depthwise and resize loops are the two that are NOT "a reasonable f32
//! implementation": an unfused `acc += w * x` there costs one extra rounding
//! per tap, which after 39 depthwise convolutions and two resizes is a
//! different hand.
//!
//! # TWO NETWORKS, TWO WEIGHT TABLES
//!
//! Both graphs number their convolutions from zero, so an unprefixed lookup
//! reads the OTHER network's weights and returns a tensor of the wrong width.
//! The container namespaces them (`palm.conv.N`, `lm.conv.N`) and so does
//! this reader.

use crate::kernels;
use crate::pose::Tensor;
use npum_format::Container;
use npum_model::array::{ArrayRef, ConvSpec};
use serde_json::Value;
use std::path::Path;

/// The arch string this reader owns.
pub const ARCH: &str = "mediapipe_hands_palm_ssd_lm_heatmap";

/// Keypoints the palm detector emits: a wrist and six finger landmarks.
pub const PALM_LANDMARKS: i64 = 7;

/// Keypoints the landmark network emits -- MediaPipe's twenty-one.
pub const HAND_LANDMARKS: i64 = 21;

/// The palm detector's box row: 4 box terms and 7 keypoint pairs.
pub const PALM_ROW_TERMS: i64 = 18;

/// The activation, as the graph records it.
///
/// Distinct from "what the epilogue can compute" on purpose: a container that
/// recorded one thing while the runtime believed another is the failure this
/// enum prevents by refusing anything but its four spellings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    /// The identity.
    None,
    /// `max(0, x)`.
    Relu,
    /// `min(max(0, x), 6)`.
    Relu6,
    /// Per-channel slope: `x >= 0 ? x : slope[c] * x`.
    PreLu,
}

/// The graph's op kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// A dense convolution.
    Conv,
    /// One filter per channel.
    DwConv,
    /// The residual, with the activation possibly fused onto the sum.
    Add,
    /// The pooling.
    MaxPool,
    /// Append zero channels.
    PadC,
    /// ONNX Resize, linear, half_pixel.
    Resize,
}

impl Op {
    /// The name the container spells.
    pub fn as_str(self) -> &'static str {
        match self {
            Op::Conv => "conv",
            Op::DwConv => "dwconv",
            Op::Add => "add",
            Op::MaxPool => "maxpool",
            Op::PadC => "pad_c",
            Op::Resize => "resize",
        }
    }
}

/// One node of one network's packed graph.
#[derive(Debug, Clone)]
pub struct Layer {
    /// What the node does.
    pub op: Op,
    /// The convolution's index into the geometry's table.
    pub conv: i32,
    /// The fused activation, where the op has an epilogue for one.
    pub act: Act,
    /// Which PReLU slope vector, or -1.
    pub prelu: i32,
    /// The stride's rows.
    pub stride_h: i64,
    /// The stride's columns.
    pub stride_w: i64,
    /// The padding: top, left, bottom, right -- FOUR terms, because MediaPipe
    /// evaluates SAME padding at a stride and the stem is [1,1,2,2].
    pub pad: [i64; 4],
    /// The pooling kernel's rows.
    pub kernel_h: i64,
    /// The pooling kernel's columns.
    pub kernel_w: i64,
    /// The Resize's vertical scale.
    pub scale_h: f64,
    /// The Resize's horizontal scale.
    pub scale_w: f64,
    /// PadC: how many zero channels to append.
    pub count: i64,
    /// The declared output channels.
    pub out_c: i64,
    /// The declared output rows.
    pub out_h: i64,
    /// The declared output columns.
    pub out_w: i64,
    /// The array's stream this convolution runs on, when the container was
    /// packed with `--npu`.
    ///
    /// It is the SHAPE the convolution is dispatched against -- "conv64x128"
    /// -- and it is EMPTY for a container packed without the panels: such a
    /// container names no stream, and a convolution with no stream is a
    /// convolution the array cannot be asked about.
    pub stream: String,
    /// The network this node belongs to -- "palm" or "lm".
    ///
    /// It is the NAMESPACE of the network's tensors, and it is what makes a
    /// panel's name: the two graphs number their convolutions from zero, so
    /// "conv.3" is two different weights in one container, and the panel
    /// that goes with it is named by the network too.
    pub net: String,
    /// The operands; -1 is the image.
    pub inputs: Vec<i32>,
}

/// One convolution's weights, `[cout, cin, kh, kw]`, widened to fp32 at load.
#[derive(Debug, Clone)]
pub struct ConvW {
    /// Its own index in the geometry's table.
    pub index: i32,
    /// Output channels.
    pub cout: i64,
    /// Input channels *per group*.
    pub cin: i64,
    /// Kernel rows.
    pub kh: i64,
    /// Kernel columns.
    pub kw: i64,
    /// 1 for dense, `cout` for depthwise.
    pub group: i64,
    /// The filter, `[cout, cin, kh, kw]` row-major.
    pub w: Vec<f32>,
    /// The bias, `[cout]`, or empty.
    pub b: Vec<f32>,
}

impl ConvW {
    /// The reduction width the dense GEMM needs.
    pub fn kin(&self) -> i64 {
        self.cin * self.kh * self.kw
    }

    /// Whether this is the depthwise case.
    pub fn depthwise(&self) -> bool {
        self.group > 1
    }
}

/// The hands family's dense convolutions, as the array's list.
///
/// THE LIST IS THE FAMILY'S OWN, and it is built from the GRAPHS rather than
/// from the container's directory: a panel exists for a convolution the graph
/// RUNS, and a panel that no op calls is a panel nobody stages.
///
/// A convolution with no stream is left out, and that is what a container
/// packed WITHOUT `--npu` looks like: the packer writes the stream into the
/// node when it stages the panel, so a container without panels names no
/// stream at all. The refusal for that container is the CLI's, and it names
/// the packing flag.
///
/// THE DEPTHWISE CONVOLUTIONS ARE LEFT OUT ON PURPOSE. A depthwise filter
/// reduces WITHIN one channel and has no `[M, N]` product in it at all: the
/// array's instruction stream wants a K-reduction across N output columns,
/// and a filter that reduces along K within one column is not that.
pub fn array_specs(geometry: &Geometry) -> Vec<ConvSpec> {
    let mut specs = Vec::new();
    for (prefix, graph, convs) in [
        ("palm", &geometry.palm_graph, &geometry.palm_convs),
        ("lm", &geometry.lm_graph, &geometry.lm_convs),
    ] {
        for l in graph {
            if l.op != Op::Conv || l.stream.is_empty() {
                continue;
            }
            let Some(cw) = convs.get(l.conv as usize) else {
                continue;
            };
            if cw.depthwise() || cw.w.is_empty() {
                continue;
            }
            specs.push(ConvSpec {
                weight: format!("{prefix}.conv.{}.w", cw.index),
                stream: l.stream.clone(),
                k: cw.kin() as usize,
                n: cw.cout as usize,
                bias: if cw.b.is_empty() {
                    None
                } else {
                    Some(cw.b.clone())
                },
            });
        }
    }
    specs
}

/// One PReLU's slope vector.
#[derive(Debug, Clone)]
pub struct Slope {
    /// The slopes, one per channel.
    pub v: Vec<f32>,
}

/// The palm detector's head, per pyramid level.
///
/// `box` and `score` are GRAPH NODE indices, not tensor indices: the head
/// consumes four feature maps that live in the middle of the graph.
#[derive(Debug, Clone)]
pub struct PalmLevel {
    /// The node holding the box regressions.
    pub box_: i32,
    /// The node holding the scores.
    pub score_: i32,
    /// The level's cells per side.
    pub h: i64,
    /// The level's cells per side (columns).
    pub w: i64,
    /// Anchors per cell.
    pub anchors_per_cell: i64,
    /// `h * w * anchors_per_cell`, recorded and checked.
    pub rows: i64,
}

/// One projection of the landmark head.
#[derive(Debug, Clone)]
pub struct LmProj {
    /// `[in, out]`, ONNX's Gemm order.
    pub w: Vec<f32>,
    /// The bias, or empty.
    pub b: Vec<f32>,
    /// How many outputs.
    pub n: i64,
}

/// One of the landmark head's four declared outputs.
#[derive(Debug, Clone)]
pub struct LmOutSpec {
    /// How many numbers.
    pub n: i64,
    /// Whether the numbers are logits the head must squash.
    pub sigmoid: bool,
    /// Which projection they come from.
    pub proj: usize,
}

/// The landmark network's head: one pool, then four Gemms.
#[derive(Debug, Clone)]
pub struct LmHead {
    /// The graph node whose output is pooled.
    pub pool: i32,
    /// The pooled width.
    pub features: i64,
    /// The pooled map's height, recorded.
    pub height: i64,
    /// The pooled map's width, recorded.
    pub width: i64,
    /// The projections.
    pub projs: Vec<LmProj>,
    /// The declared outputs, in the container's order.
    pub outputs: Vec<LmOutSpec>,
}

/// One crop's constants: a shift as a fraction of the box, and an enlargement.
#[derive(Debug, Clone)]
pub struct Crop {
    /// The shift, in fractions of the box's own size.
    pub shift: [f64; 2],
    /// The enlargement factor about the shifted centre.
    pub enlarge: f64,
}

/// The model's geometry, as the container states it.
#[derive(Debug, Clone)]
pub struct Geometry {
    /// The palm detector's input side.
    pub palm_input_size: i64,
    /// The landmark network's input side.
    pub lm_input_size: i64,
    /// How many hand landmarks.
    pub num_landmarks: i64,
    /// Per-channel means, shared by both networks.
    pub mean: Vec<f32>,
    /// Per-channel standard deviations, shared by both networks.
    pub std_dev: Vec<f32>,
    /// How many anchors the palm head emits.
    pub palm_num_anchors: i64,
    /// How many terms per anchor row.
    pub palm_row_terms: i64,
    /// Which palm keypoint is the wrist.
    pub palm_lm_wrist: i64,
    /// Which one is the base of the middle finger.
    pub palm_lm_middle_base: i64,
    /// The rotation crop's geometry.
    pub palm_pre: Crop,
    /// The final crop's geometry.
    pub palm_post: Crop,
    /// The reported box's geometry.
    pub hand: Crop,
    /// The score threshold, from the container.
    pub score_threshold: f64,
    /// The NMS IoU threshold, from the container.
    pub nms_threshold: f64,
    /// The cap on survivors, from the container.
    pub top_k: i64,
    /// The pyramid, per level.
    pub palm_levels: Vec<PalmLevel>,
    /// The landmark head.
    pub lm_head: LmHead,
    /// The palm graph.
    pub palm_graph: Vec<Layer>,
    /// The landmark graph.
    pub lm_graph: Vec<Layer>,
    /// The palm convolutions.
    pub palm_convs: Vec<ConvW>,
    /// The landmark convolutions.
    pub lm_convs: Vec<ConvW>,
    /// Palm node names, for refusals that quote one.
    pub palm_nodes: Vec<String>,
    /// Landmark node names.
    pub lm_nodes: Vec<String>,
    /// The palm PReLU slopes.
    pub palm_slopes: Vec<Slope>,
    /// The landmark PReLU slopes.
    pub lm_slopes: Vec<Slope>,
    /// The 2016 anchor centres, GENERATED in the graph's own row order.
    pub anchors: Vec<f64>,
}

impl Geometry {
    /// How many dense convolutions the palm detector has.
    pub fn palm_dense(&self) -> i64 {
        self.palm_convs.iter().filter(|c| !c.depthwise()).count() as i64
    }

    /// How many depthwise convolutions the palm detector has.
    pub fn palm_depthwise(&self) -> i64 {
        self.palm_convs.iter().filter(|c| c.depthwise()).count() as i64
    }

    /// How many dense convolutions the landmark network has.
    pub fn lm_dense(&self) -> i64 {
        self.lm_convs.iter().filter(|c| !c.depthwise()).count() as i64
    }

    /// How many depthwise convolutions the landmark network has.
    pub fn lm_depthwise(&self) -> i64 {
        self.lm_convs.iter().filter(|c| c.depthwise()).count() as i64
    }
}

/// What the palm detector's head produces: `[n, 18]` rows and `n` scores.
#[derive(Debug, Clone, Default)]
pub struct PalmOut {
    /// `n * row_terms`, row-major.
    pub boxes: Vec<f64>,
    /// `n` raw scores -- LOGITS, so the decode applies the sigmoid.
    pub scores: Vec<f64>,
    /// How many anchors.
    pub n: i64,
}

/// What the landmark network produces, in the container's output order.
#[derive(Debug, Clone, Default)]
pub struct LmOut {
    /// One vector per declared output.
    pub outs: Vec<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// Reading the container
// ---------------------------------------------------------------------------

/// The f32 view of a tensor payload.
fn f32s(container: &Container, name: &str, label: &str) -> Result<Vec<f32>, String> {
    let bytes = container
        .tensor_bytes(name)
        .map_err(|_| format!("{label}: the container holds no tensor named {name}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// The byte count a tensor's payload has, and its shape.
fn tensor_shape(container: &Container, name: &str) -> Option<(Vec<i64>, u64)> {
    container
        .directory()
        .tensor(name)
        .map(|e| (e.logical_shape.clone(), e.nbytes))
}

/// A container key as doubles.
fn doubles(container: &Container, key: &str, want: usize, label: &str) -> Result<Vec<f64>, String> {
    let value = container
        .config()
        .get(key)
        .ok_or_else(|| format!("{label}: {key} is not in the container"))?;
    let array = value
        .as_array()
        .ok_or_else(|| format!("{label}: {key} is not readable JSON"))?;
    let out: Vec<f64> = array.iter().filter_map(|v| v.as_f64()).collect();
    if want > 0 && out.len() != want {
        return Err(format!(
            "{label}: {key} has {} entries, expected {want}",
            out.len()
        ));
    }
    Ok(out)
}

/// A container key as parsed JSON (the graphs and heads are JSON strings).
fn json(container: &Container, key: &str, label: &str) -> Result<Value, String> {
    let raw = container
        .config()
        .get_str(key)
        .ok_or_else(|| format!("{label}: {key} is not in the container"))?;
    serde_json::from_str(raw)
        .map_err(|e| format!("{label}: {key} is not readable JSON ({e}): {raw}"))
}

/// A required integer.
fn need_i64(container: &Container, key: &str, label: &str) -> Result<i64, String> {
    container.config().get_i64(key).ok_or_else(|| {
        format!(
            "{label}: no '{key}'. The packed graph and this runtime have to agree \
             about the model's own fields, and a key that is not there is a packer \
             and a runtime from different days (arch {}).",
            container.config().get_str("arch").unwrap_or("?")
        )
    })
}

/// A required double.
fn need_f64(container: &Container, key: &str, label: &str) -> Result<f64, String> {
    container
        .config()
        .get_f64(key)
        .ok_or_else(|| format!("{label}: {key} is not in the container"))
}

/// A pair, as two integers.
fn pair(node: &Value, key: &str, label: &str) -> Result<(i64, i64), String> {
    let array = node
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{label}: '{key}' is not a pair"))?;
    if array.len() != 2 {
        return Err(format!(
            "{label}: '{key}' is not a pair. Every stride, kernel and scale in these \
             graphs is per axis and both axes have to be stated: these two networks \
             are square so far, but 'square so far' is not something to assume about \
             a packed graph."
        ));
    }
    Ok((
        array[0].as_i64().unwrap_or(0),
        array[1].as_i64().unwrap_or(0),
    ))
}

/// The four padding terms, refused unless there are four.
fn pad4(node: &Value, label: &str) -> Result<[i64; 4], String> {
    let array = node
        .get("pad")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{label}: 'pad' is not four terms [top, left, bottom, right]"))?;
    if array.len() != 4 {
        return Err(format!(
            "{label}: 'pad' is not four terms [top, left, bottom, right]. It is four \
             and not two because arch=7 pads ASYMMETRICALLY: MediaPipe evaluates \
             SAME padding at a stride, so the palm detector's 3x3 stride-2 stem is \
             [1,1,2,2]. Transcribing that as a symmetric pad shifts the whole \
             pyramid by one pixel, which is invisible in the output's shape and \
             moves every detection."
        ));
    }
    Ok([
        array[0].as_i64().unwrap_or(0),
        array[1].as_i64().unwrap_or(0),
        array[2].as_i64().unwrap_or(0),
        array[3].as_i64().unwrap_or(0),
    ])
}

/// The op, by name.
fn op_from(name: &str, label: &str) -> Result<Op, String> {
    Ok(match name {
        "conv" => Op::Conv,
        "dwconv" => Op::DwConv,
        "add" => Op::Add,
        "maxpool" => Op::MaxPool,
        "pad_c" => Op::PadC,
        "resize" => Op::Resize,
        other => {
            return Err(format!(
                "{label}: unknown op '{other}' in the packed graph. This build walks \
                 conv, dwconv, add, maxpool, pad_c and resize; a packer that emits \
                 anything else produced a network this runtime would have to guess \
                 at. The two networks' op vocabularies are their own -- arch=7 has no \
                 upsample and no concat because neither network has one -- so this \
                 list is not arch=6's."
            ));
        }
    })
}

/// The activation, by name.
fn act_from(name: &str, label: &str) -> Result<Act, String> {
    Ok(match name {
        "none" | "null" => Act::None,
        "relu" => Act::Relu,
        "relu6" => Act::Relu6,
        "prelu" => Act::PreLu,
        other => {
            return Err(format!(
                "{label}: activation '{other}' is not one of none/relu/relu6/prelu. \
                 These two networks fuse their activation into the convolution's \
                 epilogue or into the residual add, and an unknown one is not a pass \
                 this runtime can invent -- it would be a different network."
            ));
        }
    })
}

/// The slope vector's length: the product of the shape, not `shape[0]`.
///
/// The slope is stored `[1, C, 1, 1]`, and reading `shape[0]` gives 1 -- a
/// PReLU with one slope for a 32-channel tensor.
fn slope_len(shape: &[i64]) -> i64 {
    shape.iter().copied().product::<i64>().max(1)
}

/// One convolution, widened or dequantised to fp32.
///
/// `dtype` is the container's `conv_weight_dtype` and `i4_group` its
/// `conv_int4_group`. The BYTE COUNT is checked against the dtype before
/// anything is read: a payload that contradicts its declared precision used
/// to be read as a plausible fp32 array of the wrong length.
fn read_conv(
    container: &Container,
    prefix: &str,
    index: i32,
    want_depthwise: bool,
    dtype: &str,
    i4_group: i64,
    label: &str,
) -> Result<ConvW, String> {
    let base = format!("{prefix}.conv.{index}.w");
    let (shape, nbytes) = tensor_shape(container, &base)
        .ok_or_else(|| format!("{label}: the container holds no tensor named {base}"))?;
    if shape.len() != 4 {
        return Err(format!("{label}: {base} is not a 4-D filter"));
    }
    let (cout, cin, kh, kw) = (shape[0], shape[1], shape[2], shape[3]);
    if want_depthwise && cin != 1 {
        return Err(format!(
            "{label}: {base} is a dwconv with {cin} input channels. A depthwise \
             filter is ONE filter per channel; a dense GEMM over this one would \
             compute a different network."
        ));
    }
    let want = cout * cin * kh * kw;
    let n = cout;
    let k = cin * kh * kw;
    let w: Vec<f32> = match dtype {
        "f32" => {
            if nbytes as i64 != want * 4 {
                return Err(format!(
                    "{label}: {base} is {nbytes} bytes, which is not {} fp32 values \
                     for a [{cout}, {cin}, {kh}, {kw}] weight.",
                    want * 4
                ));
            }
            f32s(container, &base, label)?
        }
        "bf16" => {
            if nbytes as i64 != want * 2 {
                return Err(format!(
                    "{label}: {base} is {nbytes} bytes, which is not {} for {want} \
                     bf16 values.",
                    want * 2
                ));
            }
            let bytes = container
                .tensor_bytes(&base)
                .map_err(|_| format!("{label}: the container holds no tensor named {base}"))?;
            bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect()
        }
        "i8" | "i4" => {
            let packed = if dtype == "i8" { k } else { (k + 1) / 2 };
            if nbytes as i64 != n * packed {
                return Err(format!(
                    "{label}: {base} is {nbytes} bytes, which is not {} for a [{n}, \
                     {k}] {dtype} weight.",
                    n * packed
                ));
            }
            let scale_name = format!("{prefix}.conv.{index}.wscale");
            let scale = f32s(container, &scale_name, label).map_err(|_| {
                format!(
                    "{label}: {scale_name} is not in the container, and {base} says \
                     its weights are {dtype}. An integer payload without its scale is \
                     bytes whose meaning is one multiply away and not present in the \
                     file."
                )
            })?;
            if scale.len() as i64 != n {
                return Err(format!(
                    "{label}: {scale_name} has {} entries for {n} output channels.",
                    scale.len()
                ));
            }
            let bytes = container
                .tensor_bytes(&base)
                .map_err(|_| format!("{label}: the container holds no tensor named {base}"))?;
            let mut out = vec![0.0f32; want as usize];
            if dtype == "i8" {
                for nn in 0..n {
                    for kk in 0..k {
                        let q = bytes[(nn * k + kk) as usize] as i8;
                        out[(nn * k + kk) as usize] = q as f32 * scale[nn as usize];
                    }
                }
            } else {
                let gname = format!("{prefix}.conv.{index}.gscale");
                let gscale = f32s(container, &gname, label).map_err(|_| {
                    format!(
                        "{label}: {gname} is not in the container, and {base} says its \
                         weights are i4 with a group of {i4_group}."
                    )
                })?;
                let g = if i4_group > 0 {
                    (k + i4_group - 1) / i4_group
                } else {
                    1
                };
                if gscale.len() as i64 != g * n {
                    return Err(format!(
                        "{label}: {gname} is {} entries, not {} -- [{g}, {n}] for a \
                         group of {i4_group} over K={k}.",
                        gscale.len(),
                        g * n
                    ));
                }
                for nn in 0..n {
                    for kk in 0..k {
                        let byte = bytes[(nn * ((k + 1) / 2) + kk / 2) as usize];
                        let mut v = if kk & 1 == 1 {
                            ((byte >> 4) & 0x0F) as i32
                        } else {
                            (byte & 0x0F) as i32
                        };
                        if v >= 8 {
                            v -= 16;
                        }
                        let gsel = if i4_group > 0 { kk / i4_group } else { 0 };
                        out[(nn * k + kk) as usize] =
                            v as f32 * scale[nn as usize] * gscale[(gsel * n + nn) as usize];
                    }
                }
            }
            out
        }
        other => {
            return Err(format!(
                "{label}: conv_weight_dtype is '{other}', and this runtime reads f32, \
                 bf16, i8 and i4. There is no fp16 or fp8 convolution weight here, and \
                 reading either as one of the four produces a network with the right \
                 shapes and no relationship to the checkpoint."
            ));
        }
    };
    let bname = format!("{prefix}.conv.{index}.b");
    let b = if tensor_shape(container, &bname).is_some() {
        f32s(container, &bname, label)?
    } else {
        Vec::new()
    };
    if !b.is_empty() && b.len() as i64 != cout {
        return Err(format!(
            "{label}: {bname} holds {} values and the convolution has {cout} output \
             channels.",
            b.len()
        ));
    }
    Ok(ConvW {
        index,
        cout,
        cin,
        kh,
        kw,
        group: if want_depthwise { cout } else { 1 },
        w,
        b,
    })
}

/// One network's graph, weights and slopes.
struct GraphRead {
    layers: Vec<Layer>,
    convs: Vec<ConvW>,
    nodes: Vec<String>,
    slopes: Vec<Slope>,
}

/// Read one network. `prefix` namespaces its tensors.
fn read_one_graph(
    container: &Container,
    prefix: &str,
    graph_key: &str,
    nodes_key: &str,
    num_convs: i64,
    dtype: &str,
    i4_group: i64,
    label: &str,
) -> Result<GraphRead, String> {
    let ops = json(container, graph_key, label)?;
    let names = json(container, nodes_key, label)?;
    let ops = ops
        .as_array()
        .ok_or_else(|| format!("{label}: {graph_key} is not a list of nodes"))?;
    let names = names
        .as_array()
        .ok_or_else(|| format!("{label}: {nodes_key} is not a list of names"))?;
    if ops.len() != names.len() {
        return Err(format!(
            "{label}: {graph_key} has {} nodes and {nodes_key} has {}. The second list \
             is what a refusal quotes by name, and a refusal that quotes node 7 when \
             node 7 does not exist is worse than none.",
            ops.len(),
            names.len()
        ));
    }
    let mut layers = Vec::with_capacity(ops.len());
    let mut node_names = Vec::with_capacity(ops.len());
    let mut max_conv = -1i64;
    let mut max_prelu = -1i64;
    for (i, node) in ops.iter().enumerate() {
        let name = names[i].as_str().unwrap_or("?");
        let nl = format!("{label}/{graph_key}: node [{i}] '{name}'");
        let op = op_from(node.get("op").and_then(|v| v.as_str()).unwrap_or(""), &nl)?;
        // conv, dwconv and add all have an epilogue to fuse into, so `act` is
        // REQUIRED there: a packer that omitted it would be saying "and no
        // activation", and this runtime cannot tell that apart from a packer
        // that forgot. The other three have no epilogue and read the absence
        // as `none`.
        let act = match op {
            Op::Conv | Op::DwConv | Op::Add => act_from(
                node.get("act")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("{nl}: op '{}' carries no 'act'", op.as_str()))?,
                &nl,
            )?,
            _ => act_from(
                node.get("act").and_then(|v| v.as_str()).unwrap_or("none"),
                &nl,
            )?,
        };
        // An explicit JSON null is an ABSENT value, not a zero: every op whose
        // activation is not a PReLU writes null, and reading that as 0 would
        // make all twenty-six of arch=7's PReLUs into the first one's slopes.
        let prelu = node
            .get("prelu")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32)
            .unwrap_or(-1);
        if act == Act::PreLu {
            if prelu < 0 {
                return Err(format!(
                    "{nl}: the op records a prelu activation and no slope index"
                ));
            }
            max_prelu = max_prelu.max(prelu as i64);
        }
        if act != Act::PreLu && prelu >= 0 {
            return Err(format!(
                "{nl}: carries prelu slope {prelu} and the activation none. Both are \
                 written by the same packer pass over the same fused-activation node, \
                 so this is a half-transcribed op rather than a network whose \
                 activation happens to be unused."
            ));
        }
        let mut layer = Layer {
            op,
            conv: -1,
            act,
            prelu,
            stride_h: 1,
            stride_w: 1,
            pad: [0; 4],
            kernel_h: 1,
            kernel_w: 1,
            scale_h: 1.0,
            scale_w: 1.0,
            count: 0,
            out_c: 0,
            out_h: 0,
            out_w: 0,
            stream: String::new(),
            net: prefix.to_string(),
            inputs: Vec::new(),
        };
        match op {
            Op::Conv | Op::DwConv => {
                layer.conv = node
                    .get("conv")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| format!("{nl}: no 'conv' index"))?
                    as i32;
                if layer.conv < 0 {
                    return Err(format!("{nl}: no 'conv' index"));
                }
                if layer.conv as i64 >= num_convs {
                    return Err(format!(
                        "{nl}: conv {} is past the container's {num_convs} \
                         convolutions for this network",
                        layer.conv
                    ));
                }
                max_conv = max_conv.max(layer.conv as i64);
                layer.pad = pad4(node, &nl)?;
                let (sh, sw) = pair(node, "stride", &nl)?;
                layer.stride_h = sh;
                layer.stride_w = sw;
                // THE STREAM, when the container names one.
                //
                // It is the packer's word: the packer writes the design's
                // stream into the node it stages the panel for, and a
                // container packed WITHOUT the panels names none. A default
                // invented here would be a stream the design does not carry,
                // and a convolution dispatched against a missing slot is a
                // wrong answer rather than an error.
                layer.stream = node
                    .get("stream")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
            }
            Op::MaxPool => {
                let (kh, kw) = pair(node, "kernel", &nl)?;
                layer.kernel_h = kh;
                layer.kernel_w = kw;
                let (sh, sw) = pair(node, "stride", &nl)?;
                layer.stride_h = sh;
                layer.stride_w = sw;
                layer.pad = pad4(node, &nl)?;
            }
            Op::Resize => {
                let array = node
                    .get("scale")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| format!("{nl}: 'scale' is not a pair"))?;
                if array.len() != 2 {
                    return Err(format!("{nl}: 'scale' is not a pair"));
                }
                layer.scale_h = array[0].as_f64().unwrap_or(1.0);
                layer.scale_w = array[1].as_f64().unwrap_or(1.0);
            }
            Op::PadC => {
                layer.count = node
                    .get("count")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| format!("{nl}: no 'count'"))?;
                if layer.count < 0 {
                    return Err(format!("{nl}: pad_c count is negative"));
                }
            }
            Op::Add => {}
        }
        let out = node
            .get("out")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("{nl}: no 'out'"))?;
        if out.len() != 4 || out[0].as_i64() != Some(1) {
            return Err(format!("{nl}: 'out' is not a [1, C, H, W] shape"));
        }
        layer.out_c = out[1].as_i64().unwrap_or(0);
        layer.out_h = out[2].as_i64().unwrap_or(0);
        layer.out_w = out[3].as_i64().unwrap_or(0);
        layer.inputs = node
            .get("inputs")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_i64())
                    .map(|v| v as i32)
                    .collect()
            })
            .unwrap_or_default();
        if layer.inputs.is_empty() {
            return Err(format!("{nl}: no 'inputs'"));
        }
        for input in &layer.inputs {
            if *input != -1 && *input >= 0 && (*input as usize) >= i {
                return Err(format!(
                    "{nl}: takes operand {input}, which is not an earlier op and not \
                     the image (-1). This graph is evaluated in order, so a forward \
                     reference is not a cycle the runtime could resolve -- it is a \
                     graph whose outputs it cannot compute."
                ));
            }
        }
        layers.push(layer);
        node_names.push(name.to_string());
    }

    let n_conv = num_convs.max(max_conv + 1) as usize;
    let mut convs: Vec<Option<ConvW>> = Vec::with_capacity(n_conv);
    for i in 0..n_conv as i64 {
        let base = format!("{prefix}.conv.{i}.w");
        if tensor_shape(container, &base).is_none() {
            // A sparse index: the packer numbered the convolutions from the
            // ONNX's own order, and the op list says which indices are real.
            convs.push(None);
            continue;
        }
        let dw = layers.iter().any(|l| {
            l.conv == i as i32 && (l.op == Op::Conv || l.op == Op::DwConv) && l.op == Op::DwConv
        });
        convs.push(Some(read_conv(
            container, prefix, i as i32, dw, dtype, i4_group, label,
        )?));
    }
    let convs: Vec<ConvW> = convs.into_iter().flatten().collect();
    // The slope count is DERIVED from the op list rather than declared in the
    // config: an activation naming a slope index is the only consumer of these
    // vectors, and a declared count could disagree with the list.
    let mut slopes = Vec::with_capacity((max_prelu + 1).max(0) as usize);
    for i in 0..=max_prelu {
        let base = format!("{prefix}.prelu.{i}.slope");
        let (shape, _) = tensor_shape(container, &base).ok_or_else(|| {
            format!("{label}: the graph fuses prelu {i} and {base} is not in the container")
        })?;
        let v = f32s(container, &base, label)?;
        if v.len() as i64 != slope_len(&shape) {
            return Err(format!(
                "{label}: {base} holds {} values and its shape says {}.",
                v.len(),
                slope_len(&shape)
            ));
        }
        slopes.push(Slope { v });
    }
    Ok(GraphRead {
        layers,
        convs,
        nodes: node_names,
        slopes,
    })
}

/// Read the geometry out of a container.
pub fn read_geometry(container: &Container, label: &str) -> Result<Geometry, String> {
    let arch = container.config().get_str("arch").unwrap_or("");
    if arch != ARCH {
        return Err(format!(
            "{label}: this is an arch={arch} container and this build reads arch={ARCH}. \
             The two hand networks are not a pose model with its head turned off: they \
             carry depthwise convolutions the array cannot express, an asymmetric \
             padding convention, two different heads, and no normalisation at all."
        ));
    }
    let palm_input_size = need_i64(container, "palm_input_size", label)?;
    let lm_input_size = need_i64(container, "lm_input_size", label)?;
    let num_landmarks = need_i64(container, "num_landmarks", label)?;
    if num_landmarks != HAND_LANDMARKS {
        return Err(format!(
            "{label}: num_landmarks is {num_landmarks}, this architecture emits \
             {HAND_LANDMARKS} (MediaPipe's twenty-one: a wrist, four chains of four, \
             and four across the palm)"
        ));
    }
    let palm_num_anchors = need_i64(container, "palm_num_anchors", label)?;
    let palm_row_terms = need_i64(container, "palm_row_terms", label)?;
    if palm_row_terms != PALM_ROW_TERMS {
        return Err(format!(
            "{label}: palm_row_terms is {palm_row_terms}, expected {PALM_ROW_TERMS} \
             (4 box terms + 7 keypoints x 2)"
        ));
    }
    let palm_lm_wrist = need_i64(container, "palm_lm_wrist", label)?;
    let palm_lm_middle_base = need_i64(container, "palm_lm_middle_base", label)?;
    if !(0..PALM_LANDMARKS).contains(&palm_lm_wrist)
        || !(0..PALM_LANDMARKS).contains(&palm_lm_middle_base)
    {
        return Err(format!(
            "{label}: the rotation is taken between palm keypoints {palm_lm_wrist} and \
             {palm_lm_middle_base}, and the palm detector emits {PALM_LANDMARKS}. WHICH \
             two choose the rotation decides where the landmark network looks; a \
             different pair gives a plausible hand pointing sideways."
        ));
    }

    let crop = |skey: &str, ekey: &str| -> Result<Crop, String> {
        let v = doubles(container, skey, 2, label)?;
        Ok(Crop {
            shift: [v[0], v[1]],
            enlarge: need_f64(container, ekey, label)?,
        })
    };
    let palm_pre = crop("palm_pre_shift", "palm_pre_enlarge")?;
    let palm_post = crop("palm_shift", "palm_enlarge")?;
    let hand = crop("hand_shift", "hand_enlarge")?;
    let score_threshold = need_f64(container, "score_threshold", label)?;
    let nms_threshold = need_f64(container, "nms_threshold", label)?;
    let top_k = need_i64(container, "top_k", label)?;

    let mean: Vec<f32> = doubles(container, "image_mean", 3, label)?
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let std_dev: Vec<f32> = doubles(container, "image_std", 3, label)?
        .into_iter()
        .map(|v| v as f32)
        .collect();
    if std_dev.iter().any(|s| *s == 0.0) {
        return Err(format!(
            "{label}: image_std has a zero. Both networks normalise by it, so that is \
             a division by zero on every pixel of every frame."
        ));
    }

    // -- the palm head, one entry per pyramid level
    let head = json(container, "palm_head", label)?;
    let head = head
        .as_array()
        .ok_or_else(|| format!("{label}: palm_head is not a list of levels"))?;
    if head.is_empty() {
        return Err(format!("{label}: palm_head lists no levels"));
    }
    let mut palm_levels = Vec::with_capacity(head.len());
    for (i, level) in head.iter().enumerate() {
        let el = format!("{label}/palm_head[{i}]");
        let get = |key: &str| -> Result<i64, String> {
            level
                .get(key)
                .and_then(|v| v.as_i64())
                .ok_or_else(|| format!("{el}: no '{key}'"))
        };
        let l = PalmLevel {
            box_: get("box")? as i32,
            score_: get("score")? as i32,
            h: get("h")?,
            w: get("w")?,
            anchors_per_cell: get("anchors_per_cell")?,
            rows: get("rows")?,
        };
        if l.rows != l.h * l.w * l.anchors_per_cell {
            return Err(format!(
                "{el}: rows={} against h*w*a = {}. `rows` is the level's share of the \
                 anchor table and the decode reads it, so a disagreement here is a \
                 table read at the wrong stride.",
                l.rows,
                l.h * l.w * l.anchors_per_cell
            ));
        }
        palm_levels.push(l);
    }

    // -- the landmark head
    let h = json(container, "lm_head", label)?;
    let hl = format!("{label}/lm_head");
    let pool = h
        .get("pool")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| format!("{hl}: no 'pool'"))? as i32;
    let features = h
        .get("features")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| format!("{hl}: no 'features'"))?;
    let height = h
        .get("height")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| format!("{hl}: no 'height'"))?;
    let width = h
        .get("width")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| format!("{hl}: no 'width'"))?;
    let mut projs = Vec::new();
    for entry in h
        .get("projs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{hl}: no 'projs'"))?
    {
        let wname = entry
            .get("w")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{hl}: a projection has no 'w'"))?;
        let bname = entry
            .get("b")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{hl}: a projection has no 'b'"))?;
        let n = entry
            .get("n")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| format!("{hl}: a projection has no 'n'"))?;
        let (shape, _) = tensor_shape(container, wname)
            .ok_or_else(|| format!("{hl}: the container holds no tensor named {wname}"))?;
        if shape.len() != 2 {
            return Err(format!("{hl}: {wname} is not a 2-D projection"));
        }
        // Stored [in, out] and ONNX's Gemm with transB=0 is A @ W, so the
        // projection is W.T @ pooled. The other way round is a shape error
        // here and a silent TRANSPOSE in a runtime.
        if shape[1] != n {
            return Err(format!(
                "{hl}: {wname} has {} outputs, the config says {n}",
                shape[1]
            ));
        }
        if shape[0] != features {
            return Err(format!(
                "{hl}: {wname} takes {} inputs, the pooled map is {features} wide",
                shape[0]
            ));
        }
        projs.push(LmProj {
            w: f32s(container, wname, label)?,
            b: if tensor_shape(container, bname).is_some() {
                f32s(container, bname, label)?
            } else {
                Vec::new()
            },
            n,
        });
    }
    let mut outputs = Vec::new();
    for entry in h
        .get("outputs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{hl}: no 'outputs'"))?
    {
        let n = entry.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
        let sigmoid = entry
            .get("sigmoid")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let proj = entry
            .get("proj")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| format!("{hl}: an output has no 'proj'"))?;
        if proj < 0 || proj as usize >= projs.len() {
            return Err(format!(
                "{hl}: one output names projection {proj} of {}",
                projs.len()
            ));
        }
        if n != projs[proj as usize].n {
            return Err(format!(
                "{hl}: n={n} against projection {proj}'s {}. The two are read from one \
                 JSON and disagreeing means one of them is a transcription -- and the \
                 failure would be a hand with a plausible number of joints and the \
                 wrong ones.",
                projs[proj as usize].n
            ));
        }
        outputs.push(LmOutSpec {
            n,
            sigmoid,
            proj: proj as usize,
        });
    }
    let lm_head = LmHead {
        pool,
        features,
        height,
        width,
        projs,
        outputs,
    };

    // -- the weight precision
    let dtype = container
        .config()
        .get_str("conv_weight_dtype")
        .unwrap_or("f32");
    if !matches!(dtype, "f32" | "bf16" | "i8" | "i4") {
        return Err(format!(
            "{label}: conv_weight_dtype is '{dtype}', and this runtime reads f32, bf16, \
             i8 and i4."
        ));
    }
    let i4_group = match container.config().get_i64("conv_int4_group") {
        Some(v) => v,
        None if dtype == "i4" => {
            return Err(format!(
                "{label}: conv_weight_dtype is 'i4' and conv_int4_group is not in the \
                 container. The group size is what places each weight against its own \
                 group's scale, so a reader that had to default it would choose the \
                 stride of the dequantisation itself."
            ));
        }
        None => 0,
    };
    if i4_group < 0 {
        return Err(format!(
            "{label}: conv_int4_group is {i4_group}. Zero means one group over the \
             whole input axis -- per-channel int4 -- and is the only value below 1 \
             with a meaning."
        ));
    }
    if i4_group > 0 && dtype != "i4" {
        return Err(format!(
            "{label}: conv_int4_group is {i4_group} and conv_weight_dtype is '{dtype}'. \
             The group only means anything for four-bit weights, so a container that \
             states both has a scale the reader cannot place."
        ));
    }

    // -- the two graphs and their weights
    let palm = read_one_graph(
        container,
        "palm",
        "palm_graph",
        "palm_graph_nodes",
        need_i64(container, "palm_num_convs", label)?,
        dtype,
        i4_group,
        &format!("{label}/palm"),
    )?;
    let lm = read_one_graph(
        container,
        "lm",
        "lm_graph",
        "lm_graph_nodes",
        need_i64(container, "lm_num_convs", label)?,
        dtype,
        i4_group,
        &format!("{label}/lm"),
    )?;

    // -- the anchor table, GENERATED, in the graph's own row order
    //
    // The order is h outer and w inner, so the flat row index advances w -- the
    // X coordinate -- fastest, which is why the first component of a row is x.
    // The head reshapes its [H, W, a*18] map to [H*W*a, 18] that way; a C-major
    // flatten permutes every row and still leaves 2016 well-shaped rows.
    //
    // Cross-checked against palm_anchor_levels, a SECOND spelling of the same
    // pyramid. Neither is derived from the other at run time and they are
    // compared: two hand-written spellings of one list catch each other's
    // typos, and what they cannot catch is a formula wrong in both.
    let levels = json(container, "palm_anchor_levels", label)?;
    let levels = levels
        .as_array()
        .ok_or_else(|| format!("{label}: palm_anchor_levels is not a list"))?;
    if levels.len() != palm_levels.len() {
        return Err(format!(
            "{label}: palm_anchor_levels lists {} levels and the head has {}. These are \
             two spellings of one pyramid and they are compared here rather than \
             derived, so a disagreement is a refusal and not a 2016-row table of the \
             wrong shape.",
            levels.len(),
            palm_levels.len()
        ));
    }
    let mut anchors: Vec<f64> = Vec::with_capacity((palm_num_anchors * 2) as usize);
    for (i, level) in levels.iter().enumerate() {
        let array = level
            .as_array()
            .ok_or_else(|| format!("{label}: palm_anchor_levels[{i}] is not a list"))?;
        if array.len() != 3 {
            return Err(format!(
                "{label}: palm_anchor_levels[{i}] is not [h, w, anchors_per_cell]"
            ));
        }
        let (h, w, a) = (
            array[0].as_i64().unwrap_or(0),
            array[1].as_i64().unwrap_or(0),
            array[2].as_i64().unwrap_or(0),
        );
        let l = &palm_levels[i];
        if h != l.h || w != l.w || a != l.anchors_per_cell {
            return Err(format!(
                "{label}: palm_anchor_levels[{i}] is {h}x{w} with {a} anchors; the \
                 head's level is {}x{} with {}",
                l.h, l.w, l.anchors_per_cell
            ));
        }
        for y in 0..h {
            for x in 0..w {
                for _ in 0..a {
                    anchors.push((x as f64 + 0.5) / w as f64);
                    anchors.push((y as f64 + 0.5) / h as f64);
                }
            }
        }
    }
    if anchors.len() as i64 / 2 != palm_num_anchors {
        return Err(format!(
            "{label}: the levels generate {} anchors and the container declares \
             {palm_num_anchors}",
            anchors.len() / 2
        ));
    }

    Ok(Geometry {
        palm_input_size,
        lm_input_size,
        num_landmarks,
        mean,
        std_dev,
        palm_num_anchors,
        palm_row_terms,
        palm_lm_wrist,
        palm_lm_middle_base,
        palm_pre,
        palm_post,
        hand,
        score_threshold,
        nms_threshold,
        top_k,
        palm_levels,
        lm_head,
        palm_graph: palm.layers,
        lm_graph: lm.layers,
        palm_convs: palm.convs,
        lm_convs: lm.convs,
        palm_nodes: palm.nodes,
        lm_nodes: lm.nodes,
        palm_slopes: palm.slopes,
        lm_slopes: lm.slopes,
        anchors,
    })
}

// ---------------------------------------------------------------------------
// The network
// ---------------------------------------------------------------------------

/// One graph's `[K, N]` weight tables, transposed once at load.
struct NetWeights {
    /// Every dense filter, `[K, N]`, one after another.
    wmat: Vec<f32>,
    /// Where each convolution's matrix starts.
    at: Vec<usize>,
}

impl NetWeights {
    /// Transpose a graph's dense filters from the container's `[N, K]`.
    ///
    /// Depthwise filters are NOT transposed: their reduction is over kh*kw
    /// WITHIN one channel, so `dwconv` indexes the container's own layout and
    /// there is no `[K, N]` view to build. The entry stays at zero and `conv`
    /// refuses a dense convolution that lands on one.
    fn build(convs: &[ConvW]) -> NetWeights {
        let mut wmat = Vec::new();
        let mut at = Vec::with_capacity(convs.len());
        for conv in convs {
            at.push(wmat.len());
            if conv.depthwise() {
                continue;
            }
            let n = conv.cout as usize;
            let k = conv.kin() as usize;
            let base = wmat.len();
            wmat.resize(base + n * k, 0.0);
            for nn in 0..n {
                for kk in 0..k {
                    wmat[base + kk * n + nn] = conv.w[nn * k + kk];
                }
            }
        }
        NetWeights { wmat, at }
    }

    /// One convolution's matrix, or empty when it is depthwise.
    fn get(&self, conv: &ConvW) -> &[f32] {
        let start = self.at.get(conv.index as usize).copied().unwrap_or(0);
        let len = if conv.depthwise() {
            0
        } else {
            (conv.cout * conv.kin()) as usize
        };
        &self.wmat[start..start + len]
    }
}

/// What one graph node cost, for a dump that wants to say where the time went.
///
/// The CLI prints the split per image, and the interesting number on this
/// architecture is the SPLIT between the dense convolutions a panel could take
/// and the depthwise ones it cannot -- so the two are counted apart.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cost {
    /// Dense convolutions that ran on the host.
    pub convs_host: i64,
    /// Depthwise convolutions, which no array can take.
    pub dwconvs_host: i64,
    /// add, pad_c, resize and maxpool.
    pub elementwise_ops: i64,
    /// Seconds in im2col.
    pub t_im2col: f64,
    /// Seconds in the GEMM.
    pub t_gemm: f64,
    /// Seconds transposing `[M, N]` into NCHW.
    pub t_transpose: f64,
    /// Seconds in the depthwise convolution.
    pub t_dwconv: f64,
    /// Seconds in the elementwise passes.
    pub t_elementwise: f64,
    /// Seconds in the landmark head's global average pool.
    pub t_pool: f64,
    /// Seconds in the array's convolutions, im2col included.
    ///
    /// It is the WHOLE span of a convolution on the array -- from the im2col
    /// that fills A to the NCHW transpose of C -- and not the device's busy
    /// time: the array's own counters say the device is idle for most of it,
    /// and a status line that printed only the busy time would report a
    /// speedup the picture does not have.
    pub t_array_s: f64,
    /// Seconds inside the array's dispatch, the A repack excluded.
    pub t_array_gemm_s: f64,
    /// Seconds the host spent turning its activations into the device's dtype.
    pub t_array_repack_s: f64,
    /// Seconds transposing the array's result back into NCHW.
    pub t_array_transpose_s: f64,
    /// How many convolutions ran on the array.
    pub convs_array: i64,
    /// How many dispatches the array saw.
    pub dispatches: i64,
}

/// Both networks, one container.
pub struct Hands {
    geom: Geometry,
    palm: NetWeights,
    lm: NetWeights,
    threads: usize,
}

/// `act1` -- the three activations, as a function of one element.
///
/// One definition, called from the dense epilogue and the depthwise kernel
/// alike: the epilogue exists only because a GEMM's natural output is `[M, N]`
/// and the depthwise kernel never has that intermediate, and both must compute
/// the same arithmetic.
fn act1(a: Act, v: f32, slope: f32) -> f32 {
    match a {
        Act::None => v,
        Act::Relu => {
            if v > 0.0 {
                v
            } else {
                0.0
            }
        }
        Act::Relu6 => {
            if v < 0.0 {
                0.0
            } else if v > 6.0 {
                6.0
            } else {
                v
            }
        }
        Act::PreLu => {
            if v >= 0.0 {
                v
            } else {
                slope * v
            }
        }
    }
}

/// The graph's declared output extent, recomputed from the op and REFUSED if
/// it disagrees.
///
/// Every op carries its own `out`, written by the packer from the ONNX.
/// Recomputing it and comparing means a container whose padding or stride does
/// not divide its input is refused at load rather than producing a tensor of
/// the declared shape filled with whatever the kernel happened to write.
fn check_extent(
    layer: &Layer,
    h: i64,
    w: i64,
    kh: i64,
    kw: i64,
    label: &str,
) -> Result<(i64, i64), String> {
    let eh = h + layer.pad[0] + layer.pad[2] - kh;
    let ew = w + layer.pad[1] + layer.pad[3] - kw;
    if eh < 0 || ew < 0 || eh % layer.stride_h != 0 || ew % layer.stride_w != 0 {
        return Err(format!(
            "{label}: op '{}' has pad [{},{},{},{}] on a {h}x{w} input with a {kh}x{kw} \
             kernel at stride {},{} , which does not give a whole-number output. Every \
             convolution in these graphs pads asymmetrically on purpose -- MediaPipe's \
             SAME padding evaluated at a stride -- so this is the check that keeps a \
             symmetric transcription of it out.",
            layer.op.as_str(),
            layer.pad[0],
            layer.pad[1],
            layer.pad[2],
            layer.pad[3],
            layer.stride_h,
            layer.stride_w
        ));
    }
    let oh = eh / layer.stride_h + 1;
    let ow = ew / layer.stride_w + 1;
    if oh != layer.out_h || ow != layer.out_w {
        return Err(format!(
            "{label}: op '{}' computes {oh}x{ow} and the container says its output is \
             {}x{}. The two are written by the packer and here from the same op, so a \
             disagreement is a container whose op list was edited.",
            layer.op.as_str(),
            layer.out_h,
            layer.out_w
        ));
    }
    Ok((oh, ow))
}

/// im2col with FOUR padding terms.
///
/// The k index is `(ci*kh + i)*kw + j` and the row index is `y*W + x`, which is
/// the order the weights are read in too, so the GEMM's K reduction walks the
/// kernel exactly as the checkpoint stores it. Zero outside the window, and not
/// a clamp: a clamped edge replicates the border pixel, which is a different
/// network.
fn im2col(
    input: &Tensor,
    kh: i64,
    kw: i64,
    pad: [i64; 4],
    sh: i64,
    sw: i64,
    oh: i64,
    ow: i64,
) -> Vec<f32> {
    let (c, h, w) = (input.c, input.h, input.w);
    let m = oh * ow;
    let k = c * kh * kw;
    let mut a = vec![0.0f32; (m * k) as usize];
    for y in 0..oh {
        for x in 0..ow {
            let row = &mut a[((y * ow + x) * k) as usize..][..k as usize];
            for i in 0..kh {
                let sy = y * sh + i - pad[0];
                if sy < 0 || sy >= h {
                    continue;
                }
                for j in 0..kw {
                    let sx = x * sw + j - pad[1];
                    if sx < 0 || sx >= w {
                        continue;
                    }
                    for ch in 0..c {
                        row[((ch * kh + i) * kw + j) as usize] =
                            input.d[((ch * h + sy) * w + sx) as usize];
                    }
                }
            }
        }
    }
    a
}

/// The `[M, N]` to NCHW transpose, with the activation fused into it.
///
/// Tiled 32 rows at a time: a naive n-outer / m-inner copy writes M runs of one
/// float each for every channel and streams the destination as N simultaneous
/// write pointers.
fn transpose_mn_to_nchw(src: &[f32], m: i64, n: i64, dst: &mut [f32], act: Act, slope: &[f32]) {
    const TILE: i64 = 32;
    let mut m0 = 0i64;
    while m0 < m {
        let m1 = (m0 + TILE).min(m);
        for ch in 0..n {
            let sl = slope.get(ch as usize).copied().unwrap_or(0.0);
            let d = &mut dst[(ch * m) as usize..][..m as usize];
            for mm in m0..m1 {
                d[mm as usize] = act1(act, src[(mm * n + ch) as usize], sl);
            }
        }
        m0 = m1;
    }
}

/// The GEMM over ROW RANGES, one per thread.
///
/// A row's accumulation is over k in order, so splitting ROWS changes which
/// thread computes a value and never the value itself -- the split is bit-exact
/// by construction and a comparison against the oracle does not have to know
/// how many cores the machine has.
fn gemm_rows(
    a: &[f32],
    m: usize,
    b: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
    threads: usize,
) {
    let threads = threads.min(m).max(1);
    if threads <= 1 || m == 0 {
        kernels::gemm_nt(a, m, b, k, n, bias, out);
        return;
    }
    let chunk = m.div_ceil(threads);
    let parts: Vec<&mut [f32]> = out.chunks_mut(chunk * n).collect();
    std::thread::scope(|s| {
        for (ci, part) in parts.into_iter().enumerate() {
            let rows = part.len() / n;
            if rows == 0 {
                continue;
            }
            let a_part = &a[ci * chunk * k..(ci * chunk + rows) * k];
            s.spawn(move || kernels::gemm_nt(a_part, rows, b, k, n, bias, part));
        }
    });
}

/// How many threads the host convolutions are split over.
///
/// The machine's own count, asked rather than assumed: the status block prints
/// this number, and a printed count that did not come from the machine is a
/// claim about a run that did not happen.
pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

impl Hands {
    /// Read both graphs out of a container.
    pub fn open(path: &Path) -> Result<Hands, String> {
        let container = Container::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let label = path.display().to_string();
        let geom = read_geometry(&container, &label)?;
        let palm = NetWeights::build(&geom.palm_convs);
        let lm = NetWeights::build(&geom.lm_convs);
        Ok(Hands {
            geom,
            palm,
            lm,
            threads: default_threads(),
        })
    }

    /// The model's geometry.
    pub fn geometry(&self) -> &Geometry {
        &self.geom
    }

    /// How many threads the host convolutions are split over.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Walk the palm detector's graph. Returns every node's output, in order.
    pub fn run_palm(
        &self,
        input: &Tensor,
        hook: &mut dyn FnMut(usize, &Tensor),
    ) -> Result<Vec<Tensor>, String> {
        self.walk(
            input,
            &self.geom.palm_graph,
            &self.geom.palm_convs,
            &self.palm,
            &self.geom.palm_slopes,
            &self.geom.palm_nodes,
            "palm",
            hook,
            None,
        )
    }

    /// Walk the landmark network's graph. Returns every node's output.
    pub fn run_lm(
        &self,
        input: &Tensor,
        hook: &mut dyn FnMut(usize, &Tensor),
    ) -> Result<Vec<Tensor>, String> {
        self.walk(
            input,
            &self.geom.lm_graph,
            &self.geom.lm_convs,
            &self.lm,
            &self.geom.lm_slopes,
            &self.geom.lm_nodes,
            "lm",
            hook,
            None,
        )
    }

    /// The walk: the graph in order, the hook on every node.
    ///
    /// EVERY NODE IS KEPT, not the live ones, because the heads read nodes in
    /// the MIDDLE of the graph -- the palm head's two levels are nodes 74/75
    /// and 85/86 of 87. A walk that freed them would have to be told which to
    /// keep, and being told is being wrong: a head that read a freed buffer
    /// would get a plausible 2016 rows.
    #[allow(clippy::too_many_arguments)]
    fn walk(
        &self,
        input: &Tensor,
        graph: &[Layer],
        convs: &[ConvW],
        net: &NetWeights,
        slopes: &[Slope],
        names: &[String],
        which: &str,
        hook: &mut dyn FnMut(usize, &Tensor),
        array: ArrayRef<'_, '_>,
    ) -> Result<Vec<Tensor>, String> {
        let mut cost = Cost::default();
        self.walk_costed(
            input, graph, convs, net, slopes, names, which, hook, array, &mut cost,
        )
    }

    /// The walk, taking the cost counters out.
    #[allow(clippy::too_many_arguments)]
    fn walk_costed(
        &self,
        input: &Tensor,
        graph: &[Layer],
        convs: &[ConvW],
        net: &NetWeights,
        slopes: &[Slope],
        names: &[String],
        which: &str,
        hook: &mut dyn FnMut(usize, &Tensor),
        mut array: ArrayRef<'_, '_>,
        cost: &mut Cost,
    ) -> Result<Vec<Tensor>, String> {
        let mut vals: Vec<Tensor> = vec![Tensor::default(); graph.len()];
        for (i, layer) in graph.iter().enumerate() {
            let name = names.get(i).map(|s| s.as_str()).unwrap_or("?");
            let nl = format!("{which} node {i} '{name}'");
            let out = self.eval(
                layer,
                &vals,
                input,
                convs,
                net,
                slopes,
                &nl,
                cost,
                array.as_deref_mut(),
            )?;
            vals[i] = out;
            // The hook fires on the node's OUTPUT, so it sees the activation
            // too: an activation fused into the convolution is part of what the
            // node produced, and a hook placed before it would report the
            // pre-activation tensor for every node that has one.
            hook(i, &vals[i]);
        }
        Ok(vals)
    }

    /// One node.
    #[allow(clippy::too_many_arguments)]
    fn eval(
        &self,
        layer: &Layer,
        vals: &[Tensor],
        input: &Tensor,
        convs: &[ConvW],
        net: &NetWeights,
        slopes: &[Slope],
        label: &str,
        cost: &mut Cost,
        array: ArrayRef<'_, '_>,
    ) -> Result<Tensor, String> {
        let operand = |idx: i32| -> Result<&Tensor, String> {
            if idx == -1 {
                Ok(input)
            } else if idx >= 0 && (idx as usize) < vals.len() {
                Ok(&vals[idx as usize])
            } else {
                Err(format!("{label}: operand {idx} is not a node"))
            }
        };
        let slope: Option<&[f32]> = if layer.act == Act::PreLu {
            let idx = layer.prelu;
            if idx < 0 || idx as usize >= slopes.len() {
                return Err(format!(
                    "{label}: prelu {idx} is past the network's {} slope vectors",
                    slopes.len()
                ));
            }
            let v = &slopes[idx as usize].v;
            if v.len() as i64 != layer.out_c {
                return Err(format!(
                    "{label}: prelu {idx} has {} slopes for a {}-channel tensor. \
                     PReLU's slope is PER CHANNEL, so a length mismatch has no \
                     per-pixel meaning.",
                    v.len(),
                    layer.out_c
                ));
            }
            Some(v)
        } else {
            None
        };
        let first = operand(layer.inputs[0])?;
        match layer.op {
            Op::Conv => {
                let conv = convs.get(layer.conv as usize).filter(|c| !c.w.is_empty());
                let Some(conv) = conv else {
                    return Err(format!(
                        "{label}: op 'conv' names conv {}, which the container has no \
                         weights for. Both lists are written by the same packer pass, \
                         so a disagreement here means the container was edited or \
                         half-written -- and a convolution with no weights is a graph \
                         that would run on whatever was in the buffer.",
                        layer.conv
                    ));
                };
                self.conv(first, layer, conv, net, slope, label, cost, array)
            }
            Op::DwConv => {
                let conv = convs.get(layer.conv as usize).filter(|c| !c.w.is_empty());
                let Some(conv) = conv else {
                    return Err(format!(
                        "{label}: op 'dwconv' names conv {}, which the container has no \
                         weights for.",
                        layer.conv
                    ));
                };
                self.dwconv(first, layer, conv, slope, label, cost)
            }
            Op::Add => {
                if layer.inputs.len() != 2 {
                    return Err(format!(
                        "{label}: add has {} operands, not two",
                        layer.inputs.len()
                    ));
                }
                let second = operand(layer.inputs[1])?;
                self.add(first, second, layer, slope, label, cost)
            }
            Op::MaxPool => self.maxpool(first, layer, label, cost),
            Op::PadC => self.pad_c(first, layer, label, cost),
            Op::Resize => self.resize(first, layer, label, cost),
        }
    }

    /// The dense convolution: im2col, the shared GEMM, and the transpose.
    ///
    /// THE ARRAY, OR THE HOST, and it is ONE branch in ONE place.
    ///
    /// The im2col above the branch is the host's either way: it is a gather,
    /// and the array's MMAC multiplies matrices. So the two paths cannot
    /// differ in the part that is easy to get wrong -- the column order --
    /// and they differ only in who multiplies.
    ///
    /// They do NOT produce the same numbers, and that is not a bug: the
    /// array multiplies the weights rounded to bf16 and the host multiplies
    /// them in f32, so a run on the array is the same network with a coarser
    /// weight, and the difference is the width of bf16.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        input: &Tensor,
        layer: &Layer,
        w: &ConvW,
        net: &NetWeights,
        slope: Option<&[f32]>,
        label: &str,
        cost: &mut Cost,
        mut array: ArrayRef<'_, '_>,
    ) -> Result<Tensor, String> {
        let (oh, ow) = check_extent(layer, input.h, input.w, w.kh, w.kw, label)?;
        if w.cin != input.c {
            return Err(format!(
                "{label}: conv {} takes {} channels and the tensor has {}. Reading the \
                 other one would index past the end of the im2col row.",
                w.index, w.cin, input.c
            ));
        }
        if w.cout != layer.out_c {
            return Err(format!(
                "{label}: conv {} has {} outputs and the graph says {}",
                w.index, w.cout, layer.out_c
            ));
        }
        let wmat = net.get(w);
        if wmat.is_empty() {
            return Err(format!(
                "{label}: conv {} has no [K,N] weight table. The transpose is built at \
                 load for every DENSE filter, so a missing one means this convolution \
                 is depthwise and was routed here by name -- which would compute a \
                 different network rather than fail.",
                w.index
            ));
        }
        let t = std::time::Instant::now();
        let a = im2col(
            input,
            w.kh,
            w.kw,
            layer.pad,
            layer.stride_h,
            layer.stride_w,
            oh,
            ow,
        );
        cost.t_im2col += t.elapsed().as_secs_f64();
        let m = (oh * ow) as usize;
        let k = w.kin() as usize;
        let n = w.cout as usize;
        let mut c = vec![0.0f32; m * n];
        // THE ARRAY'S SPAN, AND THE HOST'S.
        //
        // The array's own time is measured around the dispatch, and the
        // A repack inside it is measured by the backend -- the host's own
        // work, and not the device's, so it is reported separately rather
        // than folded into the device's number.
        let mut array_s = 0.0f64;
        let mut repack_s = 0.0f64;
        let mut dispatched = 0i64;
        let mut convs_array = 0i64;
        let gemm_s = match array.as_deref_mut() {
            Some(array) => {
                // The panel's name is the weight's own, and the two graphs'
                // convolutions are namespaced -- "palm.conv.3.w" and
                // "lm.conv.3.w" are two different panels.
                let name = format!("{}.conv.{}.w", layer.net, w.index);
                let before_dispatch = array.dispatches() as i64;
                let before_convert = array.convert_seconds();
                let t = std::time::Instant::now();
                array
                    .conv(&name, &a, m, k, n, &mut c)
                    .map_err(|e| format!("{label}: {e}"))?;
                array_s = t.elapsed().as_secs_f64();
                repack_s = array.convert_seconds() - before_convert;
                // The dispatch is what is left once the repack is taken out:
                // the device's multiply, the DMA and the wait.
                dispatched = array.dispatches() as i64 - before_dispatch;
                convs_array = 1;
                (array_s - repack_s).max(0.0)
            }
            None => {
                let t = std::time::Instant::now();
                gemm_rows(&a, m, wmat, k, n, &w.b, &mut c, self.threads);
                t.elapsed().as_secs_f64()
            }
        };
        cost.t_gemm += gemm_s;
        let mut out = Tensor {
            c: w.cout,
            h: oh,
            w: ow,
            d: vec![0.0f32; m * n],
        };
        let t = std::time::Instant::now();
        let empty: &[f32] = &[];
        transpose_mn_to_nchw(
            &c,
            m as i64,
            n as i64,
            &mut out.d,
            layer.act,
            slope.unwrap_or(empty),
        );
        let transpose_s = t.elapsed().as_secs_f64();
        cost.t_transpose += transpose_s;
        if convs_array > 0 {
            // The array's span INCLUDES the im2col, because the oracle's does
            // -- its timer starts before the im2col -- and the two numbers
            // are only comparable if they span the same work.
            cost.t_array_s += array_s + transpose_s;
            cost.t_array_gemm_s += gemm_s;
            cost.t_array_repack_s += repack_s;
            cost.t_array_transpose_s += transpose_s;
            cost.convs_array += convs_array;
            cost.dispatches += dispatched;
        } else {
            cost.convs_host += 1;
        }
        Ok(out)
    }

    /// The depthwise convolution: one filter per channel, no GEMM.
    ///
    /// That is exactly the shape the array's instruction stream cannot express
    /// -- it wants a K-reduction across N output columns -- and it is why 39 of
    /// this container's 100 convolutions stay on the host by name. Looping per
    /// channel rather than building a block-diagonal GEMM is deliberate: the
    /// block-diagonal form computes K*C times the useful arithmetic.
    fn dwconv(
        &self,
        input: &Tensor,
        layer: &Layer,
        w: &ConvW,
        slope: Option<&[f32]>,
        label: &str,
        cost: &mut Cost,
    ) -> Result<Tensor, String> {
        let (oh, ow) = check_extent(layer, input.h, input.w, w.kh, w.kw, label)?;
        if w.cout != input.c {
            return Err(format!(
                "{label}: dwconv {} has {} filters and the tensor has {} channels. One \
                 filter per channel means they are the same number, and reading it as a \
                 dense convolution would compute a different network rather than fail.",
                w.index, w.cout, input.c
            ));
        }
        if layer.out_c != input.c {
            return Err(format!(
                "{label}: dwconv {} changes the channel count from {} to {}. A depthwise \
                 convolution cannot: it would need a projection, which is what the \
                 `conv` after it is for.",
                w.index, input.c, layer.out_c
            ));
        }
        let t = std::time::Instant::now();
        let h = input.h;
        let wd = input.w;
        let mut out = Tensor {
            c: layer.out_c,
            h: oh,
            w: ow,
            d: vec![0.0f32; (layer.out_c * oh * ow) as usize],
        };
        // Only the TOP and LEFT terms are read here: they are the offsets of
        // the gather. The bottom and right terms are checked by check_extent()
        // against the declared output, which is what makes them matter.
        let (pt, pl) = (layer.pad[0], layer.pad[1]);
        let plane_in = (h * wd) as usize;
        let plane_out = (oh * ow) as usize;
        for c in 0..input.c {
            let src = &input.d[(c as usize) * plane_in..][..plane_in];
            let filt = &w.w[(c * w.kh * w.kw) as usize..][..(w.kh * w.kw) as usize];
            let b = w.b.get(c as usize).copied().unwrap_or(0.0);
            let sl = slope
                .and_then(|s| s.get(c as usize).copied())
                .unwrap_or(0.0);
            let dst = &mut out.d[(c as usize) * plane_out..][..plane_out];
            for y in 0..oh {
                for x in 0..ow {
                    let mut acc = b;
                    for i in 0..w.kh {
                        let sy = y * layer.stride_h + i - pt;
                        if sy < 0 || sy >= h {
                            continue;
                        }
                        for j in 0..w.kw {
                            let sx = x * layer.stride_w + j - pl;
                            if sx < 0 || sx >= wd {
                                continue;
                            }
                            let wt = filt[(i * w.kw + j) as usize];
                            let v = src[(sy * wd + sx) as usize];
                            // The oracle contracts this tap into one rounding
                            // (`vfmadd231ss` in the disassembly).
                            acc = wt.mul_add(v, acc);
                        }
                    }
                    dst[(y * ow + x) as usize] = act1(layer.act, acc, sl);
                }
            }
        }
        cost.t_dwconv += t.elapsed().as_secs_f64();
        cost.dwconvs_host += 1;
        Ok(out)
    }

    /// The mobilenet residual, with the activation possibly fused onto the sum.
    fn add(
        &self,
        a: &Tensor,
        b: &Tensor,
        layer: &Layer,
        slope: Option<&[f32]>,
        label: &str,
        cost: &mut Cost,
    ) -> Result<Tensor, String> {
        if a.c != b.c || a.h != b.h || a.w != b.w {
            return Err(format!(
                "{label}: add joins a {}x{}x{} and a {}x{}x{}. Neither network has a \
                 broadcast add in its graph, so the two shapes are a transcription \
                 error rather than something to broadcast: broadcasting here would \
                 silently accept a block that resized instead of adding.",
                a.c, a.h, a.w, b.c, b.h, b.w
            ));
        }
        if layer.out_c != a.c || layer.out_h != a.h || layer.out_w != a.w {
            return Err(format!(
                "{label}: add's declared output is {}x{}x{} and its operands are \
                 {}x{}x{}",
                layer.out_c, layer.out_h, layer.out_w, a.c, a.h, a.w
            ));
        }
        let t = std::time::Instant::now();
        let plane = (a.h * a.w) as usize;
        let mut out = Tensor {
            c: a.c,
            h: a.h,
            w: a.w,
            d: vec![0.0f32; a.d.len()],
        };
        for i in 0..a.d.len() {
            let c = i / plane;
            let sl = slope.and_then(|s| s.get(c).copied()).unwrap_or(0.0);
            // A plain add, NOT a fused multiply-add: the oracle's sum is a
            // `vaddss` and there is no product to fuse.
            out.d[i] = act1(layer.act, a.d[i] + b.d[i], sl);
        }
        cost.t_elementwise += t.elapsed().as_secs_f64();
        cost.elementwise_ops += 1;
        Ok(out)
    }

    /// The pooling, with four padding terms for the same reason.
    fn maxpool(
        &self,
        input: &Tensor,
        layer: &Layer,
        label: &str,
        cost: &mut Cost,
    ) -> Result<Tensor, String> {
        let (oh, ow) = check_extent(
            layer,
            input.h,
            input.w,
            layer.kernel_h,
            layer.kernel_w,
            label,
        )?;
        if layer.out_c != input.c {
            return Err(format!("{label}: maxpool changes the channel count"));
        }
        let t = std::time::Instant::now();
        let (h, wd) = (input.h, input.w);
        let plane = (h * wd) as usize;
        let (pt, pl) = (layer.pad[0], layer.pad[1]);
        let mut out = Tensor {
            c: layer.out_c,
            h: oh,
            w: ow,
            d: vec![0.0f32; (layer.out_c * oh * ow) as usize],
        };
        for c in 0..input.c {
            let src = &input.d[(c as usize) * plane..][..plane];
            let dst = &mut out.d[(c as usize) * (oh * ow) as usize..][..(oh * ow) as usize];
            for y in 0..oh {
                for x in 0..ow {
                    let mut best = f32::NEG_INFINITY;
                    for i in 0..layer.kernel_h {
                        let sy = y * layer.stride_h + i - pt;
                        if sy < 0 || sy >= h {
                            continue;
                        }
                        for j in 0..layer.kernel_w {
                            let sx = x * layer.stride_w + j - pl;
                            if sx < 0 || sx >= wd {
                                continue;
                            }
                            best = best.max(src[(sy * wd + sx) as usize]);
                        }
                    }
                    // A window entirely in the padding has no samples. -inf is
                    // the honest value for a max over nothing, and a 0 default
                    // would be a different tensor that only ever shows up as a
                    // slightly wrong pixel.
                    dst[(y * ow + x) as usize] = best;
                }
            }
        }
        cost.t_elementwise += t.elapsed().as_secs_f64();
        cost.elementwise_ops += 1;
        Ok(out)
    }

    /// pad_c: append `count` ZERO CHANNELS.
    ///
    /// The mobilenet "pad to 2x" step: a channel-wise concat with a constant,
    /// which is what makes the following 1x1 convolution a widening projection.
    /// Reading it as a pixel pad gives the right total channel count and a
    /// wrong tensor.
    fn pad_c(
        &self,
        input: &Tensor,
        layer: &Layer,
        label: &str,
        cost: &mut Cost,
    ) -> Result<Tensor, String> {
        if layer.out_c != input.c + layer.count {
            return Err(format!(
                "{label}: pad_c appends {} channels to {} and the graph says the output \
                 is {} wide. `count` is the number of ZERO CHANNELS, appended at the \
                 channel axis.",
                layer.count, input.c, layer.out_c
            ));
        }
        if layer.out_h != input.h || layer.out_w != input.w {
            return Err(format!("{label}: pad_c changes the spatial extent"));
        }
        let t = std::time::Instant::now();
        let plane = (input.h * input.w) as usize;
        let mut out = Tensor {
            c: layer.out_c,
            h: layer.out_h,
            w: layer.out_w,
            d: vec![0.0f32; (layer.out_c as usize) * plane],
        };
        for c in 0..layer.out_c {
            let dst = &mut out.d[(c as usize) * plane..][..plane];
            if c < input.c {
                dst.copy_from_slice(&input.d[(c as usize) * plane..][..plane]);
            }
        }
        cost.t_elementwise += t.elapsed().as_secs_f64();
        cost.elementwise_ops += 1;
        Ok(out)
    }

    /// ONNX Resize, linear, half_pixel, exclude_outside=0.
    ///
    /// Two passes, and each weight belongs to the axis its pass moves along:
    /// `bot` differs from `top` in the ROW index so it is blended by wy, and
    /// `right` differs from `left` in the COLUMN index so it is blended by wx.
    /// Putting them the other way round is a TRANSPOSITION -- still bilinear,
    /// still the right shape, and still a detector that finds a hand.
    ///
    /// half_pixel: `src = (i + 0.5) / scale - 0.5`. The alternatives
    /// (align_corners, asymmetric, pytorch_half) are each off by half a source
    /// pixel.
    ///
    /// The rounding is the oracle's: its compiler turned each blend into one
    /// rounded multiply and one fused multiply-add (`vmulss` then
    /// `vfmadd231ss`), so this is `a.mul_add(1 - w, b * w)` and not
    /// `a * (1 - w) + b * w`.
    fn resize(
        &self,
        input: &Tensor,
        layer: &Layer,
        label: &str,
        cost: &mut Cost,
    ) -> Result<Tensor, String> {
        if layer.out_c != input.c {
            return Err(format!("{label}: resize changes the channel count"));
        }
        if layer.scale_h <= 0.0 || layer.scale_w <= 0.0 {
            return Err(format!("{label}: resize has a non-positive scale"));
        }
        let t = std::time::Instant::now();
        let (sh, sw) = (input.h, input.w);
        let (dh, dw) = (layer.out_h, layer.out_w);
        let mut y0 = vec![0i64; dh as usize];
        let mut y1 = vec![0i64; dh as usize];
        let mut wy = vec![0.0f32; dh as usize];
        for i in 0..dh {
            let s = (i as f64 + 0.5) / layer.scale_h - 0.5;
            let f = s.floor();
            y0[i as usize] = (f as i64).clamp(0, sh - 1);
            y1[i as usize] = ((f as i64) + 1).clamp(0, sh - 1);
            wy[i as usize] = (s - f) as f32;
        }
        let mut x0 = vec![0i64; dw as usize];
        let mut x1 = vec![0i64; dw as usize];
        let mut wx = vec![0.0f32; dw as usize];
        for i in 0..dw {
            let s = (i as f64 + 0.5) / layer.scale_w - 0.5;
            let f = s.floor();
            x0[i as usize] = (f as i64).clamp(0, sw - 1);
            x1[i as usize] = ((f as i64) + 1).clamp(0, sw - 1);
            wx[i as usize] = (s - f) as f32;
        }
        let mut out = Tensor {
            c: layer.out_c,
            h: dh,
            w: dw,
            d: vec![0.0f32; (layer.out_c * dh * dw) as usize],
        };
        let plane_in = (sh * sw) as usize;
        let plane_out = (dh * dw) as usize;
        let mut row = vec![0.0f32; dw as usize];
        for c in 0..input.c {
            let src = &input.d[(c as usize) * plane_in..][..plane_in];
            let dst = &mut out.d[(c as usize) * plane_out..][..plane_out];
            for i in 0..dh {
                let a = &src[(y0[i as usize] * sw) as usize..];
                let b = &src[(y1[i as usize] * sw) as usize..];
                let w = wy[i as usize];
                let ow = 1.0f32 - w;
                for j in 0..dw as usize {
                    row[j] = a[x0[j] as usize].mul_add(ow, b[x0[j] as usize] * w);
                }
                for j in 0..dw as usize {
                    let l0 = row[j];
                    let l1 = a[x1[j] as usize].mul_add(ow, b[x1[j] as usize] * w);
                    dst[(i * dw) as usize + j] = l0.mul_add(1.0 - wx[j], l1 * wx[j]);
                }
            }
        }
        cost.t_elementwise += t.elapsed().as_secs_f64();
        cost.elementwise_ops += 1;
        Ok(out)
    }

    /// The node a head names, by GRAPH index.
    fn node<'a>(&self, nodes: &'a [Tensor], i: i32) -> Result<&'a Tensor, String> {
        if i < 0 || i as usize >= nodes.len() {
            return Err(format!(
                "graph node {} is not one of the network's {} nodes",
                i,
                nodes.len()
            ));
        }
        Ok(&nodes[i as usize])
    }

    /// The palm detector's head: four NCHW maps into `[n, 18]` and `[n]`.
    ///
    /// The reshape is the delicate part and it is done in the graph's own axis
    /// order: the checkpoint transposes NCHW to NHWC and reshapes [H,W,C] to
    /// [H*W*A, terms], so row (h*W + w)*A + a, term t, is channel
    /// a*terms + t of cell (h, w). The flat row index therefore advances w --
    /// the X coordinate -- fastest, which is the order the anchor table is
    /// generated in, and a mismatch between the two is a detector whose best
    /// cell is a mirror image away.
    pub fn palm_head(&self, nodes: &[Tensor]) -> Result<PalmOut, String> {
        let g = &self.geom;
        let mut boxes: Vec<f64> = Vec::new();
        let mut scores: Vec<f64> = Vec::new();
        for (li, level) in g.palm_levels.iter().enumerate() {
            let el = format!("palm level {li}");
            let b = self.node(nodes, level.box_)?;
            let s = self.node(nodes, level.score_)?;
            if b.c != level.anchors_per_cell * g.palm_row_terms || b.h != level.h || b.w != level.w
            {
                return Err(format!(
                    "{el}: the box feature is {}x{}x{} and the head says {}x{}x{}. Both \
                     are written by the packer from the ONNX's own output names, so a \
                     disagreement is one of them a transcription.",
                    b.c,
                    b.h,
                    b.w,
                    level.anchors_per_cell * g.palm_row_terms,
                    level.h,
                    level.w
                ));
            }
            if s.c != level.anchors_per_cell || s.h != level.h || s.w != level.w {
                return Err(format!(
                    "{el}: the score feature is {}x{}x{} against {}x{}x{}",
                    s.c, s.h, s.w, level.anchors_per_cell, level.h, level.w
                ));
            }
            boxes.reserve((level.rows * g.palm_row_terms) as usize);
            scores.reserve(level.rows as usize);
            let cells = (level.h * level.w) as usize;
            let cell = |y: i64, x: i64| (y * level.w + x) as usize;
            for y in 0..level.h {
                for x in 0..level.w {
                    let c = cell(y, x);
                    for a in 0..level.anchors_per_cell {
                        for t in 0..g.palm_row_terms {
                            let ch = (a * g.palm_row_terms + t) as usize;
                            boxes.push(b.d[ch * cells + c] as f64);
                        }
                        scores.push(s.d[(a as usize) * cells + c] as f64);
                    }
                }
            }
        }
        let n = scores.len() as i64;
        if n != g.palm_num_anchors {
            return Err(format!(
                "the levels produced {n} anchors and the container declares {}. The \
                 decode reads a table of the declared width, so this is a refusal \
                 rather than a read past the end.",
                g.palm_num_anchors
            ));
        }
        Ok(PalmOut { boxes, scores, n })
    }

    /// The landmark head: one global average pool, then four Gemms.
    ///
    /// The pool is a MEAN over the 7x7 map and not a max and not a stride-7
    /// conv: MediaPipe's hand-landmark head is a squeeze-then-Gemm. The
    /// projections are stored [in, out] and ONNX's Gemm with transB=0 is A @ W,
    /// so this is W.T @ pooled; the other way round is a shape error in the
    /// config and a silent TRANSPOSE in a runtime.
    pub fn lm_head(&self, nodes: &[Tensor], cost: &mut Cost) -> Result<LmOut, String> {
        let g = &self.geom;
        let head = &g.lm_head;
        let x = self.node(nodes, head.pool)?;
        if x.c != head.features || x.h != head.height || x.w != head.width {
            return Err(format!(
                "lm landmark head: the pooled node is {}x{}x{} and the head says \
                 {}x{}x{}. The width is the projection's input size, so a \
                 disagreement here is a projection that cannot be applied at all.",
                x.c, x.h, x.w, head.features, head.height, head.width
            ));
        }
        let t = std::time::Instant::now();
        let plane = (x.h * x.w) as usize;
        let mut pooled = vec![0.0f32; x.c as usize];
        for c in 0..x.c {
            let p = &x.d[(c as usize) * plane..][..plane];
            let mut acc = 0.0f32;
            for v in p {
                acc += *v;
            }
            pooled[c as usize] = acc / plane as f32;
        }
        cost.t_pool += t.elapsed().as_secs_f64();

        let mut proj: Vec<Vec<f64>> = Vec::with_capacity(head.projs.len());
        for p in &head.projs {
            let mut out = vec![0.0f64; p.n as usize];
            for j in 0..p.n as usize {
                let mut acc = p.b.get(j).copied().unwrap_or(0.0) as f64;
                for c in 0..x.c as usize {
                    // The oracle's projection accumulates with a fused
                    // multiply-add, so it rounds once per term.
                    acc = (pooled[c] as f64).mul_add(p.w[c * p.n as usize + j] as f64, acc);
                }
                out[j] = acc;
            }
            proj.push(out);
        }
        let mut outs = Vec::with_capacity(head.outputs.len());
        for spec in &head.outputs {
            let mut v = proj[spec.proj].clone();
            if spec.sigmoid {
                for e in v.iter_mut() {
                    *e = 1.0 / (1.0 + (-*e).exp());
                }
            }
            outs.push(v);
        }
        Ok(LmOut { outs })
    }

    /// The walk with its cost counters handed back, for the CLI's diagnostics.
    #[allow(clippy::too_many_arguments)]
    pub fn run_palm_costed(
        &self,
        input: &Tensor,
        hook: &mut dyn FnMut(usize, &Tensor),
        cost: &mut Cost,
    ) -> Result<Vec<Tensor>, String> {
        self.walk_costed(
            input,
            &self.geom.palm_graph,
            &self.geom.palm_convs,
            &self.palm,
            &self.geom.palm_slopes,
            &self.geom.palm_nodes,
            "palm",
            hook,
            None,
            cost,
        )
    }

    /// The same for the landmark network.
    pub fn run_lm_costed(
        &self,
        input: &Tensor,
        hook: &mut dyn FnMut(usize, &Tensor),
        cost: &mut Cost,
    ) -> Result<Vec<Tensor>, String> {
        self.walk_costed(
            input,
            &self.geom.lm_graph,
            &self.geom.lm_convs,
            &self.lm,
            &self.geom.lm_slopes,
            &self.geom.lm_nodes,
            "lm",
            hook,
            None,
            cost,
        )
    }

    /// The palm detector's walk, on the array's convolutions.
    ///
    /// THE ARRAY, OR THE HOST -- and the flag that chose is not read here:
    /// the caller passes the array's backend when it wants the convolutions
    /// on the board, and passes nothing when it does not. A run that asked
    /// for the array and dispatched nothing is a run that lied, and the way
    /// to keep it from lying is that the array's backend is the only thing
    /// that can dispatch at all.
    #[allow(clippy::too_many_arguments)]
    pub fn run_palm_costed_on_array(
        &self,
        input: &Tensor,
        hook: &mut dyn FnMut(usize, &Tensor),
        array: ArrayRef<'_, '_>,
        cost: &mut Cost,
    ) -> Result<Vec<Tensor>, String> {
        self.walk_costed(
            input,
            &self.geom.palm_graph,
            &self.geom.palm_convs,
            &self.palm,
            &self.geom.palm_slopes,
            &self.geom.palm_nodes,
            "palm",
            hook,
            array,
            cost,
        )
    }

    /// The landmark network's walk, on the array's convolutions.
    pub fn run_lm_costed_on_array(
        &self,
        input: &Tensor,
        hook: &mut dyn FnMut(usize, &Tensor),
        array: ArrayRef<'_, '_>,
        cost: &mut Cost,
    ) -> Result<Vec<Tensor>, String> {
        self.walk_costed(
            input,
            &self.geom.lm_graph,
            &self.geom.lm_convs,
            &self.lm,
            &self.geom.lm_slopes,
            &self.geom.lm_nodes,
            "lm",
            hook,
            array,
            cost,
        )
    }
}
