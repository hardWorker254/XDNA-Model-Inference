//! The MediaPipe body-pose networks: two packed graphs, one container, two heads.
//!
//! # WHAT IS IN THE CONTAINER
//!
//! One container holds BOTH networks and the front end's constants:
//!
//! * **det** -- MediaPipe's person detector, 224x224, 93 graph nodes, 45 dense
//!   convolutions, 28 depthwise ones, three `d2s`, and a head that emits 2254
//!   rows of 12 numbers plus 2254 scores;
//! * **pose** -- MediaPipe's pose-landmark network, 256x256, 120 nodes, 54 dense
//!   and 34 depthwise convolutions, and a head of five single convolutions.
//!
//! They are one file because a person's landmarks are computed on a crop the
//! FIRST network found and rotated by an angle it predicted. The dependency
//! cannot live between two containers.
//!
//! # THE VOCABULARY IS SIX OPS, AND IT IS NOT HANDS' SIX
//!
//! `conv, dwconv, add, maxpool, resize, d2s`. Against arch=7 exactly one op
//! arrives and one leaves:
//!
//! * **d2s** -- DepthToSpace. It is how the detector builds its three pyramid
//!   levels out of a single 7x7x1152 map, so there is no rearrangement of conv,
//!   dwconv and add that expresses it. Three of them, one per level.
//! * **pad_c** -- gone. arch=7's channel-widening step, which neither of these
//!   graphs contains.
//!
//! The op list is PARSED rather than assumed, and an unknown one is refused by
//! name: a build that guessed would be running a different network.
//!
//! # THE D2S ORDER IS CARRIED IN THE OP AND NOT DECIDED HERE
//!
//! ONNX names the two orders DCR and CRD and they differ only in how the input
//! CHANNEL axis is split across the block. The default when the attribute is
//! absent is DCR, and this model's nodes carry `blocksize` and nothing else --
//! so the graph states DCR by saying nothing. Taking the other order is not a
//! shape error and not a range error: every channel lands on a different pixel
//! of a map with the right shape and a plausible distribution, and the next
//! convolution consumes it happily. It shows up as a detector that finds people
//! in the wrong places, which is why both orders are implemented and the order
//! is read.
//!
//! # THE ARITHMETIC IS TRANSCRIBED FROM THE ORACLE'S OBJECT FILE
//!
//! Where the reference source leaves a multiply and an add adjacent, the compiler
//! fused some of them and not others, and the difference is one rounding. Read
//! off the oracle's disassembly (`objdump -d runtime/build/libnpue_core.a`,
//! `vfmadd`/`vmulss`):
//!
//! | site | the oracle's instruction | this port |
//! |---|---|---|
//! | dense conv | `gemm_nt`, bias added after the K reduction | `kernels::gemm_nt`, shared |
//! | dwconv | `vfmadd231ss` in the tap loop | `w.mul_add(x, acc)` |
//! | resize | `vmulss` then `vfmadd231ss`, both passes | the same split, both passes |
//! | add | `vaddss` only | a plain `+` |
//! | d2s | no arithmetic at all, plane copies | the same |
//!
//! # THE EXTENT CHECK IS A FLOOR, AND THAT IS THE PADDING HERE
//!
//! arch=6's and arch=7's check requires the padded extent to divide by the
//! stride, which their "SAME padding evaluated at a stride" satisfies. This
//! checkpoint's does NOT: the detector's stem is a 3x3 at stride 2 with a
//! symmetric pad of [1,1,1,1], so 224 + 1 + 1 - 3 = 223 and 223 / 2 is 111.5.
//! ONNX takes floor, the declared output is 112, and a check demanding an exact
//! division refuses the model's own first convolution.

use crate::kernels;
use crate::pose::Tensor;
use npum_format::Container;
use npum_model::array::{ArrayRef, ConvSpec};
use serde_json::Value;
use std::path::Path;

/// The arch string this reader owns.
pub const ARCH: &str = "mediapipe_pose_det_ssd_lm_regress";

/// MediaPipe's pose network emits THIRTY-NINE rows: the thirty-three keypoints a
/// skeleton is drawn from, plus six auxiliary ones.
pub const NUM_LANDMARKS: i64 = 39;

/// The thirty-three a skeleton is drawn from. Recorded so the block can say so.
pub const NUM_KEYPOINTS: i64 = 33;

/// Each landmark is `[x, y, z, visibility, presence]`.
pub const LM_COLS: i64 = 5;

/// Each world point is `[x, y, z]`, in metres.
pub const WORLD_COLS: i64 = 3;

/// The detector's box row: xy1x2 then 4 x 2 keypoint offsets.
pub const DET_ROW_TERMS: i64 = 12;

/// The keypoints the detector's row carries, and the two the rotation uses.
pub const DET_LANDMARKS: i64 = 4;

/// The activation, as the graph records it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    /// The identity.
    None,
    /// `max(0, x)`.
    Relu,
    /// `min(max(0, x), 6)`.
    Relu6,
    /// Per-channel slope.
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
    /// ONNX Resize, linear, half_pixel.
    Resize,
    /// DepthToSpace.
    D2S,
}

impl Op {
    /// The name the container spells.
    pub fn as_str(self) -> &'static str {
        match self {
            Op::Conv => "conv",
            Op::DwConv => "dwconv",
            Op::Add => "add",
            Op::MaxPool => "maxpool",
            Op::Resize => "resize",
            Op::D2S => "d2s",
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
    /// The padding: top, left, bottom, right.
    pub pad: [i64; 4],
    /// The pooling kernel's rows.
    pub kernel_h: i64,
    /// The pooling kernel's columns.
    pub kernel_w: i64,
    /// The Resize's vertical scale.
    pub scale_h: f64,
    /// The Resize's horizontal scale.
    pub scale_w: f64,
    /// D2S: the block side.
    pub block: i64,
    /// D2S: true for CRD, false for DCR (ONNX's default).
    pub crd: bool,
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
    /// The network this node belongs to -- "det" or "pose".
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

/// The mppose family's dense convolutions, as the array's list.
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
///
/// THE TWO GRAPHS ARE NAMESPACED, because they number their convolutions
/// from zero: "det.conv.3.w" and "pose.conv.3.w" are two different weights
/// in one container, and so are two different panels.
pub fn array_specs(geometry: &Geometry) -> Vec<ConvSpec> {
    let mut specs = Vec::new();
    for (prefix, graph, convs) in [
        ("det", &geometry.det_graph, &geometry.det_convs),
        ("pose", &geometry.pose_graph, &geometry.pose_convs),
    ] {
        for l in graph {
            if l.op != Op::Conv || l.stream.is_empty() {
                continue;
            }
            // The convolution is found by its OWN index and not by its
            // position in the table: the packer numbered the convolutions
            // from the ONNX's own order, and a sparse index leaves a hole
            // that a position would slide past.
            let Some(cw) = convs.iter().find(|c| c.index == l.conv) else {
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

/// The detector's head, per pyramid level.
///
/// `box_op` and `score_op` are GRAPH NODE indices, not tensor indices: the head
/// consumes two 1x1 convolution outputs that live in the MIDDLE of the graph.
#[derive(Debug, Clone)]
pub struct DetLevel {
    /// The node holding the box regressions.
    pub box_op: i32,
    /// The node holding the scores.
    pub score_op: i32,
    /// The level's cells per side.
    pub h: i64,
    /// The level's cells per side (columns).
    pub w: i64,
    /// Anchors per cell.
    pub per: i64,
    /// `h * w * per`, recorded and checked.
    pub rows: i64,
    /// Terms per row -- 12 for every level here.
    pub terms: i64,
}

/// One of the landmark head's five declared outputs.
#[derive(Debug, Clone)]
pub struct PoseOut {
    /// The name the container spells.
    pub name: String,
    /// The GRAPH NODE whose output this is.
    pub op: i32,
    /// The graph's declared shape, any rank.
    pub decl: Vec<i64>,
    /// The convolution's `[1, C, H, W]`.
    pub conv_out: [i64; 4],
    /// Whether the output is transposed on the way out.
    pub transposed: bool,
    /// Whether the head applies the sigmoid.
    pub sigmoid: bool,
}

/// The five names, in the order the zoo returns them.
pub const POSE_OUT_NAMES: [&str; 5] = ["landmarks", "conf", "mask", "heatmap", "world"];

/// The model's geometry, as the container states it.
#[derive(Debug, Clone)]
pub struct Geometry {
    /// The detector's input side.
    pub det_input_size: i64,
    /// The landmark network's input side.
    pub pose_input_size: i64,
    /// How many landmarks -- 39.
    pub num_landmarks: i64,
    /// How many of them are keypoints -- 33.
    pub num_keypoints: i64,
    /// Columns per landmark -- 5.
    pub lm_cols: i64,
    /// Columns per world point -- 3.
    pub world_cols: i64,
    /// The detector's per-channel means. Its own, not the landmark net's.
    pub det_mean: Vec<f32>,
    /// The detector's per-channel standard deviations.
    pub det_std: Vec<f32>,
    /// The landmark network's per-channel means.
    pub pose_mean: Vec<f32>,
    /// The landmark network's per-channel standard deviations.
    pub pose_std: Vec<f32>,
    /// How many anchor rows the detector emits.
    pub det_num_anchors: i64,
    /// How many terms per row.
    pub det_row_terms: i64,
    /// How many keypoints its row carries.
    pub det_landmarks: i64,
    /// The detector's score threshold, from the container.
    pub score_threshold: f64,
    /// The NMS IoU threshold, from the container.
    pub nms_threshold: f64,
    /// The cap on survivors, from the container.
    pub top_k: i64,
    /// The pyramid, per level.
    pub det_levels: Vec<DetLevel>,
    /// Which detector keypoint is the mid-hip.
    pub person_lm_mid_hip: i64,
    /// Which one is the full-body point.
    pub person_lm_full_body: i64,
    /// Applied to the square BEFORE the rotation.
    pub person_box_pre_enlarge: f64,
    /// Applied to the REPORTED box.
    pub person_box_enlarge: f64,
    /// The landmark network's confidence threshold.
    pub pose_conf_threshold: f64,
    /// The landmark head's five outputs.
    pub pose_head: Vec<PoseOut>,
    /// The outputs this front end computes and does not read.
    pub pose_unused: Vec<String>,
    /// The detector graph.
    pub det_graph: Vec<Layer>,
    /// The landmark graph.
    pub pose_graph: Vec<Layer>,
    /// The detector convolutions.
    pub det_convs: Vec<ConvW>,
    /// The landmark convolutions.
    pub pose_convs: Vec<ConvW>,
    /// Detector node names, for refusals that quote one.
    pub det_nodes: Vec<String>,
    /// Landmark node names.
    pub pose_nodes: Vec<String>,
    /// The detector PReLU slopes.
    pub det_slopes: Vec<Slope>,
    /// The landmark PReLU slopes.
    pub pose_slopes: Vec<Slope>,
    /// The 2254 anchor centres, STORED, in the graph's own row order.
    pub anchors: Vec<f64>,
}

impl Geometry {
    /// How many dense convolutions the detector has.
    pub fn det_dense(&self) -> i64 {
        self.det_convs.iter().filter(|c| !c.depthwise()).count() as i64
    }

    /// How many depthwise convolutions the detector has.
    pub fn det_depthwise(&self) -> i64 {
        self.det_convs.iter().filter(|c| c.depthwise()).count() as i64
    }

    /// How many dense convolutions the landmark network has.
    pub fn pose_dense(&self) -> i64 {
        self.pose_convs.iter().filter(|c| !c.depthwise()).count() as i64
    }

    /// How many depthwise convolutions the landmark network has.
    pub fn pose_depthwise(&self) -> i64 {
        self.pose_convs.iter().filter(|c| c.depthwise()).count() as i64
    }

    /// One named output of the landmark head, or a refusal that lists them.
    pub fn pose_out(&self, name: &str, label: &str) -> Result<&PoseOut, String> {
        if let Some(p) = self.pose_head.iter().find(|p| p.name == name) {
            return Ok(p);
        }
        let have: Vec<&str> = self.pose_head.iter().map(|p| p.name.as_str()).collect();
        Err(format!(
            "{label}: the landmark head has no '{name}'; it has {{{}}}. MediaPipe Pose \
             declares five outputs and this front end reads all five but the heatmap, so \
             one of them being absent is a different network rather than a head with \
             fewer parts.",
            have.join(", ")
        ))
    }
}

/// What the detector's head produces, in its own shape.
#[derive(Debug, Clone, Default)]
pub struct DetOut {
    /// `n * row_terms`, row-major, row `i` being anchor `i`.
    pub boxes: Vec<f64>,
    /// `n` raw scores -- LOGITS, so the decode applies the sigmoid.
    pub scores: Vec<f64>,
    /// How many anchors.
    pub n: i64,
}

/// What the landmark network produces, in the five shapes the container declares.
#[derive(Debug, Clone, Default)]
pub struct PoseOuts {
    /// 39 x 5 screen landmarks, in the crop's own coordinates.
    pub landmarks: Vec<f64>,
    /// Already sigmoid'd by the head.
    pub conf: f64,
    /// 256 x 256, row-major `[y, x]`.
    pub mask: Vec<f32>,
    /// 39 x 3, metres.
    pub world: Vec<f64>,
    /// 64 x 64 x 39, NHWC. Computed, carried for a dump, and read by nothing.
    pub heatmap: Vec<f32>,
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
            "{label}: no '{key}'. The packed graph and this runtime have to agree about \
             the model's own fields, and a key that is not there is a packer and a runtime \
             from different days (arch {}).",
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

/// A required integer inside a JSON object.
fn jint(node: &Value, key: &str, label: &str) -> Result<i64, String> {
    node.get(key).and_then(|v| v.as_i64()).ok_or_else(|| {
        format!(
            "{label}: no '{key}'. The packed graph and this runtime have to agree about the \
             model's own fields, and a key that is not there is a packer and a runtime from \
             different days."
        )
    })
}

/// A pair, as two integers.
fn pair(node: &Value, key: &str, label: &str) -> Result<(i64, i64), String> {
    let array = node
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{label}: '{key}' is not a pair"))?;
    if array.len() != 2 {
        return Err(format!(
            "{label}: '{key}' is not a pair. Every stride, kernel and scale in these graphs \
             is per axis and both axes have to be stated: these two networks are square so \
             far, but 'square so far' is not something to assume about a packed graph."
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
            "{label}: 'pad' is not four terms [top, left, bottom, right]. It is four and not \
             two because MediaPipe evaluates SAME padding at a stride, so a 3x3 stride-2 stem \
             is [1,1,1,1] here and not [1,1] -- and the two stems in this container happen to \
             be symmetric, which means a transcription error here would not show up in a shape \
             and would move every detection instead."
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
        "resize" => Op::Resize,
        "d2s" => Op::D2S,
        other => {
            return Err(format!(
                "{label}: unknown op '{other}' in the packed graph. This build walks conv, \
                 dwconv, add, maxpool, resize and d2s. d2s is arch=8's own and is not a \
                 spelling of anything arch=6 or arch=7 has -- it is the detector's \
                 DepthToSpace, which builds the three pyramid levels out of one map and \
                 cannot be folded away -- while this build has no pad_c, which is arch=7's \
                 channel-widening step and which neither of these two graphs contains."
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
                "{label}: activation '{other}' is not one of none/relu/relu6/prelu. Both of \
                 these networks fuse their activation into the convolution's epilogue or into \
                 the residual add, and an unknown one is not a pass this runtime can invent -- \
                 it would be a different network."
            ));
        }
    })
}

/// The slope vector's length: the product of the shape, not `shape[0]`.
fn slope_len(shape: &[i64]) -> i64 {
    shape.iter().copied().product::<i64>().max(1)
}

/// One convolution, widened or dequantised to fp32.
///
/// The BYTE COUNT is checked against the declared dtype before anything is
/// read: a payload that contradicts its own precision is refused here rather
/// than read as a plausible fp32 array of the wrong length.
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
            "{label}: {base} is a dwconv with {cin} input channels. A depthwise filter is ONE \
             filter per channel; a dense GEMM over this one would compute a different network, \
             and the difference would be this architecture's accuracy rather than a crash."
        ));
    }
    let want = cout * cin * kh * kw;
    let n = cout;
    let k = cin * kh * kw;
    let w: Vec<f32> = match dtype {
        "f32" => {
            if nbytes as i64 != want * 4 {
                return Err(format!(
                    "{label}: {base} is {nbytes} bytes, which is not {} fp32 values for a \
                     [{cout}, {cin}, {kh}, {kw}] weight. A weight whose byte count disagrees \
                     with its shape is not a weight at a different precision; it is a number \
                     of the wrong size.",
                    want * 4
                ));
            }
            f32s(container, &base, label)?
        }
        "bf16" => {
            if nbytes as i64 != want * 2 {
                return Err(format!(
                    "{label}: {base} is {nbytes} bytes, which is not {} for {want} bf16 values.",
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
                    "{label}: {base} is {nbytes} bytes, which is not {} for a [{n}, {k}] \
                     {dtype} weight.",
                    n * packed
                ));
            }
            let scale_name = format!("{prefix}.conv.{index}.wscale");
            let scale = f32s(container, &scale_name, label).map_err(|_| {
                format!(
                    "{label}: {scale_name} is not in the container, and {base} says its weights \
                     are {dtype}. An integer payload without its scale is bytes whose meaning \
                     is one multiply away and not present in the file."
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
                        "{label}: {gname} is not in the container, and {base} says its weights \
                         are i4 with a group of {i4_group}."
                    )
                })?;
                let g = if i4_group > 0 {
                    (k + i4_group - 1) / i4_group
                } else {
                    1
                };
                if gscale.len() as i64 != g * n {
                    return Err(format!(
                        "{label}: {gname} is {} entries, not {} -- [{g}, {n}] for a group of \
                         {i4_group} over K={k}.",
                        gscale.len(),
                        g * n
                    ));
                }
                for nn in 0..n {
                    for kk in 0..k {
                        let byte = bytes[(nn * ((k + 1) / 2) + kk / 2) as usize];
                        // Four bits, low nibble first, SIGN-EXTENDED. A nibble of 0xF is -1
                        // and not 15: reading them unsigned maps the most negative weight
                        // onto the most positive one, in every group.
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
                "{label}: conv_weight_dtype is '{other}', and this runtime reads f32, bf16, i8 \
                 and i4. There is no fp16 or fp8 convolution weight here, and reading either as \
                 one of the four produces a network with the right shapes and no relationship \
                 to the checkpoint."
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
            "{label}: {bname} holds {} values and the convolution has {cout} output channels.",
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
///
/// It is not decoration: both networks number their convolutions from zero, so
/// an unprefixed lookup reads the OTHER network's weights and returns a tensor
/// of the wrong width.
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
            "{label}: {graph_key} has {} nodes and {nodes_key} has {}. The second list is what \
             a refusal quotes by name, and a refusal that quotes node 7 when node 7 does not \
             exist is worse than none.",
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
        // REQUIRED there. maxpool, resize and d2s have no epilogue -- they are
        // not convolutions and there is nothing to fuse into -- so the key is
        // absent on them and its absence is read as `none`.
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
        // activation is not a PReLU writes null, and reading that as 0 would make
        // all of them into the first one's slopes.
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
                "{nl}: carries prelu slope {prelu} and the activation none. Both are written by \
                 the same packer pass over the same fused-activation node, so this is a \
                 half-transcribed op rather than a network whose activation happens to be \
                 unused."
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
            block: 0,
            crd: false,
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
                        "{nl}: conv {} is past the container's {num_convs} convolutions for \
                         this network",
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
                if layer.scale_h <= 0.0 || layer.scale_w <= 0.0 {
                    return Err(format!("{nl}: resize has a non-positive scale"));
                }
            }
            Op::D2S => {
                layer.block = jint(node, "block", &nl)?;
                if layer.block < 1 {
                    return Err(format!(
                        "{nl}: d2s block is {}, and a blocksize below one is not a blocksize",
                        layer.block
                    ));
                }
                // ONNX NAMES THE TWO ORDERS DCR AND CRD, and DCR is the default when the
                // attribute is absent -- which is this model's case: its DepthToSpace
                // nodes carry `blocksize` and nothing else. The default has to be the
                // schema's own, because taking the other one moves every channel to a
                // different pixel while still producing a tensor of the right shape.
                let mode = node
                    .get("mode")
                    .and_then(|v| v.as_str())
                    .filter(|v| !v.is_empty())
                    .unwrap_or("dcr");
                if mode != "dcr" && mode != "crd" {
                    return Err(format!(
                        "{nl}: d2s mode is '{mode}'. The ONNX schema for DepthToSpace names \
                         exactly two -- DCR and CRD -- and the default when the attribute is \
                         absent is DCR. They differ only in how the input channel axis is split \
                         across the block, and the wrong one is a rearrangement that still fits."
                    ));
                }
                layer.crd = mode == "crd";
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
                    "{nl}: takes operand {input}, which is not an earlier op and not the image \
                     (-1). This graph is evaluated in order, so a forward reference is not a \
                     cycle the runtime could resolve -- it is a graph whose outputs it cannot \
                     compute."
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
            // A sparse index; `conv_of` refuses an op that names one.
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
    // The slope count is DERIVED from the op list rather than declared: an
    // activation naming a slope index is the only consumer of these vectors.
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
            "{label}: this is an arch={arch} container and this build reads arch={ARCH}. The \
             two networks here carry depthwise convolutions the array cannot express, a \
             DepthToSpace the other architectures do not have, two different normalisations, \
             and a detector that decodes xy1/xy2 rather than cx/cy/w/h."
        ));
    }
    let det_input_size = need_i64(container, "det_input_size", label)?;
    let pose_input_size = need_i64(container, "pose_input_size", label)?;
    let num_landmarks = need_i64(container, "num_landmarks", label)?;
    let num_keypoints = need_i64(container, "num_keypoints", label)?;
    let lm_cols = need_i64(container, "lm_cols", label)?;
    let world_cols = need_i64(container, "world_cols", label)?;
    if num_landmarks != NUM_LANDMARKS
        || num_keypoints != NUM_KEYPOINTS
        || lm_cols != LM_COLS
        || world_cols != WORLD_COLS
    {
        return Err(format!(
            "{label}: the container declares {num_landmarks} landmarks of {lm_cols} columns, \
             {num_keypoints} keypoints and {world_cols} world columns; MediaPipe Pose emits 39 \
             rows of 5 (33 keypoints plus six auxiliary, each x/y/z/visibility/presence) and 39 \
             of 3. A landmark head of the wrong width produces a skeleton with a plausible \
             number of joints and the wrong ones."
        ));
    }
    if det_input_size != 224 || pose_input_size != 256 {
        return Err(format!(
            "{label}: the two input sizes are {det_input_size} and {pose_input_size}. This \
             architecture's detector is 224x224 and its landmark net 256x256 -- the second is \
             larger because the region of interest is already a rotated square of the right \
             shape and only has to be resampled."
        ));
    }

    // -- the two normalisations, which are TWO MODELS' worth of preprocessing
    let det_mean: Vec<f32> = doubles(container, "det_image_mean", 3, label)?
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let det_std: Vec<f32> = doubles(container, "det_image_std", 3, label)?
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let pose_mean: Vec<f32> = doubles(container, "pose_image_mean", 3, label)?
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let pose_std: Vec<f32> = doubles(container, "pose_image_std", 3, label)?
        .into_iter()
        .map(|v| v as f32)
        .collect();
    for (name, sd) in [("det_image_std", &det_std), ("pose_image_std", &pose_std)] {
        if sd.iter().any(|s| *s == 0.0) {
            return Err(format!(
                "{label}: {name} has a zero. Both front ends normalise by it, so that is a \
                 division by zero on every pixel of every frame."
            ));
        }
    }

    // -- the detector's constants
    let det_num_anchors = need_i64(container, "det_num_anchors", label)?;
    let det_row_terms = need_i64(container, "det_row_terms", label)?;
    let det_landmarks = need_i64(container, "det_landmarks", label)?;
    if det_row_terms != DET_ROW_TERMS || det_landmarks != DET_LANDMARKS {
        return Err(format!(
            "{label}: det_row_terms is {det_row_terms} with {det_landmarks} landmarks, expected \
             12 with 4 (xy1, xy2, then four keypoints x 2). This detector decodes CORNERS \
             against an anchor centre; reading that as arch=7's cx,cy,w,h doubles every box's \
             position error."
        ));
    }
    let score_threshold = need_f64(container, "score_threshold", label)?;
    let nms_threshold = need_f64(container, "nms_threshold", label)?;
    let top_k = need_i64(container, "top_k", label)?;

    // -- the landmark network's geometry
    let person_lm_mid_hip = need_i64(container, "person_lm_mid_hip", label)?;
    let person_lm_full_body = need_i64(container, "person_lm_full_body", label)?;
    if !(0..DET_LANDMARKS).contains(&person_lm_mid_hip)
        || !(0..DET_LANDMARKS).contains(&person_lm_full_body)
    {
        return Err(format!(
            "{label}: the rotation is taken between the detector's keypoints \
             {person_lm_mid_hip} and {person_lm_full_body}, and the detector emits \
             {DET_LANDMARKS}. WHICH two choose the rotation decides where the landmark network \
             looks; a different pair shows it a tilted person and it returns 39 rows that are \
             all slightly wrong."
        ));
    }
    let person_box_pre_enlarge = need_f64(container, "person_box_pre_enlarge", label)?;
    let person_box_enlarge = need_f64(container, "person_box_enlarge", label)?;
    let pose_conf_threshold = need_f64(container, "pose_conf_threshold", label)?;

    // -- the detector's head, one entry per pyramid level
    let head = json(container, "det_head", label)?;
    let head = head
        .as_array()
        .ok_or_else(|| format!("{label}: det_head is not a list of levels"))?;
    if head.is_empty() {
        return Err(format!("{label}: det_head lists no levels"));
    }
    let mut det_levels = Vec::with_capacity(head.len());
    for (i, level) in head.iter().enumerate() {
        let el = format!("{label}/det_head[{i}]");
        let l = DetLevel {
            box_op: jint(level, "box_op", &el)? as i32,
            score_op: jint(level, "score_op", &el)? as i32,
            h: jint(level, "h", &el)?,
            w: jint(level, "w", &el)?,
            per: jint(level, "per", &el)?,
            rows: jint(level, "rows", &el)?,
            terms: jint(level, "terms", &el)?,
        };
        if l.rows != l.h * l.w * l.per {
            return Err(format!(
                "{el}: rows={} against h*w*per = {}. `rows` is the level's share of the anchor \
                 table and the decode reads it, so a disagreement here is a table read at the \
                 wrong stride.",
                l.rows,
                l.h * l.w * l.per
            ));
        }
        if l.terms != det_row_terms {
            return Err(format!(
                "{el}: terms={} against det_row_terms={det_row_terms}",
                l.terms
            ));
        }
        if l.per < 1 || l.h < 1 || l.w < 1 {
            return Err(format!("{el}: a level with no cells or no anchors"));
        }
        det_levels.push(l);
    }

    // -- the landmark head: five outputs, each from one graph node
    let hl = format!("{label}/pose_head");
    let h = json(container, "pose_head", label)?;
    if !h.is_object() {
        return Err(format!("{hl}: pose_head is not an object of named outputs"));
    }
    let mut pose_head = Vec::with_capacity(POSE_OUT_NAMES.len());
    for name in POSE_OUT_NAMES {
        let e = h.get(name).ok_or_else(|| format!("{hl}: no '{name}'"))?;
        let ol = format!("{hl}/{name}");
        let op = jint(e, "op", &ol)? as i32;
        let transposed = e
            .get("transposed")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let sigmoid = e.get("sigmoid").and_then(|v| v.as_bool()).unwrap_or(false);
        // `decl` is ANY rank -- MediaPipe declares the landmarks as [1,195] and the
        // mask as [1,256,256,1], and a Reshape is what connects them. Only
        // `conv_out` is pinned to [1,C,H,W].
        let d = e
            .get("decl")
            .and_then(|v| v.as_array())
            .filter(|a| !a.is_empty())
            .ok_or_else(|| format!("{ol}: 'decl' is missing or empty"))?;
        let decl: Vec<i64> = d.iter().map(|v| v.as_i64().unwrap_or(0)).collect();
        let c = e
            .get("conv_out")
            .and_then(|v| v.as_array())
            .filter(|a| a.len() == 4)
            .ok_or_else(|| format!("{ol}: 'conv_out' is not a [1, C, H, W] shape"))?;
        let conv_out = [
            c[0].as_i64().unwrap_or(0),
            c[1].as_i64().unwrap_or(0),
            c[2].as_i64().unwrap_or(0),
            c[3].as_i64().unwrap_or(0),
        ];
        // A Reshape and a Transpose each preserve the element count, so equality of
        // the two products is the invariant that covers all five without assuming a
        // layout -- and it catches a re-export that changed 39 rows into something
        // else while every shape still parsed.
        let dn: i64 = decl.iter().product();
        let cn: i64 = conv_out.iter().product();
        if dn != cn {
            return Err(format!(
                "{ol}: declares {dn} values and its convolution writes {cn}. One of them is a \
                 transcription."
            ));
        }
        if decl[0] != 1 || conv_out[0] != 1 {
            return Err(format!(
                "{ol}: these graphs run one image at a time, so neither shape has a batch axis \
                 to carry anything but 1"
            ));
        }
        pose_head.push(PoseOut {
            name: name.to_string(),
            op,
            decl,
            conv_out,
            transposed,
            sigmoid,
        });
    }
    let conf = pose_head
        .iter()
        .find(|p| p.name == "conf")
        .expect("conf is in the five names");
    if !conf.sigmoid {
        return Err(format!(
            "{hl}: the confidence carries no sigmoid. The zoo compares this number against \
             confThreshold={pose_conf_threshold} and returns it, and both assume a probability \
             rather than a logit; an unsquashed one would read about -9.7 and silently reject \
             every person."
        ));
    }
    let mut pose_unused = Vec::new();
    for e in json(container, "pose_unused", label)?
        .as_array()
        .cloned()
        .unwrap_or_default()
    {
        if let Some(s) = e.as_str() {
            pose_unused.push(s.to_string());
        }
    }

    // -- the weight precision
    let dtype = container
        .config()
        .get_str("conv_weight_dtype")
        .unwrap_or("f32");
    if !matches!(dtype, "f32" | "bf16" | "i8" | "i4") {
        return Err(format!(
            "{label}: conv_weight_dtype is '{dtype}', and this runtime reads f32, bf16, i8 and \
             i4. There is no fp16 or fp8 convolution weight here, and reading either as one of \
             the four produces a network with the right shapes and no relationship to the \
             checkpoint."
        ));
    }
    let i4_group = match container.config().get_i64("conv_int4_group") {
        Some(v) => v,
        None if dtype == "i4" => {
            return Err(format!(
                "{label}: conv_weight_dtype is 'i4' and conv_int4_group is not in the \
                 container. The group size is what places each weight against its own group's \
                 scale, so a reader that had to default it would choose the stride of the \
                 dequantisation itself."
            ));
        }
        None => 0,
    };
    if i4_group < 0 {
        return Err(format!(
            "{label}: conv_int4_group is {i4_group}. Zero means one group over the whole input \
             axis -- per-channel int4 -- and is the only value below 1 with a meaning."
        ));
    }
    if i4_group > 0 && dtype != "i4" {
        return Err(format!(
            "{label}: conv_int4_group is {i4_group} and conv_weight_dtype is '{dtype}'. The \
             group only means anything for four-bit weights, so a container that states both \
             has a scale the reader cannot place."
        ));
    }

    // -- the two graphs and their weights
    let det = read_one_graph(
        container,
        "det",
        "det_graph",
        "det_graph_nodes",
        need_i64(container, "det_num_convs", label)?,
        dtype,
        i4_group,
        &format!("{label}/det"),
    )?;
    let pose = read_one_graph(
        container,
        "pose",
        "pose_graph",
        "pose_graph_nodes",
        need_i64(container, "pose_num_convs", label)?,
        dtype,
        i4_group,
        &format!("{label}/pose"),
    )?;

    // -- the anchor table, STORED, and cross-checked against the levels
    //
    // The table is read rather than generated, because the packer generated it and
    // checked it against the OpenCV zoo's literal 2254 x 2 table bit for bit. A
    // table that has been verified once should be shipped rather than recomputed
    // on every reader. `det_anchor_levels` is still read and still compared against
    // the head's own levels: two spellings of one pyramid.
    let an = "det_anchors";
    let (ashape, _) = tensor_shape(container, an).ok_or_else(|| {
        format!(
            "{label}: {an} is not in the container. The 2254 SSD anchor centres are STORED, not \
             regenerated here: the packer derives them from the graph's three DepthToSpace \
             levels and reproduces the zoo's literal table bit for bit, and re-deriving them on \
             a second machine would make that a fact about every machine's arithmetic rather \
             than about this one."
        )
    })?;
    if ashape.len() != 2 || ashape[0] != det_num_anchors || ashape[1] != 2 {
        return Err(format!(
            "{label}: {an} is {}x{} and the container declares {det_num_anchors} anchors of 2.",
            ashape.first().copied().unwrap_or(0),
            ashape.get(1).copied().unwrap_or(0)
        ));
    }
    let anchors: Vec<f64> = f32s(container, an, label)?
        .into_iter()
        .map(|v| v as f64)
        .collect();
    let levels = json(container, "det_anchor_levels", label)?;
    let levels = levels
        .as_array()
        .ok_or_else(|| format!("{label}: det_anchor_levels is not a list"))?;
    if levels.len() != det_levels.len() {
        return Err(format!(
            "{label}: det_anchor_levels lists {} levels and the head has {}. These are two \
             spellings of one pyramid and they are compared here rather than derived, so a \
             disagreement is a refusal and not a 2254-row table read at the wrong stride.",
            levels.len(),
            det_levels.len()
        ));
    }
    for (i, level) in levels.iter().enumerate() {
        let array = level
            .as_array()
            .ok_or_else(|| format!("{label}: det_anchor_levels[{i}] is not a list"))?;
        if array.len() != 3 {
            return Err(format!(
                "{label}: det_anchor_levels[{i}] is not [h, w, anchors_per_cell]"
            ));
        }
        let (h, w, a) = (
            array[0].as_i64().unwrap_or(0),
            array[1].as_i64().unwrap_or(0),
            array[2].as_i64().unwrap_or(0),
        );
        let l = &det_levels[i];
        if h != l.h || w != l.w || a != l.per {
            return Err(format!(
                "{label}: det_anchor_levels[{i}] is {h}x{w} with {a} anchors; the head's level \
                 is {}x{} with {}",
                l.h, l.w, l.per
            ));
        }
    }

    Ok(Geometry {
        det_input_size,
        pose_input_size,
        num_landmarks,
        num_keypoints,
        lm_cols,
        world_cols,
        det_mean,
        det_std,
        pose_mean,
        pose_std,
        det_num_anchors,
        det_row_terms,
        det_landmarks,
        score_threshold,
        nms_threshold,
        top_k,
        det_levels,
        person_lm_mid_hip,
        person_lm_full_body,
        person_box_pre_enlarge,
        person_box_enlarge,
        pose_conf_threshold,
        pose_head,
        pose_unused,
        det_graph: det.layers,
        pose_graph: pose.layers,
        det_convs: det.convs,
        pose_convs: pose.convs,
        det_nodes: det.nodes,
        pose_nodes: pose.nodes,
        det_slopes: det.slopes,
        pose_slopes: pose.slopes,
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
    /// there is no `[K, N]` view to build. Passing an untransposed dense filter
    /// is a shape-compatible mistake -- both are K*N floats -- that yields a
    /// convolution with a TRANSPOSED filter.
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

/// What one network's walk cost.
///
/// The split that matters on this architecture is between the dense convolutions
/// an array panel could take and the depthwise ones it cannot.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cost {
    /// Dense convolutions that ran on the host.
    pub convs_host: i64,
    /// Depthwise convolutions, which no array can take.
    pub dwconvs_host: i64,
    /// add, maxpool, resize and d2s.
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
    /// Seconds in DepthToSpace.
    pub t_d2s: f64,
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
pub struct MpPose {
    geom: Geometry,
    det: NetWeights,
    pose: NetWeights,
    threads: usize,
}

/// `act1` -- the four activations, as a function of one element.
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

/// The graph's declared output extent, recomputed from the op and REFUSED if it
/// disagrees.
///
/// FLOOR, NOT EXACT DIVISION, and that is the whole point about the padding
/// here: the detector's stem is 223/2 = 111.5 and ONNX takes floor. A check
/// demanding an exact division refuses the model's own first convolution.
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
    if eh < 0 || ew < 0 {
        return Err(format!(
            "{label}: op '{}' has pad [{},{},{},{}] with a {kh}x{kw} kernel on a {h}x{w} input, \
             so the kernel is larger than the padded input and there is no output at all.",
            layer.op.as_str(),
            layer.pad[0],
            layer.pad[1],
            layer.pad[2],
            layer.pad[3]
        ));
    }
    let oh = eh / layer.stride_h + 1;
    let ow = ew / layer.stride_w + 1;
    if oh != layer.out_h || ow != layer.out_w {
        return Err(format!(
            "{label}: op '{}' computes {oh}x{ow} and the container says its output is {}x{}. \
             The two are written by the packer and here from the same op, so a disagreement is \
             a container whose op list was edited -- and the usual cause is a four-term pad \
             transcribed as a symmetric one.",
            layer.op.as_str(),
            layer.out_h,
            layer.out_w
        ));
    }
    Ok((oh, ow))
}

/// The host-side scratch for one walk: the im2col rows and the GEMM output.
///
/// A walk runs up to 161 dense convolutions and the detector's stem alone
/// gathers tens of thousands of rows, so these are megabyte buffers. Allocating
/// and dropping a fresh pair per convolution is allocator work the frame does
/// not need; resizing a buffer that is kept grows it to the largest shape and
/// then reuses it, never shrinking.
struct ConvHostScratch {
    a: Vec<f32>,
    c: Vec<f32>,
}

/// im2col with FOUR padding terms, into a caller-owned buffer.
///
/// The k index is `(ci*kh + i)*kw + j` and the row index is `y*W + x`, which is
/// the order the weights are read in too. Zero outside the window, and not a
/// clamp: a clamped edge replicates the border pixel, which is a different
/// network.
///
/// The buffer is resized rather than replaced, so the walk's scratch survives
/// between convolutions. The gather only writes the IN-window columns; the
/// out-of-window ones must be zero, and a reused buffer still holds the
/// previous convolution's values there, so the row span is cleared first.
fn im2col_into(
    input: &Tensor,
    kh: i64,
    kw: i64,
    pad: [i64; 4],
    sh: i64,
    sw: i64,
    oh: i64,
    ow: i64,
    a: &mut Vec<f32>,
) {
    let (c, h, w) = (input.c, input.h, input.w);
    let m = oh * ow;
    let k = c * kh * kw;
    let len = (m * k) as usize;
    a.resize(len, 0.0);
    a.fill(0.0);
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
}

/// The `[M, N]` to NCHW transpose, with the activation fused into it.
fn transpose_mn_to_nchw(src: &[f32], m: i64, n: i64, dst: &mut [f32], act: Act, slope: &[f32]) {
    const BM: usize = 32;
    const BN: usize = 32;
    let (m, n) = (m as usize, n as usize);
    // A blocked transpose: gather a BM x BN tile of the [M, N] result into a
    // small L1-resident buffer, then scatter it into the NCHW destination one
    // plane run at a time. Both sides are contiguous runs. The old version
    // tiled only M and then swept all N channels, so for each 32-row block it
    // kept N separate source strides live at once; at the landmark net's
    // widths that is hundreds of cache lines in flight for a 32-float write.
    let mut tile = [0f32; BM * BN];
    let mut m0 = 0usize;
    while m0 < m {
        let m1 = (m0 + BM).min(m);
        let mut n0 = 0usize;
        while n0 < n {
            let n1 = (n0 + BN).min(n);
            for r in 0..m1 - m0 {
                let row = &src[(m0 + r) * n + n0..][..n1 - n0];
                tile[r * BN..r * BN + (n1 - n0)].copy_from_slice(row);
            }
            for c in 0..n1 - n0 {
                let sl = slope.get(n0 + c).copied().unwrap_or(0.0);
                let d = &mut dst[(n0 + c) * m + m0..][..m1 - m0];
                for r in 0..m1 - m0 {
                    d[r] = act1(act, tile[r * BN + c], sl);
                }
            }
            n0 = n1;
        }
        m0 = m1;
    }
}

/// The GEMM over ROW RANGES, one per thread.
///
/// A row's accumulation is over k in order, so splitting ROWS changes which
/// thread computes a value and never the value itself -- the split is bit-exact
/// by construction.
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
    // One policy for every host GEMM: shared pool, work threshold and a
    // minimum number of rows per worker, all bit-exact by construction.
    kernels::gemm_nt_rows(a, m, b, k, n, bias, out, threads);
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

impl MpPose {
    /// Read both graphs out of a container.
    pub fn open(path: &Path) -> Result<MpPose, String> {
        let container = Container::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let label = path.display().to_string();
        let geom = read_geometry(&container, &label)?;
        let det = NetWeights::build(&geom.det_convs);
        let pose = NetWeights::build(&geom.pose_convs);
        Ok(MpPose {
            geom,
            det,
            pose,
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

    /// Walk the detector's graph. Returns every node's output, in order.
    ///
    /// `array` is the array's backend, when the run asked for the array: the
    /// detector's DENSE convolutions then dispatch to the board and the
    /// depthwise ones stay on the host -- a depthwise filter reduces WITHIN
    /// one channel and has no `[M, N]` product in it to dispatch.
    pub fn run_det(
        &self,
        input: &Tensor,
        hook: &mut dyn FnMut(usize, &Tensor),
        cost: &mut Cost,
        mut array: ArrayRef<'_, '_>,
    ) -> Result<Vec<Tensor>, String> {
        self.walk(
            input,
            &self.geom.det_graph,
            &self.geom.det_convs,
            &self.det,
            &self.geom.det_slopes,
            &self.geom.det_nodes,
            "det",
            hook,
            cost,
            array.as_deref_mut(),
        )
    }

    /// Walk the landmark network's graph. Returns every node's output.
    ///
    /// The landmark network runs ONCE PER PERSON, so its share of the array's
    /// work is counted as many times as there are people -- and the DISPATCH
    /// COUNT is not: one backend serves both graphs, and it is asked for its
    /// own count once, at the end of the frame.
    pub fn run_pose(
        &self,
        input: &Tensor,
        hook: &mut dyn FnMut(usize, &Tensor),
        cost: &mut Cost,
        mut array: ArrayRef<'_, '_>,
    ) -> Result<Vec<Tensor>, String> {
        self.walk(
            input,
            &self.geom.pose_graph,
            &self.geom.pose_convs,
            &self.pose,
            &self.geom.pose_slopes,
            &self.geom.pose_nodes,
            "pose",
            hook,
            cost,
            array.as_deref_mut(),
        )
    }

    /// The walk: the graph in order, the hook on every node.
    ///
    /// EVERY NODE IS KEPT, not the live ones, because the heads read nodes in
    /// the MIDDLE of the graph: the detector's three box convolutions are nodes
    /// 90, 74 and 58 of 93. A walk that freed them would have to be told which to
    /// keep, and a head that read a freed buffer would get 2254 well-shaped rows.
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
        cost: &mut Cost,
        mut array: ArrayRef<'_, '_>,
    ) -> Result<Vec<Tensor>, String> {
        let mut vals: Vec<Tensor> = vec![Tensor::default(); graph.len()];
        // ONE pair of host buffers for the whole walk. The convolutions reuse
        // them; the storage grows to the largest shape and is never dropped
        // between nodes.
        let mut host = ConvHostScratch {
            a: Vec::new(),
            c: Vec::new(),
        };
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
                &mut host,
                array.as_deref_mut(),
            )?;
            vals[i] = out;
            // AFTER the op, so the hook sees the activation too: it fires on the
            // node's OUTPUT, and an activation fused into the convolution is part
            // of what the node produced.
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
        host: &mut ConvHostScratch,
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
                    "{label}: prelu {idx} has {} slopes for a {}-channel tensor. PReLU's slope \
                     is PER CHANNEL, so a length mismatch has no per-pixel meaning.",
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
                        "{label}: op 'conv' names conv {}, which the container has no weights \
                         for. Both lists are written by the same packer pass, so a disagreement \
                         here means the container was edited or half-written -- and a \
                         convolution with no weights is a graph that would run on whatever was \
                         in the buffer.",
                        layer.conv
                    ));
                };
                self.conv(first, layer, conv, net, slope, label, cost, host, array)
            }
            Op::DwConv => {
                let conv = convs.get(layer.conv as usize).filter(|c| !c.w.is_empty());
                let Some(conv) = conv else {
                    return Err(format!(
                        "{label}: op 'dwconv' names conv {}, which the container has no weights \
                         for.",
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
            Op::Resize => self.resize(first, layer, label, cost),
            Op::D2S => self.d2s(first, layer, label, cost),
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
    /// They do NOT produce the same numbers, and that is not a bug: the array
    /// multiplies the weights rounded to bf16 and the host multiplies them
    /// in f32, so a run on the array is the same network with a coarser
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
        host: &mut ConvHostScratch,
        mut array: ArrayRef<'_, '_>,
    ) -> Result<Tensor, String> {
        let (oh, ow) = check_extent(layer, input.h, input.w, w.kh, w.kw, label)?;
        if w.cin != input.c {
            return Err(format!(
                "{label}: conv {} takes {} channels and the tensor has {}. Reading the other \
                 one would index past the end of the im2col row.",
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
                "{label}: conv {} has no [K,N] weight table. The transpose is built at load for \
                 every DENSE filter, so a missing one means this convolution is depthwise and \
                 was routed here by name -- which would compute a different network rather than \
                 fail.",
                w.index
            ));
        }
        let t = std::time::Instant::now();
        im2col_into(
            input,
            w.kh,
            w.kw,
            layer.pad,
            layer.stride_h,
            layer.stride_w,
            oh,
            ow,
            &mut host.a,
        );
        cost.t_im2col += t.elapsed().as_secs_f64();
        let m = (oh * ow) as usize;
        let k = w.kin() as usize;
        let n = w.cout as usize;
        // The walk's output buffer, resized to this convolution's shape. The
        // storage survives across convolutions; the contents are fully written
        // by either path below.
        host.c.resize(m * n, 0.0);
        // THE ARRAY'S SPAN, AND THE HOST'S.
        //
        // The array's own time is measured around the dispatch, and the A
        // repack inside it is measured by the backend -- the host's own work,
        // and not the device's, so it is reported separately rather than folded
        // into the device's number.
        let mut array_s = 0.0f64;
        let mut repack_s = 0.0f64;
        let mut dispatched = 0i64;
        let mut convs_array = 0i64;
        let gemm_s = match array.as_deref_mut() {
            Some(array) => {
                // The panel's name is the weight's own, and the two graphs'
                // convolutions are namespaced -- "det.conv.3.w" and
                // "pose.conv.3.w" are two different panels.
                let name = format!("{}.conv.{}.w", layer.net, w.index);
                let before_dispatch = array.dispatches() as i64;
                let before_convert = array.convert_seconds();
                let t = std::time::Instant::now();
                array
                    .conv(&name, &host.a, m, k, n, &mut host.c)
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
                gemm_rows(&host.a, m, wmat, k, n, &w.b, &mut host.c, self.threads);
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
            &host.c,
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
    /// -- it wants a K-reduction across N output columns -- and it is why 62 of
    /// this container's 161 convolutions stay on the host by name.
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
                "{label}: dwconv {} has {} filters and the tensor has {} channels. One filter \
                 per channel means they are the same number, and reading it as a dense \
                 convolution would compute a different network rather than fail.",
                w.index, w.cout, input.c
            ));
        }
        if layer.out_c != input.c {
            return Err(format!(
                "{label}: dwconv {} changes the channel count from {} to {}. A depthwise \
                 convolution cannot: it would need a projection, which is what the `conv` after \
                 it is for.",
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
        // Only the TOP and LEFT terms are read here: they are the offsets of the
        // gather. The bottom and right terms are checked by check_extent()
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
                "{label}: add joins a {}x{}x{} and a {}x{}x{}. Neither network has a broadcast \
                 add in its graph, so the two shapes are a transcription error rather than \
                 something to broadcast: broadcasting here would silently accept a block that \
                 resized instead of adding.",
                a.c, a.h, a.w, b.c, b.h, b.w
            ));
        }
        if layer.out_c != a.c || layer.out_h != a.h || layer.out_w != a.w {
            return Err(format!(
                "{label}: add's declared output is {}x{}x{} and its operands are {}x{}x{}",
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
                    // A window entirely in the padding has no samples. -inf is the
                    // honest value for a max over nothing; a 0 default would be a
                    // different tensor.
                    dst[(y * ow + x) as usize] = best;
                }
            }
        }
        cost.t_elementwise += t.elapsed().as_secs_f64();
        cost.elementwise_ops += 1;
        Ok(out)
    }

    /// ONNX Resize, linear, half_pixel, exclude_outside=0.
    ///
    /// `src = (i + 0.5) / scale - 0.5`, and each weight belongs to the axis its
    /// pass moves along. The rounding is the oracle's: its compiler turned each
    /// blend into one rounded multiply and one fused multiply-add (`vmulss` then
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

    /// DepthToSpace.
    ///
    /// `out[co, h*b + d1, w*b + d2] = in[ci, h, w]`, where the input channel
    /// index `ci` is the only thing the two ONNX orders disagree about:
    ///
    /// ```text
    /// CRD  ci = co*b*b + d1*b + d2      the axis splits as (co, d1, d2)
    /// DCR  ci = (d1*b + d2)*cout + co   the axis splits as (d1, d2, co) -- the
    ///                                   DEFAULT when the attribute is absent
    /// ```
    ///
    /// Everything else is a copy, so the work is `b*b` plane copies per output
    /// channel and no arithmetic at all.
    fn d2s(
        &self,
        input: &Tensor,
        layer: &Layer,
        label: &str,
        cost: &mut Cost,
    ) -> Result<Tensor, String> {
        let b = layer.block;
        if b < 1 {
            return Err(format!("{label}: d2s has blocksize {b}"));
        }
        let bb = b * b;
        if input.c % bb != 0 {
            return Err(format!(
                "{label}: d2s has {} input channels and a blocksize of {b}, and {} is not a \
                 multiple of b^2. DepthToSpace divides the CHANNEL count by the block area; \
                 there is no other reading of an indivisible one.",
                input.c, input.c
            ));
        }
        let cout = input.c / bb;
        if layer.out_c != cout || layer.out_h != input.h * b || layer.out_w != input.w * b {
            return Err(format!(
                "{label}: d2s goes {}x{}x{} -> {}x{}x{} with blocksize {b}, which is not the \
                 [C/b^2, H*b, W*b] this operator means.",
                input.c, input.h, input.w, layer.out_c, layer.out_h, layer.out_w
            ));
        }
        let t = std::time::Instant::now();
        let crd = layer.crd;
        let (h, w) = (input.h, input.w);
        let mut out = Tensor {
            c: layer.out_c,
            h: layer.out_h,
            w: layer.out_w,
            d: vec![0.0f32; (layer.out_c * layer.out_h * layer.out_w) as usize],
        };
        let plane_in = (h * w) as usize;
        let plane_out = (layer.out_h * layer.out_w) as usize;
        for co in 0..cout {
            let dst = &mut out.d[(co as usize) * plane_out..][..plane_out];
            for d1 in 0..b {
                for d2 in 0..b {
                    // DCR is ONNX's default and this model's case: `mode` is
                    // absent and the schema says DCR.
                    let ci = if crd {
                        (co * b + d1) * b + d2
                    } else {
                        (d1 * b + d2) * cout + co
                    };
                    let src = &input.d[(ci as usize) * plane_in..][..plane_in];
                    // OUT ROW = y*b + d1 AND OUT COLUMN = x*b + d2: both put the
                    // WITHIN-BLOCK position in the low part. d1 and d2 are BOTH
                    // outer loops because each SELECTS THE SOURCE PLANE -- ci
                    // depends on both -- and neither is a broadcast.
                    for y in 0..h {
                        let drow = &mut dst[((y * b + d1) * layer.out_w) as usize..];
                        let srow = &src[(y * w) as usize..][..w as usize];
                        for x in 0..w {
                            drow[(x * b + d2) as usize] = srow[x as usize];
                        }
                    }
                }
            }
        }
        cost.t_d2s += t.elapsed().as_secs_f64();
        cost.elementwise_ops += 1;
        Ok(out)
    }

    /// The detector's head: three levels of two 1x1 outputs into `[n, 12]` and `[n]`.
    ///
    /// The reshape is done in the graph's own axis order: the checkpoint
    /// transposes NCHW to NHWC and reshapes [H,W,C] to [H*W*A, terms], so row
    /// (h*W + w)*A + a, term t, is channel a*terms + t of cell (h, w). The flat
    /// row index therefore advances w -- the X coordinate -- fastest, which is
    /// the order the STORED anchor table is in.
    pub fn det_head(&self, nodes: &[Tensor]) -> Result<DetOut, String> {
        let g = &self.geom;
        let mut boxes: Vec<f64> = Vec::new();
        let mut scores: Vec<f64> = Vec::new();
        for (li, level) in g.det_levels.iter().enumerate() {
            let el = format!("det level {li}");
            let b = self.node(nodes, level.box_op, &g.det_nodes, "det")?;
            let s = self.node(nodes, level.score_op, &g.det_nodes, "det")?;
            if b.c != level.per * g.det_row_terms || b.h != level.h || b.w != level.w {
                return Err(format!(
                    "{el}: the box feature is {}x{}x{} and the head says {}x{}x{}. Both are \
                     written by the packer from the ONNX's own output names, so a disagreement \
                     is one of them a transcription.",
                    b.c,
                    b.h,
                    b.w,
                    level.per * g.det_row_terms,
                    level.h,
                    level.w
                ));
            }
            if s.c != level.per || s.h != level.h || s.w != level.w {
                return Err(format!(
                    "{el}: the score feature is {}x{}x{} against {}x{}x{}",
                    s.c, s.h, s.w, level.per, level.h, level.w
                ));
            }
            boxes.reserve((level.rows * g.det_row_terms) as usize);
            scores.reserve(level.rows as usize);
            let cells = (level.h * level.w) as usize;
            for y in 0..level.h {
                for x in 0..level.w {
                    let cell = (y * level.w + x) as usize;
                    for a in 0..level.per {
                        for t in 0..g.det_row_terms {
                            let ch = (a * g.det_row_terms + t) as usize;
                            boxes.push(b.d[ch * cells + cell] as f64);
                        }
                        scores.push(s.d[(a as usize) * cells + cell] as f64);
                    }
                }
            }
        }
        let n = scores.len() as i64;
        if n != g.det_num_anchors {
            return Err(format!(
                "the levels produced {n} anchors and the container declares {}. The decode reads \
                 a table of the declared width, so this is a refusal rather than a read past the \
                 end.",
                g.det_num_anchors
            ));
        }
        Ok(DetOut { boxes, scores, n })
    }

    /// The landmark head: five single convolutions, in the container's order.
    pub fn pose_head(&self, nodes: &[Tensor]) -> Result<PoseOuts, String> {
        let g = &self.geom;
        let hl = "pose landmark head";
        let mut o = PoseOuts::default();

        // landmarks: declared [1, 195] and the convolution writes [1, 195, 1, 1],
        // so the flat order is the graph's own and nothing is reordered.
        {
            let p = g.pose_out("landmarks", hl)?;
            let t = self.node(nodes, p.op, &g.pose_nodes, hl)?;
            if p.transposed {
                return Err(format!(
                    "{hl}/landmarks: the container records a transpose on the way out, and this \
                     output is the screen landmark table -- 39 rows of x/y/z/visibility/presence \
                     -- which a transpose would turn into 195 coordinates that are individually \
                     plausible."
                ));
            }
            let want = (g.num_landmarks * g.lm_cols) as usize;
            if t.d.len() != want {
                return Err(format!(
                    "{hl}/landmarks: node {} holds {} values and {}x{} = {want} were expected. \
                     The zoo's _postprocess reshapes this to 39 rows of 5, so a different count \
                     is a different model.",
                    p.op,
                    t.d.len(),
                    g.num_landmarks,
                    g.lm_cols
                ));
            }
            o.landmarks = t.d.iter().map(|v| *v as f64).collect();
        }

        // conf: one value, and the SIGMOID IS THIS HEAD'S JOB. The graph's Sigmoid
        // is a node of its own; the packer absorbed it and recorded that it did.
        // Reading the node raw would compare a logit against 0.5.
        {
            let p = g.pose_out("conf", hl)?;
            let t = self.node(nodes, p.op, &g.pose_nodes, hl)?;
            if t.d.len() != 1 {
                return Err(format!(
                    "{hl}/conf: node {} holds {} values; the confidence is one number",
                    p.op,
                    t.d.len()
                ));
            }
            if !p.sigmoid {
                return Err(format!(
                    "{hl}/conf: the container does not record a sigmoid on this output. It is \
                     compared against {} and returned to the caller, both of which assume a \
                     probability; an unsquashed logit reads about -9.7 and silently rejects \
                     every person.",
                    g.pose_conf_threshold
                ));
            }
            // The one definition, in float as the oracle's `sigmoid1` is.
            o.conf = (1.0f32 / (1.0f32 + (-t.d[0]).exp())) as f64;
        }

        // mask: declared [1, 256, 256, 1] and the convolution writes [1, 1, 256, 256].
        // The single channel is dropped and the flat order is unchanged -- a
        // Reshape, not a Transpose -- so the result is row-major [y, x].
        {
            let p = g.pose_out("mask", hl)?;
            let t = self.node(nodes, p.op, &g.pose_nodes, hl)?;
            if p.transposed {
                return Err(format!(
                    "{hl}/mask: the container records a transpose on the way out, and a \
                     transposed segmentation mask is still a segmentation mask -- of the wrong \
                     body."
                ));
            }
            let want = (g.pose_input_size * g.pose_input_size) as usize;
            if t.d.len() != want {
                return Err(format!(
                    "{hl}/mask: node {} holds {} values and a {}x{} mask is {want}",
                    p.op,
                    t.d.len(),
                    g.pose_input_size,
                    g.pose_input_size
                ));
            }
            o.mask = t.d.clone();
        }

        // world: declared [1, 117] over 39 rows of 3, in metres.
        {
            let p = g.pose_out("world", hl)?;
            let t = self.node(nodes, p.op, &g.pose_nodes, hl)?;
            if p.transposed {
                return Err(format!(
                    "{hl}/world: the container records a transpose on the way out, and the \
                     world output is 39 rows of a metric x/y/z -- transposed it is 117 numbers \
                     that are individually plausible and jointly wrong."
                ));
            }
            let want = (g.num_landmarks * g.world_cols) as usize;
            if t.d.len() != want {
                return Err(format!(
                    "{hl}/world: node {} holds {} values and {}x{} = {want} were expected",
                    p.op,
                    t.d.len(),
                    g.num_landmarks,
                    g.world_cols
                ));
            }
            o.world = t.d.iter().map(|v| *v as f64).collect();
        }

        // heatmap: declared [1, 64, 64, 39] and the convolution writes [1, 39, 64, 64].
        // This one IS transposed, so decl[y][x][c] = node[c][y][x] and the copy
        // has to be strided rather than linear. It is read so that a gate can
        // compare it, and NOT USED by anything.
        {
            let p = g.pose_out("heatmap", hl)?;
            let t = self.node(nodes, p.op, &g.pose_nodes, hl)?;
            if p.decl.len() != 4 {
                return Err(format!(
                    "{hl}/heatmap: declared {} axes and it is the one output whose ORDER is the \
                     content -- it is transposed [0,2,3,1] on the way out of a [1,39,64,64] \
                     convolution, so a rank-2 or rank-3 spelling would leave [h,w,c] and [c,h,w] \
                     indistinguishable.",
                    p.decl.len()
                ));
            }
            let (cc, hh, ww) = (p.decl[3], p.decl[1], p.decl[2]);
            if t.c != cc || t.h != hh || t.w != ww {
                return Err(format!(
                    "{hl}/heatmap: declared [{cc}, {hh}, {ww}] and the node holds {}x{}x{}. The \
                     heatmap is the one output where the declared axes and the convolution's are \
                     different -- it is transposed on the way out -- so they are compared here \
                     rather than assumed equal.",
                    t.c, t.h, t.w
                ));
            }
            o.heatmap = vec![0.0f32; t.d.len()];
            let plane = (hh * ww) as usize;
            for y in 0..hh {
                for x in 0..ww {
                    for c in 0..cc {
                        o.heatmap[((y * ww + x) * cc + c) as usize] =
                            t.d[(c as usize) * plane + (y * ww + x) as usize];
                    }
                }
            }
        }

        Ok(o)
    }

    /// The node a head names, by GRAPH index.
    fn node<'a>(
        &self,
        nodes: &'a [Tensor],
        i: i32,
        names: &[String],
        which: &str,
    ) -> Result<&'a Tensor, String> {
        if i < 0 || i as usize >= nodes.len() {
            return Err(format!(
                "{which}: the head names graph node {i}, which is not one of the network's {} \
                 nodes",
                nodes.len()
            ));
        }
        let _ = names;
        Ok(&nodes[i as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blocked transpose must be the plain n-outer / m-inner copy, value
    /// for value, across tile boundaries and for every activation. The block
    /// is only meant to be faster, never a different tensor.
    #[test]
    fn blocked_transpose_matches_plain() {
        for &(m, n) in &[(1usize, 1usize), (5, 3), (32, 32), (40, 70), (129, 131)] {
            let src: Vec<f32> = (0..m * n).map(|i| (i as f32 - 7.0) / 11.0).collect();
            let slope: Vec<f32> = (0..n).map(|i| (i as f32) / 20.0).collect();
            let mut got = vec![0f32; m * n];
            let mut want = vec![0f32; m * n];
            for act in [Act::None, Act::Relu, Act::Relu6, Act::PreLu] {
                transpose_mn_to_nchw(&src, m as i64, n as i64, &mut got, act, &slope);
                for mm in 0..m {
                    for ch in 0..n {
                        want[ch * m + mm] = act1(act, src[mm * n + ch], slope[ch]);
                    }
                }
                assert_eq!(got, want, "m={m} n={n} act={act:?}");
            }
        }
    }
}
