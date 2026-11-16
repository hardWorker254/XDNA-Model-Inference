//! The pose network: the packed graph, the host convolution and the head.
//!
//! # WHAT IS IN THE CONTAINER AND WHAT IS HERE
//!
//! The container carries the GRAPH -- the op list, the convolutions' shapes,
//! their strides and their paddings -- as data, because a runtime that
//! re-derived the graph from the checkpoint would be a second
//! implementation of the exporter. This file walks the graph the container
//! describes.
//!
//! # THE CONVOLUTION IS AN im2col AND A GEMM
//!
//! The host path builds the [M, K] matrix of windows and multiplies it by
//! the [K, N] weight matrix. The index order is the whole of it: the k
//! index is `(ci*kh + i)*kw + j` and the row index is `y*W + x`, which is the
//! order the weights are read in too, so the GEMM's reduction walks the
//! kernel exactly as the checkpoint stores it.
//!
//! The weight matrix is transposed ONCE, at load: the container holds
//! `[Cout, Cin, kh, kw]` and the GEMM wants `[K, N]` -- k outer, n
//! contiguous. Passing the container's layout through unchanged is a
//! shape-compatible mistake that produces a convolution with a TRANSPOSED
//! filter, which is not a wrong answer but a different network.
//!
//! # THE HEAD
//!
//! The exported graph's last dozen nodes -- the DFL softmax, the grid
//! anchor addition, the stride multiply, the keypoint affine -- are not
//! convolutions, and this runtime's graph engine walks convolutions. So the
//! boundary is drawn at that arithmetic and it is drawn EXPLICITLY, in the
//! container, in three fields: the box format, the score kind and the
//! keypoint visibility kind.
//!
//! The DFL's PROJECTION is folded in analytically rather than run:
//!
//! ```text
//! sum_bin w[0][bin] * p[bin] + b  ==  w[0] . E[bin] + b   (sum p == 1)
//! ```
//!
//! so the head needs the WEIGHTS, not a place in the op list. The weights
//! are APPLIED, not assumed to be the bin index: they happen to be 0..15 in
//! this checkpoint, so reading `w[b]` as `b` would give the same number
//! -- for this checkpoint.

use npum_format::Container;
use npum_model::array::{ArrayRef, ConvSpec};
use serde_json::Value;
use std::path::Path;

/// The graph's op kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// One convolution, with an optional fused SiLU.
    Conv,
    /// Channel-wise, any number of operands.
    Concat,
    /// A contiguous run of CHANNELS out of a tensor.
    Slice,
    /// The residual.
    Add,
    /// The pooling.
    MaxPool,
    /// Nearest-neighbour, the neck's up path.
    Upsample,
    /// The head.
    Detect,
}

/// How to read the box head's channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxFormat {
    /// 4 * bins channels of logits: the head softmaxes, takes the
    /// expectation over the bins, adds the grid anchor and scales by the
    /// stride.
    LtrbDfl,
    /// 4 channels, already corners, in pixels.
    Xyxy,
    /// 4 channels, already centre + full width/height, in pixels.
    CxCyWh,
}

/// Whether a channel is already a probability or a logit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreKind {
    /// Already squashed to 0..1 by the packed graph.
    Sigmoid,
    /// Raw; the head applies the sigmoid.
    Logit,
}

/// What the packed graph's `detect` node is fed, and what it emits.
#[derive(Debug, Clone)]
pub struct HeadInfo {
    /// How to read the box head's channels.
    pub box_format: BoxFormat,
    /// Whether the class head's output is a logit or a probability.
    pub score: ScoreKind,
    /// Likewise for the keypoint head.
    pub kpt_visibility: ScoreKind,
    /// Which convolution is the DFL's projection, and -1 when the container
    /// states none.
    pub dfl_conv: i32,
    /// Whether that convolution carries a bias.
    pub dfl_bias: bool,
    /// The box grid's half-cell offset, in CELLS.
    pub box_grid_offset: f32,
    /// The keypoint grid's half-cell offset, in CELLS.
    pub kpt_grid_offset: f32,
}

impl Default for HeadInfo {
    fn default() -> HeadInfo {
        HeadInfo {
            box_format: BoxFormat::Xyxy,
            score: ScoreKind::Sigmoid,
            kpt_visibility: ScoreKind::Sigmoid,
            dfl_conv: -1,
            dfl_bias: false,
            box_grid_offset: 0.5,
            kpt_grid_offset: 0.0,
        }
    }
}

/// One node of the packed graph.
#[derive(Debug, Clone)]
pub struct Layer {
    /// What the node does.
    pub op: Op,
    /// The convolution's index, for `Op::Conv`.
    pub conv: i32,
    /// Whether the activation is fused into the convolution's epilogue.
    pub silu: bool,
    /// The pooling's kernel.
    pub k: i64,
    /// The convolution's stride, and the pooling's stride.
    pub stride: i64,
    /// The rows added above and below.
    pub pad_h: i64,
    /// The columns added left and right.
    pub pad_w: i64,
    /// The upsampling factor.
    pub scale: i64,
    /// The slice's first channel.
    pub chan: i64,
    /// The slice's channel count.
    pub nch: i64,
    /// The array's stream this convolution runs on, when the container was
    /// packed with `--npu`.
    ///
    /// It is the SHAPE the convolution is dispatched against -- "conv64x128"
    /// -- and it is EMPTY for a container packed without the panels: such a
    /// container names no stream, and a convolution with no stream is a
    /// convolution the array cannot be asked about.
    pub stream: String,
    /// The operands.
    pub inputs: Vec<i32>,
}

/// One convolution's weights.
#[derive(Debug, Clone)]
pub struct ConvW {
    /// Its own position in the geometry's list.
    pub index: i32,
    /// The output channels.
    pub cout: i64,
    /// The input channels.
    pub cin: i64,
    /// The kernel's rows.
    pub kh: i64,
    /// The kernel's columns.
    pub kw: i64,
    /// The weights, `[cout, cin, kh, kw]` row-major.
    pub w: Vec<f32>,
    /// The bias, empty for a bias-free convolution.
    pub b: Vec<f32>,
}

/// The model's geometry, as the container states it.
#[derive(Debug, Clone)]
pub struct Geometry {
    /// The input's side, in pixels.
    pub input_size: i64,
    /// How many keypoints.
    pub num_keypoints: i64,
    /// How many classes.
    pub num_classes: i64,
    /// The DFL's bin count.
    pub dfl_bins: i64,
    /// One entry per level: the stride in pixels.
    pub strides: Vec<i64>,
    /// One entry per level: the cells per side.
    pub grid: Vec<i64>,
    /// The per-channel means.
    pub mean: Vec<f32>,
    /// The per-channel standard deviations.
    pub std_dev: Vec<f32>,
    /// The graph.
    pub graph: Vec<Layer>,
    /// The convolutions.
    pub convs: Vec<ConvW>,
    /// The channels the tensor leaving `Detect` has to have.
    pub detect_cout: i64,
    /// The head's own conventions.
    pub head: HeadInfo,
}

impl Geometry {
    /// The channels the tensor leaving `Detect` has to have.
    pub fn out_channels(&self) -> i64 {
        4 + self.num_classes + self.num_keypoints * 3
    }

    /// How many cells the head emits.
    pub fn n_anchors(&self) -> i64 {
        self.grid.iter().map(|g| g * g).sum()
    }

    /// How many levels.
    pub fn n_levels(&self) -> usize {
        self.strides.len()
    }
}

/// A tensor: `[c, h, w]`, channel slowest.
#[derive(Debug, Clone, Default)]
pub struct Tensor {
    /// The channels.
    pub c: i64,
    /// The rows.
    pub h: i64,
    /// The columns.
    pub w: i64,
    /// The values, channel slowest.
    pub d: Vec<f32>,
}

impl Tensor {
    /// One channel's plane.
    pub fn chw(&self, c: i64) -> &[f32] {
        let plane = (self.h * self.w) as usize;
        let start = (c as usize) * plane;
        &self.d[start..start + plane]
    }
}

/// Read the geometry out of a container.
pub fn read_geometry(container: &Container, label: &str) -> Result<Geometry, String> {
    let config = container.config();
    let get_i64 = |key: &str| -> Result<i64, String> {
        config
            .get_i64(key)
            .ok_or_else(|| format!("{label}: the container carries no {key}"))
    };
    let get_f32s = |key: &str| -> Result<Vec<f32>, String> {
        let array = config
            .get(key)
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("{label}: the container carries no {key}"))?;
        Ok(array
            .iter()
            .filter_map(|v| v.as_f64())
            .map(|v| v as f32)
            .collect())
    };
    let get_i64s = |key: &str| -> Result<Vec<i64>, String> {
        let array = config
            .get(key)
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("{label}: the container carries no {key}"))?;
        Ok(array
            .iter()
            .filter_map(|v| v.as_i64())
            .collect())
    };

    let input_size = get_i64("input_size")?;
    let num_keypoints = get_i64("num_keypoints")?;
    let num_classes = get_i64("num_classes")?;
    let dfl_bins = get_i64("dfl_bins")?;
    let strides = get_i64s("strides")?;
    let grid = get_i64s("grid")?;
    let mean = get_f32s("image_mean")?;
    let std_dev = get_f32s("image_std")?;
    let num_convs = get_i64("num_convs")? as usize;

    // The head's conventions, read out of the container rather than guessed.
    let mut head = HeadInfo {
        box_format: match config.get_str("head_box_format") {
            Some("ltrb_dfl") => BoxFormat::LtrbDfl,
            Some("xyxy") => BoxFormat::Xyxy,
            Some("cxcywh") => BoxFormat::CxCyWh,
            other => {
                return Err(format!(
                    "{label}: head_box_format is {other:?}. A heuristic that \
                     guesses \"the boxes look like corners\" is a heuristic that \
                     will be wrong on the one model where it matters, and \
                     reading xyxy as cxcywh puts every box inside the top-left \
                     quadrant and NMS then deletes most of them -- so the \
                     failure presents as an empty result rather than as a \
                     visibly wrong one."
                ))
            }
        },
        score: match config.get_str("head_score") {
            Some("logit") => ScoreKind::Logit,
            _ => ScoreKind::Sigmoid,
        },
        kpt_visibility: match config.get_str("head_keypoint_visibility") {
            Some("logit") => ScoreKind::Logit,
            _ => ScoreKind::Sigmoid,
        },
        dfl_conv: config.get_i64("head_dfl_conv").unwrap_or(-1) as i32,
        dfl_bias: false,
        box_grid_offset: config.get_f64("head_box_grid_offset").unwrap_or(0.5) as f32,
        kpt_grid_offset: config.get_f64("head_kpt_grid_offset").unwrap_or(0.0) as f32,
    };

    // The graph, as JSON in the config.
    let raw = config
        .get_str("graph")
        .ok_or_else(|| format!("{label}: the container carries no graph"))?;
    let parsed: Value = serde_json::from_str(raw)
        .map_err(|e| format!("{label}: graph is not readable JSON ({e})"))?;
    let nodes = parsed
        .as_array()
        .ok_or_else(|| format!("{label}: the graph is not a list of nodes"))?;
    let mut graph: Vec<Layer> = Vec::with_capacity(nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        let where_ = format!("{label}: graph node {i}");
        let op = match node.get("op").and_then(|v| v.as_str()) {
            Some("conv") => Op::Conv,
            Some("concat") => Op::Concat,
            Some("slice") => Op::Slice,
            Some("add") => Op::Add,
            Some("maxpool") => Op::MaxPool,
            Some("upsample") => Op::Upsample,
            Some("detect") => Op::Detect,
            other => return Err(format!("{where_}: unknown op {other:?}")),
        };
        let inputs: Vec<i32> = node
            .get("inputs")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_i64()).map(|v| v as i32).collect())
            .unwrap_or_default();
        let mut layer = Layer {
            op,
            conv: -1,
            silu: false,
            k: 0,
            stride: 0,
            pad_h: 0,
            pad_w: 0,
            scale: 0,
            chan: 0,
            nch: 0,
            stream: String::new(),
            inputs,
        };
        match op {
            Op::Conv => {
                layer.conv = node
                    .get("conv")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| format!("{where_}: a conv node carries no \"conv\" index"))?
                    as i32;
                layer.silu = node.get("silu").and_then(|v| v.as_i64()).unwrap_or(0) != 0;
                // THE STREAM, when the container names one.
                //
                // It is the packer's word, not this runtime's: the packer
                // writes the design's stream into the node it dispatches
                // that convolution against, and a container packed WITHOUT
                // the panels names none. A default invented here would be a
                // stream the design does not carry, and a convolution
                // dispatched against a missing slot is a wrong answer rather
                // than an error.
                layer.stream = node
                    .get("stream")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                // The stride and the padding are READ, not assumed: the stem
                // is stride 2 and the 1x1s pad nothing, so a default would be
                // right for most of the network and would silently resize the
                // two layers that set every resolution below them.
                layer.stride = node
                    .get("stride")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        format!(
                            "{where_}: conv carries no \"stride\". A default of 1 is \
                             right for most of this network and wrong for the stem, \
                             which downsamples by 2."
                        )
                    })?;
                if layer.stride < 1 {
                    return Err(format!(
                        "{where_}: conv stride {} is not a forward step.",
                        layer.stride
                    ));
                }
                let pad = node
                    .get("pad")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| {
                        format!(
                            "{where_}: conv carries no \"pad\". The padding is zero \
                             on a 1x1 and one on a 3x3, and which is which is the \
                             difference between a 642x642 first layer and a 640x640 \
                             one."
                        )
                    })?;
                if pad.len() != 2 {
                    return Err(format!("{where_}: conv pad has {} entries, not 2.", pad.len()));
                }
                layer.pad_h = pad[0].as_i64().unwrap_or(0);
                layer.pad_w = pad[1].as_i64().unwrap_or(0);
                if layer.pad_h < 0 || layer.pad_w < 0 {
                    return Err(format!(
                        "{where_}: conv pad is [{}, {}]. Negative padding CROPS, and \
                         this runtime pads with zeros and never crops.",
                        layer.pad_h, layer.pad_w
                    ));
                }
                if layer.conv < 0 || layer.conv as usize >= num_convs {
                    return Err(format!(
                        "{where_}: conv index {} is outside the {num_convs} \
                         convolutions this container holds",
                        layer.conv
                    ));
                }
            }
            Op::Slice => {
                layer.chan = node.get("chan").and_then(|v| v.as_i64()).unwrap_or(0);
                layer.nch = node.get("nch").and_then(|v| v.as_i64()).unwrap_or(0);
            }
            Op::MaxPool => {
                layer.k = node.get("k").and_then(|v| v.as_i64()).unwrap_or(1);
                layer.stride = node.get("stride").and_then(|v| v.as_i64()).unwrap_or(1);
            }
            Op::Upsample => {
                layer.scale = node.get("scale").and_then(|v| v.as_i64()).unwrap_or(1);
            }
            Op::Concat | Op::Add | Op::Detect => {}
        }
        graph.push(layer);
    }

    // The convolutions' weights.
    let bias_free: Vec<i64> = config
        .get("bias_free")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_i64()).collect())
        .unwrap_or_default();
    let mut convs = Vec::with_capacity(num_convs);
    for i in 0..num_convs {
        let name = format!("conv.{i}.w");
        let bytes = container
            .tensor_bytes(&name)
            .map_err(|_| format!("{label}: the container holds no tensor named {name}"))?;
        let w: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // The shape is the weight's own: [Cout, Cin, kh, kw]. The first
        // dimension is the output channels and the rest multiply out to the
        // element count.
        let entry = container
            .directory()
            .tensor(&name)
            .ok_or_else(|| format!("{label}: the container holds no tensor named {name}"))?;
        let shape = &entry.logical_shape;
        if shape.len() != 4 {
            return Err(format!(
                "{label}: {name} is {} dimensions, not 4 ([Cout, Cin, kh, kw]).",
                shape.len()
            ));
        }
        let (cout, cin, kh, kw) = (shape[0], shape[1], shape[2], shape[3]);
        if (cout * cin * kh * kw) as usize != w.len() {
            return Err(format!(
                "{label}: {name} holds {} values and its shape says {}.",
                w.len(),
                cout * cin * kh * kw
            ));
        }
        let mut b: Vec<f32> = Vec::new();
        if !bias_free.contains(&(i as i64)) {
            let bname = format!("conv.{i}.b");
            let bytes = container.tensor_bytes(&bname).map_err(|_| {
                format!("{label}: the container holds no tensor named {bname}")
            })?;
            b = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            if b.len() != cout as usize {
                return Err(format!(
                    "{label}: {bname} holds {} values and the convolution has \
                     {cout} output channels.",
                    b.len()
                ));
            }
        }
        convs.push(ConvW {
            index: i as i32,
            cout,
            cin,
            kh,
            kw,
            w,
            b,
        });
    }

    // The DFL's projection: its convolution must exist and must be a 1x1 over
    // the bins, because the identity above needs exactly that.
    if head.box_format == BoxFormat::LtrbDfl {
        if head.dfl_conv < 0 || head.dfl_conv as usize >= convs.len() {
            return Err(format!(
                "{label}: the DFL's projection is convolution {}, and this \
                 container holds {} convolutions.",
                head.dfl_conv,
                convs.len()
            ));
        }
        let dfl = &convs[head.dfl_conv as usize];
        if dfl.cout != 1 || dfl.kin() != dfl_bins {
            return Err(format!(
                "{label}: the DFL's projection is [{}, {}, {}, {}] and the \
                 identity the head folds it into needs a [1, {dfl_bins}, 1, 1] \
                 weight.",
                dfl.cout, dfl.cin, dfl.kh, dfl.kw
            ));
        }
        head.dfl_bias = !dfl.b.is_empty();
    }

    let detect_cout = get_i64("detect_cout")?;
    let geometry = Geometry {
        input_size,
        num_keypoints,
        num_classes,
        dfl_bins,
        strides,
        grid,
        mean,
        std_dev,
        graph,
        convs,
        detect_cout,
        head,
    };
    if geometry.detect_cout != geometry.out_channels() {
        return Err(format!(
            "{label}: the container's detect_cout is {} and this head produces {}. \
             The four are not optional: the decode reads row[0..3] for the box, \
             so a tensor without them would be read as if the first keypoint \
             were the box.",
            geometry.detect_cout,
            geometry.out_channels()
        ));
    }
    Ok(geometry)
}

impl ConvW {
    /// The reduction width: the input channels times the kernel's area.
    pub fn kin(&self) -> i64 {
        self.cin * self.kh * self.kw
    }
}

/// The pose family's dense convolutions, as the array's list.
///
/// THE LIST IS THE FAMILY'S OWN, and it is built from the GRAPH rather than
/// from the container's directory: a panel exists for a convolution the
/// graph RUNS, and a panel that no op calls is a panel nobody stages.
///
/// A convolution with no stream is left out, and that is what a container
/// packed WITHOUT `--npu` looks like: the packer writes the stream into the
/// node when it stages the panel, so a container without panels names no
/// stream at all. The refusal for that container is the CLI's, and it names
/// the packing flag.
pub fn array_specs(geometry: &Geometry) -> Vec<ConvSpec> {
    let mut specs = Vec::new();
    for l in &geometry.graph {
        if l.op != Op::Conv || l.stream.is_empty() {
            continue;
        }
        let Some(cw) = geometry.convs.get(l.conv as usize) else {
            continue;
        };
        if cw.w.is_empty() {
            continue;
        }
        specs.push(ConvSpec {
            weight: format!("conv.{}.w", cw.index),
            stream: l.stream.clone(),
            k: cw.kin() as usize,
            n: cw.cout as usize,
            bias: if cw.b.is_empty() { None } else { Some(cw.b.clone()) },
        });
    }
    specs
}

/// The pose network: the graph walk and the host GEMM.
/// Where the host convolutions' time went, accumulated over one run.
///
/// The oracle prints this split under every image, and it is the number that
/// says whether the next change should go into the im2col, the GEMM or the
/// transpose. It is measured, not estimated: a stage that is reported with no
/// timer is a stage nobody can budget against.
#[derive(Debug, Clone, Copy, Default)]
pub struct NetTimings {
    /// Seconds in im2col.
    pub im2col_s: f64,
    /// Seconds in the GEMM.
    pub gemm_s: f64,
    /// Seconds in the `[M, N]` to NCHW transpose.
    pub transpose_s: f64,
    /// Seconds applying the SiLU to the GEMM's output.
    pub elementwise_s: f64,
    /// Seconds transposing the WEIGHTS. Zero after the first image: the
    /// transpose happens once, at load.
    pub weight_s: f64,
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

pub struct Network {
    geometry: Geometry,
    /// The weights, transposed to `[K, N]` once, at load.
    wmat: Vec<f32>,
    /// Where each convolution's matrix starts.
    wmat_at: Vec<usize>,
    /// How many threads the GEMM's rows are split over.
    threads: usize,
}

impl Network {
    /// Build the network out of a container.
    pub fn open(path: &Path) -> Result<Network, String> {
        let container = Container::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let label = path.display().to_string();
        let geometry = read_geometry(&container, &label)?;
        // The weight transposes happen HERE, once, and not per frame: the
        // filter is the same on every frame.
        let mut wmat = Vec::new();
        let mut wmat_at = Vec::with_capacity(geometry.convs.len());
        for conv in &geometry.convs {
            wmat_at.push(wmat.len());
            let (cout, k) = (conv.cout as usize, conv.kin() as usize);
            let base = wmat.len();
            wmat.resize(base + cout * k, 0.0);
            // dst[k*N + n] = src[n*K + k]: the container holds [N, K] and the
            // GEMM wants [K, N].
            for n in 0..cout {
                for kk in 0..k {
                    wmat[base + kk * cout + n] = conv.w[n * k + kk];
                }
            }
        }
        Ok(Network {
            geometry,
            wmat,
            wmat_at,
            threads: default_threads(),
        })
    }

    /// How many threads the host convolutions are split over.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// The model's geometry.
    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    /// Walk the graph and return the canonical head tensor.
    ///
    /// The canonical tensor is `[4 + nc + nk*3, cells]`: xyxy boxes in
    /// letterbox pixels, the class score and each keypoint's visibility
    /// already as probabilities, keypoints in letterbox pixels.
    pub fn run(&self, input: &Tensor) -> Result<Tensor, String> {
        self.run_with_hook(input, &mut |_, _| {})
    }

    /// The same walk, with the time each stage of the host convolutions took.
    ///
    /// Per RUN and not per network: the numbers belong to the image that was
    /// just processed, and a field on the network would be shared by two
    /// requests that have nothing to do with each other.
    pub fn run_timed(&self, input: &Tensor) -> Result<(Tensor, NetTimings), String> {
        let mut timings = NetTimings::default();
        let head = self.walk(input, &mut |_, _| {}, &mut timings, None)?;
        Ok((head, timings))
    }

    /// The same walk, on the array's convolutions.
    ///
    /// THE ARRAY, OR THE HOST -- and the flag that chose is not read here:
    /// the caller passes the array's backend when it wants the convolutions
    /// on the board, and passes nothing when it does not. A run that asked
    /// for the array and dispatched nothing is a run that lied, and the way
    /// to keep it from lying is that the array's backend is the only thing
    /// that can dispatch at all.
    pub fn run_timed_on_array(
        &self,
        input: &Tensor,
        array: ArrayRef<'_, '_>,
    ) -> Result<(Tensor, NetTimings), String> {
        let mut timings = NetTimings::default();
        let head = self.walk(input, &mut |_, _| {}, &mut timings, array)?;
        Ok((head, timings))
    }

    /// The same walk, with a hook that sees every node's output.
    ///
    /// The hook is what localises a disagreement: end to end, a wrong first
    /// convolution, a wrong strided stem and a wrong DFL fold are all the
    /// same sentence -- "the detections are wrong" -- and the node table says
    /// which, on the first run.
    pub fn run_with_hook(
        &self,
        input: &Tensor,
        hook: &mut dyn FnMut(usize, &Tensor),
    ) -> Result<Tensor, String> {
        self.walk(input, hook, &mut NetTimings::default(), None)
    }

    /// The same walk, on the array's convolutions, with a hook.
    ///
    /// THE DUMP MUST COME FROM THE RUN THAT ANSWERED. A dump written by the
    /// host path while the answer came from the array would compare two
    /// different networks and call the difference the array's.
    pub fn run_with_hook_on_array(
        &self,
        input: &Tensor,
        hook: &mut dyn FnMut(usize, &Tensor),
        array: ArrayRef<'_, '_>,
    ) -> Result<Tensor, String> {
        self.walk(input, hook, &mut NetTimings::default(), array)
    }

    /// The walk itself: the graph in order, the hook on every node, and the
    /// time each stage of each convolution took.
    fn walk(
        &self,
        input: &Tensor,
        hook: &mut dyn FnMut(usize, &Tensor),
        timings: &mut NetTimings,
        mut array: ArrayRef<'_, '_>,
    ) -> Result<Tensor, String> {
        let g = &self.geometry;
        if input.c != 3 {
            return Err(format!(
                "pose net: the input has {} channels; this front end produces 3 (RGB)",
                input.c
            ));
        }
        if input.h != g.input_size || input.w != g.input_size {
            return Err(format!(
                "pose net: the input is {}x{} and the model's input_size is {}",
                input.h, input.w, g.input_size
            ));
        }
        let mut vals: Vec<Tensor> = vec![Tensor::default(); g.graph.len()];
        // Operand -1 is the image. vals[0] is node 0's OUTPUT, so the image
        // cannot live there.
        // The photograph is node -1 in the dump, and it is compared like any
        // other node: a front-end difference would show up there rather than
        // hide inside the network.
        hook(usize::MAX, input);
        for (i, layer) in g.graph.iter().enumerate() {
            let out = match layer.op {
                Op::Conv => {
                    let input_tensor = operand(&vals, input, layer.inputs[0]);
                    self.conv(
                        &input_tensor,
                        &g.convs[layer.conv as usize],
                        layer.silu,
                        layer.stride,
                        layer.pad_h,
                        layer.pad_w,
                        timings,
                        array.as_deref_mut(),
                    )?
                }
                Op::Concat => {
                    let ins: Vec<&Tensor> = layer
                        .inputs
                        .iter()
                        .map(|idx| operand(&vals, input, *idx))
                        .collect();
                    concat(&ins)?
                }
                Op::Slice => {
                    let t = operand(&vals, input, layer.inputs[0]);
                    slice(&t, layer.chan, layer.nch)?
                }
                Op::Add => {
                    let a = operand(&vals, input, layer.inputs[0]);
                    let b = operand(&vals, input, layer.inputs[1]);
                    add(&a, &b)?
                }
                Op::MaxPool => {
                    let t = operand(&vals, input, layer.inputs[0]);
                    maxpool(&t, layer.k)?
                }
                Op::Upsample => {
                    let t = operand(&vals, input, layer.inputs[0]);
                    upsample(&t, layer.scale)?
                }
                Op::Detect => {
                    let ins: Vec<&Tensor> = layer
                        .inputs
                        .iter()
                        .map(|idx| operand(&vals, input, *idx))
                        .collect();
                    self.head(&ins)?
                }
            };
            vals[i] = out;
            hook(i, &vals[i]);
        }
        let head = vals
            .last()
            .ok_or_else(|| "pose net: the graph produced no head tensor".to_string())?;
        Ok(head.clone())
    }

    /// One convolution: im2col, the GEMM, and the transpose into NCHW.
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
    fn conv(
        &self,
        input: &Tensor,
        w: &ConvW,
        silu: bool,
        stride: i64,
        pad_h: i64,
        pad_w: i64,
        timings: &mut NetTimings,
        mut array: ArrayRef<'_, '_>,
    ) -> Result<Tensor, String> {
        if input.c != w.cin {
            return Err(format!(
                "pose conv: input {} channels, weight wants {}",
                input.c, w.cin
            ));
        }
        let k = w.kin();
        let oh = (input.h + 2 * pad_h - w.kh) / stride + 1;
        let ow = (input.w + 2 * pad_w - w.kw) / stride + 1;
        let m = oh * ow;
        let n = w.cout;
        if m <= 0 {
            return Err(format!(
                "pose conv: a {}x{} input padded by {}/{} is smaller than the \
                 {}x{} window, so the output extent would be negative",
                input.h, input.w, pad_h, pad_w, w.kh, w.kw
            ));
        }
        let t = std::time::Instant::now();
        let a = im2col(input, w.kh, w.kw, pad_h, pad_w, stride, stride)?;
        let im2col_s = t.elapsed().as_secs_f64();
        let mut c = vec![0.0f32; (m * n) as usize];

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
        let wmat = &self.wmat[self.wmat_at[w.index as usize]..];
        let gemm_s = match array.as_deref_mut() {
            Some(array) => {
                let name = format!("conv.{}.w", w.index);
                let before_dispatch = array.dispatches() as i64;
                let before_convert = array.convert_seconds();
                let t = std::time::Instant::now();
                array
                    .conv(&name, &a, m as usize, k as usize, n as usize, &mut c)
                    .map_err(|e| format!("pose conv {name}: {e}"))?;
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
                gemm_rows(
                    &a,
                    m as usize,
                    wmat,
                    k as usize,
                    n as usize,
                    &w.b,
                    &mut c,
                    self.threads,
                );
                t.elapsed().as_secs_f64()
            }
        };
        let mut out = Tensor {
            c: n,
            h: oh,
            w: ow,
            d: vec![0.0f32; (m * n) as usize],
        };
        let t = std::time::Instant::now();
        transpose_mn_to_nchw(&c, m, n, &mut out.d, false);
        let transpose_s = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        if silu {
            for v in out.d.iter_mut() {
                *v = *v / (1.0 + (-*v).exp());
            }
        }
        let elementwise_s = if silu { t.elapsed().as_secs_f64() } else { 0.0 };
        timings.im2col_s += im2col_s;
        timings.transpose_s += transpose_s;
        timings.elementwise_s += elementwise_s;
        if convs_array > 0 {
            // The array's span INCLUDES the im2col, because the oracle's does
            // -- its timer starts before the im2col -- and the two numbers
            // are only comparable if they span the same work.
            timings.t_array_s += im2col_s + array_s + transpose_s;
            timings.t_array_gemm_s += gemm_s;
            timings.t_array_repack_s += repack_s;
            timings.t_array_transpose_s += transpose_s;
            timings.convs_array += convs_array;
            timings.dispatches += dispatched;
        } else {
            timings.gemm_s += gemm_s;
        }
        Ok(out)
    }

    /// The head: the per-level outputs into the ONE canonical tensor.
    fn head(&self, inputs: &[&Tensor]) -> Result<Tensor, String> {
        let g = &self.geometry;
        let lv = g.n_levels();
        if inputs.len() != 2 * lv {
            return Err(format!(
                "pose head: the detect node lists {} inputs and the container \
                 declares {lv} levels.",
                inputs.len()
            ));
        }
        let bins = g.dfl_bins;
        let box_ch = if g.head.box_format == BoxFormat::LtrbDfl {
            4 * bins
        } else {
            4
        };
        let cls_ch = g.num_classes;
        let kpt_ch = g.num_keypoints * 3;
        let c = g.out_channels();
        let mut cells = 0i64;
        for gg in &g.grid {
            cells += gg * gg;
        }
        let mut rows = vec![0.0f32; (cells * c) as usize];
        let mut offset = 0i64;
        for l in 0..lv {
            let bc = inputs[2 * l];
            let kp = inputs[2 * l + 1];
            let s = g.grid[l];
            let stride = g.strides[l] as f32;
            if bc.c != box_ch + cls_ch {
                return Err(format!(
                    "pose head: level {l} feeds the head a {}-channel box+class \
                     tensor; the format needs {} channels.",
                    bc.c,
                    box_ch + cls_ch
                ));
            }
            if kp.c != kpt_ch {
                return Err(format!(
                    "pose head: level {l} feeds the head a {}-channel keypoint \
                     tensor, and {} keypoints x 3 is {kpt_ch}.",
                    kp.c, g.num_keypoints
                ));
            }
            if bc.h != s || bc.w != s || kp.h != s || kp.w != s {
                return Err(format!(
                    "pose head: level {l} declares grid {s} but the head's tensors \
                     are {}x{} and {}x{}. The head maps cell (i, j) to pixel \
                     (i*s, j*s), so the grid and the tensor have to be the same \
                     grid or every keypoint belongs to a neighbouring cell.",
                    bc.h, bc.w, kp.h, kp.w
                ));
            }
            let plane = (s * s) as usize;
            // The box and the keypoint tensors are read CHANNEL FIRST: the
            // 16 bins of a corner are 16 different planes, and the corner q's
            // bin b is at element (q*bins + b) * plane + cell. So these two
            // are the whole tensors and not one channel's plane -- the class
            // score IS one plane, and it is read as one.
            let box_data = &bc.d[..];
            let cls_data = bc.chw(box_ch);
            let kpc = &kp.d[..];
            // corner q, bin b -> its element in the channels-first tensor.
            let bin = |q: i64, b: i64, cell: usize| -> f32 {
                box_data[((q * bins + b) as usize) * plane + cell]
            };
            for i in 0..s {
                for j in 0..s {
                    let cell = (i * s + j) as usize;
                    let row = &mut rows[((offset + cell as i64) * c) as usize..][..c as usize];
                    // -- the box.
                    if g.head.box_format == BoxFormat::LtrbDfl {
                        let dfl_w = &g.convs[g.head.dfl_conv as usize].w;
                        let mut d = [0.0f64; 4];
                        for q in 0..4i64 {
                            // The max is subtracted before the exp. It cancels
                            // in the ratio, so this is not an approximation --
                            // it is the difference between a correct answer and
                            // an inf. A DFL logit of 40 is not unusual on a
                            // confident box.
                            let mut mx = bin(q, 0, cell);
                            for b in 1..bins {
                                mx = mx.max(bin(q, b, cell));
                            }
                            let mut num = 0.0f64;
                            let mut den = 0.0f64;
                            for b in 0..bins {
                                let e = (bin(q, b, cell) - mx) as f64;
                                num += e.exp() * dfl_w[b as usize] as f64;
                                den += e.exp();
                            }
                            d[q as usize] = num / den;
                        }
                        let gx = j as f64 + g.head.box_grid_offset as f64;
                        let gy = i as f64 + g.head.box_grid_offset as f64;
                        row[0] = ((gx - d[0]) * stride as f64) as f32;
                        row[1] = ((gy - d[1]) * stride as f64) as f32;
                        row[2] = ((gx + d[2]) * stride as f64) as f32;
                        row[3] = ((gy + d[3]) * stride as f64) as f32;
                    } else {
                        let corner = |q: i64| -> f32 { box_data[(q as usize) * plane + cell] };
                        if g.head.box_format == BoxFormat::Xyxy {
                            row[0] = corner(0);
                            row[1] = corner(1);
                            row[2] = corner(2);
                            row[3] = corner(3);
                        } else {
                            let hw = corner(2) as f64 * 0.5;
                            let hh = corner(3) as f64 * 0.5;
                            row[0] = (corner(0) as f64 - hw) as f32;
                            row[1] = (corner(1) as f64 - hh) as f32;
                            row[2] = (corner(0) as f64 + hw) as f32;
                            row[3] = (corner(1) as f64 + hh) as f32;
                        }
                    }
                    // -- the class score, at index 4: after the four box values.
                    let mut sc = cls_data[cell];
                    if g.head.score == ScoreKind::Logit {
                        sc = 1.0 / (1.0 + (-sc).exp());
                    }
                    row[4] = sc;
                    // -- the keypoints: (anchor + 2 * offset) * stride, and the
                    // 2 is the exporter's -- the keypoint branch predicts
                    // offsets twice the size the box branch's are.
                    for k in 0..g.num_keypoints {
                        let o = 4 + g.num_classes + k * 3;
                        let base = (k * 3) as usize * plane + cell;
                        row[o as usize] = (j as f32 + g.head.kpt_grid_offset
                            + 2.0 * kpc[base])
                            * stride;
                        row[o as usize + 1] = (i as f32 + g.head.kpt_grid_offset
                            + 2.0 * kpc[base + plane])
                            * stride;
                        let mut v = kpc[base + 2 * plane];
                        if g.head.kpt_visibility == ScoreKind::Logit {
                            v = 1.0 / (1.0 + (-v).exp());
                        }
                        row[o as usize + 2] = v;
                    }
                }
            }
            offset += s * s;
        }
        // The rows were built in the order the head produces them -- [cell][C]
        // -- and the canonical tensor is [C][cells].
        let mut out = Tensor {
            c,
            h: cells,
            w: 1,
            d: vec![0.0f32; (cells * c) as usize],
        };
        transpose_mn_to_nchw(&rows, cells, c, &mut out.d, false);
        Ok(out)
    }
}

/// One operand: -1 is the photograph.
fn operand<'a>(vals: &'a [Tensor], input: &'a Tensor, idx: i32) -> &'a Tensor {
    if idx == -1 {
        input
    } else {
        &vals[idx as usize]
    }
}

/// im2col for a stride/padding generalisation.
///
/// The k index is `(ci*kh + i)*kw + j` and the row index is `y*W + x`, which
/// is the order the weights are read in too, so the GEMM's K reduction walks
/// the kernel exactly as the checkpoint stores it.
///
/// Zero outside the window, and not a clamp: a clamped edge replicates the
/// border pixel, which is a different network.
pub fn im2col(
    input: &Tensor,
    kh: i64,
    kw: i64,
    ph: i64,
    pw: i64,
    sh: i64,
    sw: i64,
) -> Result<Vec<f32>, String> {
    let (c, h, w) = (input.c, input.h, input.w);
    let oh = (h + 2 * ph - kh) / sh + 1;
    let ow = (w + 2 * pw - kw) / sw + 1;
    let m = oh * ow;
    let k = c * kh * kw;
    if m <= 0 || k <= 0 {
        return Ok(Vec::new());
    }
    // The zero border is written ONCE per tensor rather than tested per
    // element -- but only where there IS a border to write: with no padding
    // the gather can never fall outside the image.
    let mut a = if ph == 0 && pw == 0 {
        vec![0.0f32; (m * k) as usize]
    } else {
        vec![0.0f32; (m * k) as usize]
    };
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
    Ok(a)
}

/// The `[M, N]` to NCHW transpose, with the activation fused into it.
///
/// The copy is TILED 32 rows at a time: a naive n-outer / m-inner copy
/// writes M runs of one float each for every channel and streams the
/// destination as N simultaneous write pointers; tiling makes each
/// destination run 32 floats long.
///
/// The activation is fused HERE, into the pass that writes the tensor: the
/// bias is already in the value, so this is the same number a separate pass
/// would read.
pub fn transpose_mn_to_nchw(src: &[f32], m: i64, n: i64, dst: &mut [f32], silu: bool) {
    const TILE: i64 = 32;
    let mut m0 = 0i64;
    while m0 < m {
        let m1 = (m0 + TILE).min(m);
        for ch in 0..n {
            let d = &mut dst[(ch * m) as usize..][..m as usize];
            for mm in m0..m1 {
                let v = src[(mm * n + ch) as usize];
                d[mm as usize] = if silu {
                    v / (1.0 + (-v).exp())
                } else {
                    v
                };
            }
        }
        m0 = m1;
    }
}

/// How many threads the host convolutions are split over.
///
/// The machine's own count, asked rather than assumed: the status block
/// prints this number, and a printed "16 threads" on a machine with four is
/// a claim about a run that did not happen.
fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// The GEMM over ROW RANGES, one per thread.
///
/// Each output row is one im2col row times the weight matrix, and a row's
/// accumulation is over `k` in order -- so splitting ROWS changes which
/// thread computes a value and never the value itself. That is what makes
/// the threads free: the split is bit-exact by construction, and a
/// comparison against the oracle does not have to know how many cores the
/// machine has.
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
    if threads <= 1 {
        crate::kernels::gemm_nt(a, m, b, k, n, bias, out);
        return;
    }
    let chunk = m.div_ceil(threads);
    let parts: Vec<&mut [f32]> = out.chunks_mut(chunk * n).collect();
    std::thread::scope(|s| {
        for (ci, part) in parts.into_iter().enumerate() {
            let rows = part.len() / n;
            let a_part = &a[ci * chunk * k..(ci * chunk + rows) * k];
            s.spawn(move || crate::kernels::gemm_nt(a_part, rows, b, k, n, bias, part));
        }
    });
}

/// A channel-wise join.
///
/// Per CHANNEL, and that is not only for the threads: copying one plane per
/// OPERAND wrote 16 channels' worth of offsets but only a quarter of each
/// one's data.
fn concat(inputs: &[&Tensor]) -> Result<Tensor, String> {
    if inputs.len() < 2 {
        return Err(format!(
            "pose concat: {} operand(s). A join needs at least two, and a \
             one-operand join would silently be a copy, and a copy reads as a \
             plausible layer.",
            inputs.len()
        ));
    }
    let mut c = 0i64;
    for (i, t) in inputs.iter().enumerate() {
        if i > 0 && (t.h != inputs[0].h || t.w != inputs[0].w) {
            return Err(format!(
                "pose concat: operand 0 is {}x{} and operand {i} is {}x{}. The \
                 join is channel-wise, so every operand has to be the same size.",
                inputs[0].h, inputs[0].w, t.h, t.w
            ));
        }
        c += t.c;
    }
    let plane = (inputs[0].h * inputs[0].w) as usize;
    let mut out = Tensor {
        c,
        h: inputs[0].h,
        w: inputs[0].w,
        d: vec![0.0f32; (c as usize) * plane],
    };
    let mut oc = 0usize;
    for t in inputs {
        for k in 0..t.c {
            let dst = &mut out.d[(oc * plane)..][..plane];
            dst.copy_from_slice(&t.chw(k)[..plane]);
            oc += 1;
        }
    }
    Ok(out)
}

/// A contiguous run of channels.
fn slice(input: &Tensor, chan: i64, nch: i64) -> Result<Tensor, String> {
    if nch <= 0 || chan < 0 || chan + nch > input.c {
        return Err(format!(
            "pose slice: channels [{chan}, {}) out of a {}-channel tensor. A \
             split that is not a contiguous run of channels is a different op \
             and is not approximated by clamping.",
            chan + nch,
            input.c
        ));
    }
    let plane = (input.h * input.w) as usize;
    let mut out = Tensor {
        c: nch,
        h: input.h,
        w: input.w,
        d: vec![0.0f32; (nch as usize) * plane],
    };
    for c in 0..nch {
        let dst = &mut out.d[(c as usize) * plane..][..plane];
        dst.copy_from_slice(&input.chw(chan + c)[..plane]);
    }
    Ok(out)
}

/// The residual.
fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.c != b.c || a.h != b.h || a.w != b.w {
        return Err("pose add: shapes differ".to_string());
    }
    let mut out = Tensor {
        c: a.c,
        h: a.h,
        w: a.w,
        d: vec![0.0f32; a.d.len()],
    };
    for i in 0..a.d.len() {
        out.d[i] = a.d[i] + b.d[i];
    }
    Ok(out)
}

/// The pooling.
///
/// The maximum is initialised to -inf rather than to 0, because a
/// pad-averaged zero would let the padding raise a maximum.
fn maxpool(input: &Tensor, k: i64) -> Result<Tensor, String> {
    let pad = k / 2;
    let mut out = Tensor {
        c: input.c,
        h: input.h,
        w: input.w,
        d: vec![0.0f32; input.d.len()],
    };
    for ci in 0..input.c {
        let src = input.chw(ci);
        let plane = (input.h * input.w) as usize;
        let dst = &mut out.d[(ci as usize) * plane..][..plane];
        for y in 0..input.h {
            for x in 0..input.w {
                let mut best = f32::NEG_INFINITY;
                for i in 0..k {
                    let sy = y + i - pad;
                    if sy < 0 || sy >= input.h {
                        continue;
                    }
                    for j in 0..k {
                        let sx = x + j - pad;
                        if sx < 0 || sx >= input.w {
                            continue;
                        }
                        best = best.max(src[(sy * input.w + sx) as usize]);
                    }
                }
                dst[(y * input.w + x) as usize] = best;
            }
        }
    }
    Ok(out)
}

/// Nearest-neighbour upsampling.
fn upsample(input: &Tensor, scale: i64) -> Result<Tensor, String> {
    let mut out = Tensor {
        c: input.c,
        h: input.h * scale,
        w: input.w * scale,
        d: vec![0.0f32; (input.c * input.h * input.w * scale * scale) as usize],
    };
    let plane = (input.h * input.w) as usize;
    for ci in 0..input.c {
        let src = input.chw(ci);
        let dst = &mut out.d[(ci as usize) * plane * (scale * scale) as usize..];
        for y in 0..input.h {
            for x in 0..input.w {
                let v = src[(y * input.w + x) as usize];
                for dy in 0..scale {
                    let row = &mut dst[((y * scale + dy) * out.w) as usize..][..out.w as usize];
                    for dx in 0..scale {
                        row[(x * scale + dx) as usize] = v;
                    }
                }
            }
        }
    }
    Ok(out)
}
