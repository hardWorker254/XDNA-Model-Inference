//! The mppose family's status block, its per-image lines and its node dump.
//!
//! # The block is the C++'s, row for row
//!
//! `npuimage mppose` printed a block before every run and the golden transcript
//! holds it, so the block is a contract and not a greeting. Every row says
//! WHERE an operation runs and, where it is not obvious, WHY it runs there.
//!
//! # Two stages, two rows, and the split is the point
//!
//! The person detector and the landmark network are reported SEPARATELY,
//! including how many of each one's convolutions are depthwise, because that
//! split is the interesting number on this architecture: 28 of the detector's
//! 73 convolutions and 34 of the landmark network's 88 are depthwise, and no
//! array can take those. A single "161 convolutions" line would hide exactly the
//! fact that decides whether an array design set is worth building.
//!
//! # `d2s` gets its own row
//!
//! It is this architecture's own op and the one reason the graph walk is not
//! hands': three DepthToSpace nodes build the detector's 28/14/7 pyramid out of
//! one 7x7 map, so a reader looking for where that happens should find it named.
//!
//! # The numbers in the block are this container's
//!
//! Every count is read from the geometry: the sides, the anchors, the pyramid's
//! levels, the node counts, the depthwise split. Nothing here is a constant that
//! happens to match.
//!
//! # The dump is the C++'s format, byte for byte
//!
//! 64 KiB of header, then every node's fp32 NCHW in graph order, for BOTH
//! networks, with the header rewritten once the node count is known. The reader
//! looks for `"i": -1` as the network's INPUT, which is the photograph after the
//! front end: a front-end difference shows up there as a difference in 150k
//! pixels rather than as an amplified one 90 nodes later.

use npum_cpu::mppose::{Geometry, MpPose};
use npum_cpu::mppose_decode::Run;
use npum_cpu::pose::Tensor;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

/// The 64 KiB the dump reserves for its header.
const DUMP_HEADER: usize = 1 << 16;

/// The block, printed to stderr before the first image.
pub fn print_block(name: &str, g: &Geometry, setup_s: f64) {
    eprintln!("NpuEmbeddings -- mppose (MediaPipe person det + pose landmarks) on the CPU");
    eprintln!(
        "  model      {}: detector {}px, landmarks {}px, {} anchors over {} levels, {} rows \
         of {} (of which {} are keypoints)",
        name,
        g.det_input_size,
        g.pose_input_size,
        g.det_num_anchors,
        g.det_levels.len(),
        g.num_landmarks,
        g.lm_cols,
        g.num_keypoints
    );
    eprintln!(
        "  rotation   between the detector's keypoints {} (mid-hip) and {} (full body), both \
         read from the container; the RoI is the square on the hip whose half-side is that \
         distance",
        g.person_lm_mid_hip, g.person_lm_full_body
    );
    eprintln!(
        "  thresholds score {:.3}, nms IoU {:.3}, top_k {}, pose conf {:.3}, box enlarge \
         {:.2} then {:.2} -- from the container, not a flag",
        g.score_threshold,
        g.nms_threshold,
        g.top_k,
        g.pose_conf_threshold,
        g.person_box_pre_enlarge,
        g.person_box_enlarge
    );
    eprintln!(
        "  normalise  detector (x/255 - {:.1})/{:.1}, border 0.0 in TENSOR space; landmarks \
         (x/255 - {:.1})/{:.1}, border 0.0 in the raster",
        g.det_mean.first().copied().unwrap_or(0.0),
        g.det_std.first().copied().unwrap_or(1.0),
        g.pose_mean.first().copied().unwrap_or(0.0),
        g.pose_std.first().copied().unwrap_or(1.0)
    );
    eprintln!("  ops        (npu = dispatched, host = this process)");
    eprintln!(
        "             {:<26} {:<5} {}",
        format!(
            "{} x conv + {} x dwconv",
            g.det_dense(),
            g.det_depthwise()
        ),
        "host",
        format!(
            "person detector, {} nodes; dwconv has no [M,N] GEMM to dispatch, so it would \
             not move even with a design set",
            g.det_graph.len()
        )
    );
    eprintln!(
        "             {:<26} {:<5} {}",
        format!(
            "{} x conv + {} x dwconv",
            g.pose_dense(),
            g.pose_depthwise()
        ),
        "host",
        format!(
            "landmark network, {} nodes, once per person",
            g.pose_graph.len()
        )
    );
    eprintln!(
        "             {:<26} {:<5} {}",
        "add, maxpool, resize", "host", "elementwise and memory passes"
    );
    eprintln!(
        "             {:<26} {:<5} {}",
        "d2s (DepthToSpace)",
        "host",
        "this architecture's own op: three of them build the detector's 28/14/7 pyramid out \
         of one 7x7 map, DCR order"
    );
    eprintln!(
        "             {:<26} {:<5} {}",
        "image front end",
        "host",
        format!(
            "bilinear letterbox to {}px, normalise in float and pad with 0 in TENSOR space; \
             per person a rotated INTER_AREA crop to {}px",
            g.det_input_size, g.pose_input_size
        )
    );
    eprintln!(
        "             {:<26} {:<5} {}",
        "threshold, nms, inverse letterbox",
        "host",
        "host-side by construction: a list of survivors, not a tensor"
    );
    eprintln!(
        "  setup      {:.3} s (both graphs read, weights transposed once)",
        setup_s
    );
}

/// The `--mppose-dump` line, once the path is known.
pub fn print_dump_header(path: &Path, g: &Geometry) {
    eprintln!(
        "  dump       {} + {} graph nodes per image -> {} (fp32 NCHW, JSON header in the \
         first 64 KiB)",
        g.det_graph.len(),
        g.pose_graph.len(),
        path.display()
    );
}

/// One image's line, on stderr.
pub fn print_run(image: &str, r: &Run) {
    let nd = r.detections.len();
    let np = r.people.len();
    eprintln!(
        "  {:<40} {}x{}  {} detection{} -> {} pose{}  {:.1} ms (front {:.1} / det {:.1} / nms \
         {:.1} / crop {:.1} / pose {:.1} / post {:.1})",
        image,
        r.width,
        r.height,
        nd,
        if nd == 1 { "" } else { "s" },
        np,
        if np == 1 { "" } else { "s" },
        r.total_s * 1e3,
        r.front_end_s * 1e3,
        r.det_s * 1e3,
        r.nms_s * 1e3,
        r.crop_s * 1e3,
        r.pose_s * 1e3,
        r.post_s * 1e3
    );
}

/// The second stage's geometry, on stderr, before the JSON object.
///
/// Not part of a person's answer -- the landmarks are the answer -- but it is
/// the one intermediate whose being wrong produces plausible landmarks: an angle
/// off by a degree, or a pad_bias off by the square's own padding, moves every
/// joint by a fraction of its distance from the hip and nothing else in the
/// output says so.
pub fn print_crop(r: &Run) {
    eprintln!(
        "             crop: angle {:.4} deg, square {:.0}x{:.0}, pad_bias {:.1} {:.1}, crop \
         mean {:.4}",
        r.angle_deg, r.rot_bbox[2], r.rot_bbox[3], r.crop_pad_bias[0], r.crop_pad_bias[1],
        r.crop_mean
    );
}

/// The node-dump writer, installed for the lifetime of the process.
///
/// One hook on each network, and it stays installed: it is a call per node, 213
/// of them, which is a rounding error against a frame, and re-installing it per
/// image would be a per-frame allocation to avoid.
pub struct Dump {
    file: std::fs::File,
    index: Vec<String>,
    /// Which image the records belong to, counting from zero.
    pub image: usize,
}

impl Dump {
    /// Open the dump file and reserve the header.
    pub fn create(path: &Path) -> Result<Dump, String> {
        let mut file = std::fs::File::create(path).map_err(|_| {
            format!(
                "--mppose-dump {}: cannot open it for writing",
                path.display()
            )
        })?;
        file.write_all(&vec![b' '; DUMP_HEADER])
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Dump {
            file,
            index: Vec::new(),
            image: 0,
        })
    }

    /// The hook both networks call: one node's output, in graph order.
    pub fn record(&mut self, net: &str, node: usize, t: &Tensor) {
        let off = self.file.stream_position().unwrap_or(0);
        // SIZE_MAX is the graph-input marker and prints as the -1 the reader
        // looks for.
        let node = if node == usize::MAX {
            -1i64
        } else {
            node as i64
        };
        self.index.push(format!(
            "    {{\"image\": {}, \"net\": \"{}\", \"i\": {}, \"c\": {}, \"h\": {}, \"w\": {}, \
             \"n\": {}, \"off\": {}}}",
            self.image,
            net,
            node,
            t.c,
            t.h,
            t.w,
            t.d.len(),
            off
        ));
        let mut bytes = Vec::with_capacity(t.d.len() * 4);
        for v in &t.d {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let _ = self.file.write_all(&bytes);
    }

    /// Rewrite the reserved header and close.
    ///
    /// A header that does not fit is a REFUSAL, not a truncation: a short header
    /// reads as a short graph, and a graph that quietly lost its last sixty nodes
    /// is the failure this dump exists to catch.
    pub fn finish(mut self, path: &Path) -> Result<usize, String> {
        let mut header = String::from("{\n  \"nodes\": [\n");
        for (i, line) in self.index.iter().enumerate() {
            if i > 0 {
                header.push_str(",\n");
            }
            header.push_str(line);
        }
        header.push_str("\n  ]\n}\n");
        if header.len() > DUMP_HEADER {
            return Err(format!(
                "--mppose-dump {}: the header needs {} bytes for {} node records and the \
                 reserved area is {}. Raise DUMP_HEADER in cli/src/mppose.rs; do NOT let it \
                 overflow, because the reader would then take tensor bytes for header text.",
                path.display(),
                header.len(),
                self.index.len(),
                DUMP_HEADER
            ));
        }
        header.push_str(&" ".repeat(DUMP_HEADER - header.len()));
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        self.file
            .write_all(header.as_bytes())
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(self.index.len())
    }
}

/// Open a container and hand back the networks, for the CLI's block.
pub fn open(path: &Path) -> Result<MpPose, String> {
    MpPose::open(path)
}
