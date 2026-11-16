//! The hands family's status block, its per-image lines and its node dump.
//!
//! # The block is the C++'s, row for row
//!
//! `npuimage hands` printed a block before every run and the golden transcript
//! holds it, so the block is a contract and not a greeting. Every row says
//! WHERE an operation runs and, where it is not obvious, WHY it runs there.
//!
//! # Two stages, two rows, and the split is the point
//!
//! The palm detector and the landmark network are reported SEPARATELY,
//! including how many of each one's convolutions are depthwise, because that
//! split is the interesting number on this architecture: 23 of the palm's 53
//! convolutions and 16 of the landmark network's 47 are depthwise, and no array
//! can take those. A single "145 convolutions" line would hide exactly the fact
//! that decides whether an array design set is worth building.
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
//! pixels rather than as an amplified one 58 nodes later.

use npum_cpu::hands::{Geometry, Hands};
use npum_cpu::hands_decode::Run;
use npum_cpu::pose::Tensor;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

/// The 64 KiB the dump reserves for its header.
const DUMP_HEADER: usize = 1 << 16;

/// What the array says about itself, in the block.
///
/// The numbers are the DESIGN's: how many shapes it can compute, where it
/// came from, and how many bytes of panels the container carries.
pub struct ArrayInfo {
    /// How many streams the design set carries.
    pub streams: usize,
    /// The design set's directory, as the reader named it.
    ///
    /// It is EMPTY when the design came from the container: a container that
    /// carries its own set needs nothing beside it, and there is no path to
    /// print.
    pub designs: String,
    /// How many rows one dispatch computes.
    pub rows: usize,
    /// How many bytes of weight panels are staged on the board.
    pub staged_bytes: usize,
}

/// The block, printed to stderr before the first image.
pub fn print_block(name: &str, g: &Geometry, setup_s: f64, array: Option<&ArrayInfo>) {
    eprintln!("NpuEmbeddings -- hands (MediaPipe palm + landmark) on the CPU");
    eprintln!(
        "  model      {}: palm {}px, landmark {}px, {} palm anchors over {} levels, {} \
         landmarks",
        name,
        g.palm_input_size,
        g.lm_input_size,
        g.palm_num_anchors,
        g.palm_levels.len(),
        g.num_landmarks
    );
    eprintln!(
        "  rotation   from palm keypoints {} (wrist) and {} (middle base), both read \
         from the container",
        g.palm_lm_wrist, g.palm_lm_middle_base
    );
    eprintln!(
        "  thresholds score {:.3}, nms IoU {:.3}, top_k {}, crop enlarge {:.2} then \
         {:.2} -- from the container, not from a flag",
        g.score_threshold, g.nms_threshold, g.top_k, g.palm_pre.enlarge, g.palm_post.enlarge
    );
    eprintln!("  ops        (npu = dispatched, host = this process)");
    // WHERE THE CONVOLUTIONS RAN.
    //
    // The side is read from the SESSION and not from the flag: a row that
    // printed "npu" because --npu-ops conv was on the command line would say
    // so for a run where the design set was missing and everything fell
    // back, which is the one thing a status block must not do.
    match array {
        Some(info) => {
            eprintln!(
                "             {:<26} {:<5} {}",
                format!(
                    "{} x conv + {} x dwconv",
                    g.palm_dense(),
                    g.palm_depthwise()
                ),
                "npu",
                format!(
                    "palm detector, {} nodes, on the array's {} padded (K, N) slots; \
                     dwconv has no [M,N] GEMM to dispatch, so it stays here",
                    g.palm_graph.len(),
                    info.streams
                )
            );
            eprintln!(
                "             {:<26} {:<5} {}",
                format!("{} x conv + {} x dwconv", g.lm_dense(), g.lm_depthwise()),
                "npu",
                format!(
                    "landmark network, {} nodes, once per hand, on the same {} slots \
                     under a disjoint range",
                    g.lm_graph.len(),
                    info.streams
                )
            );
        }
        None => {
            eprintln!(
                "             {:<26} {:<5} {}",
                format!(
                    "{} x conv + {} x dwconv",
                    g.palm_dense(),
                    g.palm_depthwise()
                ),
                "host",
                format!(
                    "palm detector, {} nodes; dwconv has no [M,N] GEMM to dispatch, so it \
                     would not move even with a design set",
                    g.palm_graph.len()
                )
            );
            eprintln!(
                "             {:<26} {:<5} {}",
                format!("{} x conv + {} x dwconv", g.lm_dense(), g.lm_depthwise()),
                "host",
                format!(
                    "landmark network, {} nodes, once per hand",
                    g.lm_graph.len()
                )
            );
        }
    }
    eprintln!(
        "             {:<26} {:<5} {}",
        "add, maxpool, pad_c, resize", "host", "elementwise and memory passes"
    );
    eprintln!(
        "             {:<26} {:<5} {}",
        "image front end",
        "host",
        format!(
            "decode, bilinear letterbox to {}px padded with 0, (x/255 - mean)/std; per \
             hand a rotated INTER_AREA crop to {}px",
            g.palm_input_size, g.lm_input_size
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
    // THE PANELS, staged once, at load.
    //
    // The row is here and not in the per-image lines because it is a LOAD's
    // number and not a run's: the panels are staged once and every image
    // reuses them.
    if let Some(info) = array {
        eprintln!(
            "  array      {:.1} MB of bf16 panels staged, once, at load",
            info.staged_bytes as f64 / (1024.0 * 1024.0)
        );
    }
}

/// The `--hands-dump` line, once the path is known.
pub fn print_dump_header(path: &Path, g: &Geometry) {
    eprintln!(
        "  dump       {} + {} graph nodes per image -> {} (fp32 NCHW, JSON header in \
         the first 64 KiB)",
        g.palm_graph.len(),
        g.lm_graph.len(),
        path.display()
    );
}

/// One image's line, on stderr.
pub fn print_run(image: &str, r: &Run) {
    let nd = r.detections.len();
    let nh = r.hands.len();
    eprintln!(
        "  {:<40} {}x{}  {} detection{} -> {} hand{}  {:.1} ms (front {:.1} / palm \
         {:.1} / nms {:.1} / crop {:.1} / lm {:.1} / post {:.1})",
        image,
        r.width,
        r.height,
        nd,
        if nd == 1 { "" } else { "s" },
        nh,
        if nh == 1 { "" } else { "s" },
        r.total_s * 1e3,
        r.front_end_s * 1e3,
        r.palm_s * 1e3,
        r.nms_s * 1e3,
        r.crop_s * 1e3,
        r.lm_s * 1e3,
        r.post_s * 1e3
    );
    // THE ARRAY'S LINE IS PER IMAGE AND NOT IN THE BLOCK, because every
    // number in it is a property of a RUN and the block is printed before
    // there has been one. It would read "0 dispatches, 0.0 ms" in the block,
    // which is true and useless.
    if r.array_convs > 0 {
        eprintln!(
            "             array: {} dispatches, {:.1} ms on the device, {:.1} ms repacking \
             A and {:.1} ms transposing C on the host",
            r.array_dispatches,
            r.array_s * 1e3,
            r.array_repack_s * 1e3,
            r.array_transpose_s * 1e3
        );
    }
}

/// The second stage's geometry, on stderr, before the JSON object.
///
/// Not part of a hand's answer -- the landmarks are the answer -- but it is the
/// one intermediate whose being wrong produces plausible landmarks: an angle
/// off by a degree moves every joint by a fraction of its distance from the
/// wrist and nothing else in the output says so.
pub fn print_crop(r: &Run) {
    eprintln!(
        "             crop: angle {:.4} deg, palm kp box {:.1} {:.1} {:.1} {:.1}, \
         rot_bbox {:.1} {:.1} {:.1} {:.1}, pad_bias {:.1} {:.1}, crop mean {:.4}",
        r.angle_deg,
        r.palm_kp_box[0],
        r.palm_kp_box[1],
        r.palm_kp_box[2],
        r.palm_kp_box[3],
        r.rot_bbox[0],
        r.rot_bbox[1],
        r.rot_bbox[2],
        r.rot_bbox[3],
        r.crop_pad_bias[0],
        r.crop_pad_bias[1],
        r.crop_mean
    );
}

/// The node-dump writer, installed for the lifetime of the process.
///
/// One hook on each network, and it stays installed: it is a call per node, 145
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
                "--hands-dump {}: cannot open it for writing",
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
            "    {{\"image\": {}, \"net\": \"{}\", \"i\": {}, \"c\": {}, \"h\": {}, \
             \"w\": {}, \"n\": {}, \"off\": {}}}",
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
    /// A header that does not fit is a REFUSAL, not a truncation: a short
    /// header reads as a short graph, and a graph that quietly lost its last
    /// sixty nodes is the failure this dump exists to catch.
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
                "--hands-dump {}: the header needs {} bytes for {} node records and the \
                 reserved area is {}. Raise DUMP_HEADER in cli/src/hands.rs; do NOT let \
                 it overflow, because the reader would then take tensor bytes for \
                 header text.",
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
pub fn open(path: &Path) -> Result<Hands, String> {
    Hands::open(path)
}
