//! The mppose family's status block, its per-image lines and its node dump.
//!
//! # The block is the reference's, row for row
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
//! # The dump is the reference's format, byte for byte
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

use crate::flags::Args;
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
    // WHERE THE CONVOLUTIONS RAN.
    //
    // The side is read from the SESSION and not from the flag: a row that
    // printed "npu" because --npu-ops conv was on the command line would say
    // so for a run where the design set was missing and everything fell back,
    // which is the one thing a status block must not do.
    match array {
        Some(info) => {
            eprintln!(
                "             {:<26} {:<5} {}",
                format!("{} x conv + {} x dwconv", g.det_dense(), g.det_depthwise()),
                "npu",
                format!(
                    "person detector, {} nodes, on the array's {} padded (K, N) slots; \
                     3x3 dwconv uses the separate dwconv9x3 design when --npu-ops dwconv is named",
                    g.det_graph.len(),
                    info.streams
                )
            );
            eprintln!(
                "             {:<26} {:<5} {}",
                format!(
                    "{} x conv + {} x dwconv",
                    g.pose_dense(),
                    g.pose_depthwise()
                ),
                "npu",
                format!(
                    "landmark network, {} nodes, once per person, on the same {} slots \
                     under a disjoint range",
                    g.pose_graph.len(),
                    info.streams
                )
            );
        }
        None => {
            eprintln!(
                "             {:<26} {:<5} {}",
                format!("{} x conv + {} x dwconv", g.det_dense(), g.det_depthwise()),
                "host",
                format!(
                    "person detector, {} nodes; 3x3 dwconv moves only with --npu-ops dwconv \
                     and a dwconv9x3 design",
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
        }
    }
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
/// Not part of a person's answer -- the landmarks are the answer -- but it is
/// the one intermediate whose being wrong produces plausible landmarks: an angle
/// off by a degree, or a pad_bias off by the square's own padding, moves every
/// joint by a fraction of its distance from the hip and nothing else in the
/// output says so.
pub fn print_crop(r: &Run) {
    eprintln!(
        "             crop: angle {:.4} deg, square {:.0}x{:.0}, pad_bias {:.1} {:.1}, crop \
         mean {:.4}",
        r.angle_deg,
        r.rot_bbox[2],
        r.rot_bbox[3],
        r.crop_pad_bias[0],
        r.crop_pad_bias[1],
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

pub fn command(args: &Args, root: &Path) -> crate::families::Outcome {
    let Some(model) = args.positional.first() else {
        return crate::needs_a_model("mppose");
    };
    let images: Vec<&String> = args.positional.iter().skip(1).collect();
    if images.is_empty() {
        eprintln!("error: this is a MediaPipe Pose model, so say whose body to find:");
        eprintln!("    npuimage mppose <model> <image.png> [more.png ...]");
        eprintln!("  (PNG and JPEG; anything else is refused rather than guessed at)");
        eprintln!("  --max-people 1    run the landmark network on at most N detections");
        eprintln!("  --text            a human summary instead of JSON");
        eprintln!("  --mppose-dump FILE  every graph node's output, for the gate");
        eprintln!(
            "  (every threshold comes from the container, not from a flag: they are\n   \
             part of the checkpoint -- see the refusal above)"
        );
        return crate::EXIT_REFUSED;
    }
    crate::families::with_model(model, npum_families::Family::MpPose, root, |path, arch| {
        if let Some(reason) = crate::backend::refuse_if_vulkan(args, npum_families::Family::MpPose)
        {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        if let Some(reason) =
            crate::families::refuse_op_not_in_arch(args, npum_families::Family::MpPose)
        {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        // The other architectures' own flags are REFUSED rather than ignored: they
        // read as "thresholds for the detection" on a mode that also has a
        // detector, and silently dropping them would produce a run whose people
        // differ from the one that was asked for with nothing in the output
        // saying so.
        for f in [
            "--conf",
            "--iou",
            "--kpt",
            "--max-det",
            "--pose",
            "--pose-dump",
            "--bo-mode",
            "--max-hands",
            "--hands-dump",
        ] {
            if args.value(f).is_some() || args.switch(f) {
                eprintln!(
                    "error: {f}: not a flag of `mppose`. The detector's score and NMS \
                     thresholds and the landmark net's confidence threshold are read from the \
                     container (score_threshold, nms_threshold, top_k, pose_conf_threshold, \
                     person_box_pre_enlarge, person_box_enlarge), because they are part of how \
                     this checkpoint was trained and the OpenCV zoo demo that recorded the \
                     golden used the container's values, not a command line's. For --pose: use \
                     `pose`, which is a different model with different flags. The three dumps \
                     are separate because they record different graphs, and one reader cannot \
                     read another's file."
                );
                return crate::EXIT_REFUSED;
            }
        }
        // --max-people: a CAP on how many detections reach the second stage, not a
        // score threshold. Every person is an independent rotated crop of the
        // original frame, so there is no way to batch them and the only lever is
        // how many to run.
        let mut max_people = -1i64;
        if let Some(v) = args.value("--max-people") {
            match v.parse::<i64>() {
                Ok(n) if (1..=10000).contains(&n) => max_people = n,
                _ => {
                    eprintln!(
                        "error: --max-people '{v}': expected an integer in 1..10000, or omit \
                         it to run the landmark network on every detection"
                    );
                    return crate::EXIT_REFUSED;
                }
            }
        }
        let text = args.switch("--text");
        let started = std::time::Instant::now();
        let network = match crate::mppose::open(path) {
            Ok(n) => n,
            Err(reason) => {
                eprintln!("error: {reason}");
                return crate::EXIT_REFUSED;
            }
        };
        let setup_s = started.elapsed().as_secs_f64();
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| arch.to_string());
        // THE ARRAY, OR THE HOST.
        //
        // The flag is a question about where the run happens, and it is
        // answered here and nowhere else: dense convolutions go to the
        // array when `conv` is named, and 3x3 depthwise goes to its own
        // design when `dwconv` is named. A missing design set is refused
        // HERE, before the first image, with the reason.
        let wants_dw = crate::npu::wants_op(args, "dwconv");
        let mut array = if crate::ocr::wants_conv(args) || wants_dw {
            let specs = npum_cpu::mppose::array_specs(network.geometry());
            let opened = if wants_dw {
                crate::arrayconv::open_with_dw(path, &specs, args, root)
            } else {
                crate::arrayconv::open(path, &specs, args, root)
            };
            match opened {
                Ok(backend) => Some(backend),
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            }
        } else {
            None
        };
        let array_info = array.as_ref().map(|backend| crate::mppose::ArrayInfo {
            streams: backend.stream_count(),
            // THE DESIGN'S OWN PATH, or nothing at all: a container that
            // carries its own set needs nothing beside it, and there is no
            // path to print.
            designs: args
                .value("--artifacts")
                .map(|s| s.to_string())
                .unwrap_or_default(),
            rows: backend.rows_per_dispatch(),
            // The panels the container carries, staged once, at load.
            staged_bytes: backend.staged_bytes(),
        });
        crate::mppose::print_block(&name, network.geometry(), setup_s, array_info.as_ref());
        let _ = arch;

        let dump_path = args.value("--mppose-dump").map(std::path::PathBuf::from);
        let mut dump = match dump_path.as_deref() {
            Some(p) => match crate::mppose::Dump::create(p) {
                Ok(d) => Some(d),
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            },
            None => None,
        };
        if let (Some(p), Some(_)) = (dump_path.as_deref(), dump.as_ref()) {
            crate::mppose::print_dump_header(p, network.geometry());
        }

        let mut exit = 0;
        for image in images {
            let decoded = match npum_cpu::vit_image::decode_image(std::path::Path::new(image)) {
                Ok(d) => d,
                Err(reason) => {
                    eprintln!("  {image}: {reason}");
                    exit = 1;
                    continue;
                }
            };
            let run = {
                let mut hook = |net: &str, node: usize, t: &npum_cpu::pose::Tensor| {
                    if let Some(d) = dump.as_mut() {
                        d.record(net, node, t);
                    }
                };
                npum_cpu::mppose_decode::detect(
                    &network,
                    &decoded,
                    max_people,
                    &mut hook,
                    array.as_deref_mut(),
                )
            };
            let run = match run {
                Ok(r) => r,
                Err(reason) => {
                    eprintln!("  {image}: {reason}");
                    exit = 1;
                    continue;
                }
            };
            if let Some(d) = dump.as_mut() {
                d.image += 1;
            }
            crate::mppose::print_run(image, &run);
            if text {
                print!("{}", npum_cpu::mppose_decode::text_summary(image, &run));
                continue;
            }
            crate::mppose::print_crop(&run);
            print!("{}", npum_cpu::mppose_decode::result_json(&run, image));
        }

        if let (Some(p), Some(d)) = (dump_path.as_deref(), dump) {
            match d.finish(p) {
                Ok(n) => eprintln!("             {n} node tensors written"),
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            }
        }
        exit
    })
}
