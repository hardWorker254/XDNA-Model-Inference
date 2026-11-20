//! The pose family's status block, its summary and its node dump.
//!
//! # The block is the reference's, row for row
//!
//! `npuimage pose` printed a block before every run, and the golden
//! transcripts hold it, so the block is a contract and not a greeting. Every
//! row says WHERE an operation runs and, where it is not obvious, WHY it runs
//! there: a row that only named the operation would leave the reader to
//! guess whether the array had been considered.
//!
//! # The numbers in the block are this run's
//!
//! The convolution split under each image is measured, not estimated. Two
//! rows are NOT this run's measurement and say so in their own text:
//!
//! * the `16 threads` is the machine's count, asked at load;
//! * the `torch's` baseline is a recorded reference (20.5 ms for this
//!   network on the machine the reference was taken on), because a Rust
//!   runtime cannot measure torch. It is quoted as the constant it is, and
//!   the ratio next to it is this run's network time over that constant.
//!
//! # The dump is the reference's format, byte for byte
//!
//! 64 KiB of header, then every node's fp32 NCHW in graph order, with the
//! header rewritten once the node count is known. `tools/verify/diff_pose_dump.py`
//! reads either side, which is the point: a diagnostic whose two writers
//! disagree about their own format is a diagnostic that reports a difference
//! it made itself.

use npum_cpu::pose::{Geometry, NetTimings, Network};
use npum_cpu::pose_decode::Person;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use crate::flags::Args;
/// The reference the split line compares against, in milliseconds.
///
/// A CONSTANT, and the only honest thing it can be: it is torch's time for
/// this network, measured once on the machine the reference was taken on.
/// Re-measuring it here would need torch, and this runtime does not have it.
pub const TORCH_REF_MS: f64 = 20.5;

/// The 64 KiB the dump reserves for its header.
const DUMP_HEADER: usize = 1 << 16;

/// Refuse a threshold that cannot be parsed instead of quietly falling back
/// to the default: a typo in `--conf` must not silently decode with a
/// different threshold.
pub fn refuse_bad_params(args: &crate::flags::Args) -> Option<String> {
    for name in ["--conf", "--iou", "--kpt", "--max-det"] {
        let Some(value) = args.value(name) else {
            continue;
        };
        let valid = if name == "--max-det" {
            value.parse::<i64>().map(|n| n >= 1).unwrap_or(false)
        } else {
            value.parse::<f64>().map(|f| f.is_finite()).unwrap_or(false)
        };
        if !valid {
            return Some(format!("{name}='{value}' is not a valid number"));
        }
    }
    None
}

/// The decode parameters, with the command line's overrides applied.
///
/// The DEFAULTS are the container's absence of an opinion: a container that
/// carried its own thresholds would be a different model, and this one does
/// not, so the four numbers are the CLI's predict defaults unless the
/// command line moves them.
pub fn params_from(args: &crate::flags::Args) -> npum_cpu::pose_decode::DecodeParams {
    let mut p = npum_cpu::pose_decode::DecodeParams::default();
    if let Some(v) = args.value("--conf").and_then(|v| v.parse::<f64>().ok()) {
        p.conf = v;
    }
    if let Some(v) = args.value("--iou").and_then(|v| v.parse::<f64>().ok()) {
        p.iou = v;
    }
    if let Some(v) = args.value("--kpt").and_then(|v| v.parse::<f64>().ok()) {
        p.keypoint = v;
    }
    if let Some(v) = args.value("--max-det").and_then(|v| v.parse::<i64>().ok()) {
        p.max_det = v;
    }
    p
}

/// What the array says about itself, in the block.
///
/// The three numbers are the DESIGN's: how many shapes it can compute, where
/// it came from, and how many rows it computes at a time.
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
}

/// The block, printed to stderr before the first image.
pub fn print_block(
    name: &str,
    g: &Geometry,
    threads: usize,
    setup_s: f64,
    array: Option<&ArrayInfo>,
) {
    let strides = g.strides.clone();
    let stride_txt = format!(
        "{}/{}/{}",
        strides.first().copied().unwrap_or(0),
        strides.get(1).copied().unwrap_or(0),
        strides.get(2).copied().unwrap_or(0)
    );
    eprintln!(
        "NpuEmbeddings -- body pose (YOLOv8-pose) on the {}",
        if array.is_some() { "AMD NPU" } else { "CPU" }
    );
    eprintln!(
        "  model      {}: {}px letterbox, {} levels (strides {}), {} cells, head {} = 4 box + {} class + {} keypoints x 3",
        name,
        g.input_size,
        g.n_levels(),
        stride_txt,
        g.n_anchors(),
        g.detect_cout,
        g.num_classes,
        g.num_keypoints
    );
    // WHERE THE CONVOLUTIONS RUN, and the two lines that go with it.
    //
    // The row names the side, and the line under it says what the other side
    // costs -- because a reader who has just been told "the array" deserves
    // to know that the host is the faster side here, and the oracle measured
    // that on this network.
    match array {
        Some(info) => {
            eprintln!(
                "  backend    conv on the array ({} streams), everything else on the host",
                info.streams
            );
            eprintln!("             designs   {}", info.designs);
            eprintln!(
                "             {} rows per dispatch; compare against a host run, the two \
                 numbers are printed per image below.",
                info.rows
            );
        }
        None => {
            eprintln!("  backend    conv on the host (default), everything else on the host");
            eprintln!(
                "             pass --npu-ops conv to measure the array instead; see \
                 runtime/include/pose/net.hpp for why the host wins here (2.6x \
                 arithmetic padding waste on N <= 64, 100 of the network's 436 \
                 dispatches for the stem alone, and 660 us per dispatch of which 140 \
                 us is the device's GEMM)."
            );
        }
    }
    let d = npum_cpu::pose_decode::DecodeParams::default();
    eprintln!(
        "  decode     conf {:.2}, iou {:.2}, keypoint {:.2}, max_det {}",
        d.conf, d.iou, d.keypoint, d.max_det
    );
    eprintln!("  setup      {:.2} s", setup_s);
    eprintln!("  ops        (npu = dispatched, host = this process)");
    // The container carries 73 convolutions and the graph runs 72: the last
    // is the head's [1,16,1,1] DFL projection, folded analytically into the
    // head. Printing 73 would claim a convolution neither path runs.
    let n_convs = g.convs.len().saturating_sub(1);
    // WHERE THE CONVOLUTIONS RAN. The row says it, and the text under it
    // says what the split looks like on that side -- a run that dispatched
    // them to the array and printed "host" would be the one thing the flag
    // exists to prevent.
    match array {
        Some(_) => eprintln!(
            "             {:<26} {:<5} {}",
            format!("{n_convs} x conv"),
            "npu",
            "im2col on the host, GEMM on the array, bias by the design, SiLU in the host epilogue"
        ),
        None => eprintln!(
            "             {:<26} {:<5} {}",
            format!("{n_convs} x conv"),
            "host",
            format!("im2col + a blocked fp32 GEMM over rows, {threads} threads")
        ),
    }
    eprintln!(
        "             {:<26} {:<5} {}",
        "silu",
        "host",
        "fused into the convolution's epilogue; it is elementwise on the GEMM's output"
    );
    eprintln!(
        "             {:<26} {:<5} {}",
        "concat, add", "host", "channel joins and the C2f residual, memory-bound"
    );
    eprintln!(
        "             {:<26} {:<5} {}",
        "maxpool (SPPF)", "host", "size-preserving 5x5/1, pad 2"
    );
    eprintln!(
        "             {:<26} {:<5} {}",
        "upsample", "host", "nearest x2, the neck's up path"
    );
    eprintln!(
        "             {:<26} {:<5} {}",
        "image front end",
        "host",
        format!(
            "decode, bilinear letterbox to {}px, pad 114, (x/255 - mean)/std",
            g.input_size
        )
    );
    eprintln!(
        "             {:<26} {:<5} {}",
        "dfl",
        "folded",
        "the exported graph softmaxes its 16 bins and sums them, and the 1x1 \
         projection above it is applied as w . E[bin] + b on that mean -- \
         which is why it is not in the 72"
    );
    eprintln!(
        "             {:<26} {:<5} {}",
        "threshold, nms, inverse letterbox",
        "host",
        "host-side by construction: a list of survivors, not a tensor"
    );
}

/// What one image produced, and where its time went.
pub struct Run {
    /// The people, in the source image's pixels.
    pub people: Vec<Person>,
    /// The source's width.
    pub width: i64,
    /// The source's height.
    pub height: i64,
    /// Seconds for the whole image.
    pub total_s: f64,
    /// Seconds in the front end.
    pub front_end_s: f64,
    /// Seconds in the network.
    pub network_s: f64,
    /// Seconds in the decode.
    pub decode_s: f64,
    /// Where the network's time went, per stage.
    pub timings: NetTimings,
}

/// The per-image lines, on stderr.
pub fn print_run(image: &str, r: &Run, g: &Geometry) {
    let n_convs = g.convs.len().saturating_sub(1);
    // WHERE THE CONVOLUTIONS RAN, and how many dispatches that took.
    //
    // The count of dispatches is the one that says whether the flag did what
    // it said: a run that asked for the array and dispatched nothing is a run
    // that lied, and this is where the lie shows.
    let on_array = r.timings.convs_array > 0;
    eprintln!(
        "  {:<40} {:.2} s  {} people  ({} convs {}, {} dispatches)",
        image,
        r.total_s,
        r.people.len(),
        n_convs,
        if on_array { "array" } else { "host" },
        r.timings.dispatches
    );
    eprintln!(
        "             {:.1} ms front end, {:.1} ms network, {:.1} ms decode, \
         {}x{} px source",
        r.front_end_s * 1e3,
        r.network_s * 1e3,
        r.decode_s * 1e3,
        r.width,
        r.height
    );
    let t = r.timings;
    // THE HOST'S SPLIT, and only on the host's path: on the array's path no
    // convolution ran on the host, and a split of nothing is a number with
    // no meaning.
    if !on_array {
        eprintln!(
            "             host conv split: {:.1} ms gemm, {:.1} ms im2col, \
             {:.1} ms [M,N] transpose, {:.1} ms weight transpose, {:.1} ms \
             elementwise ({:.1}x torch's {:.1})",
            t.gemm_s * 1e3,
            t.im2col_s * 1e3,
            t.transpose_s * 1e3,
            t.weight_s * 1e3,
            t.elementwise_s * 1e3,
            r.network_s * 1e3 / TORCH_REF_MS,
            TORCH_REF_MS
        );
    }
    // THE ARRAY'S SPLIT, and the same span as the host's: every convolution
    // on the array, from the im2col that fills A to the NCHW transpose of C.
    // MEASURED, not the device-busy figure -- the array's own counters say
    // the device is idle for most of it, and a status line that printed only
    // the busy time would report a speedup the picture does not have. The
    // two numbers to put side by side are this and network_s.
    if on_array {
        let us_each = if t.dispatches > 0 {
            t.t_array_s * 1e6 / t.dispatches as f64
        } else {
            0.0
        };
        eprintln!(
            "             array conv: {:.1} ms in {} dispatches \
             ({:.0} us each) -- {:.1} ms GEMM on the device, {:.1} ms the \
             host's own A repack, {:.1} ms the C transpose. The other \
             {:.1} ms of the {:.1} ms network is im2col and the \
             elementwise ops, on the host either way, of which \
             {:.1} ms is the non-convolution span. im2col itself, which \
             t_array also spans because t0 is taken before it, is \
             {:.1} ms.",
            t.t_array_s * 1e3,
            t.dispatches,
            us_each,
            t.t_array_gemm_s * 1e3,
            t.t_array_repack_s * 1e3,
            t.t_array_transpose_s * 1e3,
            (r.network_s - t.t_array_s) * 1e3,
            r.network_s * 1e3,
            t.elementwise_s * 1e3,
            t.im2col_s * 1e3
        );
    }
}

/// The human summary `--text` prints.
///
/// One line per person, then the skeleton by name -- which is the thing a
/// person looking at this wants and the thing the JSON makes them assemble.
pub fn text_summary(image: &str, r: &Run) -> String {
    let mut s = format!(
        "{}: {} {}\n",
        image,
        r.people.len(),
        if r.people.len() == 1 {
            "person"
        } else {
            "people"
        }
    );
    for (i, p) in r.people.iter().enumerate() {
        let visible = p.keypoints.iter().filter(|k| k.visible).count();
        s.push_str(&format!(
            "  person {}  score {:.3}  box {:.0},{:.0} {:.0}x{:.0}  {}/{} \
             keypoints visible\n",
            i,
            p.score,
            p.x1,
            p.y1,
            p.x2 - p.x1,
            p.y2 - p.y1,
            visible,
            p.keypoints.len()
        ));
        for edge in npum_cpu::pose_decode::SKELETON_EDGES.iter() {
            let Some((a, b)) = edge_pair(edge) else {
                continue;
            };
            let (ka, kb) = (&p.keypoints[a], &p.keypoints[b]);
            if !ka.visible || !kb.visible {
                continue;
            }
            s.push_str(&format!(
                "    {:<34} ({:.0},{:.0}) -> ({:.0},{:.0})\n",
                edge, ka.x, ka.y, kb.x, kb.y
            ));
        }
    }
    s
}

/// The two endpoints of an edge, by the skeleton's indices.
///
/// The edges are stored as `a-b`, and the names contain no dash, so the
/// first dash is the separator. An edge whose names do not resolve is
/// skipped rather than printed with the wrong joints: the tables come from
/// one place, and a mismatch means they drifted.
fn edge_pair(edge: &str) -> Option<(usize, usize)> {
    let dash = edge.find('-')?;
    let names = npum_cpu::pose_decode::SKELETON_NAMES;
    let a = names.iter().position(|n| *n == &edge[..dash])?;
    let b = names.iter().position(|n| *n == &edge[dash + 1..])?;
    Some((a, b))
}

/// Write the node dump, in the reference's format.
///
/// The header area is reserved and rewritten at the end: the tensors are
/// transient, so the writer sees each one once and cannot go back to count
/// them. A header that does not fit is a refusal, because a truncated list
/// would read as a short graph.
///
/// THE DUMP IS THE RUN'S, and not a second run: the array, when there is
/// one, is passed in and the dump comes from the same walk the answer comes
/// from. A dump written by the host while the answer came from the array
/// would compare two different networks.
pub fn write_dump(
    path: &Path,
    net: &Network,
    input: &npum_cpu::pose::Tensor,
    array: npum_model::array::ArrayRef<'_, '_>,
) -> Result<(), String> {
    let mut file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    file.write_all(&vec![b' '; DUMP_HEADER])
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut index: Vec<String> = Vec::new();
    {
        let mut hook = |i: usize, t: &npum_cpu::pose::Tensor| {
            let off = file.stream_position().unwrap_or(0);
            index.push(format!(
                "    {{\"image\": 0, \"i\": {}, \"c\": {}, \"h\": {}, \"w\": {}, \
                 \"n\": {}, \"off\": {}}}",
                if i == usize::MAX { -1i64 } else { i as i64 },
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
            let _ = file.write_all(&bytes);
        };
        net.run_with_hook_on_array(input, &mut hook, array)?;
    }
    let mut header = String::from("{\n  \"nodes\": [\n");
    for (i, line) in index.iter().enumerate() {
        if i > 0 {
            header.push_str(",\n");
        }
        header.push_str(line);
    }
    header.push_str("\n  ]\n}\n");
    if header.len() > DUMP_HEADER {
        return Err(format!(
            "--pose-dump {}: the header needs {} bytes for {} nodes and the \
             reserved area is {}. Raise DUMP_HEADER in cli/src/pose.rs; do NOT \
             let it overflow, because the reader would then take tensor bytes \
             for header text.",
            path.display(),
            header.len(),
            index.len(),
            DUMP_HEADER
        ));
    }
    header.push_str(&" ".repeat(DUMP_HEADER - header.len()));
    file.seek(SeekFrom::Start(0))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    file.write_all(header.as_bytes())
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

pub fn command(args: &Args, root: &Path) -> crate::families::Outcome {
    let Some(model) = args.positional.first() else {
        return crate::needs_a_model("pose");
    };
    // Every leading non-flag argument after the model is an image, so
    // `pose <model> a.png b.png --text` runs both.
    let images: Vec<&String> = args.positional.iter().skip(1).collect();
    if images.is_empty() {
        eprintln!("error: `pose` needs an image:");
        eprintln!("    npuimage pose <model> <image.png> [more.png ...]");
        eprintln!("    npuimage serve <model>            (POST /v1/pose)");
        eprintln!(
            "  (PNG and JPEG; anything else is refused rather than guessed at. \
             --text prints a human summary instead of the JSON object, \
             --conf/--iou/--kpt/--max-det move the thresholds, --pose-dump FILE \
             writes every graph node's output)"
        );
        return crate::EXIT_REFUSED;
    }
    crate::families::with_model(model, npum_families::Family::Pose, root, |path, arch| {
        if let Some(reason) = crate::backend::refuse_if_vulkan(args, npum_families::Family::Pose) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        if let Some(reason) =
            crate::families::refuse_op_not_in_arch(args, npum_families::Family::Pose)
        {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        if let Some(reason) = crate::pose::refuse_bad_params(args) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        // `--npu-ops conv` moves the convolutions to the array, and the
        // refusal for a container that cannot do it is the array's own:
        // a container packed without the panels says so, and the message
        // says how to pack it.
        let started = std::time::Instant::now();
        let network = match npum_cpu::pose::Network::open(path) {
            Ok(n) => n,
            Err(reason) => {
                eprintln!("error: {reason}");
                return crate::EXIT_REFUSED;
            }
        };
        let setup_s = started.elapsed().as_secs_f64();
        let geometry = network.geometry().clone();
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| arch.to_string());
        let params = crate::pose::params_from(args);
        // THE ARRAY, OR THE HOST.
        //
        // The flag is a question about where the run happens, and it is
        // answered here and nowhere else: the convolutions go to the array
        // and the run says how many it dispatched. A container that cannot
        // do it is refused HERE, before the first image, with the reason.
        let mut array = if crate::ocr::wants_conv(args) {
            let specs = npum_cpu::pose::array_specs(&geometry);
            match crate::arrayconv::open(path, &specs, args, root) {
                Ok(backend) => Some(backend),
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            }
        } else {
            None
        };
        let array_info = array.as_ref().map(|backend| crate::pose::ArrayInfo {
            streams: backend.stream_count(),
            // THE DESIGN'S OWN PATH, or nothing at all: a container that
            // carries its own set needs nothing beside it, and there is no
            // path to print.
            designs: args
                .value("--artifacts")
                .map(|s| s.to_string())
                .unwrap_or_default(),
            rows: backend.rows_per_dispatch(),
        });
        crate::pose::print_block(
            &name,
            &geometry,
            network.threads(),
            setup_s,
            array_info.as_ref(),
        );
        let _ = arch;

        let text = args.switch("--text");
        let dump = args.value("--pose-dump").map(std::path::PathBuf::from);
        for image in images {
            let decoded = match npum_cpu::vit_image::decode_image(std::path::Path::new(image)) {
                Ok(d) => d,
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            };
            let all = std::time::Instant::now();
            let front = std::time::Instant::now();
            let (input, letterbox) =
                match npum_cpu::pose_decode::letterbox_normalise(&decoded, &geometry) {
                    Ok(v) => v,
                    Err(reason) => {
                        eprintln!("error: {reason}");
                        return crate::EXIT_REFUSED;
                    }
                };
            let front_end_s = front.elapsed().as_secs_f64();

            if let Some(dump) = dump.as_deref() {
                if let Err(reason) =
                    crate::pose::write_dump(dump, &network, &input, array.as_deref_mut())
                {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            }

            let net = std::time::Instant::now();
            let (head, timings) = match network.run_timed_on_array(&input, array.as_deref_mut()) {
                Ok(h) => h,
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            };
            let network_s = net.elapsed().as_secs_f64();

            let dec = std::time::Instant::now();
            let people = match npum_cpu::pose_decode::decode(&head, &geometry, &letterbox, &params)
            {
                Ok(p) => p,
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            };
            let decode_s = dec.elapsed().as_secs_f64();

            let run = crate::pose::Run {
                people,
                width: letterbox.src_w,
                height: letterbox.src_h,
                total_s: all.elapsed().as_secs_f64(),
                front_end_s,
                network_s,
                decode_s,
                timings,
            };
            crate::pose::print_run(image, &run, &geometry);
            if text {
                print!("{}", crate::pose::text_summary(image, &run));
            } else {
                let detection = npum_cpu::pose_decode::Detection {
                    people: run.people,
                    front_end_s,
                    network_s,
                    decode_s,
                    total_s: run.total_s,
                    width: run.width,
                    height: run.height,
                    scale: letterbox.scale,
                    pad_x: letterbox.pad_x,
                    pad_y: letterbox.pad_y,
                    // WHERE THE CONVOLUTIONS RAN. The JSON says it as well as
                    // the status block does, because the JSON is what a
                    // program reads: a reader that asked for the array and
                    // got a host's numbers would have no way to tell.
                    dispatches: timings.dispatches,
                    convs_host: if timings.convs_array > 0 {
                        0
                    } else {
                        geometry.convs.len().saturating_sub(1) as i64
                    },
                };
                print!(
                    "{}",
                    npum_cpu::pose_decode::result_json(
                        &detection,
                        &geometry,
                        &params,
                        image,
                        timings.convs_array > 0
                    )
                );
            }
        }
        0
    })
}
