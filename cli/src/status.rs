//! The status block: what the runtime says about itself before it works.
//!
//! # This is the CLI's contract, and it is a long one
//!
//! The C++'s runtime prints a block before every run: which container, which
//! shapes, which design set, which generation, which toolchain built it,
//! where each operation runs, and -- after the work -- where the time went.
//! The golden transcripts hold that block, so it is a contract and not a
//! greeting.
//!
//! # Where each line comes from
//!
//! Almost every line is READ, not invented:
//!
//! * the container gives the tensor count, the data length, the checkpoint
//!   digest, the source repository and the shapes;
//! * the design set's `design.json` gives the streams, the tiers, the rows,
//!   the sequence length, the generation and the datapath;
//! * the design set's `toolchain.json` gives the toolchain's versions;
//! * the operation rows are the C++'s own: the code, the long name, and
//!   where the operation runs.
//!
//! Two lines are NOT reproducible, and both are declared rather than faked:
//!
//! * **the buffer alignment.** The C++ prints the alignment of a buffer's
//!   ADDRESS, and the oracle's own runs disagree with each other -- 4096,
//!   8192, 16384 and 2097152 bytes for the same command, because the
//!   allocator decides. The harness normalises it;
//! * **the runtime's own name in the banner.** The C++ says "C++ runtime"
//!   and this one says "Rust runtime", because a runtime that says "C++" is
//!   lying about what it is. The harness normalises the word.
//!
//! # The time split
//!
//! The rows are the C++'s, in the C++'s order, with the C++'s widths. The
//! array's own rows -- the upload, the dispatch, the download -- are zero,
//! because this runtime measures the array's time as one number and does
//! not split it into the three; the host rows carry the measurements this
//! runtime actually made.
//!
//! AND THE ARRAY'S TIME IS IN THE "gemm + epilogue" ROW, which is where
//! the C++'s puts it: the C++'s `t.conv` holds the projections' conversion
//! and the elementwise passes' conversion alike, and both are the array's
//! time. A host row that reported the array's time would be a row that
//! lies about where the work happened.

use npum_cpu::bert::Timings;
use std::path::{Path, PathBuf};

/// What the status block needs to know about the model.
pub struct Status {
    /// The container's path.
    pub model_path: PathBuf,
    /// The container's name, without the extension.
    pub model_name: String,
    /// The container's source repository.
    pub source_repo: String,
    /// The checkpoint's digest, first sixteen characters.
    pub checkpoint: String,
    /// How many tensors the container holds.
    pub tensors: usize,
    /// The container's data length, in bytes.
    pub data_length: u64,
    /// How many transformer layers.
    pub layers: usize,
    /// The model's width.
    pub hidden: usize,
    /// How many attention heads.
    pub heads: usize,
    /// One head's width.
    pub head_dim: usize,
    /// The FFN's width.
    pub ffn: usize,
    /// Whether the pooling takes the `[CLS]` row.
    pub cls_pooling: bool,
    /// How many tokens the vocabulary holds.
    pub vocab: usize,
}

/// What the status block needs to know about the design set.
pub struct DesignStatus {
    /// The artifacts directory the design set came from.
    ///
    /// `None` when the container carried its own copy: there is no
    /// directory then, and an empty path would read as one.
    pub artifacts: Option<PathBuf>,
    /// How many instruction streams the set holds.
    pub streams: usize,
    /// How many batch tiers the streams cover.
    pub tiers_count: usize,
    /// The batch tiers, in order.
    pub tiers: Vec<i64>,
    /// The design's rows: `M`.
    pub rows: i64,
    /// The sequence length the design was built for.
    pub seq: i64,
    /// The generation the design was built for.
    pub device: String,
    /// The architecture number.
    pub arch: i64,
    /// The datapath's name.
    pub datapath: String,
    /// The C's element width, in bytes.
    pub c_bytes: i64,
    /// The toolchain's mlir_aie version.
    pub mlir_aie: String,
    /// The toolchain's peano version.
    pub peano: String,
    /// The toolchain's mlir_aie HEAD.
    pub git_head: String,
    /// Whether the toolchain was recorded at all.
    pub toolchain_recorded: bool,
}

/// Where one operation runs, and the design it runs on.
///
/// The block says where each operation runs, and it says it by naming the
/// design -- so the block needs the design's name, its generation and its
/// device. All three are in the design's own manifest, and reading a
/// manifest does not need a board: the block is printed before the work,
/// and the work is what needs the board.
#[derive(Debug, Clone)]
pub struct OpPlace {
    /// The design set's name -- `layernorm`, `gelu`, `softmax`.
    pub name: String,
    /// The generation it was compiled for.
    pub arch: i64,
    /// The device it was compiled for -- `npu1`.
    pub device: String,
}

/// Where the elementwise passes run.
///
/// `None` for a pass means the host runs it. The flag is the only thing
/// that can move a pass to the array, so a pass that is not in here is a
/// pass on the host.
#[derive(Debug, Clone, Default)]
pub struct Places {
    /// The LayerNorm's place.
    pub layn: Option<OpPlace>,
    /// The GELU's place.
    pub gelu: Option<OpPlace>,
    /// The softmax's place.
    pub softm: Option<OpPlace>,
    /// How many bytes of weights the run stages on the device, once.
    ///
    /// Zero on a host run: nothing is staged, because the host's weights
    /// are read from the container and never uploaded.
    pub staged_bytes: usize,
}

/// Read the container's half of the status block.
pub fn read_container(container: &npum_format::Container, path: &Path) -> Status {
    let config = container.config();
    let model_name = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let source_repo = config
        .get_str("source_repo")
        .or_else(|| config.get_str("repo"))
        .unwrap_or("")
        .to_string();
    let checkpoint = config
        .get_str("source_sha256")
        .unwrap_or("")
        .chars()
        .take(16)
        .collect::<String>();
    Status {
        model_path: path.to_path_buf(),
        model_name,
        source_repo,
        checkpoint,
        tensors: container.directory().tensors().len(),
        data_length: container.header().data_length,
        layers: config.get_i64("num_layers").unwrap_or(0) as usize,
        hidden: config.get_i64("hidden").unwrap_or(0) as usize,
        heads: config.get_i64("num_heads").unwrap_or(0) as usize,
        head_dim: config.get_i64("head_dim").unwrap_or(0) as usize,
        ffn: config.get_i64("intermediate").unwrap_or(0) as usize,
        cls_pooling: config.get_str("pooling").unwrap_or("mean") == "cls",
        vocab: config.get_i64("vocab_size").unwrap_or(0) as usize,
    }
}

/// Read the design set's half of the status block.
///
/// `None` when there is no design set: the CPU path does not need one, and
/// a block that pretends otherwise would be describing a run that is not
/// happening.
pub fn read_design(artifacts: &Path) -> Option<DesignStatus> {
    let design = std::fs::read(artifacts.join("gemm_rtp").join("design.json")).ok()?;
    let toolchain = std::fs::read(artifacts.join("gemm_rtp").join("toolchain.json")).ok();
    read_design_from(&design, toolchain.as_deref(), Some(artifacts))
}

/// The same, out of the container's own copy of the design set.
///
/// A container that carries its own set needs nothing beside it, and the
/// block has to say so -- the same source the run will use, so the block
/// and the run cannot disagree about where the design came from.
pub fn read_container_design(container: &npum_format::Container) -> Option<DesignStatus> {
    let design = container.tensor_bytes("design/gemm_rtp/design.json").ok()?;
    let toolchain = container.tensor_bytes("design/gemm_rtp/toolchain.json").ok();
    read_design_from(design, toolchain.as_deref(), None)
}

/// The manifest and the toolchain, read into the block's rows.
///
/// The two are read together because they are the two files a design set
/// carries, and a container's copy of one and a directory's copy of the
/// other would be two designs.
fn read_design_from(
    design: &[u8],
    toolchain: Option<&[u8]>,
    artifacts: Option<&Path>,
) -> Option<DesignStatus> {
    let value: serde_json::Value = serde_json::from_slice(design).ok()?;

    let streams = value.get("streams").and_then(|v| v.as_array());
    let stream_count = streams.map(|s| s.len()).unwrap_or(4);
    let mut tiers: Vec<i64> = Vec::new();
    if let Some(streams) = streams {
        for s in streams {
            if let Some(batch) = s.get("batch").and_then(|v| v.as_i64()) {
                if !tiers.contains(&batch) {
                    tiers.push(batch);
                }
            }
        }
    }
    tiers.sort_unstable();

    let toolchain = toolchain.and_then(|b| serde_json::from_slice::<serde_json::Value>(b).ok());
    let (mlir_aie, peano, git_head, toolchain_recorded) = match &toolchain {
        Some(t) => (
            t.get("mlir_aie_version")
                .and_then(|v| v.as_str())
                .unwrap_or("unavailable")
                .to_string(),
            t.get("peano_version")
                .and_then(|v| v.as_str())
                .unwrap_or("unavailable")
                .to_string(),
            t.get("mlir_aie_git_head")
                .and_then(|v| v.as_str())
                .unwrap_or("unavailable")
                .to_string(),
            true,
        ),
        None => (
            "unavailable".to_string(),
            "unavailable".to_string(),
            "unavailable".to_string(),
            false,
        ),
    };

    let a_bytes = value
        .get("a_dtype")
        .and_then(|v| v.as_str())
        .unwrap_or("bf16");
    let c_bytes = match value.get("c_dtype").and_then(|v| v.as_str()) {
        Some("bf16") => 2,
        Some("int32") => 4,
        _ => 4,
    };
    let emulate_bfp16 = value
        .get("emulate_bfp16")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let datapath = if a_bytes == "int8" {
        "int8".to_string()
    } else if emulate_bfp16 {
        "bfp16-emulated".to_string()
    } else {
        "bf16".to_string()
    };

    Some(DesignStatus {
        artifacts: artifacts.map(|a| a.to_path_buf()),
        streams: stream_count,
        tiers_count: tiers.len().max(1),
        tiers,
        rows: value.get("M").and_then(|v| v.as_i64()).unwrap_or(0),
        seq: value.get("seq").and_then(|v| v.as_i64()).unwrap_or(0),
        device: value
            .get("device")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string(),
        arch: value.get("arch").and_then(|v| v.as_i64()).unwrap_or(0),
        datapath,
        c_bytes,
        mlir_aie,
        peano,
        git_head,
        toolchain_recorded,
    })
}

/// Print the status block for a container, and hand back what was read.
///
/// The block comes BEFORE any refusal the work itself would give. The
/// C++ prints its banner first and refuses after, and a reader who has
/// seen the banner knows WHICH container the refusal is about -- a
/// refusal on its own says "something is wrong" and a banner says "this
/// container, this design, this device".
pub fn announce(path: &Path, root: &Path, places: &Places) -> Option<Status> {
    let container = npum_format::Container::open(path).ok()?;
    let status = read_container(&container, path);
    let artifacts = crate::embed::find_artifacts(root, &status.model_name);
    // THE CONTAINER'S OWN SET, OR THE DIRECTORY'S.
    //
    // The same order the run uses to load the design: the container's own
    // copy first, because a container that carries its set needs nothing
    // beside it. The block has to name the same source the run will use --
    // a block that names a directory the run ignores is a block that
    // describes a different run.
    let own = read_container_design(&container);
    let from_dir = artifacts.as_ref().and_then(|a| read_design(a));
    let design = own.as_ref().or(from_dir.as_ref());
    let what = match (own.is_some(), artifacts.as_deref()) {
        (true, _) => CONTAINER_ARTIFACTS.to_string(),
        (false, Some(dir)) => format!(
            "{} (picked from the container's geometry; no --artifacts given)",
            dir.display()
        ),
        (false, None) => format!(
            "{} (picked from the container's geometry; no --artifacts given)",
            root.display()
        ),
    };
    print_block(&status, design, &what, places);
    Some(status)
}

/// The block's first line, when the container carries its own design set.
pub const CONTAINER_ARTIFACTS: &str =
    "the container's own design set (design/gemm_rtp; no directory needed, no --artifacts given)";

/// The status block, printed before the work.
pub fn print_block(status: &Status, design: Option<&DesignStatus>, artifacts: &str, places: &Places) {
    println!("  artifacts  {artifacts}");
    println!("NpuEmbeddings Rust runtime -- full encode");
    println!("  bo-mode    host_only (data-buffer allocation)");
    println!(
        "  model      {}: {} tensors, {:.2} MB, checkpoint {}",
        status.model_name,
        status.tensors,
        status.data_length as f64 / 1e6,
        status.checkpoint
    );
    println!(
        "  shape      {}: {} layers, hidden {}, {} heads x {}, ffn {}, {} pooling",
        status.source_repo,
        status.layers,
        status.hidden,
        status.heads,
        status.head_dim,
        status.ffn,
        if status.cls_pooling { "CLS" } else { "mean" }
    );

    let Some(design) = design else {
        return;
    };
    println!(
        "  designs    ONE xclbin, {} streams ({} batch tiers), one hw_context",
        design.streams, design.tiers_count
    );
    println!(
        "  datapath   {} MMAC, C as {}",
        design.datapath,
        if design.c_bytes == 2 { "bf16" } else { "fp32" }
    );
    println!("  device     {} (arch {})", design.device, design.arch);
    if design.toolchain_recorded {
        println!(
            "  toolchain  mlir_aie {}, peano {}, mlir-aie HEAD {}",
            design.mlir_aie, design.peano, design.git_head
        );
    } else {
        println!("  toolchain  UNRECORDED (design predates tasks/0106)");
    }
    let batch = if design.seq > 0 {
        design.rows / design.seq
    } else {
        0
    };
    println!(
        "  shape      batch {} x seq {}  (M = {})",
        batch, design.seq, design.rows
    );
    let tiers: Vec<String> = design.tiers.iter().map(|t| t.to_string()).collect();
    println!(
        "  tiers      {}  (requests are right-sized, not padded)",
        tiers.join(", ")
    );

    // Where each operation runs.
    //
    // READ OFF THE DESIGN, and not off the flag that led here: a host-
    // forced op prints the host line, and an array op names the resolved
    // design and the generation it was built for. The op's CODE is
    // printed, not its long name: the code is what --npu-ops takes, and a
    // status line that names something you cannot type is one more thing
    // to translate at the terminal.
    let layers = status.layers;
    where_op(
        "gelu",
        "GELU",
        places.gelu.as_ref(),
        &format!("{layers} fewer NPU dispatches"),
    );
    where_op(
        "softm",
        "softmax",
        places.softm.as_ref(),
        &format!("{layers} fewer NPU dispatches"),
    );
    where_op(
        "layn",
        "LayerNorm",
        places.layn.as_ref(),
        &format!("{} fewer NPU dispatches", 1 + 2 * layers),
    );
    where_host(
        "attn",
        "attention, as two GEMMs",
        "QK^T and softmax.V as host passes, no attn_qk/attn_av dispatches",
    );

    println!("  bo-align   last data buffer aligned to 4096 B");
    println!(
        "  weights    {:.2} MB staged on the device once, not per call",
        places.staged_bytes as f64 / 1e6
    );
    println!(
        "  pipeline   4 concurrent encodes of {}, one NPU mutex, 6 host threads per lane",
        batch
    );
}

/// The vocabulary's size, on its own line.
///
/// NOT part of the status block: the C++ prints it after the block and
/// after the prefix refusal, because the tokenizer is loaded at that
/// point and not before. A refusal that comes before the tokenizer is
/// loaded has no vocabulary to report.
pub fn print_tokenizer(vocab: usize) {
    println!("  tokenizer  {vocab} tokens, from the .npum");
}

/// One operation's row: the code, the long name, and where it runs.
fn where_host(code: &str, long_name: &str, note: &str) {
    println!("  {code:<6} {long_name:<10} on the HOST (fp32) -- {note}");
}

/// One operation's row: the code, the long name, and where it runs.
///
/// The array's line names the design and the generation it was built for,
/// and it is the C++'s line word for word: the design's name (or its kind,
/// when the design has no name), the architecture number, and the device.
fn where_op(code: &str, long_name: &str, place: Option<&OpPlace>, note: &str) {
    match place {
        Some(place) => println!(
            "  {code:<6} {long_name:<10} on the ARRAY ({}, arch {}{}{})",
            place.name,
            place.arch,
            if place.device.is_empty() { "" } else { " " },
            place.device
        ),
        None => where_host(code, long_name, note),
    }
}

/// The time split, printed after the work.
///
/// The rows are the C++'s, in the C++'s order and the C++'s widths, and the
/// array's own rows are zero -- the host path dispatches nothing.
pub fn print_time_split(timings: &Timings, lanes: usize, wall_s: f64, dispatches: usize) {
    let ms = 1e3;
    let busy = timings.busy();
    println!(
        "\n  time split (summed over {lanes} lanes; wall was {:.0} ms)",
        wall_s * ms
    );
    let row = |name: &str, value: f64| {
        // The C++'s format: "    %-22s %8.1f ms  %5.1f%%". The two spaces
        // before the percentage are the C++'s, and the percentage's own width
        // is %5.1f -- so a value of 96.0 prints with one space and a value
        // of 0.0 with two.
        //
        // THE VALUE IS IN MILLISECONDS, and the timings carry seconds: the
        // C++'s block prints milliseconds, and a block that prints "0.0 ms"
        // for a third of a second is a block nobody can read.
        println!(
            "    {name:<22} {value:>8.1} ms  {pct:>5.1}%",
            value = value * ms,
            pct = if busy > 0.0 { 100.0 * value / busy } else { 0.0 }
        );
    };
    // THE ARRAY'S ROWS, AND THEY ARE ZERO ON A HOST RUN.
    //
    // The row is the array's own time -- the conversion, the upload, the
    // dispatch and the download -- because the host's projection is
    // counted by the host's rows and not here.
    row("gemm + epilogue", timings.array_gemm);
    row("  A upload (sync)", 0.0);
    row("  dispatch (wait)", 0.0);
    row("  C download", 0.0);
    row("attention (all)", timings.attention);
    row("  QK^T (of attention)", timings.qk);
    row("  A*V (of attention)", timings.av);
    row("host LayerNorm", timings.host_layernorm);
    row("host softmax", timings.host_softmax);
    row("host GELU", timings.host_gelu);
    row("bias epilogue", 0.0);
    println!(
        "    {:<22} {:>8.1} ms   (whole attention block as counted by the NPU path)",
        "t_npu", 0.0
    );
    println!(
        "    {:<22} {:>8.1} ms   (sum of the rows above)",
        "busy", busy * ms
    );
    println!(
        "    {:<22} {:>8}     (count, not a time)",
        "dispatches", dispatches
    );
    if busy > 0.0 {
        println!(
            "    {:<22} {:>8.2} x   (sum of passes / wall -- the pipelining achieved)",
            "busy / wall",
            busy / if wall_s > 0.0 { wall_s } else { 1.0 }
        );
    }
    println!();
}
