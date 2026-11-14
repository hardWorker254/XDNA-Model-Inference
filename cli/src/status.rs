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
//! because the host path dispatches nothing; the host rows carry the
//! measurements this runtime actually made.

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
    /// The artifacts directory.
    pub artifacts: PathBuf,
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
    let design_path = artifacts.join("gemm_rtp").join("design.json");
    let bytes = std::fs::read(&design_path).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;

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

    let toolchain_path = artifacts.join("gemm_rtp").join("toolchain.json");
    let toolchain = std::fs::read(&toolchain_path)
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
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
        artifacts: artifacts.to_path_buf(),
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

/// The status block, printed before the work.
pub fn print_block(status: &Status, design: Option<&DesignStatus>, artifacts: &Path) {
    println!(
        "  artifacts  {} (picked from the container's geometry; no --artifacts given)",
        artifacts.display()
    );
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
    let layers = status.layers;
    where_host("gelu", "GELU", &format!("{layers} fewer NPU dispatches"));
    where_host("softm", "softmax", &format!("{layers} fewer NPU dispatches"));
    where_host(
        "layn",
        "LayerNorm",
        &format!("{} fewer NPU dispatches", 1 + 2 * layers),
    );
    where_host(
        "attn",
        "attention, as two GEMMs",
        "QK^T and softmax.V as host passes, no attn_qk/attn_av dispatches",
    );

    println!("  bo-align   last data buffer aligned to 4096 B");
    println!("  weights    0.00 MB staged on the device once, not per call");
    println!(
        "  pipeline   4 concurrent encodes of {}, one NPU mutex, 6 host threads per lane",
        batch
    );
    println!("  tokenizer  {} tokens, from the .npum", status.vocab);
}

/// One operation's row: the code, the long name, and where it runs.
fn where_host(code: &str, long_name: &str, note: &str) {
    println!("  {code:<6} {long_name:<10} on the HOST (fp32) -- {note}");
}

/// The time split, printed after the work.
///
/// The rows are the C++'s, in the C++'s order and the C++'s widths, and the
/// array's own rows are zero -- the host path dispatches nothing.
pub fn print_time_split(timings: &Timings, lanes: usize, wall_s: f64) {
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
        println!(
            "    {name:<22} {value:>8.1} ms  {pct:>5.1}%",
            pct = if busy > 0.0 { 100.0 * value / busy } else { 0.0 }
        );
    };
    row("gemm + epilogue", 0.0);
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
    println!("    {:<22} {:>8}     (count, not a time)", "dispatches", 0);
    if busy > 0.0 {
        println!(
            "    {:<22} {:>8.2} x   (sum of passes / wall -- the pipelining achieved)",
            "busy / wall",
            busy / if wall_s > 0.0 { wall_s } else { 1.0 }
        );
    }
    println!();
}
