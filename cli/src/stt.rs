//! The speech family's status block.
//!
//! # A different block, because it is a different architecture
//!
//! The text family's block describes a BERT-shaped model and the ViT's
//! describes a patch-and-classify model. The speech family's block has to
//! say where THIRTEEN operations run -- the transform and the mel bank are
//! separable, the encoder and the decoder have their own GEMM sets, and the
//! attention, the logits and the decode are three more -- and a block that
//! merged them would hide exactly the ones that had moved.
//!
//! # THE ROWS ARE READ, NOT WRITTEN
//!
//! The GEMM names, the row counts and the tier counts come off the LOADED
//! design sets (`gemm_rtp` and `gemm_rtp_dec`), so the table cannot
//! describe an export other than the one that is there. The host rows are
//! this build's own schedule: there is no design for them to be read from.
//!
//! # THE DECODER'S NAMES GO IN ONCE
//!
//! The decoder set repeats its seven streams once per batch tier, and the
//! status line prints the tier row counts separately -- so the NAMES go in
//! once. Printing seven names three times reads as twenty-one GEMMs per
//! layer.

use npum_xdna::array_gemms::GemmSpec;
use serde_json::Value;
use std::path::{Path, PathBuf};

use crate::flags::Args;
/// The speech family's geometry, as the container states it.
pub struct Geometry {
    /// The container's name.
    pub model_name: String,
    /// The model's width.
    pub d_model: i64,
    /// How many encoder layers.
    pub enc_layers: i64,
    /// How many decoder layers.
    pub dec_layers: i64,
    /// The encoder's FFN width.
    pub enc_intermediate: i64,
    /// The decoder's FFN width.
    pub dec_intermediate: i64,
    /// How many heads.
    pub heads: i64,
    /// One head's width.
    pub head_dim: i64,
    /// How many mel bands.
    pub mel_bins: i64,
    /// The vocabulary's size.
    pub vocab: i64,
    /// The encoder's position count.
    pub max_seq: i64,
}

/// Read the geometry out of a container.
///
/// Every key is REQUIRED. A container that does not carry the geometry is
/// refused rather than run with a default: the numbers below are printed
/// back to the reader, and a printed default is a claim about the model.
pub fn read_geometry(container: &npum_format::Container, path: &Path) -> Option<Geometry> {
    let config = container.config();
    let model_name = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    Some(Geometry {
        model_name,
        d_model: config.get_i64("d_model")?,
        enc_layers: config.get_i64("num_encoder_layers")?,
        dec_layers: config.get_i64("num_decoder_layers")?,
        enc_intermediate: config.get_i64("encoder_intermediate")?,
        dec_intermediate: config.get_i64("decoder_intermediate")?,
        heads: config.get_i64("num_heads")?,
        head_dim: config.get_i64("head_dim")?,
        mel_bins: config.get_i64("num_mel_bins")?,
        vocab: config.get_i64("vocab_size")?,
        max_seq: config.get_i64("max_seq_len")?,
    })
}

/// The encoder's projections, in the order they are dispatched.
///
/// The shape of each projection comes from the geometry the CPU pipeline also
/// reads: `attn_out` is square, `qkv` is three widths, and the FFN's up and
/// down projections are the transpose of one another. A design whose streams
/// disagree with these numbers is a design for another model, and the array's
/// own loader refuses it before the first dispatch.
pub fn encoder_specs(g: &Geometry) -> Vec<GemmSpec> {
    let d = g.d_model as usize;
    let f = g.enc_intermediate as usize;
    let mut specs = Vec::with_capacity(g.enc_layers as usize * 4);
    for layer in 0..g.enc_layers as usize {
        for (op, k, n) in [
            ("qkv", d, 3 * d),
            ("attn_out", d, d),
            ("ffn_up", d, f),
            ("ffn_down", f, d),
        ] {
            specs.push(GemmSpec {
                weight: format!("encoder.layers.{layer}.{op}"),
                op: op.to_string(),
                k,
                n,
            });
        }
    }
    specs
}

/// The decoder's projections, in the order they are dispatched.
///
/// `cross_kv` is the cross-attention cache, built once per window rather than
/// once per token. The decoder set DOES have the stream (K=512, N=1024), so
/// it is staged like the other projections and dispatched through the same
/// GEMM seam from `set_source`.
pub fn decoder_specs(g: &Geometry) -> Vec<GemmSpec> {
    let d = g.d_model as usize;
    let f = g.dec_intermediate as usize;
    let mut specs = Vec::with_capacity(g.dec_layers as usize * 6);
    for layer in 0..g.dec_layers as usize {
        for (op, k, n) in [
            ("self_qkv", d, 3 * d),
            ("self_attn_out", d, d),
            ("cross_q", d, d),
            ("cross_kv", d, 2 * d),
            ("cross_attn_out", d, d),
            ("ffn_up", d, f),
            ("ffn_down", f, d),
        ] {
            specs.push(GemmSpec {
                weight: format!("decoder.layers.{layer}.{op}"),
                op: op.to_string(),
                k,
                n,
            });
        }
    }
    specs
}

/// Where each requested operation actually runs, for the status block.
///
/// The array's place needs TWO things: the flag must name the code, and the
/// design set the code would use must be present. A design on disk is not a
/// dispatch, and a flag asking for a pass without a design is a refusal -- so
/// the block says the host for it, and the refusal that follows says why.
#[derive(Debug, Clone, Copy)]
pub struct Places {
    pub enc_gemm: bool,
    pub dec_gemm: bool,
    pub attn: bool,
    pub fft: bool,
    pub mproj: bool,
    pub layn: bool,
    pub gelu: bool,
    pub softm: bool,
    pub logit: bool,
    pub conv: bool,
}

impl Places {
    pub fn none() -> Self {
        Self {
            enc_gemm: false,
            dec_gemm: false,
            attn: false,
            fft: false,
            mproj: false,
            layn: false,
            gelu: false,
            softm: false,
            logit: false,
            conv: false,
        }
    }

    pub fn any(&self) -> bool {
        self.enc_gemm
            || self.dec_gemm
            || self.attn
            || self.fft
            || self.mproj
            || self.layn
            || self.gelu
            || self.softm
            || self.logit
            || self.conv
    }
}

/// Read the places from the flag and the designs that were found.
pub fn places(
    container: &npum_format::Container,
    path: &Path,
    root: &Path,
    args: &Args,
    encoder: Option<&Design>,
    decoder: Option<&Design>,
) -> Places {
    let named = args.value("--artifacts");
    let elt = |set: &str| crate::npu::eltwise_design_present(container, path, root, named, set);
    let gemm = crate::npu::wants_op(args, "gemm");
    // A container can carry its own design set, in which case there is no
    // directory for `read_design` to read. That is still a design the run
    // will open, so it counts as present here.
    let enc = encoder.is_some() || crate::npu::has_own_design(container, "gemm_rtp");
    let dec = decoder.is_some() || crate::npu::has_own_design(container, "gemm_rtp_dec");
    let stream_present = |op: &str| {
        encoder
            .map(|d| d.streams.iter().any(|s| s == op))
            .unwrap_or(false)
    };
    let logit_present = |op: &str| {
        decoder
            .map(|d| d.streams.iter().any(|s| s == op))
            .unwrap_or(false)
    };
    Places {
        enc_gemm: gemm && enc,
        dec_gemm: gemm && dec,
        attn: crate::npu::wants_op(args, "attn") && enc,
        fft: crate::npu::wants_op(args, "fft") && stream_present("dft400"),
        mproj: crate::npu::wants_op(args, "mproj") && stream_present("mel_proj"),
        layn: crate::npu::wants_op(args, "layn") && (elt("layernorm") || elt("rmsnorm")),
        gelu: crate::npu::wants_op(args, "gelu") && elt("gelu"),
        softm: crate::npu::wants_op(args, "softm") && elt("softmax"),
        logit: crate::npu::wants_op(args, "logit") && logit_present("logits_0"),
        conv: crate::npu::wants_op(args, "conv") && enc,
    }
}

/// What one design set says about itself.
pub struct Design {
    /// The directory the set is in.
    pub path: PathBuf,
    /// The datapath's name: `bf16`, `bfp16-emulated` or `int8`.
    pub datapath: String,
    /// The C operand's element type: `bf16` or `fp32`.
    pub c_type: String,
    /// The GEMM streams' names, in the design's order.
    pub gemm_ops: Vec<String>,
    /// Every stream's op name, including the baked front-end ones.
    pub streams: Vec<String>,
    /// The encoder's row count, or the decoder's tiers.
    pub rows: i64,
    /// The decoder's row count per tier.
    pub tier_rows: Vec<i64>,
}

/// Read a design set's `design.json`.
///
/// The GEMM streams are the ones whose op maps to the `gemm` code; the
/// attention's `attn_qk`/`attn_av`, the mel bank's `mel_proj` and the
/// transform's `dft400` are other codes and have their own rows.
pub fn read_design(dir: &Path, decoder: bool) -> Option<Design> {
    let name = if decoder { "gemm_rtp_dec" } else { "gemm_rtp" };
    let bytes = std::fs::read(dir.join(name).join("design.json")).ok()?;
    let value: Value = serde_json::from_slice(&bytes).ok()?;
    let datapath = if value.get("a_dtype").and_then(|v| v.as_str()) == Some("int8") {
        "int8".to_string()
    } else if value.get("emulate_bfp16").and_then(|v| v.as_bool()) == Some(true) {
        "bfp16-emulated".to_string()
    } else {
        "bf16".to_string()
    };
    let c_type = match value.get("c_dtype").and_then(|v| v.as_str()) {
        Some("bf16") => "bf16".to_string(),
        _ => "fp32".to_string(),
    };
    let mut gemm_ops: Vec<String> = Vec::new();
    let mut all_ops: Vec<String> = Vec::new();
    // The decoder set repeats its streams once per batch tier, and the tier
    // row counts are the TIERS and not the batch numbers: the tiers are
    // 1, 2 and 4, and the rows are 64, 128 and 256.
    let mut tier_rows: Vec<(i64, i64)> = Vec::new();
    if let Some(streams) = value.get("streams").and_then(|v| v.as_array()) {
        for stream in streams {
            let Some(op) = stream.get("op").and_then(|v| v.as_str()) else {
                continue;
            };
            if !all_ops.iter().any(|o| o == op) {
                all_ops.push(op.to_string());
            }
            if code_for_stream(op) == "gemm" && !gemm_ops.iter().any(|o| o == op) {
                gemm_ops.push(op.to_string());
            }
            // The decoder set repeats its streams per tier: the row count of
            // the tier's own `self_qkv` is that tier's row count.
            if decoder && op == "self_qkv" {
                if let (Some(batch), Some(rows)) = (
                    stream.get("batch").and_then(|v| v.as_i64()),
                    stream.get("M").and_then(|v| v.as_i64()),
                ) {
                    if !tier_rows.iter().any(|(b, _)| *b == batch) {
                        tier_rows.push((batch, rows));
                    }
                }
            }
        }
    }
    tier_rows.sort_by_key(|(batch, _)| *batch);
    let tier_rows: Vec<i64> = tier_rows.into_iter().map(|(_, rows)| rows).collect();
    let rows = if decoder {
        0
    } else {
        value.get("M").and_then(|v| v.as_i64()).unwrap_or(0)
    };
    Some(Design {
        path: dir.to_path_buf(),
        datapath,
        c_type,
        gemm_ops,
        streams: all_ops,
        rows,
        tier_rows,
    })
}

/// Which `--npu-ops` code a stream belongs to.
///
/// The same mapping the reference's `code_for_stream` makes: the attention's two
/// streams are one code, the mel bank's one, the transform's one, and
/// everything else is a GEMM.
fn code_for_stream(op: &str) -> &'static str {
    if op.starts_with("attn_qk") || op.starts_with("attn_av") {
        "attn"
    } else if op.starts_with("mel_proj") {
        "mproj"
    } else if op.starts_with("dft") {
        "fft"
    } else if op.starts_with("conv") {
        "conv"
    } else if op.starts_with("logits") {
        "logit"
    } else {
        "gemm"
    }
}

/// The block, printed to the stderr.
///
/// The rows are the reference's, in the reference's order, with the reference's widths: a
/// row is a name, a place and a reason, and the reason is what makes the
/// row worth reading.
pub fn print_block(
    g: &Geometry,
    designs: &Path,
    encoder: Option<&Design>,
    decoder: Option<&Design>,
    places: &Places,
) {
    eprintln!("NpuEmbeddings -- Whisper speech to text on the AMD NPU");
    eprintln!(
        "  model      {}: d_model {}, {}+{} layers, {} heads x {}, {} mel bins, vocab {}",
        g.model_name,
        g.d_model,
        g.enc_layers,
        g.dec_layers,
        g.heads,
        g.head_dim,
        g.mel_bins,
        g.vocab
    );
    eprintln!("  designs    {}", designs.display());
    let datapath = encoder.map(|d| format!("{} MMAC, C as {}", d.datapath, d.c_type));
    eprintln!(
        "  datapath   {}",
        datapath.unwrap_or_else(|| "UNRECORDED (no design set)".to_string())
    );
    eprintln!("  ops        (npu = dispatched, host = this process)");
    row(
        "transform, 400 points",
        if places.fft { "npu" } else { "host" },
        if places.fft {
            "3000 transforms as one GEMM against bf16 twiddles in gemm_rtp"
        } else {
            "3001 mixed-radix transforms of 400 points, fp64"
        },
    );
    row(
        "slaney mel bank",
        if places.mproj { "npu" } else { "host" },
        if places.mproj {
            "the bank as a GEMM against bf16 weights in gemm_rtp"
        } else {
            "the checkpoint's own filter bank, 201 x n_mels"
        },
    );
    row(
        "conv1, conv2",
        if places.conv { "npu" } else { "host" },
        if places.conv {
            "windowed A rows gathered on the host; generated bf16 panels reuse attn_out and ffn_down"
        } else {
            "fp32, d^2 MACs; --npu-ops conv sends it to the array"
        },
    );
    let enc_ops = encoder.map(|d| d.gemm_ops.join(", ")).unwrap_or_default();
    let enc_rows = encoder.map(|d| d.rows).unwrap_or(0);
    row(
        &format!("encoder GEMMs, {} layers", g.enc_layers),
        if places.enc_gemm { "npu" } else { "host" },
        &format!("{enc_ops}, M={enc_rows} rows; --npu-ops gemm sends them to the array"),
    );
    let dec_ops = decoder.map(|d| d.gemm_ops.join(", ")).unwrap_or_default();
    let tiers = decoder
        .map(|d| {
            d.tier_rows
                .iter()
                .map(|r| r.to_string())
                .collect::<Vec<String>>()
                .join("/")
        })
        .unwrap_or_default();
    row(
        &format!("decoder GEMMs, {} layers", g.dec_layers),
        if places.dec_gemm { "npu" } else { "host" },
        &format!("{dec_ops}, M={tiers} rows per tier; --npu-ops gemm sends them to the array"),
    );
    row(
        "layer_norm",
        if places.layn { "npu" } else { "host" },
        "fp32 on this process; --npu-ops layn sends it to the array",
    );
    row(
        "gelu",
        if places.gelu { "npu" } else { "host" },
        "fp32 on this process; --npu-ops gelu sends it to the array",
    );
    row(
        "softmax",
        if places.softm { "npu" } else { "host" },
        "fp32 on this process; --npu-ops softm sends it to the array",
    );
    // The score matrix attention allocates per window, in MB. It is the one
    // host cost that scales with the model's WIDTH as well as its length, and
    // it is why attention is a host pass at this size.
    let score_mb = g.max_seq as f64 * g.heads as f64 * g.max_seq as f64 * 4.0 / (1024.0 * 1024.0);
    row(
        "attention",
        if places.attn { "npu" } else { "host" },
        &format!(
            "one score row at a time: {} x {} x {} = {} MB at full length",
            g.max_seq, g.heads, g.max_seq, score_mb as i64
        ),
    );
    row(
        "logits + argmax",
        if places.logit { "npu" } else { "host" },
        &format!(
            "tied embedding, {} x {} on the last row only",
            g.vocab, g.d_model
        ),
    );
    row(
        "decode + merge",
        "host",
        "the container's own BPE table; the window merge is transformers' longest-common-sequence over ids",
    );
}

/// One row of the ops table: a name, a place, and the reason.
fn row(name: &str, place: &str, why: &str) {
    eprintln!("             {name:<26} {place:<5} {why}");
}

/// The lines that follow the ops table: the setup, the language, the task
/// and the windowing.
///
/// They are the same in the CLI and in the server: a server that described
/// its windows differently from the command line would be two runtimes
/// wearing one name.
///
/// The `chunks` and the `time` lines are NOT here: they need a request,
/// and the CLI's are printed after the work.
pub fn print_schedule(
    setup_s: f64,
    options: &npum_cpu::whisper_session::Options,
    language_given: bool,
) {
    // The wording is the reference's, and the reference's line names what the reference's
    // setup did: it opened the device, loaded two design sets and staged the
    // weights on it. This runtime reads the design sets and the weights and
    // runs the arithmetic on the host, so the numbers are the host's.
    eprintln!("  setup      {setup_s:.2} s (device, two design sets, weights staged on it)");
    eprintln!(
        "  language   {}{}",
        options.language,
        if language_given {
            ""
        } else {
            "   ASSUMED: this build does not detect languages, so --language defaults to en and says so here"
        }
    );
    eprintln!("  task       {}", options.task);
    // The step between window starts is printed only when it is a real
    // step: `--chunk-seconds 5 --stride-seconds 5` is two strides across
    // one window, and a status block that printed "windows start -5 s
    // apart" before the refusal landed is a line nobody can act on.
    let step_s = options.chunk_seconds - 2 * options.stride_seconds;
    if step_s > 0 {
        eprintln!(
            "  long form  {} s windows, {} s stride each side, so windows start {} s apart",
            options.chunk_seconds, options.stride_seconds, step_s
        );
    } else {
        eprintln!(
            "  long form  {} s windows, {} s stride each side",
            options.chunk_seconds, options.stride_seconds
        );
    }
}

/// The context budget, checked BEFORE anything is built.
///
/// Two design sets are the floor, and every elementwise op on the array is
/// one more xclbin and one more hw_context: the npu1 driver measured six,
/// so a run that asks for four elementwise ops on top of the two GEMM sets
/// cannot be served and has to be told so while there is still nothing to
/// unwind.
///
/// False when the run must not go on.
pub fn check_context_budget(args: &crate::flags::Args) -> bool {
    // Two GEMM sets are the floor, and every elementwise pass the flag names
    // is one more xclbin and one more hw_context. Count what this run will
    // actually open, so a five-context run cannot pass a two-context budget.
    let mut contexts = 0usize;
    let dense = crate::npu::wants_op(args, "gemm") || crate::npu::wants_op(args, "attn");
    if dense {
        contexts += 2;
    } else if crate::npu::wants_op(args, "fft") || crate::npu::wants_op(args, "mproj") {
        // The front end lives in the ENCODER's set; with no dense work asked
        // for, that is the only context this run opens.
        contexts += 1;
    }
    for code in ["layn", "gelu", "softm"] {
        if crate::npu::wants_op(args, code) {
            contexts += 1;
        }
    }
    // A host run still checks the floor: `serve` calls this before it knows
    // a request's codes, and the check there is a probe of the BOARD.
    if contexts == 0 {
        contexts = 2;
    }
    let allow_contention = args.switch("--allow-contention");
    let report = npum_xdna::contention::survey_contexts();
    if npum_xdna::contention::require_context_budget(&report, contexts, allow_contention) {
        return true;
    }
    eprintln!(
        "error: NPU context budget: refusing to load {contexts} concurrent hw_context(s) -- see the report above (close the other process, or pass --allow-contention)"
    );
    false
}

/// The options one request runs with, from the command line.
///
/// The defaults are the reference's: English, transcribe, 30 s windows with a 5 s
/// stride on each side.
pub fn options_from(args: &crate::flags::Args) -> npum_cpu::whisper_session::Options {
    let default = npum_cpu::whisper_session::Options::default();
    npum_cpu::whisper_session::Options {
        language: args.value("--language").unwrap_or("en").to_string(),
        task: args.value("--task").unwrap_or("transcribe").to_string(),
        chunk_seconds: args
            .value_i64("--chunk-seconds")
            .unwrap_or(default.chunk_seconds),
        stride_seconds: args
            .value_i64("--stride-seconds")
            .unwrap_or(default.stride_seconds),
        // The flag table and the help both spell this `--max-new`. Reading
        // `--max-new-tokens` silently ignored the user's cap.
        max_new: args.value_i64("--max-new").unwrap_or(0).max(0) as usize,
    }
}

/// The design set's directory for a model, if there is one.
///
/// A speech model needs TWO sets -- `gemm_rtp` for the encoder stack and
/// `gemm_rtp_dec` for the decoder's seven streams -- and a directory with
/// only one of them is not a design set this model can run on.
pub fn find_designs(root: &Path, model_name: &str, named: Option<&str>) -> Option<PathBuf> {
    // The search is shared with every family, including `--artifacts`; a
    // speech model still needs BOTH sets before the directory is usable.
    crate::npu::find_artifacts(root, model_name, named).filter(|dir| {
        dir.join("gemm_rtp").join("design.json").is_file()
            && dir.join("gemm_rtp_dec").join("design.json").is_file()
    })
}

/// Print the family's status block, before any refusal the work would give.
///
/// The block is a description of the run the flag asked for. Printing it
/// after the `--npu-ops` gate would mean a refused command said nothing about
/// which container and which designs it was refused for.
pub fn announce(container: &npum_format::Container, path: &Path, args: &Args, root: &Path) {
    let Some(geometry) = read_geometry(container, path) else {
        return;
    };
    let designs = find_designs(root, &geometry.model_name, args.value("--artifacts"))
        .unwrap_or_else(|| {
            root.join("runtime")
                .join("artifacts")
                .join(&geometry.model_name)
        });
    let encoder = read_design(&designs, false);
    let decoder = read_design(&designs, true);
    let places = places(
        container,
        path,
        root,
        args,
        encoder.as_ref(),
        decoder.as_ref(),
    );
    print_block(
        &geometry,
        &designs,
        encoder.as_ref(),
        decoder.as_ref(),
        &places,
    );
}

/// Transcribe one file, and print the answers.
///
/// The order of the stderr is the reference's: the context budget BEFORE
/// anything is built, the block before the work, and the chunks and the
/// time after it.
pub fn run(
    container: &npum_format::Container,
    path: &Path,
    audio: &str,
    args: &crate::flags::Args,
    root: &Path,
) -> i32 {
    // The context budget, BEFORE the session builds anything. Two design
    // sets are the floor here, and every elementwise op on the array is one
    // more xclbin and one more hw_context: the npu1 driver measured six, so
    // a run that asks for four elementwise ops on top of the two GEMM sets
    // cannot be served and has to be told so while there is still nothing
    // to unwind.
    if !check_context_budget(args) {
        return crate::EXIT_REFUSED;
    }

    let Some(geometry) = read_geometry(container, path) else {
        eprintln!(
            "error: {}: the container does not carry the speech family's geometry",
            path.display()
        );
        return crate::EXIT_REFUSED;
    };
    // THE ARRAY, OR THE HOST. The flag is a question about where the work
    // happens, and the answer is the same one the status block just gave.
    // Whisper has TWO design sets: the encoder's `gemm_rtp` and the decoder's
    // `gemm_rtp_dec`. Both are opened when the flag asks for dense work,
    // because a decoder left on the host while the command line says "array"
    // would be a run that answered the flag with a lie.
    let dense = crate::npu::wants_op(args, "gemm")
        || crate::npu::wants_op(args, "attn")
        || crate::npu::wants_op(args, "fft")
        || crate::npu::wants_op(args, "mproj")
        || crate::npu::wants_op(args, "logit")
        || crate::npu::wants_op(args, "conv");
    let decode_dense = crate::npu::wants_op(args, "gemm")
        || crate::npu::wants_op(args, "attn")
        || crate::npu::wants_op(args, "logit");
    let mut array_enc: Option<Box<dyn npum_model::array::ArrayText>> = None;
    let mut array_dec: Option<Box<dyn npum_model::array::ArrayText>> = None;
    if dense {
        array_enc = match crate::npu::open_array_text_specs(
            container,
            path,
            root,
            args,
            encoder_specs(&geometry),
            "gemm_rtp",
        ) {
            Ok(a) => Some(a as Box<dyn npum_model::array::ArrayText>),
            Err(reason) => {
                eprintln!("error: {reason}");
                return crate::EXIT_REFUSED;
            }
        };
        if decode_dense {
            array_dec = match crate::npu::open_array_text_specs(
                container,
                path,
                root,
                args,
                decoder_specs(&geometry),
                "gemm_rtp_dec",
            ) {
                Ok(a) => Some(a as Box<dyn npum_model::array::ArrayText>),
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            };
        }
    }
    // The elementwise passes are separate designs and separate contexts: only
    // the ones the flag named are opened, and a pass without a design stays on
    // the host (and is named in the block above).
    let mut eltwise = match crate::npu::open_eltwise(container, path, root, args) {
        Ok(backend) => backend.map(|e| e as Box<dyn npum_model::array::ArrayEltwise>),
        Err(reason) => {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
    };

    let started = std::time::Instant::now();
    let encoder = match npum_cpu::whisper::Encoder::open(path) {
        Ok(e) => e,
        Err(reason) => {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
    };
    let mut decoder = match npum_cpu::whisper::Decoder::open(path) {
        Ok(d) => d,
        Err(reason) => {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
    };
    let table = match container.tensor_bytes("tokenizer.whisper_table") {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {}: {e}", path.display());
            return crate::EXIT_REFUSED;
        }
    };
    let tokenizer = match npum_cpu::whisper_tokenizer::Tokenizer::from_table(table) {
        Ok(t) => t,
        Err(reason) => {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
    };
    let weights = npum_cpu::whisper_session::FrontEndWeights {
        conv1_weight: match tensor_f32(&container, "frontend.conv1.weight") {
            Ok(w) => w,
            Err(reason) => {
                eprintln!("error: {reason}");
                return crate::EXIT_REFUSED;
            }
        },
        conv1_bias: match tensor_f32(&container, "frontend.conv1.bias") {
            Ok(w) => w,
            Err(reason) => {
                eprintln!("error: {reason}");
                return crate::EXIT_REFUSED;
            }
        },
        conv2_weight: match tensor_f32(&container, "frontend.conv2.weight") {
            Ok(w) => w,
            Err(reason) => {
                eprintln!("error: {reason}");
                return crate::EXIT_REFUSED;
            }
        },
        conv2_bias: match tensor_f32(&container, "frontend.conv2.bias") {
            Ok(w) => w,
            Err(reason) => {
                eprintln!("error: {reason}");
                return crate::EXIT_REFUSED;
            }
        },
    };
    let setup_s = started.elapsed().as_secs_f64();

    let options = options_from(args);
    print_schedule(setup_s, &options, args.value("--language").is_some());

    let samples = match npum_cpu::whisper_features::read_wav(Path::new(audio)) {
        Ok(a) => a.samples,
        Err(refusal) => {
            eprintln!("error: {}", refusal.0);
            return crate::EXIT_REFUSED;
        }
    };
    let mut transcript = match npum_cpu::whisper_session::transcribe_samples(
        &encoder,
        &mut decoder,
        &tokenizer,
        &samples,
        &options,
        &weights,
        array_enc.as_deref_mut(),
        array_dec.as_deref_mut(),
        eltwise.as_deref_mut(),
    ) {
        Ok(t) => t,
        Err(reason) => {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
    };

    // The dispatch count is the ARRAY's own, not an estimate: a count of zero
    // on a run whose flag asked for array work would be the lie the flag
    // exists to prevent. Each handle reports the calls it actually made.
    transcript.n_dispatches = (array_enc
        .as_ref()
        .map(|a| {
            a.dispatches() + a.attn_dispatches() + a.frontend_dispatches()
        })
        .unwrap_or(0)
        + array_dec
            .as_ref()
            .map(|a| a.dispatches() + a.attn_dispatches())
            .unwrap_or(0)
        + eltwise.as_ref().map(|e| e.dispatches()).unwrap_or(0))
        as i64;

    eprintln!(
        "  chunks     {} of {:.1} s of audio",
        transcript.n_chunks,
        transcript.n_samples as f64 / npum_cpu::whisper_session::SAMPLE_RATE as f64
    );
    eprintln!(
        "  time       mel {:.2} s, conv {:.2} s, encoder {:.2} s, decoder {:.2} s, {} dispatches",
        transcript.mel_seconds,
        transcript.conv_seconds,
        transcript.encoder_seconds,
        transcript.decoder_seconds,
        transcript.n_dispatches
    );
    if transcript.n_chunks > 1 {
        eprintln!("  windows    (each on its own, before the merge):");
        for segment in &transcript.segments {
            eprintln!(
                "    {:6.1} s - {:6.1} s  {}",
                segment.start_s, segment.end_s, segment.text
            );
        }
    }
    if args.switch("--json") {
        print!("{}", transcript_json(&transcript, true));
    } else {
        println!("{}", transcript.text);
    }
    0
}

/// The OpenAI-shaped answer, the same object the server returns.
///
/// The server's own function: the CLI and the server must not drift, and
/// two implementations of one JSON shape is how they drift.
pub fn transcript_json(t: &npum_cpu::whisper_session::Transcript, verbose: bool) -> String {
    npuembeddings_server::routes::transcript_json(t, verbose)
}

/// A tensor's values.
fn tensor_f32(container: &npum_format::Container, name: &str) -> Result<Vec<f32>, String> {
    let bytes = container
        .tensor_bytes(name)
        .map_err(|_| format!("the container holds no tensor named {name}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn command(args: &Args, root: &Path) -> crate::families::Outcome {
    let Some((model, audio)) = args.positional.first().zip(args.positional.get(1)) else {
        return crate::needs_a_model("transcribe");
    };
    #[rustfmt::skip]
    let outcome = crate::families::with_model_open(model, npum_families::Family::Stt, root, |path, arch, container| {
        if !Path::new(audio).exists() {
            eprintln!("error: the audio file {audio} does not exist.");
            return crate::EXIT_REFUSED;
        }
        // The block comes BEFORE the refusals: it says which container and
        // which design sets a refused command was refused for. The places it
        // prints come from the flag AND the designs that were found, so a
        // request for a pass with no design says "host" and is then refused
        // by the loader with the missing set's name.
        crate::stt::announce(container, path, args, root);
        if let Some(reason) = crate::backend::refuse_if_vulkan(args, npum_families::Family::Stt) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        if let Some(reason) =
            crate::families::refuse_op_not_in_arch(args, npum_families::Family::Stt)
        {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        // Five of the eight codes are now real seams: the encoder and the
        // decoder GEMMs, the attention, and the LayerNorm/GELU/softmax passes.
        // The rest still name operations with no design or no seam, and are
        // refused BY NAME rather than accepted and computed on the host.
        if let Some(reason) = crate::families::refuse_array_not_ported(
            npum_families::Family::Stt,
            args,
            &["gemm", "attn", "fft", "mproj", "layn", "gelu", "softm", "logit", "conv"],
        ) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        let _ = arch;
        crate::stt::run(container, path, audio, args, root)
    });
    outcome
}
