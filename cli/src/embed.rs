//! `embed`: the text-to-vector path.
//!
//! # Where the sequence length comes from, and why it is not the container's
//!
//! The container says how many positions it can feed -- 256 for
//! all-MiniLM-L6-v2 -- and the DESIGN SET says how many it was compiled for,
//! which is 64. The C++ runtime takes the DESIGN's number, and the
//! difference is not academic: a text of a hundred tokens is REFUSED at 64
//! and answered at 256, and the vector of the first sixty-four tokens is
//! not the vector of the text.
//!
//! So the length is looked up in the same places the C++ looks: the design
//! set on disk, then the design the container carries, and only then the
//! container's own limit -- which is the last resort, because a CPU-only
//! run has no reason to need a design set and must not fail without one.
//!
//! # The batching
//!
//! The C++ encodes in tiers of at most sixteen sequences, because that is
//! what its design sets were built for. The Rust encodes in chunks of
//! sixteen for the same reason -- the memory -- and NOT because the numbers
//! depend on it: every row of the forward pass is independent of every other
//! row, so a batch of four and a batch of sixteen give the same vectors.
//! What the batch DOES decide is the width of the score matrix, and that
//! width is the sequence length, not the batch.

use crate::flags::Args;
use npum_cpu::bert::BertModel;
use npum_tokenizers::wordpiece::WordPiece;
use std::path::{Path, PathBuf};

/// The result of a refusal, and the message that goes with it.
pub struct Refusal {
    /// The exit code.
    pub code: i32,
    /// The message, without a trailing newline.
    pub message: String,
}

/// Encode the texts and write the vectors.
///
/// `texts` are the lines of the input file, in order, empty ones included:
/// the C++ pushes every line it reads, and a line that is empty is a text
/// that embeds to something.
pub fn run(
    path: &Path,
    arch: &str,
    texts: &[String],
    prefix: Option<&str>,
    out_path: Option<&str>,
    allow_truncation: bool,
    root: &Path,
    args: &crate::flags::Args,
) -> Result<Vec<f32>, Refusal> {
    let container = npum_format::Container::open(path).map_err(|e| Refusal {
        code: crate::EXIT_REFUSED,
        message: format!("{}: {e}", path.display()),
    })?;
    let tokenizer = load_tokenizer(&container, arch).map_err(|e| Refusal {
        code: crate::EXIT_REFUSED,
        message: format!("{}: {e}", path.display()),
    })?;
    let model = BertModel::open(path).map_err(|e| Refusal {
        code: crate::EXIT_REFUSED,
        message: format!("{}: {e}", path.display()),
    })?;

    // THE ARRAY, OR THE HOST.
    //
    // The flag is a question about where the run happens, and it is
    // answered here and nowhere else: the projections go to the array, the
    // elementwise passes stay on the host, and the run says how many it
    // dispatched.
    let mut array: Option<Box<dyn npum_model::array::ArrayText>> =
        if wants_gemm(args) || wants_attn(args) {
            Some(open_array(path, &model, root, args).map_err(|reason| Refusal {
                code: crate::EXIT_REFUSED,
                message: reason,
            })?)
        } else {
            None
        };
    // THE ELEMENTWISE PASSES, AND ONLY THE ONES THE FLAG NAMED.
    //
    // Each pass is its own design and its own context, and a design that
    // was not opened is a pass that stays on the host.
    let mut eltwise: Option<Box<dyn npum_model::array::ArrayEltwise>> =
        match open_eltwise(path, root, args) {
            Ok(backend) => backend,
            Err(reason) => {
                return Err(Refusal {
                    code: crate::EXIT_REFUSED,
                    message: reason,
                })
            }
        };

    // The status block was printed by the caller, before any refusal this
    // run might give: the banner says WHICH container, and a refusal on
    // its own does not.
    let artifacts = find_artifacts(root, &container_name(path));
    let seq = sequence_length(path, &container, root, artifacts.as_deref());
    let hidden = model.spec.hidden;

    // The vocabulary's size, and then the texts. The C++'s order, and the
    // reason for it: the tokenizer is loaded here and not before.
    crate::status::print_tokenizer(
        npum_tokenizers::Tokenizer::vocab_size(&tokenizer),
    );
    println!("  input      {} texts", texts.len());
    let started = std::time::Instant::now();
    let mut timings = npum_cpu::bert::Timings::default();
    let mut vectors: Vec<f32> = Vec::with_capacity(texts.len() * hidden);
    for (index, text) in texts.iter().enumerate() {
        let raw = match prefix {
            Some(p) => format!("{p}{text}"),
            None => text.clone(),
        };
        let (encoding, truncated) = tokenizer.encode_padded(&raw, seq).map_err(|e| Refusal {
            code: crate::EXIT_REFUSED,
            message: e.to_string(),
        })?;
        if truncated && !allow_truncation {
            // The C++'s refusal, word for word. The token count is the
            // UNTRUNCATED one, including [CLS] and [SEP], because the
            // reader's question is "how long is my text" and not "how long
            // is the part that fitted".
            let n_tokens = encoding.ids.len().min(seq) + 2;
            return Err(Refusal {
                code: crate::EXIT_REFUSED,
                message: format!(
                    "input {index} is {n_tokens} tokens, but this runtime is running at sequence length {seq}. Split the text into shorter pieces, or run a design exported for a longer sequence (tools/export/export_gemm_rtp.py --seq N). Passing --allow-truncation cuts it at {seq} tokens instead and DISCARDS the rest."
                ),
            });
        }
        let chunk = model
            .embed_timed(
                &mut timings,
                &encoding.ids,
                &encoding.mask,
                1,
                seq,
                array.as_deref_mut(),
                eltwise.as_deref_mut(),
            )
            .map_err(|e| Refusal {
                code: crate::EXIT_REFUSED,
                message: e.to_string(),
            })?;
        vectors.extend_from_slice(&chunk);
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "  embedded   {} texts in {:.2} s  ->  {:.1} seq/s",
        texts.len(),
        elapsed,
        texts.len() as f64 / elapsed
    );
    let dispatches = array.as_ref().map(|a| a.dispatches()).unwrap_or(0)
        + array.as_ref().map(|a| a.attn_dispatches()).unwrap_or(0)
        + eltwise.as_ref().map(|e| e.dispatches()).unwrap_or(0);
    crate::status::print_time_split(&timings, 4, elapsed, dispatches);

    if let Some(out) = out_path {
        let mut bytes = Vec::with_capacity(vectors.len() * 4);
        for v in &vectors {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(out, &bytes).map_err(|e| Refusal {
            code: crate::EXIT_REFUSED,
            message: format!("failed writing {out}: {e}"),
        })?;
        println!(
            "  wrote      {out}  [{}, {hidden}] fp32",
            vectors.len() / hidden
        );
    } else {
        // The C++'s rendering: the first four vectors, the first six
        // components of each, and an ellipsis.
        for b in 0..texts.len().min(4) {
            print!("  [{b}]");
            for c in 0..6 {
                print!(" {:+.4}", vectors[b * hidden + c]);
            }
            println!(" ...");
        }
    }
    Ok(vectors)
}

/// The tokenizer a container holds.
///
/// The ARCH decides, not the file name: a container that says it is a BERT
/// model has a BERT vocabulary, whatever it is called on disk.
fn load_tokenizer(
    container: &npum_format::Container,
    arch: &str,
) -> Result<WordPiece, String> {
    if arch.contains("whisper") || arch.contains("gemma") || arch.contains("gte_new_rope") {
        return Err(format!(
            "the {arch} family has its own tokenizer, and this command does not \
             run it yet"
        ));
    }
    let blob = container
        .tensor_bytes("tokenizer.vocab")
        .map_err(|_| "the container holds no vocabulary".to_string())?;
    let text = String::from_utf8_lossy(blob);
    WordPiece::from_lines(&text).map_err(|e| format!("the vocabulary is unusable: {e}"))
}

/// The sequence length this model's design set was built for.
pub fn sequence_length(
    path: &Path,
    container: &npum_format::Container,
    root: &Path,
    artifacts: Option<&Path>,
) -> usize {
    // 1. The design set on disk.
    if let Some(artifacts) = artifacts {
        if let Some(seq) = read_seq(&artifacts.join("gemm_rtp").join("design.json")) {
            return seq;
        }
    }
    let name = path.file_stem().map(|s| s.to_string_lossy().to_string());
    if let Some(name) = &name {
        for base in [
            root.join("runtime").join("artifacts"),
            root.join("artifacts"),
            root.join("runtime"),
            root.to_path_buf(),
        ] {
            for sub in ["", "artifacts_npu1", "artifacts_npu2", "npu1", "npu2"] {
                let dir = if sub.is_empty() {
                    base.join(name)
                } else {
                    base.join(name).join(sub)
                };
                if let Some(seq) = read_seq(&dir.join("gemm_rtp").join("design.json")) {
                    return seq;
                }
            }
        }
    }
    // 2. The design the container carries.
    for name in ["design/gemm_rtp/design.json", "design/gemm_rtp_dec/design.json"] {
        if let Some(bytes) = container.tensor_bytes(name).ok() {
            if let Some(seq) = read_seq_bytes(bytes) {
                return seq;
            }
        }
    }
    // 3. The container's own limit.
    container
        .config()
        .get_i64("max_seq_len")
        .map(|v| v as usize)
        .unwrap_or(64)
}

/// Read the sequence length out of a design set's `design.json`.
fn read_seq(path: &Path) -> Option<usize> {
    let bytes = std::fs::read(path).ok()?;
    read_seq_bytes(&bytes)
}

/// Read the sequence length out of a design set's `design.json`.
fn read_seq_bytes(bytes: &[u8]) -> Option<usize> {
    let value: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    value.get("seq")?.as_u64().map(|v| v as usize)
}

/// The texts of an input file.
///
/// Every line is a text, empty ones included, and a trailing `\r` is
/// dropped: the C++ does exactly this, and a file written on Windows would
/// otherwise embed a `\r` into every text.
pub fn read_texts(path: &str) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot open {path}: {e}"))?;
    Ok(text
        .lines()
        .map(|l| l.strip_suffix('\r').unwrap_or(l).to_string())
        .collect())
}

/// Whether the arguments ask for the output to be written to a file.
pub fn wants_out_file(args: &Args) -> Option<String> {
    args.positional.get(2).cloned()
}

/// The design set's directory, if there is one.
///
/// The C++'s order: the root's `runtime/artifacts/<model>/artifacts_npu1`
/// first, because that is where the exporter puts it and where the status
/// block says it is.
pub fn find_artifacts(root: &Path, model_name: &str) -> Option<PathBuf> {
    if model_name.is_empty() {
        return None;
    }
    for base in [
        root.join("runtime").join("artifacts"),
        root.join("artifacts"),
        root.join("runtime"),
        root.to_path_buf(),
    ] {
        for sub in ["artifacts_npu1", "artifacts_npu2", "", "npu1", "npu2"] {
            let dir = if sub.is_empty() {
                base.join(model_name)
            } else {
                base.join(model_name).join(sub)
            };
            if dir.join("gemm_rtp").join("design.json").is_file() {
                return Some(dir);
            }
        }
    }
    None
}

/// Whether the run asked for the LayerNorm to go on the array.
pub fn wants_layernorm(args: &crate::flags::Args) -> bool {
    wants_op(args, "layn")
}

/// Whether the run asked for the GELU to go on the array.
pub fn wants_gelu(args: &crate::flags::Args) -> bool {
    wants_op(args, "gelu")
}

/// Whether the run asked for the softmax to go on the array.
pub fn wants_softmax(args: &crate::flags::Args) -> bool {
    wants_op(args, "softm")
}

/// Whether the run asked for the attention to go on the array.
///
/// The attention is two GEMMs -- `attn_qk` and `attn_av` -- and they are
/// streams of the projections' OWN design set, so asking for them costs no
/// second context and no second design set.
pub fn wants_attn(args: &crate::flags::Args) -> bool {
    wants_op(args, "attn")
}

/// Where the elementwise passes run, read off the flag and the design sets.
///
/// The status block is printed BEFORE the work, and it says where each
/// operation runs -- so the answer is read off the design sets' manifests
/// and not off the board: a manifest holds the design's name, its
/// generation and its device, and reading one does not need a board.
pub fn places(path: &Path, root: &Path, args: &Args) -> crate::status::Places {
    let mut places = crate::status::Places::default();
    if wants_layernorm(args) {
        places.layn = op_place(path, root, "layernorm");
    }
    if wants_gelu(args) {
        places.gelu = op_place(path, root, "gelu");
    }
    if wants_softmax(args) {
        places.softm = op_place(path, root, "softmax");
    }
    if wants_attn(args) {
        // The attention's streams are inside the projections' set, so the
        // place it names is that set's.
        places.attn = op_place(path, root, "gemm_rtp");
    }
    places.staged_bytes = staged_bytes(path, root, args);
    places
}

/// How many bytes of weights the array run stages on the device, once.
///
/// TWO THINGS ARE STAGED, and both are counted before the run and not
/// after it:
///
/// * the PANELS -- the projections' weights, already rounded to bf16 and
///   already tiled into the MMAC's byte order. They are the container's
///   own tensors, so their sizes are the container's;
/// * the LAYERNORM'S PARAMETERS -- one slot per site, and a site is the
///   embedding plus two per layer.
///
/// The number is the run's, and the block reports it before the run
/// because the block is a description of the run rather than a report on
/// it -- and a block that reports the intention rather than the value is
/// this project's recurring fail-open.
pub fn staged_bytes(path: &Path, root: &Path, args: &Args) -> usize {
    let mut bytes = 0usize;
    if wants_gemm(args) {
        bytes += panel_bytes(path);
    }
    if wants_layernorm(args) {
        if let (Some(cols), Some(layers)) = (design_cols(path, root, "layernorm"), layers(path)) {
            // gamma and beta, f32, two of them per row.
            bytes += (1 + 2 * layers) * 2 * cols * 4;
        }
    }
    bytes
}

/// The projections' panels, as the container holds them.
fn panel_bytes(path: &Path) -> usize {
    let Ok(container) = npum_format::Container::open(path) else {
        return 0;
    };
    container
        .directory()
        .tensors()
        .iter()
        .filter(|t| t.role == "gemm_b")
        .map(|t| t.nbytes as usize)
        .sum()
}

/// One design set's row width, from its own manifest.
fn design_cols(path: &Path, root: &Path, set: &str) -> Option<usize> {
    let value = design_manifest(path, root, set)?;
    value.get("cols")?.as_u64().map(|v| v as usize)
}

/// How many transformer layers the container's model has.
fn layers(path: &Path) -> Option<usize> {
    let container = npum_format::Container::open(path).ok()?;
    container.config().get_i64("num_layers").map(|v| v as usize)
}

/// One design set's place, from its own manifest.
fn op_place(path: &Path, root: &Path, set: &str) -> Option<crate::status::OpPlace> {
    let value = design_manifest(path, root, set)?;
    Some(crate::status::OpPlace {
        name: value
            .get("name")
            .and_then(|v| v.as_str())
            .filter(|n| !n.is_empty())
            .unwrap_or(set)
            .to_string(),
        arch: value.get("arch").and_then(|v| v.as_i64()).unwrap_or(0),
        device: value
            .get("device")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
    })
}

/// One design set's manifest, from the container or from the tree.
///
/// The container is the PREFERRED source and the directory is the
/// fallback, and the two are read the same way the design itself is.
fn design_manifest(path: &Path, root: &Path, set: &str) -> Option<serde_json::Value> {
    if let Ok(container) = npum_format::Container::open(path) {
        if let Some(bytes) = container
            .tensor_bytes(&format!("design/{set}/design.json"))
            .ok()
        {
            if let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) {
                return Some(value);
            }
        }
    }
    let dir = find_artifacts(root, &container_name(path))?;
    let bytes = std::fs::read(dir.join(set).join("design.json")).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Whether the run asked for one operation to go on the array.
///
/// The flag is a SET of codes, and a code is asked for by NAME.
fn wants_op(args: &crate::flags::Args, code: &str) -> bool {
    args.value("--npu-ops")
        .map(|v| v.split(',').any(|c| c.trim() == code))
        .unwrap_or(false)
}

/// The array's elementwise passes, when the run asked for them.
///
/// The three passes are three designs and three contexts, and only the ones
/// the flag named are opened: a design that was not opened is a pass that
/// stays on the host.
pub fn open_eltwise(
    path: &Path,
    root: &Path,
    args: &crate::flags::Args,
) -> Result<Option<Box<dyn npum_model::array::ArrayEltwise>>, String> {
    use npum_xdna::eltwise::{Eltwise, EltwiseDesign};

    let mut layernorm = None;
    if wants_layernorm(args) {
        let design = open_design(path, root, "layernorm", "layn")?;
        layernorm = Some(EltwiseDesign::open(design).map_err(|e| e.to_string())?);
    }
    let mut gelu = None;
    if wants_gelu(args) {
        let design = open_design(path, root, "gelu", "gelu")?;
        gelu = Some(EltwiseDesign::open(design).map_err(|e| e.to_string())?);
    }
    let mut softmax = None;
    if wants_softmax(args) {
        let design = open_design(path, root, "softmax", "softm")?;
        softmax = Some(EltwiseDesign::open(design).map_err(|e| e.to_string())?);
    }
    if layernorm.is_none() && gelu.is_none() && softmax.is_none() {
        return Ok(None);
    }
    let backend = Eltwise::new(layernorm, gelu, softmax).map_err(|e| e.to_string())?;
    Ok(Some(Box::new(backend)))
}

/// The array's design set, from the container's own copy or from the tree.
///
/// The container is the PREFERRED source and the directory is the
/// fallback: a container that carries its own design set needs nothing
/// beside it -- no directory, no MLIR-AIE on the reader's machine, no
/// artifacts to install.
fn open_design(
    path: &Path,
    root: &Path,
    set: &str,
    code: &str,
) -> Result<npum_xdna::array::ArrayDesign<crate::transport::Transport>, String> {
    use npum_xdna::array::ArrayDesign;

    let container =
        npum_format::Container::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let own = container
        .directory()
        .tensor(&format!("design/{set}/design.json"))
        .is_some();
    let transport = crate::transport::open()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if own {
        ArrayDesign::open_from_container(path, set, transport).map_err(|e| e.to_string())
    } else {
        let dir = find_artifacts(root, &container_name(path)).ok_or_else(|| {
            format!(
                "{}: --npu-ops {code} needs the {set} design set: this container does not carry \
                 one, and no design set was found for it. Pass --artifacts <dir> naming one.",
                path.display()
            )
        })?;
        ArrayDesign::open(&dir.join(set), transport).map_err(|e| e.to_string())
    }
}

/// Whether the run asked for the projections to go on the array.
///
/// The flag is a SET of codes, and this family honours exactly one of
/// them: `gemm`. Anything else is refused by the family's own check.
pub fn wants_gemm(args: &crate::flags::Args) -> bool {
    args.value("--npu-ops")
        .map(|v| v.split(',').any(|c| c.trim() == "gemm"))
        .unwrap_or(false)
}

/// The array's backend for the text family, when the run asked for it.
///
/// # THE TWO THINGS THE ARRAY NEEDS ARE SEPARATE
///
/// * the PANELS -- the container's weights as the array's operands: bf16,
///   already tiled into the MMAC's byte order. A container packed for the
///   host keeps its weights in fp32 and has no panel at all, and the
///   refusal says how to repack it.
///
/// * the DESIGN SET -- the compiled core. The container's own copy is
///   preferred, because a container that carries it needs nothing beside
///   it; a directory is the fallback.
///
/// # ONE DESIGN, ONE CONTEXT
///
/// The attention's two GEMMs are streams of the projections' OWN set --
/// `attn_qk` and `attn_av` -- and not a set of their own. So asking for the
/// attention asks for the same design set, and the run holds ONE hardware
/// context for the whole of the text family's dense work.
pub fn open_array(
    path: &Path,
    model: &BertModel,
    root: &Path,
    args: &crate::flags::Args,
) -> Result<Box<dyn npum_model::array::ArrayText>, String> {
    use npum_xdna::array::ArrayDesign;
    use npum_xdna::array_gemms::{ArrayGemms, GemmSpec};

    let container =
        npum_format::Container::open(path).map_err(|e| format!("{}: {e}", path.display()))?;

    let gemm = wants_gemm(args);
    let attn = wants_attn(args);

    // THE PANELS.
    //
    // Only the projections need them: the attention's operands are
    // ACTIVATIONS, built on the host, and a container packed for the host
    // has everything the attention needs.
    if gemm {
        let panel = container.directory().tensor("layer.0.qkv").ok_or_else(|| {
            format!(
                "{}: --npu-ops gemm needs the array's panels, and this container does not carry \
                 any: it has no layer.0.qkv. Repack it with the packer's --dtype bf16, which \
                 stores every projection's weights as the pre-tiled bf16 panel the array \
                 multiplies by.",
                path.display()
            )
        })?;
        if panel.dtype != "BF16" {
            return Err(format!(
                "{}: --npu-ops gemm needs the array's panels, and this container does not carry \
                 any: layer.0.qkv is {}, not BF16. Repack it with the packer's --dtype bf16, \
                 which stores every projection's weights as the pre-tiled bf16 panel the array \
                 multiplies by.",
                path.display(),
                panel.dtype
            ));
        }
    }

    // THE DESIGN.
    let own = container
        .directory()
        .tensor("design/gemm_rtp/design.json")
        .is_some();
    let transport = crate::transport::open()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let design = if own {
        ArrayDesign::open_from_container(path, "gemm_rtp", transport).map_err(|e| e.to_string())?
    } else {
        let dir = find_artifacts(root, &container_name(path)).ok_or_else(|| {
            format!(
                "{}: --npu-ops {} needs a design set: this container does not carry one, \
                 and no design set was found for it. Pass --artifacts <dir> naming one, or \
                 build it with the exporter. The flag has no default to fall back on, \
                 because this model's design is its own.",
                path.display(),
                if gemm { "gemm" } else { "attn" }
            )
        })?;
        ArrayDesign::open(&dir.join("gemm_rtp"), transport).map_err(|e| e.to_string())?
    };

    // THE PROJECTIONS: the four of every layer.
    let spec = &model.spec;
    let hidden = spec.hidden;
    let ffn = spec.ffn;
    let mut specs = Vec::with_capacity(spec.layers * 4);
    if gemm {
        for layer in 0..spec.layers {
            for (op, k, n) in [
                ("qkv", hidden, 3 * hidden),
                ("attn_out", hidden, hidden),
                ("ffn_up", hidden, ffn),
                ("ffn_down", ffn, hidden),
            ] {
                specs.push(GemmSpec {
                    weight: format!("layer.{layer}.{op}"),
                    op: op.to_string(),
                    k,
                    n,
                });
            }
        }
    }
    let backend = ArrayGemms::open(path, design, &specs, attn).map_err(|e| e.to_string())?;
    Ok(Box::new(backend))
}

/// A container's name, without the directory and without the extension.
fn container_name(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default()
}
