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

    // The status block: what this runtime says about itself before it works.
    let status = crate::status::read_container(&container, path);
    let artifacts = find_artifacts(root, &status.model_name);
    let design = artifacts.as_ref().and_then(|a| crate::status::read_design(a));
    crate::status::print_block(
        &status,
        design.as_ref(),
        artifacts.as_deref().unwrap_or(root),
    );

    let seq = sequence_length(path, &container, root, artifacts.as_deref());
    let hidden = model.spec.hidden;

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
            .embed_timed(&mut timings, &encoding.ids, &encoding.mask, 1, seq)
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
    crate::status::print_time_split(&timings, 4, elapsed);

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
