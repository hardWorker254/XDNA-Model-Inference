//! `embed`: the text-to-vector path.
//!
//! # Where the sequence length comes from, and why it is not the container's
//!
//! The container says how many positions it can feed -- 256 for
//! all-MiniLM-L6-v2 -- and the DESIGN SET says how many it was compiled for,
//! which is 64. The reference runtime takes the DESIGN's number, and the
//! difference is not academic: a text of a hundred tokens is REFUSED at 64
//! and answered at 256, and the vector of the first sixty-four tokens is
//! not the vector of the text.
//!
//! So the length is looked up in the same places the reference looks: the design
//! set on disk, then the design the container carries, and only then the
//! container's own limit -- which is the last resort, because a CPU-only
//! run has no reason to need a design set and must not fail without one.
//!
//! # The batching
//!
//! The reference encodes in tiers of at most sixteen sequences, because that is
//! what its design sets were built for. The Rust encodes in chunks of
//! sixteen for the same reason -- the memory -- and NOT because the numbers
//! depend on it: every row of the forward pass is independent of every other
//! row, so a batch of four and a batch of sixteen give the same vectors.
//! What the batch DOES decide is the width of the score matrix, and that
//! width is the sequence length, not the batch.

use crate::flags::Args;
use npum_cpu::bert::BertModel;
use npum_cpu::gte::GteModel;
use npum_tokenizers::wordpiece::WordPiece;
use npum_tokenizers::xlmr::XlmrTokenizer;
use std::path::{Path, PathBuf};

/// The reference encodes in tiers of at most sixteen sequences, and the NPU
/// designs were built for the same. Chunking the CLI's inputs this way lets
/// one forward pass amortise its dispatch, conversion and kernel-launch cost
/// over several texts instead of paying it once per text.
const BATCH: usize = 16;

/// A tokenizer for the embed/text family (WordPiece or XLM-R Unigram).
pub enum EmbedTokenizer {
    WordPiece(WordPiece),
    Xlmr(XlmrTokenizer),
}

impl EmbedTokenizer {
    pub fn vocab_size(&self) -> usize {
        match self {
            EmbedTokenizer::WordPiece(t) => npum_tokenizers::Tokenizer::vocab_size(t),
            EmbedTokenizer::Xlmr(t) => npum_tokenizers::Tokenizer::vocab_size(t),
        }
    }

    pub fn encode_padded(
        &self,
        text: &str,
        max_len: usize,
    ) -> npum_tokenizers::Result<(npum_tokenizers::Encoding, bool)> {
        match self {
            EmbedTokenizer::WordPiece(t) => t.encode_padded(text, max_len),
            EmbedTokenizer::Xlmr(t) => t.encode_padded(text, max_len),
        }
    }
}

/// The result of a refusal, and the message that goes with it.
#[derive(Debug)]
pub struct Refusal {
    /// The exit code.
    pub code: i32,
    /// The message, without a trailing newline.
    pub message: String,
}

/// Tokenise every text up front, at the design's sequence length.
///
/// Padding to `seq` is deliberate: a text that does not fit is either refused
/// here or cut here, exactly as before, so `--allow-truncation` and the
/// per-input refusal message keep their meaning. The returned per-text
/// lengths (the count of unmasked tokens) let a batch shrink its sequence
/// length later without ever dropping a real token.
fn tokenize_all(
    tokenizer: &EmbedTokenizer,
    texts: &[String],
    prefix: Option<&str>,
    seq: usize,
    allow_truncation: bool,
) -> Result<(Vec<u32>, Vec<u32>, Vec<usize>), Refusal> {
    let mut ids = Vec::with_capacity(texts.len() * seq);
    let mut mask = Vec::with_capacity(texts.len() * seq);
    let mut lengths = Vec::with_capacity(texts.len());
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
            let n_tokens = encoding.ids.len().min(seq) + 2;
            return Err(Refusal {
                code: crate::EXIT_REFUSED,
                message: format!(
                    "input {index} is {n_tokens} tokens, but this runtime is running at sequence length {seq}. Split the text into shorter pieces, or run a design exported for a longer sequence (tools/export/export_gemm_rtp.py --seq N). Passing --allow-truncation cuts it at {seq} tokens instead and DISCARDS the rest."
                ),
            });
        }
        lengths.push(encoding.mask.iter().filter(|&&m| m > 0).count());
        ids.extend_from_slice(&encoding.ids);
        mask.extend_from_slice(&encoding.mask);
    }
    Ok((ids, mask, lengths))
}

/// The sequence length one batch actually needs.
///
/// When the score width follows the sequence (host and GPU softmax, no array
/// design fixing it) the trailing all-padding columns can be dropped: every
/// real token keeps its original position, and the columns dropped were
/// masked out, so the vectors are unchanged. The win is largest for short
/// texts against a long design window -- attention is quadratic in `seq`.
///
/// `fix_width` is true when an array design fixes the score width (array
/// attention's `n_kv` or an array softmax's columns); then the full design
/// `seq` is kept and batching alone amortises the per-forward overhead.
fn batch_seq(design_seq: usize, lengths: &[usize], fix_width: bool) -> usize {
    if fix_width {
        return design_seq;
    }
    lengths
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(1)
        .min(design_seq)
}

/// Pack one chunk's rows into flat `ids`/`mask` at `eff_seq`.
///
/// Each text was padded to `design_seq` with padding at the tail, so taking
/// the first `eff_seq` entries keeps every real token (`eff_seq` is at least
/// the longest text in the chunk) and drops only padding.
fn pack_batch(
    ids: &[u32],
    mask: &[u32],
    design_seq: usize,
    begin: usize,
    batch: usize,
    eff_seq: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut out_ids = Vec::with_capacity(batch * eff_seq);
    let mut out_mask = Vec::with_capacity(batch * eff_seq);
    for i in 0..batch {
        let base = (begin + i) * design_seq;
        out_ids.extend_from_slice(&ids[base..base + eff_seq]);
        out_mask.extend_from_slice(&mask[base..base + eff_seq]);
    }
    (out_ids, out_mask)
}

/// Encode the texts and write the vectors.
///
/// `texts` are the lines of the input file, in order, empty ones included:
/// the reference pushes every line it reads, and a line that is empty is a text
/// that embeds to something.
pub fn run(
    container: &npum_format::Container,
    path: &Path,
    arch: &str,
    texts: &[String],
    prefix: Option<&str>,
    out_path: Option<&str>,
    allow_truncation: bool,
    root: &Path,
    args: &crate::flags::Args,
    mut ops: Box<dyn npum_model::Ops>,
) -> Result<Vec<f32>, Refusal> {
    let tokenizer = load_tokenizer(container, arch).map_err(|e| Refusal {
        code: crate::EXIT_REFUSED,
        message: format!("{}: {e}", path.display()),
    })?;

    // GTE has its own forward pass (RoPE, GEGLU, CLS pooling).
    if arch == "gte_new_rope_geglu" || (arch.contains("gte") && arch.contains("rope")) {
        let model = GteModel::from_container(container).map_err(|e| Refusal {
            code: crate::EXIT_REFUSED,
            message: format!("{}: {e}", path.display()),
        })?;

        // THE ARRAY, OR THE HOST. GTE's own forward pass now has the same
        // seams as BERT's: four projections per layer, the attention block
        // and the LayerNorm/GELU/softmax passes. As in BERT, the flag is the
        // only thing that opens a design; a design on disk is not a dispatch.
        let mut array: Option<Box<dyn npum_model::array::ArrayText>> = if wants_gemm(args)
            || wants_attn(args)
        {
            Some(
                open_array_gte(container, path, &model, root, args).map_err(|reason| Refusal {
                    code: crate::EXIT_REFUSED,
                    message: reason,
                })?,
            )
        } else {
            None
        };
        // ONE DESIGN, ONE CONTEXT per elementwise pass. Only the passes the
        // flag named are opened; a pass with no design stays on the host.
        let mut eltwise: Option<Box<dyn npum_model::array::ArrayEltwise>> =
            match open_eltwise(container, path, root, args) {
                Ok(backend) => backend,
                Err(reason) => {
                    return Err(Refusal {
                        code: crate::EXIT_REFUSED,
                        message: reason,
                    });
                }
            };

        let artifacts =
            crate::npu::find_artifacts(root, &container_name(path), args.value("--artifacts"));
        let seq = sequence_length(path, container, root, artifacts.as_deref());
        let hidden = model.spec.hidden;

        crate::status::print_tokenizer(tokenizer.vocab_size());
        println!("  input      {} texts", texts.len());
        let (all_ids, all_mask, lengths) =
            tokenize_all(&tokenizer, texts, prefix, seq, allow_truncation)?;
        let started = std::time::Instant::now();
        let mut timings = npum_cpu::gte::Timings::default();
        let mut vectors: Vec<f32> = Vec::with_capacity(texts.len() * hidden);
        // An array design fixes the score row's width (array attention's
        // padded key count or an array softmax's columns), so those runs keep
        // the full design sequence and rely on batching alone. Host and GPU
        // runs shrink `seq` per batch.
        let fix_width = array.is_some() || eltwise.is_some();
        let mut offset = 0usize;
        while offset < texts.len() {
            let batch = (texts.len() - offset).min(BATCH);
            let eff_seq = batch_seq(seq, &lengths[offset..offset + batch], fix_width);
            let (ids, mask) = pack_batch(&all_ids, &all_mask, seq, offset, batch, eff_seq);
            let chunk = model
                .embed_timed_with(
                    ops.as_mut(),
                    &mut timings,
                    &ids,
                    &mask,
                    batch,
                    eff_seq,
                    array.as_deref_mut(),
                    eltwise.as_deref_mut(),
                )
                .map_err(|e| Refusal {
                    code: crate::EXIT_REFUSED,
                    message: e.to_string(),
                })?;
            vectors.extend_from_slice(&chunk);
            offset += batch;
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
        let gpu = match ops.place() {
            npum_model::dispatcher::Placement::Vulkan => Some(crate::status::Gpu {
                device: ops.device().unwrap_or("the GPU").to_string(),
                dispatches: ops.dispatches(),
            }),
            _ => None,
        };
        let total = dispatches + gpu.as_ref().map(|g| g.dispatches).unwrap_or(0);
        // print_time_split expects bert timings; map gte timings fields
        let bt = npum_cpu::bert::Timings {
            host_layernorm: timings.host_layernorm,
            host_softmax: timings.host_softmax,
            host_gelu: timings.host_gelu,
            attention: timings.attention,
            qk: timings.qk,
            av: timings.av,
            array_gemm: timings.array_gemm,
            gpu: timings.gpu,
        };
        crate::status::print_time_split(&bt, 4, elapsed, total, gpu.as_ref());

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
            for b in 0..texts.len().min(4) {
                print!("  [{b}]");
                for c in 0..6 {
                    print!(" {:+.4}", vectors[b * hidden + c]);
                }
                println!(" ...");
            }
        }
        return Ok(vectors);
    }

    let model = BertModel::from_container(container).map_err(|e| Refusal {
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
            Some(
                open_array(container, path, &model, root, args).map_err(|reason| Refusal {
                    code: crate::EXIT_REFUSED,
                    message: reason,
                })?,
            )
        } else {
            None
        };
    // THE ELEMENTWISE PASSES, AND ONLY THE ONES THE FLAG NAMED.
    //
    // Each pass is its own design and its own context, and a design that
    // was not opened is a pass that stays on the host.
    let mut eltwise: Option<Box<dyn npum_model::array::ArrayEltwise>> =
        match open_eltwise(container, path, root, args) {
            Ok(backend) => backend,
            Err(reason) => {
                return Err(Refusal {
                    code: crate::EXIT_REFUSED,
                    message: reason,
                });
            }
        };

    // The status block was printed by the caller, before any refusal this
    // run might give: the banner says WHICH container, and a refusal on
    // its own does not.
    let artifacts =
        crate::npu::find_artifacts(root, &container_name(path), args.value("--artifacts"));
    let seq = sequence_length(path, container, root, artifacts.as_deref());
    let hidden = model.spec.hidden;

    // The vocabulary's size, and then the texts. The reference's order, and the
    // reason for it: the tokenizer is loaded here and not before.
    crate::status::print_tokenizer(tokenizer.vocab_size());
    println!("  input      {} texts", texts.len());
    let (all_ids, all_mask, lengths) =
        tokenize_all(&tokenizer, texts, prefix, seq, allow_truncation)?;
    let started = std::time::Instant::now();
    let mut timings = npum_cpu::bert::Timings::default();
    let mut vectors: Vec<f32> = Vec::with_capacity(texts.len() * hidden);
    // One workspace and one outgoing buffer for the whole run. The wrapper
    // `embed_timed` builds both fresh for every batch, which is a dozen large
    // scratch allocations per batch when there are thousands of texts.
    let mut workspace = npum_cpu::bert::BertWorkspace::new();
    let mut chunk = Vec::new();
    // An array design fixes the score width (array attention's `n_kv` or an
    // array softmax's columns), so those runs keep the full design `seq` and
    // rely on batching alone. Host and GPU runs shrink `seq` per batch.
    let fix_width = array.is_some() || eltwise.is_some();
    let mut offset = 0usize;
    while offset < texts.len() {
        let batch = (texts.len() - offset).min(BATCH);
        let eff_seq = batch_seq(seq, &lengths[offset..offset + batch], fix_width);
        let (ids, mask) = pack_batch(&all_ids, &all_mask, seq, offset, batch, eff_seq);
        model
            .embed_timed_with_into(
                ops.as_mut(),
                &mut timings,
                &ids,
                &mask,
                batch,
                eff_seq,
                array.as_deref_mut(),
                eltwise.as_deref_mut(),
                &mut workspace,
                &mut chunk,
            )
            .map_err(|e| Refusal {
                code: crate::EXIT_REFUSED,
                message: e.to_string(),
            })?;
        vectors.extend_from_slice(&chunk[..batch * hidden]);
        offset += batch;
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
    let gpu = match ops.place() {
        npum_model::dispatcher::Placement::Vulkan => Some(crate::status::Gpu {
            device: ops.device().unwrap_or("the GPU").to_string(),
            dispatches: ops.dispatches(),
        }),
        _ => None,
    };
    let total = dispatches + gpu.as_ref().map(|g| g.dispatches).unwrap_or(0);
    crate::status::print_time_split(&timings, 4, elapsed, total, gpu.as_ref());

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
/// The ARCH decides, not the file name.
fn load_tokenizer(
    container: &npum_format::Container,
    arch: &str,
) -> Result<EmbedTokenizer, String> {
    if arch.contains("whisper") || arch.contains("gemma") {
        return Err(format!(
            "the {arch} family has its own tokenizer, and this command does not \
             run it yet"
        ));
    }
    if arch.contains("gte_new_rope") || arch.contains("gte") && arch.contains("rope") {
        let blob = container
            .tensor_bytes("tokenizer.xlmr_table")
            .map_err(|_| {
                "the container holds no XLM-R tokenizer table (tokenizer.xlmr_table)".to_string()
            })?;
        let tok = XlmrTokenizer::from_table_bytes(blob)?;
        return Ok(EmbedTokenizer::Xlmr(tok));
    }
    let blob = container
        .tensor_bytes("tokenizer.vocab")
        .map_err(|_| "the container holds no vocabulary".to_string())?;
    let text = String::from_utf8_lossy(blob);
    let wp =
        WordPiece::from_lines(&text).map_err(|e| format!("the vocabulary is unusable: {e}"))?;
    Ok(EmbedTokenizer::WordPiece(wp))
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
    for name in [
        "design/gemm_rtp/design.json",
        "design/gemm_rtp_dec/design.json",
    ] {
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
/// dropped: the reference does exactly this, and a file written on Windows would
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
/// The reference's order: the root's `runtime/artifacts/<model>/artifacts_npu1`
/// first, because that is where the exporter puts it and where the status
/// block says it is.
pub fn find_artifacts(root: &Path, model_name: &str) -> Option<PathBuf> {
    // The search itself lives with the other array-opening rules in `npu`.
    // Callers that have `--artifacts` pass it to `npu::find_artifacts`
    // directly; the design-opening paths in this file do.
    crate::npu::find_artifacts(root, model_name, None)
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
///
/// # AND THE GPU, WHEN THE RUN ASKED FOR IT
///
/// `ops` is the arithmetic the run will use. When it is the GPU's, the
/// projections' GEMM, the LayerNorm, the GELU and the softmax are the
/// GPU's -- EXCEPT the passes the array took, because the array and the
/// GPU are not two answers to the same question: the run splits the work
/// between them, and the block says where each pass went.
///
/// The ATTENTION IS NEVER THE GPU'S: the family's score rows are laid
/// out in the family's own packed layout, with the heads side by side and
/// the row at the sequence's width, and the shader's attention takes a
/// different one. So the attention's line says the host, and the run does
/// the attention on the host.
pub fn places(
    container: &npum_format::Container,
    path: &Path,
    root: &Path,
    args: &Args,
    ops: &dyn npum_model::Ops,
) -> crate::status::Places {
    let mut places = crate::status::Places::default();
    let gpu = match ops.place() {
        npum_model::dispatcher::Placement::Vulkan => {
            ops.device().map(|device| crate::status::OpWhere::Gpu {
                device: device.to_string(),
            })
        }
        _ => None,
    };
    // THE ARRAY'S PLACE FIRST, AND THE GPU'S IF THE ARRAY DOES NOT HAVE
    // IT.
    //
    // The order is the order the run uses: a pass the flag named for the
    // array is the array's, and a pass the array is not running is the
    // GPU's when the run asked for the GPU. The host is what is left.
    //
    // AND THE FLAG IS WHAT DECIDES THE ARRAY'S PASSES, not the design
    // set's existence: a design set on the disk is not a dispatch, and a
    // status block that said "on the ARRAY" for a pass the host computed
    // would be a block that lies. The reference's block reads it the same way.
    let named = args.value("--artifacts");
    places.layn = array_place(
        container,
        path,
        root,
        named,
        "layernorm",
        wants_layernorm(args),
    )
    .map(crate::status::OpWhere::Array)
    .or_else(|| gpu.clone());
    places.gelu = array_place(container, path, root, named, "gelu", wants_gelu(args))
        .map(crate::status::OpWhere::Array)
        .or_else(|| gpu.clone());
    places.softm = array_place(container, path, root, named, "softmax", wants_softmax(args))
        .map(crate::status::OpWhere::Array)
        .or_else(|| gpu.clone());
    if wants_attn(args) {
        // The attention's streams are inside the projections' set, so the
        // place it names is that set's.
        places.attn =
            op_place(container, path, root, named, "gemm_rtp").map(crate::status::OpWhere::Array);
    }
    // THE GPU'S OWN PASSES, for the backend's line.
    //
    // The line names what the GPU runs and nothing else. The projections
    // are the array's when the flag asked the array for them, and the
    // elementwise passes are the array's when the flag asked for those --
    // so a run that split the work says exactly how it split it.
    if let Some(device) = ops.device() {
        let mut passes = Vec::new();
        if !wants_gemm(args) {
            passes.push("the projections");
        }
        if is_gpu(&places.layn) {
            passes.push("the LayerNorm");
        }
        if is_gpu(&places.gelu) {
            passes.push("the GELU");
        }
        if is_gpu(&places.softm) {
            passes.push("the softmax");
        }
        places.gpu = Some(crate::status::GpuPlace {
            device: device.to_string(),
            passes,
        });
    }
    places.staged_bytes = staged_bytes(container, path, root, args);
    places
}

/// Whether a pass runs on the GPU.
fn is_gpu(place: &Option<crate::status::OpWhere>) -> bool {
    matches!(place, Some(crate::status::OpWhere::Gpu { .. }))
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
pub fn staged_bytes(
    container: &npum_format::Container,
    path: &Path,
    root: &Path,
    args: &Args,
) -> usize {
    let mut bytes = 0usize;
    if wants_gemm(args) {
        bytes += panel_bytes(container);
    }
    if wants_layernorm(args) {
        if let (Some(cols), Some(layers)) = (
            design_cols(
                container,
                path,
                root,
                args.value("--artifacts"),
                "layernorm",
            ),
            layers(container),
        ) {
            // gamma and beta, f32, two of them per row.
            bytes += (1 + 2 * layers) * 2 * cols * 4;
        }
    }
    bytes
}

/// The projections' panels, as the container holds them.
fn panel_bytes(container: &npum_format::Container) -> usize {
    container
        .directory()
        .tensors()
        .iter()
        .filter(|t| t.role == "gemm_b")
        .map(|t| t.nbytes as usize)
        .sum()
}

/// One design set's row width, from its own manifest.
fn design_cols(
    container: &npum_format::Container,
    path: &Path,
    root: &Path,
    named: Option<&str>,
    set: &str,
) -> Option<usize> {
    let value = design_manifest(container, path, root, named, set)?;
    value.get("cols")?.as_u64().map(|v| v as usize)
}

/// How many transformer layers the container's model has.
fn layers(container: &npum_format::Container) -> Option<usize> {
    container.config().get_i64("num_layers").map(|v| v as usize)
}

/// One design set's place, when the run asked for it.
///
/// The flag is what makes a pass the ARRAY's, and the design set is only
/// where that pass's name comes from. A design set that is on the disk and
/// that the run did not ask for is a design set nobody dispatched, and
/// saying "on the ARRAY" about it would be a status block that lies.
fn array_place(
    container: &npum_format::Container,
    path: &Path,
    root: &Path,
    named: Option<&str>,
    set: &str,
    asked: bool,
) -> Option<crate::status::OpPlace> {
    if !asked {
        return None;
    }
    op_place(container, path, root, named, set)
}

/// One design set's place, from its own manifest.
fn op_place(
    container: &npum_format::Container,
    path: &Path,
    root: &Path,
    named: Option<&str>,
    set: &str,
) -> Option<crate::status::OpPlace> {
    let value = design_manifest(container, path, root, named, set)?;
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
fn design_manifest(
    container: &npum_format::Container,
    path: &Path,
    root: &Path,
    named: Option<&str>,
    set: &str,
) -> Option<serde_json::Value> {
    if let Some(bytes) = container
        .tensor_bytes(&format!("design/{set}/design.json"))
        .ok()
    {
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) {
            return Some(value);
        }
    }
    let dir = crate::npu::find_artifacts(root, &crate::npu::model_name(path), named)?;
    let bytes = std::fs::read(dir.join(set).join("design.json")).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Whether the run asked for one operation to go on the array.
///
/// The flag is a SET of codes, and a code is asked for by NAME.
fn wants_op(args: &crate::flags::Args, code: &str) -> bool {
    crate::npu::wants_op(args, code)
}

/// The array's elementwise passes, when the run asked for them.
///
/// The passes are separate designs and separate contexts, and only the ones
/// the flag named are opened: a design that was not opened is a pass that
/// stays on the host. The opening rules live in [`crate::npu`], shared with
/// every other family that asks for elementwise work.
pub fn open_eltwise(
    container: &npum_format::Container,
    path: &Path,
    root: &Path,
    args: &crate::flags::Args,
) -> Result<Option<Box<dyn npum_model::array::ArrayEltwise>>, String> {
    let backend = crate::npu::open_eltwise(container, path, root, args)?;
    Ok(backend.map(|b| b as Box<dyn npum_model::array::ArrayEltwise>))
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
    container: &npum_format::Container,
    path: &Path,
    model: &BertModel,
    root: &Path,
    args: &crate::flags::Args,
) -> Result<Box<dyn npum_model::array::ArrayText>, String> {
    let spec = &model.spec;
    let backend = crate::npu::open_array_text(
        container,
        path,
        root,
        args,
        crate::npu::TextGeometry::bert(spec.layers, spec.hidden, spec.ffn),
    )?;
    Ok(backend as Box<dyn npum_model::array::ArrayText>)
}

/// Array backend for GTE (fused up_gate makes ffn_up [hidden, 2*ffn]).
fn open_array_gte(
    container: &npum_format::Container,
    path: &Path,
    model: &GteModel,
    root: &Path,
    args: &crate::flags::Args,
) -> Result<Box<dyn npum_model::array::ArrayText>, String> {
    let spec = &model.spec;
    let backend = crate::npu::open_array_text(
        container,
        path,
        root,
        args,
        crate::npu::TextGeometry::gte(spec.layers, spec.hidden, spec.ffn),
    )?;
    Ok(backend as Box<dyn npum_model::array::ArrayText>)
}

/// A container's name, without the directory and without the extension.
fn container_name(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default()
}

pub fn command(args: &Args, root: &Path) -> crate::families::Outcome {
    let Some((model, input)) = args.positional.first().zip(args.positional.get(1)) else {
        return crate::needs_a_model("embed");
    };
    let prefix = args.value("--prefix").map(|s| s.to_string());
    let allow_truncation = args.switch("--allow-truncation");
    // The provenance warning comes FIRST, before the banner: it is about
    // the MODEL and not about this run.
    if !crate::families::looks_like_container_path(model) {
        crate::families::warn_if_unpinned(model);
    }
    #[rustfmt::skip]
    let outcome = crate::families::with_model_open(model, npum_families::Family::Embed, root, |path, arch, container| {
        // THE BACKEND, and the arithmetic it names.
        //
        // The choice is made HERE, before the banner: a machine with no
        // Vulkan device is refused before a status block that would claim
        // the GPU is printed.
        let choice = match crate::backend::choose(args) {
            Ok(choice) => choice,
            Err(reason) => {
                eprintln!("error: {reason}");
                return crate::EXIT_REFUSED;
            }
        };
        // The banner comes before the refusal, as in the reference.
        crate::status::announce(
            container,
            path,
            root,
            args.value("--artifacts"),
            &crate::embed::places(container, path, root, args, choice.ops.as_ref()),
        );
        // ONE GEMM design plus one context per elementwise pass. Both of this
        // family's forward passes (BERT and GTE) open the same set, so the
        // budget is checked here for both before any context is created.
        if !crate::npu::check_array_budget(args, 1) {
            return crate::EXIT_REFUSED;
        }
        // `--npu-ops` is a valid request and a valid question, and this
        // runtime answers it for five of the codes: the projections, the
        // LayerNorm, the GELU, the softmax and the attention go to the
        // array. The rest are not ported yet, and accepting them and
        // running the host would leave the command line saying "array"
        // about a run that dispatched nothing -- which is the one thing
        // the flag exists to prevent.
        //
        // The array path itself is NOT missing: the transport, the design
        // loader, the array's GEMM, its elementwise passes and its
        // attention are ported and checked on the board. What is missing is
        // the SEAM for the other families.
        //
        // BERT honours all five codes. GTE's FFN is GATED: GELU is computed
        // between ffn_up and ffn_down, and the exporter builds no standalone
        // `gelu/` design for a gated FFN -- so `gelu` must be refused BY NAME
        // here rather than accepted and silently left on the host.
        let honoured: &[&str] = if arch == "gte_new_rope_geglu"
            || (arch.contains("gte") && arch.contains("rope"))
        {
            &["gemm", "layn", "softm", "attn"]
        } else {
            &["gemm", "layn", "gelu", "softm", "attn"]
        };
        if let Some(reason) =
            crate::families::refuse_array_not_ported(npum_families::Family::Embed, args, honoured)
        {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        if let Some(prefix) = prefix.as_deref() {
            if let Some(reason) = crate::families::refuse_unknown_prefix(container, prefix) {
                eprintln!("error: {reason}");
                return crate::EXIT_REFUSED;
            }
        }
        let texts = match crate::embed::read_texts(input) {
            Ok(t) => t,
            Err(reason) => {
                eprintln!("error: {reason}");
                return crate::EXIT_REFUSED;
            }
        };
        match crate::embed::run(
            container,
            path,
            arch,
            &texts,
            prefix.as_deref(),
            crate::embed::wants_out_file(args).as_deref(),
            allow_truncation,
            root,
            args,
            choice.ops,
        ) {
            Ok(_) => 0,
            Err(refusal) => {
                eprintln!("error: {}", refusal.message);
                refusal.code
            }
        }
    });
    outcome
}

pub fn tokenize_command(args: &Args, root: &Path) -> crate::families::Outcome {
    let Some((model, file)) = args.positional.first().zip(args.positional.get(1)) else {
        return crate::needs_a_model("tokenize");
    };
    let max_len = args.value_i64("--max-len").unwrap_or(64).max(0) as usize;
    #[rustfmt::skip]
    let outcome = crate::families::with_model_open(model, npum_families::Family::Embed, root, |path, arch, container| {
        let tokenizer = match crate::families::load_tokenizer(container, arch) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: {}: {e}", path.display());
                return crate::EXIT_REFUSED;
            }
        };
        let text = match std::fs::read_to_string(file) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: cannot open {file}: {e}");
                return crate::EXIT_REFUSED;
            }
        };

        let mut lines = 0usize;
        let mut cut = 0usize;
        for line in text.lines() {
            let line = line.strip_suffix('\r').unwrap_or(line);
            let (encoding, truncated) = match tokenizer.encode_padded(line, max_len) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: {e}");
                    return crate::EXIT_REFUSED;
                }
            };
            lines += 1;
            if truncated {
                cut += 1;
            }
            let ids: Vec<String> = encoding.ids.iter().map(|id| id.to_string()).collect();
            println!("{}", ids.join(" "));
        }
        if cut > 0 {
            eprintln!(
                "  NOTE  {cut} of {lines} lines exceeded max_len={max_len} and were truncated."
            );
        }
        0
    });
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> EmbedTokenizer {
        let vocab = "[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\nworld\npre\n";
        EmbedTokenizer::WordPiece(WordPiece::from_lines(vocab).unwrap())
    }

    #[test]
    fn batch_seq_shrinks_to_the_longest_text_or_keeps_the_design_width() {
        // Host/GPU: the score width follows the sequence, so trailing padding
        // can go. The longest text wins; an empty chunk still needs one row.
        assert_eq!(batch_seq(64, &[3, 5, 2], false), 5);
        assert_eq!(batch_seq(64, &[], false), 1);
        // A text that filled the design window keeps it (the truncated case).
        assert_eq!(batch_seq(64, &[64, 1], false), 64);
        // An array design fixes the width: full sequence, no shrinking.
        assert_eq!(batch_seq(64, &[3, 5, 2], true), 64);
    }

    #[test]
    fn pack_batch_drops_only_padding_and_offsets_each_text() {
        // Three texts of design_seq 6, packed at eff_seq 4 (chunk = texts 1..3).
        let ids: Vec<u32> = (0..18).collect();
        let mask: Vec<u32> = (0..18).map(|i| if i % 6 < 4 { 1 } else { 0 }).collect();
        let (out_ids, out_mask) = pack_batch(&ids, &mask, 6, 1, 2, 4);
        assert_eq!(out_ids, vec![6, 7, 8, 9, 12, 13, 14, 15]);
        assert_eq!(out_mask, vec![1, 1, 1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn tokenize_all_flattens_and_counts_real_tokens() {
        let tok = tokenizer();
        let texts = vec!["hello".to_string(), "hello world".to_string()];
        let (ids, mask, lengths) =
            tokenize_all(&tok, &texts, None, 6, false).expect("the texts encode");
        assert_eq!(ids.len(), 12);
        assert_eq!(mask.len(), 12);
        assert_eq!(lengths, vec![3, 4]);
        // [CLS], hello, [SEP], then padding for the shorter text.
        assert_eq!(&ids[..6], &[2, 4, 3, 0, 0, 0]);
        assert_eq!(&mask[..6], &[1, 1, 1, 0, 0, 0]);

        // A prefix is folded into the text before tokenising.
        let (_, _, prefixed) = tokenize_all(&tok, &texts, Some("pre "), 6, false).unwrap();
        assert_eq!(prefixed, vec![4, 5]);
    }

    #[test]
    fn tokenize_all_refuses_a_truncated_text_with_its_index() {
        let tok = tokenizer();
        // design_seq 4 leaves room for two word pieces, so a three-piece text
        // is cut (the shorter first text still fits).
        let texts = vec!["hello".to_string(), "hello world pre".to_string()];
        let err = tokenize_all(&tok, &texts, None, 4, false).unwrap_err();
        assert_eq!(err.code, crate::EXIT_REFUSED);
        assert!(err.message.starts_with("input 1 is "), "{}", err.message);

        // With truncation allowed the same input encodes to the design width.
        let (ids, _, _) = tokenize_all(&tok, &texts, None, 4, true).unwrap();
        assert_eq!(ids.len(), 8);
    }
}
