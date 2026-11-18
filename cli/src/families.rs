//! The family verbs: embed, add, tokenize, transcribe, classify, pose,
//! hands, mppose and ocr.
//!
//! Each one is a THIN wrapper: it resolves the model, checks the family,
//! and hands over to the family's implementation. The checks are the same
//! for all of them, and that sameness is why they live here rather than
//! being copied nine times.
//!
//! # The order of the checks
//!
//! 1. The model resolves. A path that is not there is named.
//! 2. The container's arch names a family.
//! 3. The family is the one this verb is for.
//! 4. The family's own requirements -- a prompt table, an image -- are
//!    checked.
//!
//! The family check comes BEFORE the family's requirements, because a
//! Whisper container handed to `embed` is a wrong-family refusal and not a
//! "this model has no prompt table" refusal, and the first is the useful
//! message.
//!
//! # The refusals are word for word
//!
//! A refusal is the text a reader sees when something is wrong, so it is
//! the one place where a wording change is FELT rather than merely
//! observed. The golden transcripts check these strings.

use crate::flags::Args;
use std::path::{Path, PathBuf};

/// The result of a family verb.
pub type Outcome = i32;

/// `embed`.
///
/// The vectors come from the CPU backend, and they are the reference's vectors:
/// the same container through the same arithmetic gives the same bytes, and
/// `backends/cpu/tests/exact.rs` is the gate that says so.

/// The refusal for a `--prefix` this model does not have, if it is one.
///
/// Two refusals, and they are DIFFERENT refusals:
///
/// * a model with NO task prefixes at all -- its container carries no
///   `prompts` table -- is refused because nothing would be prepended, and
///   a vector that is not what was asked for is a vector that looks right;
/// * a model WITH a table, asked for a prefix that is not in it, is refused
///   with the table's names, because the reader typed one of them wrong
///   and the list is the answer.
///
/// The names are listed in SORTED order, and not in the table's order: a
/// list that a reader has to scan is more useful sorted, and the table's
/// order is the order the checkpoint happened to write them in.
pub(crate) fn refuse_unknown_prefix(path: &Path, prefix: &str) -> Option<String> {
    let container = npum_format::Container::open(path).ok()?;
    let config = container.config();
    let Some(prompts) = config.get_object("prompts") else {
        return Some(format!(
            "--prefix '{prefix}' was given, but this model has no task \
             prefixes -- its container carries no 'prompts' table, so nothing \
             would be prepended. Refusing rather than returning vectors that \
             are not what was asked for."
        ));
    };
    let mut names: Vec<&str> = prompts.keys().map(|k| k.as_str()).collect();
    names.sort_unstable();
    if names.iter().any(|n| *n == prefix) {
        return None;
    }
    Some(format!(
        "--prefix '{prefix}' is not one of this model's task prefixes: [{}]",
        names.join(", ")
    ))
}

/// The warning for a model this build cannot verify.
///
/// The reference's `warn_if_unpinned`: loud, and first, because what it warns
/// about -- weights that NOTHING checks -- is a property of the model and
/// not of the run. A run that goes quietly is how an unverified model
/// ends up in production.
pub(crate) fn warn_if_unpinned(name: &str) {
    let Some(entry) = npum_families::catalogue::find(name) else {
        return;
    };
    if !entry.sha256.is_empty() {
        return;
    }
    println!();
    println!(
        "  !! '{}' has NO sha256 pin in this build ({}).",
        entry.name, entry.repo
    );
    println!("  !! Its weights are NOT verified against anything.");
    println!("  !! To pin it:  npuembeddings add {} <sha256>", entry.repo);
    println!();
}

/// The refusal for an operation this architecture does not have.
///
/// The check lives in the FAMILY's path and not in the global gate,
/// because what an architecture has is the architecture's business: the
/// same code is valid for one family and not for another, and the
/// refusal has to say which.
pub(crate) fn refuse_op_not_in_arch(args: &Args, family: npum_families::Family) -> Option<String> {
    let listing = args.value("--npu-ops")?;
    // The codes were checked against the ones this build knows when the
    // command line was parsed; a failure here would be a different bug.
    let codes = crate::flags::parse_npu_ops(listing).ok()?;
    npum_families::ops::refuse_op_not_in_arch(family, listing, &codes)
}

/// The refusal for a `--npu-ops` this runtime accepts but cannot honour.
///
/// THE FLAG IS A QUESTION ABOUT WHERE A RUN HAPPENS, and the answer has to
/// be the truth. A family that accepts `--npu-ops gemm` and then runs the
/// host has answered "the array" with a run that dispatched nothing, and the
/// command line says something the run did not do -- which is the one thing
/// the flag exists to prevent.
///
/// So the families that have no array path refuse the flag, and say what is
/// missing AND what is not: the transport, the design loader and the array's
/// GEMM are ported and checked on the board, and what is missing is the
/// SEAM -- the place in this family where the call would go to the array
/// instead of the host.
///
/// The conv families have their own refusals, in their own words, because
/// their question is a different one: they have exactly one array-able op,
/// and their message is about what that op would move.
pub(crate) fn refuse_array_not_ported(
    family: npum_families::Family,
    args: &Args,
    honoured: &[&str],
) -> Option<String> {
    let listing = args.value("--npu-ops")?;
    let codes = crate::flags::parse_npu_ops(listing).ok()?;
    let refused: Vec<&str> = codes
        .iter()
        .copied()
        .filter(|c| !honoured.contains(c))
        .collect();
    if refused.is_empty() {
        return None;
    }
    let what = if refused.len() == 1 {
        "that operation"
    } else {
        "those operations"
    };
    let has = if honoured.is_empty() {
        "This family has no array-able operation in this runtime yet.".to_string()
    } else {
        format!(
            "The {} family runs {} on the array here.",
            family.as_str(),
            honoured.join(", ")
        )
    };
    Some(format!(
        "--npu-ops {}: the {} family runs {what} on the host in this runtime, and the array \
         path for it is not ported into this family yet -- the transport, the design loader and \
         the array's own operations are done and checked on the board, and what is missing is \
         the SEAM that would route this operation to the array here. {has} Accepting the flag and \
         running the host would make the command line say something the run did not do.",
        refused.join(", "),
        family.as_str(),
    ))
}

/// The refusal for `--npu-ops` on GTE.
///
/// GTE has its own forward pass and no seam that would route projections,
/// attention or elementwise work to the array, so every requested code must
/// be refused with that specific cause rather than the generic "no array-able
/// operation in this runtime yet" wording.
pub(crate) fn refuse_gte_array_ops(args: &Args, arch: &str) -> Option<String> {
    let listing = args.value("--npu-ops")?;
    let codes = crate::flags::parse_npu_ops(listing).ok()?;
    if codes.is_empty() {
        return None;
    }
    Some(format!(
        "--npu-ops {}: --npu-ops is not yet implemented for GTE models ({}). \
         The array/eltwise dispatch seam is missing from the GTE forward pass, \
         so accepting the flag would run on the host while reporting array work. \
         Remove --npu-ops to run on the host.",
        codes.join(", "),
        arch,
    ))
}

/// `add`.
///
/// The sha256 is checked BEFORE anything else, and a pin that can never
/// match is refused rather than stored: a stored pin that cannot match is
/// worse than no pin, because it looks like a pin.
pub fn add(args: &Args) -> Outcome {
    let Some(repo) = args.positional.first() else {
        eprintln!("error: `add` needs a HuggingFace repository:");
        eprintln!("    npuembeddings add <org/model> [<sha256>]");
        return crate::EXIT_REFUSED;
    };
    let sha = args.positional.get(1).map(|s| s.as_str()).unwrap_or("");
    if !sha.is_empty() && sha.len() != 64 {
        eprintln!(
            "error: '{sha}' is not a sha256 (expected 64 hex chars). \
             Refusing rather than storing a pin that can never match."
        );
        return crate::EXIT_REFUSED;
    }
    eprintln!("  repo       {repo}");
    eprintln!("  WARNING    no sha256 given: the weights will NOT be verified.");
    0
}

/// `tokenize`.

pub(crate) enum TokenizeTokenizer {
    WordPiece(npum_tokenizers::wordpiece::WordPiece),
    Xlmr(npum_tokenizers::xlmr::XlmrTokenizer),
}

impl TokenizeTokenizer {
    pub(crate) fn encode_padded(
        &self,
        text: &str,
        max_len: usize,
    ) -> npum_tokenizers::Result<(npum_tokenizers::Encoding, bool)> {
        match self {
            TokenizeTokenizer::WordPiece(t) => t.encode_padded(text, max_len),
            TokenizeTokenizer::Xlmr(t) => t.encode_padded(text, max_len),
        }
    }
}

/// The tokenizer a container holds.
///
/// The ARCH decides, not the file name.
pub(crate) fn load_tokenizer(
    container: &npum_format::Container,
    arch: &str,
) -> std::result::Result<TokenizeTokenizer, String> {
    if arch.contains("whisper") || arch.contains("gemma") {
        return Err(format!(
            "the {arch} family has its own tokenizer, and this command does not \
             run it yet"
        ));
    }
    if arch.contains("gte_new_rope") || (arch.contains("gte") && arch.contains("rope")) {
        let blob = container
            .tensor_bytes("tokenizer.xlmr_table")
            .map_err(|_| {
                "the container holds no XLM-R tokenizer table (tokenizer.xlmr_table)".to_string()
            })?;
        return Ok(TokenizeTokenizer::Xlmr(
            npum_tokenizers::xlmr::XlmrTokenizer::from_table_bytes(blob)?,
        ));
    }
    let blob = container
        .tensor_bytes("tokenizer.vocab")
        .map_err(|_| "the container holds no vocabulary".to_string())?;
    let text = String::from_utf8_lossy(blob);
    Ok(TokenizeTokenizer::WordPiece(
        npum_tokenizers::wordpiece::WordPiece::from_lines(&text)
            .map_err(|e| format!("the vocabulary is unusable: {e}"))?,
    ))
}

/// `transcribe`.

/// `classify`.

/// `pose`.

/// `hands`.
///
/// Two networks, one container, one image in and N hands out. The decode
/// between the two stages is where the whole of this family's subtlety lives,
/// and it is in `npum_cpu::hands_decode`; this function is the block, the
/// per-image lines and the loop.

/// `mppose`.
///
/// Two networks, one container, one image in and N people out. The decode
/// between the two stages is where the whole of this family's subtlety lives,
/// and it is in `npum_cpu::mppose_decode`; this function is the block, the
/// per-image lines and the loop.

/// `ocr`.
///
/// Two graphs, one container, one image in and N boxes of text out. The
/// detector finds the boxes and the recogniser reads each of them, and the
/// decode between the two stages is where this family's subtlety lives; it is
/// in `npum_cpu::ocr_decode`. This function is the block, the per-image lines
/// and the loop.

/// Resolve the model, check the family, and run the family's body.
pub(crate) fn with_model<F>(
    model: &str,
    wanted: npum_families::Family,
    root: &Path,
    body: F,
) -> Outcome
where
    F: FnOnce(&PathBuf, &str) -> Outcome,
{
    let path = match resolve_model(model, root) {
        Ok(p) => p,
        Err(reason) => {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
    };
    let container = match npum_format::Container::open(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {}: {e}", path.display());
            return crate::EXIT_REFUSED;
        }
    };
    let arch = container.config_str("arch").unwrap_or_default().to_string();
    let family = npum_families::registry::family_of_arch(&arch);
    if family != wanted {
        eprintln!(
            "error: {}",
            npum_families::registry::refuse_other_family(
                &path.display().to_string(),
                &arch,
                family,
                wanted.binary(),
            )
        );
        return crate::EXIT_REFUSED;
    }
    body(&path, &arch)
}

/// Resolve a model's name or path to a container on disk.
///
/// A NAME is looked up in `<root>/models/`; a PATH is used as given and
/// never re-resolved by name.
///
/// Two refusals, and they are DIFFERENT refusals:
///
/// * a PATH that is not there says the path is not there -- "a path is used
///   exactly as given, so the file has to exist where you said it does";
/// * a NAME that is not installed and is not in the catalogue says so, and
///   LISTS the models this build knows.
///
/// One message for "you pointed at a file that is not there", from the flag
/// form and from the subcommands alike: two spellings of the same refusal
/// is how they drift apart.
pub fn resolve_model(model: &str, root: &Path) -> Result<PathBuf, String> {
    if looks_like_container_path(model) {
        let path = PathBuf::from(model);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "no such container: {model}\n  \
             A path is used exactly as given, so the file has to exist where you\n  \
             said it does. A model NAME has no separator and no .npum suffix, and\n  \
             is looked up under {}/models/<name>.npum",
            root.display()
        ));
    }

    for ext in ["npue", "npum"] {
        let candidate = root.join("models").join(format!("{model}.{ext}"));
        if candidate.is_file() {
            return Ok(candidate);
        }
    }

    // Phase 10 layout: `weights/<family>/<model>/` (and flat files inside it).
    // The family is not known from the name alone, so search one level of
    // family directories under `weights/`.
    let weights = root.join("weights");
    if let Ok(families) = std::fs::read_dir(&weights) {
        for fam in families.flatten() {
            let fam_path = fam.path();
            if !fam_path.is_dir() {
                continue;
            }
            for ext in ["npue", "npum"] {
                let direct = fam_path.join(format!("{model}.{ext}"));
                if direct.is_file() {
                    return Ok(direct);
                }
            }
            let named = fam_path.join(model);
            if named.is_dir() {
                for ext in ["npue", "npum"] {
                    for name in ["model", model] {
                        let candidate = named.join(format!("{name}.{ext}"));
                        if candidate.is_file() {
                            return Ok(candidate);
                        }
                    }
                }
                // As a fallback, take any single container in that model directory.
                if let Ok(entries) = std::fs::read_dir(&named) {
                    let mut found = None;
                    for e in entries.flatten() {
                        let p = e.path();
                        if p.is_file() && looks_like_container_path(&p.to_string_lossy()) {
                            found = Some(p);
                            break;
                        }
                    }
                    if let Some(p) = found {
                        return Ok(p);
                    }
                }
            }
        }
    }

    if npum_families::catalogue::find(model).is_some() {
        // It IS a model this build knows. It is not installed, and
        // fetching it is a network operation the runtime does when it is
        // asked to serve rather than when it is asked to refuse.
        return Err(format!(
            "'{model}' is not installed. `serve` downloads and verifies it; \
             `embed` needs it on disk."
        ));
    }

    let mut known = String::new();
    for name in npum_families::catalogue::names() {
        known.push_str("\n    ");
        known.push_str(name);
    }
    Err(format!(
        "'{model}' is not installed and is not a model this build knows how to \
         fetch. Known models:{known}\n  \
         A container you packed yourself is used by name once it is in models/ \
         or weights/<family>/<model>/."
    ))
}

/// Whether the argument looks like a path to a container.
///
/// A path has a separator or a suffix; a name has neither.
pub(crate) fn looks_like_container_path(arg: &str) -> bool {
    arg.contains('/') || arg.contains('\\') || arg.ends_with(".npue") || arg.ends_with(".npum")
}

/// Delegates to the embed command handler.
pub fn embed(args: &Args, root: &Path) -> Outcome {
    crate::embed::command(args, root)
}

/// Delegates to the embed command handler.
pub fn tokenize(args: &Args, root: &Path) -> Outcome {
    crate::embed::tokenize_command(args, root)
}

/// Delegates to the stt command handler.
pub fn transcribe(args: &Args, root: &Path) -> Outcome {
    crate::stt::command(args, root)
}

/// Delegates to the vit command handler.
pub fn classify(args: &Args, root: &Path) -> Outcome {
    crate::vit::command(args, root)
}

/// Delegates to the pose command handler.
pub fn pose(args: &Args, root: &Path) -> Outcome {
    crate::pose::command(args, root)
}

/// Delegates to the hands command handler.
pub fn hands(args: &Args, root: &Path) -> Outcome {
    crate::hands::command(args, root)
}

/// Delegates to the mppose command handler.
pub fn mppose(args: &Args, root: &Path) -> Outcome {
    crate::mppose::command(args, root)
}

/// Delegates to the ocr command handler.
pub fn ocr(args: &Args, root: &Path) -> Outcome {
    crate::ocr::command(args, root)
}
