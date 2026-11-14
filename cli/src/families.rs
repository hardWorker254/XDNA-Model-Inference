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
/// The vectors come from the CPU backend, and they are the C++'s vectors:
/// the same container through the same arithmetic gives the same bytes, and
/// `backends/cpu/tests/exact.rs` is the gate that says so.
pub fn embed(args: &Args, root: &Path) -> Outcome {
    let Some((model, input)) = args.positional.first().zip(args.positional.get(1)) else {
        return crate::needs_a_model("embed");
    };
    let prefix = args.value("--prefix").map(|s| s.to_string());
    let allow_truncation = args.switch("--allow-truncation");
    // The provenance warning comes FIRST, before the banner: it is about
    // the MODEL and not about this run.
    if !looks_like_container_path(model) {
        warn_if_unpinned(model);
    }
    with_model(model, npum_families::Family::Embed, root, |path, arch| {
        // The banner comes before the refusal, as in the C++.
        crate::status::announce(path, root);
        if let Some(prefix) = prefix.as_deref() {
            if let Some(reason) = refuse_unknown_prefix(path, prefix) {
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
            path,
            arch,
            &texts,
            prefix.as_deref(),
            crate::embed::wants_out_file(args).as_deref(),
            allow_truncation,
            root,
        ) {
            Ok(_) => 0,
            Err(refusal) => {
                eprintln!("error: {}", refusal.message);
                refusal.code
            }
        }
    })
}

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
fn refuse_unknown_prefix(path: &Path, prefix: &str) -> Option<String> {
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
/// The C++'s `warn_if_unpinned`: loud, and first, because what it warns
/// about -- weights that NOTHING checks -- is a property of the model and
/// not of the run. A run that goes quietly is how an unverified model
/// ends up in production.
fn warn_if_unpinned(name: &str) {
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

/// The refusal for a family this runtime does not run yet.
///
/// The families arrive in the plan's order, and the ones that have not
/// arrived say so BY NAME. Exiting 0 with nothing on the stdout would be
/// a lie about the answer: a script would read it as "the model said
/// nothing", and the model's answer is a sentence.
fn refuse_not_ported(family: npum_families::Family) -> Option<String> {
    family.not_ported_refusal()
}

/// The refusal for an operation this architecture does not have.
///
/// The check lives in the FAMILY's path and not in the global gate,
/// because what an architecture has is the architecture's business: the
/// same code is valid for one family and not for another, and the
/// refusal has to say which.
fn refuse_op_not_in_arch(args: &Args, family: npum_families::Family) -> Option<String> {
    let listing = args.value("--npu-ops")?;
    // The codes were checked against the ones this build knows when the
    // command line was parsed; a failure here would be a different bug.
    let codes = crate::flags::parse_npu_ops(listing).ok()?;
    npum_families::ops::refuse_op_not_in_arch(family, listing, &codes)
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
pub fn tokenize(args: &Args, root: &Path) -> Outcome {
    let Some((model, file)) = args.positional.first().zip(args.positional.get(1)) else {
        return crate::needs_a_model("tokenize");
    };
    let max_len = args.value_i64("--max-len").unwrap_or(64).max(0) as usize;
    with_model(model, npum_families::Family::Embed, root, |path, arch| {
        let container = match npum_format::Container::open(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: {}: {e}", path.display());
                return crate::EXIT_REFUSED;
            }
        };
        let tokenizer = match load_tokenizer(&container, arch) {
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
    })
}

/// The tokenizer a container holds.
///
/// The ARCH decides, not the file name: a container that says it is a BERT
/// model has a BERT vocabulary, whatever it is called on disk.
fn load_tokenizer(
    container: &npum_format::Container,
    arch: &str,
) -> std::result::Result<npum_tokenizers::wordpiece::WordPiece, String> {
    // The arch is checked, and the check is not a formality: a Whisper
    // container's vocabulary is a BYTE-LEVEL one and a BERT container's is a
    // WORD-PIECE one, and running the wrong tokenizer over the right
    // vocabulary produces ids that are all valid and all wrong.
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
    npum_tokenizers::wordpiece::WordPiece::from_lines(&text)
        .map_err(|e| format!("the vocabulary is unusable: {e}"))
}

/// `transcribe`.
pub fn transcribe(args: &Args, root: &Path) -> Outcome {
    let Some((model, audio)) = args.positional.first().zip(args.positional.get(1)) else {
        return crate::needs_a_model("transcribe");
    };
    with_model(model, npum_families::Family::Stt, root, |path, arch| {
        if !Path::new(audio).exists() {
            eprintln!("error: the audio file {audio} does not exist.");
            return crate::EXIT_REFUSED;
        }
        eprintln!("  model      {}: {}", path.display(), arch);
        if let Some(reason) = refuse_not_ported(npum_families::Family::Stt) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        0
    })
}

/// `classify`.
pub fn classify(args: &Args, root: &Path) -> Outcome {
    let Some(model) = args.positional.first() else {
        return crate::needs_a_model("classify");
    };
    with_model(model, npum_families::Family::Cls, root, |path, arch| {
        if let Some(reason) = refuse_op_not_in_arch(args, npum_families::Family::Cls) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        // The ViT's own block, on the stderr: the geometry, the design set,
        // and where every operation runs.
        let started = std::time::Instant::now();
        let container = match npum_format::Container::open(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: {}: {e}", path.display());
                return crate::EXIT_REFUSED;
            }
        };
        let Some(geometry) = crate::vit::read_geometry(&container, path) else {
            eprintln!(
                "error: {}: the container does not carry the ViT's geometry",
                path.display()
            );
            return crate::EXIT_REFUSED;
        };
        let designs = crate::vit::find_designs(root, &geometry.model_name)
            .unwrap_or_else(|| root.join("runtime").join("artifacts").join(&geometry.model_name));
        crate::vit::print_block(&geometry, &designs, started.elapsed().as_secs_f64());
        if let Some(reason) = refuse_not_ported(npum_families::Family::Cls) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        let _ = arch;
        0
    })
}

/// `pose`.
pub fn pose(args: &Args, root: &Path) -> Outcome {
    let Some(model) = args.positional.first() else {
        return crate::needs_a_model("pose");
    };
    with_model(model, npum_families::Family::Pose, root, |path, arch| {
        if let Some(reason) = refuse_op_not_in_arch(args, npum_families::Family::Pose) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        eprintln!("  model      {}: {}", path.display(), arch);
        if let Some(reason) = refuse_not_ported(npum_families::Family::Pose) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        0
    })
}

/// `hands`.
pub fn hands(args: &Args, root: &Path) -> Outcome {
    let Some(model) = args.positional.first() else {
        return crate::needs_a_model("hands");
    };
    with_model(model, npum_families::Family::Hands, root, |path, arch| {
        if let Some(reason) = refuse_op_not_in_arch(args, npum_families::Family::Hands) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        eprintln!("  model      {}: {}", path.display(), arch);
        if let Some(reason) = refuse_not_ported(npum_families::Family::Hands) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        0
    })
}

/// `mppose`.
pub fn mppose(args: &Args, root: &Path) -> Outcome {
    let Some(model) = args.positional.first() else {
        return crate::needs_a_model("mppose");
    };
    with_model(model, npum_families::Family::MpPose, root, |path, arch| {
        if let Some(reason) = refuse_op_not_in_arch(args, npum_families::Family::MpPose) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        eprintln!("  model      {}: {}", path.display(), arch);
        if let Some(reason) = refuse_not_ported(npum_families::Family::MpPose) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        0
    })
}

/// `ocr`.
pub fn ocr(args: &Args, root: &Path) -> Outcome {
    let Some(model) = args.positional.first() else {
        return crate::needs_a_model("ocr");
    };
    with_model(model, npum_families::Family::Ocr, root, |path, arch| {
        if let Some(reason) = refuse_op_not_in_arch(args, npum_families::Family::Ocr) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        eprintln!("  model      {}: {}", path.display(), arch);
        if let Some(reason) = refuse_not_ported(npum_families::Family::Ocr) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        0
    })
}

/// Resolve the model, check the family, and run the family's body.
fn with_model<F>(model: &str, wanted: npum_families::Family, root: &Path, body: F) -> Outcome
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
         A container you packed yourself is used by name once it is in models/."
    ))
}

/// Whether the argument looks like a path to a container.
///
/// A path has a separator or a suffix; a name has neither.
fn looks_like_container_path(arg: &str) -> bool {
    arg.contains('/') || arg.contains('\\') || arg.ends_with(".npue") || arg.ends_with(".npum")
}
