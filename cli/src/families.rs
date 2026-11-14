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
        if let Some(reason) = refuse_op_not_in_arch(args, npum_families::Family::Stt) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        let _ = arch;
        crate::stt::run(path, audio, args, root)
    })
}

/// `classify`.
pub fn classify(args: &Args, root: &Path) -> Outcome {
    let Some(model) = args.positional.first() else {
        return crate::needs_a_model("classify");
    };
    // Every leading non-flag argument after the model is an image, so
    // `classify <model> a.png b.png --top-k` classifies both.
    let images: Vec<&String> = args.positional.iter().skip(1).collect();
    if images.is_empty() {
        eprintln!("error: `classify` needs an image:");
        eprintln!("    npuimage classify <model> <image.png> [more.png ...]");
        eprintln!("    npuimage serve <model>            (POST /v1/classify)");
        eprintln!(
            "  (PNG and JPEG; anything else is refused rather than guessed at. \
             --top-k prints the runners-up to stderr, --json to stdout)"
        );
        return crate::EXIT_REFUSED;
    }
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
        // The setup is the weights arriving: the container is open, and the
        // head and the stack are read before the block is printed, because
        // the block reports how long that took.
        let vit = match npum_cpu::vit::Vit::open(path) {
            Ok(v) => v,
            Err(reason) => {
                eprintln!("error: {reason}");
                return crate::EXIT_REFUSED;
            }
        };
        crate::vit::print_block(&geometry, &designs, started.elapsed().as_secs_f64());
        let _ = arch;

        let json = args.switch("--json");
        let top_k = args.value_i64("--top-k").unwrap_or(0) > 0;
        let mut out = String::from("[\n");
        for (i, image) in images.iter().enumerate() {
            let decoded = match npum_cpu::vit_image::decode_image(std::path::Path::new(image)) {
                Ok(d) => d,
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            };
            let front = std::time::Instant::now();
            let patches = match npum_cpu::vit_image::preprocess(&decoded, &geometry) {
                Ok(p) => p,
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            };
            let front_end_s = front.elapsed().as_secs_f64();

            let enc = std::time::Instant::now();
            let hidden = vit.encode(&patches);
            let encoder_s = enc.elapsed().as_secs_f64();

            let head = std::time::Instant::now();
            let d = geometry.hidden as usize;
            let logits = vit.classify(&hidden[..d]);
            let prediction = match npum_cpu::vit::predict(&logits, vit.labels()) {
                Ok(p) => p,
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            };
            let head_s = head.elapsed().as_secs_f64();

            if json {
                out.push_str("  ");
                out.push_str(&crate::vit::prediction_json(
                    &prediction,
                    image,
                    front_end_s,
                    encoder_s,
                    head_s,
                ));
                out.push_str(if i + 1 < images.len() { ",\n" } else { "\n" });
            } else {
                eprintln!(
                    "  {:<40} {:.2} s  0 dispatches",
                    image,
                    front_end_s + encoder_s + head_s
                );
                if top_k {
                    // A ranking of the whole row, not the top two: the
                    // argmax is on stdout alone and the runners-up are a
                    // status detail, so the full sort is printed rather than
                    // a truncated one that would read as "there is nothing
                    // else close".
                    let mut order: Vec<usize> = (0..prediction.logits.len()).collect();
                    order.sort_by(|a, b| {
                        prediction.logits[*b]
                            .partial_cmp(&prediction.logits[*a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    for (k, index) in order.iter().take(10).enumerate() {
                        eprintln!(
                            "    {:2}  {:<40} logit {:+.4}",
                            k + 1,
                            vit.labels().get(*index).cloned().unwrap_or_default(),
                            prediction.logits[*index]
                        );
                    }
                }
                println!("{}", prediction.name);
            }
        }
        if json {
            out.push_str("]\n");
            print!("{out}");
        }
        0
    })
}

/// `pose`.
pub fn pose(args: &Args, root: &Path) -> Outcome {
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
    with_model(model, npum_families::Family::Pose, root, |path, arch| {
        if let Some(reason) = refuse_op_not_in_arch(args, npum_families::Family::Pose) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        // `--npu-ops conv` is a valid pose request and a valid question, and
        // this runtime has no answer to it yet: the host path is the ported
        // one. Accepting the flag and running the host would leave the
        // command line saying "array" about a run that dispatched nothing.
        if args
            .value("--npu-ops")
            .map(|v| v.split(',').any(|c| c.trim() == "conv"))
            .unwrap_or(false)
        {
            eprintln!(
                "error: --npu-ops conv: the pose family runs its convolutions on \
                 the host in this runtime, and the array path is not ported yet. \
                 Accepting the flag and running the host would make the command \
                 line say something the run did not do."
            );
            return crate::EXIT_REFUSED;
        }
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
        crate::pose::print_block(&name, &geometry, network.threads(), setup_s);
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
                if let Err(reason) = crate::pose::write_dump(dump, &network, &input) {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            }

            let net = std::time::Instant::now();
            let (head, timings) = match network.run_timed(&input) {
                Ok(h) => h,
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            };
            let network_s = net.elapsed().as_secs_f64();

            let dec = std::time::Instant::now();
            let people = match npum_cpu::pose_decode::decode(&head, &geometry, &letterbox, &params) {
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
                    dispatches: 0,
                    convs_host: geometry.convs.len().saturating_sub(1) as i64,
                };
                print!(
                    "{}",
                    npum_cpu::pose_decode::result_json(
                        &detection,
                        &geometry,
                        &params,
                        image,
                        false
                    )
                );
            }
        }
        0
    })
}

/// `hands`.
///
/// Two networks, one container, one image in and N hands out. The decode
/// between the two stages is where the whole of this family's subtlety lives,
/// and it is in `npum_cpu::hands_decode`; this function is the block, the
/// per-image lines and the loop.
pub fn hands(args: &Args, root: &Path) -> Outcome {
    let Some(model) = args.positional.first() else {
        return crate::needs_a_model("hands");
    };
    let images: Vec<&String> = args.positional.iter().skip(1).collect();
    if images.is_empty() {
        eprintln!("error: this is a hands model, so say whose hands to find:");
        eprintln!("    npuimage hands <model> <image.png> [more.png ...]");
        eprintln!("  (PNG and JPEG; anything else is refused rather than guessed at)");
        eprintln!("  --max-hands 1     run the second stage on at most N detections");
        eprintln!("  --text            a human summary instead of JSON");
        eprintln!("  --hands-dump FILE every graph node's output, for the gate");
        eprintln!(
            "  (the score and NMS thresholds come from the container, not from a\n   \
             flag: they are part of the checkpoint -- see the refusal above)"
        );
        return crate::EXIT_REFUSED;
    }
    with_model(model, npum_families::Family::Hands, root, |path, arch| {
        if let Some(reason) = refuse_op_not_in_arch(args, npum_families::Family::Hands) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        // `--npu-ops conv` is a valid hands request and a valid question, and
        // this runtime has no answer to it yet: the host path is the ported
        // one. Accepting the flag and running the host would leave the command
        // line saying "array" about a run that dispatched nothing -- which is
        // the one thing this project treats as the worst outcome.
        if args
            .value("--npu-ops")
            .map(|v| v.split(',').any(|c| c.trim() == "conv"))
            .unwrap_or(false)
        {
            eprintln!(
                "error: --npu-ops conv: the hands family runs its convolutions on the \
                 host in this runtime, and the array path is not ported yet. Accepting \
                 the flag and running the host would make the command line say \
                 something the run did not do."
            );
            return crate::EXIT_REFUSED;
        }
        // pose's own flags are REFUSED rather than ignored: they read as
        // "thresholds for the detection" on a mode that also has a detector,
        // and silently dropping them would produce a run whose boxes differ
        // from the one that was asked for with nothing in the output saying so.
        for f in [
            "--conf",
            "--iou",
            "--kpt",
            "--max-det",
            "--pose",
            "--pose-dump",
            "--bo-mode",
        ] {
            if args.value(f).is_some() || args.switch(f) {
                eprintln!(
                    "error: {f}: not a flag of `hands`. The score, NMS and crop \
                     thresholds are read from the container (score_threshold, \
                     nms_threshold, the two enlarge factors and the two shifts), \
                     because they are part of how this checkpoint was trained and the \
                     OpenCV zoo demo that recorded the golden used the container's \
                     values, not a command line's. For --pose: use `pose`, which is a \
                     different model with different flags. The two dumps are separate \
                     because they record different graphs, and one reader cannot read \
                     the other's file."
                );
                return crate::EXIT_REFUSED;
            }
        }
        // --max-hands: a CAP on how many detections reach the second stage, not
        // a score threshold. Every hand is an independent crop of the original
        // frame, so the only lever is how many to run. A cap of 0 is a typo and
        // a cap of -1 is the session's own "all of them".
        let mut max_hands = -1i64;
        if let Some(v) = args.value("--max-hands") {
            match v.parse::<i64>() {
                Ok(n) if (1..=10000).contains(&n) => max_hands = n,
                _ => {
                    eprintln!(
                        "error: --max-hands '{v}': expected an integer in 1..10000, or \
                         omit it to run the second stage on every detection"
                    );
                    return crate::EXIT_REFUSED;
                }
            }
        }
        let text = args.switch("--text");
        let started = std::time::Instant::now();
        let network = match crate::hands::open(path) {
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
        crate::hands::print_block(&name, network.geometry(), setup_s);
        let _ = arch;

        let dump_path = args.value("--hands-dump").map(std::path::PathBuf::from);
        let mut dump = match dump_path.as_deref() {
            Some(p) => match crate::hands::Dump::create(p) {
                Ok(d) => Some(d),
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            },
            None => None,
        };
        if let (Some(p), Some(_)) = (dump_path.as_deref(), dump.as_ref()) {
            crate::hands::print_dump_header(p, network.geometry());
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
                npum_cpu::hands_decode::detect(&network, &decoded, max_hands, &mut hook)
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
            crate::hands::print_run(image, &run);
            if text {
                print!("{}", npum_cpu::hands_decode::text_summary(image, &run));
                continue;
            }
            crate::hands::print_crop(&run);
            print!("{}", npum_cpu::hands_decode::result_json(&run, image));
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

/// `mppose`.
///
/// Two networks, one container, one image in and N people out. The decode
/// between the two stages is where the whole of this family's subtlety lives,
/// and it is in `npum_cpu::mppose_decode`; this function is the block, the
/// per-image lines and the loop.
pub fn mppose(args: &Args, root: &Path) -> Outcome {
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
    with_model(model, npum_families::Family::MpPose, root, |path, arch| {
        if let Some(reason) = refuse_op_not_in_arch(args, npum_families::Family::MpPose) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        // `--npu-ops conv` is a valid mppose request and a valid question, and
        // this runtime has no answer to it yet: the host path is the ported one.
        // Accepting the flag and running the host would leave the command line
        // saying "array" about a run that dispatched nothing -- which is the one
        // thing this project treats as the worst outcome.
        if args
            .value("--npu-ops")
            .map(|v| v.split(',').any(|c| c.trim() == "conv"))
            .unwrap_or(false)
        {
            eprintln!(
                "error: --npu-ops conv: the mppose family runs its convolutions on the host \
                 in this runtime, and the array path is not ported yet. Accepting the flag and \
                 running the host would make the command line say something the run did not do."
            );
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
        crate::mppose::print_block(&name, network.geometry(), setup_s);
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
                npum_cpu::mppose_decode::detect(&network, &decoded, max_people, &mut hook)
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

/// `ocr`.
///
/// Two graphs, one container, one image in and N boxes of text out. The
/// detector finds the boxes and the recogniser reads each of them, and the
/// decode between the two stages is where this family's subtlety lives; it is
/// in `npum_cpu::ocr_decode`. This function is the block, the per-image lines
/// and the loop.
pub fn ocr(args: &Args, root: &Path) -> Outcome {
    let Some(model) = args.positional.first() else {
        return crate::needs_a_model("ocr");
    };
    let images: Vec<&String> = args.positional.iter().skip(1).collect();
    if images.is_empty() {
        eprintln!("error: this is an OCR model, so say what to read:");
        eprintln!("    npuimage ocr <model> <image.png> [more.png ...]");
        eprintln!("  (PNG and JPEG; anything else is refused rather than guessed at)");
        eprintln!("  --max-boxes 1      read at most N boxes");
        eprintln!("  --text             a human summary instead of JSON");
        eprintln!("  --ocr-dump FILE    every graph node's output, for the gate");
        eprintln!(
            "  (every threshold comes from the container, not from a flag: they are\n   \
             part of the checkpoint -- see the refusal above)"
        );
        return crate::EXIT_REFUSED;
    }
    with_model(model, npum_families::Family::Ocr, root, |path, arch| {
        if let Some(reason) = refuse_op_not_in_arch(args, npum_families::Family::Ocr) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        // The other architectures' own flags are REFUSED rather than ignored: they
        // read as "thresholds for the detection" on a mode that also has a detector,
        // and silently dropping them would produce a run whose boxes differ from the
        // one that was asked for with nothing in the output saying so.
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
            "--max-people",
            "--mppose-dump",
        ] {
            if args.value(f).is_some() || args.switch(f) {
                eprintln!(
                    "error: {f}: not a flag of `ocr`. The detector's thresholds and the \
                     recogniser's are read from the container (det_thresh, det_box_thresh, \
                     det_unclip_ratio, det_max_candidates, det_min_size), because they are \
                     part of how this checkpoint was trained and the demo that recorded the \
                     golden used the container's values, not a command line's. For --pose: use \
                     `pose`, which is a different model with different flags; for \
                     --max-people: use `mppose`, whose second stage is a landmark network \
                     and not a recogniser. The dumps are separate because they record \
                     different graphs, and one reader cannot read another's file."
                );
                return crate::EXIT_REFUSED;
            }
        }
        // --max-boxes: a CAP on how many boxes reach the recogniser, not a score
        // threshold. Every box is an independent crop of the original frame, so there
        // is no way to batch them and the only lever is how many to read.
        let mut max_boxes = -1i64;
        if let Some(v) = args.value("--max-boxes") {
            match v.parse::<i64>() {
                Ok(n) if (1..=10000).contains(&n) => max_boxes = n,
                _ => {
                    eprintln!(
                        "error: --max-boxes '{v}': expected an integer in 1..10000, or omit it \
                         to read every box the detector found"
                    );
                    return crate::EXIT_REFUSED;
                }
            }
        }
        let text = args.switch("--text");
        let started = std::time::Instant::now();
        let mut session = match crate::ocr::open(path) {
            Ok(s) => s,
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
        let geometry = session.geometry().clone();
        crate::ocr::print_block(&name, &geometry, setup_s);
        let _ = arch;

        let dump_path = args.value("--ocr-dump").map(std::path::PathBuf::from);
        let mut dump = match dump_path.as_deref() {
            Some(p) => match crate::ocr::Dump::create(p) {
                Ok(d) => Some(d),
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            },
            None => None,
        };
        if let (Some(p), Some(_)) = (dump_path.as_deref(), dump.as_ref()) {
            crate::ocr::print_dump_header(p, &geometry);
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
                let mut hook = |net: &str, node: i64, t: &npum_cpu::ocr::Tensor| {
                    if let Some(d) = dump.as_mut() {
                        d.record(net, node, t);
                    }
                };
                session.read(&decoded, max_boxes, &mut hook)
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
            crate::ocr::print_run(image, &run);
            crate::ocr::print_split(
                session.det_cost(),
                session.rec_cost(),
                geometry.det_dense(),
                geometry.det_depthwise(),
                geometry.det_transposed(),
                // The recogniser's counters accumulate over EVERY box, so the
                // split reports the multiplication the oracle's does: the
                // recogniser runs once per box, and its 24 dense and 14
                // depthwise convolutions are counted once per crop.
                geometry.rec_dense() * run.boxes_read,
                geometry.rec_depthwise() * run.boxes_read,
                geometry.rec_transposed() * run.boxes_read,
            );
            if text {
                print!("{}", npum_cpu::ocr_decode::text_summary(image, &run));
                continue;
            }
            print!("{}", npum_cpu::ocr_decode::result_json(&run, image));
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
