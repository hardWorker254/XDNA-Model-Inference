//! The flag table and the parser.
//!
//! # Why this is a hand-written parser and not clap
//!
//! Three properties of the C++ parser are CONTRACT, not implementation
//! detail, and clap's derive API does not have all three:
//!
//! 1. **"The last occurrence wins."** `--pipeline 4 --pipeline 1` disables
//!    pipelining. The C++ tree got this wrong once -- `serve <whisper>
//!    --threads 8` answered with 24 threads -- and it cost a bug report.
//!
//! 2. **A flag may follow a positional argument.** `embed model in.txt
//!    --pipeline 1` is a valid command line.
//!
//! 3. **`--serve [PORT]` takes a port only if the next argument starts with
//!    a DIGIT, and `--serve=8080` is not accepted at all.** Nothing has ever
//!    documented `--serve=8080`, and accepting it now would be a new
//!    spelling that nobody asked for.
//!
//! Plus one more that is about the TABLE rather than the parser: the flag
//! table includes the REFUSED-BY-NAME families -- `--npu-eltwise`,
//! `--host-gelu`, `--extra-ops` -- because the parser must let them through
//! so that the refusal downstream can name the replacement. A parser that
//! refused them as "unrecognised" would be strictly less useful on exactly
//! the flags most likely to appear in a stale command line.
//!
//! clap is still the tool that generates the CLI reference in the
//! documentation; it is just not the tool that parses.

use std::collections::HashMap;

/// How many values a flag swallows.
///
/// The parser does not USE the arity to skip arguments, and that is
/// deliberate: every consumer reads the argument vector itself, so consuming
/// the value here would only mean the value stopped being visible to them.
/// The arity is recorded because it is the next thing anyone will want from
/// the table.
pub fn flag_arity(name: &str) -> Option<i32> {
    let arity = match name {
        // -- the ones the parser reads into the arguments struct.
        "--bench" => 1,
        "--threads" => 1,
        "--pipeline" => 1,
        "--prefix" => 1,
        "--artifacts" => 1,
        "--model" => 1,
        "--serve" => 0,
        "--port" => 1,
        "--bind" => 1,
        "--embed" => 1,
        "--add" => 2,
        "--tokenize" => 1,
        "--prepare-model" => 2,
        "--list-models" => 0,
        "--help" => 0,
        "--allow-truncation" => 0,
        "--allow-contention" => 0,
        "--max-len" => 1,
        "--source-repo" => 1,
        "--gemma-host-only" => 0,
        "--tile-k" => 1,
        "--tile-n" => 1,
        "--npu-ops" => 1,
        // -- the backend choice: where the ARITHMETIC happens. The host is
        // the default and the C++'s arithmetic; the GPU is the other one.
        // A name that is neither is refused by name, like every other
        // flag this runtime does not accept.
        "--backend" => 1,
        "--root" => 1,
        // -- read straight off the argument vector by the runtime and the
        // setup headers.
        "--dev" => 1,
        "--bo-mode" => 1,
        "--cpu" => 0,
        "--no-fuse-ffn" => 0,
        "--sim-c-bf16" => 0,
        // -- the gemma mode. `--guard-contention` is not in --help because it
        // is not typed by a person: the release benchmark passes it, and
        // without this row the unrecognised-option refusal would break the
        // release benchmark with an error about a flag it never mentions.
        "--guard-contention" => 0,
        // -- the execute and probe modes.
        "--probe" => 0,
        "--probe-streams" => 0,
        "--probe-pair" => 0,
        "--probe-ctx" => 0,
        "--probe-design" => 1,
        "--probe-insts" => 2,
        "--probe-rtp" => 2,
        "--probe-bo" => 3,
        "--soak-npu" => 1,
        "--soak-cpu" => 1,
        "--encode-file" => 1,
        "--transcribe" => 1,
        "--audio" => 1,
        // -- the vit mode and the whisper modes.
        "--classify" => 1,
        "--json" => 0,
        "--top-k" => 0,
        "--convert" => 0,
        "--language" => 1,
        "--task" => 1,
        "--max-new" => 1,
        "--chunk-seconds" => 1,
        "--stride-seconds" => 1,
        // -- the pose mode. `--text` is a no-argument policy flag that
        // switches the mode from its default JSON to a human summary; the
        // four others are the decode thresholds, and each takes a value so
        // that `flag_takes_value()` swallows the number rather than opening
        // it as a PNG.
        "--pose" => 1,
        "--text" => 0,
        "--conf" => 1,
        "--iou" => 1,
        "--kpt" => 1,
        "--max-det" => 1,
        // Diagnostic: dump every graph node's output for a node-by-node diff
        // against the pose gate.
        "--pose-dump" => 1,
        // -- the hands mode. `--hands` is the image list, arity 1, and it
        // REPEATS -- the mode accumulates every occurrence, the same
        // treatment `--classify` and `--pose` get. `--max-hands` caps how
        // many detections reach the landmark network; it is a COUNT and not
        // a threshold, because every hand is an independent crop and the
        // only lever on the second stage is how many of them to run.
        // This mode's score, NMS and crop thresholds are NOT here on purpose:
        // they come from the container, and the mode REFUSES
        // --conf/--iou/--kpt/--max-det by name rather than accepting and
        // ignoring them.
        "--hands" => 1,
        "--max-hands" => 1,
        // Diagnostic: dump every graph node's output of BOTH networks.
        "--hands-dump" => 1,
        // -- the mppose mode. Same treatment and the same reasons:
        // `--mppose` is the image list, arity 1, REPEATED, and `--max-people`
        // caps how many detections reach the landmark network -- a COUNT,
        // because every person is an independent ROTATED crop. No threshold
        // flags again: the detector's score and NMS and the landmark net's
        // confidence all come from the container, and the mode REFUSES
        // --conf/--iou/--kpt/--max-det AND the other modes' by name rather
        // than accepting and ignoring them -- because two different
        // containers called `pose` and `hands` answer to those flags with
        // different semantics, and this mode is the third.
        "--mppose" => 1,
        "--max-people" => 1,
        // Diagnostic: dump every graph node's output of BOTH graphs.
        "--mppose-dump" => 1,
        // -- the ocr mode. Same treatment and the same reasons: `--ocr` is
        // the image list, arity 1, REPEATED, and `--max-boxes` caps how many
        // boxes reach the recogniser -- a COUNT, because every box is an
        // independent STRAIGHTENED crop. No threshold flags again: the
        // detector's thresholds are in the container, and the mode REFUSES
        // --conf/--iou/--kpt/--max-det AND the other modes' by name rather
        // than accepting and ignoring them.
        "--ocr" => 1,
        "--max-boxes" => 1,
        // Diagnostic: dump every graph node's output of BOTH graphs.
        "--ocr-dump" => 1,
        "--token" => 1,
        // -- REFUSED BY NAME, not accepted. Listed so that the parser lets
        // them through to the refusal, whose message names the replacement.
        // `--npu-extra-ops` is here for the same reason and used to be
        // absent: it was this build's LIVE flag when that table was written,
        // so nothing noticed it being refused rather than accepted. The
        // flag-table gate is what caught it, and it caught it because it
        // checks for a second row with the same name rather than only
        // checking that every accepted flag parses.
        "--npu-eltwise" => 0,
        "--host-gelu" => 0,
        "--host-ln" => 0,
        "--host-sm" => 0,
        "--npu-extra-ops" => 0,
        "--extra-ops" => 0,
        _ => return None,
    };
    Some(arity)
}

/// Whether the flag is one this binary knows.
pub fn flag_is_known(name: &str) -> bool {
    flag_arity(name).is_some()
}

/// Whether the flag takes a value.
///
/// `-h` is the one short option. It is spelled out rather than folded into
/// the table because the table is keyed on the long spelling everywhere else,
/// and a one-off two-character special case reads more clearly here than it
/// does as a row whose every sibling starts with "--".
pub fn flag_is_known_short(name: &str) -> bool {
    name == "-h"
}

/// The flags that are REFUSED BY NAME rather than accepted.
///
/// Each one's refusal names the replacement, so a stale command line cannot
/// look like it worked.
///
/// The replacements are the C++ runtime's, word for word, and they are not
/// all the same shape: `--npu-eltwise` becomes a LIST of codes, because it
/// used to be three flags in one, while `--host-gelu` becomes the same flag
/// with a code REMOVED from it. That difference is the whole reason these
/// are a table and not a list of strings: "the same flag, minus one code"
/// and "three flags in one" are different sentences.
pub const REMOVED_OP_FLAGS: &[(&str, &str)] = &[
    ("--npu-eltwise", "--npu-ops gelu,layn,softm"),
    ("--host-gelu", "--npu-ops without gelu"),
    ("--host-ln", "--npu-ops without layn"),
    ("--host-sm", "--npu-ops without softm"),
    ("--npu-extra-ops", "--npu-ops"),
];

/// The flags that belong to the EXPORTER and not to the runtime.
///
/// `--extra-ops` is `tools/export/export_eltwise.py`'s own spelling and
/// stays that tool's: it is the low-level per-kernel builder, and the flag
/// only narrows that for a direct call. At run time it selected nothing and
/// was dropped without a word -- so a command line that asked for an op
/// quietly ran without it. The GEMM exporter has no flag at all any more, so
/// nobody reaches this one by accident from a top-level export -- but a
/// script that does reach for it, having learned it from a README that
/// predates the split, is told what to type instead.
pub const EXPORTER_ONLY_FLAGS: &[(&str, &str)] = &[("--extra-ops", "--npu-ops")];

/// The operation codes `--npu-ops` takes, in TABLE order.
///
/// The order matters: a message that lists what was asked for should read
/// the same every time for the same set, and "layn,softm" is easier to
/// match against a help block than a permutation of itself.
///
/// The second column is the long name, for the messages that have to say
/// what a code means.
pub const NPU_OP_CODES: &[(&str, &str)] = &[
    ("gemm", "the per-layer GEMMs"),
    ("gelu", "GELU"),
    ("layn", "LayerNorm"),
    ("softm", "softmax"),
    ("conv", "conv1d (Whisper's audio front end)"),
    ("attn", "attention, as two GEMMs"),
    ("mproj", "Whisper's mel filter bank, as a GEMM"),
    ("fft", "Whisper's 400-point transform, as a GEMM"),
    ("logit", "the vocabulary projection, as a GEMM"),
];

/// The codes, comma-joined, the way the refusal lists them.
pub fn npu_op_codes() -> String {
    NPU_OP_CODES
        .iter()
        .map(|(code, _)| *code)
        .collect::<Vec<_>>()
        .join(", ")
}

/// The refusal for a removed flag, if one of them was given.
///
/// The refusal is checked against the WHOLE argument vector, not against the
/// parsed flags: a removed flag must be refused even when it arrives after
/// the positional arguments, and even when it arrives with a value the
/// parser would have swallowed.
pub fn refuse_removed_op_flags(args: &[String]) -> Option<String> {
    for name in args {
        for (old, new) in REMOVED_OP_FLAGS {
            if name == old {
                return Some(format!(
                    "{old} is gone: one flag now says which ops go on the array, \
                     and an op is on the host when it is NOT listed. Use {new}. \
                     (The old flags are refused rather than ignored so a stale \
                     command line cannot look like it worked.)"
                ));
            }
        }
    }
    None
}

/// The refusal for an exporter's flag, if one of them was given.
pub fn refuse_exporter_only_flags(args: &[String]) -> Option<String> {
    for name in args {
        for (old, new) in EXPORTER_ONLY_FLAGS {
            if name == old {
                return Some(format!(
                    "{old} is tools/export/export_eltwise.py's BUILD flag: at run \
                     time it selected nothing and was dropped without a word, so a \
                     command line that asked for an op quietly ran without it. To \
                     send ops to the array, use {new}."
                ));
            }
        }
    }
    None
}

/// The refusal for an operation code this build does not know.
///
/// The codes are listed, because a user who typed a code that does not exist
/// needs to know which ones do -- and a refusal that says only "no" is a
/// refusal that sends the reader to the source.
pub fn refuse_unknown_npu_op(code: &str) -> String {
    format!(
        "--npu-ops: '{code}' is not an op this build knows. Valid codes: [{}] \
         (gemm = the per-layer GEMMs, layn = LayerNorm, softm = softmax, gelu = \
         GELU, conv = Whisper's conv1/conv2, attn = attention as two GEMMs \
         (QK^T and softmax.V), mproj = the mel filter bank as a GEMM, fft = the \
         400-point transform as a GEMM, logit = the vocabulary projection as a \
         GEMM). Nothing listed means every op runs on the host, which is the \
         measured-faster path.",
        npu_op_codes()
    )
}

/// The codes `--npu-ops` was given.
///
/// The LAST occurrence of the flag wins, and an empty component is skipped,
/// so a trailing comma and `--npu-ops ""` are the same thing.
pub fn parse_npu_ops(list: &str) -> Result<Vec<&'static str>, String> {
    let mut out = Vec::new();
    // The codes come back SORTED, because the C++'s are a `std::set` and
    // every message that names them walks that set. The order the user
    // typed them in is echoed unchanged where it is echoed at all.
    for item in list.split(',') {
        let code = item.trim();
        if code.is_empty() {
            continue;
        }
        match NPU_OP_CODES.iter().find(|(known, _)| *known == code) {
            Some((known, _)) => {
                if !out.contains(known) {
                    out.push(*known);
                }
            }
            None => return Err(refuse_unknown_npu_op(code)),
        }
    }
    out.sort_unstable();
    Ok(out)
}

/// The parsed command line.
#[derive(Debug, Clone, Default)]
pub struct Args {
    /// The positional arguments, in order.
    pub positional: Vec<String>,
    /// Every flag's value, by name. The LAST occurrence wins.
    pub flags: HashMap<String, String>,
    /// The flags that appeared with no value.
    pub switches: Vec<String>,
    /// The verb, when one was given.
    pub subcommand: Option<String>,
}

impl Args {
    /// A flag's value, if it was given.
    pub fn value(&self, name: &str) -> Option<&str> {
        self.flags.get(name).map(|s| s.as_str())
    }

    /// A flag's value as an integer.
    pub fn value_i64(&self, name: &str) -> Option<i64> {
        self.value(name).and_then(|v| v.parse().ok())
    }

    /// Whether a switch was given.
    pub fn switch(&self, name: &str) -> bool {
        self.switches.iter().any(|s| s == name)
    }

    /// Whether a flag was given at all, with or without a value.
    pub fn present(&self, name: &str) -> bool {
        self.flags.contains_key(name) || self.switch(name)
    }
}

/// What the parser found.
#[derive(Debug)]
pub enum ParseError {
    /// An option this binary does not know.
    UnknownOption(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnknownOption(name) => write!(
                f,
                "unrecognised option `{name}` -- run with --help for the list"
            ),
        }
    }
}

/// The verbs, in the order they are listed.
pub const SUBCOMMANDS: &[&str] = &[
    "list",
    "serve",
    "embed",
    "add",
    "tokenize",
    "transcribe",
    "classify",
    "pose",
    "hands",
    "mppose",
    "ocr",
];

/// Parse the argument vector.
///
/// `argv` is the vector WITHOUT the program name.
///
/// The rules, and each of them is a rule rather than a default:
///
/// * the FIRST positional is the verb, and the rest are its arguments;
/// * a flag may appear anywhere, including after the positional arguments;
/// * the LAST occurrence of a flag wins;
/// * an option that takes a value takes the NEXT argument, whatever it is,
///   unless that argument is itself an option -- in which case the value is
///   missing and the flag is recorded with no value, so that the consumer
///   can say "this flag needs a value" rather than "unknown option";
/// * an option that takes no value is recorded as a switch.
///
/// # There is no `--flag=value` spelling
///
/// The WHOLE argument is the flag's name, and an `=` in it makes it a name
/// nobody has. `--serve=8080` is refused as an unrecognised option, and it
/// is not a gap in the parser: nothing has ever documented that spelling,
/// and accepting it now would be a new spelling that nobody asked for.
///
/// A parser that split on the `=` would also have to decide what
/// `--serve=8080` MEANS, and there is no answer that is not a new rule:
/// the port would be 8080 in one path and the flag would be refused in
/// another, depending on which reader looked at it.
pub fn parse(argv: &[String]) -> Result<Args, ParseError> {
    let mut args = Args::default();
    let mut seen_verb = false;
    let mut i = 0;

    while i < argv.len() {
        let item = argv[i].clone();
        if item.starts_with("--") {
            let name = item.clone();
            if !flag_is_known(&name) {
                return Err(ParseError::UnknownOption(name));
            }
            let arity = flag_arity(&name).unwrap_or(0);
            if arity == 0 {
                args.switches.push(name);
                i += 1;
                continue;
            }
            // Otherwise the value is the next argument, if it is not itself
            // an option.
            if i + 1 < argv.len() && !argv[i + 1].starts_with('-') {
                args.flags.insert(name, argv[i + 1].clone());
                i += 2;
            } else {
                // The value is missing. The flag is still RECORDED, so the
                // consumer can refuse it by name -- "the parser does not
                // know this flag" would be the wrong message for a flag the
                // binary does know.
                args.switches.push(name);
                i += 1;
            }
            continue;
        }
        if item.starts_with('-') && item.len() > 1 {
            if flag_is_known_short(&item) {
                args.switches.push("--help".to_string());
                i += 1;
                continue;
            }
            return Err(ParseError::UnknownOption(item));
        }
        // A positional.
        if !seen_verb {
            args.subcommand = Some(item);
            seen_verb = true;
        } else {
            args.positional.push(item);
        }
        i += 1;
    }
    Ok(args)
}

/// The `--serve` port and bind address.
///
/// The grammar is deliberately the loose one `serve` has always accepted:
/// `--serve [PORT]`, where PORT is the next argument and only if it starts
/// with a DIGIT. `--serve=8080` is not accepted, because nothing has ever
/// documented it.
///
/// ONE reader, and that is the point: the C++ tree had three hand-written
/// scans of the same two flags, and that is how `--serve` with no port came
/// to mean port 0 in the speech mode -- `atoi("")` is 0, and 0 is a legal
/// request to bind(), so the server started, printed its banner and
/// listened on an ephemeral port it never named.
#[derive(Debug, Clone)]
pub struct ServeTarget {
    /// The port.
    pub port: u16,
    /// The bind address.
    pub bind: String,
}

impl Default for ServeTarget {
    fn default() -> Self {
        Self {
            port: 8080,
            bind: "127.0.0.1".to_string(),
        }
    }
}

/// Read the `--serve` flag out of an argument vector.
///
/// `from` is the first index to read, because the callers do not all pass
/// the same slice: the arch modes read from 1 and the embedding path has
/// always started at 2.
pub fn read_serve(argv: &[String], from: usize) -> (Option<u16>, String, bool) {
    let mut port: Option<u16> = None;
    let mut bind = "127.0.0.1".to_string();
    let mut found = false;
    let mut i = from;
    while i < argv.len() {
        if argv[i] == "--serve" {
            found = true;
            if i + 1 < argv.len() && argv[i + 1].starts_with(|c: char| c.is_ascii_digit()) {
                port = argv[i + 1].parse().ok();
                i += 1;
            }
        }
        i += 1;
    }
    i = from;
    while i + 1 < argv.len() {
        if argv[i] == "--bind" {
            bind = argv[i + 1].clone();
        }
        i += 1;
    }
    (port, bind, found)
}
