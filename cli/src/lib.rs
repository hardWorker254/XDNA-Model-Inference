//! The CLI's shared body.
//!
//! Three executables, one body. Each `main()` says which family it is -- in
//! ONE line -- and calls `entry`; from there down it is one codebase. The
//! split is what a reader meets by typing: `npuaudio transcribe ...` and
//! `npuembeddings transcribe ...` cannot both be the answer to "how do I
//! transcribe?", and a binary that quietly ran a mode it was not named for
//! would leave the naming to be learned from the source.
//!
//! # The family is decided TWICE, and both answers are needed
//!
//! * from the SPELLING of the verb -- `transcribe` is Audio whatever the
//!   arguments say. Checked before any handler runs, so refusing costs
//!   nothing and no container is opened.
//! * from the ARCH of the container -- a Whisper container is Audio whatever
//!   the verb was. Checked where a path has just been resolved, because the
//!   flag form has NO verb to check.
//!
//! Both refusals quote the same roster, so a reader refused by either route
//! gets the same three lines and the same program name to type next.
//!
//! # The refusals are the CONTRACT, and they are word for word
//!
//! Every refusal below is transcribed from the C++ runtime's, and the
//! golden transcripts check them. That is deliberate: a refusal is the text
//! a reader sees when something is wrong, and it is the ONE place where a
//! wording change is felt rather than merely observed. The C++ runtime said
//! things there that this one keeps saying:
//!
//! * an unrecognised option says WHY it is refused rather than ignored --
//!   "an ignored flag changes what gets built without saying so";
//! * a pin that can never match is refused rather than stored;
//! * a container of another family names the program that runs it.

pub mod embed;
pub mod families;
pub mod flags;
pub mod hands;
pub mod help;
pub mod list;
pub mod mppose;
pub mod pose;
pub mod root;
pub mod serve;
pub mod status;
pub mod stt;
pub mod vit;

use flags::{parse, Args, ParseError};
use npum_families::registry::{owns, refuse_other_family, refuse_subcommand, ROSTER};
use std::path::PathBuf;
use std::process::ExitCode;

/// The exit code every refusal uses.
///
/// ONE code for every refusal, so that a script can tell "this refused" from
/// "this ran and produced nothing" without reading the message.
pub const EXIT_REFUSED: i32 = 2;

/// The body of `main()`, shared by all three binaries.
///
/// `argv` is the argument vector WITHOUT the program name.
pub fn entry(binary: &'static str, argv: &[String]) -> i32 {
    // The tracing subscriber is set up once, and only if the environment
    // asks for it: a runtime that logs by default makes a transcript of its
    // own output unusable, and the parity harness reads those transcripts.
    if std::env::var_os("NPUE_LOG").is_some() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init();
    }

    let args = match parse(argv) {
        Ok(args) => args,
        Err(ParseError::UnknownOption(name)) => {
            eprintln!(
                "error: unrecognised option {name}. Run `{binary} --help` for the \
                 options this build accepts; a flag that is not recognised is \
                 refused rather than ignored, because an ignored flag changes what \
                 gets built without saying so."
            );
            return EXIT_REFUSED;
        }
    };

    if args.switch("--help") {
        print!("{}", help::print_usage(binary));
        return 0;
    }

    // The verb, and whether THIS binary owns it.
    let Some(verb) = args.subcommand.as_deref() else {
        // No verb at all. The help is the answer, and it goes to stdout --
        // it is what the reader asked for by typing the program's name.
        print!("{}", help::print_usage(binary));
        return 0;
    };

    if !owns(binary, verb) {
        let owner = npum_families::registry::owner_of(verb);
        eprintln!("error: {}", refuse_subcommand(verb, owner, binary));
        return EXIT_REFUSED;
    }

    // The removed flags are refused HERE, before anything is opened and
    // before the model is even resolved.
    //
    // The C++ runtime refuses them in the subcommand's own gate, which is
    // the same place in the sequence: the gate that forwards the flags the
    // runtime reads is also the gate that refuses the ones it does not. A
    // flag that silently disappears is the failure the gate exists to
    // prevent, and a removed flag is the case where it is most likely,
    // because the command line is old.
    if let Some(reason) = flags::refuse_removed_op_flags(argv) {
        eprintln!("error: {reason}");
        return EXIT_REFUSED;
    }
    if let Some(reason) = flags::refuse_exporter_only_flags(argv) {
        eprintln!("error: {reason}");
        return EXIT_REFUSED;
    }

    // `--npu-ops` is checked against the codes this build knows, and
    // against nothing else: a misspelled code that was ignored would leave
    // the operation on the host and the user would not know.
    if let Some(listing) = args.value("--npu-ops") {
        if let Err(reason) = flags::parse_npu_ops(listing) {
            eprintln!("error: {reason}");
            return EXIT_REFUSED;
        }
    }

    // The generation is checked against the names this build knows, and it
    // is checked HERE -- before any design is looked for.
    //
    // The C++ runtime did not check it: `--dev npu3` was carried into the
    // design search and came back as a refusal that named the generation it
    // did not recognise. That is a refusal that happens to be right, and it
    // is right for the wrong reason -- the answer depends on there being no
    // design set for it.
    //
    // Here the generation is the DEVICE's, and an override that names a
    // generation that does not exist is a refusal by name.
    if let Some(device) = args.value("--dev") {
        if device != "npu1" && device != "npu2" {
            eprintln!(
                "error: --dev '{device}' is not a generation this build knows. \
                 The generations are npu1 and npu2."
            );
            return EXIT_REFUSED;
        }
    }

    let root = resolve_root(&args, argv);

    match verb {
        "list" => list::print_list(&root, binary),
        "serve" => run_serve(binary, &args, argv, &root),
        "embed" => families::embed(&args, &root),
        "add" => families::add(&args),
        "tokenize" => families::tokenize(&args, &root),
        "transcribe" => families::transcribe(&args, &root),
        "classify" => families::classify(&args, &root),
        "pose" => families::pose(&args, &root),
        "hands" => families::hands(&args, &root),
        "mppose" => families::mppose(&args, &root),
        "ocr" => families::ocr(&args, &root),
        other => {
            eprintln!("error: `{other}` is not a verb this binary has.");
            eprint!("{ROSTER}");
            EXIT_REFUSED
        }
    }
}

/// Run the shared body and turn its code into an exit status.
pub fn run(binary: &'static str) -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    ExitCode::from(entry(binary, &argv) as u8)
}

/// Where the models and the design sets are.
///
/// `--root` wins, and the default search is the C++ runtime's: the
/// executable's own directory first, and only then upwards.
fn resolve_root(args: &Args, argv: &[String]) -> PathBuf {
    if let Some(root) = args.value("--root") {
        return PathBuf::from(root);
    }
    // The flag table records `--root` as taking a value, and the parser
    // stored it -- but the flag form (`--root` after the verb) is read from
    // the argument vector as well, because the C++ runtime read it that way
    // and a command line that worked there must work here.
    let mut i = 0;
    while i + 1 < argv.len() {
        if argv[i] == "--root" {
            return PathBuf::from(&argv[i + 1]);
        }
        i += 1;
    }
    root::default_root(&argv.first().cloned().unwrap_or_default())
}

/// The refusal for a verb with too few arguments.
///
/// ONE message per verb, and it says what the verb needs.
pub fn needs_a_model(verb: &str) -> i32 {
    eprintln!("error: `{verb}` needs a model name or a path to a .npum container");
    EXIT_REFUSED
}

fn run_serve(binary: &'static str, args: &Args, argv: &[String], root: &PathBuf) -> i32 {
    let Some(model) = args.positional.first() else {
        return needs_a_model("serve");
    };
    let (port, bind, _found) = flags::read_serve(argv, 1);
    let port = args
        .value_i64("--port")
        .map(|p| p as u16)
        .or(port)
        .unwrap_or(8080);
    let bind = args
        .value("--bind")
        .map(|s| s.to_string())
        .unwrap_or(bind);
    serve::serve(binary, model, port, &bind, root, args)
}

/// The refusal for a container of another family, re-exported for the
/// families module.
pub fn refuse_family(path: &str, arch: &str, owner: npum_families::Family, binary: &str) -> String {
    refuse_other_family(path, arch, owner, binary)
}
