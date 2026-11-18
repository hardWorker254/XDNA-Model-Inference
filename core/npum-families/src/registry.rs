//! The family registry: which family runs which container.
//!
//! # The refusal, and what it says
//!
//! The reference runtime's refusal named the OTHER binary and printed the roster:
//!
//! ```text
//! models/whisper-base.npue is `whisper_encdec_gelu`, which npuaudio runs,
//! and this is npuembeddings.
//!     npuembeddings   embed, list, add, tokenize, serve
//!     npuaudio        transcribe, serve
//!     npuimage        classify, pose, hands, mppose, ocr, serve
//! ```
//!
//! Three lines, the same three lines every time, from one string. A refusal
//! that names a program the help text does not list is the failure the split
//! exists to make impossible, and the roster is one constant so that the
//! help text and the refusals cannot drift apart.

use crate::Family;

/// The roster, as one string.
///
/// Every refusal quotes this, so a reader refused by any route gets the same
/// three lines and the same program name to type next.
pub const ROSTER: &str = "    npuembeddings   embed, list, add, tokenize, serve\n\
                          \x20   npuaudio        transcribe, serve\n\
                          \x20   npuimage        classify, pose, hands, mppose, ocr, serve";

/// The binary that runs a family.
pub fn binary_of(family: Family) -> &'static str {
    family.binary()
}

/// The family a container's arch names.
pub fn family_of_arch(arch: &str) -> Family {
    Family::from_arch(arch).unwrap_or(Family::Embed)
}

/// The refusal for a container of another family.
///
/// `path` is the container, `arch` its arch, `owner` the family that runs
/// it, and `this_binary` the binary that was asked.
pub fn refuse_other_family(path: &str, arch: &str, owner: Family, this_binary: &str) -> String {
    format!(
        "{path} is `{arch}`, which {} runs, and this is {this_binary}.\n{ROSTER}",
        binary_of(owner)
    )
}

/// The refusal for a verb that belongs to another binary.
pub fn refuse_subcommand(sub: &str, owner: Family, this_binary: &str) -> String {
    format!(
        "`{sub}` runs in {}, and this is {this_binary}.\n{ROSTER}",
        binary_of(owner)
    )
}

/// Which binary a verb belongs to, for the refusal to name.
pub fn owner_of(sub: &str) -> Family {
    match sub {
        "transcribe" => Family::Stt,
        "classify" => Family::Cls,
        "pose" => Family::Pose,
        "hands" => Family::Hands,
        "mppose" => Family::MpPose,
        "ocr" => Family::Ocr,
        _ => Family::Embed,
    }
}

/// Whether a binary owns a verb.
///
/// `serve` is owned by every family -- the endpoint is picked by the
/// container's arch -- and no other verb is.
pub fn owns(binary: &str, sub: &str) -> bool {
    match binary {
        "npuaudio" => sub == "transcribe" || sub == "serve",
        "npuimage" => matches!(
            sub,
            "classify" | "pose" | "hands" | "mppose" | "ocr" | "serve"
        ),
        _ => matches!(sub, "embed" | "list" | "add" | "tokenize" | "serve"),
    }
}

/// The verbs a binary owns, as one line of prose.
pub fn verbs_of(binary: &str) -> &'static str {
    match binary {
        "npuaudio" => "transcribe, serve",
        "npuimage" => "classify, pose, hands, mppose, ocr, serve",
        _ => "embed, list, add, tokenize, serve",
    }
}
