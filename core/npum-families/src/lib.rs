//! The model families: what each one does and how it is run.
//!
//! # Why a family is a trait and not a flag
//!
//! The C++ runtime had `maybe_*_mode()` functions that returned `-1` when a
//! mode did not apply. A `-1` is a value the caller must remember to check,
//! and the check was forgotten once per mode per call site. The result was a
//! family that answered questions that belonged to another family.
//!
//! A trait removes the possibility: a family that does not do something does
//! not have the method, so the call does not compile.
//!
//! # Which binary owns which family
//!
//! | family | binary |
//! |---|---|
//! | embed | `npuembeddings` |
//! | stt | `npuaudio` |
//! | cls, pose, hands, mppose, ocr | `npuimage` |
//!
//! The split is decided TWICE, and both answers are needed:
//!
//! * from the SPELLING of the verb -- `transcribe` is Audio whatever the
//!   arguments say;
//! * from the ARCH of the container -- a Whisper container is Audio whatever
//!   the verb was.
//!
//! The second check matters because the flag form has no verb to check.

pub mod catalogue;
pub mod registry;

use npum_model::ModelSpec;
use serde::{Deserialize, Serialize};

/// Which family a model belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Family {
    /// A BERT-family text embedder.
    Embed,
    /// A speech-to-text model.
    Stt,
    /// An image classifier.
    Cls,
    /// A pose detector.
    Pose,
    /// A hand detector.
    Hands,
    /// A whole-body pose detector.
    MpPose,
    /// A text detector and recogniser.
    Ocr,
}

impl Family {
    /// Every family.
    pub const ALL: [Family; 7] = [
        Family::Embed,
        Family::Stt,
        Family::Cls,
        Family::Pose,
        Family::Hands,
        Family::MpPose,
        Family::Ocr,
    ];

    /// The family's name.
    pub fn as_str(self) -> &'static str {
        match self {
            Family::Embed => "embed",
            Family::Stt => "stt",
            Family::Cls => "cls",
            Family::Pose => "pose",
            Family::Hands => "hands",
            Family::MpPose => "mppose",
            Family::Ocr => "ocr",
        }
    }

    /// The binary that runs this family.
    pub fn binary(self) -> &'static str {
        match self {
            Family::Embed => "npuembeddings",
            Family::Stt => "npuaudio",
            Family::Cls | Family::Pose | Family::Hands | Family::MpPose | Family::Ocr => {
                "npuimage"
            }
        }
    }

    /// The family's arch string, as the container spells it.
    ///
    /// A container's arch names its family. `embed` is the default and covers
    /// every BERT/nomic/Gemma arch, which is also why an arch this build has
    /// no mode for stays in `npuembeddings`: an unknown arch is at worst
    /// text.
    pub fn from_arch(arch: &str) -> Option<Self> {
        match arch {
            "whisper_encdec_gelu" | "whisper_encdec" => Some(Family::Stt),
            "vit_cls" | "vit" => Some(Family::Cls),
            "yolov8_pose" | "pose" => Some(Family::Pose),
            "mediapipe_hands" | "hands" => Some(Family::Hands),
            "mediapipe_pose" | "mppose" => Some(Family::MpPose),
            "paddleocr_db_svtr_ctc" | "ocr" => Some(Family::Ocr),
            _ => Some(Family::Embed),
        }
    }
}

/// What a family needs to run one request.
pub trait ModelFamily {
    /// The family.
    fn family(&self) -> Family;

    /// Check the container is one this family can run.
    ///
    /// A refusal here is a refusal BY NAME, not a plausible answer: a
    /// container whose arch does not match is refused, and the message says
    /// which binary would run it.
    fn accepts(&self, spec: &ModelSpec) -> Result<(), String>;

    /// The endpoints this family answers over HTTP.
    fn endpoints(&self) -> &'static [&'static str];
}

/// What can go wrong when a family refuses a model.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Refused(String),
}
