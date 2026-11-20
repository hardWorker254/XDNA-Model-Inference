//! Family/reference runtime pipelines.
//!
//! These modules implement model-family execution on top of CPU arithmetic and
//! shared core abstractions (loading, encode/decode/session loops, family-specific
//! tensor flows and postprocessing helpers). They are not low-level ISA kernels.

pub mod bert;
pub mod dwconv_array;
pub mod gemma;
pub mod gte;
pub mod hands;
pub mod hands_decode;
pub mod mppose;
pub mod mppose_decode;
pub mod ocr;
pub mod ocr_decode;
pub mod ocr_session;
pub mod operand;
pub mod pose;
pub mod pose_decode;
pub mod vit;
pub mod whisper;
pub mod whisper_session;
pub mod whisper_tokenizer;
