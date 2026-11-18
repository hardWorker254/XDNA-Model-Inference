//! Input decoding and frontend preprocessing helpers.
//!
//! This module groups media/file decoding and model input preparation used by
//! CPU reference pipelines (e.g. image decode/preprocess, WAV/mel frontends).
//! Low-level math kernels remain in `crate::kernels`.

pub mod vit_image;
pub mod whisper_features;
