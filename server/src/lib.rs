//! The HTTP endpoints.
//!
//! # The eight, and they do not change
//!
//! | method | path | family |
//! |---|---|---|
//! | POST | `/v1/embeddings` | embed |
//! | POST | `/v1/classify` | ViT |
//! | POST | `/v1/pose` | pose / hands / mppose |
//! | POST | `/v1/audio/transcriptions` | Whisper |
//! | GET | `/v1/models` | the catalogue |
//! | GET | `/v1/labels` | the classifier's vocabulary |
//! | GET | `/health` | liveness |
//! | GET | `/` | the root |
//!
//! The response BODIES are 1:1 -- the same JSON, the same keys, the same
//! numbers -- and so are the status codes and the `Content-Type`. The one
//! thing that does change is the implementation: the previous implementation had FOUR
//! separate HTTP implementations, each with its own accept loop, and this
//! is one router with eight routes.
//!
//! # Composition
//!
//! * [`routes`] owns HTTP routing, request/response DTO mapping and
//!   protocol-level errors.
//! * [`engines`] owns concrete model/container loading and request execution.
//! * This crate root is the composition surface: it wires modules and exposes
//!   the router and engine types.
//!
//! # The mode is chosen by the CONTAINER'S ARCH, never by a flag
//!
//! `serve` is the one verb all three binaries have, and the endpoint is
//! picked by what the container is. A binary serving a container of another
//! family is refused BY NAME, before the listener opens -- rather than
//! answered by an endpoint this process is not named for.
//!
//! # A wrong path is a 404 that names the right one
//!
//! Never another endpoint's answer. A client that posts to `/v1/embeddings`
//! on a Whisper container gets a 404 that says which endpoint to use, and
//! not a transcription.

pub mod engines;
pub mod multipart;
pub mod routes;

use axum::Router;

pub use engines::{
    Classification, Classifier, DEFAULT_DECODER_POOL, Embedder, PoseDetector, PoseResult,
    Transcriber,
};

/// The result type of this crate.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// The eight routes, as one router.
pub fn router(state: routes::ServerState) -> Router {
    routes::router(state)
}
