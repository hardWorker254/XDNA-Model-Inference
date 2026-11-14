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
//! thing that does change is the implementation: the C++ tree had FOUR
//! separate HTTP implementations, each with its own accept loop, and this
//! is one router with eight routes.
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

pub mod routes;

use axum::Router;

/// The result type of this crate.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// The embedding endpoint's engine.
///
/// The tokenizer and the model, and the sequence length they run at. The
/// numbers it produces are the CPU backend's, and they are the C++'s
/// numbers: the same container through the same arithmetic gives the same
/// bytes.
#[derive(Clone)]
pub struct Embedder {
    tokenizer: npum_tokenizers::wordpiece::WordPiece,
    model: npum_cpu::bert::BertModel,
    seq: usize,
}

impl Embedder {
    /// Build an engine from a container.
    pub fn open(path: &std::path::Path, seq: usize) -> std::result::Result<Self, String> {
        let container = npum_format::Container::open(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let blob = container
            .tensor_bytes("tokenizer.vocab")
            .map_err(|_| "the container holds no vocabulary".to_string())?;
        let tokenizer = npum_tokenizers::wordpiece::WordPiece::from_lines(&String::from_utf8_lossy(blob))
            .map_err(|e| format!("the vocabulary is unusable: {e}"))?;
        let model = npum_cpu::bert::BertModel::open(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self {
            tokenizer,
            model,
            seq,
        })
    }

    /// The vocabulary's size.
    pub fn vocab_size(&self) -> usize {
        self.model.spec.vocab
    }

    /// The sequence length this engine runs at.
    pub fn seq(&self) -> usize {
        self.seq
    }

    /// Embed the texts.
    ///
    /// The vectors and the token count, which the usage block reports.
    pub fn embed(
        &self,
        texts: &[String],
    ) -> std::result::Result<(Vec<Vec<f32>>, usize), String> {
        let hidden = self.model.spec.hidden;
        let mut vectors = Vec::with_capacity(texts.len());
        let mut tokens = 0usize;
        for text in texts {
            let (encoding, _) = self
                .tokenizer
                .encode_padded(text, self.seq)
                .map_err(|e| e.to_string())?;
            tokens += encoding.mask.iter().filter(|m| **m > 0).count();
            let chunk = self
                .model
                .embed(&encoding.ids, &encoding.mask, 1, self.seq)
                .map_err(|e| e.to_string())?;
            vectors.push(chunk);
        }
        let _ = hidden;
        Ok((vectors, tokens))
    }
}

/// The eight routes, as one router.
pub fn router(state: routes::ServerState) -> Router {
    routes::router(state)
}
