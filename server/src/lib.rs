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

pub mod multipart;
pub mod routes;

use axum::Router;

/// The result type of this crate.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// The speech endpoint's engine.
///
/// The encoder, the decoder with its KV cache, the tokenizer and the front
/// end's weights. The decoder is behind a lock because it holds state --
/// the cache and the "first generated step" counter the decoding policy is
/// keyed on -- and two requests must not share it: a request that fed its
/// tokens into another request's cache would answer with the other
/// request's words.
///
/// The work is the CPU backend's and the numbers are the C++'s numbers: the
/// same container through the same arithmetic gives the same bytes.
pub struct Transcriber {
    encoder: npum_cpu::whisper::Encoder,
    decoder: std::sync::Mutex<npum_cpu::whisper::Decoder>,
    tokenizer: npum_cpu::whisper_tokenizer::Tokenizer,
    weights: npum_cpu::whisper_session::FrontEndWeights,
    options: npum_cpu::whisper_session::Options,
}

impl Transcriber {
    /// Build an engine from a container.
    pub fn open(
        path: &std::path::Path,
        options: npum_cpu::whisper_session::Options,
    ) -> std::result::Result<Self, String> {
        let container = npum_format::Container::open(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let encoder = npum_cpu::whisper::Encoder::open(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let decoder = npum_cpu::whisper::Decoder::open(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let table = container
            .tensor_bytes("tokenizer.whisper_table")
            .map_err(|_| format!("{}: the container holds no tokenizer table", path.display()))?;
        let tokenizer = npum_cpu::whisper_tokenizer::Tokenizer::from_table(table)?;
        let weights = npum_cpu::whisper_session::FrontEndWeights {
            conv1_weight: tensor_f32(&container, "frontend.conv1.weight")?,
            conv1_bias: tensor_f32(&container, "frontend.conv1.bias")?,
            conv2_weight: tensor_f32(&container, "frontend.conv2.weight")?,
            conv2_bias: tensor_f32(&container, "frontend.conv2.bias")?,
        };
        Ok(Self {
            encoder,
            decoder: std::sync::Mutex::new(decoder),
            tokenizer,
            weights,
            options,
        })
    }

    /// The vocabulary's size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.len()
    }

    /// The encoder's position count.
    pub fn positions(&self) -> usize {
        self.encoder.geometry().max_seq
    }

    /// How many mel bands.
    pub fn mel_bins(&self) -> usize {
        self.encoder.geometry().mel_bins
    }

    /// Transcribe the audio in a WAV file's BYTES.
    ///
    /// The name is what the refusals name: the upload's name, so a client
    /// that sent the wrong file is told which file.
    pub fn transcribe_bytes(
        &self,
        audio: &[u8],
        name: &str,
        options: &npum_cpu::whisper_session::Options,
    ) -> std::result::Result<npum_cpu::whisper_session::Transcript, String> {
        let read = npum_cpu::whisper_features::read_wav_bytes(audio, name).map_err(|r| r.0)?;
        let mut decoder = self
            .decoder
            .lock()
            .map_err(|_| "the decoder's state is poisoned".to_string())?;
        npum_cpu::whisper_session::transcribe_samples(
            &self.encoder,
            &mut decoder,
            &self.tokenizer,
            &read.samples,
            options,
            &self.weights,
        )
    }

    /// The options this engine runs with by default.
    pub fn options(&self) -> &npum_cpu::whisper_session::Options {
        &self.options
    }
}

/// A tensor's values.
fn tensor_f32(container: &npum_format::Container, name: &str) -> std::result::Result<Vec<f32>, String> {
    let bytes = container
        .tensor_bytes(name)
        .map_err(|_| format!("the container holds no tensor named {name}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

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
