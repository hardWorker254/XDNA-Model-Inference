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

/// The classifier endpoint's engine.
///
/// The ViT's weights and the geometry they describe. Unlike the speech
/// engine this one holds no STATE between requests -- the encoder is a
/// function of the image -- so it needs no lock.
pub struct Classifier {
    vit: npum_cpu::vit::Vit,
}

impl Classifier {
    /// Build an engine from a container.
    pub fn open(path: &std::path::Path) -> std::result::Result<Self, String> {
        let vit = npum_cpu::vit::Vit::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self { vit })
    }

    /// How many labels.
    pub fn num_labels(&self) -> usize {
        self.vit.geometry().num_labels as usize
    }

    /// The input's side, in pixels.
    pub fn image_size(&self) -> i64 {
        self.vit.geometry().image_size
    }

    /// The patch's side, in pixels.
    pub fn patch_size(&self) -> i64 {
        self.vit.geometry().patch_size
    }

    /// Classify an image's BYTES.
    ///
    /// The name is what the refusals name: the upload's name, so a client
    /// that sent the wrong file is told which file.
    ///
    /// The three timings are this run's measurement of this image, and they
    /// are reported because the answer's cost is part of the answer: a
    /// client that sizes a fleet reads them.
    pub fn classify_bytes(
        &self,
        image: &[u8],
        name: &str,
    ) -> std::result::Result<Classification, String> {
        let decoded = npum_cpu::vit_image::decode_image_bytes(image, name)?;
        let geometry = self.vit.geometry();
        let front = std::time::Instant::now();
        let patches = npum_cpu::vit_image::preprocess(&decoded, geometry)?;
        let front_end_s = front.elapsed().as_secs_f64();
        let enc = std::time::Instant::now();
        let hidden = self.vit.encode(&patches);
        let encoder_s = enc.elapsed().as_secs_f64();
        let head = std::time::Instant::now();
        let d = geometry.hidden as usize;
        let logits = self.vit.classify(&hidden[..d]);
        let prediction = npum_cpu::vit::predict(&logits, self.vit.labels())?;
        let head_s = head.elapsed().as_secs_f64();
        Ok(Classification {
            prediction,
            front_end_s,
            encoder_s,
            head_s,
        })
    }
}

/// One classification and what it cost.
pub struct Classification {
    /// The label, its name and its probability.
    pub prediction: npum_cpu::vit::Prediction,
    /// The front end's time.
    pub front_end_s: f64,
    /// The encoder's time.
    pub encoder_s: f64,
    /// The head's time.
    pub head_s: f64,
}

/// The pose endpoint's engine.
///
/// The YOLOv8-pose network and the geometry it describes. Like the
/// classifier this holds no state between requests -- a detection is a
/// function of the image and the thresholds -- so it needs no lock.
pub struct PoseDetector {
    net: npum_cpu::pose::Network,
}

impl PoseDetector {
    /// Build an engine from a container.
    pub fn open(path: &std::path::Path) -> std::result::Result<Self, String> {
        let net = npum_cpu::pose::Network::open(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self { net })
    }

    /// The model's geometry.
    pub fn geometry(&self) -> &npum_cpu::pose::Geometry {
        self.net.geometry()
    }

    /// How many threads the host convolutions use.
    pub fn threads(&self) -> usize {
        self.net.threads()
    }

    /// Detect people in an image's BYTES.
    ///
    /// The name is what the answer calls the image: the upload's filename,
    /// so a client that sent several pictures can tell the answers apart.
    pub fn detect_bytes(
        &self,
        image: &[u8],
        name: &str,
        params: &npum_cpu::pose_decode::DecodeParams,
    ) -> std::result::Result<PoseResult, String> {
        let all = std::time::Instant::now();
        let decoded = npum_cpu::vit_image::decode_image_bytes(image, name)?;
        let geometry = self.net.geometry();
        let front = std::time::Instant::now();
        let (input, letterbox) = npum_cpu::pose_decode::letterbox_normalise(&decoded, geometry)?;
        let front_end_s = front.elapsed().as_secs_f64();
        let net = std::time::Instant::now();
        let (head, _) = self.net.run_timed(&input)?;
        let network_s = net.elapsed().as_secs_f64();
        let dec = std::time::Instant::now();
        let people = npum_cpu::pose_decode::decode(&head, geometry, &letterbox, params)?;
        let decode_s = dec.elapsed().as_secs_f64();
        Ok(PoseResult {
            people,
            width: letterbox.src_w,
            height: letterbox.src_h,
            scale: letterbox.scale,
            pad_x: letterbox.pad_x,
            pad_y: letterbox.pad_y,
            front_end_s,
            network_s,
            decode_s,
            total_s: all.elapsed().as_secs_f64(),
        })
    }
}

/// One pose request and what it cost.
pub struct PoseResult {
    /// The people, in the source image's pixels.
    pub people: Vec<npum_cpu::pose_decode::Person>,
    /// The source's width.
    pub width: i64,
    /// The source's height.
    pub height: i64,
    /// Model pixels per source pixel.
    pub scale: f64,
    /// The left padding, in model pixels.
    pub pad_x: i64,
    /// The top padding, in model pixels.
    pub pad_y: i64,
    /// The front end's time.
    pub front_end_s: f64,
    /// The network's time.
    pub network_s: f64,
    /// The decode's time.
    pub decode_s: f64,
    /// The whole request's time.
    pub total_s: f64,
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
            // THE ARITHMETIC GOES THROUGH THE SEAM, and on the server it is
            // the host's: the HTTP path answers with the CPU backend's
            // bytes, which are the C++'s bytes.
            let mut ops = npum_cpu::ops::CpuOps::best();
            let chunk = self
                .model
                .embed(&encoding.ids, &encoding.mask, 1, self.seq, &mut ops, None, None)
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
