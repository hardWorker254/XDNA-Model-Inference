//! Concrete execution engines used by HTTP routes.
//!
//! This module owns model/container loading and request execution for the
//! families exposed by the server. Route handlers and HTTP DTO/error mapping live
//! in `routes`; this layer focuses on turning requests into model outputs.

/// The speech endpoint's engine.
///
/// The encoder, the decoder with its KV cache, the tokenizer and the front
/// end's weights. The decoder is behind a lock because it holds state --
/// the cache and the "first generated step" counter the decoding policy is
/// keyed on -- and two requests must not share it: a request that fed its
/// tokens into another request's cache would answer with the other
/// request's words.
///
/// The work is the CPU backend's and the numbers are the reference's numbers: the
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
        let container =
            npum_format::Container::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
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
        let net =
            npum_cpu::pose::Network::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
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
fn tensor_f32(
    container: &npum_format::Container,
    name: &str,
) -> std::result::Result<Vec<f32>, String> {
    let bytes = container
        .tensor_bytes(name)
        .map_err(|_| format!("the container holds no tensor named {name}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[derive(Clone)]
enum EmbedTokenizer {
    WordPiece(npum_tokenizers::wordpiece::WordPiece),
    Xlmr(npum_tokenizers::xlmr::XlmrTokenizer),
}

impl EmbedTokenizer {
    fn encode_padded(
        &self,
        text: &str,
        max_len: usize,
    ) -> npum_tokenizers::Result<(npum_tokenizers::Encoding, bool)> {
        match self {
            EmbedTokenizer::WordPiece(t) => t.encode_padded(text, max_len),
            EmbedTokenizer::Xlmr(t) => t.encode_padded(text, max_len),
        }
    }
}

#[derive(Clone)]
enum EmbedModel {
    Bert(npum_cpu::bert::BertModel),
    Gte(npum_cpu::gte::GteModel),
}

/// The embedding endpoint's engine.
///
/// The tokenizer and the model, and the sequence length they run at.
/// By default the arithmetic is the host's (CPU backend bytes, the reference
/// reference). When Vulkan ops are selected, the chosen operations run on
/// the GPU through the same `Ops` seam.
#[derive(Clone)]
pub struct Embedder {
    tokenizer: EmbedTokenizer,
    model: EmbedModel,
    seq: usize,
    vulkan_ops: Option<(bool, bool, bool, bool)>,
}

impl Embedder {
    /// Build an engine from a container (host arithmetic).
    pub fn open(path: &std::path::Path, seq: usize) -> std::result::Result<Self, String> {
        Self::open_with_vulkan(path, seq, None)
    }

    /// Build an engine from a container, optionally selecting Vulkan ops
    /// `(gemm, layn, gelu, softm)`.
    pub fn open_with_vulkan(
        path: &std::path::Path,
        seq: usize,
        vulkan_ops: Option<(bool, bool, bool, bool)>,
    ) -> std::result::Result<Self, String> {
        let container =
            npum_format::Container::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let arch = container.config_str("arch").unwrap_or_default().to_string();
        let tokenizer = if arch.contains("gte_new_rope")
            || (arch.contains("gte") && arch.contains("rope"))
        {
            let blob = container
                .tensor_bytes("tokenizer.xlmr_table")
                .map_err(|_| {
                    "the container holds no XLM-R tokenizer table (tokenizer.xlmr_table)"
                        .to_string()
                })?;
            EmbedTokenizer::Xlmr(npum_tokenizers::xlmr::XlmrTokenizer::from_table_bytes(
                blob,
            )?)
        } else {
            let blob = container
                .tensor_bytes("tokenizer.vocab")
                .map_err(|_| "the container holds no vocabulary".to_string())?;
            EmbedTokenizer::WordPiece(
                npum_tokenizers::wordpiece::WordPiece::from_lines(&String::from_utf8_lossy(blob))
                    .map_err(|e| format!("the vocabulary is unusable: {e}"))?,
            )
        };
        let model =
            if arch == "gte_new_rope_geglu" || (arch.contains("gte") && arch.contains("rope")) {
                EmbedModel::Gte(
                    npum_cpu::gte::GteModel::open(path)
                        .map_err(|e| format!("{}: {e}", path.display()))?,
                )
            } else {
                EmbedModel::Bert(
                    npum_cpu::bert::BertModel::open(path)
                        .map_err(|e| format!("{}: {e}", path.display()))?,
                )
            };
        Ok(Self {
            tokenizer,
            model,
            seq,
            vulkan_ops,
        })
    }

    /// The vocabulary's size.
    pub fn vocab_size(&self) -> usize {
        match &self.model {
            EmbedModel::Bert(m) => m.spec.vocab,
            EmbedModel::Gte(m) => m.spec.vocab,
        }
    }

    /// The sequence length this engine runs at.
    pub fn seq(&self) -> usize {
        self.seq
    }

    /// Embed the texts.
    ///
    /// The vectors and the token count, which the usage block reports.
    pub fn embed(&self, texts: &[String]) -> std::result::Result<(Vec<Vec<f32>>, usize), String> {
        let mut vectors = Vec::with_capacity(texts.len());
        let mut tokens = 0usize;
        for text in texts {
            let (encoding, _) = self
                .tokenizer
                .encode_padded(text, self.seq)
                .map_err(|e| e.to_string())?;
            tokens += encoding.mask.iter().filter(|m| **m > 0).count();
            let chunk = match &self.model {
                EmbedModel::Bert(model) => {
                    if let Some((gemm, layn, gelu, softm)) = self.vulkan_ops {
                        let mut ops =
                            npum_vulkan::ops::VulkanOps::with_ops(gemm, layn, gelu, softm);
                        if !ops.has_gpu() {
                            return Err(
                                "no Vulkan device was opened on this machine for --vk-ops/--backend vulkan"
                                    .to_string(),
                            );
                        }
                        model
                            .embed(
                                &encoding.ids,
                                &encoding.mask,
                                1,
                                self.seq,
                                &mut ops,
                                None,
                                None,
                            )
                            .map_err(|e| e.to_string())?
                    } else {
                        let mut ops = npum_cpu::ops::CpuOps::best();
                        model
                            .embed(
                                &encoding.ids,
                                &encoding.mask,
                                1,
                                self.seq,
                                &mut ops,
                                None,
                                None,
                            )
                            .map_err(|e| e.to_string())?
                    }
                }
                EmbedModel::Gte(model) => {
                    if let Some((gemm, layn, gelu, softm)) = self.vulkan_ops {
                        let mut ops =
                            npum_vulkan::ops::VulkanOps::with_ops(gemm, layn, gelu, softm);
                        if !ops.has_gpu() {
                            return Err(
                                "no Vulkan device was opened on this machine for --vk-ops/--backend vulkan"
                                    .to_string(),
                            );
                        }
                        let mut timings = npum_cpu::gte::Timings::default();
                        model
                            .embed_timed_with(
                                &mut ops,
                                &mut timings,
                                &encoding.ids,
                                &encoding.mask,
                                1,
                                self.seq,
                                None,
                                None,
                            )
                            .map_err(|e| e.to_string())?
                    } else {
                        let mut ops = npum_cpu::ops::CpuOps::best();
                        let mut timings = npum_cpu::gte::Timings::default();
                        model
                            .embed_timed_with(
                                &mut ops,
                                &mut timings,
                                &encoding.ids,
                                &encoding.mask,
                                1,
                                self.seq,
                                None,
                                None,
                            )
                            .map_err(|e| e.to_string())?
                    }
                }
            };
            vectors.push(chunk);
        }
        Ok((vectors, tokens))
    }
}
