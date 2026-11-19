//! Concrete execution engines used by HTTP routes.
//!
//! This module owns model/container loading and request execution for the
//! families exposed by the server. Route handlers and HTTP DTO/error mapping live
//! in `routes`; this layer focuses on turning requests into model outputs.

use std::sync::Arc;
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::Duration;

/// The speech endpoint's engine.
///
/// The encoder, the tokenizer and the front end's weights are shared, while
/// each decoder instance keeps its own KV cache/state and is guarded by a
/// lock. A pool larger than one allows independent requests to decode in
/// parallel at the cost of loading the decoder weights/state multiple times.
///
/// The work is the CPU backend's and the numbers are the reference's numbers: the
/// same container through the same arithmetic gives the same bytes.
pub struct Transcriber {
    encoder: npum_cpu::whisper::Encoder,
    decoders: Vec<Mutex<npum_cpu::whisper::Decoder>>,
    tokenizer: npum_cpu::whisper_tokenizer::Tokenizer,
    weights: npum_cpu::whisper_session::FrontEndWeights,
    options: npum_cpu::whisper_session::Options,
    /// Bumped every time a decoder is released, and the predicate the waiters
    /// sleep on. The counter is read BEFORE the `try_lock` sweep, so a release
    /// in the window between that sweep and the wait cannot be missed.
    avail_gen: Mutex<u64>,
    avail_cv: Condvar,
}

impl Transcriber {
    /// Build an engine from a container.
    pub fn open(
        path: &std::path::Path,
        options: npum_cpu::whisper_session::Options,
    ) -> std::result::Result<Self, String> {
        Self::open_with_pool(path, options, 1)
    }

    /// Build an engine from a container with a pool of decoder instances.
    ///
    /// `pool` is clamped to at least one. Each instance keeps its own
    /// KV cache/state, so a larger pool enables concurrent transcriptions
    /// but uses more memory.
    pub fn open_with_pool(
        path: &std::path::Path,
        options: npum_cpu::whisper_session::Options,
        pool: usize,
    ) -> std::result::Result<Self, String> {
        let container =
            npum_format::Container::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let encoder = npum_cpu::whisper::Encoder::open(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let pool = pool.max(1);
        let mut decoders = Vec::with_capacity(pool);
        for _ in 0..pool {
            decoders.push(Mutex::new(
                npum_cpu::whisper::Decoder::open(path)
                    .map_err(|e| format!("{}: {e}", path.display()))?,
            ));
        }
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
            decoders,
            tokenizer,
            weights,
            options,
            avail_gen: Mutex::new(0),
            avail_cv: Condvar::new(),
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

    /// Wait for ANY free decoder, not just the first one.
    ///
    /// The old strategy tried every instance once and then queued on
    /// `decoders[0]`. With a pool larger than one that turns all overflow
    /// requests into a FIFO behind decoder0 even when decoder1 frees up first;
    /// this waits on a condition variable that EVERY release signals, so the
    /// next request takes whichever instance actually becomes idle. The
    /// timeout is a guard against a lost wake-up, not the scheduling policy:
    /// the counter read before the sweep closes the race with a release that
    /// lands between the sweep and the wait.
    fn acquire_decoder(
        &self,
    ) -> std::result::Result<MutexGuard<'_, npum_cpu::whisper::Decoder>, String> {
        let poisoned = || format!("all {} decoder instances are poisoned", self.decoders.len());
        loop {
            let generation = *self.avail_gen.lock().map_err(|_| poisoned())?;
            if let Some(guard) = self.decoders.iter().find_map(|m| m.try_lock().ok()) {
                return Ok(guard);
            }
            if self.decoders.iter().all(Mutex::is_poisoned) {
                return Err(poisoned());
            }
            let guard = self.avail_gen.lock().map_err(|_| poisoned())?;
            let (guard, _) = self
                .avail_cv
                .wait_timeout_while(guard, Duration::from_millis(20), |current| {
                    *current == generation
                })
                .map_err(|_| poisoned())?;
            drop(guard);
        }
    }

    /// Publish a release to every waiter. Called after the guard is dropped,
    /// so a woken waiter can actually take the instance it raced for.
    fn notify_decoder_free(&self) {
        let mut generation = match self.avail_gen.lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };
        *generation += 1;
        drop(generation);
        self.avail_cv.notify_all();
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
        let result = {
            let mut decoder = self.acquire_decoder()?;
            npum_cpu::whisper_session::transcribe_samples(
                &self.encoder,
                &mut decoder,
                &self.tokenizer,
                &read.samples,
                options,
                &self.weights,
            )
        };
        self.notify_decoder_free();
        result
    }

    /// The options this engine runs with by default.
    pub fn options(&self) -> &npum_cpu::whisper_session::Options {
        &self.options
    }
}

/// The classifier endpoint's engine.
///
/// One classification's reusable scratch. The encoder used to allocate eight
/// large buffers per layer and two result vectors per image; those now live
/// here and are handed to `encode_into` / `classify_into`.
struct ClassifierScratch {
    workspace: npum_cpu::vit::VitWorkspace,
    hidden: Vec<f32>,
    logits: Vec<f32>,
}

impl ClassifierScratch {
    fn new() -> Self {
        Self {
            workspace: npum_cpu::vit::VitWorkspace::new(),
            hidden: Vec::new(),
            logits: Vec::new(),
        }
    }
}

/// The ViT's weights and the geometry they describe, plus a small pool of
/// reusable encoder scratch. Scratch is state about ALLOCATION, not about the
/// model: the encoder is still a pure function of the image. A second slot
/// lets two classifications overlap; a third request falls back to a local
/// scratch rather than waiting behind a lock.
pub struct Classifier {
    vit: npum_cpu::vit::Vit,
    scratch: Vec<std::sync::Mutex<ClassifierScratch>>,
}

impl Classifier {
    /// Build an engine from a container.
    pub fn open(path: &std::path::Path) -> std::result::Result<Self, String> {
        let vit = npum_cpu::vit::Vit::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self {
            vit,
            scratch: (0..2)
                .map(|_| std::sync::Mutex::new(ClassifierScratch::new()))
                .collect(),
        })
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
        // Take a free scratch slot; when both are busy, use a local one.
        // The encoding itself is unchanged -- same kernels, same order.
        let mut fallback = std::cell::OnceCell::new();
        let mut slot = self.scratch.iter().find_map(|m| m.try_lock().ok());
        let scratch: &mut ClassifierScratch = match slot.as_mut() {
            Some(guard) => &mut **guard,
            None => {
                let _ = fallback.set(ClassifierScratch::new());
                fallback.get_mut().unwrap()
            }
        };
        let enc = std::time::Instant::now();
        self.vit
            .encode_into(&patches, &mut scratch.workspace, &mut scratch.hidden);
        let encoder_s = enc.elapsed().as_secs_f64();
        let head = std::time::Instant::now();
        let d = geometry.hidden as usize;
        let nlab = self.vit.geometry().num_labels as usize;
        scratch.logits.resize(nlab, 0.0);
        self.vit
            .classify_into(&scratch.hidden[..d], &mut scratch.logits[..nlab]);
        let prediction = npum_cpu::vit::predict(&scratch.logits[..nlab], self.vit.labels())?;
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
    /// The device context, opened once and shared by every request.
    ///
    /// `VulkanOps::with_ops` opens a Vulkan device. Doing that per request
    /// paid device setup on each HTTP call even though the model, the shaders
    /// and the queues are the same every time. The `Option` inside the lock
    /// is empty until either `with_vulkan_device` hands in the device the
    /// startup probe opened or the first GPU request opens one, so a
    /// host-only server stays host-only. The lock is held for the whole batch
    /// loop, which also keeps two requests from issuing work to the same
    /// device context at once.
    vulkan: Option<Arc<std::sync::Mutex<Option<npum_vulkan::ops::VulkanOps>>>>,
    /// Reusable BERT scratch and output, shared across requests by cloning
    /// the `Arc` (the engine itself is cloned into every `ServerState`).
    /// Two slots let two requests run concurrently; a third allocates a
    /// local scratch instead of waiting.
    bert_scratch: Arc<Vec<std::sync::Mutex<BertSlot>>>,
}

type BertSlot = (npum_cpu::bert::BertWorkspace, Vec<f32>);

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
            vulkan: vulkan_ops.map(|_| Arc::new(std::sync::Mutex::new(None))),
            vulkan_ops,
            bert_scratch: Arc::new(
                (0..2)
                    .map(|_| {
                        std::sync::Mutex::new((npum_cpu::bert::BertWorkspace::new(), Vec::new()))
                    })
                    .collect(),
            ),
        })
    }

    /// Give the engine a Vulkan device that is already open.
    ///
    /// `serve` probes for a GPU before it opens a model, and the probe is a
    /// real `VulkanOps`. Keeping that object here means the first request
    /// reuses the device the startup check opened instead of opening a
    /// second one. A host-only engine ignores it.
    pub fn with_vulkan_device(self, device: Option<npum_vulkan::ops::VulkanOps>) -> Self {
        if let (Some(shared), Some(device)) = (self.vulkan.as_ref(), device) {
            *shared.lock().expect("the Vulkan device mutex was poisoned") = Some(device);
        }
        self
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

    /// Run `f` with a free BERT scratch slot (or a local one when both are
    /// in use). The vectors are written by `f` into the caller's `vectors`;
    /// only the scratch is reused, never the arithmetic.
    fn with_bert_slot<R>(&self, f: impl FnOnce(&mut BertSlot) -> R) -> R {
        let mut slot = self.bert_scratch.iter().find_map(|m| m.try_lock().ok());
        if let Some(guard) = slot.as_mut() {
            f(&mut **guard)
        } else {
            f(&mut (npum_cpu::bert::BertWorkspace::new(), Vec::new()))
        }
    }

    /// Embed the texts.
    ///
    /// The vectors and the token count, which the usage block reports.
    pub fn embed(&self, texts: &[String]) -> std::result::Result<(Vec<Vec<f32>>, usize), String> {
        if texts.is_empty() {
            return Ok((Vec::new(), 0));
        }

        // Tokenise and pad every text up front so the model can be run in
        // batches instead of one forward pass per input. The Vulkan device
        // opened for the first such request stays alive across all later
        // batches and requests; the CPU ops instance is per request.
        let mut all_ids = Vec::with_capacity(texts.len() * self.seq);
        let mut all_mask = Vec::with_capacity(texts.len() * self.seq);
        let mut tokens = 0usize;
        for text in texts {
            let (encoding, _) = self
                .tokenizer
                .encode_padded(text, self.seq)
                .map_err(|e| e.to_string())?;
            tokens += encoding.mask.iter().filter(|m| **m > 0).count();
            all_ids.extend_from_slice(&encoding.ids);
            all_mask.extend_from_slice(&encoding.mask);
        }

        let hidden = match &self.model {
            EmbedModel::Bert(model) => model.spec.hidden,
            EmbedModel::Gte(model) => model.spec.hidden,
        };

        // Bound workspace/activation growth for large requests while still
        // amortising tokenisation, dispatch and device setup.
        const BATCH: usize = 16;

        let mut vectors = Vec::with_capacity(texts.len());
        let mut offset = 0usize;
        if let (Some((gemm, layn, gelu, softm)), Some(shared)) =
            (self.vulkan_ops, self.vulkan.as_ref())
        {
            // OPEN ONCE. Every later request reuses the same device context
            // instead of paying Vulkan setup again. The open/check is a short
            // critical section; below, the lock is taken PER BATCH so a second
            // request can slip in between this one's batches instead of
            // waiting for the whole request to finish. The device is still
            // touched by only one thread at a time.
            {
                let mut guard = shared.lock().expect("the Vulkan device mutex was poisoned");
                let ops = guard.get_or_insert_with(|| {
                    npum_vulkan::ops::VulkanOps::with_ops(gemm, layn, gelu, softm)
                });
                if !ops.has_gpu() {
                    return Err(
                        "no Vulkan device was opened on this machine for --vk-ops/--backend vulkan"
                            .to_string(),
                    );
                }
            }
            while offset < texts.len() {
                let batch = (texts.len() - offset).min(BATCH);
                let start = offset * self.seq;
                let end = (offset + batch) * self.seq;
                let ids = &all_ids[start..end];
                let mask = &all_mask[start..end];
                let mut guard = shared.lock().expect("the Vulkan device mutex was poisoned");
                let ops = guard
                    .as_mut()
                    .expect("the Vulkan device was opened above and is never removed");
                self.with_bert_slot(|(workspace, flat)| {
                    match &self.model {
                        EmbedModel::Bert(model) => {
                            model
                                .embed_into(
                                    ids, mask, batch, self.seq, ops, None, None, workspace, flat,
                                )
                                .map_err(|e| e.to_string())?;
                        }
                        EmbedModel::Gte(model) => {
                            let mut timings = npum_cpu::gte::Timings::default();
                            model
                                .embed_timed_with(
                                    ops,
                                    &mut timings,
                                    ids,
                                    mask,
                                    batch,
                                    self.seq,
                                    None,
                                    None,
                                )
                                .map_err(|e| e.to_string())?;
                        }
                    }
                    for i in 0..batch {
                        vectors.push(flat[i * hidden..(i + 1) * hidden].to_vec());
                    }
                    Ok::<(), String>(())
                })?;
                offset += batch;
            }
        } else {
            let mut ops = npum_cpu::ops::CpuOps::best();
            while offset < texts.len() {
                let batch = (texts.len() - offset).min(BATCH);
                let start = offset * self.seq;
                let end = (offset + batch) * self.seq;
                let ids = &all_ids[start..end];
                let mask = &all_mask[start..end];
                self.with_bert_slot(|(workspace, flat)| {
                    match &self.model {
                        EmbedModel::Bert(model) => model
                            .embed_into(
                                ids, mask, batch, self.seq, &mut ops, None, None, workspace, flat,
                            )
                            .map_err(|e| e.to_string())?,
                        EmbedModel::Gte(model) => {
                            let mut timings = npum_cpu::gte::Timings::default();
                            model
                                .embed_timed_with(
                                    &mut ops,
                                    &mut timings,
                                    ids,
                                    mask,
                                    batch,
                                    self.seq,
                                    None,
                                    None,
                                )
                                .map_err(|e| e.to_string())?;
                        }
                    }
                    for i in 0..batch {
                        vectors.push(flat[i * hidden..(i + 1) * hidden].to_vec());
                    }
                    Ok::<(), String>(())
                })?;
                offset += batch;
            }
        }

        Ok((vectors, tokens))
    }
}
