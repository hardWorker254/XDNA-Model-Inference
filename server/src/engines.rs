//! Concrete execution engines used by HTTP routes.
//!
//! This module owns model/container loading and request execution for the
//! families exposed by the server. Route handlers and HTTP DTO/error mapping live
//! in `routes`; this layer focuses on turning requests into model outputs.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use npum_model::array::{ArrayConvs, ArrayEltwise, ArrayText};
use npum_tokenizers::gemma::Gemma;
use npum_tokenizers::Tokenizer;

/// The array handles one `serve` process opened at startup.
///
/// A transport is `Send` but not `Sync`, so the ONE handle per family is kept
/// behind a `Mutex`: concurrent requests serialise on the hardware context
/// instead of handing the same `&mut` to two calls. The CLI's `npu` module
/// opens these; the server only holds and hands them to the CPU pipeline.
pub struct XdnaForStt {
    pub enc: Option<Box<dyn ArrayText + Send>>,
    pub dec: Option<Box<dyn ArrayText + Send>>,
    pub eltwise: Option<Box<dyn ArrayEltwise + Send>>,
}

/// The ViT's array handles, with the flags that decide which of the staged
/// projections are dispatched. Staging a panel is not a dispatch, so the
/// engine carries the flags the CLI read and the pipeline is told explicitly.
pub struct XdnaForCls {
    pub array: Option<Box<dyn ArrayText + Send>>,
    pub eltwise: Option<Box<dyn ArrayEltwise + Send>>,
    pub gemm: bool,
    pub conv: bool,
    pub logit: bool,
}

/// The embedding endpoint's array handles, opened once at startup.
///
/// Both of this family's forward passes -- BERT and GTE -- take the same two
/// seams: `ArrayText` for the four projections and the attention block, and
/// `ArrayEltwise` for LayerNorm/GELU/softmax. A handle the flag did not name
/// stays `None`, and the pipeline then does that pass on the host.
pub struct XdnaForEmbed {
    pub array: Option<Box<dyn ArrayText + Send>>,
    pub eltwise: Option<Box<dyn ArrayEltwise + Send>>,
}

/// The pose endpoint's convolution handle, opened once at startup.
pub struct XdnaForPose {
    pub array: Box<dyn ArrayConvs + Send>,
}

/// The speech endpoint's engine.
///
/// The encoder, the tokenizer and the front end's weights are shared, while
/// each decoder instance keeps its own KV cache/state and is guarded by a
/// lock. A pool larger than one allows independent requests to decode in
/// parallel at the cost of loading the decoder weights/state multiple times.
///
/// The sequence length one embedded batch actually needs.
///
/// Tokenisation pads every text to the model's design window. The trailing
/// all-padding columns carry no information, and with the host/Vulkan softmax
/// the score width follows the sequence: dropping them is the same answer
/// with a much smaller attention matrix. An array attention or an array
/// softmax fixes the score row at the DESIGN's width, so those runs keep the
/// full window (`fix_width`) and rely on batching alone -- as the CLI does.
fn batch_seq(design_seq: usize, lengths: &[usize], fix_width: bool) -> usize {
    if fix_width {
        return design_seq;
    }
    lengths
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(1)
        .min(design_seq)
}

/// Repack one batch's padded rows into `eff_seq` columns.
fn pack_batch(
    ids: &[u32],
    mask: &[u32],
    design_seq: usize,
    begin: usize,
    batch: usize,
    eff_seq: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut out_ids = Vec::with_capacity(batch * eff_seq);
    let mut out_mask = Vec::with_capacity(batch * eff_seq);
    for i in 0..batch {
        let base = (begin + i) * design_seq;
        out_ids.extend_from_slice(&ids[base..base + eff_seq]);
        out_mask.extend_from_slice(&mask[base..base + eff_seq]);
    }
    (out_ids, out_mask)
}

#[cfg(test)]
mod seq_tests {
    use super::*;

    #[test]
    fn a_short_batch_drops_only_padding() {
        assert_eq!(batch_seq(512, &[3, 5, 2], false), 5);
        assert_eq!(batch_seq(512, &[], false), 1);
        assert_eq!(batch_seq(4, &[9, 1], false), 4);
    }

    #[test]
    fn the_array_keeps_the_design_width() {
        assert_eq!(batch_seq(512, &[3, 5, 2], true), 512);
        assert_eq!(batch_seq(512, &[], true), 512);
    }

    #[test]
    fn packing_keeps_each_texts_original_positions() {
        // Three rows padded to seq 8, only the first three tokens real.
        let ids: Vec<u32> = (0..24).collect();
        let mask: Vec<u32> = (0..24).map(|i| u32::from(i % 8 < 3)).collect();
        let (packed_ids, packed_mask) = pack_batch(&ids, &mask, 8, 1, 2, 4);
        assert_eq!(packed_ids, vec![8, 9, 10, 11, 16, 17, 18, 19]);
        assert_eq!(packed_mask, vec![1, 1, 1, 0, 1, 1, 1, 0]);
    }
}

/// How many decoders the plain `open` builds.
///
/// Two is enough to overlap two transcriptions while keeping the decoder's
/// KV-cache memory reasonable; a caller that knows its concurrency can ask
/// for more with [`Transcriber::open_with_pool`].
pub const DEFAULT_DECODER_POOL: usize = 2;

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
    /// The array handles, opened once by `serve`. They are kept behind
    /// SEPARATE mutexes: one hardware context per handle, and one `&mut` per
    /// request. `None` is a host run: no device, no design set, no array work.
    xdna_enc: Option<Arc<Mutex<Box<dyn ArrayText + Send>>>>,
    xdna_dec: Option<Arc<Mutex<Box<dyn ArrayText + Send>>>>,
    xdna_elt: Option<Arc<Mutex<Box<dyn ArrayEltwise + Send>>>>,
}

impl Transcriber {
    /// Build an engine from a container.
    pub fn open(
        path: &std::path::Path,
        options: npum_cpu::whisper_session::Options,
    ) -> std::result::Result<Self, String> {
        Self::open_with_pool(path, options, DEFAULT_DECODER_POOL)
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
            xdna_enc: None,
            xdna_dec: None,
            xdna_elt: None,
        })
    }

    /// Give the transcriber the array handles `serve` opened at startup.
    ///
    /// The engine does not open a design or a transport itself: the one place
    /// that knows the search rules, the panel checks and `--artifacts` is the
    /// CLI's `npu` module. Sharing its result here keeps a host run and an
    /// array run identical apart from the seams the run actually dispatches.
    pub fn with_xdna(self, xdna: Option<XdnaForStt>) -> Self {
        match xdna {
            Some(h) => Self {
                xdna_enc: h.enc.map(|e| Arc::new(Mutex::new(e))),
                xdna_dec: h.dec.map(|e| Arc::new(Mutex::new(e))),
                xdna_elt: h.eltwise.map(|e| Arc::new(Mutex::new(e))),
                ..self
            },
            None => self,
        }
    }

    /// Whether this transcriber will dispatch array work.
    pub fn has_xdna(&self) -> bool {
        self.xdna_enc.is_some() || self.xdna_dec.is_some() || self.xdna_elt.is_some()
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
            // ONE hardware context per set, so each handle is locked for the
            // length of the pass. A second request waits here rather than
            // handing the same `&mut` to two dispatches.
            let mut enc_guard = self.xdna_enc.as_ref().map(|m| m.lock().unwrap());
            let mut dec_guard = self.xdna_dec.as_ref().map(|m| m.lock().unwrap());
            let mut elt_guard = self.xdna_elt.as_ref().map(|m| m.lock().unwrap());
            let enc: Option<&mut dyn ArrayText> = match enc_guard.as_deref_mut() {
                Some(b) => Some(b.as_mut()),
                None => None,
            };
            let dec: Option<&mut dyn ArrayText> = match dec_guard.as_deref_mut() {
                Some(b) => Some(b.as_mut()),
                None => None,
            };
            let elt: Option<&mut dyn ArrayEltwise> = match elt_guard.as_deref_mut() {
                Some(b) => Some(b.as_mut()),
                None => None,
            };
            npum_cpu::whisper_session::transcribe_samples(
                &self.encoder,
                &mut decoder,
                &self.tokenizer,
                &read.samples,
                options,
                &self.weights,
                enc,
                dec,
                elt,
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
    /// The one array handle for this engine, opened by `serve` at startup.
    xdna_array: Option<Arc<Mutex<Box<dyn ArrayText + Send>>>>,
    xdna_elt: Option<Arc<Mutex<Box<dyn ArrayEltwise + Send>>>>,
    /// Staging is not dispatching: the CLI's flags decide which of the
    /// staged projections the encoder actually submits to the array.
    xdna_gemm: bool,
    xdna_conv: bool,
    xdna_head: bool,
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
            xdna_array: None,
            xdna_elt: None,
            xdna_gemm: false,
            xdna_conv: false,
            xdna_head: false,
        })
    }

    /// Give the classifier the array handles `serve` opened at startup.
    pub fn with_xdna(self, xdna: Option<XdnaForCls>) -> Self {
        match xdna {
            Some(h) => Self {
                xdna_array: h.array.map(|a| Arc::new(Mutex::new(a))),
                xdna_elt: h.eltwise.map(|e| Arc::new(Mutex::new(e))),
                xdna_gemm: h.gemm,
                xdna_conv: h.conv,
                xdna_head: h.logit,
                ..self
            },
            None => self,
        }
    }

    /// Whether this classifier will dispatch array work.
    pub fn has_xdna(&self) -> bool {
        self.xdna_array.is_some() || self.xdna_elt.is_some()
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
        let mut array_guard = self.xdna_array.as_ref().map(|m| m.lock().unwrap());
        let mut elt_guard = self.xdna_elt.as_ref().map(|m| m.lock().unwrap());
        let mut backend: Option<&mut dyn ArrayText> = match array_guard.as_deref_mut() {
            Some(b) => Some(b.as_mut()),
            None => None,
        };
        let eltwise: Option<&mut dyn ArrayEltwise> = match elt_guard.as_deref_mut() {
            Some(b) => Some(b.as_mut()),
            None => None,
        };
        self.vit.encode_into(
            &patches,
            &mut scratch.workspace,
            &mut scratch.hidden,
            backend.as_deref_mut(),
            eltwise,
            self.xdna_gemm,
            self.xdna_conv,
        )?;
        let encoder_s = enc.elapsed().as_secs_f64();
        let head = std::time::Instant::now();
        let d = geometry.hidden as usize;
        let nlab = self.vit.geometry().num_labels as usize;
        scratch.logits.resize(nlab, 0.0);
        let head_on = backend.as_ref().is_some_and(|b| b.head_on_array()) && self.xdna_head;
        self.vit
            .classify_into_with(
                &scratch.hidden[..d],
                &mut scratch.logits[..nlab],
                backend.as_deref_mut(),
                head_on,
            )?;
        let prediction = npum_cpu::vit::predict(&scratch.logits[..nlab], self.vit.labels())?;
        let head_s = head.elapsed().as_secs_f64();
        let dispatches = array_guard
            .as_ref()
            .map(|b| b.dispatches() + b.attn_dispatches() + b.head_dispatches())
            .unwrap_or(0);
        let elt_dispatches = elt_guard.as_ref().map(|e| e.dispatches()).unwrap_or(0);
        Ok(Classification {
            dispatches,
            elt_dispatches,
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
    /// The array's own dispatch counters for this run.
    pub dispatches: usize,
    pub elt_dispatches: usize,
}

/// Hands: the MediaPipe palm detector + landmark net, on the host.
///
/// There is no XDNA handle here by design: these families have no installed
/// design set, and `--npu-ops` is refused by name before `serve` starts. The
/// JSON comes from the same `hands_decode::result_json` the CLI prints, so
/// the two paths cannot drift.
pub struct HandsEngine {
    net: npum_cpu::hands::Hands,
    xdna: Option<Arc<Mutex<Box<dyn ArrayConvs + Send>>>>,
}

impl HandsEngine {
    pub fn geometry(&self) -> &npum_cpu::hands::Geometry {
        self.net.geometry()
    }

    /// Give the engine the convolution handle `serve` opened at startup.
    pub fn with_xdna(mut self, array: Option<Box<dyn ArrayConvs + Send>>) -> Self {
        self.xdna = array.map(|a| Arc::new(Mutex::new(a)));
        self
    }

    pub fn open(path: &std::path::Path) -> std::result::Result<Self, String> {
        Ok(Self {
            net: npum_cpu::hands::Hands::open(path).map_err(|e| e.to_string())?,
            xdna: None,
        })
    }

    pub fn detect_bytes(
        &self,
        image: &[u8],
        name: &str,
        max_hands: i64,
    ) -> std::result::Result<String, String> {
        let decoded = npum_cpu::vit_image::decode_image_bytes(image, name)?;
        let mut hook = |_: &str, _: usize, _: &npum_cpu::pose::Tensor| {};
        let mut guard = self.xdna.as_ref().map(|m| m.lock().unwrap());
        let array = guard.as_deref_mut().map(|a| a.as_mut() as &mut dyn ArrayConvs);
        let run = npum_cpu::hands_decode::detect(&self.net, &decoded, max_hands, &mut hook, array)?;
        Ok(npum_cpu::hands_decode::result_json(&run, name))
    }
}

/// Multi-person MediaPipe Pose, on the host -- same JSON as `npuimage mppose`.
pub struct MpPoseEngine {
    net: npum_cpu::mppose::MpPose,
    xdna: Option<Arc<Mutex<Box<dyn ArrayConvs + Send>>>>,
}

impl MpPoseEngine {
    pub fn geometry(&self) -> &npum_cpu::mppose::Geometry {
        self.net.geometry()
    }

    pub fn with_xdna(mut self, array: Option<Box<dyn ArrayConvs + Send>>) -> Self {
        self.xdna = array.map(|a| Arc::new(Mutex::new(a)));
        self
    }

    pub fn open(path: &std::path::Path) -> std::result::Result<Self, String> {
        Ok(Self {
            net: npum_cpu::mppose::MpPose::open(path).map_err(|e| e.to_string())?,
            xdna: None,
        })
    }

    pub fn detect_bytes(
        &self,
        image: &[u8],
        name: &str,
        max_people: i64,
    ) -> std::result::Result<String, String> {
        let decoded = npum_cpu::vit_image::decode_image_bytes(image, name)?;
        let mut hook = |_: &str, _: usize, _: &npum_cpu::pose::Tensor| {};
        let mut guard = self.xdna.as_ref().map(|m| m.lock().unwrap());
        let array = guard.as_deref_mut().map(|a| a.as_mut() as &mut dyn ArrayConvs);
        let run = npum_cpu::mppose_decode::detect(&self.net, &decoded, max_people, &mut hook, array)?;
        Ok(npum_cpu::mppose_decode::result_json(&run, name))
    }
}

/// PaddleOCR, on the host. The session owns mutable detector/recogniser
/// scratch, so one request at a time holds the mutex.
pub struct OcrEngine {
    session: Mutex<npum_cpu::ocr_session::Ocr>,
}

// SAFETY: this engine is constructed with `array = None` and never accepts
// one. The `Option<Box<dyn ArrayConvs>>` inside `Ocr` is therefore always
// empty, and the rest of the session is plain host tensors behind a Mutex.
// No non-Send transport handle ever crosses a thread boundary.
unsafe impl Send for OcrEngine {}
unsafe impl Sync for OcrEngine {}

impl OcrEngine {
    pub fn geometry(&self) -> npum_cpu::ocr::Geometry {
        self.session
            .lock()
            .expect("the OCR session mutex was poisoned")
            .geometry()
            .clone()
    }

    pub fn with_xdna(self, array: Option<Box<dyn ArrayConvs + Send>>) -> Self {
        if let Some(array) = array {
            self.session
                .lock()
                .expect("the OCR session mutex was poisoned")
                .set_array(Some(array));
        }
        self
    }

    pub fn open(path: &std::path::Path) -> std::result::Result<Self, String> {
        Ok(Self {
            session: Mutex::new(npum_cpu::ocr_session::Ocr::open(path, None).map_err(|e| e.to_string())?),
        })
    }

    pub fn read_bytes(
        &self,
        image: &[u8],
        name: &str,
        max_boxes: i64,
    ) -> std::result::Result<String, String> {
        let decoded = npum_cpu::vit_image::decode_image_bytes(image, name)?;
        let mut hook = |_: &str, _: i64, _: &npum_cpu::ocr::Tensor| {};
        let mut session = self.session.lock().map_err(|e| e.to_string())?;
        let result = session.read(&decoded, max_boxes, &mut hook)?;
        Ok(npum_cpu::ocr_decode::result_json(&result, name))
    }
}

/// The pose endpoint's engine.
///
/// The YOLOv8-pose network and the geometry it describes. Like the
/// classifier this holds no state between requests -- a detection is a
/// function of the image and the thresholds -- so it needs no lock.
pub struct PoseDetector {
    net: npum_cpu::pose::Network,
    /// The array's convolutions, behind a lock: one hardware context must not
    /// be handed to two concurrent requests. `None` runs the host's kernels.
    xdna: Option<Arc<Mutex<Box<dyn ArrayConvs + Send>>>>,
}

impl PoseDetector {
    /// Build an engine from a container.
    pub fn open(path: &std::path::Path) -> std::result::Result<Self, String> {
        let net =
            npum_cpu::pose::Network::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self { net, xdna: None })
    }

    /// Give the engine the convolution handle `serve` opened at startup.
    pub fn with_xdna(self, xdna: Option<XdnaForPose>) -> Self {
        match xdna {
            Some(h) => Self {
                xdna: Some(Arc::new(Mutex::new(h.array))),
                ..self
            },
            None => self,
        }
    }

    /// Whether this engine will actually dispatch convolutions to the array.
    pub fn has_xdna(&self) -> bool {
        self.xdna.is_some()
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
        let (head, _) = match self.xdna.as_ref() {
            Some(shared) => {
                let mut guard = shared
                    .lock()
                    .expect("the array convolution mutex was poisoned");
                self.net.run_timed_on_array(&input, Some(&mut **guard))?
            }
            None => self.net.run_timed(&input)?,
        };
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
    Gemma(Gemma),
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
            EmbedTokenizer::Gemma(t) => {
                let enc = t.encode(text)?;
                let mut ids = vec![2u32];
                ids.extend_from_slice(&enc.ids);
                ids.push(1u32);
                let truncated = ids.len() > max_len;
                ids.truncate(max_len);
                let mut mask = vec![1u32; ids.len()];
                let mut types = vec![0u32; ids.len()];
                ids.resize(max_len, 0);
                mask.resize(max_len, 0);
                types.resize(max_len, 0);
                Ok((npum_tokenizers::Encoding { ids, mask, types }, truncated))
            }
        }
    }
}

/// Read the GEMATOK1 table the packer embeds in a Gemma container.
fn gemma_vocab(blob: &[u8]) -> std::result::Result<HashMap<String, u32>, String> {
    if blob.len() < 56 || &blob[..8] != b"GEMATOK1" {
        return Err("the Gemma tokenizer table has no GEMATOK1 header".to_string());
    }
    let count = u32::from_le_bytes(blob[12..16].try_into().unwrap()) as usize;
    let mut vocab = HashMap::new();
    let mut pos = 56usize;
    for id in 0..count {
        let len = u16::from_le_bytes(
            blob.get(pos..pos + 2)
                .ok_or_else(|| "the Gemma tokenizer table is cut short".to_string())?
                .try_into()
                .unwrap(),
        ) as usize;
        pos += 2;
        let piece = std::str::from_utf8(
            blob.get(pos..pos + len)
                .ok_or_else(|| "the Gemma tokenizer table is cut short".to_string())?,
        )
        .map_err(|e| format!("the Gemma tokenizer table is not UTF-8: {e}"))?;
        pos += len;
        vocab.insert(piece.to_string(), id as u32);
    }
    Ok(vocab)
}

#[derive(Clone)]
enum EmbedModel {
    Bert(npum_cpu::bert::BertModel),
    Gte(npum_cpu::gte::GteModel),
    Gemma(Arc<npum_cpu::gemma::GemmaModel>),
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
    /// GTE's scratch, shared across the engine's clones. GTE's forward used
    /// to allocate seven buffers and both RoPE tables per request; one lock
    /// here keeps a single set for the whole server.
    gte_workspace: Arc<std::sync::Mutex<npum_cpu::GteWorkspace>>,
    /// Requests that arrived while another caller is gathering a batch.
    ///
    /// `Arc` so every clone of the engine (every `ServerState`) feeds the
    /// SAME queue: without that, ten concurrent requests would still run ten
    /// forwards of batch one.
    pending: Arc<Mutex<Vec<PendingEmbed>>>,
    pending_cv: Arc<Condvar>,
    /// The startup-opened array handles, behind separate locks: a transport is
    /// `Send` but not `Sync`, so concurrent requests must serialise on the
    /// hardware context instead of sharing one `&mut`. The two handles are
    /// separate because a run can stage projections without staging an
    /// eltwise pass (and the other way round).
    xdna_array: Option<Arc<Mutex<Box<dyn ArrayText + Send>>>>,
    xdna_elt: Option<Arc<Mutex<Box<dyn ArrayEltwise + Send>>>>,
}

type BertSlot = (npum_cpu::bert::BertWorkspace, Vec<f32>);

/// The result one `/v1/embeddings` caller waits for.
type EmbedResult = std::result::Result<(Vec<Vec<f32>>, usize), String>;

/// One request waiting to be coalesced into a shared forward pass.
struct PendingEmbed {
    texts: Vec<String>,
    reply: std::sync::mpsc::Sender<EmbedResult>,
}

/// How long the first arrival waits for company.
///
/// Long enough to gather the requests a burst of small clients sends in the
/// same few milliseconds, short enough that a lone request is not delayed
/// noticeably. The forward pass is what costs time here, not the wait.
const COALESCE_WINDOW: Duration = Duration::from_millis(10);

/// Stop gathering once this many texts are queued: a bigger forward pass
/// would spend more time in one batch than it saves in dispatch overhead.
const COALESCE_MAX_TEXTS: usize = 64;

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
        let tokenizer = if arch.contains("gemma") {
            let blob = container
                .tensor_bytes("tokenizer.gemma_table")
                .map_err(|_| "the container holds no Gemma tokenizer table".to_string())?;
            EmbedTokenizer::Gemma(Gemma::new(gemma_vocab(blob)?))
        } else if arch.contains("gte_new_rope")
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
            if arch.contains("gemma") {
                EmbedModel::Gemma(Arc::new(
                    npum_cpu::gemma::GemmaModel::open(path)
                        .map_err(|e| format!("{}: {e}", path.display()))?,
                ))
            } else if arch == "gte_new_rope_geglu" || (arch.contains("gte") && arch.contains("rope")) {
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
            gte_workspace: Arc::new(std::sync::Mutex::new(npum_cpu::GteWorkspace::default())),
            pending: Arc::new(Mutex::new(Vec::new())),
            pending_cv: Arc::new(Condvar::new()),
            xdna_array: None,
            xdna_elt: None,
        })
    }

    /// Give the engine the array handles `serve` opened at startup.
    pub fn with_xdna(self, xdna: Option<XdnaForEmbed>) -> Self {
        match xdna {
            Some(h) => Self {
                xdna_array: h.array.map(|a| Arc::new(Mutex::new(a))),
                xdna_elt: h.eltwise.map(|e| Arc::new(Mutex::new(e))),
                ..self
            },
            None => self,
        }
    }

    /// Whether this engine will actually dispatch to the array.
    pub fn has_xdna(&self) -> bool {
        self.xdna_array.is_some() || self.xdna_elt.is_some()
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
            EmbedModel::Gemma(m) => m.spec.vocab,
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

    /// Embed one coalesced batch. The vectors and the token count of EACH
    /// input text, so the leader can hand every caller its own usage number.
    fn embed_impl(
        &self,
        texts: &[String],
    ) -> std::result::Result<(Vec<Vec<f32>>, Vec<usize>), String> {
        if texts.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        // Tokenise and pad every text up front so the model can be run in
        // batches instead of one forward pass per input. The Vulkan device
        // opened for the first such request stays alive across all later
        // batches and requests; the CPU ops instance is per request.
        let mut all_ids = Vec::with_capacity(texts.len() * self.seq);
        let mut all_mask = Vec::with_capacity(texts.len() * self.seq);
        let mut lengths = Vec::with_capacity(texts.len());
        let mut per_text = Vec::with_capacity(texts.len());
        for text in texts {
            let (encoding, _) = self
                .tokenizer
                .encode_padded(text, self.seq)
                .map_err(|e| e.to_string())?;
            let real = encoding.mask.iter().filter(|m| **m > 0).count();
            per_text.push(real);
            lengths.push(real);
            all_ids.extend_from_slice(&encoding.ids);
            all_mask.extend_from_slice(&encoding.mask);
        }

        let hidden = match &self.model {
            EmbedModel::Bert(model) => model.spec.hidden,
            EmbedModel::Gte(model) => model.spec.hidden,
            EmbedModel::Gemma(model) => model.spec.hidden,
        };

        // Bound workspace/activation growth for large requests while still
        // amortising tokenisation, dispatch and device setup.
        const BATCH: usize = 16;

        // The array's score row is the DESIGN's, so an array attention/softmax
        // keeps the full window. The handles are held for the whole request:
        // one hardware context must not be handed to two callers at once.
        let fix_width = self.xdna_array.is_some() || self.xdna_elt.is_some();
        let mut array_guard = self.xdna_array.as_ref().map(|m| m.lock().unwrap());
        let mut elt_guard = self.xdna_elt.as_ref().map(|m| m.lock().unwrap());

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
                let eff_seq = batch_seq(self.seq, &lengths[offset..offset + batch], fix_width);
                let (ids, mask) = pack_batch(&all_ids, &all_mask, self.seq, offset, batch, eff_seq);
                let mut guard = shared.lock().expect("the Vulkan device mutex was poisoned");
                let ops = guard
                    .as_mut()
                    .expect("the Vulkan device was opened above and is never removed");
                let array = array_guard
                    .as_deref_mut()
                    .map(|a| a.as_mut() as &mut (dyn ArrayText + '_));
                let eltwise = elt_guard
                    .as_deref_mut()
                    .map(|e| e.as_mut() as &mut (dyn ArrayEltwise + '_));
                self.with_bert_slot(|(workspace, flat)| {
                    let gte = match &self.model {
                        EmbedModel::Bert(model) => {
                            model
                                .embed_into(
                                    &ids, &mask, batch, eff_seq, ops, array, eltwise, workspace,
                                    flat,
                                )
                                .map_err(|e| e.to_string())?;
                            None
                        }
                        EmbedModel::Gte(model) => {
                            let mut timings = npum_cpu::gte::Timings::default();
                            let mut scratch = self
                                .gte_workspace
                                .lock()
                                .expect("the GTE workspace mutex was poisoned");
                            // GTE returns the pooled vectors. It never writes
                            // `flat`, which is BERT's output slot.
                            Some(
                                model
                                    .embed_timed_with_workspace(
                                        ops,
                                        &mut timings,
                                        &ids,
                                        &mask,
                                        batch,
                                        eff_seq,
                                        array,
                                        eltwise,
                                        &mut scratch,
                                    )
                                    .map_err(|e| e.to_string())?,
                            )
                        }
                        EmbedModel::Gemma(model) => {
                            let mut timings = npum_cpu::gemma::Timings::default();
                            Some(
                                model
                                    .embed_timed_with(
                                        ops,
                                        &mut timings,
                                        &ids,
                                        &mask,
                                        batch,
                                        eff_seq,
                                        array,
                                        eltwise,
                                    )
                                    .map_err(|e| e.to_string())?,
                            )
                        }
                    };
                    match gte {
                        Some(out) => {
                            // GTE pools into one flat buffer, like BERT's
                            // `flat`; split it into one vector per input.
                            for row in out.chunks(hidden) {
                                vectors.push(row.to_vec());
                            }
                        }
                        None => {
                            for i in 0..batch {
                                vectors.push(flat[i * hidden..(i + 1) * hidden].to_vec());
                            }
                        }
                    }
                    Ok::<(), String>(())
                })?;
                offset += batch;
            }
        } else {
            let mut ops = npum_cpu::ops::CpuOps::best();
            while offset < texts.len() {
                let batch = (texts.len() - offset).min(BATCH);
                let eff_seq = batch_seq(self.seq, &lengths[offset..offset + batch], fix_width);
                let (ids, mask) = pack_batch(&all_ids, &all_mask, self.seq, offset, batch, eff_seq);
                let array = array_guard
                    .as_deref_mut()
                    .map(|a| a.as_mut() as &mut (dyn ArrayText + '_));
                let eltwise = elt_guard
                    .as_deref_mut()
                    .map(|e| e.as_mut() as &mut (dyn ArrayEltwise + '_));
                self.with_bert_slot(|(workspace, flat)| {
                    let gte = match &self.model {
                        EmbedModel::Bert(model) => {
                            model
                                .embed_into(
                                    &ids, &mask, batch, eff_seq, &mut ops, array, eltwise,
                                    workspace, flat,
                                )
                                .map_err(|e| e.to_string())?;
                            None
                        }
                        EmbedModel::Gte(model) => {
                            let mut timings = npum_cpu::gte::Timings::default();
                            let mut scratch = self
                                .gte_workspace
                                .lock()
                                .expect("the GTE workspace mutex was poisoned");
                            Some(
                                model
                                    .embed_timed_with_workspace(
                                        &mut ops,
                                        &mut timings,
                                        &ids,
                                        &mask,
                                        batch,
                                        eff_seq,
                                        array,
                                        eltwise,
                                        &mut scratch,
                                    )
                                    .map_err(|e| e.to_string())?,
                            )
                        }
                        EmbedModel::Gemma(model) => {
                            let mut timings = npum_cpu::gemma::Timings::default();
                            Some(
                                model
                                    .embed_timed_with(
                                        &mut ops,
                                        &mut timings,
                                        &ids,
                                        &mask,
                                        batch,
                                        eff_seq,
                                        array,
                                        eltwise,
                                    )
                                    .map_err(|e| e.to_string())?,
                            )
                        }
                    };
                    match gte {
                        Some(out) => {
                            // GTE pools into one flat buffer, like BERT's
                            // `flat`; split it into one vector per input.
                            for row in out.chunks(hidden) {
                                vectors.push(row.to_vec());
                            }
                        }
                        None => {
                            for i in 0..batch {
                                vectors.push(flat[i * hidden..(i + 1) * hidden].to_vec());
                            }
                        }
                    }
                    Ok::<(), String>(())
                })?;
                offset += batch;
            }
        }

        Ok((vectors, per_text))
    }

    /// Embed the texts, coalescing concurrent small requests into one forward.
    ///
    /// The first arrival becomes the leader: it gathers whatever arrives in
    /// `COALESCE_WINDOW` (or until the batch is full), runs ONE forward for
    /// all of it, and hands each caller back only its own vectors and token
    /// count. The alternative -- every request running its own batch of one
    /// -- wastes the NPU's batch width on a hundred small HTTP calls.
    pub fn embed(&self, texts: &[String]) -> EmbedResult {
        if texts.is_empty() {
            return Ok((Vec::new(), 0));
        }
        let (reply, receiver) = std::sync::mpsc::channel();
        let leader = {
            let mut pending = self
                .pending
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let was_empty = pending.is_empty();
            pending.push(PendingEmbed {
                texts: texts.to_vec(),
                reply,
            });
            was_empty
        };
        self.pending_cv.notify_one();
        if !leader {
            return receiver.recv().unwrap_or_else(|error| {
                Err(format!(
                    "the embedding batcher stopped before answering: {error}"
                ))
            });
        }

        // Gather. A `wait_timeout` that returns early because someone arrived
        // re-checks the count; one that times out ends the wait with what we
        // have. Both paths leave with the queue drained by this thread alone.
        let started = Instant::now();
        let batch = loop {
            let mut pending = self
                .pending
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let queued: usize = pending.iter().map(|request| request.texts.len()).sum();
            if queued >= COALESCE_MAX_TEXTS || started.elapsed() >= COALESCE_WINDOW {
                break std::mem::take(&mut *pending);
            }
            let remaining = COALESCE_WINDOW.saturating_sub(started.elapsed());
            let (guard, _) = self
                .pending_cv
                .wait_timeout(pending, remaining)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            drop(guard);
        };

        let mut all_texts = Vec::new();
        let mut owners = Vec::with_capacity(batch.len());
        for request in batch {
            all_texts.extend(request.texts.iter().cloned());
            owners.push((request.texts.len(), request.reply));
        }

        // A panic in the forward pass must not leave the other callers blocked
        // forever on a channel whose sender was dropped without a value.
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.embed_impl(&all_texts)))
                .unwrap_or_else(|_| Err("the embedding forward pass panicked".to_string()));

        match outcome {
            Ok((vectors, per_text)) => {
                let mut at = 0usize;
                for (count, reply) in owners {
                    let mine = vectors[at..at + count].to_vec();
                    let tokens = per_text[at..at + count].iter().sum();
                    at += count;
                    let _ = reply.send(Ok((mine, tokens)));
                }
            }
            Err(reason) => {
                for (_, reply) in owners {
                    let _ = reply.send(Err(reason.clone()));
                }
            }
        }
        receiver.recv().unwrap_or_else(|error| {
            Err(format!(
                "the embedding batcher stopped before answering: {error}"
            ))
        })
    }
}
