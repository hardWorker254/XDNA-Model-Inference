//! One Whisper transcription session: the windowing, the prompt and the
//! merge.
//!
//! # THE WINDOWS
//!
//! A window of C seconds is C*sample_rate samples, which the front end
//! turns into C*sample_rate/hop frames and conv2's stride 2 halves -- 50
//! positions a second at Whisper's 16 kHz / hop 160. The encoder's position
//! count is the MODEL's own (1500 for every shipped size), so a longer
//! window would be cut by the encoder's own guard with nothing to say so
//! at this level. Refused here instead, naming the number: a silently cut
//! window is a transcript of the first 30 s presented as the whole
//! recording.
//!
//! transformers strides BOTH sides, so consecutive windows start one
//! (chunk - 2*stride) apart, not (chunk - stride): 20 s for the default
//! 30/5.
//!
//! # THE PROMPT'S ORDER IS LOAD-BEARING
//!
//! transformers' order, which is not the obvious one: the language comes
//! SECOND, before the task token. Primed the other way round the model
//! still answers fluently and the whole transcript is in a different
//! language, so the order is a correctness property and not a style
//! choice.
//!
//! # THE MERGE
//!
//! transformers' merge, line for line: `tokenization_whisper`'s
//! `_find_longest_common_sequence` with no timestamps.
//!
//! Windows overlap, so consecutive windows say the same words twice. This
//! slides a match of length i across the boundary -- the last i tokens of
//! the left against the first i of the right -- scores it as
//! matches/i + i/10000 (the epsilon breaks ties towards the LONGER
//! overlap, which is what keeps a repeated "the the the" from being
//! swallowed) and keeps the best one with more than a single match. The
//! seam is then cut at the MIDPOINT of that overlap: the left half is
//! kept, the right half dropped, and the search continues on the tail of
//! the right window. With nothing matching, the midpoint lands at the end
//! of the left window and the effect is plain concatenation -- which is
//! the right behaviour for windows that share no words at all.

use crate::whisper::{Decoder, Encoder, EncoderWorkspace};
use crate::whisper_tokenizer::Tokenizer;

/// The sample rate every Whisper model runs at.
pub const SAMPLE_RATE: usize = 16_000;

/// The encoder's position count: conv2's stride 2 over 3000 mel frames.
pub const ENCODER_POSITIONS: usize = 1500;

/// The options one transcription runs with.
#[derive(Debug, Clone)]
pub struct Options {
    /// The language the decoder is primed with.
    pub language: String,
    /// `transcribe` or `translate`.
    pub task: String,
    /// The window's length, in seconds.
    pub chunk_seconds: i64,
    /// The stride on each side of the window, in seconds.
    pub stride_seconds: i64,
    /// The cap on GENERATED tokens; 0 means the position bound only.
    pub max_new: usize,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            language: "en".to_string(),
            task: "transcribe".to_string(),
            chunk_seconds: 30,
            stride_seconds: 5,
            max_new: 0,
        }
    }
}

/// One window's transcript.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Where the window starts, in samples.
    pub start_sample: usize,
    /// Where it ends, in samples.
    pub end_sample: usize,
    /// Where it starts, in seconds.
    pub start_s: f64,
    /// Where it ends, in seconds.
    pub end_s: f64,
    /// The generated ids.
    pub ids: Vec<i32>,
    /// The decoded text.
    pub text: String,
}

/// What one transcription produced.
#[derive(Debug, Clone)]
pub struct Transcript {
    /// The merged text, decoded once.
    pub text: String,
    /// The windows, each on its own.
    pub segments: Vec<Segment>,
    /// The language the decoder was primed with.
    pub language: String,
    /// The task.
    pub task: String,
    /// How many samples were transcribed.
    pub n_samples: usize,
    /// How many windows there were.
    pub n_chunks: usize,
    /// Where the time went, per request.
    pub mel_seconds: f64,
    pub conv_seconds: f64,
    pub encoder_seconds: f64,
    pub decoder_seconds: f64,
    /// How many array dispatches there were.
    pub n_dispatches: i64,
}

/// The decoder's prompt: the start token, the language, the task and the
/// timestamps flag.
pub fn prompt(tokenizer: &Tokenizer, options: &Options) -> Result<Vec<i32>, String> {
    let sot = tokenizer.token_id("<|startoftranscript|>")?;
    let lang = format!("<|{}|>", options.language);
    let task = format!("<|{}|>", options.task);
    let lang_id = tokenizer.token_id(&lang)?;
    let task_id = tokenizer.token_id(&task)?;
    let nots = tokenizer.token_id("<|notimestamps|>")?;
    Ok(vec![sot, lang_id, task_id, nots])
}

/// Transcribe a run of samples.
///
/// The encoder and the decoder are borrowed separately because the decoder
/// keeps its cross-attention cache: it is rebuilt per WINDOW, not per
/// session, because it depends on the encoder's output and a second window
/// is a different audio.
pub fn transcribe_samples(
    encoder: &Encoder,
    decoder: &mut Decoder,
    tokenizer: &Tokenizer,
    samples: &[f32],
    options: &Options,
    weights: &FrontEndWeights,
) -> Result<Transcript, String> {
    if samples.is_empty() {
        return Err("transcribe: no audio samples".into());
    }
    let geometry = encoder.geometry();
    let pos_per_second = (SAMPLE_RATE / 2 / geometry.hop_length) as i64;
    if options.chunk_seconds <= 0
        || options.chunk_seconds * pos_per_second > geometry.max_seq as i64
    {
        return Err(format!(
            "transcribe: a {} s window is {} encoder positions, and this container's position table holds {}.\n  Re-pack it with the full table -- that is the fix:\n    python tools/pack/pack_npue.py --model-dir <ckpt> --out <this file> --device <npu1|npu2> --max-seq {}\n  Shrinking --chunk-seconds to {} would NOT be a fix: the window would cover {} s of every {} s of audio, and the rest would be silence the model would transcribe as something.",
            options.chunk_seconds,
            options.chunk_seconds * pos_per_second,
            geometry.max_seq,
            geometry.max_seq,
            geometry.max_seq as i64 / pos_per_second,
            geometry.max_seq as i64 / pos_per_second,
            options.chunk_seconds
        ));
    }
    if options.stride_seconds < 0 || options.stride_seconds * 2 > options.chunk_seconds {
        return Err(
            "transcribe: the stride must be at most half the chunk length (transformers' own rule; two strides span the overlap)".into(),
        );
    }

    let mut transcript = Transcript {
        text: String::new(),
        segments: Vec::new(),
        language: options.language.clone(),
        task: options.task.clone(),
        n_samples: samples.len(),
        n_chunks: 0,
        mel_seconds: 0.0,
        conv_seconds: 0.0,
        encoder_seconds: 0.0,
        decoder_seconds: 0.0,
        n_dispatches: 0,
    };
    let chunk = options.chunk_seconds as usize * SAMPLE_RATE;
    let stride = options.stride_seconds as usize * SAMPLE_RATE;
    // transformers strides BOTH sides, so consecutive windows start one
    // (chunk - 2*stride) apart, not (chunk - stride).
    let step = chunk - 2 * stride;
    let n = samples.len();
    let mut window_ids: Vec<Vec<i32>> = Vec::new();
    // One workspace for the whole transcription: the encoder's scratch is
    // reused across every window, not just across its layers.
    let mut encoder_ws = EncoderWorkspace::new();

    let mut position = 0usize;
    loop {
        let end = n.min(position + chunk);
        let segment = run_window(
            encoder,
            decoder,
            &mut encoder_ws,
            tokenizer,
            &samples[position..end],
            position,
            end,
            options,
            weights,
            &mut transcript,
        )?;
        window_ids.push(segment.ids.clone());
        transcript.segments.push(segment);
        transcript.n_chunks += 1;
        // The last window covers the tail, so no redundant sliver after it.
        if end >= n {
            break;
        }
        position += step;
    }
    let merged = merge_windows(&window_ids);
    let bytes = tokenizer.decode(&merged)?;
    transcript.text = String::from_utf8_lossy(&bytes).into_owned();
    Ok(transcript)
}

/// The front end's weights, read out of the container.
pub struct FrontEndWeights {
    /// conv1's weights, `[d_model, n_mels, 3]`.
    pub conv1_weight: Vec<f32>,
    /// conv1's bias.
    pub conv1_bias: Vec<f32>,
    /// conv2's weights, `[d_model, d_model, 3]`.
    pub conv2_weight: Vec<f32>,
    /// conv2's bias.
    pub conv2_bias: Vec<f32>,
}

/// One window: the front end, the encoder and the greedy chain.
#[allow(clippy::too_many_arguments)]
fn run_window(
    encoder: &Encoder,
    decoder: &mut Decoder,
    encoder_ws: &mut EncoderWorkspace,
    tokenizer: &Tokenizer,
    window: &[f32],
    start: usize,
    end: usize,
    options: &Options,
    weights: &FrontEndWeights,
    transcript: &mut Transcript,
) -> Result<Segment, String> {
    let geometry = encoder.geometry();
    let mel_start = std::time::Instant::now();
    let (mel, _used) = crate::whisper_features::log_mel_30s(window, geometry.mel_bins);
    transcript.mel_seconds += mel_start.elapsed().as_secs_f64();

    let conv_start = std::time::Instant::now();
    let conv = crate::whisper_features::conv_front_end(
        &mel,
        &weights.conv1_weight,
        &weights.conv1_bias,
        &weights.conv2_weight,
        &weights.conv2_bias,
        geometry.d_model,
    );
    transcript.conv_seconds += conv_start.elapsed().as_secs_f64();

    let encoder_start = std::time::Instant::now();
    let hidden = encoder.run_into(&conv, ENCODER_POSITIONS, encoder_ws);
    transcript.encoder_seconds += encoder_start.elapsed().as_secs_f64();

    // The cross-attention cache is built once per WINDOW, not once per
    // session: it depends on the encoder output, and a second window is a
    // different audio.
    decoder.set_source(&hidden, ENCODER_POSITIONS)?;

    let decoder_start = std::time::Instant::now();
    let ids = decoder.greedy(
        &prompt(tokenizer, options)?,
        tokenizer.token_id("<|endoftext|>")?,
        options.max_new,
    )?;
    transcript.decoder_seconds += decoder_start.elapsed().as_secs_f64();
    let bytes = tokenizer.decode(&ids)?;

    Ok(Segment {
        start_sample: start,
        end_sample: end,
        start_s: start as f64 / SAMPLE_RATE as f64,
        end_s: end as f64 / SAMPLE_RATE as f64,
        ids,
        text: String::from_utf8_lossy(&bytes).into_owned(),
    })
}

/// Merge the windows' ids, transformers' way.
///
/// The merge is over IDS and not over text: two windows that spell the same
/// word differently are different ids, and the reference implementation
/// compares ids.
pub fn merge_windows(seqs: &[Vec<i32>]) -> Vec<i32> {
    if seqs.is_empty() {
        return Vec::new();
    }
    let mut total: Vec<i32> = Vec::new();
    let mut left: Vec<i32> = seqs[0].clone();
    for right in &seqs[1..] {
        let ll = left.len() as i64;
        let rl = right.len() as i64;
        let mut best = 0.0f64;
        let mut ls = ll;
        let mut lp = ll;
        let mut rs = 0i64;
        let mut rp = 0i64;
        for i in 1..(ll + rl) {
            let a = 0.max(ll - i);
            let b = ll.min(ll + rl - i);
            let c = 0.max(i - ll);
            let d = rl.min(i);
            if b - a != d - c {
                continue;
            }
            let mut matches = 0i64;
            for k in 0..(b - a) {
                if left[(a + k) as usize] == right[(c + k) as usize] {
                    matches += 1;
                }
            }
            let score = matches as f64 / i as f64 + i as f64 / 10000.0;
            if matches > 1 && score > best {
                best = score;
                ls = a;
                lp = b;
                rs = c;
                rp = d;
            }
        }
        let left_mid = (lp + ls) / 2;
        let right_mid = (rp + rs) / 2;
        total.extend_from_slice(&left[..left_mid as usize]);
        left = right[right_mid as usize..].to_vec();
    }
    total.extend_from_slice(&left);
    total
}
