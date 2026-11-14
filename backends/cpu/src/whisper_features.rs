//! Whisper's audio front end: the log-mel spectrogram and the two
//! convolutions that follow it.
//!
//! # Everything here is DEFINED BY HUGGINGFACE, not chosen
//!
//! The container stores the convolutions' weights; the spectrogram is a
//! fixed function of the samples, and a container that computed a
//! different one would be a model that transcribes nothing while
//! looking perfectly healthy. So the constants are Whisper's own, they
//! are named here, and the order of the steps is written down once.
//!
//! # The pipeline, in the order that matters
//!
//! ```text
//! samples            float32 mono, 16 kHz
//! zero-pad           to 30 s = 480000 samples (a 5 s file is NOT 3000
//!                    frames of its own audio followed by nothing; it is
//!                    5 s of audio and 25 s of silence, and the encoder
//!                    always sees 1500 positions)
//! reflect-pad        200 samples each side, WITHOUT repeating the edge
//!                    sample (numpy's "reflect", not "symmetric")
//! frames             400 samples, hop 160, 3001 of them
//! window             periodic Hann, w[n] = 0.5 - 0.5cos(2*pi*n/400)
//!                    -- w[399] is 6.17e-05 and NOT zero, which is what
//!                    makes it periodic; a symmetric window is a
//!                    different filter
//! FFT                exactly 400 points, NOT zero-padded to 512: a
//!                    longer transform is a different spectrum
//! power              |X|^2, 201 bins
//! mel                201 x n_mels slaney filter bank, area-normalised
//! log10              of max(mel, 1e-10)
//! drop               the LAST frame -> 3000
//! floor              maximum(log_spec, global_max - 8.0)
//! scale              (x + 4.0) / 4.0
//! ```
//!
//! # The transform is EXACTLY 400 points
//!
//! 400 = 2^4 * 5^2, and it has to be exactly that: zero-padding the
//! frame to 512 and running a power-of-two FFT is a different
//! transform of a different signal, and the resulting spectrogram is
//! wrong in a way no tolerance argument survives. So this is a
//! mixed-radix Cooley-Tukey over the factors 2 and 5, in fp64.
//!
//! # The reflect pad, and the two ways to get it wrong
//!
//! numpy's `pad(..., mode="reflect")` does NOT repeat the edge
//! sample: for `[a, b, c]` with one sample of padding the result is
//! `[b, a, b, c, b]`. The leftmost padded sample is the one TWO back,
//! and the right side is the mirror of the left with the direction
//! FLIPPED and starting AFTER the data.

use std::path::Path;

/// Whisper's own numbers. 30 s at 16 kHz is 480000 samples; / 160 is
/// 3000 frames; conv2's stride 2 makes 1500 encoder positions; and
/// 1500 is max_source_positions for every shipped size.
pub const SAMPLE_RATE: usize = 16000;
/// 30 s of samples.
pub const CHUNK_SAMPLES: usize = 30 * SAMPLE_RATE;
/// The frames' hop, in samples.
pub const HOP_LENGTH: usize = 160;
/// The transform's length, in samples.
pub const NFFT: usize = 400;
/// The power spectrum's bins.
pub const MEL_BINS: usize = NFFT / 2 + 1;
/// The frames after the last one is dropped.
pub const MEL_FRAMES: usize = 3000;
/// The encoder's positions: conv2's stride 2 halves the frames.
pub const ENCODER_POSITIONS: usize = MEL_FRAMES / 2;

/// The audio, as the model is given it.
pub struct Audio {
    /// The samples, mono, in [-1, 1], at 16 kHz.
    pub samples: Vec<f32>,
    /// The sample rate.
    pub sample_rate: u32,
}

/// A refusal, with the reason.
#[derive(Debug)]
pub struct Refusal(pub String);

/// Read a WAV file.
///
/// Strict: a 16-bit or a 32-bit-float PCM WAV at 16 kHz, mono, is read
/// directly, and anything else is REFUSED with what to do about it.
/// A resample that is 0.1% off produces a transcript that is wrong in
/// no visible way, which is the worst failure there is.
pub fn read_wav(path: &Path) -> Result<Audio, Refusal> {
    let raw = std::fs::read(path).map_err(|e| Refusal(format!("cannot open {}: {e}", path.display())))?;
    let p = &raw[..];
    let n = raw.len();
    if n < 12 || &p[0..4] != b"RIFF" || &p[8..12] != b"WAVE" {
        return Err(Refusal(format!(
            "{}: not a RIFF/WAVE file. convert it: --convert (ffmpeg -f s16le -ar 16000 -ac 1)",
            path.display()
        )));
    }

    // Walk the chunks. A WAV is a list of typed blocks and the two we
    // want are not adjacent, not first, and not necessarily in this
    // order -- LIST and fact chunks sit between them often enough that
    // assuming otherwise is a bug waiting for a real file.
    let mut have_fmt = false;
    let mut format: u16 = 0;
    let mut channels: u16 = 0;
    let mut bits: u16 = 0;
    let mut rate: u32 = 0;
    let mut data: Option<&[u8]> = None;
    let mut off = 12usize;
    while off + 8 <= n {
        let id = &p[off..off + 4];
        let sz = u32::from_le_bytes([p[off + 4], p[off + 5], p[off + 6], p[off + 7]]) as usize;
        let body = off + 8;
        if body + sz > n {
            break; // truncated: use what is there, fmt decides
        }
        if id == b"fmt " && sz >= 16 {
            format = u16::from_le_bytes([p[body], p[body + 1]]);
            channels = u16::from_le_bytes([p[body + 2], p[body + 3]]);
            rate = u32::from_le_bytes([p[body + 4], p[body + 5], p[body + 6], p[body + 7]]);
            bits = u16::from_le_bytes([p[body + 14], p[body + 15]]);
            // WAVE_FORMAT_EXTENSIBLE (0xFFFE): the real format is the
            // first two bytes of the SubFormat GUID, which is where every
            // encoder puts it.
            if format == 0xFFFE && sz >= 26 {
                format = u16::from_le_bytes([p[body + 24], p[body + 25]]);
            }
            have_fmt = true;
        } else if id == b"data" {
            data = Some(&p[body..body + sz]);
        }
        off = body + sz + (sz & 1); // chunks are word-aligned
    }

    let (Some(data), true) = (data, have_fmt) else {
        return Err(Refusal(format!(
            "{}: no fmt /data chunk. it is not a usable PCM WAV; convert it: --convert",
            path.display()
        )));
    };
    if !(format == 3 && bits == 32) && !(format == 1 && bits == 16) {
        return Err(Refusal(format!(
            "{}: format {format}, {bits} bits per sample (need 16-bit PCM or 32-bit float). convert it: --convert",
            path.display()
        )));
    }
    if channels != 1 {
        return Err(Refusal(format!(
            "{}: {channels} channels. Whisper is 16 kHz MONO; downmix with --convert",
            path.display()
        )));
    }
    if rate != SAMPLE_RATE as u32 {
        return Err(Refusal(format!(
            "{}: {rate} Hz. Whisper is 16 kHz; resample with --convert",
            path.display()
        )));
    }

    let mut samples = Vec::with_capacity(data.len() / 2);
    if format == 1 {
        let count = data.len() / 2;
        for i in 0..count {
            let v = i16::from_le_bytes([data[2 * i], data[2 * i + 1]]);
            samples.push(v as f32 / 32768.0);
        }
    } else {
        let count = data.len() / 4;
        for i in 0..count {
            let v = f32::from_le_bytes([data[4 * i], data[4 * i + 1], data[4 * i + 2], data[4 * i + 3]]);
            samples.push(v);
        }
    }
    Ok(Audio {
        samples,
        sample_rate: rate,
    })
}

/// The log-mel spectrogram, `(n_mels, frames)` row-major, frame index fastest.
pub struct MelSpec {
    /// How many mel bands.
    pub n_mels: usize,
    /// How many frames.
    pub frames: usize,
    /// The values.
    pub data: Vec<f32>,
}

impl MelSpec {
    /// One value: the band and the frame.
    pub fn at(&self, mel: usize, frame: usize) -> f32 {
        self.data[mel * self.frames + frame]
    }
}

/// The log-mel spectrogram of a 30 s chunk.
///
/// `samples_used` receives how many samples were real, so a caller can
/// tell a 5 s file from a 30 s one instead of assuming.
pub fn log_mel_30s(samples: &[f32], n_mels: usize) -> (MelSpec, usize) {
    // 1. 30 s of audio, zero-padded. The encoder is fixed at 1500
    //    positions, so a short file is padded rather than given a
    //    shorter feature tensor.
    let n_used = samples.len().min(CHUNK_SAMPLES);
    let mut wave = vec![0.0f64; CHUNK_SAMPLES];
    for (i, s) in samples.iter().take(n_used).enumerate() {
        wave[i] = *s as f64;
    }

    // 2. reflect-pad, 200 each side -> 480400, which is 3001 frames of
    //    400 with hop 160. The last of them is dropped after the mel
    //    projection.
    let padded = reflect_pad(&wave, NFFT / 2);
    let num_frames = 1 + (padded.len() - NFFT) / HOP_LENGTH;
    debug_assert_eq!(num_frames, MEL_FRAMES + 1);

    // 3. periodic Hann. w[399] = 6.17e-05, not 0: that is what
    //    "periodic" means and a symmetric window is a different filter.
    let mut window = vec![0.0f64; NFFT];
    for (n, w) in window.iter_mut().enumerate() {
        *w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / NFFT as f64).cos();
    }

    // 4. frames -> windowed samples, and then the power spectrum.
    let power = power_30s(&padded, num_frames, &window);
    let mut mel = project_and_log(&power, n_mels);
    mel_floor_scale(&mut mel);
    (
        MelSpec {
            n_mels,
            frames: MEL_FRAMES,
            data: mel,
        },
        n_used,
    )
}

/// numpy's `pad(..., mode="reflect")`.
fn reflect_pad(x: &[f64], pad: usize) -> Vec<f64> {
    if pad == 0 {
        return x.to_vec();
    }
    let mut out = vec![0.0f64; x.len() + 2 * pad];
    // out[i] = x[pad - i], NOT x[pad - 1 - i]: numpy's reflect for
    // [0,1,2,3] with pad 2 is [2,1,0,1,2,3,2,1], so the leftmost padded
    // sample is the one TWO back.
    for i in 0..pad {
        out[i] = x[pad - i];
    }
    out[pad..pad + x.len()].copy_from_slice(x);
    // The right side is the mirror of the left with the direction
    // FLIPPED, and it starts AFTER the data: out[pad + n + i] =
    // x[n - 2 - i].
    for i in 0..pad {
        out[pad + x.len() + i] = x[x.len() - 2 - i];
    }
    out
}

/// The `(n_bins, frames)` power spectrum, channels-first.
fn power_30s(padded: &[f64], num_frames: usize, window: &[f64]) -> Vec<f32> {
    let n_bins = NFFT / 2 + 1;
    let mut power = vec![0.0f32; n_bins * MEL_FRAMES];
    // The windowed frame goes through FLOAT32 before the transform, and
    // that rounding is part of the contract: the C++ keeps the frames in
    // a float32 matrix (it is the GEMM's A operand) and reads them back
    // as float64 for the transform. A frame that stays in double all the
    // way through is a different frame, and the log10 downstream turns a
    // relative difference of 6e-08 into an absolute one of 3.6e-06.
    let mut frame = vec![0.0f64; NFFT];
    let mut windowed = vec![0.0f32; NFFT];
    for t in 0..num_frames {
        let src = &padded[t * HOP_LENGTH..t * HOP_LENGTH + NFFT];
        for i in 0..NFFT {
            windowed[i] = (src[i] * window[i]) as f32;
        }
        if t == num_frames - 1 {
            continue; // the frame Whisper drops
        }
        for i in 0..NFFT {
            frame[i] = windowed[i] as f64;
        }
        let spec = power_spectrum(&frame);
        for (k, v) in spec.iter().enumerate() {
            power[k * MEL_FRAMES + t] = *v as f32;
        }
    }
    power
}

/// The power spectrum of one frame: `|X|^2` for the first 201 bins.
fn power_spectrum(frame: &[f64]) -> Vec<f64> {
    let n = frame.len();
    let mut buf: Vec<(f64, f64)> = frame.iter().map(|v| (*v, 0.0)).collect();
    fft_rec(&mut buf);
    (0..n / 2 + 1).map(|i| {
        let (re, im) = buf[i];
        re * re + im * im
    })
    .collect()
}

/// A mixed-radix Cooley-Tukey over the factors 2 and 5.
///
/// Recursive rather than iterative because the permutation falls out of
/// the decimation and a 400-point transform is small enough that the
/// recursion is not the bottleneck; the arithmetic is.
fn fft_rec(a: &mut [(f64, f64)]) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    // The radix is the SMALLEST prime factor, so: 2 if even, else 5.
    // Growing p multiplicatively instead (2, 5, 10, 20, ...) walks off
    // the end of size_t on a size this does not factor, and the division
    // by zero that follows is a SIGFPE with no message.
    let p = if n % 2 == 0 { 2 } else { 5 };
    if n % p != 0 {
        panic!("whisper features: FFT length {n} is not 2^a*5^b");
    }
    let m = n / p;

    let mut sub: Vec<Vec<(f64, f64)>> = Vec::with_capacity(p);
    for q in 0..p {
        let mut s = vec![(0.0, 0.0); m];
        for k in 0..m {
            s[k] = a[k * p + q];
        }
        fft_rec(&mut s);
        sub.push(s);
    }
    // X[k + m*t] = sum_q A_q[k] * exp(-2*pi*i*q*(k + m*t)/n), and the
    // exponent splits: exp(-2*pi*i*qk/n) * exp(-2*pi*i*qmt/n). The
    // first factor is the usual per-level twiddle, the second depends
    // only on (q, t) and is hoisted.
    let theta = -2.0 * std::f64::consts::PI / n as f64;
    for q in 0..p {
        for k in 0..m {
            let ph = theta * (q * k) as f64;
            let (c, s) = (ph.cos(), ph.sin());
            let (re, im) = sub[q][k];
            sub[q][k] = (re * c - im * s, re * s + im * c);
        }
    }
    let mut v = vec![vec![(0.0f64, 0.0f64); p]; p];
    for q in 0..p {
        for t in 0..p {
            let ph = theta * (q * m * t) as f64;
            v[q][t] = (ph.cos(), ph.sin());
        }
    }
    let mut out = vec![(0.0, 0.0); n];
    for k in 0..m {
        for t in 0..p {
            let mut acc = (0.0f64, 0.0f64);
            for q in 0..p {
                let (ar, ai) = sub[q][k];
                let (vr, vi) = v[q][t];
                acc = (
                    acc.0 + ar * vr - ai * vi,
                    acc.1 + ar * vi + ai * vr,
                );
            }
            out[k + m * t] = acc;
        }
    }
    a.copy_from_slice(&out);
}

/// The Slaney mel filter bank, `(201, n_mels)` row-major.
pub fn mel_filter_bank(n_mels: usize, sample_rate: usize, n_fft: usize) -> Vec<f64> {
    let n_bins = n_fft / 2 + 1;
    // n_mels + 2 centre frequencies: the two extra ones are the outer
    // edges the first and last triangles climb to.
    let mut centre = vec![0.0f64; n_mels + 2];
    let mel_min = hz_to_mel_slaney(0.0);
    let mel_max = hz_to_mel_slaney(sample_rate as f64 / 2.0);
    for (i, c) in centre.iter_mut().enumerate() {
        *c = mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64);
    }

    // Triangular in FREQUENCY space (transformers' default:
    // triangularize_in_mel_space=False), each filter zero outside its own
    // band.
    let mut bank = vec![0.0f64; n_bins * n_mels];
    for j in 0..n_mels {
        let lo = centre[j];
        let mid = centre[j + 1];
        let hi = centre[j + 2];
        // Slaney area normalisation: constant energy per channel, so a
        // narrow low band is not quieter than a wide high one purely by
        // being narrow.
        let enorm = 2.0 / (hi - lo);
        for k in 0..n_bins {
            let hz = sample_rate as f64 / 2.0 * k as f64 / (n_bins - 1) as f64;
            let mut w = 0.0;
            if hz > lo && hz < hi {
                w = if hz <= mid {
                    (hz - lo) / (mid - lo)
                } else {
                    (hi - hz) / (hi - mid)
                };
            }
            bank[k * n_mels + j] = w * enorm;
        }
    }
    bank
}

/// Slaney's mel scale: linear below 1000 Hz, logarithmic above.
///
/// HTK's 2595*log10(1+f/700) is a different function and produces a
/// different bank.
fn hz_to_mel_slaney(hz: f64) -> f64 {
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    let logstep = 27.0 / 6.4f64.ln();
    let linear = 3.0 * hz / 200.0;
    if hz < MIN_LOG_HZ {
        return linear;
    }
    MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() * logstep
}

/// Slaney's mel scale, inverted.
fn mel_to_hz_slaney(mel: f64) -> f64 {
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    let logstep = 6.4f64.ln() / 27.0;
    let linear = 200.0 * mel / 3.0;
    if mel < MIN_LOG_MEL {
        return linear;
    }
    MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
}

/// The Slaney bank over the power spectrum, then log10, in that order.
///
/// Writing the projection as `log_spec @ bank` instead is HF's
/// formulation and produces a different spectrogram.
fn project_and_log(power: &[f32], n_mels: usize) -> Vec<f32> {
    let n_bins = NFFT / 2 + 1;
    let bank = mel_filter_bank(n_mels, SAMPLE_RATE, NFFT);
    let mut mel = vec![0.0f32; n_mels * MEL_FRAMES];
    for t in 0..MEL_FRAMES {
        for j in 0..n_mels {
            let mut acc = 0.0f64;
            for k in 0..n_bins {
                acc += bank[k * n_mels + j] * power[k * MEL_FRAMES + t] as f64;
            }
            mel[j * MEL_FRAMES + t] = acc as f32;
        }
    }
    // The 1e-10 floor BEFORE the log, not after: log10(0) is -inf, and
    // an -inf in the tensor poisons the global max on the way to the
    // floor.
    for v in mel.iter_mut() {
        *v = (*v as f64).max(1e-10).log10() as f32;
    }
    mel
}

/// transformers' floor and scale, over a FINISHED tensor.
///
/// The maximum is taken after the last frame is dropped: taking it
/// before would let a dropped frame raise the floor of every frame that
/// stayed.
fn mel_floor_scale(mel: &mut [f32]) {
    let mut global_max = -1e300f64;
    for v in mel.iter() {
        global_max = global_max.max(*v as f64);
    }
    let floor_v = global_max - 8.0;
    for v in mel.iter_mut() {
        *v = (((*v as f64).max(floor_v) + 4.0) / 4.0) as f32;
    }
}

/// conv1 -> GELU -> conv2 -> GELU -> permute: Whisper's whole audio front
/// end.
///
/// The weights are the container's, in Conv1d order: `w1` is
/// `[d, n_mels, 3]`, `w2` is `[d, d, 3]`. The result is `(1500, d)`
/// row-major, which is the M x K the encoder's first GEMM wants.
pub fn conv_front_end(
    mel: &MelSpec,
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    hidden: usize,
) -> Vec<f32> {
    let t = mel.frames;
    let m = mel.n_mels;
    let t2 = t / 2;

    // conv1: (M, T) -> (hidden, T), kernel 3, padding 1, GELU.
    let mut h1 = vec![0.0f32; hidden * t];
    for o in 0..hidden {
        let w = &w1[o * m * 3..(o + 1) * m * 3];
        for frame in 0..t {
            let mut acc = b1[o];
            for c in 0..m {
                let wc = &w[c * 3..c * 3 + 3];
                for k in 0..3 {
                    let ti = frame as isize + k as isize - 1; // padding 1
                    if ti < 0 || ti >= t as isize {
                        continue;
                    }
                    // An FMA, and not a multiply and an add: the oracle
                    // is built with `-march=x86-64-v3` and its compiler
                    // contracts `acc += w * x` into one rounding. The
                    // product's rounding is the difference between the
                    // two, and 240 of them add up.
                    acc = wc[k].mul_add(mel.at(c, ti as usize), acc);
                }
            }
            h1[o * t + frame] = gelu_erf_exact(acc);
        }
    }

    // conv2: (hidden, T) -> (hidden, T/2), kernel 3, stride 2, padding 1,
    // GELU.
    let mut h2 = vec![0.0f32; hidden * t2];
    for o in 0..hidden {
        let w = &w2[o * hidden * 3..(o + 1) * hidden * 3];
        for frame in 0..t2 {
            let centre = frame * 2;
            let mut acc = b2[o];
            for c in 0..hidden {
                let wc = &w[c * 3..c * 3 + 3];
                for k in 0..3 {
                    let ti = centre as isize + k as isize - 1;
                    if ti < 0 || ti >= t as isize {
                        continue;
                    }
                    acc = wc[k].mul_add(h1[c * t + ti as usize], acc);
                }
            }
            h2[o * t2 + frame] = gelu_erf_exact(acc);
        }
    }

    // permute to (T2, hidden): Conv1d is channels-first, the encoder's
    // GEMM is rows-are-tokens. Doing it here means the encoder never
    // sees a transposed operand and cannot get the stride wrong by
    // accident.
    let mut out = vec![0.0f32; t2 * hidden];
    for o in 0..hidden {
        for frame in 0..t2 {
            out[frame * hidden + o] = h2[o * t2 + frame];
        }
    }
    out
}

/// Exact erf GELU: `0.5*x*(1+erf(x/sqrt(2)))`.
///
/// Computed in double like the numpy oracle and rounded once at the
/// end. The float32 version is NOT the same number, and the difference
/// is a decade above the double-vs-float noise the rest of the stack
/// carries.
pub fn gelu_erf_exact(x: f32) -> f32 {
    let xd = x as f64;
    (0.5 * xd * (1.0 + erf(xd * 0.70710678118654752440))) as f32
}

/// The error function.
///
/// `erf(x) = P(1/2, x^2)`, and the regularised lower incomplete gamma
/// is evaluated the way every numerical library does it: a series of
/// POSITIVE terms where the series converges, and a continued fraction
/// where it does not. The positivity matters -- an alternating series
/// for `x` near 2.5 loses three digits to cancellation, and the GELU's
/// result would carry them.
///
/// The accuracy is about one ulp of f64, which is six decades more than
/// the f32 the GELU rounds to needs.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let a = x.abs();
    if a.is_infinite() {
        return sign;
    }
    if a < 1e-300 {
        return x;
    }
    sign * gamma_p(0.5, a * a)
}

/// The regularised lower incomplete gamma function `P(a, x)`.
fn gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // The series: P(a,x) = e^{-x} x^a / Gamma(a+1) * sum_{n>=0}
        // x^n / ((a+1)(a+2)...(a+n)).
        let gln = ln_gamma(a + 1.0);
        // The prefactor carries Gamma(a+1), so the series starts at 1 and
        // its denominators run (a+1), (a+1)(a+2), ... . Starting the sum
        // at 1/a as well would divide the answer by a twice.
        let mut ap = a;
        let mut sum = 1.0;
        let mut del = 1.0;
        for _ in 0..1000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-17 {
                break;
            }
        }
        (-x + a * x.ln() - gln).exp() * sum
    } else {
        // The continued fraction for Q(a,x) = 1 - P(a,x), by Lentz's
        // method.
        //
        // The prefactor here carries Gamma(a) and NOT Gamma(a+1): the
        // continued fraction's own factor is 1/x, and the series branch
        // above already carries the a in its first term. Getting this
        // wrong divides Q by a -- which is a factor of two at a = 0.5,
        // and it shows up only where erf is nearly 1, which is exactly
        // where the GELU's tail lives.
        let gln = ln_gamma(a);
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / 1e-300;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-300 {
                d = 1e-300;
            }
            c = b + an / c;
            if c.abs() < 1e-300 {
                c = 1e-300;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-17 {
                break;
            }
        }
        1.0 - (-x + a * x.ln() - gln).exp() * h
    }
}

/// `ln Gamma(x)`, by the Lanczos approximation.
fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Gamma(x)Gamma(1-x) = pi / sin(pi x).
        return std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, g) in G.iter().enumerate().skip(1) {
        // The denominators are x+1 .. x+8: the series is the one in
        // Numerical Recipes' gammln, and the coefficients are indexed
        // against x+j and not against x+j+1.
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}
