//! The Whisper front end's baked streams, on the real board.
//!
//! `dft400` and `mel_proj` carry their B operands inside the exported design
//! set but NOT inside the container: the twiddles and the Slaney bank are
//! generated here, tiled in the design's own order and staged like any other
//! panel. No mock can prove that path works -- this test needs the device.
//! It skips without one, and with no whisper container.

use std::path::PathBuf;

use npum_model::array::ArrayFrontend;
use npum_xdna::array::ArrayDesign;
use npum_xdna::array_gemms::ArrayGemms;
use npum_xdna::transport::SelectedTransport as Board;
use npum_xdna::transport::Transport;

const NFFT: usize = 400;
const N_BINS: usize = NFFT / 2 + 1;
const N_MELS: usize = 80;
const FRAMES: usize = 4;

fn board_present() -> bool {
    std::path::Path::new("/dev/accel/accel0").exists()
        || std::path::Path::new("/dev/dri/accel0").exists()
}

fn container() -> Option<PathBuf> {
    let p = std::env::var("NPUEMBEDDINGS_WHISPER_NPU").ok().map(PathBuf::from)?;
    if p.is_file() { Some(p) } else { None }
}

/// Direct O(400*201) reference, in f64, for the few frames this test uses.
fn reference_power(wave: &[f32]) -> Vec<f64> {
    let mut power = vec![0.0f64; N_BINS * FRAMES];
    for t in 0..FRAMES {
        for k in 0..N_BINS {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for n in 0..NFFT {
                let a = 2.0 * std::f64::consts::PI * k as f64 * n as f64 / NFFT as f64;
                re += wave[t * NFFT + n] as f64 * a.cos();
                im -= wave[t * NFFT + n] as f64 * a.sin();
            }
            power[k * FRAMES + t] = re * re + im * im;
        }
    }
    power
}

#[test]
fn baked_dft_and_mel_streams_match_the_reference() {
    if !board_present() {
        eprintln!("skipping: no NPU device node");
        return;
    }
    let Some(container) = container() else {
        eprintln!("skipping: set NPUEMBEDDINGS_WHISPER_NPU to a whisper container");
        return;
    };
    let transport = match Board::open(0) {
        Ok(transport) => transport,
        Err(e) => {
            eprintln!("skipping: no NPU board: {e}");
            return;
        }
    };
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../runtime/artifacts/whisper-base/artifacts_npu1/gemm_rtp");
    let design = ArrayDesign::open(&dir, transport).expect("the design set loads");
    let mut backend = ArrayGemms::open(&container, design, &[], false, false, false, false)
        .expect("the front end streams open without staged GEMM panels");
    assert!(backend.frontend_on_array());

    let mut wave = vec![0.0f32; FRAMES * NFFT];
    for (t, frame) in wave.chunks_exact_mut(NFFT).enumerate() {
        for (n, sample) in frame.iter_mut().enumerate() {
            let freq = 40.0 + 7.0 * t as f64;
            *sample = (2.0 * std::f64::consts::PI * freq * n as f64 / NFFT as f64).sin() as f32;
        }
    }

    let mut power = vec![0.0f32; N_BINS * FRAMES];
    backend
        .dft400(&wave, FRAMES, N_BINS, &mut power)
        .expect("the DFT dispatches");
    let reference = reference_power(&wave);
    let mut dft_err = 0.0f64;
    for i in 0..N_BINS * FRAMES {
        let e = (power[i] as f64 - reference[i]).abs()
            / reference[i].max(1.0);
        dft_err = dft_err.max(e);
    }
    eprintln!("DFT: max relative error {dft_err:.3e}");
    assert!(dft_err < 2e-2, "the DFT's bf16 twiddles are too far off");
    assert_eq!(backend.frontend_dispatches(), 1);

    let mut mel = vec![0.0f32; N_MELS * FRAMES];
    backend
        .mproj(&power, FRAMES, N_BINS, N_MELS, &mut mel)
        .expect("the mel projection dispatches");
    let bank = npum_cpu::whisper_features::mel_filter_bank(N_MELS, 16000, NFFT);
    let mut mel_err = 0.0f64;
    for j in 0..N_MELS {
        for t in 0..FRAMES {
            let mut want = 0.0f64;
            for k in 0..N_BINS {
                want += bank[k * N_MELS + j] * power[k * FRAMES + t] as f64;
            }
            let e = (mel[j * FRAMES + t] as f64 - want).abs() / want.max(1.0);
            mel_err = mel_err.max(e);
        }
    }
    eprintln!("mel: max relative error {mel_err:.3e}");
    assert!(mel_err < 3e-2, "the mel bank's bf16 weights are too far off");
    assert_eq!(backend.frontend_dispatches(), 2);
}
