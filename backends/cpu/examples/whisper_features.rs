//! Whisper's audio front end, dumped as hex.
//!
//! The C++ oracle has its own test binary for this
//! (`runtime/tests/test_whisper_features.cpp`), and it prints the same
//! three lines: the samples, the mel tensor and the convolution's
//! output. This example is the Rust's counterpart of it, and the two
//! are compared BYTE for byte -- a mismatch in the transcript says
//! "something is wrong", and these three lines say WHERE.
//!
//! Usage:
//!   whisper_features <model.npum> <audio.wav>

use npum_cpu::whisper;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: whisper_features <model.npum> <audio.wav>");
        std::process::exit(2);
    }
    let model = Path::new(&args[1]);
    let audio = Path::new(&args[2]);

    let container = npum_format::Container::open(model).expect("the container opens");
    let config = container.config();
    let d_model = config.get_i64("d_model").expect("d_model") as usize;
    let n_mels = config.get_i64("num_mel_bins").expect("num_mel_bins") as usize;

    let samples = match whisper::read_wav(audio) {
        Ok(a) => a.samples,
        Err(e) => {
            eprintln!("{}", e.0);
            std::process::exit(2);
        }
    };
    println!("samples {} {}", samples.len(), to_hex(&samples));

    let (mel, _used) = whisper::log_mel_30s(&samples, n_mels);
    println!(
        "mel {} {} {}",
        mel.n_mels,
        mel.frames,
        to_hex(&mel.data)
    );

    let w1 = tensor_f32(&container, "frontend.conv1.weight");
    let b1 = tensor_f32(&container, "frontend.conv1.bias");
    let w2 = tensor_f32(&container, "frontend.conv2.weight");
    let b2 = tensor_f32(&container, "frontend.conv2.bias");
    let emb = whisper::conv_front_end(&mel, &w1, &b1, &w2, &b2, d_model);
    println!("conv {} {} {}", whisper::ENCODER_POSITIONS, d_model, to_hex(&emb));
}

/// A tensor's bytes, as f32.
fn tensor_f32(container: &npum_format::Container, name: &str) -> Vec<f32> {
    let bytes = container
        .tensor_bytes(name)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The bytes, as hex.
fn to_hex(values: &[f32]) -> String {
    let mut out = String::with_capacity(values.len() * 8);
    for v in values {
        for b in v.to_le_bytes() {
            out.push_str(&format!("{b:02x}"));
        }
    }
    out
}
