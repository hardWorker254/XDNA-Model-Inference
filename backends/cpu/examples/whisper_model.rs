//! Whisper's encoder, dumped as hex.
//!
//! The C++ oracle has its own test binary for this
//! (`runtime/tests/test_whisper_model.cpp`), and it prints the same
//! line: `enc <rows> <cols> <hex>`. This example is the Rust's
//! counterpart of it, and the two are compared BYTE for byte.
//!
//! Usage:
//!   whisper_model <model.npum> <audio.wav>

use npum_cpu::whisper;
use npum_cpu::whisper_features;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: whisper_model <model.npum> <audio.wav>");
        std::process::exit(2);
    }
    let model = Path::new(&args[1]);
    let audio = Path::new(&args[2]);

    let container = npum_format::Container::open(model).expect("the container opens");
    let encoder = whisper::Encoder::open(model).expect("the encoder's weights load");
    let g = encoder.geometry().clone();

    let samples = match whisper_features::read_wav(audio) {
        Ok(a) => a.samples,
        Err(e) => {
            eprintln!("{}", e.0);
            std::process::exit(2);
        }
    };

    let (mel, _used) = whisper_features::log_mel_30s(&samples, g.mel_bins);
    let w1 = tensor_f32(&container, "frontend.conv1.weight");
    let b1 = tensor_f32(&container, "frontend.conv1.bias");
    let w2 = tensor_f32(&container, "frontend.conv2.weight");
    let b2 = tensor_f32(&container, "frontend.conv2.bias");
    let conv = whisper_features::conv_front_end(&mel, &w1, &b1, &w2, &b2, g.d_model);

    let n_src = conv.len() / g.d_model;
    let hidden = encoder.run(&conv, n_src);
    println!("enc {} {} {}", n_src, g.d_model, to_hex(&hidden));
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
