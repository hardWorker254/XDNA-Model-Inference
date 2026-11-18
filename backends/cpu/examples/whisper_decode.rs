//! Whisper's decoder, dumped as hex.
//!
//! The C++ oracle has its own test binary for this
//! (`runtime/tests/test_whisper_model.cpp`), and it prints the same
//! lines: `step <pos> <top1> <cols> <hex>`, `logits <pos> <vocab> <hex>`,
//! `greedy <n> <ids...>` and `text <hex>`. This example is the Rust's
//! counterpart of it, and the two are compared BYTE for byte.
//!
//! Usage:
//!   whisper_decode <model.npum> <audio.wav> [--ids a,b,c] [--greedy a,b,c]
//!                  [--max-new N]

use npum_cpu::whisper;
use npum_cpu::whisper_features;
use npum_cpu::whisper_tokenizer::Tokenizer;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: whisper_decode <model.npum> <audio.wav> [--ids a,b,c] [--greedy a,b,c] [--max-new N]"
        );
        std::process::exit(2);
    }
    let model = Path::new(&args[1]);
    let audio = Path::new(&args[2]);

    let mut ids: Vec<i32> = Vec::new();
    let mut prompt: Vec<i32> = Vec::new();
    let mut max_new = 0usize;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--ids" => {
                ids = parse_ids(&args[i + 1]);
                i += 2;
            }
            "--greedy" => {
                prompt = parse_ids(&args[i + 1]);
                i += 2;
            }
            "--max-new" => {
                max_new = args[i + 1].parse().unwrap();
                i += 2;
            }
            other => {
                eprintln!("unknown flag {other}");
                std::process::exit(2);
            }
        }
    }

    let container = npum_format::Container::open(model).expect("the container opens");
    let encoder = whisper::Encoder::open(model).expect("the encoder's weights load");
    let mut decoder = whisper::Decoder::open(model).expect("the decoder's weights load");
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
    decoder
        .set_source(&hidden, n_src)
        .expect("the cross-attention cache");

    for (pos, &token) in ids.iter().enumerate() {
        let mut h = Vec::new();
        let mut lg = Vec::new();
        let top = decoder
            .step(token, pos, Some(&mut h), Some(&mut lg))
            .expect("the step");
        println!("step {} {} {} {}", pos, top, h.len(), to_hex(&h));
        println!("logits {} {} {}", pos, lg.len(), to_hex(&lg));
    }

    if !prompt.is_empty() {
        let table = container
            .tensor_bytes("tokenizer.whisper_table")
            .expect("the tokenizer's table");
        let tokenizer = Tokenizer::from_table(table).expect("the tokenizer's table");
        let stop = tokenizer.token_id("<|endoftext|>").expect("the stop token");
        let out = decoder
            .greedy(&prompt, stop, max_new)
            .expect("the greedy chain");
        print!("greedy {}", out.len());
        for id in &out {
            print!(" {id}");
        }
        println!();
        let text = tokenizer.decode(&out).expect("the text");
        println!("text {}", to_hex_bytes(&text));
    }
}

/// A comma-separated list of ids.
fn parse_ids(s: &str) -> Vec<i32> {
    s.split(',')
        .filter(|p| !p.is_empty())
        .map(|p| p.trim().parse().expect("an id"))
        .collect()
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

/// The bytes, as hex.
fn to_hex_bytes(values: &[u8]) -> String {
    let mut out = String::with_capacity(values.len() * 2);
    for b in values {
        out.push_str(&format!("{b:02x}"));
    }
    out
}
