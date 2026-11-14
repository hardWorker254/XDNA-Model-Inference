//! The CPU backend against the golden.
//!
//! # What this test is
//!
//! The golden transcripts in `tools/parity/golden` were recorded from the C++
//! binaries. This test runs the same container through the Rust's CPU backend
//! and compares the vector against the golden's, element by element.
//!
//! The comparison is on SEVEN SIGNIFICANT DIGITS, because that is what the
//! golden holds: the C++'s HTTP endpoint prints `%.7g`. Seven digits is not
//! bit-exactness, and the difference matters -- see below.
//!
//! # What it is NOT
//!
//! It is not a bit-exactness gate. Two things are bit-exactness gates and
//! neither is this one:
//!
//! * the AVX2 path against the AVX-512 path, on the same input, comparing
//!   bytes -- in `kernels.rs`'s own tests, where both paths exist;
//! * the C++'s own `encoding_format: "base64"`, which carries the float32
//!   BYTES rather than a decimal rendering of them.
//!
//! A seven-digit comparison catches a wrong arithmetic -- a missing FMA, a
//! different summation order, a transposed weight -- and it does not catch
//! a one-ULP difference. That is the honest limit of what the golden holds,
//! and it is stated here rather than left to be discovered.

use std::path::{Path, PathBuf};

/// The repository root, from this crate's manifest directory.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("the repository root exists")
}

/// The container, if it is installed.
fn model_path() -> Option<PathBuf> {
    let path = repo_root().join("models/all-MiniLM-L6-v2.npue");
    if path.is_file() { Some(path) } else { None }
}

/// The golden's vector, if it is recorded.
fn golden_vector() -> Option<Vec<f64>> {
    let path = repo_root().join("tools/parity/golden/http/embeddings/body");
    let text = std::fs::read_to_string(path).ok()?;
    let body: serde_json::Value = serde_json::from_str(&text).ok()?;
    let first = body["data"][0]["embedding"].as_array()?;
    Some(
        first
            .iter()
            .map(|v| v.as_f64().unwrap_or(f64::NAN))
            .collect(),
    )
}

/// The sequence length the oracle's design set was built for.
///
/// It is 64, and it is the DESIGN's number and not the container's: the
/// container's `max_seq_len` is 256, and the C++ runtime truncates and
/// refuses at the design's width.
const SEQ: usize = 64;

#[test]
fn the_embedding_matches_the_golden() {
    let (Some(model_path), Some(golden)) = (model_path(), golden_vector()) else {
        eprintln!("skipped: the container or the golden is not installed");
        return;
    };
    let model = npum_cpu::bert::BertModel::open(&model_path).expect("the container opens");

    let container = npum_format::Container::open(&model_path).expect("the container opens");
    let blob = container
        .tensor_bytes("tokenizer.vocab")
        .expect("the container holds a vocabulary");
    let tokenizer = npum_tokenizers::wordpiece::WordPiece::from_lines(&String::from_utf8_lossy(blob))
        .expect("the vocabulary is usable");

    let (encoding, truncated) = tokenizer
        .encode_padded("a golden transcript is a contract", SEQ)
        .expect("the text encodes");
    assert!(!truncated, "the golden's text fits");

    let vectors = model
        .embed(&encoding.ids, &encoding.mask, 1, SEQ)
        .expect("the model runs");

    assert_eq!(vectors.len(), golden.len(), "the vector's width");
    let mut worst: f64 = 0.0;
    for (i, (got, want)) in vectors.iter().zip(golden.iter()).enumerate() {
        let got = *got as f64;
        let diff = (got - want).abs();
        let rel = diff / want.abs().max(1e-12);
        if rel > worst {
            worst = rel;
        }
        assert!(
            rel < 1e-6,
            "element {i}: the runtime says {got}, the golden says {want} (relative {rel:e})"
        );
    }
    eprintln!("the worst relative difference is {worst:e}");
}
