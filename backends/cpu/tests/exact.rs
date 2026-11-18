//! The CPU backend against the oracle's EXACT vector.
//!
//! # Why this file exists and the seven-digit one does not
//!
//! `golden.rs` compares against the golden transcripts, which hold SEVEN
//! SIGNIFICANT DIGITS -- enough to catch a wrong arithmetic, and not enough
//! to prove a right one. The C++'s HTTP endpoint also speaks
//! `encoding_format: "base64"`, and that carries the float32 BYTES.
//!
//! So this file is the bit-exactness gate: the same container, the same
//! text, the same sequence length, and the two vectors compared BYTE BY
//! BYTE. A difference of one ULP in one element fails the test, and that is
//! the point: the golden is a contract, and a contract is a number.
//!
//! # The fixture, and where it came from
//!
//! `tools/parity/fixtures/exact/*.b64` holds the oracle's own bytes,
//! produced by `tools/parity/exact_vector.py` against the C++ binaries. It
//! is a fixture and not a golden: it is the INPUT to this comparison, not
//! the transcript of a run.

use std::path::{Path, PathBuf};

/// The repository root, from this crate's manifest directory.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("the repository root exists")
}

/// The oracle's bytes, if the fixture is installed.
fn oracle_bytes() -> Option<Vec<u8>> {
    let path = repo_root().join(
        "tools/parity/fixtures/exact/all-MiniLM-L6-v2__a-golden-transcript-is-a-contract.b64",
    );
    let text = std::fs::read_to_string(path).ok()?;
    Some(base64_decode(text.trim()))
}

/// A base64 decoder, because a test does not need a dependency for forty
/// lines of arithmetic.
///
/// The standard alphabet, with the padding, and nothing else: the oracle's
/// output is machine-generated and always padded.
fn base64_decode(text: &str) -> Vec<u8> {
    fn value(c: u8) -> u8 {
        match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => 0,
        }
    }
    let bytes: Vec<u8> = text.bytes().filter(|c| !c.is_ascii_whitespace()).collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut buffer = 0u32;
    let mut bits = 0u32;
    for c in bytes {
        if c == b'=' {
            break;
        }
        buffer = (buffer << 6) | u32::from(value(c));
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buffer >> bits) as u8);
        }
    }
    out
}

/// The sequence length the oracle's design set was built for.
const SEQ: usize = 64;

#[test]
fn the_vector_is_bit_identical_to_the_oracles() {
    let (Some(model_path), Some(expected)) = (
        Some(repo_root().join("models/all-MiniLM-L6-v2.npue")),
        oracle_bytes(),
    ) else {
        eprintln!("skipped: the container or the fixture is not installed");
        return;
    };
    if !model_path.is_file() {
        eprintln!("skipped: the container is not installed");
        return;
    }
    let model = npum_cpu::bert::BertModel::open(&model_path).expect("the container opens");

    let container = npum_format::Container::open(&model_path).expect("the container opens");
    let blob = container
        .tensor_bytes("tokenizer.vocab")
        .expect("the container holds a vocabulary");
    let tokenizer =
        npum_tokenizers::wordpiece::WordPiece::from_lines(&String::from_utf8_lossy(blob))
            .expect("the vocabulary is usable");

    let (encoding, truncated) = tokenizer
        .encode_padded("a golden transcript is a contract", SEQ)
        .expect("the text encodes");
    assert!(!truncated, "the fixture's text fits");

    let vectors = model
        .embed(
            &encoding.ids,
            &encoding.mask,
            1,
            SEQ,
            &mut npum_cpu::ops::CpuOps::best(),
            None,
            None,
        )
        .expect("the model runs");

    let mut got = Vec::with_capacity(vectors.len() * 4);
    for v in &vectors {
        got.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(got.len(), expected.len(), "the vector's width in bytes");

    let mut differing = 0usize;
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        if a != b {
            differing += 1;
            if differing <= 8 {
                let element = i / 4;
                let got = f32::from_le_bytes([got[i], got[i + 1], got[i + 2], got[i + 3]]);
                let want = f32::from_le_bytes([
                    expected[i],
                    expected[i + 1],
                    expected[i + 2],
                    expected[i + 3],
                ]);
                eprintln!(
                    "element {element}: the runtime says {got:?} ({:08x}), the oracle says {want:?} ({:08x})",
                    got.to_bits(),
                    want.to_bits()
                );
            }
        }
    }
    assert_eq!(
        differing,
        0,
        "{differing} of {} bytes differ from the oracle's",
        got.len()
    );
}
