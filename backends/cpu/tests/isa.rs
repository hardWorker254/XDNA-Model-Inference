//! The ISAs must agree BYTE FOR BYTE.
//!
//! # What this gate is for
//!
//! The plan's 5.1a asks for an AVX-512 path for the CPU and says, in the
//! same breath, that the paths must agree bit for bit. That is not a
//! formality: the GEMM's reduction runs over K, and the order of that
//! reduction does not depend on how many output columns are computed at
//! once -- so the 512-bit GEMM IS the same number as the 256-bit one. But
//! the LayerNorm and the softmax sum a row in EIGHT lanes and fold them
//! with a fixed tree, and a sixteen-lane tree is a different number. The
//! difference between "the width is free" and "the width is the answer" is
//! exactly what this gate measures.
//!
//! # How it is measured
//!
//! The same input, through every ISA this host has, and the BYTES compared.
//! Not a tolerance: a tolerance would let a one-ULP difference through and
//! would call a real difference "close enough".
//!
//! The portable path is in the comparison on purpose. It is the slowest by
//! two orders of magnitude -- `mul_add` without a hardware FMA is a call
//! into libm -- and it is the path that proves the others are not
//! accidentally right.

use npum_cpu::kernels::{available_isas, gemm_nt_on, Isa};

/// A deterministic pseudo-random f32, so the test needs no random seed and
/// repeats exactly.
fn values(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn the_gemm_agrees_across_every_isa() {
    let isas = available_isas();
    eprintln!("this host has: {isas:?}");

    let (m, k, n) = (37usize, 129usize, 70usize);
    let a = values(m * k, 0x1234_5678);
    let b = values(k * n, 0x9abc_def0);
    let bias = values(n, 0x0f0f_0f0f);

    let mut reference: Option<(Isa, Vec<u8>)> = None;
    for isa in isas {
        let mut out = vec![0.0f32; m * n];
        assert!(
            gemm_nt_on(isa, &a, m, &b, k, n, &bias, &mut out),
            "the host does not have {isa:?}"
        );
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        match &reference {
            None => reference = Some((isa, bytes)),
            Some((first_isa, first_bytes)) => {
                assert_eq!(
                    &bytes, first_bytes,
                    "the {isa:?} path differs from the {:?} path",
                    first_isa
                );
            }
        }
    }
}

#[test]
fn the_row_kernels_agree_across_every_isa() {
    let isas = available_isas();

    let gamma = values(384, 0x1111_2222);
    let beta = values(384, 0x3333_4444);
    let row = values(384, 0x5555_6666);

    let mut reference: Option<(Isa, Vec<u8>)> = None;
    for isa in isas {
        let mut norm = row.clone();
        npum_cpu::kernels::layernorm_row_on(isa, &mut norm, &gamma, &beta).expect("the row lays out");
        let bytes: Vec<u8> = norm.iter().flat_map(|v| v.to_le_bytes()).collect();
        match &reference {
            None => reference = Some((isa, bytes)),
            Some((first_isa, first_bytes)) => {
                assert_eq!(
                    &bytes, first_bytes,
                    "the {isa:?} path differs from the {:?} path",
                    first_isa
                );
            }
        }
    }
}

/// The whole model, on every ISA this host has.
///
/// The gate above is on the kernels; this one is on the ANSWER, because a
/// kernel can be bit-exact and the composition of two bit-exact kernels
/// still is not -- if the order in which they are called differs.
#[test]
fn the_model_agrees_across_every_isa() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("the repository root exists");
    let model_path = root.join("models/all-MiniLM-L6-v2.npue");
    if !model_path.is_file() {
        eprintln!("skipped: the container is not installed");
        return;
    }
    let model = npum_cpu::bert::BertModel::open(&model_path).expect("the container opens");
    let container = npum_format::Container::open(&model_path).expect("the container opens");
    let blob = container
        .tensor_bytes("tokenizer.vocab")
        .expect("the container holds a vocabulary");
    let tokenizer = npum_tokenizers::wordpiece::WordPiece::from_lines(&String::from_utf8_lossy(blob))
        .expect("the vocabulary is usable");
    const SEQ: usize = 64;
    let (encoding, _) = tokenizer
        .encode_padded("a golden transcript is a contract", SEQ)
        .expect("the text encodes");

    let mut reference: Option<(Isa, Vec<u8>)> = None;
    for isa in available_isas() {
        let vectors = model
            .embed_on(isa, &encoding.ids, &encoding.mask, 1, SEQ, None)
            .expect("the model runs");
        let bytes: Vec<u8> = vectors.iter().flat_map(|v| v.to_le_bytes()).collect();
        match &reference {
            None => reference = Some((isa, bytes)),
            Some((first_isa, first_bytes)) => {
                assert_eq!(
                    &bytes, first_bytes,
                    "the {isa:?} path differs from the {:?} path",
                    first_isa
                );
            }
        }
    }
}
