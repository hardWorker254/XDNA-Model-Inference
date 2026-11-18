//! The seam: the host's arithmetic against the GPU's, operation by
//! operation and model by model.
//!
//! # WHAT THESE GATES ARE
//!
//! The GPU backend's arithmetic is the TEXT family's: the projections, the
//! LayerNorm, the GELU and the softmax, in the host's f32. These gates are
//! what says that the GPU computes the SAME numbers the host computes --
//! and they say it per operation, because a difference in the model's
//! output says only that something is different, and a reader needs to
//! know WHAT.
//!
//! # THE TOLERANCES, AND WHY THEY ARE NOT ALL THE SAME
//!
//! * **The GEMM is BIT for BIT.** The shader reduces over K in the host's
//!   order, one FMA per step, and the bias is added in the epilogue on
//!   both sides. Zero is not a rounded bound here; it is the answer, and a
//!   difference of one ULP would be a bug in the shader.
//! * **The GELU is BIT for bit.** The same degree-8 polynomial, evaluated
//!   in the same order.
//! * **The LayerNorm and the softmax are within a bound.** Their SUMS are
//!   reduced in a different order -- the host in eight lanes with a fixed
//!   tree, the GPU in sixty-four -- and the GPU's `exp2` is not the host's
//!   polynomial. The bound is on the ABSOLUTE difference for the
//!   LayerNorm, whose output passes through zero, and on the RELATIVE one
//!   for the softmax, whose output is a probability.
//!
//! # WHAT A SKIPPED CASE MEANS
//!
//! On a machine with no Vulkan device the file reports that it did not
//! run. That is a skip and not a pass: the gate prints it, and the run
//! that matters is the one with a GPU.

use npum_cpu::ops::CpuOps;
use npum_model::Ops;
use npum_vulkan::ops::VulkanOps;
use std::path::{Path, PathBuf};

/// A deterministic pseudo-random f32, so a failure repeats exactly.
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

/// The largest absolute difference.
fn worst_abs(gpu: &[f32], host: &[f32]) -> f32 {
    assert_eq!(
        gpu.len(),
        host.len(),
        "the two answers have different sizes"
    );
    gpu.iter()
        .zip(host)
        .map(|(g, h)| (g - h).abs())
        .fold(0.0f32, f32::max)
}

/// The largest relative difference, with a floor so a zero is not a
/// division by zero.
fn worst_rel(gpu: &[f32], host: &[f32]) -> f32 {
    assert_eq!(
        gpu.len(),
        host.len(),
        "the two answers have different sizes"
    );
    gpu.iter()
        .zip(host)
        .map(|(g, h)| (g - h).abs() / h.abs().max(1e-3))
        .fold(0.0f32, f32::max)
}

/// The repository root, from this crate's manifest directory.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("the repository root exists")
}

/// The shapes the text family uses: 64 rows, hidden 384, ffn 1536.
const ROWS: usize = 64;
const HIDDEN: usize = 384;
const FFN: usize = 1536;

#[test]
fn the_gemm_is_bit_identical_to_the_host() {
    let mut gpu = VulkanOps::new();
    if !gpu.has_gpu() {
        eprintln!("SKIP: no Vulkan device on this machine");
        return;
    }
    let mut cpu = CpuOps::best();

    // The four projections' shapes: the QKV one and the FFN's two.
    for (m, k, n) in [
        (ROWS, HIDDEN, 3 * HIDDEN),
        (ROWS, HIDDEN, HIDDEN),
        (ROWS, HIDDEN, FFN),
        (ROWS, FFN, HIDDEN),
    ] {
        let a = values(m * k, 1);
        let b = values(k * n, 2);
        let bias = values(n, 3);
        let mut host = vec![0.0f32; m * n];
        let mut on_gpu = vec![0.0f32; m * n];
        cpu.gemm_nt(&a, m, &b, k, n, &bias, &mut host).unwrap();
        gpu.gemm_nt(&a, m, &b, k, n, &bias, &mut on_gpu).unwrap();
        assert_eq!(
            host,
            on_gpu,
            "the GEMM {m}x{k}x{n} differs from the host's by {}",
            worst_abs(&on_gpu, &host)
        );
    }
}

#[test]
fn the_gelu_is_bit_identical_to_the_host() {
    let mut gpu = VulkanOps::new();
    if !gpu.has_gpu() {
        eprintln!("SKIP: no Vulkan device on this machine");
        return;
    }
    let mut cpu = CpuOps::best();
    let activations = values(ROWS * HIDDEN, 5);
    let mut host = activations.clone();
    let mut on_gpu = activations;
    cpu.gelu(&mut host).unwrap();
    gpu.gelu(&mut on_gpu).unwrap();
    assert_eq!(
        host,
        on_gpu,
        "the GELU differs from the host's by {}",
        worst_abs(&on_gpu, &host)
    );
}

#[test]
fn the_layernorm_is_within_1e_5_of_the_host() {
    let mut gpu = VulkanOps::new();
    if !gpu.has_gpu() {
        eprintln!("SKIP: no Vulkan device on this machine");
        return;
    }
    let mut cpu = CpuOps::best();
    let activations = values(ROWS * HIDDEN, 6);
    let gamma = values(HIDDEN, 7);
    let beta = values(HIDDEN, 8);
    let mut host = activations.clone();
    let mut on_gpu = activations;
    cpu.layernorm_rows(&mut host, ROWS, HIDDEN, &gamma, &beta)
        .unwrap();
    gpu.layernorm_rows(&mut on_gpu, ROWS, HIDDEN, &gamma, &beta)
        .unwrap();
    let error = worst_abs(&on_gpu, &host);
    eprintln!("layernorm: worst absolute difference {error:e}");
    assert!(error <= 1e-5, "the LayerNorm differs by {error:e}");
}

#[test]
fn the_softmax_is_within_1e_5_of_the_host() {
    let mut gpu = VulkanOps::new();
    if !gpu.has_gpu() {
        eprintln!("SKIP: no Vulkan device on this machine");
        return;
    }
    let mut cpu = CpuOps::best();
    // The score matrix's shape: batch 1, 12 heads, 64 keys.
    let (rows, width) = (12 * 64, 64);
    let mut scores = values(rows * width, 9);
    // The padding's mask, which is what the clamps in the shader are for.
    for row in 0..rows {
        scores[row * width + 3] = -1e30;
    }
    let mut host = scores.clone();
    let mut on_gpu = scores;
    cpu.softmax_rows(&mut host, rows, width).unwrap();
    gpu.softmax_rows(&mut on_gpu, rows, width).unwrap();
    let error = worst_rel(&on_gpu, &host);
    eprintln!("softmax: worst relative difference {error:e}");
    assert!(error <= 1e-5, "the softmax differs by {error:e}");
}

/// The sequence length the family's design set was built for.
const SEQ: usize = 64;

#[test]
fn the_model_on_the_gpu_matches_the_host() {
    let mut gpu = VulkanOps::new();
    if !gpu.has_gpu() {
        eprintln!("SKIP: no Vulkan device on this machine");
        return;
    }
    let model_path = repo_root().join("models/all-MiniLM-L6-v2.npue");
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
    assert!(!truncated, "the text fits");

    let host = model
        .embed(
            &encoding.ids,
            &encoding.mask,
            1,
            SEQ,
            &mut CpuOps::best(),
            None,
            None,
        )
        .expect("the model runs on the host");
    let on_gpu = model
        .embed(&encoding.ids, &encoding.mask, 1, SEQ, &mut gpu, None, None)
        .expect("the model runs on the GPU");

    // The vectors are L2-normalised, so the cosine is the number that
    // says "the same direction". The bound is the plan's: 2e-3.
    let dot: f64 = host
        .iter()
        .zip(&on_gpu)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let host_len: f64 = host
        .iter()
        .map(|a| (*a as f64) * (*a as f64))
        .sum::<f64>()
        .sqrt();
    let gpu_len: f64 = on_gpu
        .iter()
        .map(|a| (*a as f64) * (*a as f64))
        .sum::<f64>()
        .sqrt();
    let cosine = dot / (host_len * gpu_len);
    eprintln!("the vectors' cosine: {cosine:.12}");
    eprintln!("the GPU dispatched {} operations", gpu.dispatches());
    assert!(
        (1.0 - cosine) <= 2e-3,
        "the GPU's vector differs from the host's: 1 - cos = {:e}",
        1.0 - cosine
    );
    assert!(
        gpu.dispatches() > 0,
        "the run claimed the GPU and dispatched nothing"
    );
}
