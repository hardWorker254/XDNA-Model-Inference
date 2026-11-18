//! The attention path on the board: two GEMMs per head, with the panels
//! rebuilt and double-buffered on the host.
//!
//! # WHAT THIS CHECKS
//!
//! `attention::run` was reworked to pipeline the panel preparation of the
//! next head with the last in-flight dispatch of the current head, and to
//! submit the two GEMMs through the split submit/wait path. This test checks
//! that the rework still computes the SAME attention as a float32 host
//! reference, on a real design set with real `attn_qk`/`attn_av` streams.
//!
//! # IT SKIPS WHEN THERE IS NO BOARD OR NO DESIGN SET
//!
//! Same rule as the other board tests: absence of hardware is a skip, not a
//! failure.

use std::path::PathBuf;

use npum_model::array::AttentionShape;
use npum_xdna::array::ArrayDesign;
use npum_xdna::attention::{self, Attention};
use npum_xdna::transport::SelectedTransport as Board;
use npum_xdna::transport::Transport;

/// Where a design set may be. Same search as `design_numerics`.
fn design_set() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(dir) = std::env::var("NPU_ARTIFACTS") {
        candidates.push(PathBuf::from(dir));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    candidates.push(root.join("runtime/artifacts"));
    candidates.push(root.join("../Npu-Embeddings-XDNA1/runtime/artifacts"));

    for base in candidates {
        for model in ["all-MiniLM-L6-v2", "vit-base-patch16-224"] {
            for generation in ["artifacts_npu1", "artifacts_npu2"] {
                let dir = base.join(model).join(generation).join("gemm_rtp");
                if dir.join("design.json").exists() {
                    return Some(dir);
                }
            }
        }
    }
    None
}

/// Deterministic pseudo-random f32 in [-1, 1).
fn draw(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let unit = ((*state >> 40) as f32) / (1u32 << 24) as f32;
    unit * 2.0 - 1.0
}

/// The float32 host attention: QK^T, softmax, then softmax.V.
///
/// The model's packer folds 1/sqrt(head_dim) into Q's weights, so -- as the
/// runtime's comment says -- no scale is applied here either.
fn host_attention(qkv: &[f32], shape: AttentionShape, mask: &[f32], ctx: &mut [f32]) {
    let AttentionShape {
        batch,
        seq,
        heads,
        head_dim,
        hidden,
    } = shape;
    for b in 0..batch {
        for h in 0..heads {
            for i in 0..seq {
                let qrow = b * seq * 3 * hidden + i * 3 * hidden + h * head_dim;
                let mut scores = vec![0f32; seq];
                for j in 0..seq {
                    let krow = b * seq * 3 * hidden + j * 3 * hidden + hidden + h * head_dim;
                    let mut s = 0f32;
                    for t in 0..head_dim {
                        s += qkv[qrow + t] * qkv[krow + t];
                    }
                    scores[j] = s + mask[b * seq + j];
                }
                let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in &mut scores {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for s in &mut scores {
                    *s *= inv;
                }
                for t in 0..head_dim {
                    let mut v = 0f32;
                    for j in 0..seq {
                        let vrow =
                            b * seq * 3 * hidden + j * 3 * hidden + 2 * hidden + h * head_dim;
                        v += scores[j] * qkv[vrow + t];
                    }
                    ctx[b * seq * hidden + i * hidden + h * head_dim + t] = v;
                }
            }
        }
    }
}

#[test]
fn the_attention_pipeline_matches_the_host_reference() {
    let Some(dir) = design_set() else {
        eprintln!("skipping: no design set (set NPU_ARTIFACTS to the artifacts root)");
        return;
    };
    let transport = match Board::open(0) {
        Ok(transport) => transport,
        Err(e) => {
            eprintln!("skipping: no NPU board: {e}");
            return;
        }
    };

    let mut design = ArrayDesign::open(&dir, transport).expect("the design set loads");
    let mut attn = Attention::open(&mut design).expect("the attention streams open");

    // MiniLM: hidden 384, 12 heads of 32. The design's attn_qk K is 64
    // (one head padded up to the tile) and N is 384 (the padded key count);
    // attn_av's N is 192 (one context row padded). seq 256 fits one tier.
    let shape = AttentionShape {
        batch: 1,
        seq: 256,
        heads: 12,
        head_dim: 32,
        hidden: 384,
    };
    let n = shape.batch * shape.seq * 3 * shape.hidden;
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let qkv: Vec<f32> = (0..n).map(|_| draw(&mut state)).collect();
    let mask = vec![0f32; shape.batch * shape.seq];

    // Round Q, K and V to bf16 so the reference multiplies what the array
    // multiplied: the device can be blamed for its arithmetic, not for the
    // rounding the caller performed.
    let qkv: Vec<f32> = qkv
        .iter()
        .map(|v| npum_format::bf16_to_f32(npum_format::f32_to_bf16(*v)))
        .collect();

    let mut got = vec![0f32; shape.batch * shape.seq * shape.hidden];
    let mut want = vec![0f32; got.len()];
    host_attention(&qkv, shape, &mask, &mut want);

    let wall = std::time::Instant::now();
    attention::run(
        &mut design,
        &mut attn,
        &qkv,
        shape,
        &mask,
        None, // host softmax: the array's softmax design is a different set
        &mut got,
    )
    .expect("the attention runs on the board");
    let wall = wall.elapsed().as_secs_f64();

    // The head's last dispatch of head h is in flight while head h+1's panels
    // are prepared: every head but the last takes the overlapped path.
    assert_eq!(attn.dispatches, 2 * shape.heads, "two GEMMs per head");

    let scale = want.iter().fold(0f32, |acc, v| acc.max(v.abs()));
    assert!(scale > 0.0, "the reference is all zeros");
    let mut max_err = 0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        max_err = max_err.max((*g - *w).abs() / scale);
    }
    assert!(
        max_err < 2e-2,
        "the array's attention disagrees with the host: max relative error {max_err}"
    );
    eprintln!(
        "  attention: max relative error {max_err:.3e}, dispatches {}, wall {:.3}s, qk {:.3}s, av {:.3}s",
        attn.dispatches, wall, attn.qk_seconds, attn.av_seconds
    );
}
