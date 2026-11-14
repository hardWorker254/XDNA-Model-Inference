//! The container's own design set and the container's own weights, on the
//! board.
//!
//! # WHAT THIS CLOSES
//!
//! The other checks feed the array a design from a DIRECTORY and operands
//! this test made up. This one uses what a model actually ships with: the
//! design set the container carries (`design/gemm_rtp/...`) and the weight
//! panel the container stores (`layer.0.qkv`, already pre-tiled into the
//! MMAC's byte order).
//!
//! That is the check that spans design, file and hardware at once. The
//! synthetic path cannot notice that the container and the design disagree
//! about the generation, because both sides then agree on a layout the
//! hardware does not consume -- the byte count is right, the shapes agree,
//! the recorded hash matches on both sides, and every product is plausible.
//!
//! # THE REFERENCE IS THE PANEL DE-TILED
//!
//! The container stores the weights already rounded to bf16 and already
//! tiled, so the reference is the panel de-tiled with its OWN recorded
//! layout, widened back to f32, multiplied in f32. The device is then judged
//! on its own arithmetic and not on the rounding the caller did.
//!
//! # IT SKIPS WHEN THERE IS NO BF16 CONTAINER
//!
//! The container is not in the tree: every container in the tree is f32, and
//! the array's MMAC multiplies bf16 and int8. The recipe for making one is
//! in the comment below.

use npum_format::tiling::{untile_b, TileOrder};
use npum_xdna::array::ArrayDesign;
use npum_xdna::transport::Transport;
use npum_xdna::xrt::XrtTransport;
use npum_format::Container;

/// bf16 carries 8 significand bits, so one output rounding is at most half an
/// ulp = 2^-9 relative. Anything beyond that is not a rounding artefact.
const BF16_HALF_ULP: f64 = 1.0 / 512.0;

/// The bound the Python harness uses for the relative error.
const MAX_RELATIVE: f64 = 4.0 * BF16_HALF_ULP;

/// The bound the Python harness uses for 1 - cos.
const MAX_ONE_MINUS_COS: f64 = 2e-3;

/// The container this test uses.
///
/// THE RECIPE:
///
/// ```text
/// # the model's own ONNX, as the tree's CHECKPOINT.json records it:
/// mkdir -p ~/Npu-Embeddings-XDNA1/models/all-MiniLM-L6-v2/onnx
/// curl -L -o .../onnx/model.onnx \
///   https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
///
/// # then pack it for the array, with the design set inside the container:
/// cd ~/Npu-Embeddings-XDNA1
/// python3 tools/pack/pack_npue.py \
///   --model-dir models/all-MiniLM-L6-v2 \
///   --out /tmp/minilm-bf16.npue \
///   --dtype bf16 \
///   --embed-artifacts runtime/artifacts/all-MiniLM-L6-v2/artifacts_npu1
///
/// NPU_BF16_CONTAINER=/tmp/minilm-bf16.npue cargo test --release -p npum-xdna
/// ```
fn container_path() -> Option<std::path::PathBuf> {
    let path = std::env::var("NPU_BF16_CONTAINER").ok()?;
    let path = std::path::PathBuf::from(path);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// f32 to bf16, round to nearest even.
fn to_bf16(x: f32) -> u16 {
    npum_format::f32_to_bf16(x)
}

/// bf16 to f32.
fn from_bf16(bits: u16) -> f32 {
    npum_format::bf16_to_f32(bits)
}

/// A deterministic pseudo-random f32 in [-1, 1).
fn draw(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let unit = ((*state >> 40) as f32) / (1u32 << 24) as f32;
    unit * 2.0 - 1.0
}

#[test]
fn the_container_s_own_weights_go_through_the_array() {
    let Some(path) = container_path() else {
        eprintln!("skipping: no bf16 container (set NPU_BF16_CONTAINER)");
        return;
    };
    let transport = match XrtTransport::open(0) {
        Ok(transport) => transport,
        Err(e) => {
            eprintln!("skipping: no NPU board: {e}");
            return;
        }
    };

    let mut design =
        ArrayDesign::open_from_container(&path, "gemm_rtp", transport).expect("the container's design loads");
    let info = design.info().clone();

    // THE STREAM: the qkv one, at the smallest batch tier.
    //
    // The tiers are the design's own: 4, 8 and 16 batches of 64 rows. The
    // smallest is the one a single sentence wants, and using it is what
    // proves the tier selection rather than the biggest fit.
    let stream = info
        .streams
        .iter()
        .filter(|s| s.op == "qkv" && s.m > 0)
        .min_by_key(|s| s.m)
        .expect("the design set has a qkv stream with a shape");
    let (m, k, n) = (stream.m as usize, stream.k as usize, stream.n as usize);

    // THE CONTAINER'S OWN PANEL.
    let container = Container::open(&path).expect("the container opens");
    let panel = container
        .tensor_bytes("layer.0.qkv")
        .expect("the container carries the qkv panel");
    assert_eq!(
        panel.len(),
        k * n * 2,
        "the container's qkv panel is not [{},{}] bf16",
        k,
        n
    );

    // THE PANEL DE-TILED, WITH ITS OWN RECORDED LAYOUT.
    let layout = info.b_layout.as_ref().expect("the manifest records the B layout");
    let (tile_k, tile_n) = (layout.tile_k as usize, layout.tile_n as usize);
    let (s, t) = (layout.mac_s as usize, layout.mac_t as usize);
    let weights = untile_b(
        panel,
        k,
        n,
        tile_k,
        tile_n,
        2,
        Some(s),
        Some(t),
        TileOrder::Kn,
    )
    .expect("the panel de-tiles");
    let weights: Vec<f32> = weights
        .chunks_exact(2)
        .map(|pair| from_bf16(u16::from_le_bytes([pair[0], pair[1]])))
        .collect();

    // THE A OPERAND.
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let rows = 8usize;
    let mut a_bytes = Vec::with_capacity(rows * k * 2);
    let mut a_values = Vec::with_capacity(rows * k);
    for _ in 0..(rows * k) {
        let rounded = to_bf16(draw(&mut state));
        a_bytes.extend_from_slice(&rounded.to_le_bytes());
        a_values.push(from_bf16(rounded));
    }

    // THE PANEL, STAGED AS IT IS STORED.
    let slot_b = design.stage(1, panel).expect("the panel is staged");
    design.bind(1, slot_b).expect("the panel is bound");

    let mut out = vec![0f32; rows * n];
    npum_xdna::gemm::gemm(
        &mut design,
        stream.slot as usize,
        &a_values,
        rows,
        m,
        k,
        None,
        n,
        &mut out,
    )
    .expect("the op runs");

    // THE REFERENCE: the same weights, in f32, multiplied in f32.
    let mut want = vec![0f32; rows * n];
    for i in 0..rows {
        for kk in 0..k {
            let factor = a_values[i * k + kk];
            for j in 0..n {
                want[i * n + j] += factor * weights[kk * n + j];
            }
        }
    }

    let mut dot = 0f64;
    let mut norm_got = 0f64;
    let mut norm_want = 0f64;
    let mut max_err = 0f32;
    for (g, w) in out.iter().zip(want.iter()) {
        dot += *g as f64 * *w as f64;
        norm_got += (*g as f64) * (*g as f64);
        norm_want += (*w as f64) * (*w as f64);
        max_err = max_err.max((g - w).abs());
    }
    let one_minus_cos = 1.0 - dot / (norm_got.sqrt() * norm_want.sqrt());
    let scale = want.iter().fold(0f32, |acc, v| acc.max(v.abs()));
    let max_rel = max_err as f64 / scale as f64;

    eprintln!(
        "  container  M{rows} K{k} N{n}  1-cos {one_minus_cos:.2e}  max|d|/|ref|max {max_rel:.2e}"
    );
    assert!(
        one_minus_cos <= MAX_ONE_MINUS_COS,
        "the array's answer is not the reference's: 1-cos {one_minus_cos:.6e}"
    );
    assert!(
        max_rel <= MAX_RELATIVE,
        "the array's answer is not the reference's: max|d|/|ref|max {max_rel:.6e}"
    );
}
