//! The array's CONVOLUTION: the container's own panel, on the board.
//!
//! # WHAT THIS CLOSES
//!
//! The design-level check feeds the array a design and operands this test
//! made up, and the container-level check feeds it a projection's panel. This
//! one feeds it a CONVOLUTION's panel -- the one a detector ships with -- and
//! so it is the check that exercises the two things a projection does not:
//!
//! * the PADDED COLUMNS of the A operand, which a convolution has and a
//!   projection does not: the design's K is a multiple of the tile, and the
//!   convolution's own K is not;
//!
//! * the NARROWING of the result: the design's N is a multiple of the
//!   device's columns, and the convolution's own N is not.
//!
//! Both are places where a stale value is a WRONG NUMBER rather than a
//! wrong shape, and a wrong shape is the one failure a test usually catches.
//!
//! # THE PANEL IS THE CONTAINER'S, AND THE REFERENCE IS ITS OWN DE-TILING
//!
//! The container stores the weights already rounded to bf16, already padded
//! to the design's widths and already tiled into the MMAC's byte order. The
//! reference is that panel de-tiled with its OWN recorded layout, narrowed to
//! the convolution's own shape, and multiplied in f32 -- so the device is
//! judged on its arithmetic, and the padding is checked for being zero as
//! well.
//!
//! # IT SKIPS, AND SAYS SO, WHEN THERE IS NO BOARD
//!
//! As in the other checks: a machine without an NPU still builds and still
//! runs the CPU backends, and a test that fails because the hardware is
//! absent teaches the reader to ignore it.

use std::path::PathBuf;

use npum_format::tiling::{untile_b, TileOrder};
use npum_xdna::array::ArrayDesign;
use npum_xdna::conv::{conv, ConvShape};
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

/// The design set of the OCR family: the one whose streams are CONVOLUTIONS'.
///
/// The conv families' design sets name their streams after the padded shape
/// they compute (`conv64x128`), so a design set of this kind is the only one
/// whose stream names say what they compute.
fn design_set() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = std::env::var("NPU_ARTIFACTS") {
        candidates.push(PathBuf::from(dir));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .to_path_buf();
    candidates.push(root.join("runtime/artifacts"));
    candidates.push(root.join("../Npu-Embeddings-XDNA1/runtime/artifacts"));

    for base in candidates {
        for generation in ["artifacts_npu1", "artifacts_npu2"] {
            let dir = base
                .join("paddleocr-onnx")
                .join(generation)
                .join("gemm_rtp");
            if dir.join("design.json").exists() {
                return Some(dir);
            }
        }
    }
    None
}

/// The container with a CONVOLUTION's panel in it.
///
/// The OCR family is the one whose container carries panels for its dense
/// convolutions, and the tree's own copy is the parity fixture -- the same
/// container the oracle's tree holds.
fn container_path() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .to_path_buf();
    for candidate in [
        root.join("tools/parity/fixtures/root/models/paddleocr-onnx.npue"),
        PathBuf::from("/home/prof/Npu-Embeddings-XDNA1/models/paddleocr-onnx.npue"),
    ] {
        if candidate.exists() {
            return Some(candidate);
        }
    }
    std::env::var("NPU_OCR_CONTAINER")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
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
fn the_container_s_own_convolution_panel_goes_through_the_array() {
    let Some(dir) = design_set() else {
        eprintln!("skipping: no conv design set (set NPU_ARTIFACTS)");
        return;
    };
    let Some(path) = container_path() else {
        eprintln!("skipping: no OCR container");
        return;
    };
    let transport = match XrtTransport::open(0) {
        Ok(transport) => transport,
        Err(e) => {
            eprintln!("skipping: no NPU board: {e}");
            return;
        }
    };

    let mut design = ArrayDesign::open(&dir, transport).expect("the design loads");
    let info = design.info().clone();

    // THE STREAM: the smallest (K, N) the design set has, which is the one
    // the first convolution of the detector dispatches against.
    let stream = info
        .streams
        .iter()
        .filter(|s| s.m > 0 && s.k > 0 && s.n > 0)
        .min_by_key(|s| (s.k, s.n))
        .expect("the design set has a stream with a shape");
    let (rows, padded_k, padded_n) = (stream.m as usize, stream.k as usize, stream.n as usize);

    // THE CONVOLUTION'S OWN SHAPE, FROM ITS OWN WEIGHTS.
    //
    // The container's weight is [cout, cin, kh, kw], so the GEMM's K is
    // cin*kh*kw and its N is cout. Reading them out of the container rather
    // than writing them down keeps the test honest about which convolution
    // it is checking: a test that hard-codes the shape checks the shape it
    // wrote down, and not the one in the container.
    let container = Container::open(&path).expect("the container opens");
    let entry = container
        .directory()
        .tensor("det.conv.0.w")
        .expect("the container carries the first convolution's weights");
    let shape_w: Vec<i64> = entry.logical_shape.clone();
    assert_eq!(
        shape_w.len(),
        4,
        "the convolution's weight is not [cout, cin, kh, kw]: {shape_w:?}"
    );
    let (n, cin, kh, kw) = (
        shape_w[0] as usize,
        shape_w[1] as usize,
        shape_w[2] as usize,
        shape_w[3] as usize,
    );
    let (k, n) = (cin * kh * kw, n);
    assert!(
        k <= padded_k && n <= padded_n,
        "the design's {padded_k}x{padded_n} does not hold the convolution's {k}x{n}"
    );
    assert!(
        k < padded_k || n < padded_n,
        "this convolution fits its design's widths exactly, so the padding and \
         the narrowing are not exercised -- pick another convolution"
    );

    // THE PANEL, STAGED AS IT IS STORED.
    let panel = container
        .tensor_bytes("det.conv.0.btile")
        .expect("the container carries the first convolution's panel");
    assert_eq!(
        panel.len(),
        padded_k * padded_n * 2,
        "the container's panel is not [{},{}] bf16",
        padded_k,
        padded_n
    );

    // THE PANEL DE-TILED, WITH ITS OWN RECORDED LAYOUT.
    let layout = info.b_layout.as_ref().expect("the manifest records the B layout");
    let (tile_k, tile_n) = (layout.tile_k as usize, layout.tile_n as usize);
    let (s, t) = (layout.mac_s as usize, layout.mac_t as usize);
    let weights = untile_b(
        panel,
        padded_k,
        padded_n,
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

    // THE PADDING IS ZERO, AND THAT IS EXACT.
    //
    // A padded K row is multiplied by nothing, and a padded N column is a
    // channel the tensor does not have. The packer says so, and the panel
    // says so, and this says so about the panel.
    for kk in k..padded_k {
        for j in 0..padded_n {
            assert_eq!(
                weights[kk * padded_n + j],
                0.0,
                "the panel's padding at ({kk},{j}) is not zero"
            );
        }
    }
    for kk in 0..k {
        for j in n..padded_n {
            assert_eq!(
                weights[kk * padded_n + j],
                0.0,
                "the panel's padding at ({kk},{j}) is not zero"
            );
        }
    }

    // THE BIAS, THE CONVOLUTION'S OWN.
    let bias_bytes = container
        .tensor_bytes("det.conv.0.b")
        .expect("the container carries the first convolution's bias");
    let bias: Vec<f32> = bias_bytes
        .chunks_exact(4)
        .map(|quad| f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
        .collect();
    assert_eq!(bias.len(), n, "the bias is not the convolution's own width");

    // THE A OPERAND: THE HOST'S im2col ROWS.
    //
    // The rows are the convolution's own K, not the design's: the padding is
    // the op's business, and a test that padded the rows itself would not
    // be testing that.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let n_rows = 8usize;
    let mut a_values = Vec::with_capacity(n_rows * k);
    for _ in 0..(n_rows * k) {
        a_values.push(from_bf16(to_bf16(draw(&mut state))));
    }

    // THE DISPATCH.
    let shape = ConvShape {
        rows,
        k,
        n,
        padded_k,
        padded_n,
    };
    let slot_b = design.stage(1, panel).expect("the panel is staged");
    design.bind(1, slot_b).expect("the panel is bound");

    let mut out = vec![0f32; n_rows * n];
    let mut scratch = npum_xdna::conv::ConvScratch::default();
    conv(
        &mut design,
        stream.slot as usize,
        &shape,
        Some(&bias),
        &a_values,
        &mut out,
        &mut scratch,
    )
    .expect("the op runs");

    // THE REFERENCE: the same weights, in f32, multiplied in f32.
    let mut want = vec![0f32; n_rows * n];
    for i in 0..n_rows {
        for kk in 0..k {
            let factor = a_values[i * k + kk];
            for j in 0..n {
                want[i * n + j] += factor * weights[kk * padded_n + j];
            }
        }
        for j in 0..n {
            want[i * n + j] += bias[j];
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
        "  conv       M{n_rows} K{k}->{padded_k} N{n}->{padded_n}  \
         1-cos {one_minus_cos:.2e}  max|d|/|ref|max {max_rel:.2e}"
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
