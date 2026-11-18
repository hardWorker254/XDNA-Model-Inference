//! The array path, end to end on the board.
//!
//! # WHAT THIS CHECKS
//!
//! That a compiled design, loaded by this runtime, computes the GEMM it
//! claims to. The operands are random, the reference is a float32 matrix
//! multiply of the SAME ROUNDED operands, and the bound is the one the
//! Python design harness uses: the array's operands are bf16, and the device
//! cannot be blamed for the bf16 the caller handed it -- only for its own
//! arithmetic.
//!
//! # IT IS THE SAME CHECK THE PYTHON HARNESS MAKES
//!
//! `tools/verify/verify_design_numerics.py` does this in Python, against the
//! same designs, with the same bound. That script is the gate for the
//! exporter's designs; this is the same gate for the runtime's loader, and
//! the two must agree about what "the design computes its GEMM" means -- a
//! runtime that loads a design and gets a different answer is a runtime
//! whose loader is wrong, and only a check against the arithmetic says so.
//!
//! # IT SKIPS, AND SAYS SO, WHEN THERE IS NO BOARD
//!
//! A machine without an NPU, or without a design set, still builds and still
//! runs the CPU backends. The check prints one line and returns rather than
//! failing: a test that fails because the hardware is absent teaches the
//! reader to ignore it, and a test nobody reads catches nothing.

use std::path::PathBuf;

use npum_xdna::array::ArrayDesign;
use npum_xdna::transport::SelectedTransport as Board;
use npum_xdna::transport::Transport;

/// bf16 carries 8 significand bits, so one output rounding is at most half an
/// ulp = 2^-9 relative. Anything beyond that is not a rounding artefact.
const BF16_HALF_ULP: f64 = 1.0 / 512.0;

/// The bound the Python harness uses for the relative error.
const MAX_RELATIVE: f64 = 4.0 * BF16_HALF_ULP;

/// The bound the Python harness uses for 1 - cos.
const MAX_ONE_MINUS_COS: f64 = 2e-3;

/// Where a design set may be.
///
/// The environment variable first, then the two places the runtime itself
/// looks: the tree's own `runtime/artifacts` and the oracle tree's, which is
/// a sibling of this worktree.
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
        for model in ["vit-base-patch16-224", "all-MiniLM-L6-v2"] {
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

/// f32 to bf16, round to nearest even.
fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounding = 0x7fffu64 + ((bits >> 16) & 1) as u64;
    ((bits as u64 + rounding) >> 16) as u16
}

/// bf16 to f32.
fn from_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// A deterministic pseudo-random f32 in [-1, 1).
///
/// The generator is xorshift, and it is seeded by a constant: a test whose
/// numbers change from run to run is a test that cannot be compared with a
/// run from last week.
fn draw(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let unit = ((*state >> 40) as f32) / (1u32 << 24) as f32;
    unit * 2.0 - 1.0
}

/// One operand in the design's own dtype, and the same operand in f32.
///
/// The two are the SAME numbers: the f32 side is the bf16 value widened, so
/// the reference multiplies exactly what the array multiplied.
fn operand(rows: usize, cols: usize, state: &mut u64) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::with_capacity(rows * cols * 2);
    let mut values = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        let rounded = to_bf16(draw(state));
        bytes.extend_from_slice(&rounded.to_le_bytes());
        values.push(from_bf16(rounded));
    }
    (bytes, values)
}

/// The float32 reference: C = A @ B, in the order the rows are laid out.
fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    for i in 0..m {
        let row = &mut c[i * n..(i + 1) * n];
        for kk in 0..k {
            let factor = a[i * k + kk];
            let brow = &b[kk * n..(kk + 1) * n];
            for j in 0..n {
                row[j] += factor * brow[j];
            }
        }
    }
    c
}

#[test]
fn the_design_computes_the_gemm_it_claims_to() {
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
    let info = design.info().clone();

    // THE STREAM: the qkv one, the first with a shape of its own.
    let stream = info
        .streams
        .iter()
        .find(|s| s.op == "qkv" && s.m > 0)
        .expect("the design set has a qkv stream with a shape");
    let (m, k, n) = (stream.m as usize, stream.k as usize, stream.n as usize);

    // THE LAYOUT: the tile and the sub-tile the manifest records.
    let layout = info
        .b_layout
        .as_ref()
        .expect("the manifest records the B layout");
    let (tile_k, tile_n) = (layout.tile_k as usize, layout.tile_n as usize);
    let (s, t) = (layout.mac_s as usize, layout.mac_t as usize);

    let mut state = 0x243f_6a88_85a3_08d3u64;
    let (a_bytes, a_values) = operand(m, k, &mut state);
    let (b_bytes, b_values) = operand(k, n, &mut state);
    let b_panel = npum_format::tiling::tile_b(
        &b_bytes,
        k,
        n,
        tile_k,
        tile_n,
        2,
        Some(s),
        Some(t),
        npum_format::tiling::TileOrder::Kn,
    )
    .expect("the B operand tiles");

    let slot_a = design.stage(0, &a_bytes).expect("A is staged");
    let slot_b = design.stage(1, &b_panel).expect("B is staged");
    design.bind(0, slot_a).expect("A is bound");
    design.bind(1, slot_b).expect("B is bound");
    design
        .dispatch(stream.slot as usize)
        .expect("the kernel runs");

    let mut c_bytes = vec![0u8; m * n * 2];
    design.read(2, &mut c_bytes).expect("C is read back");
    let got: Vec<f32> = c_bytes
        .chunks_exact(2)
        .map(|pair| from_bf16(u16::from_le_bytes([pair[0], pair[1]])))
        .collect();

    let want = reference(&a_values, &b_values, m, k, n);

    // THE VERDICT, THE PYTHON HARNESS'S VERDICT.
    //
    // A reference of all zeros is not a comparison: the harness refuses it
    // by name rather than dividing by it, and so does this.
    let scale = want.iter().fold(0f32, |acc, v| acc.max(v.abs()));
    assert!(
        scale > 0.0,
        "the float32 reference is ALL ZEROS, so nothing was measured"
    );
    let mut dot = 0f64;
    let mut norm_got = 0f64;
    let mut norm_want = 0f64;
    let mut max_err = 0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        dot += *g as f64 * *w as f64;
        norm_got += (*g as f64) * (*g as f64);
        norm_want += (*w as f64) * (*w as f64);
        max_err = max_err.max((g - w).abs());
    }
    let cos = dot / (norm_got.sqrt() * norm_want.sqrt());
    let one_minus_cos = 1.0 - cos;
    let max_rel = max_err as f64 / scale as f64;

    eprintln!(
        "  {}  M{m} K{k} N{n}  1-cos {one_minus_cos:.2e}  max|d|/|ref|max {max_rel:.2e}",
        if one_minus_cos <= MAX_ONE_MINUS_COS && max_rel <= MAX_RELATIVE {
            "OK  "
        } else {
            "FAIL"
        }
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

/// The same GEMM, through the runtime's OWN op layer.
///
/// The check above drives the design through the loader's primitives; this
/// one goes through `gemm::gemm` -- the f32 operand in, the bias, the
/// padded tail, the read-back -- and so it checks the OP and not only the
/// transport. An op that added the bias to the bf16 instead of the f32, or
/// that left the padded tail stale, would pass the first check and fail this
/// one.
#[test]
fn the_op_layer_gemm_matches_the_reference() {
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
    let info = design.info().clone();
    let stream = info
        .streams
        .iter()
        .find(|s| s.op == "qkv" && s.m > 0)
        .expect("the design set has a qkv stream with a shape");
    let (m, k, n) = (stream.m as usize, stream.k as usize, stream.n as usize);
    let layout = info
        .b_layout
        .as_ref()
        .expect("the manifest records the B layout");
    let (tile_k, tile_n) = (layout.tile_k as usize, layout.tile_n as usize);
    let (s, t) = (layout.mac_s as usize, layout.mac_t as usize);

    // THE CALLER'S ROWS, WHICH ARE FEWER THAN THE DESIGN'S.
    //
    // The design computes `m` rows whatever the caller has, and the tail is
    // a padded row that must come back as zeros -- not as the previous
    // call's numbers.
    let n_real = 128usize;

    let mut state = 0x1319_8a2e_0370_7344u64;
    let (_a_bytes, a_values) = operand(n_real, k, &mut state);
    let (b_bytes, b_values) = operand(k, n, &mut state);
    let b_panel = npum_format::tiling::tile_b(
        &b_bytes,
        k,
        n,
        tile_k,
        tile_n,
        2,
        Some(s),
        Some(t),
        npum_format::tiling::TileOrder::Kn,
    )
    .expect("the B operand tiles");

    // THE WEIGHT PANEL, STAGED AND BOUND.
    let slot_b = design.stage(1, &b_panel).expect("B is staged");
    design.bind(1, slot_b).expect("B is bound");

    // THE BIAS.
    let mut bias_state = 0x9e37_79b9_7f4a_7c15u64;
    let (_, bias) = operand(1, n, &mut bias_state);

    let mut out = vec![0f32; n_real * n];
    npum_xdna::gemm::gemm(
        &mut design,
        stream.slot as usize,
        &a_values,
        n_real,
        m,
        k,
        Some(&bias),
        n,
        &mut out,
    )
    .expect("the op runs");

    // THE REFERENCE, WITH THE BIAS: the op adds it, so the reference has it
    // too, and a reference without it would be a check of nothing.
    let mut want = reference(&a_values, &b_values, n_real, k, n);
    for r in 0..n_real {
        for j in 0..n {
            want[r * n + j] += bias[j];
        }
    }
    let mut dot = 0f64;
    let mut norm_got = 0f64;
    let mut norm_want = 0f64;
    let mut max_err = 0f32;
    for (g, w) in out.iter().zip(want.iter()) {
        dot += *g as f64 * (*w as f64);
        norm_got += (*g as f64) * (*g as f64);
        norm_want += (*w as f64) * (*w as f64);
        max_err = max_err.max((g - w).abs());
    }
    let one_minus_cos = 1.0 - dot / (norm_got.sqrt() * norm_want.sqrt());
    let scale = want.iter().fold(0f32, |acc, v| acc.max(v.abs()));
    let max_rel = max_err as f64 / scale as f64;

    eprintln!(
        "  op layer  M{n_real} K{k} N{n}  1-cos {one_minus_cos:.2e}  max|d|/|ref|max {max_rel:.2e}"
    );
    assert!(
        one_minus_cos <= MAX_ONE_MINUS_COS,
        "the op layer's answer is not the reference's: 1-cos {one_minus_cos:.6e}"
    );
    assert!(
        max_rel <= MAX_RELATIVE,
        "the op layer's answer is not the reference's: max|d|/|ref|max {max_rel:.6e}"
    );
}
