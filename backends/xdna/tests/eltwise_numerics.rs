//! The eltwise pipeline on the board: a softmax over more rows than one
//! dispatch covers, so the host conversion of the next chunk overlaps the
//! device's current dispatch.
//!
//! Skips without a board or a design set, like the other board tests.

use std::path::PathBuf;

use npum_xdna::array::ArrayDesign;
use npum_xdna::eltwise::EltwiseDesign;
use npum_xdna::transport::SelectedTransport as Board;
use npum_xdna::transport::Transport;

fn design_set() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(dir) = std::env::var("NPU_ARTIFACTS") {
        candidates.push(PathBuf::from(dir));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    candidates.push(root.join("runtime/artifacts"));

    for base in candidates {
        for model in ["all-MiniLM-L6-v2", "vit-base-patch16-224"] {
            let dir = base.join(model).join("artifacts_npu1").join("softmax");
            if dir.join("design.json").exists() {
                return Some(dir);
            }
        }
    }
    None
}

fn draw(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let unit = ((*state >> 40) as f32) / (1u32 << 24) as f32;
    unit * 2.0 - 1.0
}

#[test]
fn the_softmax_pipeline_matches_the_host_reference() {
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
    let design = ArrayDesign::open(&dir, transport).expect("the softmax design loads");
    let mut pass = EltwiseDesign::open(design).expect("the softmax pass opens");

    let cols = pass.cols();
    let rows_per_dispatch = pass.rows();
    // More than one dispatch's worth: the pipelined path runs.
    let n_rows = rows_per_dispatch * 4;
    let total = n_rows * cols;
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let x: Vec<f32> = (0..total)
        .map(|_| npum_format::bf16_to_f32(npum_format::f32_to_bf16(draw(&mut state))))
        .collect();
    let mut got = x.clone();

    let mut want = vec![0f32; total];
    for r in 0..n_rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = row.iter().map(|v| (*v - mx).exp()).collect();
        let sum: f32 = exp.iter().sum();
        for j in 0..cols {
            want[r * cols + j] = exp[j] / sum;
        }
    }

    pass.softmax(&mut got, n_rows).expect("the softmax runs");

    assert_eq!(
        pass.dispatches(),
        4,
        "four chunks, four dispatches (the pipelined path)"
    );
    let mut max_err = 0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        max_err = max_err.max((*g - *w).abs());
    }
    assert!(
        max_err < 5e-3,
        "the array's softmax disagrees with the host: max absolute error {max_err}"
    );
    eprintln!(
        "  softmax: {n_rows} rows in {} dispatches, max error {max_err:.3e}",
        pass.dispatches()
    );
}
