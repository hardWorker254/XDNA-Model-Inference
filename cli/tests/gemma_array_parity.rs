//! Gemma host/array parity, gated on the board and the container.
//!
//! `NPUEMBEDDINGS_GEMMA_NPU=/path/to/embeddinggemma-300m.npue`, optionally
//! `NPU_ARTIFACTS`. The ids are fixed so the same text is compared on both
//! paths; the point is the array's arithmetic, not the tokenizer.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use npuembeddings_cli::flags::Args;
use npum_cpu::ops::CpuOps;

const BATCH: usize = 2;
const SEQ: usize = 32;
const RTOL_COS: f64 = 5e-3;

fn board_present() -> bool {
    Path::new("/dev/accel/accel0").exists() || Path::new("/dev/dri/accel0").exists()
}

fn container() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("NPUEMBEDDINGS_GEMMA_NPU").ok()?);
    if p.is_file() { Some(p) } else { None }
}

fn args(codes: &str, root: Option<&Path>) -> Args {
    let mut args = Args::default();
    let mut flags = HashMap::new();
    flags.insert("--npu-ops".to_string(), codes.to_string());
    if let Some(root) = root {
        flags.insert("--artifacts".to_string(), root.display().to_string());
    }
    args.flags = flags;
    args
}

#[test]
fn gemma_array_path_matches_the_host() {
    if !board_present() {
        eprintln!("skipping: no NPU device node");
        return;
    }
    let Some(path) = container() else {
        eprintln!("skipping: set NPUEMBEDDINGS_GEMMA_NPU to a Gemma container");
        return;
    };
    let container = npum_format::Container::open(&path).expect("the container opens");
    let model = npum_cpu::gemma::Model::from_container(&container).expect("the Gemma model opens");
    let mut ids = vec![0u32; BATCH * SEQ];
    let mut mask = vec![0u32; BATCH * SEQ];
    for b in 0..BATCH {
        for s in 0..SEQ {
            ids[b * SEQ + s] = ((b * 977 + s * 131 + 7) % 200000) as u32 + 2;
            mask[b * SEQ + s] = if b == 0 || s < 16 { 1 } else { 0 };
        }
    }
    let mut ops = CpuOps::best();
    let mut host_t = npum_cpu::gemma::Timings::default();
    let host = model
        .embed_timed_with(&mut ops, &mut host_t, &ids, &mask, BATCH, SEQ, None, None)
        .expect("the host run");

    let root = std::env::var("NPU_ARTIFACTS").ok().map(PathBuf::from);
    let opts = args("gemm,softm", root.as_deref());
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut array = npuembeddings_cli::npu::open_array_text(
        &container,
        &path,
        &root_dir,
        &opts,
        npuembeddings_cli::npu::TextGeometry::gemma(
            model.spec.layers,
            model.spec.hidden,
            model.spec.ffn,
            model.spec.qkv_n,
        ),
    )
    .expect("the Gemma design opens");
    let mut eltwise = npuembeddings_cli::npu::open_eltwise(&container, &path, &root_dir, &opts)
        .expect("the softmax design opens");
    let array_ref: Option<&mut dyn npum_model::array::ArrayText> =
        Some(array.as_mut() as &mut dyn npum_model::array::ArrayText);
    let elt_ref: Option<&mut dyn npum_model::array::ArrayEltwise> =
        eltwise.as_deref_mut().map(|e| e as &mut dyn npum_model::array::ArrayEltwise);
    let mut array_t = npum_cpu::gemma::Timings::default();
    let got = model
        .embed_timed_with(
            &mut ops,
            &mut array_t,
            &ids,
            &mask,
            BATCH,
            SEQ,
            array_ref,
            elt_ref,
        )
        .expect("the array run");

    assert!(array.dispatches() > 0, "no GEMM dispatched");
    assert!(eltwise.as_ref().is_some_and(|e| e.dispatches() > 0), "no softmax dispatched");
    let hidden = model.spec.hidden;
    for b in 0..BATCH {
        let dot: f64 = host[b * hidden..(b + 1) * hidden]
            .iter()
            .zip(&got[b * hidden..(b + 1) * hidden])
            .map(|(x, y)| *x as f64 * *y as f64)
            .sum();
        let norm = |v: &[f32]| {
            v.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt()
        };
        let distance = 1.0 - dot
            / (norm(&host[b * hidden..(b + 1) * hidden])
                * norm(&got[b * hidden..(b + 1) * hidden]));
        eprintln!("row {b}: 1 - cos = {distance:.4e}");
        assert!(distance <= RTOL_COS, "row {b}: 1 - cos = {distance:.4e}");
    }
}
