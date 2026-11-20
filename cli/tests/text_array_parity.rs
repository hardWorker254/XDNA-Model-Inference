//! Device-gated parity for the text family's array path (BERT and GTE).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use npum_cpu::bert::BertModel;
use npum_cpu::gte::GteModel;
use npum_model::array::ArrayEltwise;
use npum_model::array::ArrayTextRef;

use npuembeddings_cli::flags::Args;
use npuembeddings_cli::npu;

fn board_present() -> bool {
    Path::new("/dev/accel/accel0").exists() || Path::new("/dev/dri/accel0").exists()
}

fn container(env: &str) -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var(env).ok()?);
    if p.is_file() {
        Some(p)
    } else {
        None
    }
}

/// Замерено на этой плате: bge-base 1.05e-3 (1237 диспатчей), gte-multilingual
/// 2.27e-3 (1225 диспатчей). Двухкратный запас покрывает разброс по батчам.
const RTOL_COS: f32 = 5e-3;

fn inputs(vocab: usize, seq: usize) -> (Vec<u32>, Vec<u32>) {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut ids = Vec::with_capacity(seq);
    let mut mask = Vec::with_capacity(seq);
    for _ in 0..seq {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ids.push((state as usize % vocab.max(1)) as u32);
        mask.push(1);
    }
    (ids, mask)
}

fn args(root: Option<&Path>) -> Args {
    args_with_codes("gemm,attn,layn,gelu,softm", root)
}

fn args_with_codes(codes: &str, root: Option<&Path>) -> Args {
    let mut args = Args::default();
    let mut flags = HashMap::new();
    flags.insert("--npu-ops".to_string(), codes.to_string());
    if let Some(root) = root {
        flags.insert("--artifacts".to_string(), root.display().to_string());
    }
    args.flags = flags;
    args
}

fn compare(name: &str, host: &[f32], array: &[f32], dispatches: usize) {
    assert_eq!(host.len(), array.len(), "{name}: different output sizes");
    assert!(dispatches > 0, "{name}: the array run dispatched nothing");
    let norm = |v: &[f32]| {
        let s: f64 = v.iter().map(|x| (*x as f64) * (*x as f64)).sum();
        s.sqrt()
    };
    let (hn, an) = (norm(host), norm(array));
    assert!(hn > 0.0 && an > 0.0, "{name}: a run produced all zeros");
    let dot: f64 = host
        .iter()
        .zip(array.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let distance = 1.0 - dot / (hn * an);
    eprintln!("{name}: measured 1 - cos = {distance:.4e} over {dispatches} dispatches");
    assert!(
        distance <= RTOL_COS as f64,
        "{name}: 1 - cos = {distance:.4e}, above {RTOL_COS:e} (dispatches = {dispatches})"
    );
}

fn mut_array_run<R, F>(
    path: &Path,
    container: &npum_format::Container,
    opts: &Args,
    geometry: npu::TextGeometry,
    run: F,
) -> (R, usize)
where
    F: for<'a, 'b> FnOnce(ArrayTextRef<'a, 'b>, Option<&'a mut (dyn ArrayEltwise + 'b)>) -> R,
{
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut array = npu::open_array_text(container, path, &root, opts, geometry)
        .expect("the text design opens");
    let mut eltwise =
        npu::open_eltwise(container, path, &root, opts).expect("the eltwise designs open");
    let borrowed = eltwise
        .as_deref_mut()
        .map(|e| e as &mut (dyn ArrayEltwise + '_));
    let out = run(Some(&mut *array), borrowed);
    // ПОСЛЕ прогона: счётчик говорит о том, что действительно считалось.
    let dispatches = array.dispatches()
        + array.attn_dispatches()
        + eltwise.as_ref().map(|e| e.dispatches()).unwrap_or(0);
    (out, dispatches)
}

#[test]
fn bert_text_array_path_matches_the_host() {
    if !board_present() {
        eprintln!("skipping: no NPU device node");
        return;
    }
    let Some(path) = container("NPUEMBEDDINGS_BERT_NPU") else {
        eprintln!("skipping: set NPUEMBEDDINGS_BERT_NPU to a container packed with --npu");
        return;
    };
    let root = std::env::var("NPU_ARTIFACTS").ok().map(PathBuf::from);
    let container = npum_format::Container::open(&path).expect("the container opens");
    let model = BertModel::from_container(&container).expect("the BERT model opens");
    let (ids, mask) = inputs(model.spec.vocab, model.spec.max_seq_len.min(64));
    let seq = ids.len();
    let mut ops = npum_cpu::ops::CpuOps::best();

    let host = model
        .embed_timed_with(
            &mut ops,
            &mut npum_cpu::bert::Timings::default(),
            &ids,
            &mask,
            1,
            seq,
            None,
            None,
        )
        .expect("the host run");

    let opts = args(root.as_deref());
    let (got, dispatches) = mut_array_run(
        path.as_path(),
        &container,
        &opts,
        npu::TextGeometry::bert(model.spec.layers, model.spec.hidden, model.spec.ffn),
        |array, eltwise| {
            model
                .embed_timed_with(
                    &mut ops,
                    &mut npum_cpu::bert::Timings::default(),
                    &ids,
                    &mask,
                    1,
                    seq,
                    array,
                    eltwise,
                )
                .expect("the array run")
        },
    );
    compare("BERT", &host, &got, dispatches);
}

#[test]
fn gte_text_array_path_matches_the_host() {
    if !board_present() {
        eprintln!("skipping: no NPU device node");
        return;
    }
    let Some(path) = container("NPUEMBEDDINGS_GTE_NPU") else {
        eprintln!("skipping: set NPUEMBEDDINGS_GTE_NPU to a GTE container packed with --npu");
        return;
    };
    let root = std::env::var("NPU_ARTIFACTS").ok().map(PathBuf::from);
    let container = npum_format::Container::open(&path).expect("the container opens");
    let model = GteModel::from_container(&container).expect("the GTE model opens");
    let (ids, mask) = inputs(model.spec.vocab, model.spec.max_seq_len.min(64));
    let seq = ids.len();
    let mut ops = npum_cpu::ops::CpuOps::best();

    let host = model
        .embed_timed_with(
            &mut ops,
            &mut npum_cpu::gte::Timings::default(),
            &ids,
            &mask,
            1,
            seq,
            None,
            None,
        )
        .expect("the host run");

    // GTE's FFN is gated: there is no standalone gelu/ design for it.
    let opts = args_with_codes("gemm,attn,layn,softm", root.as_deref());
    let (got, dispatches) = mut_array_run(
        path.as_path(),
        &container,
        &opts,
        npu::TextGeometry::gte(model.spec.layers, model.spec.hidden, model.spec.ffn),
        |array, eltwise| {
            model
                .embed_timed_with(
                    &mut ops,
                    &mut npum_cpu::gte::Timings::default(),
                    &ids,
                    &mask,
                    1,
                    seq,
                    array,
                    eltwise,
                )
                .expect("the array run")
        },
    );
    compare("GTE", &host, &got, dispatches);
}
