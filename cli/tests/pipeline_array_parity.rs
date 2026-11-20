//! Device-gated parity для ViT и Whisper: те же сеймы, что использует CLI.
//!
//! Пропускается без платы и без контейнера с bf16-панелями. Переменные:
//! `NPUEMBEDDINGS_VIT_NPU`, `NPUEMBEDDINGS_WHISPER_NPU`, `NPU_ARTIFACTS`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use npuembeddings_cli::flags::Args;
use npuembeddings_cli::npu;
use npum_model::array::ArrayEltwise;

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

fn cos_distance(a: &[f32], b: &[f32]) -> f64 {
    let norm = |v: &[f32]| {
        v.iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt()
    };
    let dot = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum::<f64>();
    1.0 - dot / (norm(a) * norm(b))
}

#[test]
fn vit_array_path_matches_the_host_on_the_same_image() {
    if !board_present() {
        eprintln!("skipping: no NPU device node");
        return;
    }
    let Some(path) = container("NPUEMBEDDINGS_VIT_NPU") else {
        eprintln!("skipping: set NPUEMBEDDINGS_VIT_NPU to a ViT container packed with --npu");
        return;
    };
    let image = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/parity/fixtures/bus.jpg");
    if !image.is_file() {
        eprintln!("skipping: {} is not there", image.display());
        return;
    }
    let root = std::env::var("NPU_ARTIFACTS").ok().map(PathBuf::from);
    let container = npum_format::Container::open(&path).expect("the container opens");
    let model = npum_cpu::vit::Vit::open(&path).expect("the ViT opens");
    let geometry = model.geometry().clone();
    let decoded = npum_cpu::vit_image::decode_image(&image).expect("the image decodes");
    let patches = npum_cpu::vit_image::preprocess(&decoded, &geometry).expect("the patches");

    let host_cls = model
        .encode(&patches, None, None, false, false)
        .expect("the host run");
    let host = model.classify(&host_cls);

    let opts = args("gemm,attn,layn,gelu,softm,conv", root.as_deref());
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let specs = npuembeddings_cli::vit::gemm_specs(&container, &geometry)
        .expect("the ViT geometry names its streams");
    let mut chosen = specs.layers;
    chosen.push(specs.patch);
    let mut array =
        npu::open_array_text_specs(&container, &path, &root_dir, &opts, chosen, "gemm_rtp")
            .expect("the GEMM design opens");
    let mut elt =
        npu::open_eltwise(&container, &path, &root_dir, &opts).expect("the eltwise designs open");
    let borrowed = elt
        .as_deref_mut()
        .map(|e| e as &mut (dyn ArrayEltwise + '_));
    let cls = model
        .encode(&patches, Some(&mut *array), borrowed, true, true)
        .expect("the array run");
    let dispatches = array.dispatches()
        + array.attn_dispatches()
        + elt.as_ref().map(|e| e.dispatches()).unwrap_or(0);
    assert!(dispatches > 0, "the ViT array run dispatched nothing");
    let got = model.classify(&cls);
    let distance = cos_distance(&host, &got);
    eprintln!("ViT: measured 1 - cos = {distance:.4e} over {dispatches} dispatches");
    // Замерено на bus.jpg: 1 - cos = 3.06e-5 при 374 диспатчах.
    assert!(distance <= 1e-3, "ViT logits: 1 - cos = {distance:.4e}");
}

#[test]
fn whisper_array_path_dispatches_and_transcribes_the_same_audio() {
    if !board_present() {
        eprintln!("skipping: no NPU device node");
        return;
    }
    let Some(path) = container("NPUEMBEDDINGS_WHISPER_NPU") else {
        eprintln!(
            "skipping: set NPUEMBEDDINGS_WHISPER_NPU to a Whisper container packed with --npu"
        );
        return;
    };
    let audio = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/parity/fixtures/tone.wav");
    if !audio.is_file() {
        eprintln!("skipping: {} is not there", audio.display());
        return;
    }
    let root = std::env::var("NPU_ARTIFACTS").ok().map(PathBuf::from);
    let container = npum_format::Container::open(&path).expect("the container opens");
    let samples = npum_cpu::whisper_features::read_wav(&audio)
        .expect("the WAV decodes")
        .samples;

    let encoder = npum_cpu::whisper::Encoder::open(&path).expect("the encoder opens");
    let mut host_decoder = npum_cpu::whisper::Decoder::open(&path).expect("the host decoder opens");
    let mut array_decoder =
        npum_cpu::whisper::Decoder::open(&path).expect("the array decoder opens");
    let table = container
        .tensor_bytes("tokenizer.whisper_table")
        .expect("the token table");
    let tokenizer =
        npum_cpu::whisper_tokenizer::Tokenizer::from_table(table).expect("the tokenizer");
    let f32_tensor = |name: &str| -> Vec<f32> {
        container
            .tensor_bytes(name)
            .expect("the front end weight")
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let weights = npum_cpu::whisper_session::FrontEndWeights {
        conv1_weight: f32_tensor("frontend.conv1.weight"),
        conv1_bias: f32_tensor("frontend.conv1.bias"),
        conv2_weight: f32_tensor("frontend.conv2.weight"),
        conv2_bias: f32_tensor("frontend.conv2.bias"),
    };
    let options = npum_cpu::whisper_session::Options::default();

    let host = npum_cpu::whisper_session::transcribe_samples(
        &encoder,
        &mut host_decoder,
        &tokenizer,
        &samples,
        &options,
        &weights,
        None,
        None,
        None,
    )
    .expect("the host run");

    // Both sets: the encoder's `gemm_rtp` and the decoder's `gemm_rtp_dec`.
    // Only `gemm,attn` is asked for, so no extra eltwise hw contexts are
    // opened -- LayerNorm/GELU/softmax stay on the host in this run.
    let opts = args("gemm,attn", root.as_deref());
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let geometry = npuembeddings_cli::stt::read_geometry(&container, &path).expect("the geometry");
    let mut enc = npu::open_array_text_specs(
        &container,
        &path,
        &root_dir,
        &opts,
        npuembeddings_cli::stt::encoder_specs(&geometry),
        "gemm_rtp",
    )
    .expect("the encoder design opens");
    let mut dec = npu::open_array_text_specs(
        &container,
        &path,
        &root_dir,
        &opts,
        npuembeddings_cli::stt::decoder_specs(&geometry),
        "gemm_rtp_dec",
    )
    .expect("the decoder design opens");
    let got = npum_cpu::whisper_session::transcribe_samples(
        &encoder,
        &mut array_decoder,
        &tokenizer,
        &samples,
        &options,
        &weights,
        Some(&mut *enc),
        Some(&mut *dec),
        None,
    )
    .expect("the array run");
    // `Transcript::n_dispatches` is still a placeholder (always zero), so the
    // proof of dispatch comes from the handles themselves, read after the run.
    let dispatches =
        enc.dispatches() + enc.attn_dispatches() + dec.dispatches() + dec.attn_dispatches();
    assert!(dispatches > 0, "the Whisper array run dispatched nothing");
    assert!(!got.text.is_empty(), "the array run produced no text");
    // Текст дискретен: bf16 может заменить одно слово. Проверяем, что прогон
    // остался тем же по числу окон; полный численный паритет — за платой.
    assert_eq!(host.n_chunks, got.n_chunks, "different number of windows");

    // The front end's two matrix passes ALONE: dft400/mel_proj are streams
    // inside the encoder set, generated and staged by the backend. With only
    // `fft,mproj` asked for, the encoder's GEMMs and the decoder stay on the
    // host, so any difference in the transcript is the front end's.
    let fe_opts = args("fft,mproj", root.as_deref());
    let mut fe_enc = npu::open_array_text_specs(
        &container,
        &path,
        &root_dir,
        &fe_opts,
        Vec::new(),
        "gemm_rtp",
    )
    .expect("the front-end design opens without GEMM panels");
    assert!(fe_enc.dft_on_array() && fe_enc.mproj_on_array());
    let mut fe_decoder =
        npum_cpu::whisper::Decoder::open(&path).expect("the front-end decoder opens");
    let fe = npum_cpu::whisper_session::transcribe_samples(
        &encoder,
        &mut fe_decoder,
        &tokenizer,
        &samples,
        &options,
        &weights,
        Some(&mut *fe_enc),
        None,
        None,
    )
    .expect("the front-end array run");
    assert!(fe_enc.frontend_dispatches() > 0, "no front-end dispatch");
    assert_eq!(fe.n_chunks, host.n_chunks);
    assert_eq!(fe.text.trim(), host.text.trim(), "the array front end changed the transcript");
}

/// The decoder's tied logit head, alone, against the host's dot-product tree.
///
/// `--npu-ops logit` stages eight [512, 6656] panels from `embed_tokens` and
/// dispatches one per vocabulary chunk. The host's `logits_from` is the
/// oracle; the array uses bf16 panels, so compare with a bf16-sized tolerance
/// AND check that argmax -- the only thing the greedy decoder consumes --
/// survives it.
#[test]
fn whisper_logit_streams_match_the_host() {
    use npum_model::array::ArrayLogits;

    if !board_present() {
        eprintln!("skipping: no NPU device node");
        return;
    }
    let Some(path) = container("NPUEMBEDDINGS_WHISPER_NPU") else {
        eprintln!("skipping: set NPUEMBEDDINGS_WHISPER_NPU");
        return;
    };
    let file = npum_format::Container::open(&path).expect("the container opens");
    let d = file.config().get_i64("d_model").expect("d_model") as usize;
    let vocab = file.config().get_i64("vocab_size").expect("vocab_size") as usize;
    let bytes = file.tensor_bytes("decoder.embed_tokens").expect("decoder.embed_tokens");
    let embed: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let h: Vec<f32> = (0..d).map(|i| ((i as f64 * 0.017).sin() * 0.25) as f32).collect();
    let want = npum_cpu::whisper::logits_from(&h, &embed, d, vocab);

    let root = std::env::var("NPU_ARTIFACTS").ok().map(PathBuf::from);
    let opts = args("logit", root.as_deref());
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut dec = npu::open_array_text_specs(&file, &path, &root_dir, &opts, Vec::new(), "gemm_rtp_dec")
        .expect("the decoder's logit design opens");
    assert!(ArrayLogits::logit_on_array(dec.as_ref()));
    let mut got = vec![0.0f32; vocab];
    ArrayLogits::logits(dec.as_mut(), &h, &mut got).expect("the array's logits");
    assert_eq!(ArrayLogits::logit_dispatches(dec.as_ref()), 8);

    let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    assert_eq!(argmax(&want), argmax(&got), "the greedy decoder would choose another token");
    let dist = cos_distance(&want, &got);
    eprintln!("logits: 1 - cos = {dist:.4e}, argmax {} / {}", argmax(&want), argmax(&got));
    assert!(dist <= 2e-2, "1 - cos = {dist:.4e}");
}

/// The classifier head reuses the set's `qkv` stream with a zero-padded B
/// panel. Compare its logits to the host's double-accumulation matvec on the
/// SAME CLS row, so any difference is the head alone.
#[test]
fn vit_head_reuses_the_qkv_stream() {
    use npum_model::array::ArrayHead;

    if !board_present() {
        eprintln!("skipping: no NPU device node");
        return;
    }
    let Some(path) = container("NPUEMBEDDINGS_VIT_NPU") else {
        eprintln!("skipping: set NPUEMBEDDINGS_VIT_NPU");
        return;
    };
    let image = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/parity/fixtures/bus.jpg");
    if !image.is_file() {
        return;
    }
    let root = std::env::var("NPU_ARTIFACTS").ok().map(PathBuf::from);
    let file = npum_format::Container::open(&path).expect("the container opens");
    let model = npum_cpu::vit::Vit::open(&path).expect("the ViT opens");
    let geometry = model.geometry().clone();
    let decoded = npum_cpu::vit_image::decode_image(&image).expect("the image decodes");
    let patches = npum_cpu::vit_image::preprocess(&decoded, &geometry).expect("the patches");
    let cls = model.encode(&patches, None, None, false, false).expect("the host encoder");
    let nlab = geometry.num_labels as usize;
    let want = model.classify(&cls[..geometry.hidden as usize]);

    let opts = args("logit", root.as_deref());
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let specs = npuembeddings_cli::vit::gemm_specs(&file, &geometry).expect("the ViT specs");
    let mut chosen = specs.layers;
    chosen.push(specs.patch);
    let mut array =
        npu::open_array_text_specs(&file, &path, &root_dir, &opts, chosen, "gemm_rtp")
            .expect("the head design opens");
    assert!(ArrayHead::head_on_array(array.as_ref()));
    let mut got = vec![0.0f32; nlab];
    model
        .classify_into_with(
            &cls[..geometry.hidden as usize],
            &mut got,
            Some(&mut *array),
            true,
        )
        .expect("the array head");
    assert_eq!(ArrayHead::head_dispatches(array.as_ref()), 1);
    let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    assert_eq!(argmax(&want), argmax(&got), "the head chose another label");
    let distance = cos_distance(&want, &got);
    eprintln!("ViT head: 1 - cos = {distance:.4e}, argmax {} / {}", argmax(&want), argmax(&got));
    assert!(distance <= 1e-3, "1 - cos = {distance:.4e}");
}

/// Whisper's conv1/conv2 on the encoder set's own streams, against the
/// original fp32 convolution kernels on the SAME mel spectrogram.
#[test]
fn whisper_conv_streams_match_the_host() {
    use npum_model::array::ArrayAudioConv;

    if !board_present() {
        return;
    }
    let Some(path) = container("NPUEMBEDDINGS_WHISPER_NPU") else {
        return;
    };
    let audio = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/parity/fixtures/tone.wav");
    let samples = npum_cpu::whisper_features::read_wav(&audio).expect("the WAV decodes").samples;
    let file = npum_format::Container::open(&path).expect("the container opens");
    let f32_tensor = |name: &str| -> Vec<f32> {
        file.tensor_bytes(name)
            .expect("the front end weight")
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let (mel, _) = npum_cpu::whisper_features::log_mel_30s_with_frontend(&samples, 80, None)
        .expect("the host mel");
    let hidden = 512usize;
    let w1 = f32_tensor("frontend.conv1.weight");
    let b1 = f32_tensor("frontend.conv1.bias");
    let w2 = f32_tensor("frontend.conv2.weight");
    let b2 = f32_tensor("frontend.conv2.bias");
    let want = npum_cpu::whisper_features::conv_front_end(&mel, &w1, &b1, &w2, &b2, hidden);

    let root = std::env::var("NPU_ARTIFACTS").ok().map(PathBuf::from);
    let opts = args("conv", root.as_deref());
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let geometry = npuembeddings_cli::stt::read_geometry(&file, &path).expect("the geometry");
    let mut array = npu::open_array_text_specs(
        &file,
        &path,
        &root_dir,
        &opts,
        npuembeddings_cli::stt::encoder_specs(&geometry),
        "gemm_rtp",
    )
    .expect("the conv design opens");
    assert!(ArrayAudioConv::conv_on_array(array.as_ref()));
    let got = npum_cpu::whisper_features::conv_front_end_with(
        &mel, &w1, &b1, &w2, &b2, hidden, Some(&mut *array), true,
    );
    assert!(ArrayAudioConv::conv_dispatches(array.as_ref()) > 0);
    assert_eq!(want.len(), got.len());
    let num = want.iter().zip(&got).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum::<f64>();
    let den = want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
    let rel = (num / den).sqrt();
    eprintln!("conv: relative error = {rel:.4e}, {} dispatches", ArrayAudioConv::conv_dispatches(array.as_ref()));
    assert!(rel <= 2e-2, "relative error = {rel:.4e}");
}
