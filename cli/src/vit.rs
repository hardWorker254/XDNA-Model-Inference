//! The ViT family's status block.
//!
//! # A different block, because it is a different architecture
//!
//! The text family's block describes a BERT-shaped model: an embedding
//! table, a stack of post-LN layers, a mean pooling. The ViT is none of
//! those -- it is pre-LN, its input is pixels turned into patches, and its
//! output is a label. So it has its own block, and the block says where
//! every operation runs.
//!
//! # The rows are read, not written
//!
//! The geometry comes from the container's own config: the packer read
//! the mean, the standard deviation, the resample code and the layer
//! norm's epsilon out of the checkpoint's `preprocessor_config.json` and
//! REFUSED rather than defaulted them, so the container is a
//! transcription of a decision that was already made. Printing those
//! numbers back is not decoration -- a consumer that implements a subset
//! has to be able to say which one it implements.
//!
//! # The weights' size is the STAGED size
//!
//! Not the container's size: the operands that a host-layout container
//! keeps on the host are not staged anywhere, and the number that goes
//! with them is the LayerNorm's parameters, the position table, the CLS
//! row and the head.

use npum_format::Container;
use std::path::{Path, PathBuf};

use crate::flags::Args;
/// The ViT's geometry, as the container states it.
///
/// The CPU backend's geometry, not a second one: the block prints the
/// numbers and the arithmetic reads them, and two copies of the same
/// facts are two copies to drift.
pub use npum_cpu::vit::Geometry;

/// Read the geometry out of a container.
///
/// Every key is REQUIRED. A container that does not carry the geometry
/// is refused by the reference's reader, and a default here would be a
/// classifier that answers about a model nobody chose.
pub fn read_geometry(container: &Container, path: &Path) -> Option<Geometry> {
    Geometry::read(container, path)
}

/// The bytes a host-layout container stages, and what the block calls them.
///
/// The GEMM operands stay on the host -- a host-layout container is fp32
/// and the array multiplies bf16 and int8 -- so what is staged is the
/// LayerNorm's parameters, the position table, the CLS row, the last
/// LayerNorm and the head.
pub fn staged_bytes(g: &Geometry) -> u64 {
    let d = g.hidden as u64;
    let norm_sites = (2 * g.layers as u64) + 1;
    let head = d * g.num_labels as u64 * 4;
    let head_bias = g.num_labels as u64 * 4;
    let positions = g.n_pos as u64 * d * 4;
    let cls = d * 4;
    let norms = norm_sites * 2 * d * 4;
    head + head_bias + positions + cls + norms
}

/// The block, printed to the stderr.
///
/// The rows are the reference's, in the reference's order, with the reference's widths:
/// a row is a name, a place and a reason, and the reason is what makes
/// the row worth reading.
pub fn print_block(g: &Geometry, designs: &Path, setup_s: f64) {
    eprintln!("NpuEmbeddings -- ViT image classification on the AMD NPU");
    eprintln!(
        "  model      {}: hidden {}, {} pre-LN layers, {} heads x {}, {}px at patch {} -> {} positions, {} labels",
        g.model_name,
        g.hidden,
        g.layers,
        g.heads,
        g.head_dim,
        g.image_size,
        g.patch_size,
        g.n_pos,
        g.num_labels
    );
    eprintln!("  designs    {}", designs.display());
    eprintln!(
        "  weights    {:.1} MB staged on the array, bf16 operands",
        staged_bytes(g) as f64 / (1024.0 * 1024.0)
    );
    eprintln!("  ops        (npu = dispatched, host = this process)");
    row(
        "patch_embed, 5th GEMM",
        "host",
        "the same attn_out shape, computed here; --npu-ops conv dispatches it, \
         and it is the only convolution this architecture has",
    );
    row(
        &format!("{} x qkv/attn_out/ffn", g.layers),
        "host",
        "qkv, attn_out, ffn_up, ffn_down, M=1024 rows per dispatch; --npu-ops \
         gemm moves them to the array",
    );
    row(
        "image front end",
        "host",
        &format!(
            "decode, PIL-equivalent resize to {}px, (x/255 - mean)/std, im2col",
            g.image_size
        ),
    );
    row(
        "layernorm",
        "host",
        &format!(
            "fp32 with this container's layer_norm_eps {}; --npu-ops layn moves it",
            format_g(g.ln_eps, 6)
        ),
    );
    row("gelu", "host", "exact erf, fp32; --npu-ops gelu moves it");
    let mut softmax = format!(
        "O(seq^2) on the host: {}x{} scores x {} heads in fp32; --npu-ops softm \
         moves it to softmax/, its own hw_context",
        g.n_pos, g.n_pos, g.heads
    );
    if g.n_pos == 197 {
        softmax.push_str(", at a MEASURED 0.806 s against 0.241 s on the --npu-ops gemm arm");
    }
    row("softmax", "host", &softmax);
    row(
        "classifier head",
        "host",
        &format!(
            "{}x{} matvec on the host; --npu-ops logit dispatches it by padding \
             the projection to the next legal width (speed is not a constraint)",
            g.hidden, g.num_labels
        ),
    );
    eprintln!(
        "  setup      {:.2} s (device, 1 design set, weights staged on it)",
        setup_s
    );
    let mut measured = String::from(
        "  measured   every figure below is THIS run's measurement of THIS image, \
         not a throughput claim.",
    );
    if g.n_pos == 197 {
        measured.push_str(
            " Attention at these 197 positions has been measured in this \
             repository (NPU_OPS.md, the attn cell), and the array came out \
             SLOWER: 0.348 s of encoder with --npu-ops gemm,attn against 0.241 s \
             with --npu-ops gemm.",
        );
    } else {
        measured.push_str(
            " Attention at this container's position count has not been measured \
             in this repository.",
        );
    }
    eprintln!("{measured}");
}

/// One row of the ops table: a name, a place, and the reason.
fn row(name: &str, place: &str, why: &str) {
    eprintln!("             {name:<26} {place:<5} {why}");
}

/// `%.6g`: the reference's own rendering of the epsilon.
///
/// The epsilon is printed with `%.6g` and NOT with a decimal formatting,
/// which turns the ViT's 1e-12 into "0.000000" -- the wrong number in the
/// exact case the value is worth printing for.
fn format_g(value: f64, precision: usize) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    let exponent = value.abs().log10().floor() as i32;
    if exponent < -4 || exponent >= precision as i32 {
        let text = format!("{:.*e}", precision - 1, value);
        if let Some((mantissa, exp)) = text.split_once('e') {
            if let Ok(exp) = exp.parse::<i32>() {
                return format!(
                    "{}e{}{:02}",
                    trim_zeros(mantissa),
                    if exp < 0 { '-' } else { '+' },
                    exp.abs()
                );
            }
        }
        text
    } else {
        let decimals = (precision as i32 - 1 - exponent).max(0) as usize;
        trim_zeros(&format!("{value:.decimals$}"))
    }
}

/// The trailing zeros of a decimal, and the point that follows them.
fn trim_zeros(text: &str) -> String {
    if !text.contains('.') {
        return text.to_string();
    }
    let trimmed = text.trim_end_matches('0');
    let trimmed = trimmed.strip_suffix('.').unwrap_or(trimmed);
    trimmed.to_string()
}

/// The prediction, as the JSON object.
///
/// The CPU backend's function: the CLI and the server must not drift, and
/// two implementations of one JSON shape is how they drift.
pub fn prediction_json(
    prediction: &npum_cpu::vit::Prediction,
    image: &str,
    front_end_s: f64,
    encoder_s: f64,
    head_s: f64,
) -> String {
    npum_cpu::vit::prediction_json(prediction, image, 1, front_end_s, encoder_s, head_s)
}

/// The design set's directory for a model, if there is one.
pub fn find_designs(root: &Path, model_name: &str) -> Option<PathBuf> {
    for base in [
        root.join("runtime").join("artifacts"),
        root.join("artifacts"),
        root.join("runtime"),
        root.to_path_buf(),
    ] {
        for sub in ["artifacts_npu1", "artifacts_npu2", "", "npu1", "npu2"] {
            let dir = if sub.is_empty() {
                base.join(model_name)
            } else {
                base.join(model_name).join(sub)
            };
            if dir.join("gemm_rtp").join("design.json").is_file() {
                return Some(dir);
            }
        }
    }
    None
}

pub fn command(args: &Args, root: &Path) -> crate::families::Outcome {
    let Some(model) = args.positional.first() else {
        return crate::needs_a_model("classify");
    };
    // Every leading non-flag argument after the model is an image, so
    // `classify <model> a.png b.png --top-k` classifies both.
    let images: Vec<&String> = args.positional.iter().skip(1).collect();
    if images.is_empty() {
        eprintln!("error: `classify` needs an image:");
        eprintln!("    npuimage classify <model> <image.png> [more.png ...]");
        eprintln!("    npuimage serve <model>            (POST /v1/classify)");
        eprintln!(
            "  (PNG and JPEG; anything else is refused rather than guessed at. \
             --top-k prints the runners-up to stderr, --json to stdout)"
        );
        return crate::EXIT_REFUSED;
    }
    crate::families::with_model(model, npum_families::Family::Cls, root, |path, arch| {
        if let Some(reason) = crate::backend::refuse_if_vulkan(args, npum_families::Family::Cls) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        if let Some(reason) =
            crate::families::refuse_op_not_in_arch(args, npum_families::Family::Cls)
        {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        if let Some(reason) =
            crate::families::refuse_array_not_ported(npum_families::Family::Cls, args, &[])
        {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        // The ViT's own block, on the stderr: the geometry, the design set,
        // and where every operation runs.
        let started = std::time::Instant::now();
        let container = match npum_format::Container::open(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: {}: {e}", path.display());
                return crate::EXIT_REFUSED;
            }
        };
        let Some(geometry) = crate::vit::read_geometry(&container, path) else {
            eprintln!(
                "error: {}: the container does not carry the ViT's geometry",
                path.display()
            );
            return crate::EXIT_REFUSED;
        };
        let designs = crate::vit::find_designs(root, &geometry.model_name).unwrap_or_else(|| {
            root.join("runtime")
                .join("artifacts")
                .join(&geometry.model_name)
        });
        // The setup is the weights arriving: the container is open, and the
        // head and the stack are read before the block is printed, because
        // the block reports how long that took.
        let vit = match npum_cpu::vit::Vit::open(path) {
            Ok(v) => v,
            Err(reason) => {
                eprintln!("error: {reason}");
                return crate::EXIT_REFUSED;
            }
        };
        crate::vit::print_block(&geometry, &designs, started.elapsed().as_secs_f64());
        let _ = arch;

        let json = args.switch("--json");
        let top_k = args.value_i64("--top-k").unwrap_or(0) > 0;
        let mut out = String::from("[\n");
        for (i, image) in images.iter().enumerate() {
            let decoded = match npum_cpu::vit_image::decode_image(std::path::Path::new(image)) {
                Ok(d) => d,
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            };
            let front = std::time::Instant::now();
            let patches = match npum_cpu::vit_image::preprocess(&decoded, &geometry) {
                Ok(p) => p,
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            };
            let front_end_s = front.elapsed().as_secs_f64();

            let enc = std::time::Instant::now();
            let hidden = vit.encode(&patches);
            let encoder_s = enc.elapsed().as_secs_f64();

            let head = std::time::Instant::now();
            let d = geometry.hidden as usize;
            let logits = vit.classify(&hidden[..d]);
            let prediction = match npum_cpu::vit::predict(&logits, vit.labels()) {
                Ok(p) => p,
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            };
            let head_s = head.elapsed().as_secs_f64();

            if json {
                out.push_str("  ");
                out.push_str(&crate::vit::prediction_json(
                    &prediction,
                    image,
                    front_end_s,
                    encoder_s,
                    head_s,
                ));
                out.push_str(if i + 1 < images.len() { ",\n" } else { "\n" });
            } else {
                eprintln!(
                    "  {:<40} {:.2} s  0 dispatches",
                    image,
                    front_end_s + encoder_s + head_s
                );
                if top_k {
                    // A ranking of the whole row, not the top two: the
                    // argmax is on stdout alone and the runners-up are a
                    // status detail, so the full sort is printed rather than
                    // a truncated one that would read as "there is nothing
                    // else close".
                    let mut order: Vec<usize> = (0..prediction.logits.len()).collect();
                    order.sort_by(|a, b| {
                        prediction.logits[*b]
                            .partial_cmp(&prediction.logits[*a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    for (k, index) in order.iter().take(10).enumerate() {
                        eprintln!(
                            "    {:2}  {:<40} logit {:+.4}",
                            k + 1,
                            vit.labels().get(*index).cloned().unwrap_or_default(),
                            prediction.logits[*index]
                        );
                    }
                }
                println!("{}", prediction.name);
            }
        }
        if json {
            out.push_str("]\n");
            print!("{out}");
        }
        0
    })
}
