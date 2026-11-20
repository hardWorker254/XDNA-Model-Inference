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
use npum_xdna::array_gemms::GemmSpec;
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

/// The ViT's projections, as the array's loader needs them.
///
/// The four per-layer names and shapes are the ones the CPU pipeline dispatches
/// in [`npum_cpu::vit::Vit::encode_into`]. The patch embedding is a convolution
/// rewritten as an exact GEMM -- stride equals kernel, so im2col is a
/// permutation -- and is named separately because `--npu-ops conv`, not
/// `gemm`, is the flag that moves it.
pub struct Specs {
    pub layers: Vec<GemmSpec>,
    pub patch: GemmSpec,
}

pub fn gemm_specs(container: &Container, g: &Geometry) -> Result<Specs, String> {
    let d = g.hidden as usize;
    let ffn = container
        .directory()
        .tensor("layer.0.ffn_up")
        .and_then(|e| e.logical_shape.get(1).copied())
        .ok_or_else(|| "the container does not state layer.0.ffn_up's shape".to_string())?
        as usize;
    let patch = container
        .directory()
        .tensor("frontend.patch_embed")
        .ok_or_else(|| "the container holds no frontend.patch_embed panel".to_string())?;
    let patch_k = patch
        .logical_shape
        .first()
        .copied()
        .ok_or_else(|| "frontend.patch_embed has no reduction width".to_string())?
        as usize;
    let patch_n = patch
        .logical_shape
        .get(1)
        .copied()
        .ok_or_else(|| "frontend.patch_embed has no output width".to_string())?
        as usize;

    let mut layers = Vec::with_capacity(g.layers as usize * 4);
    for layer in 0..g.layers as usize {
        for (op, k, n) in [
            ("qkv", d, 3 * d),
            ("attn_out", d, d),
            ("ffn_up", d, ffn),
            ("ffn_down", ffn, d),
        ] {
            layers.push(GemmSpec {
                weight: format!("layer.{layer}.{op}"),
                op: op.to_string(),
                k,
                n,
            });
        }
    }
    Ok(Specs {
        layers,
        // The shipped ViT design has a square `attn_out` stream at the
        // model's hidden width, which is the patch embedding's shape too. A
        // model whose patch width differs has no stream here and is refused
        // by the loader instead of being run on the host under a false flag.
        patch: GemmSpec {
            weight: "frontend.patch_embed".to_string(),
            op: "attn_out".to_string(),
            k: patch_k,
            n: patch_n,
        },
    })
}

/// Where each requested operation runs, for the status block.
#[derive(Debug, Clone, Copy)]
pub struct Places {
    pub gemm: bool,
    pub conv: bool,
    pub attn: bool,
    pub layn: bool,
    pub gelu: bool,
    pub softm: bool,
}

impl Places {
    pub fn none() -> Self {
        Self {
            gemm: false,
            conv: false,
            attn: false,
            layn: false,
            gelu: false,
            softm: false,
        }
    }
}

pub fn places(
    container: &Container,
    path: &Path,
    root: &Path,
    args: &Args,
    design_present: bool,
) -> Places {
    let named = args.value("--artifacts");
    let elt = |set: &str| crate::npu::eltwise_design_present(container, path, root, named, set);
    Places {
        gemm: crate::npu::wants_op(args, "gemm") && design_present,
        conv: crate::npu::wants_op(args, "conv") && design_present,
        attn: crate::npu::wants_op(args, "attn") && design_present,
        layn: crate::npu::wants_op(args, "layn") && (elt("layernorm") || elt("rmsnorm")),
        gelu: crate::npu::wants_op(args, "gelu") && elt("gelu"),
        softm: crate::npu::wants_op(args, "softm") && elt("softmax"),
    }
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
pub fn print_block(g: &Geometry, designs: &Path, setup_s: f64, places: &Places) {
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
        if places.conv { "npu" } else { "host" },
        "the same attn_out shape, computed here; --npu-ops conv dispatches it, \
         and it is the only convolution this architecture has",
    );
    row(
        &format!("{} x qkv/attn_out/ffn", g.layers),
        if places.gemm { "npu" } else { "host" },
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
        if places.layn { "npu" } else { "host" },
        &format!(
            "fp32 with this container's layer_norm_eps {}; --npu-ops layn moves it",
            format_g(g.ln_eps, 6)
        ),
    );
    row(
        "gelu",
        if places.gelu { "npu" } else { "host" },
        "exact erf, fp32; --npu-ops gelu moves it",
    );
    row(
        "attention",
        if places.attn { "npu" } else { "host" },
        &format!(
            "{} heads x {} keys, no mask; --npu-ops attn uses attn_qk/attn_av \
             inside the same design set as the projections",
            g.heads, g.n_pos
        ),
    );
    let mut softmax = format!(
        "O(seq^2) on the host: {}x{} scores x {} heads in fp32; --npu-ops softm \
         moves it to softmax/, its own hw_context",
        g.n_pos, g.n_pos, g.heads
    );
    if g.n_pos == 197 {
        softmax.push_str(", at a MEASURED 0.806 s against 0.241 s on the --npu-ops gemm arm");
    }
    row(
        "softmax",
        if places.softm { "npu" } else { "host" },
        &softmax,
    );
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
pub fn find_designs(root: &Path, model_name: &str, named: Option<&str>) -> Option<PathBuf> {
    // The search is shared with every family, including `--artifacts`.
    crate::npu::find_artifacts(root, model_name, named)
        .filter(|dir| dir.join("gemm_rtp").join("design.json").is_file())
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
    #[rustfmt::skip]
    let outcome = crate::families::with_model_open(model, npum_families::Family::Cls, root, |path, arch, container| {
        if let Some(reason) = crate::backend::refuse_if_vulkan(args, npum_families::Family::Cls) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        // ONE GEMM design, plus one context per elementwise pass: check the
        // board BEFORE the listener or the work opens anything.
        if !crate::npu::check_array_budget(args, 1) {
            return crate::EXIT_REFUSED;
        }
        // The ViT's own block, on the stderr: the geometry, the design set,
        // and where every operation runs. It is printed BEFORE the
        // `--npu-ops` refusals so a refused command still says which container
        // and which design it was refused for.
        let started = std::time::Instant::now();
        let Some(geometry) = crate::vit::read_geometry(container, path) else {
            eprintln!(
                "error: {}: the container does not carry the ViT's geometry",
                path.display()
            );
            return crate::EXIT_REFUSED;
        };
        let designs =
            crate::vit::find_designs(root, &geometry.model_name, args.value("--artifacts"))
                .unwrap_or_else(|| {
                    root.join("runtime")
                        .join("artifacts")
                        .join(&geometry.model_name)
                });
        let present = crate::npu::has_own_design(container, "gemm_rtp")
            || designs.join("gemm_rtp").join("design.json").is_file();
        let places = crate::vit::places(container, path, root, args, present);
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
        crate::vit::print_block(&geometry, &designs, started.elapsed().as_secs_f64(), &places);

        if let Some(reason) =
            crate::families::refuse_op_not_in_arch(args, npum_families::Family::Cls)
        {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }
        // Six codes are real seams: the four layer GEMMs, the patch-embed
        // convolution (a GEMM on the array), the attention, and the three
        // elementwise passes. `mproj`, `fft` and `logit` are still refused by
        // name: the head cannot be tiled to a legal B width.
        if let Some(reason) = crate::families::refuse_array_not_ported(
            npum_families::Family::Cls,
            args,
            &["gemm", "attn", "layn", "gelu", "softm", "conv"],
        ) {
            eprintln!("error: {reason}");
            return crate::EXIT_REFUSED;
        }

        // THE ARRAY, OR THE HOST. The flag is the only thing that opens a
        // design; a design that happens to be on disk is not a dispatch.
        let want_array = crate::npu::wants_op(args, "gemm")
            || crate::npu::wants_op(args, "attn")
            || crate::npu::wants_op(args, "conv");
        let mut array: Option<Box<dyn npum_model::array::ArrayText>> = None;
        if want_array {
            let specs = match crate::vit::gemm_specs(container, &geometry) {
                Ok(s) => s,
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            };
            // ALL of this model's projections go into the ONE backend, so a
            // later flag cannot ask for a GEMM whose panel was never staged.
            // The FLAGS, not the backend's `gemm_on_array()`, decide which of
            // them are dispatched: staging the patch embedding for a
            // `conv`-only run must not silently take over the layer GEMMs.
            let mut chosen = specs.layers;
            chosen.push(specs.patch);
            array = match crate::npu::open_array_text_specs(
                container,
                path,
                root,
                args,
                chosen,
                "gemm_rtp",
            ) {
                Ok(a) => Some(a as Box<dyn npum_model::array::ArrayText>),
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            };
        }
        let mut eltwise = match crate::npu::open_eltwise(container, path, root, args) {
            Ok(e) => e.map(|e| e as Box<dyn npum_model::array::ArrayEltwise>),
            Err(reason) => {
                eprintln!("error: {reason}");
                return crate::EXIT_REFUSED;
            }
        };
        // The patch embedding is the convolution, and it is gated by `conv`
        // and not by `gemm`: asking only for `gemm` leaves the four layer
        // projections on the array but the patch embedding on the host.
        let conv_on = places.conv && array.is_some();
        let gemm_on = places.gemm && array.is_some();
        let _ = arch;

        let json = args.switch("--json");
        // `--top-k` is a switch in the flag table, so it has no value to
        // read; reading one made the documented runners-up never print.
        let top_k = args.switch("--top-k");
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
            let hidden = match vit.encode(
                &patches,
                array.as_deref_mut(),
                eltwise.as_deref_mut(),
                gemm_on,
                conv_on,
            ) {
                Ok(h) => h,
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            };
            let encoder_s = enc.elapsed().as_secs_f64();
            // The count is the array's own, not an estimate. A run whose flag
            // asked for array work but dispatched zero calls would be the lie
            // the `--npu-ops` flag exists to prevent.
            let dispatches = array
                .as_ref()
                .map(|a| a.dispatches() + a.attn_dispatches())
                .unwrap_or(0)
                + eltwise.as_ref().map(|e| e.dispatches()).unwrap_or(0);

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
                    "  {:<40} {:.2} s  {} dispatches",
                    image,
                    front_end_s + encoder_s + head_s,
                    dispatches
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
    });
    outcome
}
