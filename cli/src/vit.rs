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

/// The ViT's geometry, as the container states it.
pub struct Geometry {
    /// The container's name.
    pub model_name: String,
    /// The model's width.
    pub hidden: i64,
    /// How many layers.
    pub layers: i64,
    /// How many heads.
    pub heads: i64,
    /// One head's width.
    pub head_dim: i64,
    /// The input's side, in pixels.
    pub image_size: i64,
    /// The patch's side, in pixels.
    pub patch_size: i64,
    /// The positions, the patches and the CLS row.
    pub n_pos: i64,
    /// How many labels.
    pub num_labels: i64,
    /// The layer norm's epsilon.
    pub ln_eps: f64,
}

/// Read the geometry out of a container.
///
/// Every key is REQUIRED. A container that does not carry the geometry
/// is refused by the C++'s reader, and a default here would be a
/// classifier that answers about a model nobody chose.
pub fn read_geometry(container: &Container, path: &Path) -> Option<Geometry> {
    let config = container.config();
    let model_name = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    Some(Geometry {
        model_name,
        hidden: config.get_i64("hidden")?,
        layers: config.get_i64("num_layers")?,
        heads: config.get_i64("num_heads")?,
        head_dim: config.get_i64("head_dim")?,
        image_size: config.get_i64("image_size")?,
        patch_size: config.get_i64("patch_size")?,
        n_pos: config.get_i64("max_seq_len")?,
        num_labels: config.get_i64("num_labels")?,
        ln_eps: config.get_f64("layer_norm_eps").unwrap_or(1e-12),
    })
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
/// The rows are the C++'s, in the C++'s order, with the C++'s widths:
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
        softmax.push_str(
            ", at a MEASURED 0.806 s against 0.241 s on the --npu-ops gemm arm",
        );
    }
    row("softmax", "host", &softmax);
    row(
        "classifier head",
        "host",
        &format!(
            "{}x{} matvec: {} is not a multiple of tile_n*cols, so no legal B \
             panel of that width exists for this array",
            g.hidden, g.num_labels, g.num_labels
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

/// `%.6g`: the C++'s own rendering of the epsilon.
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
        let (mantissa, exp) = text.split_once('e').expect("Rust's exponential form");
        let exp: i32 = exp.parse().expect("Rust's exponential form");
        format!(
            "{}e{}{:02}",
            trim_zeros(mantissa),
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        )
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
