//! The ViT: the patch embedding, the pre-LN encoder and the classifier
//! head.
//!
//! # WHAT IS DIFFERENT FROM BERT, AND FROM WHISPER
//!
//! * the INPUT is pixels turned into patches, not tokens: the front end
//!   produces a `[n_patches, patch_dim]` matrix and the first GEMM is the
//!   patch embedding;
//! * the ORDER is pre-LN -- residual, LayerNorm, attention, add -- like
//!   Whisper's and not like BERT's;
//! * the LayerNorm is the DOUBLE-accumulating one (the same function
//!   Whisper's encoder uses), and the epsilon is the CONTAINER's;
//! * the attention has NO MASK: a ViT attends over every position it was
//!   given, and every position is a real one;
//! * the attention's scale is FOLDED into Q's weight and Q's bias, so
//!   applying it again would be applying it twice -- which is a different
//!   model;
//! * the output is a LABEL and a probability, not a vector.
//!
//! # THE FINAL LAYERNORM IS NOT A DETAIL
//!
//! transformers' ViTModel applies a LayerNorm after the encoder, and that
//! normalised CLS row is what the head consumes. Skipping it changes the
//! logits by a scale, which changes which label wins, without changing any
//! shape -- so it is not a detail.

use crate::kernels;
use crate::parallel;
use crate::vit_image::Resample;
use crate::whisper::attention_into;
use crate::whisper::{gelu_erf_rows, layernorm_rows, layernorm_rows_inplace};
use npum_format::Container;
use npum_model::array::{ArrayEltwise, ArrayText, ArrayTextRef, AttentionShape, EltOp, };
use std::path::Path;

/// The ViT's geometry, as the container states it.
#[derive(Debug, Clone)]
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
    /// How many patches there are.
    pub n_patches: i64,
    /// The patch's width: the channels times the patch's area.
    pub patch_dim: i64,
    /// How many labels.
    pub num_labels: i64,
    /// The layer norm's epsilon.
    pub ln_eps: f64,
    /// The attention's scale, when it is not folded.
    pub attn_scale: f64,
    /// Whether the scale is already folded into Q's weight and bias.
    pub qkv_scale_folded: bool,
    /// The per-channel means.
    pub mean: Vec<f32>,
    /// The per-channel standard deviations.
    pub std_dev: Vec<f32>,
    /// The resample code.
    pub resample: Resample,
    /// How many channels.
    pub num_channels: i64,
}

impl Geometry {
    /// Read the geometry out of a container.
    ///
    /// Every key is REQUIRED except the ones with a measured default: a
    /// container that does not carry the geometry is refused by the reference's
    /// reader, and a default here would be a classifier that answers about
    /// a model nobody chose.
    pub fn read(container: &Container, path: &Path) -> Option<Geometry> {
        let config = container.config();
        let model_name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let mean = config
            .get("image_mean")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_f64()).collect::<Vec<f64>>())?;
        let std_dev = config
            .get("image_std")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_f64()).collect::<Vec<f64>>())?;
        let resample = Resample::from_code(config.get_i64("resample")?)?;
        Some(Geometry {
            model_name,
            hidden: config.get_i64("hidden")?,
            layers: config.get_i64("num_layers")?,
            heads: config.get_i64("num_heads")?,
            head_dim: config.get_i64("head_dim")?,
            image_size: config.get_i64("image_size")?,
            patch_size: config.get_i64("patch_size")?,
            n_pos: config.get_i64("max_seq_len")?,
            n_patches: config.get_i64("n_patches")?,
            patch_dim: config.get_i64("patch_dim")?,
            num_labels: config.get_i64("num_labels")?,
            ln_eps: config.get_f64("layer_norm_eps").unwrap_or(1e-12),
            attn_scale: config.get_f64("attention_scale").unwrap_or(0.125),
            qkv_scale_folded: config.get_str("qkv_scale_folded") != Some("false"),
            mean: mean.into_iter().map(|v| v as f32).collect(),
            std_dev: std_dev.into_iter().map(|v| v as f32).collect(),
            resample,
            num_channels: config.get_i64("num_channels").unwrap_or(3),
        })
    }
}

/// One layer's weights.
struct Layer {
    ln1_gamma: Vec<f32>,
    ln1_beta: Vec<f32>,
    qkv: Vec<f32>,
    qkv_bias: Vec<f32>,
    attn_out: Vec<f32>,
    attn_out_bias: Vec<f32>,
    ln2_gamma: Vec<f32>,
    ln2_beta: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_up_bias: Vec<f32>,
    ffn_down: Vec<f32>,
    ffn_down_bias: Vec<f32>,
}

/// The ViT: the encoder stack and the head.
pub struct Vit {
    geometry: Geometry,
    layers: Vec<Layer>,
    positions: Vec<f32>,
    cls_token: Vec<f32>,
    patch_embed: Vec<f32>,
    patch_embed_bias: Vec<f32>,
    final_gamma: Vec<f32>,
    final_beta: Vec<f32>,
    head: Vec<f32>,
    head_bias: Vec<f32>,
    labels: Vec<String>,
}

/// The encoder's reusable scratch.
///
/// Before this existed, `encode` built eight large buffers for EVERY layer
/// (two `x.clone()`s, three projections, the feed-forward pair, plus the
/// attention's own two) and built a new one again for every image. A
/// classifier sees many images with one loaded model, so the buffers are
/// kept here and grown once. Nothing about the arithmetic changes: every
/// buffer is fully written before it is read, exactly as the freshly
/// allocated `vec![0.0; ..]` was.
pub struct VitWorkspace {
    patch_out: Vec<f32>,
    x: Vec<f32>,
    norm: Vec<f32>,
    qkv: Vec<f32>,
    ctx: Vec<f32>,
    proj: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    scores: Vec<f32>,
    acc: Vec<f32>,
    /// gamma and beta, packed for the array's LayerNorm (2 * hidden).
    ln_params: Vec<f32>,
}

impl Default for VitWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

impl VitWorkspace {
    /// An empty workspace; the first `encode_into` grows it to the model.
    pub fn new() -> Self {
        Self {
            patch_out: Vec::new(),
            x: Vec::new(),
            norm: Vec::new(),
            qkv: Vec::new(),
            ctx: Vec::new(),
            proj: Vec::new(),
            up: Vec::new(),
            down: Vec::new(),
            scores: Vec::new(),
            acc: Vec::new(),
            ln_params: Vec::new(),
        }
    }

    /// Grow a scratch buffer to `len`, if needed. GEMMs and copies below
    /// overwrite the whole published range, so a grow's zero fill is only a
    /// convenience, never a dependency.
    fn grow(buf: &mut Vec<f32>, len: usize) {
        if buf.len() < len {
            buf.resize(len, 0.0);
        }
    }

    fn ensure(
        &mut self,
        n_patches: usize,
        n_pos: usize,
        d: usize,
        width: usize,
        head_dim: usize,
        score_cols: usize,
    ) {
        Self::grow(&mut self.patch_out, n_patches * d);
        Self::grow(&mut self.x, n_pos * d);
        Self::grow(&mut self.norm, n_pos * d);
        Self::grow(&mut self.qkv, n_pos * 3 * d);
        Self::grow(&mut self.ctx, n_pos * d);
        Self::grow(&mut self.proj, n_pos * d);
        Self::grow(&mut self.up, n_pos * width);
        Self::grow(&mut self.down, n_pos * d);
        Self::grow(&mut self.scores, n_pos * score_cols);
        Self::grow(&mut self.acc, head_dim);
        Self::grow(&mut self.ln_params, 2 * d);
    }
}

#[cfg(test)]
mod array_seam_tests {
    use super::*;
    use npum_model::array::{ArrayAttention, ArrayGemm};
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Default)]
    struct Log {
        gemms: Vec<String>,
        attention: usize,
        ln: usize,
        gelu: usize,
    }

    struct MockText {
        log: Rc<RefCell<Log>>,
    }
    impl ArrayGemm for MockText {
        fn name(&self) -> &'static str {
            "mock"
        }
        fn gemm_on_array(&self) -> bool {
            true
        }
        fn dispatches(&self) -> usize {
            self.log.borrow().gemms.len()
        }
        fn staged_bytes(&self) -> usize {
            0
        }
        fn gemm(
            &mut self,
            weight: &str,
            _a: &[f32],
            _rows: usize,
            _k: usize,
            _n: usize,
            _bias: Option<&[f32]>,
            out: &mut [f32],
        ) -> npum_model::array::Result<()> {
            self.log.borrow_mut().gemms.push(weight.to_string());
            out.fill(0.0);
            Ok(())
        }
    }
    impl ArrayAttention for MockText {
        fn name(&self) -> &'static str {
            "mock"
        }
        fn on_array(&self) -> bool {
            true
        }
        fn attn_dispatches(&self) -> usize {
            self.log.borrow().attention
        }
        fn attn_seconds(&self) -> (f64, f64) {
            (0.0, 0.0)
        }
        fn attention(
            &mut self,
            _qkv: &[f32],
            _shape: AttentionShape,
            _mask: &[f32],
            _softmax: Option<&mut (dyn ArrayEltwise + '_)>,
            ctx: &mut [f32],
        ) -> npum_model::array::Result<()> {
            self.log.borrow_mut().attention += 1;
            ctx.fill(0.0);
            Ok(())
        }
    }
    impl ArrayText for MockText {}

impl npum_model::array::ArrayLogits for MockText {}
use npum_model::array::ArrayFrontend;
    impl ArrayFrontend for MockText {}

    struct MockEltwise {
        log: Rc<RefCell<Log>>,
    }
    impl ArrayEltwise for MockEltwise {
        fn on_array(&self, _op: EltOp) -> bool {
            true
        }
        fn layernorm(
            &mut self,
            _params: &[f32],
            _x: &mut [f32],
            _n_rows: usize,
        ) -> npum_model::array::Result<()> {
            self.log.borrow_mut().ln += 1;
            Ok(())
        }
        fn gelu(&mut self, _x: &mut [f32]) -> npum_model::array::Result<()> {
            self.log.borrow_mut().gelu += 1;
            Ok(())
        }
        fn softmax(&mut self, _x: &mut [f32], _n_rows: usize) -> npum_model::array::Result<()> {
            Ok(())
        }
        fn dispatches(&self) -> usize {
            0
        }
    }

    fn small_vit() -> (Vit, Geometry) {
        let g = Geometry {
            model_name: "tiny".into(),
            hidden: 8,
            layers: 1,
            heads: 2,
            head_dim: 4,
            image_size: 16,
            patch_size: 8,
            n_pos: 4,
            n_patches: 3,
            patch_dim: 8,
            num_labels: 3,
            ln_eps: 1e-5,
            attn_scale: 0.5,
            qkv_scale_folded: true,
            mean: vec![0.0; 3],
            std_dev: vec![1.0; 3],
            resample: Resample::Bicubic,
            num_channels: 3,
        };
        let layer = Layer {
            ln1_gamma: vec![1.0; 8],
            ln1_beta: vec![0.0; 8],
            qkv: vec![0.0; 8 * 24],
            qkv_bias: vec![0.0; 24],
            attn_out: vec![0.0; 8 * 8],
            attn_out_bias: vec![0.0; 8],
            ln2_gamma: vec![1.0; 8],
            ln2_beta: vec![0.0; 8],
            ffn_up: vec![0.0; 8 * 16],
            ffn_up_bias: vec![0.0; 16],
            ffn_down: vec![0.0; 16 * 8],
            ffn_down_bias: vec![0.0; 8],
        };
        let vit = Vit {
            geometry: g.clone(),
            layers: vec![layer],
            positions: vec![0.0; 4 * 8],
            cls_token: vec![0.0; 8],
            patch_embed: vec![0.0; 8 * 8],
            patch_embed_bias: vec![0.0; 8],
            final_gamma: vec![1.0; 8],
            final_beta: vec![0.0; 8],
            head: vec![0.0; 8 * 3],
            head_bias: vec![0.0; 3],
            labels: vec!["a".into(), "b".into(), "c".into()],
        };
        (vit, g)
    }

    #[test]
    fn the_encoder_dispatches_patch_embed_four_layer_gemms_and_attention() {
        let (vit, g) = small_vit();
        let patches = vec![0.0; g.n_patches as usize * g.patch_dim as usize];
        let log = Rc::new(RefCell::new(Log::default()));
        let mut text = MockText { log: log.clone() };
        let mut eltwise = MockEltwise { log: log.clone() };
        let mut ws = VitWorkspace::new();
        let mut out = Vec::new();

        vit.encode_into(
            &patches,
            &mut ws,
            &mut out,
            Some(&mut text),
            Some(&mut eltwise),
            true,
            true,
        )
        .expect("the array path runs");

        let recorded = log.borrow();
        assert_eq!(recorded.gemms.len(), 5, "1 patch embed + 4 per layer");
        assert_eq!(recorded.gemms[0], "frontend.patch_embed");
        assert_eq!(recorded.gemms[1], "layer.0.qkv");
        assert_eq!(recorded.gemms[2], "layer.0.attn_out");
        assert_eq!(recorded.gemms[3], "layer.0.ffn_up");
        assert_eq!(recorded.gemms[4], "layer.0.ffn_down");
        assert_eq!(recorded.attention, 1);
        assert_eq!(recorded.gelu, 1);
        assert!(recorded.ln >= 3, "ln1, ln2 and the final norm");
    }
}

/// One projection: the array's seam when asked, the host's kernel otherwise.
#[allow(clippy::too_many_arguments)]
fn gemm_seam(
    array: Option<&mut (dyn ArrayText + '_)>,
    gemm_on: bool,
    weight: &str,
    a: &[f32],
    rows: usize,
    b: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
    threads: usize,
) -> Result<(), String> {
    match array.filter(|_| gemm_on) {
        Some(backend) => backend
            .gemm(weight, a, rows, k, n, Some(bias), out)
            .map_err(|e| e.to_string()),
        None => {
            kernels::gemm_nt_rows(a, rows, b, k, n, bias, out, threads);
            Ok(())
        }
    }
}

/// One LayerNorm with a separate destination: copy and normalise on the
/// array, or the host's fused copy-and-normalise kernel.
#[allow(clippy::too_many_arguments)]
fn layernorm_seam(
    eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
    ln_on: bool,
    params: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    src: &[f32],
    dst: &mut [f32],
    rows: usize,
    width: usize,
    eps: f64,
) -> Result<(), String> {
    if let Some(eltwise) = eltwise.filter(|_| ln_on) {
        let len = rows * width;
        dst[..len].copy_from_slice(&src[..len]);
        params[..width].copy_from_slice(gamma);
        params[width..2 * width].copy_from_slice(beta);
        eltwise
            .layernorm(&params[..2 * width], &mut dst[..len], rows)
            .map_err(|e| e.to_string())
    } else {
        layernorm_rows(src, dst, rows, width, gamma, beta, eps);
        Ok(())
    }
}

impl Vit {
    /// Read the weights out of a container.
    pub fn open(path: &Path) -> Result<Vit, String> {
        let container = Container::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let geometry = Geometry::read(&container, path).ok_or_else(|| {
            format!(
                "{}: the container does not carry the ViT's geometry",
                path.display()
            )
        })?;
        let mut layers = Vec::with_capacity(geometry.layers as usize);
        for l in 0..geometry.layers {
            let p = format!("layer.{l}.");
            layers.push(Layer {
                ln1_gamma: tensor(&container, &format!("{p}ln1.weight"))?,
                ln1_beta: tensor(&container, &format!("{p}ln1.bias"))?,
                qkv: operand(&container, &format!("{p}qkv"))?,
                qkv_bias: tensor(&container, &format!("{p}qkv.bias"))?,
                attn_out: operand(&container, &format!("{p}attn_out"))?,
                attn_out_bias: tensor(&container, &format!("{p}attn_out.bias"))?,
                ln2_gamma: tensor(&container, &format!("{p}ln2.weight"))?,
                ln2_beta: tensor(&container, &format!("{p}ln2.bias"))?,
                ffn_up: operand(&container, &format!("{p}ffn_up"))?,
                ffn_up_bias: tensor(&container, &format!("{p}ffn_up.bias"))?,
                ffn_down: operand(&container, &format!("{p}ffn_down"))?,
                ffn_down_bias: tensor(&container, &format!("{p}ffn_down.bias"))?,
            });
        }
        // The labels: the vocabulary the head's OUTPUTS are named by. A
        // head with more outputs than names would print label 999's weight
        // as if it were all of them -- a finite, confident, wrong answer.
        let labels: Vec<String> = container
            .tensor_bytes("labels.table")
            .map(|b| {
                String::from_utf8_lossy(b)
                    .split('\n')
                    .filter(|l| !l.is_empty())
                    .map(|l| l.to_string())
                    .collect()
            })
            .unwrap_or_default();
        if labels.len() as i64 != geometry.num_labels {
            return Err(format!(
                "{}: labels.table has {} names and the head has {} outputs. \
                 Printing label {} as '{}' would name a class that is not the one \
                 the argmax chose.",
                path.display(),
                labels.len(),
                geometry.num_labels,
                labels.len().saturating_sub(1),
                labels.last().map(|l| l.as_str()).unwrap_or("")
            ));
        }
        Ok(Vit {
            positions: tensor(&container, "frontend.position_embeddings")?,
            cls_token: tensor(&container, "frontend.cls_token")?,
            patch_embed: operand(&container, "frontend.patch_embed")?,
            patch_embed_bias: tensor(&container, "frontend.patch_embed.bias")?,
            final_gamma: tensor(&container, "layernorm.weight")?,
            final_beta: tensor(&container, "layernorm.bias")?,
            head: tensor(&container, "classifier.weight")?,
            head_bias: tensor(&container, "classifier.bias")?,
            layers,
            geometry,
            labels,
        })
    }

    /// The model's geometry.
    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    /// The labels, in the head's order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// The encoder: the patches to the normalised positions.
    ///
    /// `patches` is the front end's `[n_patches, patch_dim]` matrix. This
    /// convenience wrapper owns a fresh workspace; callers running more
    /// than one image should keep a [`VitWorkspace`] and use
    /// [`encode_into`](Self::encode_into).
    pub fn encode(
        &self,
        patches: &[f32],
        array: ArrayTextRef<'_, '_>,
        eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
        gemm_on: bool,
        conv_on: bool,
    ) -> Result<Vec<f32>, String> {
        let mut ws = VitWorkspace::new();
        let mut out = Vec::new();
        self.encode_into(patches, &mut ws, &mut out, array, eltwise, gemm_on, conv_on)?;
        Ok(out)
    }

    /// The encoder, writing into the caller's output buffer and reusing a
    /// workspace across images.
    ///
    /// The arithmetic is the same as [`encode`](Self::encode): the same GEMMs,
    /// the same `x.clone()` expressed as `copy_from_slice`, the same
    /// `attention_into` that `attention` itself calls, and the same residual
    /// adds. Only the addresses of the scratch buffers change.
    pub fn encode_into(
        &self,
        patches: &[f32],
        ws: &mut VitWorkspace,
        out: &mut Vec<f32>,
        mut array: ArrayTextRef<'_, '_>,
        mut eltwise: Option<&mut (dyn ArrayEltwise + '_)>,
        gemm_on: bool,
        conv_on: bool,
    ) -> Result<(), String> {
        let g = &self.geometry;
        let d = g.hidden as usize;
        let n_pos = g.n_pos as usize;
        let n_patches = g.n_patches as usize;
        let width = self
            .layers
            .iter()
            .map(|l| l.ffn_up.len() / d)
            .max()
            .unwrap_or(0);

        // The array's softmax and attention normalise the DESIGN's padded key
        // count, not the sequence. A design narrower than the sequence is
        // refused; the two designs must agree on the score row's width.
        let sc_cols = match eltwise.as_deref().and_then(|e| e.softmax_cols()) {
            Some(cols) => {
                if cols < n_pos {
                    return Err(format!(
                        "the softmax design's score row is {cols} wide, but this ViT has {n_pos} positions"
                    ));
                }
                cols
            }
            None => n_pos,
        };
        // `gemm_on` comes from the CALLER's flag, not from
        // `ArrayGemm::gemm_on_array()`: the latter only says that SOME panel
        // was staged, which is also true for a `conv`-only run that staged
        // just the patch embedding. Asking that handle for the four layer
        // projections would be a refusal at the first layer, not a dispatch.
        let attn_on = array.as_deref().is_some_and(|a| a.on_array());
        if attn_on {
            let attn_cols = array
                .as_deref()
                .and_then(|a| a.score_cols())
                .unwrap_or(n_pos);
            if attn_cols < n_pos {
                return Err(format!(
                    "the attention design's score row is {attn_cols} wide, but this ViT has {n_pos} positions"
                ));
            }
            if let Some(cols) = eltwise.as_deref().and_then(|e| e.softmax_cols()) {
                if cols != attn_cols {
                    return Err(format!(
                        "the softmax design's score row is {cols} wide and the attention's is {attn_cols} wide"
                    ));
                }
            }
        }
        let ln_on = eltwise
            .as_deref()
            .is_some_and(|e| e.on_array(EltOp::LayerNorm));
        let gelu_on = eltwise.as_deref().is_some_and(|e| e.on_array(EltOp::Gelu));
        let sm_on = eltwise
            .as_deref()
            .is_some_and(|e| e.on_array(EltOp::Softmax));

        ws.ensure(n_patches, n_pos, d, width, g.head_dim as usize, sc_cols);

        let VitWorkspace {
            patch_out,
            x,
            norm,
            qkv,
            ctx,
            proj,
            up,
            down,
            scores,
            acc,
            ln_params,
        } = ws;
        let patch_out = &mut patch_out[..n_patches * d];
        let x = &mut x[..n_pos * d];
        let norm = &mut norm[..n_pos * d];
        let qkv = &mut qkv[..n_pos * 3 * d];
        let ctx = &mut ctx[..n_pos * d];
        let proj = &mut proj[..n_pos * d];
        let up = &mut up[..n_pos * width];
        let down = &mut down[..n_pos * d];
        let mask = vec![0.0f32; n_pos];

        // The patch embedding: a conv rewritten as an exact GEMM, because
        // the stride equals the kernel, so im2col is a permutation. It is
        // gated by `conv` and not by `gemm`: the status block calls this the
        // architecture's one convolution, and a run that asked only for
        // `conv` must not silently take over the four layer projections.
        gemm_seam(
            array.as_deref_mut().filter(|_| conv_on),
            conv_on,
            "frontend.patch_embed",
            patches,
            n_patches,
            &self.patch_embed,
            g.patch_dim as usize,
            d,
            &self.patch_embed_bias,
            patch_out,
            parallel::threads(),
        )?;

        // The CLS row and the position table. Positions are added AFTER
        // the concatenation and BEFORE the first LayerNorm, which is HF's
        // order: pos_embeddings[:, 1:] + patch, then cat((cls, patches)),
        // then + pos_embeddings[:, :1]. Same sum, same tensor, and the CLS
        // row therefore holds position 0 like any other.
        //
        // The scratch may hold a previous image: zero the WHOLE row range,
        // including any padding positions the patches do not fill, so a
        // reused buffer behaves exactly like the old `vec![0.0; ..]`.
        x.fill(0.0);
        x[..d].copy_from_slice(&self.cls_token[..d]);
        for p in 0..n_patches {
            x[(p + 1) * d..(p + 2) * d].copy_from_slice(&patch_out[p * d..(p + 1) * d]);
        }
        for t in 0..n_pos {
            let pe = &self.positions[t * d..(t + 1) * d];
            let row = &mut x[t * d..(t + 1) * d];
            for j in 0..d {
                row[j] += pe[j];
            }
        }

        // The scale is inside the Q weight and the Q bias unless the
        // container was packed with --no-fold-scale. Applying it here as
        // well would be applying it twice, which is a different model.
        let scale = if g.qkv_scale_folded {
            1.0f32
        } else {
            g.attn_scale as f32
        };

        let threads = parallel::threads();
        for (li, layer) in self.layers.iter().enumerate() {
            let qkv_name = format!("layer.{li}.qkv");
            let attn_out_name = format!("layer.{li}.attn_out");
            let ffn_up_name = format!("layer.{li}.ffn_up");
            let ffn_down_name = format!("layer.{li}.ffn_down");
            // ---- self-attention, pre-LN ----
            layernorm_seam(
                eltwise.as_deref_mut(),
                ln_on,
                &mut ln_params[..2 * d],
                &layer.ln1_gamma,
                &layer.ln1_beta,
                x,
                norm,
                n_pos,
                d,
                g.ln_eps,
            )?;
            gemm_seam(
                array.as_deref_mut(),
                gemm_on,
                &qkv_name,
                norm,
                n_pos,
                &layer.qkv,
                d,
                3 * d,
                &layer.qkv_bias,
                qkv,
                threads,
            )?;
            // The fused [Q|K|V] row is Q at offset 0 and a K|V block at
            // offset d, both with row stride 3*d_model -- the same layout
            // the Whisper's encoder reads. There is NO mask.
            if attn_on {
                if let Some(backend) = array.as_deref_mut() {
                    let shape = AttentionShape {
                        batch: 1,
                        seq: n_pos,
                        heads: g.heads as usize,
                        head_dim: g.head_dim as usize,
                        hidden: d,
                    };
                    let softmax = if sm_on { eltwise.as_deref_mut() } else { None };
                    backend
                        .attention(qkv, shape, &mask, softmax, ctx)
                        .map_err(|e| e.to_string())?;
                }
            } else if sm_on {
                if let Some(softmax) = eltwise.as_deref_mut() {
                    crate::whisper::attention_eltwise_into(
                        qkv,
                        3 * d,
                        &qkv[d..],
                        3 * d,
                        n_pos,
                        n_pos,
                        d,
                        g.heads as usize,
                        g.head_dim as usize,
                        scale,
                        ctx,
                        &mut scores[..n_pos * sc_cols],
                        sc_cols,
                        acc,
                        softmax,
                    )?;
                }
            } else {
                attention_into(
                    qkv,
                    3 * d,
                    &qkv[d..],
                    3 * d,
                    n_pos,
                    n_pos,
                    d,
                    g.heads as usize,
                    g.head_dim as usize,
                    scale,
                    ctx,
                    scores,
                    acc,
                );
            }
            gemm_seam(
                array.as_deref_mut(),
                gemm_on,
                &attn_out_name,
                ctx,
                n_pos,
                &layer.attn_out,
                d,
                d,
                &layer.attn_out_bias,
                proj,
                threads,
            )?;
            kernels::add_inplace(x, &proj[..n_pos * d]);

            // ---- feed-forward, pre-LN ----
            layernorm_seam(
                eltwise.as_deref_mut(),
                ln_on,
                &mut ln_params[..2 * d],
                &layer.ln2_gamma,
                &layer.ln2_beta,
                x,
                norm,
                n_pos,
                d,
                g.ln_eps,
            )?;
            let w = layer.ffn_up.len() / d;
            let up = &mut up[..n_pos * w];
            gemm_seam(
                array.as_deref_mut(),
                gemm_on,
                &ffn_up_name,
                norm,
                n_pos,
                &layer.ffn_up,
                d,
                w,
                &layer.ffn_up_bias,
                up,
                threads,
            )?;
            if gelu_on {
                if let Some(backend) = eltwise.as_deref_mut() {
                    backend.gelu(up).map_err(|e| e.to_string())?;
                }
            } else {
                gelu_erf_rows(up);
            }
            gemm_seam(
                array.as_deref_mut(),
                gemm_on,
                &ffn_down_name,
                up,
                n_pos,
                &layer.ffn_down,
                w,
                d,
                &layer.ffn_down_bias,
                down,
                threads,
            )?;
            kernels::add_inplace(x, &down[..n_pos * d]);
        }

        // The final LayerNorm, over the whole stack and not per chunk.
        if ln_on {
            if let Some(backend) = eltwise.as_deref_mut() {
                ln_params[..d].copy_from_slice(&self.final_gamma);
                ln_params[d..2 * d].copy_from_slice(&self.final_beta);
                backend
                    .layernorm(&ln_params[..2 * d], &mut x[..n_pos * d], n_pos)
                    .map_err(|e| e.to_string())?;
            }
        } else {
            layernorm_rows_inplace(x, n_pos, d, &self.final_gamma, &self.final_beta, g.ln_eps);
        }
        out.resize(n_pos * d, 0.0);
        out.copy_from_slice(x);
        Ok(())
    }

    /// The head: the CLS row to the logits.
    ///
    /// One `d x num_labels` matvec on the host. The labels are the FAST
    /// axis of the weight, and the accumulation is in DOUBLE -- the same
    /// arithmetic as the reference's `head_matvec`.
    pub fn classify(&self, cls_row: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; self.geometry.num_labels as usize];
        self.classify_into(cls_row, &mut logits);
        logits
    }

    /// The head, writing into the caller's buffer. The double accumulation
    /// and its order are identical to [`classify`](Self::classify).
    pub fn classify_into(&self, cls_row: &[f32], logits: &mut [f32]) {
        let g = &self.geometry;
        let d = g.hidden as usize;
        let nlab = g.num_labels as usize;
        for j in 0..nlab.min(logits.len()) {
            let mut acc = self.head_bias[j] as f64;
            for t in 0..d {
                acc += cls_row[t] as f64 * self.head[t * nlab + j] as f64;
            }
            logits[j] = acc as f32;
        }
    }
}

/// A prediction: the label, its name and its probability.
pub struct Prediction {
    /// The winning label's index.
    pub label: usize,
    /// The winning label's name.
    pub name: String,
    /// The winning label's probability.
    pub p: f32,
    /// The full logit row.
    pub logits: Vec<f32>,
}

/// The argmax, and then the softmax over the FULL logit row.
///
/// The max is taken first so the exponential cannot overflow, and the
/// probability is computed from the same row rather than from the top few,
/// because a truncated softmax is not a probability and reporting it as
/// one is how a confident-looking 0.99 comes out of a model that is
/// guessing.
pub fn predict(logits: &[f32], labels: &[String]) -> Result<Prediction, String> {
    if logits.is_empty() {
        return Err("the head produced no logits".into());
    }
    let mut best = 0usize;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    let mx = logits[best];
    let mut sum = 0.0f64;
    for v in logits {
        sum += (*v as f64 - mx as f64).exp();
    }
    let p = if sum > 0.0 {
        (1.0f64 / sum) as f32
    } else {
        0.0f32
    };
    Ok(Prediction {
        label: best,
        name: labels.get(best).cloned().unwrap_or_default(),
        p,
        logits: logits.to_vec(),
    })
}

/// The prediction, as the JSON object.
///
/// The reference's shape, in the reference's order: the image, the label, its name,
/// the probability, the runners-up (when more than one was asked for),
/// the three timings and the two dispatch counts.
///
/// The numbers are `%.9g`, and the probability is a float32 printed with
/// nine significant digits -- which is still far more than a softmax
/// carries and is what the golden transcript holds.
pub fn prediction_json(
    prediction: &Prediction,
    image: &str,
    top_k: usize,
    front_end_s: f64,
    encoder_s: f64,
    head_s: f64,
) -> String {
    let mut out = format!(
        "{{\"image\": \"{}\", \"label\": {}, \"name\": \"{}\", \"p\": {}",
        json_escape(image),
        prediction.label,
        json_escape(&prediction.name),
        format_g(prediction.p as f64, 9),
    );
    if top_k > 1 && !prediction.logits.is_empty() {
        // The runners-up, by the same argmax the label came from, and out of
        // the SAME softmax -- so their probabilities sum with `p` to 1 rather
        // than being independently normalised numbers.
        let mx = prediction
            .logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for v in &prediction.logits {
            sum += (*v as f64 - mx as f64).exp();
        }
        let mut order: Vec<usize> = (0..prediction.logits.len()).collect();
        order.sort_by(|a, b| {
            prediction.logits[*b]
                .partial_cmp(&prediction.logits[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let want = top_k.min(order.len());
        out.push_str(", \"top_k\": [");
        for i in 0..want {
            if i > 0 {
                out.push_str(", ");
            }
            let p = if sum > 0.0 {
                (prediction.logits[order[i]] as f64 - mx as f64).exp() / sum
            } else {
                0.0
            };
            out.push_str(&format!(
                "{{\"label\": {}, \"p\": {}}}",
                order[i],
                format_g(p, 9)
            ));
        }
        out.push(']');
    }
    out.push_str(&format!(
        ", \"front_end_s\": {}, \"encoder_s\": {}, \"head_s\": {}, \
         \"dispatches\": 0, \"elt_dispatches\": 0}}",
        format_g(front_end_s, 9),
        format_g(encoder_s, 9),
        format_g(head_s, 9),
    ));
    out
}

/// The JSON escaping the reference's server uses.
///
/// `"`, `\\`, the three named control characters, and `\\uXXXX` for the rest
/// of the control range.
pub fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// `%.9g`: nine significant digits, C's rules.
///
/// The probability is a float32 printed with nine significant digits, and
/// the golden transcript holds those digits, so the formatting is the
/// contract and not a detail.
pub fn format_g(value: f64, precision: usize) -> String {
    if value.is_nan() {
        return if value.is_sign_negative() {
            "-nan".to_string()
        } else {
            "nan".to_string()
        };
    }
    if value.is_infinite() {
        return if value < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0".to_string()
        } else {
            "0".to_string()
        };
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

/// A tensor's values.
/// A GEMM operand: F32 row-major in a host container, BF16 panels in one
/// packed for the array. Both are valid inputs to this same forward pass.
fn operand(container: &Container, name: &str) -> Result<Vec<f32>, String> {
    super::operand::read_operand(container, name)
}

fn tensor(container: &Container, name: &str) -> Result<Vec<f32>, String> {
    let bytes = container
        .tensor_bytes(name)
        .map_err(|_| format!("the container holds no tensor named {name}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
