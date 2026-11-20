//! Opening the array's designs and specs -- ONE place, for every family.
//!
//! # WHY THIS MODULE EXISTS
//!
//! The embedding path grew a correct way to open a design set: prefer the
//! container's own copy, fall back to the tree, never pick a set silently, and
//! check the panels before anything is staged. The speech, image and
//! convolution families need the SAME rules. A rule copied into four commands
//! is a rule that will be fixed in one of them and drift in the other three,
//! so the rules live here.
//!
//! What is common is the BOARD: finding a design, opening a context, checking
//! a panel's dtype and layout, and building the projection specs a family's
//! geometry implies. What stays with the family is the WORK: which convolutions
//! it runs, which elementwise passes it asks for, and which op codes its
//! `--npu-ops` flag may name.
//!
//! # THE TWO THINGS THE ARRAY NEEDS ARE SEPARATE
//!
//! * the PANELS -- the container's pre-tiled operands. They are the weights'
//!   own bytes, rounded to the array's dtype and laid out for the MMAC. A
//!   container packed for the host has no panels, and a name-only check is how
//!   the refusal says so.
//! * the DESIGN SET -- the compiled core. It is the MODEL's own: its streams
//!   are named after this model's padded shapes, and a set from another model
//!   would dispatch another model's shapes against these panels.
//!
//! `--artifacts` names WHERE the reader's own set is. It used to be ignored by
//! the tree search, which made the "pass --artifacts" refusals lies; the search
//! here names it first.

use std::path::{Path, PathBuf};

use npum_model::array::{ArrayConvs, ArrayEltwise, ArrayText, ConvSpec};
use npum_xdna::array::ArrayDesign;
use npum_xdna::array_gemms::{ArrayGemms, GemmSpec};

/// The board transport this build speaks through.
pub type Board = crate::transport::Transport;

/// A container's name, without the directory and without the extension.
///
/// The artifacts are laid out under this name, so it is also the model name
/// the design search uses.
pub fn model_name(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default()
}

/// Whether `--npu-ops` names `code`.
///
/// A code is asked for by NAME, never inferred from a design's presence: a
/// design on disk is not a dispatch, and a status block that claimed array
/// work the host did would be a block that lies.
pub fn wants_op(args: &crate::flags::Args, code: &str) -> bool {
    args.value("--npu-ops")
        .map(|v| v.split(',').any(|c| c.trim() == code))
        .unwrap_or(false)
}

/// Check the hardware-context budget for a family's array run.
///
/// A family that opens `dense_sets` GEMM designs also opens one context per
/// elementwise pass the flag named. A HOST run opens none, so it skips the
/// check: refusing a host command because another process owns the board
/// would be a refusal with no cause.
pub fn check_array_budget(args: &crate::flags::Args, dense_sets: usize) -> bool {
    let dense = wants_op(args, "gemm") || wants_op(args, "attn") || wants_op(args, "conv");
    let mut contexts = if dense { dense_sets } else { 0 };
    for code in ["layn", "gelu", "softm"] {
        if wants_op(args, code) {
            contexts += 1;
        }
    }
    if contexts == 0 {
        return true;
    }
    let allow_contention = args.switch("--allow-contention");
    let report = npum_xdna::contention::survey_contexts();
    if npum_xdna::contention::require_context_budget(&report, contexts, allow_contention) {
        return true;
    }
    eprintln!(
        "error: NPU context budget: refusing to load {contexts} concurrent hw_context(s) -- see the report above (close the other process, or pass --allow-contention)"
    );
    false
}

/// Find a model's design-set root, honouring `--artifacts`.
///
/// `--artifacts` may name either the model's root (the directory that holds
/// `gemm_rtp/`, `layernorm/`, ...) or a set directory itself. Both resolve to
/// the model's root, because every caller appends the set's name to the answer.
///
/// The tree search is the reference runtime's order: the root's
/// `runtime/artifacts/<model>/artifacts_npu1` first, because that is where the
/// exporter puts it and where the status block says it is.
pub fn find_artifacts(root: &Path, model: &str, named: Option<&str>) -> Option<PathBuf> {
    if model.is_empty() {
        return None;
    }

    // THE READER'S OWN SET, FIRST.
    if let Some(named) = named {
        let dir = PathBuf::from(named);
        if dir.join("gemm_rtp").join("design.json").is_file() {
            return Some(dir);
        }
        // A set directory was named; its parent is the model's root.
        if dir.join("design.json").is_file() {
            return dir.parent().map(|p| p.to_path_buf());
        }
    }

    for base in [
        root.join("runtime").join("artifacts"),
        root.join("artifacts"),
        root.join("runtime"),
        root.to_path_buf(),
    ] {
        for sub in ["artifacts_npu1", "artifacts_npu2", "", "npu1", "npu2"] {
            let dir = if sub.is_empty() {
                base.join(model)
            } else {
                base.join(model).join(sub)
            };
            if dir.join("gemm_rtp").join("design.json").is_file() {
                return Some(dir);
            }
        }
    }
    None
}

/// Whether the container carries its own copy of a design set.
pub fn has_own_design(container: &npum_format::Container, set: &str) -> bool {
    container
        .directory()
        .tensor(&format!("design/{set}/design.json"))
        .is_some()
}

/// One design set, from the container's own copy or from the tree.
///
/// The container is the PREFERRED source: a container that carries its set
/// needs nothing beside it -- no directory, no MLIR-AIE, no installed
/// artifacts. A directory is the fallback, and an absent one is a refusal that
/// names `--artifacts` rather than a silent host computation.
pub fn open_design(
    container: &npum_format::Container,
    path: &Path,
    root: &Path,
    named: Option<&str>,
    set: &str,
    code: &str,
    hint: &str,
) -> Result<ArrayDesign<Board>, String> {
    use npum_xdna::array::ArrayDesign;

    let transport = crate::transport::open().map_err(|e| format!("{}: {e}", path.display()))?;
    if has_own_design(container, set) {
        ArrayDesign::open_from_container(path, set, transport).map_err(|e| e.to_string())
    } else {
        let dir = find_artifacts(root, &model_name(path), named).ok_or_else(|| {
            format!(
                "{}: --npu-ops {code} needs the {set} design set: this container does not carry \
                 one, and no design set was found for it. {hint}",
                path.display()
            )
        })?;
        ArrayDesign::open(&dir.join(set), transport).map_err(|e| e.to_string())
    }
}

/// The refusal for a container packed without the array's panels.
///
/// The check is on the WEIGHT, not on a generic config flag: the panel the
/// projection multiplies by has to be the projection's own.
pub fn check_bf16_panel(
    container: &npum_format::Container,
    path: &Path,
    weight: &str,
    code: &str,
) -> Result<(), String> {
    let panel = container.directory().tensor(weight).ok_or_else(|| {
        format!(
            "{}: --npu-ops {code} needs the array's panels, and this container does not carry \
             any: it has no {weight}. Repack it with the packer's --dtype bf16, which stores \
             every projection's weights as the pre-tiled bf16 panel the array multiplies by.",
            path.display()
        )
    })?;
    if panel.dtype != "BF16" {
        return Err(format!(
            "{}: --npu-ops {code} needs the array's panels, and this container does not carry \
             any: {weight} is {}, not BF16. Repack it with the packer's --dtype bf16, which \
             stores every projection's weights as the pre-tiled bf16 panel the array multiplies \
             by.",
            path.display(),
            panel.dtype
        ));
    }
    Ok(())
}

/// Check the B layout before anything is staged.
///
/// A panel whose sizes agree with the design but whose bytes are tiled
/// differently is the dangerous case: every shape check passes and the array
/// multiplies the wrong bytes. The container records its panel's `layout_hash`;
/// the design records the layout it was compiled for. They must be the same
/// hash. Old containers without a hash are left alone -- an absent check is
/// not a failed one.
fn check_layouts(
    design: &ArrayDesign<Board>,
    container: &npum_format::Container,
    path: &Path,
    specs: &[GemmSpec],
) -> Result<(), String> {
    let info = design.info();
    let expected = if info.b_layout_hash.is_empty() {
        info.b_layout.as_ref().map(npum_format::layout_hash)
    } else {
        Some(info.b_layout_hash.clone())
    };

    // The design's own manifest must agree with itself before it is used as
    // the standard a container is judged by.
    if let (Some(layout), Some(hash)) = (info.b_layout.as_ref(), expected.as_ref()) {
        let own = npum_format::layout_hash(layout);
        if &own != hash {
            return Err(format!(
                "{}: the design's b_layout_hash ({hash}) does not match its own b_layout ({own}). \
                 Re-export the design set; a hash that disagrees with its layout cannot be used \
                 to check the container's panels.",
                path.display()
            ));
        }
    }

    let Some(expected) = expected else {
        return Ok(());
    };
    for spec in specs {
        if let Some(entry) = container.directory().tensor(&spec.weight) {
            if let Some(got) = entry.layout_hash.as_ref() {
                if got != &expected {
                    return Err(format!(
                        "{}: {} has layout_hash {got}, but this design set was compiled for \
                         {expected}. The sizes agree, so a shape check would pass -- but the \
                         bytes are tiled differently. Repack the container with the B layout \
                         this design set expects.",
                        path.display(),
                        spec.weight
                    ));
                }
            }
        }
    }
    Ok(())
}

/// The shape of a text model's dense work.
///
/// A layer's four projections are described by their weight names and widths;
/// only the fused-up GTE differs, whose `ffn_up` is `[hidden, 2 * ffn]`.
#[derive(Debug, Clone, Copy)]
pub struct TextGeometry {
    pub layers: usize,
    pub hidden: usize,
    pub ffn: usize,
    /// `ffn_up`'s output is `ffn` times this factor: 1 for BERT, 2 for GTE.
    pub ffn_up_factor: usize,
}

impl TextGeometry {
    /// The BERT shape: `ffn_up` is `[hidden, ffn]`.
    pub fn bert(layers: usize, hidden: usize, ffn: usize) -> Self {
        Self {
            layers,
            hidden,
            ffn,
            ffn_up_factor: 1,
        }
    }

    /// The GTE shape: `ffn_up` fuses the up and gate projections.
    pub fn gte(layers: usize, hidden: usize, ffn: usize) -> Self {
        Self {
            layers,
            hidden,
            ffn,
            ffn_up_factor: 2,
        }
    }

    /// The projections of every layer, in the order the run dispatches them.
    pub fn gemm_specs(&self) -> Vec<GemmSpec> {
        let mut specs = Vec::with_capacity(self.layers * 4);
        for layer in 0..self.layers {
            for (op, k, n) in [
                ("qkv", self.hidden, 3 * self.hidden),
                ("attn_out", self.hidden, self.hidden),
                ("ffn_up", self.hidden, self.ffn * self.ffn_up_factor),
                ("ffn_down", self.ffn, self.hidden),
            ] {
                specs.push(GemmSpec {
                    weight: format!("layer.{layer}.{op}"),
                    op: op.to_string(),
                    k,
                    n,
                });
            }
        }
        specs
    }
}

/// The array's backend for a text model: the dense projections and, when the
/// flag also asks for it, the attention's two GEMMs.
///
/// Both are streams of the SAME design set -- `attn_qk` and `attn_av` live in
/// `gemm_rtp` -- so one handle is one hardware context, whether one or both
/// were asked for.
pub fn open_array_text(
    container: &npum_format::Container,
    path: &Path,
    root: &Path,
    args: &crate::flags::Args,
    geometry: TextGeometry,
) -> Result<Box<dyn ArrayText + Send>, String> {
    let specs = if wants_op(args, "gemm") {
        geometry.gemm_specs()
    } else {
        Vec::new()
    };
    open_array_text_specs(container, path, root, args, specs, "gemm_rtp")
}

/// Open an array backend for a set of projections the CALLER built.
///
/// [`open_array_text`] builds its specs from a [`TextGeometry`], which names
/// the projections `layer.N.{qkv,attn_out,ffn_up,ffn_down}`. Whisper prefixes
/// them with `encoder.`/`decoder.`, and ViT has an extra patch-embedding GEMM,
/// so those families build their own specs and call this instead.
///
/// `set` is the design set by name: `gemm_rtp` for an encoder or a classifier,
/// `gemm_rtp_dec` for Whisper's decoder. The attention's streams live inside
/// the same set, so one set is one hardware context whether the flag asked for
/// the projections, the attention, or both.
pub fn open_array_text_specs(
    container: &npum_format::Container,
    path: &Path,
    root: &Path,
    args: &crate::flags::Args,
    specs: Vec<GemmSpec>,
    set: &str,
) -> Result<Box<dyn ArrayText + Send>, String> {
    let gemm = !specs.is_empty();
    let attn = wants_op(args, "attn");
    if !gemm && !attn {
        return Err(format!(
            "{}: --npu-ops asks for no array work, so no design set should be opened",
            path.display()
        ));
    }

    // The panic-free, name-based refusal for a container packed for the host.
    // The first spec is a real projection of this model, so the panel it names
    // is the one this run would actually multiply by.
    if gemm {
        check_bf16_panel(container, path, &specs[0].weight, "gemm")?;
    }

    let design = open_design(
        container,
        path,
        root,
        args.value("--artifacts"),
        set,
        if gemm { "gemm" } else { "attn" },
        "Pass --artifacts <dir> naming one, or build it with the exporter. The flag has no \
         default to fall back on, because this model's design is its own.",
    )?;

    check_layouts(&design, container, path, &specs)?;

    let backend = ArrayGemms::open(path, design, &specs, attn).map_err(|e| e.to_string())?;
    Ok(Box::new(backend))
}

/// Whether an eltwise design set is present, in the container or on disk.
///
/// Presence is checked before opening so a real load failure (a corrupt or
/// incompatible design) is reported directly, not hidden behind an unrelated
/// fallback lookup.
pub fn eltwise_design_present(
    container: &npum_format::Container,
    path: &Path,
    root: &Path,
    named: Option<&str>,
    set: &str,
) -> bool {
    if has_own_design(container, set) {
        return true;
    }
    let Some(dir) = find_artifacts(root, &model_name(path), named) else {
        return false;
    };
    let set_dir = dir.join(set);
    set_dir.join("design.json").is_file() || set_dir.is_dir()
}

/// Open one eltwise set, container-first, and wrap it as a design.
fn open_eltwise_design(
    container: &npum_format::Container,
    path: &Path,
    root: &Path,
    named: Option<&str>,
    set: &str,
    code: &str,
) -> Result<npum_xdna::eltwise::EltwiseDesign<Board>, String> {
    use npum_xdna::eltwise::EltwiseDesign;

    let design = open_design(
        container,
        path,
        root,
        named,
        set,
        code,
        "Pass --artifacts <dir> naming one.",
    )?;
    EltwiseDesign::open(design).map_err(|e| e.to_string())
}

/// The array's elementwise passes, when the run asked for them.
///
/// Each pass is a separate design and a separate hardware context, so only
/// the ones the flag named are opened. A pass that was not opened stays on the
/// host. RMSNorm is opened under the `layn` code for models that use it.
pub fn open_eltwise(
    container: &npum_format::Container,
    path: &Path,
    root: &Path,
    args: &crate::flags::Args,
) -> Result<Option<Box<dyn ArrayEltwise + Send>>, String> {
    use npum_xdna::eltwise::Eltwise;

    let named = args.value("--artifacts");
    let mut layernorm = None;
    let mut rmsnorm = None;
    if wants_op(args, "layn") {
        // Prefer LayerNorm; for RMSNorm models the exported set is `rmsnorm/`.
        // Only fall back when the LayerNorm design is absent, so a broken
        // LayerNorm design is reported instead of hidden behind another set.
        if eltwise_design_present(container, path, root, named, "layernorm") {
            layernorm = Some(open_eltwise_design(
                container,
                path,
                root,
                named,
                "layernorm",
                "layn",
            )?);
        } else if eltwise_design_present(container, path, root, named, "rmsnorm") {
            rmsnorm = Some(open_eltwise_design(
                container, path, root, named, "rmsnorm", "layn",
            )?);
        } else {
            let _ = open_eltwise_design(container, path, root, named, "layernorm", "layn")?;
        }
    }
    let mut gelu = None;
    if wants_op(args, "gelu") {
        gelu = Some(open_eltwise_design(
            container, path, root, named, "gelu", "gelu",
        )?);
    }
    let mut softmax = None;
    if wants_op(args, "softm") {
        softmax = Some(open_eltwise_design(
            container, path, root, named, "softmax", "softm",
        )?);
    }
    if layernorm.is_none() && rmsnorm.is_none() && gelu.is_none() && softmax.is_none() {
        return Ok(None);
    }
    let backend =
        Eltwise::with_rmsnorm(layernorm, gelu, softmax, rmsnorm).map_err(|e| e.to_string())?;
    Ok(Some(Box::new(backend)))
}

/// Open the array's convolutions for a family.
///
/// `set` names the design set the family's convolutions were compiled against.
/// It is the CALLER's decision -- usually the container's own `gemm_rtp` -- and
/// is deliberately not hardcoded here: a family with a differently named set
/// must be able to say so, and a wrong set must be a refusal, never a guess.
pub fn open_convs(
    path: &Path,
    specs: &[ConvSpec],
    args: &crate::flags::Args,
    root: &Path,
    set: &str,
) -> Result<Box<dyn ArrayConvs + Send>, String> {
    use npum_xdna::array_convs::ArrayConvs;

    // THE PANELS.
    let container =
        npum_format::Container::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let streams = container.config_str("npu_streams").unwrap_or_default();
    if streams.is_empty() {
        return Err(format!(
            "{}: --npu-ops conv needs the array panels, and this container does not carry \
             any: it was packed without --npu. Repack with the packer's --npu, which stages \
             a pre-tiled bf16 B panel for every DENSE convolution. (A design set is the \
             SECOND requirement, and a separate one.)",
            path.display()
        ));
    }

    // THE CONVOLUTIONS.
    if specs.is_empty() {
        return Err(format!(
            "{}: the container names array streams, and none of them belongs to a convolution \
             this graph runs.",
            path.display()
        ));
    }

    // THE DESIGN SET: the container's own copy first, then the directory.
    let design = open_design(
        &container,
        path,
        root,
        args.value("--artifacts"),
        set,
        "conv",
        "Pass --artifacts <dir> naming one, or build it with the exporter. This container's \
         padded (K, N) designs are its own, and picking one silently is how a run reports array \
         numbers from a design set belonging to another model.",
    )?;

    let backend = ArrayConvs::open(path, design, specs).map_err(|e| e.to_string())?;
    Ok(Box::new(backend))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_name_uses_the_file_stem() {
        assert_eq!(
            model_name(Path::new("/m/bge-base-en-v1.5.npum")),
            "bge-base-en-v1.5"
        );
        assert_eq!(model_name(Path::new("whisper-base")), "whisper-base");
    }

    #[test]
    fn gemm_specs_follow_the_layer_weight_names() {
        let g = TextGeometry::bert(2, 8, 16);
        let specs = g.gemm_specs();
        assert_eq!(specs.len(), 8);
        assert_eq!(specs[0].weight, "layer.0.qkv");
        assert_eq!((specs[0].k, specs[0].n), (8, 24));
        assert_eq!(specs[2].weight, "layer.0.ffn_up");
        assert_eq!((specs[2].k, specs[2].n), (8, 16));
        assert_eq!(specs[3].weight, "layer.0.ffn_down");
        assert_eq!((specs[3].k, specs[3].n), (16, 8));
        assert_eq!(specs[4].weight, "layer.1.qkv");
    }

    #[test]
    fn gte_fuses_up_and_gate() {
        let g = TextGeometry::gte(1, 8, 16);
        let specs = g.gemm_specs();
        assert_eq!((specs[2].k, specs[2].n), (8, 32));
        assert_eq!((specs[3].k, specs[3].n), (16, 8));
    }

    #[test]
    fn shipped_gemm_designs_have_a_consistent_b_layout_hash() {
        // The layout check in `check_layouts` uses the design's hash as the
        // standard a container's panels are judged by. A design whose own hash
        // disagrees with its own layout could refuse good containers -- or
        // accept bad ones -- so guard the shipped sets against that drift.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut seen = 0;
        for model in [
            "bge-base-en-v1.5",
            "embeddinggemma-300m",
            "whisper-base",
            "vit-base-patch16-224",
        ] {
            let file = root
                .join("runtime")
                .join("artifacts")
                .join(model)
                .join("artifacts_npu1")
                .join("gemm_rtp")
                .join("design.json");
            let Ok(text) = std::fs::read_to_string(&file) else {
                continue;
            };
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
                continue;
            };
            let Some(layout) = value.get("b_layout") else {
                continue;
            };
            let Ok(layout) = serde_json::from_value::<npum_format::BLayout>(layout.clone()) else {
                continue;
            };
            let expected = npum_format::layout_hash(&layout);
            let actual = value
                .get("b_layout_hash")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            assert_eq!(
                expected,
                actual,
                "{}: design manifest disagrees with its own B layout",
                file.display()
            );
            seen += 1;
        }
        assert!(seen > 0, "no shipped GEMM design set was found to check");
    }

    #[test]
    fn find_artifacts_honours_the_named_directory() {
        let base = std::env::temp_dir().join(format!("npu-artifacts-{}", std::process::id()));
        let model_root = base.join("my-model").join("artifacts_npu1");
        let set = model_root.join("gemm_rtp");
        let _ = std::fs::create_dir_all(&set);
        let _ = std::fs::write(set.join("design.json"), "{}");
        let tree = base.join("tree");

        // A model root named directly.
        assert_eq!(
            find_artifacts(&tree, "my-model", Some(model_root.to_str().unwrap())),
            Some(model_root.clone())
        );
        // A set directory named directly resolves to its parent.
        assert_eq!(
            find_artifacts(&tree, "my-model", Some(set.to_str().unwrap())),
            Some(model_root)
        );
        // Without --artifacts, the tree search finds nothing here.
        assert_eq!(find_artifacts(&tree, "my-model", None), None);
        let _ = std::fs::remove_dir_all(&base);
    }
}
