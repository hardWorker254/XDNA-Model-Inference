//! Opening the array's convolutions for a family.
//!
//! # THE TWO THINGS THE ARRAY NEEDS ARE SEPARATE
//!
//! * the PANELS -- the container's own pre-tiled bf16 weights, one per dense
//!   convolution. A container packed without them says so in its config, and
//!   the refusal says how to pack it.
//!
//! * the DESIGN SET -- the compiled core. It is the MODEL's own and it
//!   cannot be guessed: its streams are named after the padded shapes this
//!   model's convolutions need, and a design set from another model would
//!   dispatch another model's shapes against this model's weights.
//!
//! The container's own copy of the design set is preferred when it has one,
//! and a directory is the fallback.
//!
//! # THE FAMILY DECIDES WHAT A CONVOLUTION IS
//!
//! This file opens the board; it does not decide which convolutions go to
//! it. That list is the FAMILY's, because only the family knows which
//! convolutions its graph RUNS: the OCR family's detector and recogniser
//! both have dense convolutions and the pose family's graph has one of them
//! per node, and a panel exists for a convolution that is called.

use std::path::{Path, PathBuf};

use npum_model::array::ConvSpec;

/// Open the array's convolutions for a family.
///
/// `specs` is the family's own list of the convolutions it runs; an empty
/// list is a refusal, because a design set with no panels behind it is not a
/// design set.
pub fn open(
    path: &Path,
    specs: &[ConvSpec],
    args: &crate::flags::Args,
    root: &Path,
) -> Result<Box<dyn npum_model::array::ArrayConvs>, String> {
    use npum_xdna::array::ArrayDesign;
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

    // THE DESIGN SET.
    //
    // The container's own copy first, and then the directory: the container
    // is self-sufficient when it carries the set, and a container that does
    // not needs nothing beside it.
    let own = container
        .directory()
        .tensor("design/gemm_rtp/design.json")
        .is_some();
    let dir =
        if own {
            None
        } else {
            Some(find_design(root, path, args.value("--artifacts")).ok_or_else(|| {
            format!(
                "{}: --npu-ops conv needs a design set on disk: pass --artifacts <dir> naming \
                 one, or build it with the exporter. The flag has no default to fall back on, \
                 because this container's padded (K, N) designs are its own -- and picking one \
                 silently is how a run reports array numbers from a design set belonging to \
                 another model.",
                path.display()
            )
        })?)
        };

    // THE BOARD.
    let transport = crate::transport::open().map_err(|e| format!("{}: {e}", path.display()))?;
    let design = match dir {
        Some(dir) => ArrayDesign::open(&dir, transport).map_err(|e| e.to_string())?,
        None => ArrayDesign::open_from_container(path, "gemm_rtp", transport)
            .map_err(|e| e.to_string())?,
    };

    let backend = ArrayConvs::open(path, design, specs).map_err(|e| e.to_string())?;
    Ok(Box::new(backend))
}

/// The design set's directory, if one can be found.
///
/// The `--artifacts` flag first -- it is the reader saying where their own
/// design set is -- and then the tree's own artifacts, under the model's
/// name. The model's name is the CONTAINER's file name, because that is
/// what the artifacts are laid out under.
pub fn find_design(root: &Path, model: &Path, named: Option<&str>) -> Option<PathBuf> {
    let usable = |dir: &Path| dir.join("gemm_rtp").join("design.json").is_file();
    if let Some(named) = named {
        let dir = PathBuf::from(named);
        if usable(&dir) {
            return Some(dir.join("gemm_rtp"));
        }
        if dir.join("design.json").is_file() {
            return Some(dir);
        }
    }
    let stem = model
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    if stem.is_empty() {
        return None;
    }
    crate::vit::find_designs(root, &stem).map(|dir| dir.join("gemm_rtp"))
}
