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
    // The opening rules are shared with every other family in `npu`. The set
    // is named here, by this family, and not hardcoded in the opener: a family
    // whose convolutions were compiled against a differently named set can say
    // so without editing the common module.
    let backend = crate::npu::open_convs(path, specs, args, root, "gemm_rtp")?;
    Ok(backend as Box<dyn npum_model::array::ArrayConvs>)
}

/// Open the array's convolutions AND the depthwise design set.
///
/// Both are needed when `--npu-ops` names `dwconv`: a dense panel set
/// without the second xclbin would refuse dwconv at the first layer.
pub fn open_with_dw(
    path: &Path,
    specs: &[ConvSpec],
    args: &crate::flags::Args,
    root: &Path,
) -> Result<Box<dyn npum_model::array::ArrayConvs>, String> {
    let backend = crate::npu::open_convs_with_dw(path, specs, args, root, "gemm_rtp")?;
    Ok(backend as Box<dyn npum_model::array::ArrayConvs>)
}

/// The convolution design set's directory, if one can be found.
///
/// The opener resolves the set from the container or from `--artifacts`; this
/// is the directory form kept for callers that only want to print where the
/// set came from.
pub fn find_design(root: &Path, model: &Path, named: Option<&str>) -> Option<PathBuf> {
    let stem = crate::npu::model_name(model);
    crate::npu::find_artifacts(root, &stem, named).map(|dir| dir.join("gemm_rtp"))
}
