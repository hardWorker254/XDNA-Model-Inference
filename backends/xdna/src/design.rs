//! Design sets: the compiled instruction streams, and whether they fit.
//!
//! # The fit check is a PURE FUNCTION
//!
//! `design_fits()` in the C++ runtime read the environment and compared
//! STRINGS. This one takes a `Caps`, a `Design` and a `Layout` and answers.
//! Nothing it reads is ambient:
//!
//! * the generation comes from the device, not from `NPU2=1`;
//! * the operand dtype comes from the layout, not from a flag;
//! * the sub-tile comes from `Geometry::for_generation`, not from a literal
//!   in the dispatch code.
//!
//! # Fail-closed
//!
//! A design set built for the OTHER generation is refused, even when the
//! geometry happens to agree. The C++ runtime did this by comparing the
//! `device` string that the exporter wrote into `design.json`; that check
//! stays, and it stays strict, because the alternative -- accepting a set
//! whose geometry coincides -- is how a wrong-but-plausible answer arrives.

use npum_format::layout::{Generation, OperandDtype};
use serde::{Deserialize, Serialize};

/// One compiled design, described: what it is and what it was built for.
///
/// The DESCRIPTION, not the design itself: the thing on the board is
/// `array::ArrayDesign`, and this is what the fit checks look at. The C++
/// runtime had no such split -- its `npu::Design` was both -- and the
/// difference between "the description does not fit" and "the board refused"
/// is worth keeping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Design {
    /// The design set's name -- `gemm_rtp`, `gelu`, `layernorm`, `softmax`.
    pub name: String,
    /// The generation this design was compiled for.
    pub generation: Generation,
    /// The operand dtype this design runs.
    pub dtype: OperandDtype,
    /// The tile's K extent.
    pub tile_k: i64,
    /// The tile's N extent.
    pub tile_n: i64,
    /// The instruction streams, by name.
    pub streams: Vec<String>,
}

/// A set of designs: everything a model needs to run on the array.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DesignSet {
    /// The designs, by name.
    pub designs: Vec<Design>,
}

impl DesignSet {
    /// One design, by name.
    pub fn design(&self, name: &str) -> Option<&Design> {
        self.designs.iter().find(|d| d.name == name)
    }

    /// Whether every design in this set fits the given capabilities.
    ///
    /// The whole set is checked and not one design of it, because a set that
    /// is half for npu1 and half for npu2 is exactly the failure the
    /// generation check exists to catch.
    pub fn fits(&self, generation: Generation, dtype: OperandDtype) -> Result<(), String> {
        for d in &self.designs {
            if d.generation != generation {
                return Err(format!(
                    "the design set holds `{}` for {}, and this device is {}",
                    d.name,
                    d.generation.as_str(),
                    generation.as_str()
                ));
            }
            if d.dtype != dtype {
                return Err(format!(
                    "the design set holds `{}` for {}, and this container is {}",
                    d.name,
                    d.dtype.tag(),
                    dtype.tag()
                ));
            }
        }
        Ok(())
    }
}

/// Whether a design serves a layout.
///
/// A pure function of three values and nothing else -- no environment, no
/// clock, no file system. That is what makes it testable: the table of
/// cases is a table of triples.
pub fn design_fits(
    caps_generation: Generation,
    caps_dtype: OperandDtype,
    design: &Design,
    layout_mac: (i64, i64),
) -> Result<(), String> {
    if design.generation != caps_generation {
        return Err(format!(
            "the design `{}` is built for {}, and this device is {}",
            design.name,
            design.generation.as_str(),
            caps_generation.as_str()
        ));
    }
    if design.dtype != caps_dtype {
        return Err(format!(
            "the design `{}` runs {}, and this container is {}",
            design.name,
            design.dtype.tag(),
            caps_dtype.tag()
        ));
    }
    let expected = npum_format::layout::mac_for(caps_generation, caps_dtype);
    if (design.tile_k, design.tile_n) != (layout_mac.0, layout_mac.1) {
        return Err(format!(
            "the design `{}` is tiled ({}, {}), and the container's panel is (s={}, t={})",
            design.name, design.tile_k, design.tile_n, expected.s, expected.t
        ));
    }
    Ok(())
}
