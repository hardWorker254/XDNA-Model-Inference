//! The built-in catalogue: the models this build knows how to fetch and run.
//!
//! # What the catalogue is
//!
//! Three things, and all three are its SHAPE rather than its contents:
//!
//! * the rows of the `list` table -- name, state, layers, hidden, pooling,
//!   size, note;
//! * the refusal for a model this build does not know, which LISTS the
//!   models it does know;
//! * the geometry the packer cross-checks against the downloaded
//!   `config.json`.
//!
//! # Where the data comes from
//!
//! `core/npum-families/data/catalogue.json` is the source of truth, and
//! `tools/gen/gen_catalogue.py` turns it into the table below. The JSON
//! stays in the tree so that the data is reviewable as data, and the table
//! ships so that the answer costs no parsing.
//!
//! The table was extracted from the reference runtime's own catalogue when the
//! runtime moved, and the extraction was checked row for row against the
//! oracle's `list` output and against its refusal for an unknown model.
//!
//! # The geometry is a DESCRIPTION
//!
//! The numbers here describe the model for the table and for the
//! cross-check. They are never what the packer trusts: the packer reads the
//! checkpoint's own config, and a mismatch is an error rather than a
//! correction.

use serde::Serialize;

/// One row of the built-in catalogue.
///
/// The strings are `&'static str`, not `String`: the catalogue is static
/// data, and a static table of owned strings cannot be built in a `static`
/// -- `String::from` is not a const function -- which would mean either a
/// `LazyLock` for data that never changes, or an allocation per row.
#[derive(Debug, Clone, Serialize)]
pub struct CatalogueEntry {
    /// The container stem, and what `serve <name>` takes.
    pub name: &'static str,
    /// The HuggingFace repository.
    pub repo: &'static str,
    /// The pin over the ONNX export.
    pub sha256: &'static str,
    /// The pooling: `cls` or `mean`.
    pub pooling: &'static str,
    /// The hidden size.
    pub hidden: i64,
    /// The number of layers.
    pub layers: i64,
    /// The number of attention heads.
    pub heads: i64,
    /// The feed-forward width.
    pub ffn: i64,
    /// The tile's N extent.
    pub tile_n: i64,
    /// The checkpoint's size in megabytes -- the checkpoint, not the
    /// container.
    pub download_mb: f64,
    /// One line for the table.
    pub note: &'static str,
}

include!("catalogue_data.rs");

/// The catalogue, in its own order.
///
/// The order is the order the reference runtime's table had, and it is the order
/// a reader sees in `list` and in the refusal for an unknown model. It is
/// NOT alphabetical: it groups the embedders, then the speech models, then
/// the classifier, which is the order a reader who is looking for "the
/// text one" scans in.
pub fn catalogue() -> &'static [CatalogueEntry] {
    CATALOGUE
}

/// One row, by name.
pub fn find(name: &str) -> Option<&'static CatalogueEntry> {
    CATALOGUE.iter().find(|e| e.name == name)
}

/// The names of every model this build knows.
///
/// The order is the table's order, and the refusal lists them in it.
pub fn names() -> Vec<&'static str> {
    CATALOGUE.iter().map(|e| e.name).collect()
}
