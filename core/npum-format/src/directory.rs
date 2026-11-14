//! The directory: the container's index.
//!
//! One JSON object, written compactly (no spaces, keys in insertion order),
//! that names every tensor and carries the config block:
//!
//! ```json
//! {"config": {...}, "tensors": [
//!   {"name": "...", "role": "...", "dtype": "BF16",
//!    "logical_shape": [...], "padded_shape": [...],
//!    "offset": 0, "nbytes": 123, "layout": {...}, "layout_hash": "..."}
//! ]}
//! ```
//!
//! # Why the shapes come in two
//!
//! `logical_shape` is what the mathematics means; `padded_shape` is what the
//! bytes are. A GEMM operand whose N was padded up to a legal tile width has
//! a logical shape of 384 and a padded shape of 384 -- or of 512, when 384
//! does not tile. Both shapes are recorded because a reader that used the
//! padded one as the logical one would compute the right numbers for the
//! wrong question, and a reader that used the logical one to size a copy
//! would read past the end of the tensor.

use serde::{Deserialize, Serialize};

/// The container's config block.
///
/// Free-form JSON, kept as JSON rather than as a struct of named fields. The
/// packer writes keys the families read -- `arch`, `n_layers`, `hidden_size`,
/// the tokenizer's tables -- and a struct here would be a second place where
/// the list of keys lives, to be kept in step with the packer's.
///
/// # It is TRANSPARENT
///
/// The config block IS a JSON object in the file, and this type reads it as
/// one: `{"arch": "..."}` in the container is `{"arch": "..."}` here, with no
/// wrapper object around it. A wrapper would mean the container's config
/// and the Rust struct's shape have to agree, and the shape of a free-form
/// config block is not something to pin down twice.
///
/// The `Deserialize` is written out rather than derived, because
/// `#[serde(transparent)]` and `#[serde(flatten)]` do not compose the way
/// they look like they should: `flatten` collects the REMAINING keys of a
/// map, and a `Value` has no notion of "the rest of the map".
#[derive(Debug, Clone, Default, Serialize)]
#[serde(transparent)]
pub struct Config {
    /// The config itself.
    pub fields: serde_json::Value,
}

impl<'de> Deserialize<'de> for Config {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self {
            fields: serde_json::Value::deserialize(deserializer)?,
        })
    }
}

impl Config {
    /// A config from a JSON object.
    pub fn from_value(value: serde_json::Value) -> Self {
        Self { fields: value }
    }

    /// One key, as JSON.
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.fields.get(key)
    }

    /// One key, as a string.
    ///
    /// The absence of a key is `None` and not an empty string: an empty
    /// `arch` and a missing `arch` are different facts, and the second one is
    /// what a pre-arch container looks like.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.fields.get(key).and_then(|v| v.as_str())
    }

    /// One key, as an integer.
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.fields.get(key).and_then(|v| v.as_i64())
    }

    /// One key, as a boolean.
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.as_f64())
    }

    /// Whether a key is there at all.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.fields.get(key).and_then(|v| v.as_bool())
    }

    /// One key, as an OBJECT.
    ///
    /// The absence of the key is `None` and not an empty object: a config
    /// with an empty `prompts` table and a config with NO `prompts` table
    /// are different facts, and the second one is what a model with no task
    /// prefixes looks like.
    pub fn get_object(&self, key: &str) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.fields.get(key).and_then(|v| v.as_object())
    }
}

/// One tensor's entry in the directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    /// The tensor's name. For a design set's files this is
    /// `design/<set>/<file>` -- the set name is part of the name because a
    /// self-sufficient container carries four sets, and a flat `final.xclbin`
    /// in a container holding four of them would be four files of one name.
    pub name: String,
    /// The role: `weight`, `bias`, `design`, and so on.
    #[serde(default)]
    pub role: String,
    /// The dtype tag: `F32`, `I32`, `I64`, `U16`, `BF16`, `I8`.
    pub dtype: String,
    /// The shape the mathematics means.
    pub logical_shape: Vec<i64>,
    /// The shape the bytes are.
    #[serde(default)]
    pub padded_shape: Vec<i64>,
    /// The byte offset inside the data region.
    pub offset: u64,
    /// How many bytes the payload is.
    pub nbytes: u64,
    /// The pre-tiled layout descriptor, for a B panel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<serde_json::Value>,
    /// The hash of `layout`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout_hash: Option<String>,
}

/// The whole directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Directory {
    /// The config block.
    #[serde(default)]
    pub config: Config,
    /// The tensor index, in write order.
    pub tensors: Vec<TensorEntry>,
}

impl Directory {
    /// One tensor's entry, by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Every tensor's entry, in directory order.
    pub fn tensors(&self) -> &[TensorEntry] {
        &self.tensors
    }

    /// The config block.
    pub fn config(&self) -> &Config {
        &self.config
    }
}
