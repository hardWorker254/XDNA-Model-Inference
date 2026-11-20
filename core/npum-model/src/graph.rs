//! The model graph: what a container says a model is, and what it does.

use crate::caps::OpKind;
use crate::tensor::DType;
use npum_format::Container;
use std::path::{Path, PathBuf};

/// What a container says about the model it holds.
///
/// This is the model's IDENTITY -- its architecture, its geometry, its
/// tokenizer -- and not its weights. The weights live in the container's
/// tensor index; the spec is what the runtime reads to decide whether it can
/// run the model at all.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Where the container is.
    pub path: PathBuf,
    /// The container's architecture id.
    pub arch: u32,
    /// The architecture's name, as the container spells it.
    pub arch_name: String,
    /// The number of transformer layers.
    pub layers: usize,
    /// The hidden size.
    pub hidden: usize,
    /// The number of attention heads.
    pub heads: usize,
    /// The position table's length.
    pub max_seq: usize,
    /// The pooling: `mean`, `cls`, or none.
    pub pooling: Option<String>,
    /// The task prefixes the container's prompt table holds.
    pub prompts: Vec<String>,
}

impl ModelSpec {
    /// Read a container's spec.
    pub fn read(path: &Path) -> crate::Result<Self> {
        let container = Container::open(path).map_err(|source| crate::Error::Container {
            path: path.to_path_buf(),
            source,
        })?;
        let arch_name = container.config_str("arch").unwrap_or_default().to_string();
        let arch = container.header().arch;
        let layers = container
            .config()
            .get_i64("n_layers")
            .or_else(|| container.config().get_i64("num_hidden_layers"))
            .unwrap_or(0) as usize;
        let hidden = container
            .config()
            .get_i64("hidden_size")
            .or_else(|| container.config().get_i64("n_embd"))
            .unwrap_or(0) as usize;
        let heads = container
            .config()
            .get_i64("n_heads")
            .or_else(|| container.config().get_i64("num_attention_heads"))
            .unwrap_or(0) as usize;
        let max_seq = container
            .config()
            .get_i64("max_seq_len")
            .or_else(|| container.config().get_i64("max_seq"))
            .or_else(|| container.config().get_i64("max_position_embeddings"))
            .unwrap_or(0) as usize;
        let pooling = container.config_str("pooling").map(|s| s.to_string());
        let prompts = container
            .config()
            .get("prompts")
            .and_then(|v| v.as_object())
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();
        Ok(Self {
            path: path.to_path_buf(),
            arch,
            arch_name,
            layers,
            hidden,
            heads,
            max_seq,
            pooling,
            prompts,
        })
    }
}

/// One operation in a graph.
#[derive(Debug, Clone)]
pub struct Op {
    /// Which operation.
    pub kind: OpKind,
    /// The operation's name, for diagnostics.
    pub name: String,
    /// The operation's operands.
    pub inputs: Vec<String>,
    /// Where the result goes.
    pub output: Option<String>,
    /// The operation's parameters.
    pub attrs: serde_json::Value,
}

/// A model graph: the operations, in order.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    /// The operations.
    pub ops: Vec<Op>,
}

impl Graph {
    /// The operations of one kind.
    pub fn of_kind(&self, kind: OpKind) -> impl Iterator<Item = &Op> {
        self.ops.iter().filter(move |op| op.kind == kind)
    }
}

/// The dtype a tensor entry holds, from its directory entry.
pub fn dtype_of(entry: &npum_format::TensorEntry) -> Option<DType> {
    DType::parse(&entry.dtype)
}
