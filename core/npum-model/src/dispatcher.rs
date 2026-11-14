//! The dispatcher: where the "who computes this" decision is applied.
//!
//! # Three policies, and the flag that picks between them
//!
//! | flag | what it means |
//! |---|---|
//! | `--backend xdna\|vulkan\|cpu` | the whole model on one backend |
//! | `--npu-ops gelu,layernorm` | a mixed mode: the listed operations go to the array, the rest stay on the host |
//! | `--vulkan-ops attn,softmax` | new: the GPU takes what the array does badly |
//! | `--ops-placement auto` | the backend decides, from its capabilities and the profile |
//!
//! The mixed mode is the C++ semantics, kept 1:1, because a script with
//! `--npu-ops` in it means something specific and that meaning does not
//! change because the runtime was rewritten.
//!
//! # The placement is decided ONCE
//!
//! Not per operation at run time: the dispatcher builds the placement table
//! when the model is prepared, and the walk afterwards only reads it. A
//! decision that could change mid-walk would make a run's arithmetic depend
//! on how far it had got, which is the shape of a bug that reproduces once a
//! week and never on purpose.

use crate::caps::OpKind;
use crate::{Backend, Plan};

/// Where one operation runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// On the array.
    Npu,
    /// On the GPU.
    Vulkan,
    /// On the host.
    Host,
}

impl Placement {
    /// The placement's name.
    pub fn as_str(self) -> &'static str {
        match self {
            Placement::Npu => "npu",
            Placement::Vulkan => "vulkan",
            Placement::Host => "host",
        }
    }
}

/// The dispatcher: a backend, a placement table, and the walk.
pub struct Dispatcher {
    backend: Box<dyn Backend>,
    placement: Vec<(OpKind, Placement)>,
    plan: Plan,
}

impl Dispatcher {
    /// Build a dispatcher over one backend.
    ///
    /// The placement table is built from the backend's plan and the
    /// operation lists the flags carried. An operation in neither list runs
    /// on the host, which is the C++ default and the safe one: a backend
    /// that was not asked to do something is not asked to do it.
    pub fn new(backend: Box<dyn Backend>, plan: Plan, npu_ops: &[OpKind], vulkan_ops: &[OpKind]) -> Self {
        let mut placement = Vec::new();
        for op in OpKind::ALL {
            let where_ = if vulkan_ops.contains(&op) {
                Placement::Vulkan
            } else if npu_ops.contains(&op) && plan.runs(op.as_str()) {
                Placement::Npu
            } else {
                Placement::Host
            };
            placement.push((op, where_));
        }
        Self {
            backend,
            placement,
            plan,
        }
    }

    /// The backend this dispatcher drives.
    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    /// The plan the backend returned.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Where one operation runs.
    pub fn placement_of(&self, op: OpKind) -> Placement {
        self.placement
            .iter()
            .find(|(k, _)| *k == op)
            .map(|(_, p)| *p)
            .unwrap_or(Placement::Host)
    }

    /// The whole placement table.
    pub fn placement_table(&self) -> &[(OpKind, Placement)] {
        &self.placement
    }
}
