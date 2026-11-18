# `npum-model`

Core model/runtime abstractions: tensors, graph/spec types, capability/dispatch
metadata, and the arithmetic seams used by backends and model families.

## Key abstractions

### `Backend` (tensor-oriented execution contract)
Defined in `src/lib.rs` as `npum_model::Backend`.

Use this when implementing or selecting a **backend/provider** that prepares and
executes model operations in tensor form (e.g. CPU/Vulkan/XDNA backend
implementations, graph/op dispatch paths).

- Owns preparation/planning (`prepare`) and tensor-level operations (`gemm`,
  `gelu`, etc.).
- Exposed from the core crate so both model/graph code and backend crates share
  the same trait identity.

### `Ops` (slice-oriented family arithmetic seam)
Defined in `src/ops.rs` as `npum_model::Ops`.

Use this for **family/reference forward passes** that already operate on flat
buffers/slices with explicit dimensions and need a pluggable arithmetic provider
without tensor packing/copies per op.

- Methods take flat slices (`&[f32]` / `&mut [f32]`) and explicit shape params
  (`gemm_nt`, `layernorm_rows`, `gelu`, `softmax_rows`).
- Reports execution placement/dispatch accounting (`place`, `device`,
  `dispatches`) so callers can attribute work correctly.

### `array` / `dispatcher` / `caps`
Support types for array-capable operations, placement decisions, and backend
capability description. Use these to route work and express what a backend can
honor rather than encoding backend-specific assumptions in families.

## Choosing the canonical layer

- **Implementing a backend or tensor/graph execution path:** start with
  `Backend` (+ `Caps`/`Plan`/`OpKind`).
- **Implementing or porting a family/reference forward pass over host/device
  buffers:** prefer `Ops` as the arithmetic seam.
- **Routing between host and array/device implementations:** use `dispatcher` and
  placement/capability types; keep the "where it ran" claim truthful.
- **Do not duplicate logic across layers:** if a family needs slice-level math,
  add/route through `Ops`; if the path is tensor/graph-driven, use `Backend`.
  Avoid parallel ad-hoc code paths for the same operation semantics.

This separation is intentional: tensor-level backend execution and slice-level
family seams optimize for different data shapes and copy/dispatch costs.
