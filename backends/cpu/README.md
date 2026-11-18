# `npum-cpu`

CPU backend crate for NpuEmbeddings.

## Responsibility (current)

This crate currently combines:

1. **Arithmetic/backend core (primary backend responsibility)**
   - Low-level kernels and CPU arithmetic (`src/kernels.rs`)
   - Backend/ops implementations (`src/ops.rs`, `CpuBackend` in `src/lib.rs`)
   - Reference numerics used for cross-backend comparison/parity

2. **Reference family pipelines (runtime/reference execution layer)**
   - Family/model execution modules under `src/pipelines/` (BERT/GTE/ViT/Whisper,
     pose/hands/mppose/OCR, decode/session/tokenizer helpers)

3. **Input/frontend helpers**
   - Media/input preprocessing modules under `src/frontend/`
     (e.g. image decode/preprocess, WAV/mel frontend)

The public API is preserved for compatibility (e.g. `npum_cpu::bert`,
`npum_cpu::whisper_features`), while the internal module layout now makes the
layer boundaries explicit.

## Intended direction

- Keep CPU kernels/ops/ISA-specific arithmetic as the backend core.
- Treat family pipelines and frontends as higher-level reference/runtime layers
  that may be split into separate crates/modules in a follow-up refactor to
  reduce dependency weight for consumers that only need arithmetic.
