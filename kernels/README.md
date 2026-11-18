# AIE kernels (`kernels/`)

This directory contains C++ kernel sources for the AMD XDNA / AIE execution path
(`*.cc`, `*.h`). They implement low-level operators (e.g. GELU, LayerNorm/RMSNorm,
Softmax, SiLU, narrowing/cast helpers, shared polynomial helpers) used by the
NPU design/export pipeline.

## Role in the repository

- **Source of truth for exported NPU operator kernels:** these files are inputs to
  the design/export toolchain (see `tools/export/`, especially
  `tools/export/exporters/` and `tools/export/export_gemm_rtp.py`) that produces
  runtime artifacts/design sets consumed by `backends/xdna/`.
- **Not the host CPU reference implementation:** host/reference numerics live in
  `backends/cpu/` (Rust). Kernel behavior here must stay aligned with the
  corresponding model/operator contracts and parity gates.
- **Not general application code:** files here are compiled/packaged as part of
  NPU artifact generation, not as part of the normal Rust workspace build.

## Lifecycle (high level)

1. Kernel sources in `kernels/` are referenced by export/design generation tools in
   `tools/export/`.
2. The export pipeline builds/assembles AIE/IRON designs and emits artifacts (e.g.
   xclbin/instruction streams and related metadata) for target devices.
3. `backends/xdna/` loads and dispatches those artifacts at runtime; CLI/server
   select the backend/path at the edge.

## When changing kernels

- Treat changes as operator/precision-affecting unless proven otherwise.
- Re-export affected design artifacts and run the relevant verification/parity
  gates (see `tools/README.md` and `docs/en/PARITY.md` / `docs/ru/PARITY.md`).
- If a kernel change intentionally changes numerics or behavior, record that in
  the appropriate contract/divergence documentation rather than silently updating
  goldens.
