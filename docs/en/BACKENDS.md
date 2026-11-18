# Backends

The runtime separates **arithmetic backends** and **array/NPU operation paths**:

- **CPU** (`backends/cpu`): host reference numerics/kernels.
- **Vulkan** (`backends/vulkan`): GPU compute via WGSL/`ash`. Reachable from CLI as `--backend vulkan` (all supported ops) or `--vk-ops gemm,layn,gelu,softm` (selective offload). Unsupported ops/families/paths refuse by name.
- **XDNA/NPU** (`backends/xdna`): array dispatch and transports (e.g. direct/xrt as built). Accessed via `--npu-ops` where implemented for a family.

`--npu-ops` (array execution mode) and `--backend` (arithmetic backend for non-array ops/paths) are independent axes; combinations are accepted only where supported.

## Status (high level)

- CPU: reference path for implemented families.
- Vulkan: real device execution for supported ops (gemm/gelu/layernorm/softmax/attn paths as wired; CLI-selectable subset `gemm,layn,gelu,softm`). Remaining work includes `conv/mproj/fft/logit` dispatch, staging/batching optimizations, and broader family/server coverage where required.
- XDNA: array paths implemented for selected projections and convolution families where ported (including pose/hands/mppose array conv paths in validated cases). Direct-build no-XRT/libxrt/libpython linkage constraints are enforced by `tools/verify/verify_no_xrt.py`; it can also perform an optional runtime run when hardware and a container are provided.

## Embed server

The embeddings server can run with Vulkan ops (`--vk-ops` / `--backend vulkan`). Other server engines currently run host-only and refuse unsupported backend requests by name.
