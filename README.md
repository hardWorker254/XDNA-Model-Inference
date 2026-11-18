# NpuEmbeddings

NpuEmbeddings is a Rust workspace for running model inference on AMD NPUs (XDNA1/npu1, XDNA2/npu2) with a CPU reference path and optional Vulkan acceleration for supported operations.

The project provides three command-line binaries, a shared HTTP server, model/container tooling, and documentation for building, usage, backends, and validation.

## Binaries

| Binary | Main commands | Purpose |
| --- | --- | --- |
| `npuembeddings` | `list`, `add`, `tokenize`, `embed`, `serve` | Text embedding workflows |
| `npuaudio` | `transcribe`, `serve` | Speech-to-text workflows |
| `npuimage` | `classify`, `pose`, `hands`, `mppose`, `ocr`, `serve` | Vision workflows |

## Workspace layout

```text
backends/cpu/         CPU kernels, arithmetic, and reference execution paths
backends/vulkan/      Vulkan backend
backends/xdna/        NPU transport/dispatch and device-specific execution
core/npum-format/     Container format (.npum/.npue): read/write, digests, metadata
core/npum-model/      Tensors, model/runtime abstractions, backend/ops contracts
core/npum-tokenizers/ Tokenizers
core/npum-families/   Model families, catalogue, and routing/validation rules
cli/                  CLI binaries and command dispatch
server/               HTTP server and routes
tools/                Packing, export, verification, generation, and helper scripts
docs/                 User and developer documentation (EN/RU)
kernels/              AIE/NPU kernel sources used by the export/build pipeline
models/               Local model metadata/containers (large weights are typically local-only)
```

Dependency direction (by design): `core/` does not depend on `backends/`, and `backends/` do not depend on `cli/` or `server/`. Backend/family selection happens at the edges (CLI/server).

## Build

Prerequisites depend on which paths you use (CPU-only, Vulkan, NPU/export toolchain). For the Python tooling environment used by scripts in `tools/`, the repository includes a bootstrap script.

```bash
./bootstrap.sh
cargo build --release
```

Built binaries are placed in `target/release/`:

- `npuembeddings`
- `npuaudio`
- `npuimage`

## Usage (overview)

- Models/containers can be passed by path (`.npum` or legacy `.npue`) or, where supported, resolved by name from configured model locations.
- The HTTP server mode/endpoints are determined by the container/model family (not by a separate "mode" flag).
- Backend and operation routing options (CPU/Vulkan/NPU operation selection) depend on the command, family, and available artifacts/capabilities.

For command details and supported flags, use `--help` on each binary/subcommand and refer to the documentation in `docs/`.

## Documentation

- [docs/en/BUILDING.md](docs/en/BUILDING.md) / [docs/ru/BUILDING.md](docs/ru/BUILDING.md) — build and validation
- [docs/en/USAGE.md](docs/en/USAGE.md) / [docs/ru/USAGE.md](docs/ru/USAGE.md) — CLI/server usage and model resolution
- [docs/en/BACKENDS.md](docs/en/BACKENDS.md) / [docs/ru/BACKENDS.md](docs/ru/BACKENDS.md) — backend selection and capabilities
- [docs/en/MODELS.md](docs/en/MODELS.md) / [docs/ru/MODELS.md](docs/ru/MODELS.md) — containers, models, and weights layout
- [docs/en/PARITY.md](docs/en/PARITY.md) / [docs/ru/PARITY.md](docs/ru/PARITY.md) — validation/parity checks
- [docs/en/INVARIANTS.md](docs/en/INVARIANTS.md) / [docs/ru/INVARIANTS.md](docs/ru/INVARIANTS.md) — runtime invariants and contracts
- [tools/README.md](tools/README.md) — tooling overview

## Container format

The runtime container format is `.npum` (NPU Model). Legacy `.npue` containers are accepted for compatibility; writers emit `.npum`.

## License

Apache-2.0. See [LICENSE](LICENSE), [THIRD-PARTY.md](THIRD-PARTY.md), and [PROVENANCE.md](PROVENANCE.md).
