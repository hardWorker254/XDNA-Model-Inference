# Building (Rust tree)

This document describes how to build and validate the Rust runtime in this worktree.

## Supported scenarios

| Scenario | What you need | What you get |
|---|---|---|
| **Rust-only build** | Rust toolchain (see `rust-toolchain.toml`), `cargo` | `cargo build --release` for host-side crates/binaries (subject to enabled features/deps in your environment) |
| **Rust + Python tooling** | Python 3, venv + pinned deps | Packing, reference/verify/parity scripts under `tools/`, `reference/`, `npue_tools/` |
| **Full NPU/export/parity environment** | Above + AMD XDNA driver/hardware, design artifacts, and (for some export flows) vendor wheels matching the bootstrap matrix | NPU execution paths, design-aware parity, export/packing flows that depend on vendor runtimes |

Use the smallest scenario that matches the change. Heavy model/NPU/parity checks are not required for every Rust-only edit but are required when behavior/contracts on those paths change.

## Prerequisites

- Rust toolchain pinned by [`rust-toolchain.toml`](../../rust-toolchain.toml) (`1.85` with `rustfmt`, `clippy`).
- Python 3 for tooling (`tools/`, `reference/`). `bootstrap.sh` currently targets the pinned AMD/vendor wheel matrix (notably Python **3.14 / cp314** for specific wheels) and verifies checksums where defined.
- For NPU paths: supported AMD XDNA1/XDNA2 hardware/driver and compatible design artifacts as required by the runtime (fail-closed on mismatch).
- For Vulkan backend: a working Vulkan ICD/device.

## Bootstrap (Python tooling)

```sh
./bootstrap.sh
```

`bootstrap.sh` creates a local venv and installs pinned dependencies/wheels for the supported tooling/export environment. If your interpreter or platform does not match the pinned wheel matrix, bootstrap will fail fast; in that case use a Rust-only workflow or provide a compatible environment for the tooling you need.

## Build

```sh
cargo build --release
```

Artifacts:
- `target/release/npuembeddings`
- `target/release/npuaudio`
- `target/release/npuimage`

## Formatting and linting

```sh
cargo fmt --all
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

Formatting/lint baselines are being tightened; run them for changes you touch and fix regressions introduced by your edits. Prefer not to reformat unrelated files in the same change unless you are intentionally normalizing the tree.

## Validation

- **Rust tests:** `cargo test --workspace` (or `--release` as appropriate).
- **Parity with C++ oracle:** `python3 tools/parity/run.py compare --candidate ./target/release` (requires recorded goldens and installed models/fixtures for non-skipped cases). See [PARITY.md](PARITY.md).
- **No-XRT gate:** `python3 tools/verify/verify_no_xrt.py` (when the script and relevant build context are available).

## Repositories / worktree

This tree may be a `git worktree` of the oracle repository. Oracle binaries, plans, and some tools/docs may live in the sibling/oracle tree; use absolute paths to oracle artifacts when comparing.
