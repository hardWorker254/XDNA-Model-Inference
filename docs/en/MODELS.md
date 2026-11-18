# Models and weights layout

## Container format

- Primary: `.npum` (NPU Model). Legacy `.npue` is accepted by readers; writer emits `.npum`. Layout/semantics are preserved across the magic rename except where explicitly documented.
- Containers carry weights, config/arch metadata, and (for array paths) pre-tiled panels/design-related data as required.

## Sources and packing

- Transformer-family weights are sourced from safetensors where available (per migration state); ONNX remains relevant for vision/graph sources and fallbacks.
- Packing is performed by Python tools (`tools/pack/pack_npue.py`, `npue_tools/`).
- Example (host-side f32 container, comparable anywhere):

```sh
python tools/pack/pack_npue.py --model-dir models/<name> \
  --out models/<name>.npum --dtype f32 --no-embed-artifacts
```

## Model lookup

By name, the CLI/server resolves models in:
1. `<root>/models/<name>.npue|.npum`
2. `<root>/weights/<family>/<name>/` (e.g. `model.npue`, `<name>.npue`, or a container inside the directory)

By path, the container file is used exactly as provided.

## Local vs required artifacts (source of truth)

- **Source of truth in git:** code, docs, configs, manifests/scripts (e.g. `tools/parity/fixtures/*` inputs as provided), and small metadata needed to reproduce workflows.
- **Generated/local caches (not source of truth):** `target/`, `.venv/`, `.wheel-cache/`, large downloaded model weights under `models/**` except explicitly whitelisted metadata/config/tokenizer files tracked by policy.
- **Parity goldens:** `tools/parity/golden/` are contract artifacts; treat updates as reviewed, intentional changes.

When adding or updating models, document required files, generation steps, and whether artifacts are mandatory for CI/dev smoke vs full local/NPU validation.
