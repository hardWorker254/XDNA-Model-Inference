# Usage

## Binaries and families
- `npuembeddings` — text embeddings: `list`, `add`, `tokenize`, `embed`, `serve` (`/v1/embeddings`).
- `npuaudio` — speech-to-text (Whisper): `transcribe`, `serve` (`/v1/audio/transcriptions`).
- `npuimage` — vision: `classify`, `pose`, `hands`, `mppose`, `ocr`, `serve` (`/v1/classify`, `/v1/pose`, etc. as per model/arch).

The server endpoint/mode is selected by the container `arch`, not by a flag. A container served by the wrong binary/family is refused by name.

## Model resolution
A model argument can be:
- a path to a container (`.npue`/`.npum`, or a path with separators) — used as-is;
- a model name — looked up in `<root>/models/<name>.npue|.npum`, then in `weights/<family>/<name>/` (e.g. `weights/<family>/<name>/model.npue` or `<name>.npue`).

`<root>` is the runtime root (see `--help`/CLI behavior).

## Common flags (selection)
- `--backend cpu|vulkan` — arithmetic backend (where supported).
- `--vk-ops gemm,layn,gelu,softm` — selective Vulkan offload (implies Vulkan); may be combined with supported paths. Unsupported families/paths refuse by name.
- `--npu-ops ...` — NPU/array operation path (where implemented). `--npu-ops` and `--backend` are different axes (array path vs arithmetic backend).

Flag parsing follows the project rule: the last occurrence wins.

## Examples
```sh
./target/release/npuembeddings embed models/all-MiniLM-L6-v2.npue in.txt out.f32
./target/release/npuembeddings serve models/all-MiniLM-L6-v2.npue --port 8080 --bind 127.0.0.1
./target/release/npuembeddings serve models/all-MiniLM-L6-v2.npue --vk-ops gemm
./target/release/npuimage pose models/yolov8n-pose.npue image.jpg --npu-ops conv
```

(Exact flags and availability depend on the family and build; unsupported combinations fail by name rather than silently falling back.)
