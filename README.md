# NpuEmbeddings -- the Rust runtime

Sentence embeddings for BERT-family models on an AMD NPU, with the host CPU
as a fallback path. Runs `serve` (an OpenAI-shaped HTTP endpoint) and `embed`
(a file in, a file out), and the same for speech, vision, pose and OCR.

This tree is the **Rust rewrite**. The C++ runtime it replaces lives in the
sibling tree and stays there for now: it is the parity oracle, and a rewrite
without its oracle is a rewrite with nobody to argue with.

---

## Where the two trees are

| | this tree | the oracle |
|---|---|---|
| path | `~/XDNA-Model-Inference` | `~/Npu-Embeddings-XDNA1` |
| branch | `rust/clean-tree` | `main` |
| runtime | Rust | C++ |
| role | what is being built | what the answers are checked against |

The two are the same git repository: this tree is a `git worktree` of the
other, so a change here and a change there are two branches of one history,
not two forks of one project.

**The oracle's full documentation -- the 124 KB README, the build guide, the
accuracy notes -- is in the old tree.** This README says what is true of
*this* tree, and points at the rest.

---

## The layout

```
backends/cpu        the host kernels -- the reference numerics
backends/vulkan     the shaders
backends/xdna       the array: transports, dispatch, geometry
core/npum-format    the container: .npum
core/npum-model     tensors, graphs, the Backend trait
core/npum-tokenizers  wordpiece, unigram, the whisper and gemma tables
core/npum-families  the seven families, the catalogue, the refusals
cli                 the three executables
server              the eight HTTP routes
```

**The dependency rule, and it is a rule and not an accident:** `core/` knows
nothing about `backends/`, and `backends/` know nothing about `cli/` or
`server/`. The graph walks itself and calls a trait; which implementation
answers is chosen at the edge, by the CLI. A core crate that could name a
backend would be a core crate that could not be tested without one.

---

## Building

```sh
./bootstrap.sh          # the Python venv, with pinned dependencies
cargo build --release
```

The three binaries land in `target/release/`:

```
npuembeddings   embed, list, add, tokenize, serve
npuaudio        transcribe, serve
npuimage        classify, pose, hands, mppose, ocr, serve
```

---

## The contract

The runtime's behaviour is a contract, and it is written down:

* **[docs/en/INVARIANTS.md](docs/en/INVARIANTS.md)** -- the rules the runtime
  does not break, and why each one is there.
  [По-русски](docs/ru/INVARIANTS.md).

A change that breaks an invariant is not a change, it is a regression.

---

## The parity gate

`tools/parity/` records what the C++ runtime does and checks that the Rust
runtime does the same.

```sh
# record the goldens, from the oracle
python tools/parity/run.py record --oracle ~/Npu-Embeddings-XDNA1/runtime/build

# check the Rust binaries against them
python tools/parity/run.py compare --candidate target/release
```

A case whose model is not installed is **skipped**, and the skip is printed
beside the count of the ones that ran: a run that compared nothing must not
be mistaken for a run that passed everything.

A case that differs is a **failure**, and the first difference is where the
reader starts. If the difference is intended, it is declared in
`tools/parity/divergences.txt` with its reason -- never by re-recording the
golden as a side effect of the change.

---

## The container format

`.npum` -- NPU Model. It was `.npue` once, which stood for "NPU embedding"
and stopped being true when the runtime grew an audio, a vision, a pose and
an OCR family.

The LAYOUT did not change when the name did -- only the four magic bytes --
so the reader accepts both magics and the writer emits only `NPUM`.

---

## The models

The weights are an ONNX export placed by hand beside the config the runtime
fetches. "A checkout should never have to trust a binary it did not choose."

```sh
tools/parity/fixtures/fetch_models.sh        # the pinned download manifest
python tools/pack/pack_npue.py --model-dir models/all-MiniLM-L6-v2 \
    --out models/all-MiniLM-L6-v2.npue --dtype f32 --no-embed-artifacts
```

`--dtype f32` produces a row-major host-side container: no design set, no
array, and the arithmetic on the CPU. It is the container that can be built
and compared anywhere.

---

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [THIRD-PARTY.md](THIRD-PARTY.md).
