# The parity harness

<!-- SPDX-License-Identifier: Apache-2.0 -->

It records what the C++ runtime does and checks that the Rust runtime does
the same.

```sh
# record the goldens, from the oracle
python tools/parity/run.py record --oracle ~/Npu-Embeddings-XDNA1/runtime/build

# check the Rust binaries against them
python tools/parity/run.py compare --candidate target/release

# the case table, and the normalisation rules
python tools/parity/run.py list -v
python tools/parity/run.py rules
```

---

## What a golden is

For each case, a directory under `golden/<case-id>/` holding:

| file | what |
|---|---|
| `stdout` | what the process wrote to stdout |
| `stderr` | what it wrote to stderr |
| `code` | the exit code |
| `body` | the HTTP response's body, for the http cases |
| `status` | the HTTP response's status code, for the http cases |
| `meta.json` | the case's parameters, and the rules that fired |

`meta.json` is what makes a golden REVIEWABLE: it says which binary wrote
it, on what date, and which normalisation rules were applied. A golden
without it is a mystery file.

---

## The rules of the harness

### The oracle is the oracle

`record` runs the **C++** binaries and writes the goldens. `compare` runs the
**Rust** binaries and diffs. They are the same machinery with a different
binary directory, and that is the point: a difference between the two
transcripts is a difference between the two runtimes, and not a difference
between two harnesses.

### Every run gets the SAME root

Each run is given `--root <the fixture root>`, which holds one set of
containers.

It is not inferred: the oracle's default root is the tree its BINARY lives
in, and the candidate's default root is the tree its binary lives in, and
those are two different directories holding two different containers. A
harness that let each side use its own root would be comparing two different
models and calling the difference a mismatch.

The fixture root is `tools/parity/fixtures/root/`, and its `models/` is a
SYMLINK to the repository's `models/`, so the packed containers are used
where they are and not copied by the hundred megabytes.

### The network is not used

`HF_HUB_OFFLINE=1` is set for every run, and the cases that need the network
are tagged `network` and skipped unless `NPUE_PARITY_NETWORK=1` is set.

A golden that depends on what HuggingFace answered that day is a golden of
the internet, not of the runtime.

### A skip is not a pass

A case is SKIPPED when:

* **its model is not installed** -- a golden of a refusal that says "there
  is no model" is a golden of the ABSENCE, and it would be matched by any
  implementation that also refuses;
* **it needs a compiled design set** -- the C++ runtime asks for one before
  it runs the embedding, the classification and the transcription paths, and
  it asks even for a container whose operands are row-major host-side values
  and whose every operation would run on the CPU. A design set is the
  OUTPUT of MLIR-AIE: it is a compiled AIE program with an xclbin and
  instruction binaries beside a design.json. There is no way to fetch one
  and no way to make one without the toolchain;
* **it needs the network** -- see above.

The number of skips is printed beside the number of passes, so that a run
which compared nothing cannot be mistaken for a run which passed everything.

### A failure is a difference, not a golden that needs updating

If a golden has to change, the change says what changed and why. A golden
that is updated as a side effect of a run is not a golden.

If the difference is intended AND it will stay that way, declare it in
`divergences.txt` with its reason. A declared divergence is still printed
on every run, with the diff. It just does not fail the run.

### The normalisation rules are named, finite, and printed

A rule that is not on the list is not applied, and a difference that is not
on the list is a failure. `run.py rules` prints them.

They forgive: durations, ports, temporary paths, process ids, the container
extension rename (`.npue` → `.npum`), and the absolute paths of the two
trees. They forgive nothing else.

---

## The fixture root

```
tools/parity/fixtures/
  in.txt            the text the embed and tokenize cases use
  bus.jpg           the image the vision cases use
  ocr_text.png      the image the OCR case uses
  tone.wav          the audio the transcribe case uses
  root/             THE ROOT: models/ (a symlink) and runtime/artifacts/
  fetch_models.sh   the pinned download manifest
```

`fetch_models.sh` fetches the ONNX exports the packer needs, pinned by
sha256 where the repository records one. It is a separate script because it
is a download and the rest of the harness must never touch the network.

---

## What the oracle cannot do

Some of this is the harness's fault and some of it is the machine's, and
both are worth writing down:

* **`add` crashes.** The C++ runtime ABORTS inside glibc ("invalid stdio
  handle") when the network path runs on this machine. A crash is not a
  contract: it says what the process did, not what the runtime means. The
  Rust runtime refuses cleanly, and the difference is declared in
  `divergences.txt`.
* **`list` ignores `--root`.** The C++ runtime's `list` reports the tree its
  own binary lives in, whatever `--root` says. The Rust runtime honours
  `--root`, which is the fix. The `sub/list` case therefore only compares
  the two runtimes when the oracle's default root holds the same containers
  as the fixture root.
* **The embedding, classification and transcription paths need a design
  set**, and there is none on this machine for those geometries. So those
  goldens are not recorded, and the eight HTTP endpoints are not either:
  the server loads the model and its design set BEFORE it binds the port,
  so a server with no design set never listens.

What IS recorded, and what does compare byte for byte today:

* the three `--help` texts and the no-argument behaviour;
* the `list` table, its states and its legend;
* `tokenize`, which is a real test of the wordpiece tokenizer;
* the refusals: an unknown option, a removed flag, a wrong binary, an
  unknown model, a missing argument, a container of another family, a
  `--serve=8080`;
* the flag semantics: the last occurrence wins, a flag after the
  positionals, an unknown `--npu-ops` code;
* the pose, hands, mppose and OCR transcripts, as goldens for the numerics
  work to come.
