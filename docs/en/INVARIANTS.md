# Invariants

<!-- SPDX-License-Identifier: Apache-2.0 -->

These are the rules the runtime does not break. A change that breaks one is
not a change, it is a regression, and the golden transcripts in
`tools/parity/golden/` are what say so.

Every invariant here has a reason, and the reason is written down. An
invariant without a reason is a superstition, and superstitions get broken
by people who do not know what they are protecting.

The first eight come from `RUST_MIGRATION_PLAN.md` §14. The rest were
found while the port was being made, and each of them is here because
breaking it produces an answer that LOOKS right.

---

## 1. The last occurrence of a flag wins, not the first

`--pipeline 4 --pipeline 1` disables pipelining.

This is not a preference. The C++ tree got it wrong once, and the symptom
was not an error: `serve <whisper> --threads 8` answered with 24 threads,
and the answer was a number, so nothing said that the flag had been
ignored.

## 2. `design_fits()` is fail-closed

A design set built for the other generation is refused, **even when the
geometry happens to agree**.

The alternative -- accepting a set whose tile dimensions coincide with the
device's -- is how a wrong-but-plausible answer arrives. There is no
version of this check that can be "a bit less strict" and still mean
anything.

## 3. The B panel's sub-tile is per-generation, and it is checked against the container

| | bf16 | int8 |
|---|---|---|
| npu1 / `aie2` | (8, 4) | (8, 8) |
| npu2 / `aie2p` | (8, 8) | (8, 8) |

A container packed for one board and read by a core built for the other is
**rejected nowhere**: the byte count agrees, the shapes agree, and both
sides derive the same `layout_hash` from the same wrong constant. The only
symptom is wrong numbers.

This is not hypothetical. A `bge-small` int8 design shipped with the bf16
pair on npu1, where the int8 MMAC's N sub-tile is 8 and bf16's is 4. The
exporter, the packer, the runtime's hash check and the numerics gate all
agreed with each other and all disagreed with the hardware, because all
four read the same wrong constant. End to end it read as 1-cos 8.6e-01 on
a model whose bf16 path is 1e-05.

So the sub-tile is a constant in one place, reached only through
`Geometry::for_generation`, and it is checked against what the container
says -- never against the name of a file.

## 4. The softmax mask is clamped BEFORE the widening to bf16, or `-inf` becomes NaN

Three clamps, each with its own reason:

1. **On load, in bf16, before the widening:** `xb = max(xb, -1e30)`.
   HuggingFace's additive mask is `(1 - mask) * finfo(f32).min`, which is
   finite in f32 -- but bf16's largest finite magnitude is 3.3895e38
   against f32's 3.4028e38, so the narrowing produces `-inf`. Clamping in
   f32 afterwards does not undo the NaN it already made: clamping before
   the widening left 386 NaNs, clamping before it did not.
2. **On the difference:** `d = max(x - m, -100)`. `exp(-100)` is 3.7e-44,
   which is zero in bf16 -- the right answer for a masked element.
3. **On the base-2 argument:** `arg = max(d * log2e, -120)`. The
   exponential is computed as `exp2_poly`, and a negative exponent below
   about -127 is a negative exponent FIELD -- a NaN bit pattern. Getting
   this wrong produced exactly 256 NaNs on the first attempt.

The CPU path applies the same three clamps. Not because it needs them: the
CPU path must compute the function the array computes, and a function that
differs in the clamped region is a different function.

## 5. The AVX2 path must match the scalar path, without contraction, bit for bit

Rust does not contract `a * b + c` into an FMA by default, and the C++
scalar path did not either. The gate
`runtime/tests/test_int8_host_kernels.cpp` proved the two paths agree bit
for bit under exactly that rule.

"Close" is not a property. A difference between two spellings of the same
arithmetic is a bug in one of them, and a tolerance cannot tell the reader
which.

## 6. A partially loaded model is refused BY NAME

A renamed module, a missing head, a shape that does not match -- these are
refusals that name what is wrong. They are not "plausible numbers".

## 7. A safetensors key is data, not code

It is read without `pickle`. A checkpoint is an input, and an input that
can execute code is not an input.

## 8. A failed test is a difference, not "the golden needs updating"

If a golden has to change, the change says what changed and why. A golden
that is updated as a side effect of a run is not a golden.

---

## 9. A container of another family is refused by name, and the refusal says which program to run it

```
models/whisper-tiny.npum is `whisper_encdec_gelu`, which npuaudio runs,
and this is npuembeddings.
    npuembeddings   embed, list, add, tokenize, serve
    npuaudio        transcribe, serve
    npuimage        classify, pose, hands, mppose, ocr, serve
```

Three lines, the same three lines every time, from one string. The roster
is a constant, so the help text and the refusals cannot drift apart about
which program to run next.

The family is decided TWICE and both answers are needed:

* from the SPELLING of the verb -- `transcribe` is Audio whatever the
  arguments say;
* from the ARCH of the container -- a Whisper container is Audio whatever
  the verb was.

## 10. The generation is read from the DEVICE, not from the environment

The C++ runtime read it from `NPU_DEVICE`, else `NPU2=1`, else npu1. Its
own comment said why:

> running device is not discoverable from XRT here

So the answer was "whatever the environment says", the environment said
nothing by default, and the default was npu1 -- on a machine running npu2.
That was not a bug that produced an error; it was a configuration that
produced plausible numbers.

`--dev` remains, as an OVERRIDE, and it is checked against what the device
says when the device can say. An override that contradicts the hardware is
a refusal.

## 11. `--serve [PORT]` takes a port only if the next argument starts with a digit, and `--serve=8080` is not accepted

The grammar has always been loose in exactly this way. Nothing has ever
documented `--serve=8080`, and accepting it now would be a new spelling
that nobody asked for.

ONE reader, too. The C++ tree had three hand-written scans of the same two
flags, and that is how `--serve` with no port came to mean port 0 in the
speech mode: `atoi("")` is 0, and 0 is a legal request to `bind()`, so the
server started, printed its banner and listened on an ephemeral port it
never named.

## 12. The removed flags are refused by name, and the refusal names the replacement

`--npu-eltwise`, `--host-gelu`, `--host-ln`, `--host-sm`, `--npu-extra-ops`
and `--extra-ops`.

They are listed in the flag TABLE, so the parser lets them through to the
refusal. A parser that refused them as "unrecognised" would be strictly
less useful on exactly the flags most likely to appear in a stale command
line -- the refusal should say what to use INSTEAD.

## 13. The reader accepts both container magics; the writer writes only the new one

`NPUM` is the current name. `NPUE` is the name from before the rename,
when the format stood for "NPU embedding" and the runtime had grown an
audio, a vision, a pose and an OCR family.

The LAYOUT did not change when the name did -- only the four magic bytes
-- so refusing the old magic would refuse every container that has not
been repacked, for no gain. The reader accepts both; the writer emits only
`NPUM`.

## 14. The layout hash is over the KEY-SORTED, compact JSON of the descriptor

The packer hashes `json.dumps(layout, sort_keys=True, separators=(",", ":"))`.
The Rust side must produce the same bytes, and the sort is what makes that
true: the struct's declaration order is not the sorted order.

If a hash starts disagreeing with the packer's, the first thing to check is
whether the `serde_json` `preserve_order` feature got switched on
somewhere -- it turns `Map` from a `BTreeMap` into an insertion-ordered
map, and the hash follows.

## 15. bf16 rounding is round-to-nearest-even, and it is implemented explicitly

Not taken from `std::bf16` until a gate has confirmed that the standard
library's semantics are the hardware's. The ties rule matters:
round-half-to-even and round-half-away differ on every exact .5, and a
container written with one and read with the other loses a bit of every
weight that lands on one.

## 16. The hardware context budget is `Option`, not a number

The C++ runtime's `context_budget()` returned a plain integer, and for npu2
it returned a number nobody had measured. The comparison then read as "the
budget is zero" and the process refused to run -- or, worse, the comparison
was skipped and the contention went unnoticed.

A budget that has not been measured is a budget that has not been measured,
and a runtime that says so is a runtime that can be trusted when it says
the budget IS known.

## 17. A case whose model is not installed is SKIPPED, not recorded

A golden transcript of a command that refuses because the model is missing
is a transcript of the ABSENCE. It would be matched by any implementation
that also refuses, and it would pin nothing.

The harness skips such a case, says which model is missing, and reports the
count of skips beside the count of passes -- so that a run that compared
nothing cannot be mistaken for a run that passed everything.

## 18. The golden transcripts are recorded against a FIXED root

Every run gets `--root <the fixture root>`, which points at one set of
containers.

It is not inferred: the oracle's default root is the tree its BINARY lives
in, and the candidate's default root is the tree its binary lives in, and
those are two different directories holding two different containers. A
harness that let each side use its own root would be comparing two
different models and calling the difference a mismatch.

## 19. The normalisation rules are named, finite, and printed

A rule that is not on the list is not applied, and a difference that is not
on the list is a failure. The rules are data, they are printed with the
comparison result, and a rule that changed silently would make every gate
weaker without anyone noticing.

The rules forgive: durations, ports, temporary paths, process ids, the
container extension rename, and the absolute paths of the two trees.

They forgive nothing else.

## 20. A model that is not known to this build is fetched with a checksum, and the ONNX is placed by hand

"A checkout should never have to trust a binary it did not choose."

The weights are an ONNX export placed by hand, beside the config the
runtime fetches. The runtime says exactly which file it wants and where.

## 21. `list` honours `--root`, and the C++ runtime did not

The C++ runtime's `list` ignored the flag and reported the tree its own
binary lived in. Every other subcommand read `--root` from the argument
vector, so the one command that reports **what is installed** reported a
different inventory from the one that **uses** what is installed.

That is the failure shape this project keeps meeting: nothing breaks, the
table looks right, and the table is about a different machine.

So: `--root` is honoured everywhere, and the parity harness gives both
sides the same root -- which is also why the `sub/list` case only compares
the two runtimes when the oracle's default root holds the same containers.

## 22. A host-layout container does not need a design set

The C++ runtime asked for a compiled design set before it would run the
embedding, the classification and the transcription paths -- and it asked
even when the container holds row-major host-side operands and every
operation would run on the CPU.

The check was unconditional, so a CPU-only container could not be run
without a design set the container would not use. And a design set is the
OUTPUT of MLIR-AIE: it cannot be downloaded and it cannot be made without
the toolchain. The consequence was that the one container that can be built
and compared anywhere -- the plain f32 one -- could not be run at all.

So: a container that says `gemm_layout: "host"` runs on the CPU, and
`--npu-ops gemm` on it is refused by name rather than ignored.

The difference from the C++ runtime is declared in
`tools/parity/divergences.txt`.

## 23. `--serve` takes no value in the flag table, and the port is the next argument

The table says `--serve` has arity 0. The grammar is `--serve [PORT]`,
where the port is the next argument and only if it starts with a digit.

Those two statements disagree, and the disagreement has a consequence: in
the subcommands, where the positional arguments are rewritten into
`--ocr <path>` and `--pose <path>`, the number after `--serve` is a
**positional** -- so `npuimage ocr m.png --serve 9000` reads `9000` as a
second image.

One reader, one table, and the table's arity is what the rewriters believe.
A flag with an arity of 0 and a value is a flag whose value is a
positional, and that is worth knowing before it surprises somebody.

## 24. The size in the `list` table is the PAYLOAD, not the file

`mb` is the container's **data length** -- the tensors -- divided by a
million. Not the file size.

The difference is small and constant: the header and the directory are
a few tens of kilobytes, so a 13.5 MB container prints as 13 MB by one
measure and 14 by the other. And the table is read by a person deciding
whether to download something, so it should say how big the WEIGHTS are.

The catalogue's own rows use the CHECKPOINT's size, for the same reason:
a person reading `546 MB dl` wants to know what the download will cost.
