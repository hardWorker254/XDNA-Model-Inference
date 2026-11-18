# Invariants

<!-- SPDX-License-Identifier: Apache-2.0 -->

These are rules the runtime must not break. A change that breaks one is a regression, and parity/golden checks are used to detect such regressions.

Each invariant should have a reason. An invariant without a reason is hard to evaluate and easy to break accidentally.

---

## 1. The last occurrence of a flag wins, not the first

`--pipeline 4 --pipeline 1` disables pipelining.

This is not a preference. If only the first value is honored, later overrides in a command line can be silently ignored, producing plausible but incorrect behavior (for example, a different thread count than requested).

## 2. `design_fits()` is fail-closed

A design set built for another generation is refused, **even when the geometry happens to agree**.

Accepting a set whose tile dimensions coincide by accident can produce wrong-but-plausible results. This check must not be weakened.

## 3. The B panel sub-tile is per-generation (and per operand path where applicable), and it is checked against the container

| | bf16 | int8 |
|---|---|---|
| npu1 / `aie2` | (8, 4) | (8, 8) |
| npu2 / `aie2p` | (8, 8) | (8, 8) |

A container packed for one target and consumed with the wrong panel layout can have matching byte counts and shapes while still producing incorrect numbers. The sub-tile must be defined in one place, derived consistently, and checked against container metadata/layout information (not inferred from filenames).

## 4. The softmax mask is clamped BEFORE widening to bf16, or `-inf` becomes NaN

Three clamps matter, each for a specific reason:

1. **On load, in bf16, before widening:** clamp masked values (e.g. `max(x, -1e30)` as appropriate for the implementation) so narrowing does not turn finite sentinel values into `-inf`/NaN-producing states.
2. **On the difference:** `d = max(x - m, -100)`. `exp(-100)` underflows to zero in bf16, which is the intended result for masked elements.
3. **On the base-2 argument:** clamp the exponent-domain argument (e.g. `max(d * log2e, -120)`) to avoid invalid exponent field behavior in the polynomial/exp2 path.

CPU and accelerated paths must compute the same function in these regions; differing clamp behavior means a different function.

## 5. Vectorized paths must match scalar/reference arithmetic rules (including contraction behavior), bit for bit where required

Contraction behavior (e.g. FMA vs separate multiply+add) changes results. Paths that are required to match must follow the same contraction/rounding rules. “Close” is not sufficient when exact agreement is required by a gate.

## 6. A partially loaded model is refused by name

A renamed module, missing head, or shape mismatch must produce a refusal that names what is wrong. The runtime must not continue with plausible but invalid outputs.

## 7. Safetensors/serialized tensor keys are data, not code

Model/container inputs must be parsed as data. Inputs must not be able to execute code during loading.

## 8. A failed test is a difference, not “the golden needs updating”

If an expected result changes, the change must be intentional, explained, and recorded/documented explicitly (updated golden and/or declared divergence with rationale). Expected outputs must not be rewritten implicitly as a side effect.

---

## 9. A container of another family is refused by name, and the refusal says which program to run it

Refusals should be clear and actionable, including which binary/command family owns the container/model type. The family is determined consistently (from command context and from container/model metadata/arch as applicable), and wrong-family usage must fail explicitly.

## 10. Generation/target selection must be explicit and verifiable

When hardware generation affects correctness, selection must come from a trustworthy source (device/runtime query where available) and user overrides must be validated. A wrong default or unchecked override must not silently produce plausible but incorrect results.

## 11. `--serve [PORT]` parsing is strict and unambiguous

Optional port parsing must follow a defined grammar (e.g. only consume the next argument as a port when it matches the expected numeric form). Unsupported spellings/ambiguous forms must not be accepted implicitly.

## 12. Removed flags are refused by name, and the refusal names the replacement

Deprecated/removed flags should be recognized explicitly so the refusal can explain what to use instead, rather than falling back to a generic “unrecognized option” message.

## 13. Readers accept supported legacy container magics; writers emit only the current one

The current container magic/name is emitted by writers. Readers may accept legacy compatible magics for backward compatibility when layout semantics are unchanged. Compatibility must not change interpretation of payload data.

## 14. The layout hash is computed over canonical (key-sorted, compact) JSON

Layout hash computation must use a canonical serialization form so independent implementations produce identical bytes. Serialization settings that change key order/formatting can change the hash and must be controlled.

## 15. bf16 rounding is defined and implemented explicitly

Rounding behavior (including tie handling) must be specified and consistent with the intended semantics. Implicit reliance on unspecified/default library behavior is risky for bit-exact comparisons and serialized artifacts.

## 16. Hardware/context budgets that are unknown must be represented as unknown

If a budget/limit has not been measured or cannot be determined, represent it as unknown (e.g. `Option`/absence) rather than inventing a numeric value. Callers must handle unknown explicitly.

## 17. A case whose prerequisites are missing is SKIPPED, not recorded as a pass

If required models/fixtures/artifacts/hardware are unavailable, the case must be reported as skipped with reason. A run that compared nothing must not be reported as a full pass.

## 18. Golden comparisons use a fixed/explicit root

Comparisons must run against an explicit, shared fixture/model root when reproducibility depends on it. Inferring different roots for different binaries can compare different inputs and produce misleading diffs.

## 19. Normalization rules are named, finite, and visible

Only documented normalization rules may be applied when comparing outputs. Rules must be explicit and reported with results. Differences outside the allowed normalization set are failures unless intentionally declared/documented.

## 20. Model acquisition must be verifiable

When models/checkpoints are fetched or prepared, integrity and provenance checks (checksums/pins/manifests as applicable) must be enforced. Unverified binaries/artifacts must not be trusted implicitly.

## 21. `list` honors `--root` consistently

Commands that report installed/available artifacts must respect the same root/path resolution rules as commands that use them. Inconsistent root handling can produce correct-looking but misleading inventories.

## 22. A host-layout container does not require array/NPU design artifacts

If a container/path runs entirely on host-side layouts/kernels, it must not require unrelated device design artifacts. Unsupported device-specific requests for such containers must be refused explicitly by name.

## 23. `--serve` arity and argument parsing must be consistent across command forms

Flag-table arity and positional rewriting must agree so optional values (e.g. port) are interpreted consistently in all command forms. A value intended for a flag must not be misinterpreted as an unrelated positional argument.

## 24. Reported container/model sizes must be defined clearly

If a UI/CLI reports a size (e.g. payload/data size vs file/checkpoint/download size), the definition must be consistent and meaningful for the user decision it supports. Avoid ambiguous labels when different size measures exist.
