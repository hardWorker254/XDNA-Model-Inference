# Parity and verification

## Golden transcripts (contract)

- Golden outputs live in `tools/parity/golden/` and are the behavioral contract vs the C++ oracle.
- **Compare:** `python3 tools/parity/run.py compare --candidate ./target/release`
- **Record (from oracle binaries):** `python3 tools/parity/run.py record --oracle ~/Npu-Embeddings-XDNA1/runtime/build`
- **List cases/rules:** `python3 tools/parity/run.py list -v` and `python3 tools/parity/run.py rules`

### Interpreting results

- **Skipped** cases (missing models/fixtures, missing design set, or network-gated cases when not enabled) are reported and must not be counted as passes. A run that compared nothing is not a passing run.
- **Failures** are diffs from the golden contract. Do not "fix" failures by re-recording goldens as a side effect.
- **Declared divergences:** intentional, reviewed differences belong in `tools/parity/divergences.txt` with a case id and reason. Declared divergences are still printed with diffs but do not fail the run.

### Process expectations

1. Record goldens only from the agreed oracle build/environment.
2. When behavior intentionally changes, update code/contracts and record/document the divergence or golden change explicitly with rationale (never implicitly).
3. Keep fixture inputs and normalization rules stable and named (`tools/parity/run.py rules`).
4. For PR-level confidence, prefer a smoke subset when full models/NPU are unavailable; run full parity where the change affects compared paths.

## Other checks

- `cargo test --workspace` — Rust tests (including backend/family gates where present).
- `python3 tools/verify/verify_no_xrt.py` — direct build must not link XRT/Python (gate), when applicable.
- Family-/path-specific verifiers live under `tools/verify/` (run as appropriate for the change).

See also [INVARIANTS.md](INVARIANTS.md).
