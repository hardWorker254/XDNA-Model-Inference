# Parity and verification

## Golden transcripts (contract)

- Golden outputs live in `tools/parity/golden/` and serve as a behavioral contract for covered cases.
- **Compare:** `python3 tools/parity/run.py compare --candidate ./target/release`
- **Record (from a baseline/reference build):** `python3 tools/parity/run.py record --oracle <path-to-reference-build>`
- **List cases/rules:** `python3 tools/parity/run.py list -v` and `python3 tools/parity/run.py rules`

### Interpreting results

- **Skipped** cases (missing models/fixtures, missing design artifacts, or gated cases not enabled) are reported separately and must not be counted as passes. A run that compared nothing is not a passing run.
- **Failures** are diffs from the golden contract. Do not “fix” failures by re-recording goldens as a side effect.
- **Declared divergences:** intentional, reviewed differences belong in `tools/parity/divergences.txt` with a case id and reason. Declared divergences are still printed with diffs but do not fail the run.

### Process expectations

1. Record goldens only from an agreed baseline/reference build and environment.
2. When behavior intentionally changes, update code/contracts and record/document the divergence or golden change explicitly with rationale (never implicitly).
3. Keep fixture inputs and normalization rules stable and named (`tools/parity/run.py rules`).
4. For PR-level confidence, prefer a smoke subset when full models/NPU are unavailable; run full parity where the change affects compared paths.

## Other checks

- `cargo test --workspace` — Rust tests (including backend/family gates where present).
- `python3 tools/verify/verify_no_xrt.py` — direct build must not link XRT/Python (gate), when applicable.
- Family-/path-specific verifiers live under `tools/verify/` (run as appropriate for the change).

See also [INVARIANTS.md](INVARIANTS.md).
