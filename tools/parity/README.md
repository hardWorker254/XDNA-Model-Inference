# Parity harness

<!-- SPDX-License-Identifier: Apache-2.0 -->

This directory contains tooling to record baseline ("golden") outputs and compare candidate binaries against them.

```sh
# Record goldens from a baseline/oracle build
python tools/parity/run.py record --oracle <path-to-oracle-build>

# Compare candidate binaries against recorded goldens
python tools/parity/run.py compare --candidate <path-to-candidate-build>

# Show the case table and normalization rules
python tools/parity/run.py list -v
python tools/parity/run.py rules
```

## Golden artifacts

For each case, a directory under `golden/<case-id>/` typically contains:

| File | Purpose |
| --- | --- |
| `stdout` | Captured standard output |
| `stderr` | Captured standard error |
| `code` | Exit code |
| `body` | HTTP response body (HTTP cases) |
| `status` | HTTP response status (HTTP cases) |
| `meta.json` | Case parameters and normalization metadata |

`meta.json` makes a golden reviewable (what ran, when, and which normalization rules were applied).

## Harness principles

- **Separate record/compare roles:** `record` captures baseline outputs; `compare` checks candidate outputs against them using the same harness logic.
- **Consistent fixture root:** runs should use the same fixture/model root so differences reflect behavior, not different inputs.
- **No unintended network dependence:** offline-by-default execution is preferred for reproducible goldens (network-dependent cases should be explicitly gated).
- **Skips are not passes:** if a case cannot run due to missing fixtures/artifacts/hardware, it should be reported as skipped.
- **Differences require intent:** update goldens or declare/document divergences deliberately; do not silently rewrite expected results.
- **Explicit normalization rules:** only documented normalization rules should be applied (`run.py rules`).

## Fixtures

`fixtures/` contains inputs and helper assets used by parity cases (text/image/audio samples, fixture root layout, and scripts for fetching/preparing required model data where applicable).

Some cases may require additional compiled artifacts or local models and will be skipped when prerequisites are unavailable.

## Notes

- Treat `tools/parity/golden/` as a behavioral contract for covered cases.
- If a behavior change is intentional, update the corresponding golden(s) and/or document the divergence according to project practice.
- See [docs/en/PARITY.md](../../docs/en/PARITY.md) and [docs/ru/PARITY.md](../../docs/ru/PARITY.md) for project-level parity guidance.
