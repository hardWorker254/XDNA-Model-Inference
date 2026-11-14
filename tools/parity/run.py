# SPDX-License-Identifier: Apache-2.0
"""The parity harness's entry point.

    python tools/parity/run.py record   --oracle   <the C++ binaries>
    python tools/parity/run.py compare  --candidate <the Rust binaries>
    python tools/parity/run.py list
    python tools/parity/run.py rules

WHAT A PASSING RUN MEANS
------------------------

That the candidate produces the same transcript as the oracle, for every
case whose model is installed. Not "roughly the same": the same, after the
named volatile-field rules have been applied.

WHAT A FAILING RUN MEANS
------------------------

That something changed. The report says WHICH case, WHICH stream, and
prints the diff -- and the first difference is where the reader starts.

A case whose model is not installed is SKIPPED, and the skip is printed.
A skip is not a pass: the harness prints the number of cases that were
compared, the number that were skipped, and the number that passed, so
that a run that compared nothing cannot be mistaken for a run that passed
everything.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cases as case_table  # noqa: E402
import normalize  # noqa: E402
import recorder  # noqa: E402


def select(selector: list[str]) -> list[case_table.Case]:
    """The cases the selector picks.

    A selector is a case id, a tag, or a prefix of either.
    """
    if not selector:
        return list(case_table.CASES)
    out = []
    for case in case_table.CASES:
        if case.id in selector or any(t in selector for t in case.tags):
            out.append(case)
        elif any(case.id.startswith(s) for s in selector):
            out.append(case)
    return out


def cmd_record(args) -> int:
    """Record the goldens from the oracle."""
    root = recorder.ensure_root()
    recorder.register_paths()
    bin_dir = recorder.binary_dir(args.oracle)
    cases = select(args.case)

    recorded = 0
    skipped = 0
    for case in cases:
        transcript = recorder.run_case(bin_dir, case, root)
        if transcript.skipped:
            print(f"  SKIP  {case.id}: {transcript.skipped}")
            skipped += 1
            continue
        recorder.write_golden(transcript, source=str(bin_dir))
        print(f"  ok    {case.id}")
        recorded += 1
    print()
    print(f"recorded {recorded}, skipped {skipped}")
    if skipped:
        print()
        print("A SKIP IS NOT A PASS. A skipped case needs something the fixture")
        print("root does not hold -- a model, or a compiled design set -- and it")
        print("pins nothing until it is recorded.")
    return 0


def cmd_compare(args) -> int:
    """Compare the candidate against the goldens."""
    root = recorder.ensure_root()
    recorder.register_paths()
    bin_dir = recorder.binary_dir(args.candidate)
    cases = select(args.case)
    declared = recorder.divergences()

    passed = 0
    failed = 0
    skipped = 0
    diverged = 0
    for case in cases:
        comparison = recorder.compare_case(bin_dir, case, root)
        if comparison.skipped:
            print(f"  SKIP  {case.id}: {comparison.detail}")
            skipped += 1
        elif comparison.ok:
            print(f"  ok    {case.id}")
            passed += 1
        elif case.id in declared:
            # A DECLARED DIVERGENCE: the two runtimes are known to differ
            # here, on purpose, and the reason is written down. It is
            # reported -- loudly, with the diff -- and it does not fail
            # the run, because a difference that was decided on is not an
            # accident.
            print(f"  DIVERGE  {case.id}: {declared[case.id]}")
            print(comparison.detail)
            diverged += 1
        else:
            print(f"  FAIL  {case.id}")
            print(comparison.detail)
            failed += 1
            if args.first_failure:
                break
    print()
    print(
        f"passed {passed}, failed {failed}, skipped {skipped}, "
        f"declared divergences {diverged}"
    )
    if failed:
        print()
        print("A FAILING CASE IS A DIFFERENCE, NOT A GOLDEN THAT NEEDS UPDATING.")
        print("If the difference is intended, say so in the commit message and")
        print("re-record the golden on purpose -- never as a side effect.")
        print()
        print("If it is intended AND it will stay that way, declare it in")
        print("tools/parity/divergences.txt, with the reason. A declared")
        print("divergence is still printed on every run; it just does not")
        print("fail the run.")
    return 1 if failed else 0


def cmd_list(args) -> int:
    """The case table."""
    for case in case_table.CASES:
        tags = ",".join(case.tags)
        print(f"{case.id:<44} {case.binary:<14} [{tags}]")
        if args.verbose:
            print(f"    {case.why}")
    print()
    print(f"{len(case_table.CASES)} cases")
    return 0


def cmd_rules(args) -> int:
    """The normalisation rules."""
    print(normalize.rules_documentation())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="parity", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("record", help="record the goldens from the oracle")
    p.add_argument("--oracle", default=str(recorder.REPO.parent / "Npu-Embeddings-XDNA1" / "runtime" / "build"))
    p.add_argument("--case", action="append", default=[])
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("compare", help="compare the candidate against the goldens")
    p.add_argument("--candidate", default=str(recorder.REPO / "target" / "release"))
    p.add_argument("--case", action="append", default=[])
    p.add_argument("--first-failure", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("list", help="the case table")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("rules", help="the normalisation rules")
    p.set_defaults(func=cmd_rules)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
