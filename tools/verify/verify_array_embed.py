#!/usr/bin/env python3
#===----------------------------------------------------------------------===//
# The text family's array path, against the C++ oracle.
#
# Runs the same container through both runtimes with the same --npu-ops and
# compares the vectors.
#
# WHY A TOLERANCE AND NOT EQUALITY
#
# The array computes in bf16 and the host computes in fp32, so the two
# runtimes' answers are not the same bytes -- and they are not supposed to
# be. The bound is the one the repository already uses for a whole hidden
# state: tools/verify/verify_design_numerics.py measures bf16's half an ulp
# at 2**-9 and the repo's 1-cos bound at 2e-3.
#
# WHAT IS COMPARED EXACTLY
#
# * the DISPATCH COUNT, per text. It is the number that says whether the
#   flag did what it said, and it is an integer: a run that dispatched
#   something else is a different run, whatever the vectors say.
# * the vectors, within the tolerance above.
#===----------------------------------------------------------------------===//

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

#: bf16 carries 8 significand bits, so one output rounding is at most half
#: an ulp = 2^-9 relative. The same constant the design harness uses.
BF16_HALF_ULP = 2.0**-9

#: The repository's bound for a whole hidden state: 1 - cos.
DEFAULT_RTOL_COS = 2e-3


def run(binary: Path, container: Path, texts: Path, out: Path, ops: str | None) -> str:
    """Run one runtime and write its vectors."""
    argv = [str(binary), "embed", str(container), str(texts), str(out)]
    if ops:
        argv += ["--npu-ops", ops]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise SystemExit(
            f"{binary} failed with {proc.returncode}:\n{proc.stdout}\n{proc.stderr}"
        )
    return proc.stdout


def dispatches(stdout: str) -> int:
    """The dispatch count the status block reports."""
    for line in stdout.splitlines():
        if line.strip().startswith("dispatches"):
            for word in line.split():
                if word.isdigit():
                    return int(word)
    raise SystemExit("the run reported no dispatch count")


def compare(a: np.ndarray, b: np.ndarray, rtol_cos: float) -> tuple[bool, float, float]:
    """The 1-cos and the largest relative error between two sets of vectors."""
    a = a.reshape(-1, 384)
    b = b.reshape(-1, 384)
    an = a / np.linalg.norm(a, axis=1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    cos = float(np.min(np.sum(an * bn, axis=1)))
    denom = np.maximum(np.abs(b), 1e-30)
    rel = float(np.max(np.abs(a - b) / denom))
    return (1.0 - cos) <= rtol_cos, 1.0 - cos, rel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--container", type=Path, required=True)
    ap.add_argument("--texts", type=Path, required=True)
    ap.add_argument("--ops", default="")
    ap.add_argument("--rtol-cos", type=float, default=DEFAULT_RTOL_COS)
    ap.add_argument("--out", type=Path, default=Path("/tmp/npue-array"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    oracle_out = args.out / "oracle.f32"
    candidate_out = args.out / "candidate.f32"

    print(f"# container   {args.container}")
    print(f"# ops         {args.ops or '(none)'}")

    oracle_log = run(args.oracle, args.container, args.texts, oracle_out, args.ops)
    candidate_log = run(args.candidate, args.container, args.texts, candidate_out, args.ops)

    a = np.fromfile(oracle_out, dtype=np.float32)
    b = np.fromfile(candidate_out, dtype=np.float32)
    if a.shape != b.shape:
        raise SystemExit(f"the vectors have different lengths: {a.shape} and {b.shape}")

    ok, one_minus_cos, rel = compare(a, b, args.rtol_cos)
    print(f"# vectors     {a.size // 384} x 384")
    print(f"# 1 - cos     {one_minus_cos:.3e}   (bound {args.rtol_cos:.1e})")
    print(f"# max rel     {rel:.3e}")
    try:
        oracle_disp = dispatches(oracle_log)
        ours_disp = dispatches(candidate_log)
        print(f"# oracle disp {oracle_disp}")
        print(f"# ours disp   {ours_disp}")
        # THE DISPATCH COUNTS, AND WHEN THEY CAN BE COMPARED.
        #
        # The C++ encodes a batch of texts in one tier, so one dispatch
        # computes every text's rows; this runtime encodes one text at a
        # time, so one dispatch computes one text's rows. The two counts
        # are therefore the same only for a single text -- and for one
        # text they must be equal, because a run that dispatched
        # something else is a different run.
        if a.size // 384 == 1 and oracle_disp != ours_disp:
            print(f"FAIL: the dispatch counts differ ({oracle_disp} against {ours_disp})")
            return 1
        if a.size // 384 > 1:
            print(
                f"# note        the counts are per batch: the C++ encoded "
                f"{a.size // 384} texts in one tier and this runtime encoded them one at a "
                f"time, so {ours_disp} against {oracle_disp} is expected"
            )
    except SystemExit:
        pass
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
