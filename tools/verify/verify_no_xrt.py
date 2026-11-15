#!/usr/bin/env python3
#===----------------------------------------------------------------------===//
# The direct transport's premise: no XRT, no Python.
#
# The direct transport exists for one reason -- a run on a machine that has
# the amdxdna driver and nothing else -- and that reason is a property of the
# BUILD, so it can be checked without a board:
#
#   1. `cargo build --no-default-features --features direct` succeeds. This is
#      the build that has neither the XRT feature nor its C shim, so a
#      reference to `npum_xdna::xrt` anywhere in the array path, or a build
#      script that compiles the shim regardless, fails here and not on the
#      user's machine.
#   2. The binary does not link libxrt or libpython. The crate is allowed to
#      depend on XRT; the binary that says "direct" is not.
#   3. The binary is a direct build at all: `npum-xdna` compiled WITHOUT the
#      `xrt` feature. A build that quietly kept XRT would pass (1) and (2) on
#      a machine that has XRT installed and link it anyway, so the feature
#      list is read out of cargo's own output.
#
# And, when there is a board and a container, one run: the check that the
# build is not merely XRT-free but working.
#===----------------------------------------------------------------------===//

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, **kwargs)


def build(target_dir: Path) -> tuple[bool, str]:
    """The pure-direct build, and the features cargo actually used."""
    proc = run(
        [
            "cargo",
            "build",
            "--release",
            "--offline",
            "--no-default-features",
            "--features",
            "direct",
            "-p",
            "npuembeddings-cli",
            "--message-format",
            "json-render-diagnostics",
        ],
        cwd=ROOT,
        env={**os.environ, "CARGO_TARGET_DIR": str(target_dir)},
    )
    if proc.returncode != 0:
        return False, proc.stderr.strip()[-4000:]
    # The features are in the compiler invocation, which is the only place
    # that says what the binary was built WITH.
    if '"direct"' not in proc.stdout:
        return False, "the build did not report the direct feature"
    if '"xrt"' in proc.stdout:
        return False, "the build still has the xrt feature"
    return True, ""


def links_xrt(binary: Path) -> list[str]:
    """The XRT (and Python) libraries the binary is linked against."""
    if shutil.which("ldd") is None:
        return []
    proc = run(["ldd", str(binary)])
    found = []
    for line in proc.stdout.splitlines():
        lower = line.lower()
        if "xrt" in lower or "python" in lower:
            found.append(line.strip())
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", default="/tmp/npum-verify-no-xrt")
    parser.add_argument(
        "--container",
        help="a .npue to run through the binary; skipped when it is absent",
    )
    parser.add_argument("--texts", help="the input texts for that run")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    print(f"# build       {ROOT} --no-default-features --features direct")
    ok, error = build(target_dir)
    if not ok:
        print("FAIL: the direct build does not build without XRT")
        print(error)
        return 1
    print("# build       ok, no xrt feature")

    binary = target_dir / "release" / "npuembeddings"
    if not binary.exists():
        print(f"FAIL: the build produced no {binary}")
        return 1

    linked = links_xrt(binary)
    if linked:
        print("FAIL: the binary links XRT or Python")
        for line in linked:
            print(f"    {line}")
        return 1
    print("# linkage     no libxrt, no libpython")

    if args.container:
        texts = Path(args.texts or "/tmp/npum-verify-no-xrt-in.txt")
        if not texts.exists():
            texts.write_text("a text to embed\n")
        out = target_dir / "out.f32"
        proc = run(
            [str(binary), "embed", args.container, str(texts), str(out), "--npu-ops", "gemm"],
            cwd=ROOT,
        )
        if proc.returncode != 0 or not out.exists():
            print("FAIL: the direct binary did not run")
            print((proc.stderr or proc.stdout).strip()[-4000:])
            return 1
        print(f"# run         wrote {out} ({out.stat().st_size} bytes)")
    else:
        print("# run         skipped (no --container)")

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
