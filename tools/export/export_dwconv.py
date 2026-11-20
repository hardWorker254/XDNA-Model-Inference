#!/usr/bin/env python3
# NpuEmbeddings -- build the production depthwise-conv design set.
#
# SPDX-License-Identifier: Apache-2.0
#
# One xclbin, one instruction stream, one fixed geometry: 16 channels per
# dispatch x 1024 outputs per channel. Real 3x3 depthwise layers are chunked
# into it by the runtime. The design is the same one the research driver
# compiled; exporting it means --npu-ops dwconv can now name a REAL design set
# instead of refusing by name.
#
# Usage (with aarch/IRON environment active):
#   python tools/export/export_dwconv.py --arch 1 \
#       --out runtime/artifacts/mediapipe-hands
#   python tools/export/export_dwconv.py --arch 1 \
#       --out runtime/artifacts/mediapipe-pose
#   python tools/export/export_dwconv.py --arch 1 \
#       --out runtime/artifacts/paddleocr-onnx
#
# The directory is written as <out>/artifacts_npu<N>/dwconv9x3, i.e. beside
# gemm_rtp, which is where the runtime discovers design roots.

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))

SYMBOL = "dwconv9x3_bf16_tiled"
SET_NAME = "dwconv9x3"
CHANNELS = 16
WIDTH = 1024
STRIDE_EXTRA = 32
TILE_K, TILE_N, TILE_M, ROWS, COLS = 9, 16, 1024, 4, 4


def chan_block(tile_w: int) -> int:
    stride = tile_w + STRIDE_EXTRA
    return ((3 * stride + 10 + 15) // 16) * 16


def cache_dir() -> Path:
    return Path(os.environ.get("NPU_CACHE_HOME",
                               str(Path.home() / ".npu" / "cache")))


def _has_symbol(d: Path) -> bool:
    mlir = d / "aie.mlir"
    return mlir.exists() and SYMBOL in mlir.read_text(
        encoding="utf-8", errors="ignore")


def purge_cache() -> int:
    root = cache_dir()
    if not root.is_dir():
        return 0
    n = 0
    for d in list(root.iterdir()):
        if d.is_dir() and _has_symbol(d):
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n


def find_cache() -> Path:
    root = cache_dir()
    hits = [
        d for d in root.iterdir()
        if d.is_dir()
        and (d / "aie.mlir").exists()
        and (d / "final.xclbin").exists()
        and (d / "insts.bin").exists()
        and _has_symbol(d)
    ]
    if len(hits) != 1:
        raise SystemExit(
            f"dwconv export: {len(hits)} cache candidates (expected 1) in {root}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["1"], default="1")
    ap.add_argument("--out", required=True,
                    help="artifacts root, e.g. runtime/artifacts/mediapipe-hands")
    args = ap.parse_args()

    cache = cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["NPU_CACHE_HOME"] = str(cache)
    os.environ["IRON_CACHE_DIR"] = str(cache)
    print(f"dwconv: purged {purge_cache()} cache candidate(s)")

    # Heavy imports AFTER cache setup (mlir_aie reads NPU_CACHE_HOME at import).
    import aie.iron as iron
    from aie.iron.device import from_name
    from ml_dtypes import bfloat16
    from dwconv_array import dwconv9x3_array
    from toolchain_provenance import write_toolchain_json

    iron.set_current_device(from_name("npu1", n_cols=None))
    n_elem = CHANNELS * chan_block(WIDTH)
    X = iron.zeros((n_elem,), dtype=bfloat16, device="npu")
    Y = iron.zeros((CHANNELS * WIDTH,), dtype=bfloat16, device="npu")
    dwconv9x3_array(X, Y)  # compile

    src = find_cache()
    dst = Path(args.out).expanduser() / f"artifacts_npu{args.arch}" / SET_NAME
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("final.xclbin", "insts.bin"):
        shutil.copy2(src / f, dst / f)
    a_bytes = n_elem * 2
    c_bytes = CHANNELS * WIDTH * 2
    meta = {
        "name": SET_NAME,
        "kind": SET_NAME,
        "kernel": "MLIR_AIE",
        "arch": int(args.arch),
        "device": "npu1",
        "buffers": [a_bytes, c_bytes],
        "a_dtype": "bf16",
        "c_dtype": "bf16",
        "cols": COLS,
        "tile": {"m": TILE_M, "k": TILE_K, "n": TILE_N, "rows": ROWS},
        "slot0": {"file": "insts.bin", "op": SET_NAME, "batch": 1},
        "tiers": [1],
        "batch": 1,
        "streams": [{
            "op": SET_NAME, "slot": 1, "file": "insts.bin",
            "M": TILE_M, "K": TILE_K, "N": TILE_N, "batch": 1,
        }],
        "insts_bytes": (dst / "insts.bin").stat().st_size,
        "source_cache_dir": src.name,
        "description": "16 channels x 1024 outputs of bf16 3x3 depthwise conv; "
                       "three sliding_mul_ops<16,3> per output, bias in the "
                       "input buffer; activation applied by the host",
    }
    (dst / "design.json").write_text(json.dumps(meta, indent=2),
                                     encoding="utf-8")
    write_toolchain_json(dst)
    print(f"dwconv: wrote {dst}  "
          f"xclbin {(dst / 'final.xclbin').stat().st_size / 1024:.1f} KB, "
          f"buffers {a_bytes}/{c_bytes} bytes")


if __name__ == "__main__":
    main()
