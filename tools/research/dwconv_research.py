#!/usr/bin/env python3
# NpuEmbeddings -- A/B driver for the depthwise design (research only).
#
# SPDX-License-Identifier: Apache-2.0
#
# The design itself lives in tools/lib/dwconv_array.py and is shared with the
# production exporter. This file only builds inputs, times baseline vs tiled
# and checks parity against a bf16-rounded host oracle.
#
# Usage:
#   PATH=/opt/xilinx/xrt/bin:$PATH PYTHONPATH=/opt/xilinx/xrt/python \
#     ./.venv/bin/python tools/research/dwconv_research.py \
#       --cols 4 --width 1024 --iters 20 --mode both --tile-width 256

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron.device import from_name
from aie.utils.benchmark import run_iters

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))

from dwconv_array import (  # noqa: E402
    STRIDE_EXTRA, TRACE_DIR, chan_block, dw_array,
)

def make_inputs(channels, width, tile_w, n_cols):
    """Build the padded per-tile buffers + the bf16-rounded host oracle."""
    n_tiles = width // tile_w
    stride = tile_w + STRIDE_EXTRA
    chan_in_n = chan_block(tile_w)
    rng = np.random.default_rng(7)
    img = (rng.standard_normal((channels, width + 2, width + 2)) * 0.1).astype(np.float32)
    w = (rng.standard_normal((channels, 3, 3)) * 0.1).astype(np.float32)
    bias = (rng.standard_normal(channels) * 0.1).astype(np.float32)

    x_np = np.zeros((n_tiles, channels, chan_in_n), dtype=np.float32)
    for t in range(n_tiles):
        lo = t * tile_w
        hi = lo + tile_w + 2
        for c in range(channels):
            for dy in range(3):
                x_np[t, c, dy * stride:dy * stride + tile_w + 2] = img[c, dy, lo:hi]
            x_np[t, c, 3 * stride:3 * stride + 9] = w[c].reshape(-1)
            x_np[t, c, 3 * stride + 9] = bias[c]

    # Oracle: bf16-rounded inputs, exactly what the device consumes.
    xb = img.astype(bfloat16).astype(np.float32)
    wb = w.astype(bfloat16).astype(np.float32)
    bb = bias.astype(bfloat16).astype(np.float32)
    ref = np.broadcast_to(bb[:, None], (channels, width)).astype(np.float64).copy()
    for dy in range(3):
        for dx in range(3):
            ref += wb[:, dy, dx, None].astype(np.float64) * \
                   xb[:, dy, dx:dx + width].astype(np.float64)
    return x_np.astype(bfloat16), ref


def host_time(channels, width):
    """Vectorised NumPy reference, timed like the device path."""
    rng = np.random.default_rng(11)
    img = (rng.standard_normal((channels, 3, width + 2)) * 0.1).astype(np.float32)
    w = (rng.standard_normal((channels, 3, 3)) * 0.1).astype(np.float32)
    b = (rng.standard_normal(channels) * 0.1).astype(np.float32)

    def run():
        out = np.broadcast_to(b[:, None], (channels, width)).copy()
        for dy in range(3):
            for dx in range(3):
                out += w[:, dy, dx, None] * img[:, dy, dx:dx + width]
        return out

    for _ in range(3):
        run()
    t0 = time.perf_counter()
    for _ in range(20):
        run()
    return (time.perf_counter() - t0) / 20 * 1e6


def measure(args, use_tiled_kernel: bool, x_np, ref, channels, width, tile_w,
            n_cols, tag):
    n_tiles = width // tile_w
    stride = tile_w + STRIDE_EXTRA
    chan_in_n = chan_block(tile_w)
    chan_out_n = tile_w

    cfg = None
    if args.trace:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        cfg = TraceConfig(trace_size=args.trace_size,
                          trace_file=str(TRACE_DIR / f"trace_{tag}.txt"))

    l2_mode = args.l2_mode
    cpc = channels // (4 * n_cols)
    X = iron.zeros((n_tiles * channels * chan_in_n,), dtype=bfloat16, device="npu")
    Y = iron.zeros((n_tiles * channels * chan_out_n,), dtype=bfloat16, device="npu")
    if l2_mode == "per_col":
        # [tile][channel] -> [col][row][tile][k]
        arr = x_np.reshape(n_tiles, channels, chan_in_n)
        col_major = np.zeros((n_cols, 4, n_tiles, cpc, chan_in_n),
                             dtype=bfloat16)
        for col in range(n_cols):
            for r in range(4):
                for k in range(cpc):
                    c = (col * 4 + r) * cpc + k
                    col_major[col, r, :, k, :] = arr[:, c, :]
        X[:] = col_major.reshape(-1)
    else:
        X[:] = x_np.reshape(-1)

    kw = dict(channels=channels, width=width, tile_w=tile_w,
              chan_n=chan_block(tile_w), n_cols=n_cols,
              fifo_depth=args.fifo_depth, use_tiled_kernel=use_tiled_kernel,
              l2_mode=l2_mode, trace_config=cfg, trace_row=0,
              trace_col=min(args.cols - 1, 1), trace_egress_col=0)

    t0 = time.perf_counter()
    dw_array(X, Y, **kw)
    compile_s = time.perf_counter() - t0

    got_raw = Y.numpy().astype(np.float64)
    if l2_mode == "per_col":
        # [col][row][tile][k][tile_w] -> [tile][channel][tile_w]
        cm = got_raw.reshape(n_cols, 4, n_tiles, cpc, tile_w)
        got = np.zeros((n_tiles, channels, tile_w), dtype=np.float64)
        for col in range(n_cols):
            for r in range(4):
                for k in range(cpc):
                    c = (col * 4 + r) * cpc + k
                    got[:, c, :] = cm[col, r, :, k, :]
    else:
        got = got_raw.reshape(n_tiles, channels, tile_w)
    full = np.zeros((channels, width), dtype=np.float64)
    for t in range(n_tiles):
        full[:, t * tile_w:(t + 1) * tile_w] = got[t]
    rel = float(np.linalg.norm(full - ref) / np.linalg.norm(ref))
    max_abs = float(np.max(np.abs(full - ref)))

    res = run_iters(dw_array, X, Y, warmup=args.warmup, iters=args.iters, **kw)
    npu = getattr(getattr(res, "npu", None), "avg_us", float("nan"))
    npu_min = getattr(getattr(res, "npu", None), "min_us", float("nan"))
    e2e = getattr(getattr(res, "e2e", None), "avg_us", float("nan"))
    cpc = channels // (4 * n_cols)
    row = dict(mode="tiled" if use_tiled_kernel else "baseline",
               tile=tile_w, cols=n_cols, cpc=cpc, n_tiles=n_tiles,
               fifo=args.fifo_depth, l2_mode=l2_mode, npu=npu,
               npu_min=npu_min, e2e=e2e, rel=rel, max_abs=max_abs,
               compile_s=compile_s)

    if cfg is not None:
        phys = getattr(cfg, "physical_mlir_path", None)
        if phys:
            jpath = TRACE_DIR / f"trace_{tag}.json"
            cfg.trace_to_json(phys, str(jpath))
            from aie.utils.trace.utils import get_cycles_summary
            deltas = []
            for entry in get_cycles_summary(str(jpath)):
                deltas += [d for d in entry[1:] if d is not None]
            if deltas:
                row["core_cycles"] = sum(deltas) / len(deltas)
                print(f"  trace: {len(deltas)} event0/event1 pairs, "
                      f"avg core delta = {row['core_cycles']:.0f} cycles "
                      f"(~{row['core_cycles'] / 1e9 * 1e6:.2f} us @1GHz)")
        else:
            print("  trace: no physical MLIR on cache hit (cycles unavailable)")
    return row, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--tile-width", type=int, default=256,
                    help="outputs per core iteration (128/256/512/1024)")
    ap.add_argument("--fifo-depth", type=int, default=2)
    ap.add_argument("--l2-mode", choices=["per_tile", "per_col"],
                    default="per_tile",
                    help="per_col cuts runtime fill/drain calls to one per "
                         "column, but on npu1 it exceeds the core DMA "
                         "channels (2 in/2 out) and the MemTile DMA budget; "
                         "kept as a measured wall, not a working mode")
    ap.add_argument("--mode", choices=["baseline", "tiled", "both"], default="tiled")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--trace-size", type=int, default=0x2000)
    args = ap.parse_args()

    iron.set_current_device(from_name("npu1", n_cols=None))
    channels = args.channels
    assert channels % (4 * args.cols) == 0
    tile_w = args.tile_width if args.tile_width > 0 else args.width

    host_us = host_time(channels, args.width)
    print(f"host NumPy: {host_us:.1f} us  ({channels}x{args.width}, "
          f"{channels * args.width * 9 / 1e6:.3f}M useful MAC)")
    print(f"{'mode':<9} {'tile':>5} {'cols':>4} {'cpc':>3} {'tiles':>5} "
          f"{'fifo':>4} {'NPU us':>9} {'NPU min':>9} {'e2e us':>9} "
          f"{'relfro':>9} {'max|e|':>9}")
    print("-" * 89)

    rows = []
    if args.mode in ("baseline", "both"):
        bx, ref = make_inputs(channels, args.width, args.width, args.cols)
        row, _ = measure(args, False, bx, ref, channels, args.width,
                         args.width, args.cols, f"baseline_{args.cols}c")
        rows.append(row)
    if args.mode in ("tiled", "both"):
        tx, ref = make_inputs(channels, args.width, tile_w, args.cols)
        row, _ = measure(args, True, tx, ref, channels, args.width,
                         tile_w, args.cols,
                         f"tiled_t{tile_w}_{args.cols}c_d{args.fifo_depth}_{args.l2_mode}")
        rows.append(row)
    for r in rows:
        print(f"{r['mode']:<9} {r['tile']:>5} {r['cols']:>4} {r['cpc']:>3} "
              f"{r['n_tiles']:>5} {r['fifo']:>4} {r['npu']:>9.1f} "
              f"{r['npu_min']:>9.1f} {r['e2e']:>9.1f} "
              f"{r['rel']:>9.3e} {r['max_abs']:>9.3e}")

    if len(rows) == 2:
        b, t = rows
        print(f"\nspeedup vs baseline: NPU {b['npu'] / t['npu']:.2f}x, "
              f"e2e {b['e2e'] / t['e2e']:.2f}x; "
              f"vs host: NPU {t['npu'] / host_us:.2f}x (lower is better)")
    assert all(r["rel"] <= 2e-3 for r in rows), "parity gate failed"


if __name__ == "__main__":
    main()
