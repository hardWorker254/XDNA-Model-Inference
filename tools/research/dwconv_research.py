#!/usr/bin/env python3
# NpuEmbeddings -- standalone research prototype: real depthwise 3x3 on npu1.
#
# SPDX-License-Identifier: Apache-2.0
#
# This does NOT touch the shipping design sets. It builds one small IRON
# program around kernels/dwconv9x3_bf16.cc and measures it against a host
# reference, so the question "can depthwise run on this board at all" gets a
# MEASURED answer rather than an argument about GEMM legality.
#
# Two modes:
#   baseline -- one tile per core, single acquire/compute/release, no overlap
#               (the 1009 us design recorded in XDNA_ENABLEMENT_PLAN.md).
#   tiled    -- width split into --tile-width slices, a range_ loop in every
#               core and depth=2 ObjectFifos (ping-pong) so the L3/L2/L1 DMAs
#               overlap with compute. One core can own several channels
#               (--cols 2 with 16 channels => 2 channels/core), which halves
#               the number of columns/FIFOs at the cost of fewer live cores.
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
from aie.iron import (CompileTime, In, ObjectFifo, Out, Program, Runtime,
                      TaskGroup, Worker)
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name
from aie.helpers.taplib import TensorTiler2D
from aie.iron.kernel import ExternalFunction
from aie.utils.trace import TraceConfig

# Research prototype: iron's ExternalFunction validates the numpy dtype of a
# MemRefValue against ml_dtypes.bfloat16, but a MemRefValue carries an MLIR
# bf16 TYPE object, so the check can never pass in this wheel version. Shape
# and rank are already guaranteed by the ObjectFifo types below.
ExternalFunction._validate_arg = lambda self, index, arg, expected_ty: None
from aie.iron.kernels._common import _detect_arch, _include_dirs
from aie.utils import config as aie_config
from aie.utils.benchmark import run_iters

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
KERNELS = REPO / "kernels"
STRIDE_EXTRA = 32  # slack in each of the three padded rows (32-lane loads)
TRACE_DIR = Path("/tmp/opencode")


def chan_block(tile_w: int) -> int:
    """bf16 elements per channel block, 32-byte aligned for load_v<16>."""
    stride = tile_w + STRIDE_EXTRA
    return ((3 * stride + 10 + 15) // 16) * 16


def _kernel(symbol: str, arg_types):
    include = _include_dirs()
    include.insert(0, str(KERNELS))
    include.append(str(Path(aie_config.cxx_header_path()) / "aie_kernels"))
    include.append(str(Path(aie_config.cxx_header_path()) / "aie_kernels" / _detect_arch()))
    return ExternalFunction(
        symbol,
        source_file=str(KERNELS / "dwconv9x3_bf16.cc"),
        arg_types=arg_types,
        include_dirs=include,
    )


def build_program(dev, channels: int, width: int, n_cols: int, tile_w: int,
                  fifo_depth: int = 2, use_tiled_kernel: bool = True,
                  l2_mode: str = "per_tile", trace_config=None,
                  trace_row: int = 0, trace_col: int = 0,
                  trace_egress_col: int = 0):
    """Build the IRON program.

    channels   -- total channels (16 in the acceptance test)
    n_cols     -- AIE columns used (4 = one channel/core, 2 = two/core, ...)
    tile_w     -- outputs per core iteration; n_tiles = width // tile_w
    fifo_depth -- ObjectFifo depth; 2 is ping-pong
    l2_mode   -- "per_tile": one L3->L2 fill/drain per tile (many runtime ops);
                 "per_col": one L3->L2 fill/drain per column with all tiles,
                            split into per-tile L2->L1 fifos in the core. Same
                            overlap, 1/n_tiles of the runtime task submissions.
    """
    n_rows = 4
    assert l2_mode in ("per_tile", "per_col")
    assert channels % (n_rows * n_cols) == 0, \
        f"{channels} channels do not split over {n_rows}x{n_cols} cores"
    cpc = channels // (n_rows * n_cols)  # channels per core
    assert width % tile_w == 0, f"width {width} is not a multiple of {tile_w}"
    n_tiles = width // tile_w
    if not use_tiled_kernel:
        assert n_tiles == 1 and cpc == 1, \
            "baseline kernel handles exactly one channel and one tile"

    stride = tile_w + STRIDE_EXTRA
    # With cpc>1 the second channel starts at `in + chan_in_n`; load_v<16>
    # needs a 32-byte-aligned address, hence the 16-element (32 B) rounding.
    chan_in_n = chan_block(tile_w)
    chan_out_n = tile_w
    core_in_n = cpc * chan_in_n
    core_out_n = cpc * chan_out_n

    chan_in_ty = np.ndarray[(chan_in_n,), np.dtype[bfloat16]]
    chan_out_ty = np.ndarray[(chan_out_n,), np.dtype[bfloat16]]
    core_in_ty = np.ndarray[(core_in_n,), np.dtype[bfloat16]]
    core_out_ty = np.ndarray[(core_out_n,), np.dtype[bfloat16]]

    if use_tiled_kernel:
        kernel = _kernel("dwconv9x3_bf16_tiled",
                         [core_in_ty, core_out_ty, np.int32, np.int32,
                          np.int32, np.int32])

        def core_fn(inp, out, kern):
            # depth=2 fifo + range_ loop = DMA of tile t+1 overlaps the
            # sliding_mul work on tile t.
            for _ in range_(n_tiles):
                ai = inp.acquire(1)
                ao = out.acquire(1)
                kern(ai, ao, tile_w, stride, cpc, chan_in_n)
                inp.release(1)
                out.release(1)
    else:
        kernel = _kernel("dwconv9x3_bf16",
                         [core_in_ty, core_out_ty, np.int32, np.int32])

        def core_fn(inp, out, kern):
            ai = inp.acquire(1)
            ao = out.acquire(1)
            kern(ai, ao, width, stride)
            inp.release(1)
            out.release(1)

    if l2_mode == "per_col":
        # One L2 object holds [row][tile] blocks for the whole column.
        l2_in_ty = np.ndarray[(n_rows * n_tiles * core_in_n,), np.dtype[bfloat16]]
        l2_out_ty = np.ndarray[(n_rows * n_tiles * core_out_n,), np.dtype[bfloat16]]
        in_offsets = [(r * n_tiles + t) * core_in_n
                      for r in range(n_rows) for t in range(n_tiles)]
        out_offsets = [(r * n_tiles + t) * core_out_n
                       for r in range(n_rows) for t in range(n_tiles)]
        n_splits = n_rows * n_tiles
        split_names_in = [f"dw_in_{col}_{r}_{t}" for col in range(n_cols)
                          for r in range(n_rows) for t in range(n_tiles)]
        split_names_out = [f"dw_out_{col}_{r}_{t}" for col in range(n_cols)
                           for r in range(n_rows) for t in range(n_tiles)]
    else:
        l2_in_ty = np.ndarray[(n_rows * core_in_n,), np.dtype[bfloat16]]
        l2_out_ty = np.ndarray[(n_rows * core_out_n,), np.dtype[bfloat16]]
        in_offsets = [core_in_n * r for r in range(n_rows)]
        out_offsets = [core_out_n * r for r in range(n_rows)]
        n_splits = n_rows
        split_names_in = [f"dw_in_{col}_{r}" for col in range(n_cols)
                          for r in range(n_rows)]
        split_names_out = [f"dw_out_{col}_{r}" for col in range(n_cols)
                           for r in range(n_rows)]

    def make_core_col(ins_r, outs_r):
        def core_col(*args):
            # IRON passes every Worker operand as a separate positional arg.
            inps = args[:n_tiles]
            outs = args[n_tiles:2 * n_tiles]
            kern = args[-1]
            # Static unroll: inps[t] must be a Python index into the list of
            # fifo handles (range_ would give an SSA index). Each acquire
            # triggers its own L2->L1 DMA, so tiles still ping-pong.
            for t in range(n_tiles):
                ai = inps[t].acquire(1)
                ao = outs[t].acquire(1)
                kern(ai, ao, tile_w, stride, cpc, chan_in_n)
                inps[t].release(1)
                outs[t].release(1)
        return core_col

    workers = [[None] * n_cols for _ in range(n_rows)]
    in_fifos, out_fifos = [], []
    for col in range(n_cols):
        fi = ObjectFifo(l2_in_ty, name=f"dw_in_{col}", depth=fifo_depth)
        fo = ObjectFifo(l2_out_ty, name=f"dw_out_{col}", depth=fifo_depth)
        ins = fi.cons().split(in_offsets, obj_types=[core_in_ty] * n_splits,
                              names=split_names_in[col * n_splits:(col + 1) * n_splits])
        outs = fo.prod().join(out_offsets, obj_types=[core_out_ty] * n_splits,
                              names=split_names_out[col * n_splits:(col + 1) * n_splits],
                              depths=[fifo_depth] * n_splits)
        for r in range(n_rows):
            tr = 1 if (trace_config is not None and
                       trace_row == r and trace_col == col) else None
            if l2_mode == "per_col":
                ins_r = [ins[r * n_tiles + t].cons() for t in range(n_tiles)]
                outs_r = [outs[r * n_tiles + t].prod() for t in range(n_tiles)]
                fn, operands = make_core_col(ins_r, outs_r), ins_r + outs_r + [kernel]
            else:
                fn, operands = core_fn, [ins[r].cons(), outs[r].prod(), kernel]
            workers[r][col] = Worker(fn, operands, tile=Tile(col, r + 2),
                                     stack_size=0x2000, trace=tr)
        in_fifos.append(fi)
        out_fifos.append(fo)

    if l2_mode == "per_col":
        # Host layout: [col][row][tile][channel-in-core][chan_in_n].
        x_ty = np.ndarray[(channels * n_tiles * chan_in_n,),
                          np.dtype[bfloat16]]
        y_ty = np.ndarray[(channels * n_tiles * chan_out_n,),
                          np.dtype[bfloat16]]
        x_taps = TensorTiler2D.simple_tiler(
            (channels * n_tiles, chan_in_n),
            (n_rows * n_tiles * cpc, chan_in_n))
        y_taps = TensorTiler2D.simple_tiler(
            (channels * n_tiles, chan_out_n),
            (n_rows * n_tiles * cpc, chan_out_n))
    else:
        # Host layout: [tile][channel][chan_in_n]. A tap covers one tile of
        # all channels belonging to a column (n_rows * cpc channels).
        x_ty = np.ndarray[(n_tiles * channels * chan_in_n,), np.dtype[bfloat16]]
        y_ty = np.ndarray[(n_tiles * channels * chan_out_n,), np.dtype[bfloat16]]
        x_taps = TensorTiler2D.simple_tiler(
            (n_tiles * channels, chan_in_n), (n_rows * cpc, chan_in_n))
        y_taps = TensorTiler2D.simple_tiler(
            (n_tiles * channels, chan_out_n), (n_rows * cpc, chan_out_n))

    n_ops = n_cols if l2_mode == "per_col" else n_tiles * n_cols

    def sequence(X, Y, x_prods, y_conss):
        tg = TaskGroup()
        for idx in range(n_ops):
            col = idx % n_cols
            x_prods[col].fill(X, tap=x_taps[idx], group=tg)
            y_conss[col].drain(Y, tap=y_taps[idx], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence,
                 [x_ty, y_ty,
                  [f.prod() for f in in_fifos],
                  [f.cons() for f in out_fifos]])

    program = Program(dev, rt, workers=[w for row in workers for w in row])
    if trace_config is not None:
        program.enable_trace(trace_config.trace_size,
                             workers=[workers[trace_row][trace_col]],
                             egress_shim_col=trace_egress_col)
    return program.resolve_program(), workers


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def dw_array(X: In, Y: Out, *, channels: CompileTime[int],
             width: CompileTime[int], tile_w: CompileTime[int],
             chan_n: CompileTime[int], n_cols: CompileTime[int],
             fifo_depth: CompileTime[int] = 2,
             use_tiled_kernel: CompileTime[bool] = True,
             l2_mode: CompileTime[str] = "per_tile",
             trace_config: CompileTime[TraceConfig | None] = None,
             trace_row: CompileTime[int] = 0, trace_col: CompileTime[int] = 0,
             trace_egress_col: CompileTime[int] = 0):
    assert chan_n == chan_block(tile_w), "stale JIT cache: chan_n changed"
    return build_program(iron.get_current_device(), channels, width, n_cols,
                         tile_w, fifo_depth, use_tiled_kernel, l2_mode,
                         trace_config, trace_row, trace_col,
                         trace_egress_col)[0]


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
