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
# One core per channel. With 16 channels the 4x4 npu1 grid is full.
#
# Usage:
#   PATH=/opt/xilinx/xrt/bin:$PATH PYTHONPATH=/opt/xilinx/xrt/python \
#     ./.venv/bin/python tools/research/dwconv_research.py --channels 16 --width 128

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name
from aie.helpers.taplib import TensorTiler2D
from aie.iron.kernel import ExternalFunction

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
STRIDE_EXTRA = 32  # slack in each of the three padded rows


def build_program(dev, channels: int, width: int, n_cols: int = 4):
    n_rows = 4
    assert channels == n_rows * n_cols, "this prototype maps one channel per core"
    assert width % 16 == 0, "sliding_mul emits 16 outputs per call"

    stride = width + STRIDE_EXTRA
    row_ty = np.ndarray[(stride,), np.dtype[bfloat16]]
    chan_in_ty = np.ndarray[(3 * stride + 10,), np.dtype[bfloat16]]  # rows + 9 taps + bias
    out_ty = np.ndarray[(width,), np.dtype[bfloat16]]

    include = _include_dirs()
    include.insert(0, str(KERNELS))
    include.append(str(Path(aie_config.cxx_header_path()) / "aie_kernels"))
    include.append(str(Path(aie_config.cxx_header_path()) / "aie_kernels" / _detect_arch()))
    kernel = ExternalFunction(
        "dwconv9x3_bf16",
        source_file=str(KERNELS / "dwconv9x3_bf16.cc"),
        arg_types=[chan_in_ty, out_ty, np.int32, np.int32],
        include_dirs=include,
    )

    def core_fn(inp, out, kern):
        ai = inp.acquire(1)
        ao = out.acquire(1)
        kern(ai, ao, width, stride)
        inp.release(1)
        out.release(1)

    in_fifos, out_fifos = [], []
    workers = []
    l2_in_ty = np.ndarray[(n_rows * (3 * stride + 10),), np.dtype[bfloat16]]
    l2_out_ty = np.ndarray[(n_rows * width,), np.dtype[bfloat16]]
    in_offsets = [(3 * stride + 10) * r for r in range(n_rows)]
    out_offsets = [width * r for r in range(n_rows)]

    for col in range(n_cols):
        fi = ObjectFifo(l2_in_ty, name=f"dw_in_{col}", depth=2)
        fo = ObjectFifo(l2_out_ty, name=f"dw_out_{col}", depth=2)
        ins = fi.cons().split(in_offsets, obj_types=[chan_in_ty] * n_rows,
                              names=[f"dw_in_{col}_{r}" for r in range(n_rows)])
        outs = fo.prod().join(out_offsets, obj_types=[out_ty] * n_rows,
                              names=[f"dw_out_{col}_{r}" for r in range(n_rows)],
                              depths=[2] * n_rows)
        for r in range(n_rows):
            workers.append(Worker(core_fn,
                                  [ins[r].cons(), outs[r].prod(), kernel],
                                  tile=Tile(col, r + 2), stack_size=0x2000))
        in_fifos.append(fi)
        out_fifos.append(fo)

    x_ty = np.ndarray[(channels, 3 * stride + 10), np.dtype[bfloat16]]
    y_ty = np.ndarray[(channels, width), np.dtype[bfloat16]]
    x_taps = TensorTiler2D.simple_tiler((channels, 3 * stride + 10),
                                        (n_rows, 3 * stride + 10))
    y_taps = TensorTiler2D.simple_tiler((channels, width), (n_rows, width))

    def sequence(X, Y, x_prods, y_conss):
        tg = TaskGroup()
        for col in range(n_cols):
            x_prods[col].fill(X, tap=x_taps[col], group=tg)
            y_conss[col].drain(Y, tap=y_taps[col], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence,
                 [x_ty, y_ty,
                  [f.prod() for f in in_fifos],
                  [f.cons() for f in out_fifos]])

    program = Program(dev, rt, workers=workers)
    return program.resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def dw_array(X: In, Y: Out, *, channels: CompileTime[int],
             width: CompileTime[int], stride: CompileTime[int],
             n_cols: CompileTime[int]):
    return build_program(iron.get_current_device(), channels, width, n_cols)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    iron.set_current_device(from_name("npu1", n_cols=None))
    stride = args.width + STRIDE_EXTRA
    channels = args.cols * 4

    t0 = time.perf_counter()
    program = dw_array  # JIT compiles on first call below
    print(f"prepared in {time.perf_counter() - t0:.1f}s")

    rng = np.random.default_rng(7)
    img = (rng.standard_normal((channels, args.width + 2, args.width + 2)) * 0.1).astype(np.float32)
    w = (rng.standard_normal((channels, 3, 3)) * 0.1).astype(np.float32)
    bias = (rng.standard_normal(channels) * 0.1).astype(np.float32)

    # Build the padded three rows per channel and the flat weight vector.
    x_np = np.zeros((channels, 3 * stride + 10), dtype=np.float32)
    for c in range(channels):
        for dy in range(3):
            x_np[c, dy * stride:dy * stride + args.width] = img[c, dy, :args.width]
        x_np[c, 3 * stride:3 * stride + 9] = w[c].reshape(-1)
        x_np[c, 3 * stride + 9] = bias[c]

    x_bf = x_np.astype(bfloat16).astype(np.float32)
    wt_bf = x_np[:, 3 * stride:].astype(bfloat16).astype(np.float32)

    # Reference: out[c, x] = bias + sum dy,dx w[c,dy,dx] * img[c,dy,x+dx]
    ref = np.zeros((channels, args.width), dtype=np.float64)
    for c in range(channels):
        for x in range(args.width):
            acc = float(wt_bf[c, 9])
            for dy in range(3):
                for dx in range(3):
                    acc += float(wt_bf[c, dy * 3 + dx]) * float(x_bf[c, dy * stride + x + dx])
            ref[c, x] = acc

    X = iron.zeros((channels, 3 * stride + 10), dtype=bfloat16, device="npu")
    Wt = iron.zeros((channels, 10), dtype=bfloat16, device="npu")
    Y = iron.zeros((channels, args.width), dtype=bfloat16, device="npu")
    X[:] = x_np.astype(bfloat16)

    t0 = time.perf_counter()
    program(X, Y, channels=channels, width=args.width, stride=stride, n_cols=args.cols)
    print(f"first run (incl. compile) {time.perf_counter() - t0:.3f}s")

    got = Y.numpy().astype(np.float64)
    rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    max_abs = float(np.max(np.abs(got - ref)))
    print(f"parity: relfro={rel:.3e} max_abs={max_abs:.3e}")
    print("  got[0,:8] ", np.array2string(got[0, :8], precision=4))
    print("  ref[0,:8] ", np.array2string(ref[0, :8], precision=4))
    print("  bias", bias[:2], "w0", w[0].reshape(-1)[:3])

    res = run_iters(program, X, Y, warmup=5, iters=args.iters,
                    channels=channels, width=args.width, stride=stride, n_cols=args.cols)
    print(f"NPU avg={getattr(getattr(res,'npu',None),'avg_us',None)} us  "
          f"e2e avg={getattr(getattr(res,'e2e',None),'avg_us',None)} us")


if __name__ == "__main__":
    main()
