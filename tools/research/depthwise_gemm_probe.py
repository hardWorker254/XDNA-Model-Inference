#!/usr/bin/env python3
# NpuEmbeddings -- research probe: can a depthwise 3x3 be done with the
# EXISTING dense GEMM design by materialising the block-diagonal B?
#
# SPDX-License-Identifier: Apache-2.0
#
# out[m, c] = sum_{t=0..8} x[m, t, c] * w[c, t]
#
# A dense GEMM computes sum_k A[m, k] * B[k, n].  Set
#   A[m, c*9 + t] = x[m, t, c]
#   B[c*9 + t, c] = w[c, t]          (block diagonal; everything else zero)
# so A @ B = [M, C] is EXACTLY the depthwise result -- every output column is
# used, no discarding.  The cost is the B panel: 9*C*C entries for 9*C weights.
# This probe measures that cost on the board instead of estimating it.
#
# Usage:
#   PATH=/opt/xilinx/xrt/bin:$PATH PYTHONPATH=/opt/xilinx/xrt/python \
#     ./.venv/bin/python tools/research/depthwise_gemm_probe.py \
#     --channels 16 32 64 128 256 --m 1024 --iters 20

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import kernels, str_to_dtype
from aie.iron.device import from_name
from aie.utils.benchmark import run_iters

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from gemm_pretiled import pretiled_array  # noqa: E402
from npue import tile_b  # noqa: E402

# The shipped conv64x128 stream on npu1: M=1024, K=64, N=128, tile 64x64x32.
TILE_M, TILE_K, TILE_N, COLS = 64, 64, 32, 4
TAPS = 9


def probe(channels: int, rows: int, iters: int, warmup: int) -> dict:
    k_logical = TAPS * channels
    k_pad = ((k_logical + TILE_K - 1) // TILE_K) * TILE_K
    n_pad = ((channels + TILE_N * COLS - 1) // (TILE_N * COLS)) * (TILE_N * COLS)
    m_pad = ((rows + TILE_M * 4 - 1) // (TILE_M * 4)) * (TILE_M * 4)

    dt_in, dt_out = str_to_dtype("bf16"), str_to_dtype("f32")
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((m_pad, TAPS, channels)) * 0.1).astype(np.float32)
    w = (rng.standard_normal((channels, TAPS)) * 0.1).astype(np.float32)
    ref = np.einsum("mtc,ct->mc", x, w, optimize=True)  # [M, C], float64-ish

    a_np = np.zeros((m_pad, k_pad), dtype=np.float32)
    for c in range(channels):
        a_np[:, c * TAPS:(c + 1) * TAPS] = x[:, :, c]
    b_logical = np.zeros((k_pad, n_pad), dtype=np.float32)
    for c in range(channels):
        b_logical[c * TAPS:(c + 1) * TAPS, c] = w[c]

    a = iron.zeros((m_pad, k_pad), dtype=dt_in, device="npu")
    b = iron.zeros((k_pad, n_pad), dtype=dt_in, device="npu")
    c_out = iron.zeros(m_pad * n_pad, dtype=dt_out, device="npu")
    a[:] = a_np.astype(dt_in)

    r, s, t = kernels.mm(
        dim_m=TILE_M, dim_k=TILE_K, dim_n=TILE_N,
        input_dtype=dt_in, output_dtype=dt_out,
        b_col_maj=False, c_col_maj=False, use_chess=False,
        emulate_bf16_mmul_with_bfp16=False, vectorized=True,
    ).mac_dims
    b.numpy().reshape(-1).view(np.uint16)[:] = tile_b(
        b_logical.astype(dt_in).view(np.uint16), TILE_K, TILE_N, s, t,
        order="k,n",
    )

    kw = dict(
        M=m_pad, K=k_pad, N=n_pad,
        m=TILE_M, k=TILE_K, n=TILE_N, n_aie_cols=COLS,
        dtype_in_str="bf16", dtype_out_str="f32",
        emulate_bf16_mmul_with_bfp16=False,
        pretiled=True, tile_order="k,n", inner_st=True,
        trace_config=None,
    )

    t0 = time.perf_counter()
    pretiled_array(a, b, c_out, **kw)  # compile + one run
    compile_s = time.perf_counter() - t0

    got = c_out.numpy().reshape(m_pad, n_pad)[:, :channels].astype(np.float64)
    rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    max_abs = float(np.max(np.abs(got - ref)))

    t0 = time.perf_counter()
    res = run_iters(pretiled_array, a, b, c_out,
                    warmup=warmup, iters=iters, **kw)
    bench_s = time.perf_counter() - t0
    npu_us = getattr(getattr(res, "npu", None), "avg_us", None)
    e2e_us = getattr(getattr(res, "e2e", None), "avg_us", None)

    useful_macs = m_pad * TAPS * channels
    dense_macs = m_pad * k_pad * n_pad
    return dict(
        channels=channels, rows=m_pad, k=k_pad, n=n_pad,
        panel_kb=k_pad * n_pad * 2 / 1024,
        useful_panel_entries=TAPS * channels,
        panel_entries=k_pad * n_pad,
        overhead=dense_macs / useful_macs,
        rel_fro=rel, max_abs=max_abs,
        compile_s=compile_s, bench_s=bench_s,
        npu_avg_us=npu_us, e2e_avg_us=e2e_us,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    iron.set_current_device(from_name("npu1", n_cols=None))
    print(f"depthwise-via-dense-GEMM probe: M={args.m}, iters={args.iters}")
    print(f"{'C':>5} {'Kpad':>6} {'Npad':>6} {'panel KB':>10} {'MAC x':>8} "
          f"{'relfro':>10} {'NPU us':>10} {'e2e us':>10}")
    for c in args.channels:
        try:
            r = probe(c, args.m, args.iters, args.warmup)
        except Exception as exc:  # noqa: BLE001 - research probe, report everything
            print(f"{c:>5} FAILED: {type(exc).__name__}: {exc}")
            continue
        print(f"{c:>5} {r['k']:>6} {r['n']:>6} {r['panel_kb']:>10.1f} "
              f"{r['overhead']:>8.1f} {r['rel_fro']:>10.2e} "
              f"{(r['npu_avg_us'] or float('nan')):>10.1f} "
              f"{(r['e2e_avg_us'] or float('nan')):>10.1f}")


if __name__ == "__main__":
    main()
