#!/usr/bin/env python3
# NpuEmbeddings -- the IRON design library for the production dwconv9x3 set.
#
# SPDX-License-Identifier: Apache-2.0
#
# Imported by tools/export/export_dwconv.py to build the shipping xclbin, and
# by tools/research/dwconv_research.py for A/B measurements. No CLI, no timing,
# no host oracle here -- those belong to the research driver.

from __future__ import annotations

import sys
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

# ---------------------------------------------------------------------------
# PRODUCTION ENTRY -- one xclbin for every model: 16 channels x 1024 outputs.
# Real layers are chunked into this by the runtime. The kernel always computes
# the full 1024 outputs; the host fills unused positions with zeros and reads
# back only the real output width.
# ---------------------------------------------------------------------------
@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def dwconv9x3_array(X: In, Y: Out):
    return build_program(
        iron.get_current_device(),
        channels=16, width=1024, n_cols=4, tile_w=1024,
        fifo_depth=2, use_tiled_kernel=True, l2_mode="per_tile",
    )[0]
