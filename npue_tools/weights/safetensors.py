# NpuEmbeddings -- safetensors, read as the checkpoint's own tensors.
#
# WHAT THE FORMAT GIVES THAT ONNX DOES NOT
#
# `model.safetensors` is a flat dict {name: bytes}: an 8-byte little-endian
# header length, a JSON header mapping every tensor name to its dtype, shape
# and byte range, then the bytes. Nothing in it is code, nothing is a graph,
# and the names are the ones the checkpoint had -- `encoder.layer.0.attention
# .self.query.weight`, not `onnx::MatMul_977`.
#
# That is the whole reason this reader exists. The ONNX reader
# (tools/lib/onnx_weights.py) has to undo what the exporter did -- transpose
# MatMul operands back, recover 36 names out of a graph, add back a `model.`
# root -- while a flat checkpoint needs none of it. The header is the whole
# specification.
#
# WHAT IT COSTS, AND WHAT IS REFUSED
#
# Reading is mmap + a slice per tensor, so a 3.1 GB checkpoint costs the header
# (a few hundred KB of JSON) and whatever a caller asks for. `array()` returns
# an OWNED copy -- a view would fault after `close()`, and packers hold arrays
# past it.
#
# Refused by name, with a message: an unknown or unsafe dtype; a range that
# runs past the data, overlaps another tensor, or is not a whole number of
# elements for its dtype; a header that is not JSON or not an object. A
# checkpoint that lies about its own layout is not a checkpoint to guess at --
# the alternative is plausible numbers in a container.
#
#   THE `__metadata__` ENTRY is the file's own metadata dict (author, format,
#   sometimes a repo id). It is not a tensor, it has no data range, and a
#   reader that enumerated it as one would fail on the first array() call. It
#   is exposed separately and skipped here.

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path

import numpy as np

from .base import WeightError, checkpoint_digest

#: The dtype tags a header may carry, and how each is stored on this side.
#: Integer tags stay integer (vocabulary tables, scales); the float tags are
#: widened losslessly to float32 by `_convert`.
STORAGE_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.dtype(np.uint16),
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


class SafetensorsWeights:
    """One `.safetensors` file, served under its own tensor names."""

    format = "safetensors"

    def __init__(self, path):
        self.path = Path(path)
        self._file = open(self.path, "rb")
        size = self.path.stat().st_size
        if size < 8:
            self.close()
            raise WeightError(f"{self.path}: shorter than a header length")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        header_len = struct.unpack("<Q", self._mm[0:8])[0]
        # The length is the caller's claim about a file it did not write: a
        # header that runs past the data, or that is absurdly large, is a
        # broken file and not something to allocate for.
        if header_len == 0 or 8 + header_len > size:
            self.close()
            raise WeightError(
                f"{self.path}: header length {header_len} does not fit a "
                f"{size}-byte file"
            )
        raw = self._mm[8 : 8 + header_len]
        try:
            header = json.loads(raw)
        except json.JSONDecodeError as e:
            self.close()
            raise WeightError(f"{self.path}: header is not JSON: {e}") from None
        if not isinstance(header, dict):
            self.close()
            raise WeightError(f"{self.path}: header is not an object")

        self.metadata = header.pop("__metadata__", {}) or {}
        self._data_start = 8 + header_len
        self._data_len = size - self._data_start
        self._entries = {}
        for name, e in header.items():
            self._entries[name] = self._entry(name, e)
        self._check_no_overlap()
        self.files = (str(self.path),)
        self.digest = checkpoint_digest(self.files)

    # -- header ---------------------------------------------------------
    def _entry(self, name, e):
        """One header entry, validated: dtype, shape, byte range."""
        if not isinstance(e, dict):
            raise WeightError(f"{self.path}: entry {name!r} is not an object")
        dtype = e.get("dtype")
        shape = e.get("shape")
        offsets = e.get("data_offsets")
        if dtype not in STORAGE_DTYPES:
            raise WeightError(
                f"{self.path}: {name!r} has dtype {dtype!r}; this reader "
                f"knows {', '.join(sorted(STORAGE_DTYPES))}. A dtype that is "
                f"not read exactly is a checkpoint that would pack wrong."
            )
        if not isinstance(shape, list) or not all(
            isinstance(d, int) and d >= 0 for d in shape
        ):
            raise WeightError(f"{self.path}: {name!r} has shape {shape!r}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(o, int) for o in offsets)
        ):
            raise WeightError(f"{self.path}: {name!r} has offsets {offsets!r}")
        begin, end = offsets
        if begin < 0 or end < begin or end > self._data_len:
            raise WeightError(
                f"{self.path}: {name!r} spans {begin}..{end} of "
                f"{self._data_len} data bytes"
            )
        itemsize = np.dtype(STORAGE_DTYPES[dtype]).itemsize
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if end - begin != count * itemsize:
            raise WeightError(
                f"{self.path}: {name!r} is {end - begin} bytes for "
                f"{count} x {dtype} ({count * itemsize} expected) -- shape and "
                f"range disagree"
            )
        return (dtype, tuple(shape), begin, end)

    def _check_no_overlap(self):
        """Tensors may not share bytes: the file is a dict, not an arena."""
        spans = sorted(
            (begin, end, name)
            for name, (_d, _s, begin, end) in self._entries.items()
            if end > begin
        )
        for (b0, e0, n0), (b1, _e1, n1) in zip(spans, spans[1:]):
            if b1 < e0:
                raise WeightError(
                    f"{self.path}: {n0!r} and {n1!r} share bytes "
                    f"({b0}..{e0} against {b1}..) -- a header that aliases "
                    f"tensors cannot be read as a dict"
                )

    # -- the protocol ---------------------------------------------------
    def keys(self) -> list[str]:
        return list(self._entries)

    def __contains__(self, name) -> bool:
        return name in self._entries

    def info(self, name):
        dtype, shape, begin, end = self._entries[name]
        return {"dtype": dtype, "shape": shape, "bytes": end - begin}

    def array(self, name: str) -> np.ndarray:
        if name not in self._entries:
            raise WeightError(
                f"{self.path}: no tensor {name!r}. Present: "
                f"{', '.join(sorted(self._entries)[:6])}"
                + (" ..." if len(self._entries) > 6 else "")
            )
        dtype, shape, begin, end = self._entries[name]
        raw = self._mm[self._data_start + begin : self._data_start + end]
        a = np.frombuffer(raw, dtype=STORAGE_DTYPES[dtype])
        return _convert(a.reshape(shape).copy(), dtype)

    def close(self):
        mm = getattr(self, "_mm", None)
        if mm is not None:
            try:
                mm.close()
            except (BufferError, ValueError):
                # A caller holds a view into the map. Nothing here can free
                # it, and the file closes anyway; refusing to close the file
                # would be the worse failure.
                pass
            self._mm = None
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _convert(a: np.ndarray, dtype_tag: str) -> np.ndarray:
    """The stored array, as the protocol promises it.

    F32 stays as it is; F16 widens to float32 exactly; BF16 is two bytes of
    the TOP of a float32, so shifting left by 16 and reinterpreting is the
    exact inverse of the truncation that made it; integer tags and BOOL pass
    through unchanged.
    """
    if dtype_tag in ("F32", "I64", "I32", "I16", "I8", "U8", "BOOL"):
        return a
    if dtype_tag == "F16":
        return a.astype(np.float32)
    if dtype_tag == "BF16":
        bits = a.astype(np.uint32) << np.uint32(16)
        return bits.view(np.float32)
    if dtype_tag == "F64":
        # F64 is read exactly and NARROWED only where the protocol says the
        # number is float32 -- which is a real loss, so it is refused here
        # rather than done quietly. No checkpoint in this tree is F64.
        raise WeightError(
            "F64 is not narrowed to float32 silently: pack the checkpoint in "
            "F32, or teach this reader a dtype tag the packers accept"
        )
    return a
