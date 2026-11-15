# NpuEmbeddings -- the weight source protocol, and the two questions a gate asks.
#
# WHY A PROTOCOL AND NOT A DICT
#
# The packer used to build a dict of every tensor (`{k: w.array(k) ...}`), which
# works and is what `pack_npue.load()` still returns -- but the dict is the one
# place where "the numbers came from a checkpoint" stops being knowable: once
# they are in a dict of numpy arrays, nothing says which file they came from,
# whether the orientation was already corrected, or whether the source could
# even hold the names the packer expects.
#
# The protocol keeps that answer attached. It is deliberately tiny -- names,
# one array at a time, close -- because the readers are streaming readers: a
# 3.1 GB checkpoint must not be resident just because a packer wants to hash
# it, and `array(name)` is the only call that materialises anything.

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

# THE DTYPES A CHECKPOINT MAY HOLD, and what each becomes on this side.
#
# Everything float becomes float32 -- F64 narrowed would be lossy and is
# refused; F16 and BF16 widen losslessly (bf16's 8 exponent bits and 7 mantissa
# bits are a subset of fp32's, so the shift is exact). Integer tensors stay
# integer: they are vocabulary tables and quantisation scales, and widening
# them to float would silently change what the packer does next.
#
# Anything else -- F8_E4M3, F8_E5M2, I4, U4 -- is REFUSED BY NAME. They exist
# in the wild, they are not what any checkpoint in this tree carries, and a
# reader that guessed at them would produce plausible numbers.
FLOAT_DTYPES = {
    "F32": np.float32,
    "F16": np.float32,
    "BF16": np.float32,
}


class WeightError(SystemExit):
    """A checkpoint that cannot be read, refused with a reason.

    SystemExit rather than a plain exception: every caller here is a packer,
    the refusal is the answer to the user's command, and the existing packers
    already refuse that way (`raise SystemExit(...)`), so a gate can print the
    message without a traceback.
    """


@runtime_checkable
class WeightSource(Protocol):
    """A checkpoint's tensors, under the names the checkpoint itself used."""

    #: "safetensors" or "onnx" -- which file was read, for the provenance
    #: record in the container.
    format: str
    #: The digest that identifies THIS checkpoint (see `checkpoint_digest`).
    digest: str
    #: The files it was read from, for the same record.
    files: tuple[str, ...]

    def keys(self) -> list[str]:
        ...

    def __contains__(self, name: str) -> bool:
        ...

    def array(self, name: str) -> np.ndarray:
        """One tensor as an OWNED numpy array, in the checkpoint's orientation.

        `array()` names one tensor: a reader that returned views would fault
        once `close()` ran, and the packer holds arrays past that point.
        """

    def close(self) -> None:
        ...


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Digest of one file, read once and never held in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def checkpoint_digest(files) -> str:
    """One digest identifying a checkpoint, whatever format it is in.

    The scheme is `onnx_weights.model_digest`'s, kept identical on purpose:
    each file's own digest together with its BASENAME, in a fixed order. The
    basename matters because two checks a byte-identical file under a
    different name is not the same checkpoint; the fixed order matters because
    a fingerprint that changes with the order it was enumerated is not one.

    It is not the digest of any single file, so it cannot be mistaken for one:
    `sha256_file()` is that question.
    """
    outer = hashlib.sha256()
    for p in files:
        p = Path(p)
        outer.update(p.name.encode("utf-8"))
        outer.update(b"\0")
        outer.update(sha256_file(p).encode("ascii"))
        outer.update(b"\n")
    return outer.hexdigest()


def weights_digest(source: WeightSource, only=None) -> str:
    """A digest of the NUMBERS, independent of the file that holds them.

    This is the question phase 3's gate asks: is a safetensors checkpoint the
    same weights as the ONNX export of it? File digests cannot answer that --
    the exporters rewrite names and layouts, so the files must differ -- and
    the packers cannot answer it either, because they quantise.

    So each tensor contributes its name, dtype, shape and raw bytes, in sorted
    name order, and the result is the same for both sources exactly when the
    dict they serve is the same. That is the property the container equality
    gate rests on: equal numbers in, equal container out.

    `only` restricts the digest to a set of names. The gate passes the ONNX
    source's keys, because a flat checkpoint legitimately holds tensors an
    export dropped (a position-index buffer, an unused pooler) and the question
    there is whether the weights the graph uses survived the round trip.
    """
    outer = hashlib.sha256()
    for name in sorted(source.keys() if only is None else only):
        a = np.ascontiguousarray(source.array(name))
        outer.update(name.encode("utf-8"))
        outer.update(b"\0")
        outer.update(str(a.dtype).encode("ascii"))
        outer.update(b"\0")
        outer.update(repr(tuple(a.shape)).encode("ascii"))
        outer.update(b"\0")
        outer.update(a.tobytes())
        outer.update(b"\n")
    return outer.hexdigest()
