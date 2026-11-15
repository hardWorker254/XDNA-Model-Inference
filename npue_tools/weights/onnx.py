# NpuEmbeddings -- ONNX, still the reader for what a flat checkpoint cannot give.
#
# This is the fallback, and for three families it is the only source there is:
#
#   PaddleOCR        -- the container is packed from `det.onnx`/`rec.onnx`,
#                       whose 318 weights live inside `Constant` nodes; the
#                       safetensors re-upload has readable names but the graph
#                       (572 nodes, BN folded into the convolutions, the
#                       Resize/ConvTranspose topology) is still ONNX, so ONNX
#                       stays the source both ways.
#   MediaPipe hands  -- no safetensors exists anywhere for these graphs, and
#   MediaPipe mppose    the weights are not the interesting part: the anchors
#                       are read out of the graph's DepthToSpace nodes.
#   YOLOv8n-pose     -- the ONNX file IS the specification (op list, the head's
#                       offsets). The one safetensors of it that exists is F16
#                       under Candle's module names, so it is not used.
#
# So this class is not dead weight to delete later; it is the reader for the
# vision families, and the reason `open_model` chooses per model instead of
# per project.
#
# It adds nothing to tools/lib/onnx_weights.py but the protocol: that reader
# already serves the checkpoint's names and orientation, which is what both
# sides of the seam must agree on.

from __future__ import annotations

import sys
from pathlib import Path

# tools/lib is not a package and does not need to become one here: this module
# is the only place in npue_tools that reaches into it, and a package-wide
# reorganisation is phase 2's job, not this file's.
_TOOLS_LIB = Path(__file__).resolve().parents[2] / "tools" / "lib"
if str(_TOOLS_LIB) not in sys.path:
    sys.path.insert(0, str(_TOOLS_LIB))

from onnx_weights import OnnxWeights, model_digest  # noqa: E402


class OnnxWeightsSource:
    """One ONNX model's tensors, under the checkpoint's own names."""

    format = "onnx"

    def __init__(self, path, strip="", rename=()):
        self.path = Path(path)
        self._w = OnnxWeights(self.path, strip=strip, rename=rename)
        self.files = tuple(self._w.source_files())
        self.digest = model_digest(self.path)

    def keys(self):
        return self._w.keys()

    def __contains__(self, name):
        return name in self._w

    def info(self, name):
        return self._w.info(name)

    def array(self, name):
        return self._w.array(name)

    def close(self):
        self._w.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
