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

# Ensure repo root and tools/lib are importable (phase 2 cleanup).
_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO), str(_REPO / "tools" / "lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.lib.onnx_weights import (  # noqa: E402
    EMBEDDINGGEMMA_RENAME,
    EMBEDDINGGEMMA_STRIP,
    WHISPER_DECODER_ONNX,
    WHISPER_ENCODER_ONNX,
    Combined,
    OnnxWeights,
    model_digest,
)


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


class WhisperOnnxSource:
    """A whisper export: TWO graphs behind one namespace.

    The export splits the model into an encoder and a decoder file, and the
    encoder's file dropped the `model.encoder.` root its tensors are named for
    -- `prefix=` puts it back. The packer asks for both halves in one dict and
    cannot sensibly be split by file, so this is the one place that knows the
    stitching, and it lives with the ONNX reader rather than in the packer: a
    second copy of "how the export spells whisper" is how the two drift.

    The digest covers the WHOLE model, not just the two graph files: an export
    large enough to trip protobuf's 2 GB limit keeps its weights in an
    `.onnx_data` side file, and hashing only the graphs would pin the tensor
    names while leaving every weight unguarded.
    """

    format = "onnx"

    def __init__(self, model_dir):
        self.path = Path(model_dir)
        self._w = Combined([
            OnnxWeights(self.path / WHISPER_ENCODER_ONNX, prefix="model.encoder."),
            OnnxWeights(self.path / WHISPER_DECODER_ONNX),
        ])
        self.digest = model_digest(self.path / WHISPER_ENCODER_ONNX,
                                   self.path / WHISPER_DECODER_ONNX)

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
