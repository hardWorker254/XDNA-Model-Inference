# NpuEmbeddings -- where the numbers come from, decided once per model.
#
# THE DECISION, AND WHY IT IS PER MODEL
#
# Phase 3 of the plan makes safetensors the primary weight source and keeps
# ONNX as the fallback. The plan's table also says which models cannot move,
# and the reasons are not about weights at all:
#
#   * PaddleOCR, MediaPipe hands, MediaPipe mppose, YOLOv8n-pose: the ONNX file
#     is the GRAPH -- op list, folded BatchNorm, anchors read out of
#     DepthToSpace, the head's geometry constants. MediaPipe's checkpoints have
#     no safetensors anywhere.
#   * the 16 transformer models (BERT x11, Whisper x5, ViT, Gemma, XLM-R): the
#     graph is this project's recipe, so only the numbers are needed, and a
#     flat checkpoint is strictly better at that.
#
# So the choice is made by asking which files are on disk, not by a constant:
# a transformer model that has `model.safetensors` uses it, one that does not
# still packs from ONNX, and a vision packer keeps calling with the ONNX path
# because that is what its family packs from.
#
# `prefer` exists for the gate: proving that the two sources give the same
# container needs a way to say "this one, whether or not the other is there".

from __future__ import annotations

from pathlib import Path

from .base import (
    WeightError,
    WeightSource,
    checkpoint_digest,
    sha256_file,
    weights_digest,
)
from .onnx import OnnxWeightsSource
from .safetensors import SafetensorsWeights

__all__ = [
    "SAFETENSORS_NAME",
    "ONNX_NAME",
    "OnnxWeightsSource",
    "SafetensorsWeights",
    "WeightError",
    "WeightSource",
    "checkpoint_digest",
    "export_digest",
    "has_safetensors",
    "open_model",
    "prefer",
    "set_prefer",
    "sha256_file",
    "weights_digest",
]

#: Where a checkpoint's flat weights live inside a model directory, and where
#: the ONNX export of the same checkpoint does. Both are relative to the model
#: directory `reference/fetch_model.py` populates.
SAFETENSORS_NAME = "model.safetensors"
ONNX_NAME = "onnx/model.onnx"

# THE PACK'S CHOICE, kept here rather than in each packer.
#
# The packers are four modules and two of them (whisper, vit) open their own
# weights, so a preference living in pack_npue would be invisible to exactly
# the branches that needed it most. One module-level answer, set once from
# --weights-source, read by every reader that does not say otherwise.
_PREFER = "auto"
_PREFERS = ("auto", "safetensors", "onnx")


def set_prefer(name: str) -> None:
    """Set the source a pack reads: 'auto', 'safetensors' or 'onnx'."""
    global _PREFER
    if name not in _PREFERS:
        raise WeightError(f"weights source {name!r}; expected one of {_PREFERS}")
    _PREFER = name


def prefer() -> str:
    """The source a pack reads unless it says otherwise."""
    return _PREFER


def export_digest(model_dir):
    """The digest of the ONNX export of this model, when it is still on disk.

    Goldens, `CHECKPOINT.json` pins and provenance records in this tree name
    the EXPORT, because that is what they were made from. A pack from
    model.safetensors records that digest too (`export_sha256`) so those
    records keep pointing at the right weights; the equality between the two
    sources is what tools/verify/verify_weights_source.py proves.

    None when there is no export: a checkpoint on its own has no export digest
    to name, and inventing one would be the lie this field exists to avoid.
    """
    model_dir = Path(model_dir)
    if (model_dir / ONNX_NAME).is_file():
        w = OnnxWeightsSource(model_dir / ONNX_NAME)
        try:
            return w.digest
        finally:
            w.close()
    encoder = model_dir / "onnx" / "encoder_model.onnx"
    decoder = model_dir / "onnx" / "decoder_model.onnx"
    if encoder.is_file() and decoder.is_file():
        # The two-file export (whisper): one digest over the whole model, which
        # is the rule `model_digest` documents -- hashing only the graphs would
        # leave every weight in the .onnx_data side files unguarded.
        from .onnx import model_digest

        return model_digest(encoder, decoder)
    return None


def has_safetensors(model_dir) -> bool:
    return (Path(model_dir) / SAFETENSORS_NAME).is_file()


def open_model(model_dir, *, strip="", rename=(), prefer=None) -> WeightSource:
    """Open the weight source for a model directory.

    `prefer` is "auto" (safetensors when present, ONNX otherwise),
    "safetensors" or "onnx"; None means the module-wide choice from
    `set_prefer` (which is "auto" unless --weights-source said otherwise).
    An explicit choice that is not on disk REFUSES instead of falling back: a
    gate that silently measured the other source would report equality between
    a thing and itself.

    `strip` and `rename` are the ONNX reader's tools for undoing what an
    exporter did to a name (a root added, a mid-name rewrite). A flat
    checkpoint has the checkpoint's own names, so passing them with a
    safetensors source is a contradiction and is refused -- silently ignoring
    them would pack under names the caller did not get.
    """
    model_dir = Path(model_dir)
    st_path = model_dir / SAFETENSORS_NAME
    onnx_path = model_dir / ONNX_NAME
    prefer = _PREFER if prefer is None else prefer

    if prefer not in _PREFERS:
        raise WeightError(f"prefer={prefer!r}; expected one of {_PREFERS}")
    if prefer == "safetensors" and not st_path.is_file():
        raise WeightError(
            f"{model_dir}: --weights-source safetensors, but {SAFETENSORS_NAME} "
            f"is not there"
        )
    if prefer == "onnx" and not onnx_path.is_file():
        raise WeightError(
            f"{model_dir}: --weights-source onnx, but {ONNX_NAME} is not there"
        )

    use_safetensors = prefer == "safetensors" or (
        prefer == "auto" and st_path.is_file()
    )
    if use_safetensors:
        # `strip`/`rename` describe what an EXPORTER did to a name -- a root it
        # added, a mid-name rewrite. A flat checkpoint carries the checkpoint's
        # own names, so there is nothing to repair and the arguments are simply
        # not used. Refusing them here was wrong for the one model that passes
        # them unconditionally: embeddinggemma's branch always hands its export
        # repair to `load_model`, and with model.safetensors present the pack
        # would have refused a source that needs no repair at all. What would
        # catch a genuinely needed rename is the gate, not this refusal: two
        # packs of one checkpoint either agree byte for byte or say why.
        return SafetensorsWeights(st_path)
    if not onnx_path.is_file():
        raise WeightError(
            f"{model_dir}: no weights -- neither {SAFETENSORS_NAME} nor "
            f"{ONNX_NAME} is there"
        )
    return OnnxWeightsSource(onnx_path, strip=strip, rename=rename)
