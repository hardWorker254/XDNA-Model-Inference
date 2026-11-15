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
    "has_safetensors",
    "open_model",
    "sha256_file",
    "weights_digest",
]

#: Where a checkpoint's flat weights live inside a model directory, and where
#: the ONNX export of the same checkpoint does. Both are relative to the model
#: directory `reference/fetch_model.py` populates.
SAFETENSORS_NAME = "model.safetensors"
ONNX_NAME = "onnx/model.onnx"


def has_safetensors(model_dir) -> bool:
    return (Path(model_dir) / SAFETENSORS_NAME).is_file()


def open_model(model_dir, *, strip="", rename=(), prefer="auto") -> WeightSource:
    """Open the weight source for a model directory.

    `prefer` is "auto" (safetensors when present, ONNX otherwise),
    "safetensors" or "onnx". An explicit choice that is not on disk REFUSES
    instead of falling back: a gate that silently measured the other source
    would report equality between a thing and itself.

    `strip` and `rename` are the ONNX reader's tools for undoing what an
    exporter did to a name (a root added, a mid-name rewrite). A flat
    checkpoint has the checkpoint's own names, so passing them with a
    safetensors source is a contradiction and is refused -- silently ignoring
    them would pack under names the caller did not get.
    """
    model_dir = Path(model_dir)
    st_path = model_dir / SAFETENSORS_NAME
    onnx_path = model_dir / ONNX_NAME

    if prefer not in ("auto", "safetensors", "onnx"):
        raise WeightError(
            f"prefer={prefer!r}; expected 'auto', 'safetensors' or 'onnx'"
        )
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
        if strip or rename:
            raise WeightError(
                f"{model_dir}: strip/rename are the ONNX reader's repair for "
                f"names an exporter rewrote; {SAFETENSORS_NAME} carries the "
                f"checkpoint's own names, so the repair does not apply"
            )
        return SafetensorsWeights(st_path)
    if not onnx_path.is_file():
        raise WeightError(
            f"{model_dir}: no weights -- neither {SAFETENSORS_NAME} nor "
            f"{ONNX_NAME} is there"
        )
    return OnnxWeightsSource(onnx_path, strip=strip, rename=rename)
