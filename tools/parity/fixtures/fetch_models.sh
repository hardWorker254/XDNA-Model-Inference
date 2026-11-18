#!/usr/bin/env bash
# The model files the PARITY fixtures need -- fetched from HuggingFace, pinned
# by sha256 where the repository records one.
#
#   tools/parity/fixtures/fetch_models.sh [REPO]
#
# REPO defaults to the repository root. Files land where the packer reads
# them: models/<name>/onnx/model.onnx for the transformer families, and
# models/<name>/<file>.onnx for the two MediaPipe models and the pose model,
# whose checkpoints are not in an onnx/ directory.
#
# WHAT THIS IS FOR. Phase 0 of RUST_MIGRATION_PLAN.md records golden
# transcripts of the reference binaries. A transcript of a command that refuses
# because the model is missing is not a transcript of the command -- it is a
# transcript of the absence -- so the fixtures have to be models that RUN.
#
# WHY THESE SEVEN and not all nineteen. One model per family is enough to
# cover every endpoint and every subcommand, and each extra model is another
# gigabyte and another thing that can be downloaded wrong. The coverage this
# buys is listed in the header of tools/parity/cases.py.
#
# WHY f32 CONTAINERS. The packer's default is bf16 -- a pre-tiled panel for
# the ARRAY -- and a bf16 container needs a design set, which needs MLIR-AIE to
# export. The CPU control container is plain row-major f32 and needs nothing
# beside it, so it is the one that can be produced and compared anywhere.
#
# NOT HERE: the design sets (runtime/artifacts/<model>/artifacts_npu1/) and the
# AMD/Xilinx wheels. The first are built, not downloaded; the second come from
# bootstrap.sh.

set -euo pipefail

# The repository root. This script lives at <repo>/tools/parity/fixtures/, so
# the root is three levels up -- not four, which is what the script used to
# say and which put the downloads in the PARENT of the repository, where
# nothing looks for them.
REPO="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"

# name|repo|file|destination|sha256 (empty = unpinned)
MANIFEST=$(cat <<'EOF'
all-MiniLM-L6-v2|sentence-transformers/all-MiniLM-L6-v2|onnx/model.onnx|models/all-MiniLM-L6-v2/onnx/model.onnx|6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452
vit-base-patch16-224|onnx-community/vit-base-patch16-224-ONNX|onnx/model.onnx|models/vit-base-patch16-224/onnx/model.onnx|
whisper-tiny|onnx-community/whisper-tiny-ONNX|onnx/encoder_model.onnx|models/whisper-tiny/onnx/encoder_model.onnx|8dd994fe489eaa52c229d3ecbd7aeb330ed474c1659017bb7c1f4f4768dd567e
whisper-tiny|onnx-community/whisper-tiny-ONNX|onnx/decoder_model.onnx|models/whisper-tiny/onnx/decoder_model.onnx|7e844cce0ac74edaa4592c055176f43ede671de55a3c6cb3d863b56342eeaabc
nomic-embed-text-v1.5|nomic-ai/nomic-embed-text-v1.5|onnx/model.onnx|models/nomic-embed-text-v1.5/onnx/model.onnx|147d5aa88c2101237358e17796cf3a227cead1ec304ec34b465bb08e9d952965
yolov8n-pose|Xenova/yolov8-pose-onnx|yolov8n-pose.onnx|models/yolov8n-pose/yolov8n-pose.onnx|04f6d2416266f2aba6c5ba8b26de33ed9eba3279f972ac02e33f9f1366547586
mediapipe-hands|opencv/palm_detection_mediapipe|palm_detection_mediapipe_2023feb.onnx|models/mediapipe-hands/palm_detection_mediapipe_2023feb.onnx|78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c
mediapipe-hands|opencv/handpose_estimation_mediapipe|handpose_estimation_mediapipe_2023feb.onnx|models/mediapipe-hands/handpose_estimation_mediapipe_2023feb.onnx|db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c
mediapipe-pose|opencv/person_detection_mediapipe|person_detection_mediapipe_2023mar.onnx|models/mediapipe-pose/person_detection_mediapipe_2023mar.onnx|47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f
mediapipe-pose|opencv/pose_estimation_mediapipe|pose_estimation_mediapipe_2023mar.onnx|models/mediapipe-pose/pose_estimation_mediapipe_2023mar.onnx|9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f
EOF
)

sha_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

total=0
while IFS='|' read -r name repo file dest want; do
    [ -z "$name" ] && continue
    total=$((total + 1))
    if [ -f "$dest" ]; then
        if [ -z "$want" ] || [ "$(sha_of "$dest")" = "$want" ]; then
            echo "  cached   $dest"
            continue
        fi
        echo "  stale    $dest -- digest mismatch, re-downloading"
        rm -f "$dest"
    fi
    url="https://huggingface.co/$repo/resolve/main/$file"
    mkdir -p "$(dirname "$dest")"
    echo "  fetch    $dest"
    curl -fsSL --retry 3 -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
    if [ -n "$want" ]; then
        got="$(sha_of "$dest")"
        if [ "$got" != "$want" ]; then
            rm -f "$dest"
            echo "error: sha256 mismatch for $dest" >&2
            echo "       want $want" >&2
            echo "       got  $got" >&2
            exit 1
        fi
        echo "           verified"
    fi
done <<< "$MANIFEST"

echo
echo "$total file(s) in place."
echo "Now:  .venv/bin/python tools/pack/pack_npue.py --model-dir models/<name> --out models/<name>.npue --dtype f32 --no-embed-artifacts"
