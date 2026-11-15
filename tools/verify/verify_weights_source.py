#!/usr/bin/env python3
#===----------------------------------------------------------------------===//
# Phase 3's gate: the same checkpoint, packed from two sources, is one container.
#
# THE CLAIM BEING CHECKED
#
# RUST_MIGRATION_PLAN.md phase 3 says safetensors replaces ONNX as the weight
# source for the transformer families, and that the swap has to be provably
# lossless BEFORE any runtime work depends on it: "safetensors-container byte-
# for-byte equal to the ONNX-container". Two things must therefore hold, and
# they are checked separately here because they fail for different reasons:
#
#   1. THE NUMBERS. Both readers serve the checkpoint's own names and the
#      checkpoint's own orientation, so the dicts must be identical -- same
#      key set, same shapes, same dtype, same bytes. `weights_digest` hashes
#      exactly that. A mismatch means a name was lost, a weight transposed, or
#      a dtype widened where it should not have been -- the three failure modes
#      the ONNX reader exists to undo.
#
#   2. THE CONTAINER. Packing the same dict must produce the same file. Two
#      files are compared byte for byte after masking the ONE field that is
#      SUPPOSED to differ: `source_sha256`, the digest of the checkpoint files
#      the numbers were read from (and `weights_source`, which names the format
#      when the packer writes it). Everything else -- the tensor directory,
#      every operand's tiling and layout hash, every config value, every data
#      byte -- must be identical.
#
#      The plan's phrase "byte-for-byte" is therefore exact minus that field,
#      and the masking is NOT a loosening: the source record is checked
#      instead, against the digest of the file each pack actually read.
#
# WHY THIS IS A GATE AND NOT A TEST OF THE READER
#
# tools/verify/verify_onnx_reader.py already checks the ONNX reader against a
# checkpoint. This is the other axis: whether the second source is a drop-in
# replacement. It skips (and says so) for a model that has only one source, so
# a checkout with no downloads still runs.
#===----------------------------------------------------------------------===//

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "lib"))

from npue_tools.weights import (ONNX_NAME, SAFETENSORS_NAME,  # noqa: E402
                                open_model, weights_digest)


def is_graph_temporary(name: str) -> bool:
    """Whether an ONNX initializer is an exporter's temporary, not a weight.

    Three shapes, all seen in this tree: a subgraph-scoped constant
    (`/emb_ln/Constant_output_0`, `/Constant_2_output_0`), an orphaned node
    output the exporter numbered (`onnx::Unsqueeze_159`, `onnx::MatMul_977`),
    and a shape operand (`/Unsqueeze_axes`). A REAL weight is a checkpoint
    name -- dots, module path, `.weight`/`.bias` -- and never one of these.
    """
    # A SLASH IS NEVER IN A CHECKPOINT NAME: module paths are dotted
    # (`model.encoder.layers.0...`), and a slash only appears in an exporter's
    # subgraph scoping -- so all four shapes below are reachable by asking
    # "does this name have one".
    return (
        "/" in name
        or name.startswith("onnx::")
        or "Constant" in name
        or name.endswith("_axes")
    )


#: Where an export of a model can be, relative to the model directory. Whisper
#: is the reason this is a list: its export is TWO files (an encoder and a
#: decoder) and there is no onnx/model.onnx.
ONNX_LAYOUTS = (ONNX_NAME, "onnx/encoder_model.onnx")


def sources(model_dir: Path) -> tuple[bool, bool]:
    return (
        (model_dir / SAFETENSORS_NAME).is_file(),
        any((model_dir / p).is_file() for p in ONNX_LAYOUTS),
    )


def onnx_source(model_dir: Path):
    """The ONNX reader the PACKER would use for this model, not a second one.

    Importing the packer's own function is the point: a gate that opened the
    export its own way would be checking a reader the pipeline does not use.
    """
    if (model_dir / ONNX_NAME).is_file():
        return open_model(model_dir, prefer="onnx")
    sys.path.insert(0, str(REPO / "tools" / "pack"))
    from packers.whisper import combined_onnx  # noqa: PLC0415

    return combined_onnx(model_dir)


def compare_numbers(model_dir: Path) -> list[str]:
    """The two readers must serve one dict: same names, shapes, bytes."""
    problems = []
    st = onnx = None
    try:
        st = open_model(model_dir, prefer="safetensors")
        onnx = onnx_source(model_dir)
        a, b = sorted(st.keys()), sorted(onnx.keys())
        # WHAT A KEY-SET DIFFERENCE MEANS, IN BOTH DIRECTIONS.
        #
        # The ONNX reader surfaces every INITIALIZER, and a traced export puts
        # its graph constants there under synthetic names -- nomic's ONNX
        # carries 24 of them (`/Constant_2_output_0`, `/Unsqueeze_axes`) beside
        # its 112 real weights. A constant is not a weight: the packer never
        # reads one, and its absence from a flat checkpoint is not a loss. So
        # those are counted and reported, not refused.
        #
        # The other direction is a real loss: a weight the ONNX source has that
        # the flat checkpoint does not. That is a KeyError inside an arch
        # branch, or a zero-filled tensor where a branch defaults, so it fails.
        #
        # The flat checkpoint may also hold MORE: MiniLM's has
        # `embeddings.position_ids` (an index buffer) and an unused
        # `pooler.dense` pair. Those are reported too.
        missing = [n for n in b if n not in set(a)]
        constants = [n for n in missing if is_graph_temporary(n)]
        lost = [n for n in missing if n not in set(constants)]
        if lost:
            problems.append(
                f"{len(lost)} weight(s) the ONNX source has are not in the "
                f"safetensors checkpoint: {lost[:6]}"
            )
            return problems
        extra = [n for n in a if n not in set(b)]
        print(f"# key sets    {len(a)} safetensors, {len(b)} onnx "
              f"({len(constants)} of them graph constants; "
              f"{len(extra)} only in safetensors {extra[:4]})")
        # Only the ONNX keys that ARE weights: a constant in the digest would
        # make the comparison depend on the exporter's temporaries.
        shared = set(b) - set(constants)
        sa, sb = weights_digest(st, only=shared), weights_digest(onnx, only=shared)
        if sa != sb:
            # Name the offenders: a digest alone says "differ" and leaves the
            # next hour to guesswork.
            bad = []
            for name in sorted(shared):
                x, y = st.array(name), onnx.array(name)
                if x.dtype != y.dtype or x.shape != y.shape or \
                        x.tobytes() != y.tobytes():
                    bad.append((name, x.shape, x.dtype, y.shape, y.dtype))
                if len(bad) >= 4:
                    break
            problems.append(
                f"the weights differ ({sa[:16]} against {sb[:16]}); "
                + "; ".join(
                    f"{n}: {sx}{dx} vs {sy}{dy}" for n, sx, dx, sy, dy in bad
                )
            )
        print(f"# numbers     {len(shared)} weights, digest {sa[:16]}...")
    finally:
        for w in (st, onnx):
            if w is not None:
                w.close()
    return problems


def pack(model_dir: Path, out: Path, source: str, extra: list[str]) -> str:
    cmd = [sys.executable, str(REPO / "tools" / "pack" / "pack_npue.py"),
           "--model-dir", str(model_dir), "--out", str(out),
           "--weights-source", source, *extra]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        return (proc.stderr or proc.stdout).strip()[-3000:]
    return ""


def masked(directory: dict) -> dict:
    """The container's directory with the source record taken out.

    Three keys, all about where the numbers came from and all checked
    separately below: the checkpoint that was read (`source_sha256` --
    necessarily different, it is a different file), which format that was
    (`weights_source`), and the export the safetensors pack is proven equal to
    (`export_sha256`, which the ONNX pack does not carry).
    """
    d = json.loads(json.dumps(directory))
    for key in ("source_sha256", "weights_source", "export_sha256"):
        d.get("config", {}).pop(key, None)
    return d


def container_bytes(path: Path) -> tuple[dict, bytes]:
    """(directory, tensor data bytes) of a .npue, without the header.

    Read through tools/lib/npue.Reader so this gate cannot disagree with the
    runtime about where the directory or the data begins.
    """
    from npue import Reader

    with open(path, "rb") as f:
        blob = f.read()
    r = Reader(path)
    try:
        directory = {
            "config": r.config,
            "tensors": [r.entries[n] for n in sorted(r.entries)],
        }
        data = blob[r.data_offset : r.data_offset + r.data_length]
    finally:
        r.close()
    return directory, data


def compare_containers(model_dir: Path, work: Path, extra: list[str]) -> list[str]:
    problems = []
    out = {}
    for source in ("onnx", "safetensors"):
        path = work / f"{model_dir.name}-{source}.npue"
        error = pack(model_dir, path, source, extra)
        if error:
            return [f"packing from {source} failed:\n{error}"]
        out[source] = path

    da, da_data = container_bytes(out["onnx"])
    db, db_data = container_bytes(out["safetensors"])
    if da_data != db_data:
        # Which tensor: the directory carries offsets, so a byte offset can be
        # turned into a name instead of a number.
        first = next(
            (i for i, (x, y) in enumerate(zip(da_data, db_data)) if x != y),
            min(len(da_data), len(db_data)),
        )
        name = spark = None
        for entry in da["tensors"]:
            start = entry.get("offset", 0)
            if start <= first < start + entry.get("nbytes", 0):
                name = entry["name"]
                spark = entry
                break
        problems.append(
            f"tensor data differs at byte {first}"
            + (f" (tensor {name!r}, {spark})" if name else "")
        )
    if masked(da) != masked(db):
        # Report the KEYS that differ rather than a wall of JSON.
        diff = []
        ca, cb = masked(da), masked(db)
        for key in sorted(set(ca["config"]) | set(cb["config"])):
            if ca["config"].get(key) != cb["config"].get(key):
                diff.append(f"config.{key}: {ca['config'].get(key)!r} != "
                            f"{cb['config'].get(key)!r}")
        if ca["tensors"] != cb["tensors"]:
            for x, y in zip(ca["tensors"], cb["tensors"]):
                if x != y:
                    diff.append(f"tensor entry {x['name']!r} differs: "
                                f"{x} != {y}")
                    break
        problems.append("container directories differ: " + "; ".join(diff[:6]))

    # And the field that was masked is itself checked: it must be the digest
    # of the file that pack read, and the two sources must not agree on it
    # (they are different files, and a container that claimed otherwise would
    # be provenance that cannot be trusted).
    sha_onnx = da["config"].get("source_sha256")
    sha_st = db["config"].get("source_sha256")
    # The export the flat checkpoint claims to equal must be the export that
    # was packed here -- not merely some export.
    claimed = db["config"].get("export_sha256")
    if claimed is not None and claimed != sha_onnx:
        problems.append(
            f"the safetensors container claims export_sha256 {claimed[:16]} "
            f"but the ONNX pack was made from {str(sha_onnx)[:16]}"
        )
    if not sha_onnx or not sha_st:
        problems.append("a packed container carries no source_sha256")
    elif sha_onnx == sha_st:
        problems.append(
            f"both containers record source_sha256 {sha_onnx[:16]} -- the "
            f"digest is supposed to identify the checkpoint files"
        )
    print(f"# containers  identical except the source record "
          f"({sha_onnx[:12]} vs {sha_st[:12]})")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", action="append", default=None,
                    help="a model directory; default is every one under "
                         "models/ that has both sources")
    ap.add_argument("--work", default=None, help="where to write the containers")
    ap.add_argument("--keep", action="store_true",
                    help="keep the containers (they are deleted otherwise)")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="npu1")
    args = ap.parse_args()

    dirs = ([Path(p) for p in args.model_dir] if args.model_dir
            else sorted(p for p in (REPO / "models").iterdir() if p.is_dir()))
    checked = skipped = failed = 0
    extra = ["--dtype", args.dtype, "--device", args.device]
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp())
    work.mkdir(parents=True, exist_ok=True)

    for model_dir in dirs:
        has_st, has_onnx = sources(model_dir)
        if not (has_st and has_onnx):
            if has_st or has_onnx:
                print(f"# skip        {model_dir.name}: only "
                      f"{'safetensors' if has_st else 'onnx'}")
                skipped += 1
            continue
        print(f"== {model_dir.name}")
        problems = compare_numbers(model_dir) + compare_containers(
            model_dir, work, extra)
        if problems:
            failed += 1
            for p in problems:
                print(f"  FAIL  {p}")
        else:
            checked += 1
            print("  PASS  same numbers, same container")

    if not args.keep:
        for p in work.glob("*.npue"):
            p.unlink()
    print(f"\nchecked {checked}, failed {failed}, skipped {skipped}")
    return 0 if failed == 0 and checked else (1 if failed else 0)


if __name__ == "__main__":
    sys.exit(main())
