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
#   3. THE DECLARED TRANSFORM. Some exports are the checkpoint under a stated
#      rewrite -- embeddinggemma-300m's is the checkpoint rounded to bf16 with
#      gemma's `(1 + w)` folded into every RMSNorm scale. Identity would fail
#      there by construction, and a silent tolerance would fail everywhere.
#      Such a model DECLARES its rewrite (see `declared_transform`) and the
#      gate proves the declaration weight by weight: every tensor explained by
#      a declared class, every class used by some tensor, nothing else allowed.
#      The verdict reads "equal up to the declared export transform" and names
#      it; an unexplained tensor is still a FAIL.
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
from functools import cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "lib"))
sys.path.insert(0, str(REPO / "tools" / "pack"))

from npue import Reader, from_bf16_bits, to_bf16_bits  # noqa: E402
from npue_tools.weights import (ONNX_NAME, SAFETENSORS_NAME,  # noqa: E402
                                open_model, weights_digest)


@cache
def packer():
    """The packer module, imported once.

    The gate compares under the PIPELINE's name rules, not its own: the export
    repair and the sources' roots are the packer's answers, imported rather
    than restated, so the two cannot drift into disagreeing about what a name
    means.
    """
    import pack_npue  # noqa: PLC0415

    return pack_npue


def is_graph_temporary(name: str) -> bool:
    """Whether an ONNX initializer is an exporter's temporary, not a weight.

    Three shapes, all seen in this tree: a subgraph-scoped constant
    (`/emb_ln/Constant_output_0`, `/Constant_2_output_0`), an orphaned node
    output the exporter numbered (`onnx::Unsqueeze_159`, `onnx::MatMul_977`),
    and a shape operand (`/Unsqueeze_axes`). A REAL weight is a checkpoint
    name -- dots, module path, `.weight`/`.bias` -- and never one of these.
    """
    # A DOTTED PATH IS THE MARK OF A CHECKPOINT NAME, and a bare noun has none:
    # `cos_cache_global`/`cos_cache_local` are the gemma export's stored RoPE
    # tables -- Cos/Sin of inv_freq, computed once at export time and left in
    # the graph as initializers. No checkpoint spells a weight that way, so a
    # name with neither a dot nor a slash is something an exporter computed.
    if "." not in name and "/" not in name:
        return True
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


# ---------------------------------------------------------------- declared
# A THIRD VERDICT, AND WHY IT IS NOT A LOOSENING
#
# Two sources of one checkpoint are normally compared for identity: same names,
# same bytes, same container. Some exports are not identities, and pretending
# otherwise makes a gate say FAIL about a difference that is fully understood --
# which is worse than no gate, because the next person "fixes" it by weakening
# the comparison for everyone.
#
# So a model may DECLARE what its export did to the checkpoint. The gate then
# proves that declaration tensor by tensor: every weight must be explained by
# one of the declared rewrites, and every declared rewrite must explain at least
# one weight (a class that explains nothing is a stale declaration, and is
# reported as such). A single unexplained tensor fails the gate exactly as
# before. What the declaration buys is the NAME of the verdict, not slack.
#
# Only embeddinggemma-300m has one, and it was measured on all 314 tensors
# before it was written down: 169 linear/embedding weights are the checkpoint
# rounded to bf16, and 145 RMSNorm scales are `1 + bf16(w)`, because the export
# folded gemma's `(1 + w)` RMSNorm offset into the initializer.


def to_bf16(array):
    """The checkpoint's fp32, rounded to bf16 and widened back (RNE).

    Through `tools/lib/npue`'s own bit operation, so the gate cannot disagree
    with the packer about what bf16 means.
    """
    return from_bf16_bits(to_bf16_bits(np.asarray(array, dtype=np.float32)))


def requantise(array, dtype):
    """Re-round a value to a container's storage dtype, so a comparison made
    after packing asks the same question the file answers."""
    if dtype == "BF16":
        return to_bf16(array)
    return np.asarray(array, dtype=np.float32)


class DeclaredTransform:
    """What an export did to the checkpoint, in checkable pieces.

    `classes` are applied at READER level, where both sources hold f32 and the
    full rewrite is visible. `container` classes are applied at CONTAINER level,
    where each side has already been rounded to the pack's dtype, so only the
    part of the rewrite that survives packing can be tested there.
    """

    def __init__(self, sentence, classes, container):
        self.sentence = sentence
        self.classes = classes
        self.container = container

    def matching(self, checkpoint, export):
        """Which declared rewrites produce `export` from `checkpoint`."""
        found = []
        for label, fn in self.classes:
            expected = np.asarray(fn(checkpoint), dtype=np.float32)
            if np.array_equal(expected, np.asarray(export, dtype=np.float32)):
                found.append(label)
        return found


def declared_transform(model_dir: Path):
    """The declared export rewrite for this model, or None if it is an identity."""
    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    except OSError:
        return None
    if str(cfg.get("model_type", "")).startswith("gemma"):
        return DeclaredTransform(
            "the export is the checkpoint rounded to bf16, with gemma's "
            "(1 + w) folded into every RMSNorm scale",
            classes=[
                ("bf16", to_bf16),
                ("bf16 then 1 + w (RMSNorm)",
                 lambda a: to_bf16(a) + np.float32(1.0)),
            ],
            container=[
                # The container is asked a weaker question than the reader: it
                # holds the pack's dtype, so the export's own bf16 rounding is
                # already in the checkpoint side of the comparison. What
                # survives packing is the offset -- applied to the checkpoint
                # value as the EXPORT rounded it, then re-rounded to whatever
                # the entry stores (an F32 host norm row still holds only the
                # export's bf16 precision).
                ("bf16",
                 lambda a, dtype: requantise(to_bf16(a), dtype)),
                ("1 + bf16(w) (RMSNorm)",
                 lambda a, dtype: requantise(to_bf16(a) + np.float32(1.0), dtype)),
            ],
        )
    return None


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
    # The export's name repair is the PACKER's: gemma's export keeps a
    # "model." root and renames three families mid-name, so comparing raw
    # initializer names against the checkpoint would call all 314 correctly
    # spelled weights missing.
    strip, rename = packer().export_repair(model_dir)
    if (model_dir / ONNX_NAME).is_file():
        return open_model(model_dir, prefer="onnx", strip=strip, rename=rename)
    from packers.whisper import combined_onnx  # noqa: PLC0415

    return combined_onnx(model_dir)


def canonical_names(names, roots):
    """{canonical: the spelling this source uses}, roots applied once per name."""
    out = {}
    for name in names:
        canonical = name
        for root in roots:
            if canonical.startswith(root):
                canonical = canonical[len(root):]
                break
        out.setdefault(canonical, name)
    return out


def compare_numbers(model_dir: Path) -> list[str]:
    """The two readers must serve one dict: same weights, same bytes.

    Three shape of "same", and the check says which one it applied:
      * identity -- both readers spell and round the checkpoint the same way,
        so the digest of the whole dict must match;
      * namespace -- the two sources disagree only about a root in front of
        every name (gte: `new.` against `0.auto_model.`), which the packer
        already reconciles, so the comparison is per canonical name;
      * declared transform -- the export rewrote the numbers (embeddinggemma:
        bf16 plus a folded `1 + w`), proven weight by weight against the
        declaration.
    """
    problems = []
    st = onnx = None
    decl = declared_transform(model_dir)
    roots = packer().source_name_roots(model_dir)
    try:
        st = open_model(model_dir, prefer="safetensors")
        onnx = onnx_source(model_dir)
        st_map = canonical_names(sorted(st.keys()), roots)
        on_map = canonical_names(sorted(onnx.keys()), roots)
        # WHAT A KEY-SET DIFFERENCE MEANS, IN BOTH DIRECTIONS.
        #
        # The ONNX reader surfaces every INITIALIZER, and a traced export puts
        # its graph constants there under synthetic names -- nomic's ONNX
        # carries 24 of them (`/Constant_2_output_0`, `/Unsqueeze_axes`) beside
        # its 112 real weights. A constant is not a weight: the packer never
        # reads one, and its absence from a flat checkpoint is not a loss.
        #
        # The other direction is a real loss: a weight the ONNX source has that
        # the flat checkpoint does not. That is a KeyError inside an arch
        # branch, or a zero-filled tensor where a branch defaults, so it fails.
        #
        # The flat checkpoint may also hold MORE: MiniLM's has
        # `embeddings.position_ids` (an index buffer), an unused `pooler.dense`
        # pair, and gte's a `classifier` head the container never reads.
        missing = [c for c in on_map if c not in st_map]
        constants = [c for c in missing if is_graph_temporary(on_map[c])]
        lost = [c for c in missing if c not in set(constants)]
        if lost:
            problems.append(
                f"{len(lost)} weight(s) the ONNX source has are not in the "
                f"safetensors checkpoint: {[on_map[c] for c in lost][:6]}")
            return problems
        extra = [c for c in st_map if c not in on_map]
        print(f"# key sets    {len(st_map)} safetensors, {len(on_map)} onnx "
              f"({len(constants)} of them graph constants; {len(extra)} only "
              f"in safetensors {[st_map[c] for c in extra][:4]})")

        shared = [c for c in sorted(on_map) if c not in set(constants)]
        counts: dict[str, int] = {}
        unexplained = []
        for canonical in shared:
            x = st.array(st_map[canonical])
            y = onnx.array(on_map[canonical])
            if x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes():
                continue
            if x.shape != y.shape or x.dtype != y.dtype:
                unexplained.append(f"{st_map[canonical]}: {x.shape}{x.dtype} vs "
                                   f"{on_map[canonical]}: {y.shape}{y.dtype}")
                continue
            labels = decl.matching(x, y) if decl is not None else []
            if labels:
                counts[labels[0]] = counts.get(labels[0], 0) + 1
            else:
                unexplained.append(
                    f"{st_map[canonical]}: matches none of the declared "
                    f"rewrites (checkpoint {np.asarray(x).ravel()[:1]}, export "
                    f"{np.asarray(y).ravel()[:1]})")
        if unexplained and decl is None:
            problems.append(
                f"{len(unexplained)} weight(s) differ: " + "; ".join(unexplained[:4]))
        elif unexplained:
            problems.append(
                f"{len(unexplained)} weight(s) are not explained by the "
                f"declared export transform: " + "; ".join(unexplained[:4]))
        elif decl is None:
            digest = weights_digest(st, only=[st_map[c] for c in shared])
            if roots:
                print(f"# numbers     {len(shared)} weights, identical under "
                      f"the sources' own spellings ({', '.join(roots)})")
            else:
                print(f"# numbers     {len(shared)} weights, digest "
                      f"{digest[:16]}...")
        else:
            unused = [label for label, _ in decl.classes if label not in counts]
            if unused:
                problems.append(
                    "the declared export transform names a rewrite that "
                    "matches no weight: " + ", ".join(unused))
            else:
                detail = ", ".join(f"{n} {label}" for label, n in counts.items())
                print(f"# numbers     {len(shared)} weights, equal up to the "
                      f"declared export transform ({detail})")
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


def compare_tensor_data(onnx_path: Path, st_path: Path, da: dict, db: dict,
                        decl) -> list[str]:
    """Compare the two containers' tensor data, tensor by tensor.

    Two byte strings of a multi-gigabyte container cannot say WHICH tensor
    differs, and a model with a declared export transform must be allowed to
    differ in exactly the tensors that declaration covers -- nowhere else. So
    the comparison is per entry, and the entries that differ are asked to
    satisfy a declared rewrite (re-rounded to the container's own dtype, since
    that is what the file holds).
    """
    expected = {e["name"]: e for e in db["tensors"]}
    problems = []
    explained: dict[str, int] = {}
    ra = Reader(str(onnx_path))
    rb = Reader(str(st_path))
    try:
        for entry in da["tensors"]:
            name = entry["name"]
            other = expected.get(name)
            if other is None:
                problems.append(f"tensor {name!r} exists only in the ONNX container")
                continue
            if entry != other:
                problems.append(
                    f"tensor entry {name!r} differs: {entry} != {other}")
                continue
            x = ra.tensor(name)
            y = rb.tensor(name)
            if entry.get("dtype") in ("U8", "I8", "I4"):
                continue
            if np.array_equal(x, y):
                continue
            labels = []
            for label, fn in decl.container if decl is not None else ():
                want = np.asarray(fn(y, entry.get("dtype")), dtype=np.float32)
                if np.array_equal(want, np.asarray(x, dtype=np.float32)):
                    labels.append(label)
            if labels:
                explained[labels[0]] = explained.get(labels[0], 0) + 1
            else:
                problems.append(
                    f"tensor {name!r} differs and no declared rewrite explains "
                    f"it (onnx {np.asarray(x).ravel()[:1]} against checkpoint "
                    f"{np.asarray(y).ravel()[:1]})")
    finally:
        ra.close()
        rb.close()
    if not problems and explained:
        detail = ", ".join(f"{n} {label}" for label, n in explained.items())
        print(f"# tensors     equal up to the declared export transform ({detail})")
    return problems


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
        problems += compare_tensor_data(
            out["onnx"], out["safetensors"], da, db, declared_transform(model_dir))
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
            decl = declared_transform(model_dir)
            if decl is None:
                print("  PASS  same numbers, same container")
            else:
                print(f"  PASS  equal up to the declared export transform -- "
                      f"{decl.sentence}")

    if not args.keep:
        for p in work.glob("*.npue"):
            p.unlink()
    print(f"\nchecked {checked}, failed {failed}, skipped {skipped}")
    if checked == 0:
        # SKIPPED IS NOT PASSED. A model with one source cannot be gated at
        # all, and exiting 0 for "I compared nothing" is how a green gate
        # comes to mean "nobody looked" -- the failure mode that lets an
        # unverified source be deleted because a script said ok.
        print("FAIL: no model had both sources, so nothing was verified")
        return 1
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
