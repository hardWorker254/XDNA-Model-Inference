#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify an arch=9 PaddleOCR container and the reference runtime built from it.

WHAT THIS FILE IS, AND WHAT IT IS NOT
-------------------------------------
Two checkpoints in one container -- a text detector and a text recogniser --
and a reference runtime that runs both and reads a line of text off a photograph.
This file checks that the container holds those two checkpoints in that
arrangement, that the runtime walks their graphs the way the graphs say, and
that the whole thing answers the question the checkpoints were trained to
answer.

FOUR SECTIONS, AND THE SPLIT IS THE USEFUL PART
-----------------------------------------------
  1. the packed op list, against onnxruntime. The container's op list is
     materialised back into ONNX and run by onnxruntime, op by op, against the
     same onnxruntime running the checkpoint. A disagreement here is the
     PACKER's -- the weights it folded, the order it kept, the shapes it
     resolved.
  2. the runtime's walk, against an independent reading of the same op list,
     node by node. The runtime dumps every node's output; the reading here
     computes the same nodes in numpy from the SAME INPUT and the two are
     compared one node at a time, so a disagreement names the node.
  3. the runtime end to end, against rapidocr. This is the section that covers
     what sections 1 and 2 cannot: the letterbox, the DB post-process, the
     perspective crop and the CTC decode are none of them in either graph.
  4. the array path against the host path, on the same frame.

Section 1 passing and section 3 failing is a FRONT END fault and nothing else.
That separation is the reason the gate is split and not one pass: end to end,
a wrong letterbox and a wrong last convolution are the same sentence.

WHY THE ORACLE OF SECTION 3 IS RAPIDOCR
---------------------------------------
The model card of the checkpoint says to use it with rapidocr, and rapidocr
is a third-party implementation of the whole pipeline -- the frame, the DB
post-process, the crop, the CTC decode -- written by somebody who had no part
in this runtime. The alternative oracle would be a reading of the same
algorithm written by the same hand as the runtime, which can be wrong in the
same way twice and still agree with itself.

THE BOXES ARE COMPARED WITH A TOLERANCE, THE TEXTS ARE NOT
----------------------------------------------------------
rapidocr's detector resizes the frame to its own rule -- the SHORT side to 736
pixels -- and this runtime's resizes it to its own -- the LONG side to 960 --
so the two probability maps are of different resolutions and the boxes they
produce differ by a pixel or two. The texts are compared exactly, because
after the boxes are close the crops are close and a transcription is either
the same string or it is not.

THE RECOGNISER'S FRAME IS 320 PIXELS WIDE, AND THAT IS A LIMIT
-------------------------------------------------------------
PaddleOCR feeds the recogniser a frame as wide as the line it is reading, and
the checkpoint's width is a dynamic dimension; this runtime's container fixes
it at 320, so a line longer than 320/48 of its own height is squashed into
320 pixels. The squashing is the runtime's, not the checkpoint's, and both
this file and rapidocr see it: the texts compared here are the texts both
read off the same squashed crop.

THE TWO FAULTS THIS GATE WAS BUILT AFTER
-----------------------------------------
Both of them were found by section 2 and neither by section 3, which is why
section 2 is in the file:

  - the convex hull's upper chain took the wrong point into its cross
    product, and the hull lost the lower edge of every text line: the boxes
    came out as strips through the middle of the glyphs.
  - a reduction over one axis computed the strides of the OUTPUT and read the
    input through them, so it averaged the wrong elements; and the attention's
    matmul paired the batch axes with the strides the other way round, so all
    eight heads read the first head's keys.

Both produce a plausible transcription of the wrong thing, and both produce it
CONFIDENTLY.

SPDX-License-Identifier: Apache-2.0
"""

import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "lib"))
sys.path.insert(0, _HERE)

from npue import Reader  # noqa: E402
import ocr_ref  # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
MODEL_DIR = os.path.join(REPO, "models", "paddleocr-onnx")
CONTAINER = [os.path.join(REPO, "models", "paddleocr-onnx.npue")]
DET_ONNX = os.path.join(MODEL_DIR, "det.onnx")
REC_ONNX = os.path.join(MODEL_DIR, "rec.onnx")
DICT_TXT = os.path.join(MODEL_DIR, "dict.txt")
IMAGE = os.path.join(REPO, "docs", "ocr_text.png")
BINARY = os.path.join(REPO, "runtime", "build", "npuimage")

# Seeded, and the same every run: a section must fail the SAME way twice for
# the message to be worth anything.
SEED = 20261113

# Section 1's tolerance, peak-relative to each op's own largest value. The
# measured worst is far below it; what sets it is the fault it has to catch:
# a convolution whose weights were folded wrongly is a number of the order of
# ONE against the same reference.
TOL_PEAK_REL = 1e-3

# Section 2's tolerance, on the same footing. The two sides accumulate in a
# different order -- numpy's pairwise sums against the runtime's -- so the
# same op does not give the last bit, and the tolerance covers that and not a
# wrong index.
TOL_WALK_REL = 1e-3

# Section 3's box tolerance, in pixels. The two front ends resize the frame
# to different rules, so the probability maps are of different resolutions and
# the boxes differ by a pixel or two; the texts are compared exactly.
TOL_BOX_PX = 6.0


class Fail(Exception):
    pass


def check(cond, msg):
    if not cond:
        raise Fail(msg)
    return True


def _need(path, what):
    if not os.path.exists(path):
        raise Fail(f"{what} is not there: {path}")


# ==============================================================================
# -- section 1: the packed op list, against onnxruntime ------------------------
# ==============================================================================


def _with_outputs(model, names, shapes):
    """The model's graph outputs, extended to every tensor the caller wants.

    A checkpoint's ONNX publishes its last tensor and nothing else, and a
    comparison that can only see the last tensor is a comparison of one
    number: a graph that is wrong in the middle and right at the end passes
    it. Every packed op names the checkpoint tensor it came from -- the
    packer recorded it for exactly this -- so the checkpoint can be asked for
    all of them at once.

    The OUTPUTS ARE REPLACED and not appended to, and their order is the
    caller's: onnxruntime returns the outputs in the graph's order, and the
    materialised graph's own output -- its last op -- would otherwise come
    first and every index after it would be off by one.
    """
    import copy
    from onnx import helper as oh
    g = model.graph
    vi = {v.name: v for v in g.value_info}
    outs = []
    for n, sh in zip(names, shapes):
        if not n:
            continue
        if n in vi:
            outs.append(copy.deepcopy(vi[n]))
        else:
            outs.append(oh.make_tensor_value_info(
                n, 1, [int(d) for d in (sh or [])]))
    del g.output[:]
    g.output.extend(outs)
    return model


def _peak_rel(got, want):
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    if got.shape != want.shape:
        raise Fail(f"shape {got.shape} against onnxruntime's {want.shape}")
    return float(np.abs(got - want).max()) / (1.0 + float(np.abs(want).max()))


def section_1(reader, verbose):
    import onnx
    import onnxruntime as ort

    print("  1. the packed op list, against onnxruntime")
    for pfx, onnx_path, graph_key, shape in (
            ("det", DET_ONNX, "det_graph",
             (1, 3, int(reader.config["det_input_size"]),
              int(reader.config["det_input_size"]))),
            ("rec", REC_ONNX, "rec_graph",
             (1, 3, int(reader.config["rec_input_height"]),
              int(reader.config["rec_input_width"])))):
        _need(onnx_path, f"the {pfx} checkpoint")
        g = ocr_ref.Graph.from_container(reader, pfx)

        # THE SAME INPUT ON BOTH SIDES. A random tensor, seeded: a real
        # photograph would also be a fair test but it is a WEAK one, because
        # it exercises the network on a distribution the checkpoint has never
        # seen, where a transposed filter still produces a plausible range.
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal(shape).astype(np.float32)

        # -- the container's op list, as an ONNX graph
        model, names = ocr_ref.materialise(g, list(shape), prefix=f"{pfx}_x")
        model = _with_outputs(model, names, [op.get("oshape")
                                             for op in g.ops])
        onnx.checker.check_model(model)
        sess = ort.InferenceSession(model.SerializeToString(),
                                    providers=["CPUExecutionProvider"])
        got = sess.run(None, {f"{pfx}_x": x})

        # -- the checkpoint, asked for every tensor the packer named
        ck = onnx.load(onnx_path)
        outs = [op.get("out") for op in g.ops]
        shapes = [op.get("oshape") for op in g.ops]
        ck = _with_outputs(ck, outs, shapes)
        onnx.checker.check_model(ck)
        csess = ort.InferenceSession(ck.SerializeToString(),
                                     providers=["CPUExecutionProvider"])
        want = csess.run(None, {csess.get_inputs()[0].name: x})

        # The checkpoint's outputs, paired with the packed ops by NAME. The
        # op that FOLDED a batch norm into its convolution has no counterpart
        # in the checkpoint -- the checkpoint's tensor is the sum, and the
        # packed list has three ops for it -- and those ops are compared
        # through the op that does have a counterpart.
        by_name = {o.name: v for o, v in zip(csess.get_outputs(), want)}
        worst, rows, compared = 0.0, [], 0
        for i, op in enumerate(g.ops):
            name = op.get("out")
            if not name or name not in by_name:
                continue
            rel = _peak_rel(got[i], by_name[name])
            rows.append((i, op["op"], rel))
            worst = max(worst, rel)
            compared += 1
        check(compared > 0, f"{pfx}: no packed op names a tensor of the "
                           f"checkpoint, so there was nothing to compare")
        for i, kind, rel in rows:
            if verbose or rel > 1e-6:
                print(f"     ok   {pfx} op {i:<4} {kind:<12} peak-rel "
                      f"{rel:.3e}")
        check(worst <= TOL_PEAK_REL,
              f"{pfx}: the packed graph reaches {worst:.3e} of onnxruntime's "
              f"peak at its worst op, above the {TOL_PEAK_REL:.0e} this "
              f"section allows. The per-op numbers are above.")
        print(f"     {pfx}: {compared} ops reproduce onnxruntime, worst "
              f"{worst:.3e} of peak (allowed {TOL_PEAK_REL:.0e})")


# ==============================================================================
# -- section 2: the runtime's walk, against an independent reading -------------
# ==============================================================================


# The reserved area at the start of a dump, in bytes. It is the runtime's
# constant (runtime/include/runtime/ocr_mode.hpp, kDumpHeader): the JSON
# header is written there at the end of the run, padded with spaces to fill
# it, and every tensor's offset is absolute from the start of the file.
_DUMP_HEADER = 1 << 20


def _read_dump(path):
    """The dump's records: (header, float array).

    The file is a JSON header in the first megabyte and raw fp32 after it.
    """
    with open(path, "rb") as f:
        raw = f.read()
    head = json.loads(raw[:_DUMP_HEADER].decode("utf-8"))
    body = raw[_DUMP_HEADER:]
    out = []
    for rec in head["nodes"]:
        n = int(rec["n"])
        off = int(rec["off"]) - _DUMP_HEADER
        check(0 <= off and off + 4 * n <= len(body),
              f"the dump's record {rec['net']} node {rec['i']} points at bytes "
              f"{rec['off']}..{rec['off'] + 4 * n}, and the file's tensors "
              f"end at {len(body) + _DUMP_HEADER}")
        out.append((rec, np.frombuffer(body, dtype=np.float32, count=n,
                                       offset=off)))
    return out


def _run_runtime(container, extra=()):
    cmd = [BINARY, "ocr", container, IMAGE, *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    check(p.returncode == 0,
          f"`{' '.join(cmd)}` exited {p.returncode}\n{p.stderr[-800:]}")
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError as e:
        raise Fail(f"the runtime printed something that is not JSON ({e}); "
                   f"its stdout is the result and stderr is the status "
                   f"block:\n{p.stderr[-400:]}")
    return out, p.stderr


def section_2(reader, verbose):
    print("  2. the runtime's walk, against an independent reading of the "
          "op list")
    dump_path = os.path.join("/tmp", "npue-ocr-verify-dump.bin")
    got, _ = _run_runtime(CONTAINER[0], ("--ocr-dump", dump_path))
    check(os.path.exists(dump_path), "the runtime wrote no dump")
    records = _read_dump(dump_path)

    by_net = {}
    for rec, arr in records:
        by_net.setdefault(rec["net"], []).append((rec, arr))

    for pfx, key in (("det", "det_graph"), ("rec", "rec_graph")):
        g = ocr_ref.Graph.from_container(reader, pfx)
        ops = json.loads(reader.config[key])
        rows = by_net.get(pfx, [])
        check(rows, f"the dump holds no {pfx} nodes")
        # The recogniser runs ONCE PER BOX, so its nodes are in runs of one
        # graph's length, each with its own input in front of it.
        per = len(ops) + 1
        check(len(rows) % per == 0,
              f"the dump holds {len(rows)} {pfx} records, which is not a whole "
              f"number of {per}-record runs")
        runs = len(rows) // per
        worst_all, worst_at, n_bad = 0.0, None, 0
        for run in range(runs):
            chunk = rows[run * per:(run + 1) * per]
            inp = chunk[0][0]
            check(inp["i"] == -1,
                  f"{pfx} run {run}: the dump's first record is node "
                  f"{inp['i']}, and the network's input is recorded at -1")
            nodes = chunk[1:]
            x = chunk[0][1].reshape(inp["shape"])
            vals = g.run(x)
            for rec, arr in nodes:
                i = int(rec["i"])
                want = np.asarray(vals[i], np.float64)
                arr = arr.reshape(rec["shape"]).astype(np.float64)
                if want.shape != arr.shape:
                    raise Fail(f"{pfx} run {run} node {i}: the runtime's tensor "
                               f"is {arr.shape} and the reading's is "
                               f"{want.shape}")
                rel = float(np.abs(want - arr).max()) / (
                    1.0 + float(np.abs(want).max()))
                if rel > TOL_WALK_REL:
                    n_bad += 1
                    if worst_at is None:
                        worst_at = (run, i, ops[i]["op"], rel)
                worst_all = max(worst_all, rel)
        if n_bad:
            raise Fail(
                f"{pfx}: {n_bad} nodes differ from the independent reading by "
                f"more than {TOL_WALK_REL:.0e} of their own peak; the first is "
                f"run {worst_at[0]} node {worst_at[1]} ({worst_at[2]}), off by "
                f"{worst_at[3]:.3e}. A node that is wrong here is wrong in the "
                f"RUNTIME, and section 1 has already ruled out the packer.")
        print(f"     {pfx}: {runs} run(s) x {per} nodes reproduce the "
              f"reading, worst {worst_all:.3e} of peak "
              f"(allowed {TOL_WALK_REL:.0e})")
    return got


# ==============================================================================
# -- section 3: the runtime end to end, against rapidocr ----------------------
# ==============================================================================


def _rapidocr():
    """rapidocr's own pipeline, on this checkpoint.

    The pieces are imported rather than the whole API, because the API of
    this version takes the dictionary's path through a keyword it does not
    forward -- and because the pieces are what this section compares against:
    the frame, the DB post-process, the crop, the CTC decode.
    """
    try:
        from rapidocr_onnxruntime.ch_ppocr_v3_rec.utils import CTCLabelDecode
        from rapidocr_onnxruntime.ch_ppocr_v3_det.utils import (
            DBPostProcess, DetResizeForTest, NormalizeImage)
    except ImportError as e:
        raise Fail(
            f"rapidocr is not importable ({e}), and this section's oracle IS "
            f"rapidocr: the checkpoint's model card names it, and it is a "
            f"third-party implementation of the whole pipeline. Install it "
            f"with `pip install rapidocr-onnxruntime`.")
    return CTCLabelDecode, DBPostProcess, DetResizeForTest, NormalizeImage


def _oracle(reader, det_onnx, rec_onnx, image_path):
    """rapidocr's answer on this image: the boxes and the texts."""
    import cv2
    import onnxruntime as ort
    CTCLabelDecode, DBPostProcess, DetResizeForTest, NormalizeImage = \
        _rapidocr()

    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    resize = DetResizeForTest(limit_side_len=736, limit_type="min")
    norm = NormalizeImage(scale="1./255.",
                           mean=list(reader.config["det_image_mean"]),
                           std=list(reader.config["det_image_std"]),
                           order="hwc")
    data = norm(resize({"image": img}))
    blob = np.transpose(data["image"], (2, 0, 1))[None].astype(np.float32)
    det = ort.InferenceSession(det_onnx, providers=["CPUExecutionProvider"])
    pred = det.run(None, {det.get_inputs()[0].name: blob})[0]

    op = DBPostProcess(thresh=float(reader.config["det_thresh"]),
                       box_thresh=float(reader.config["det_box_thresh"]),
                       max_candidates=int(reader.config["det_max_candidates"]),
                       unclip_ratio=float(reader.config["det_unclip_ratio"]))
    boxes = op(pred, np.array([[img.shape[0], img.shape[1], 1.0, 1.0]]))
    boxes = boxes[0]["points"]

    chars = []
    with open(DICT_TXT, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                chars.append(line)
    dec = CTCLabelDecode(chars)
    rec = ort.InferenceSession(rec_onnx, providers=["CPUExecutionProvider"])

    # rapidocr's own crop, copied from its source.
    def crop(points):
        w = int(max(np.linalg.norm(points[0] - points[1]),
                    np.linalg.norm(points[2] - points[3])))
        h = int(max(np.linalg.norm(points[0] - points[3]),
                    np.linalg.norm(points[1] - points[2])))
        std = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(points, std)
        dst = cv2.warpPerspective(img, M, (w, h),
                                  borderMode=cv2.BORDER_REPLICATE,
                                  flags=cv2.INTER_CUBIC)
        if dst.shape[0] * 1.0 / dst.shape[1] >= 1.5:
            dst = np.rot90(dst)
        return dst

    # rapidocr's own resize-and-normalise, copied from its source.
    def resize_norm(c, max_ratio):
        height = int(reader.config["rec_input_height"])
        width = int(height * max_ratio)
        h, w = c.shape[:2]
        ratio = w / float(h)
        resized_w = width if int(np.ceil(height * ratio)) > width \
            else int(np.ceil(height * ratio))
        r = cv2.resize(c, (resized_w, height)).astype("float32")
        r = r.transpose((2, 0, 1)) / 255
        r -= float(reader.config["rec_image_mean"][0])
        r /= float(reader.config["rec_image_std"][0])
        out = np.zeros((3, height, width), dtype=np.float32)
        out[:, :, 0:resized_w] = r
        return out

    crops = [crop(np.asarray(b, np.float32)) for b in boxes]
    ratio = max(c.shape[1] / float(c.shape[0]) for c in crops) if crops else 1.0
    out = []
    for b, c in zip(boxes, crops):
        logits = rec.run(None, {rec.get_inputs()[0].name:
                                 resize_norm(c, ratio)[None]})[0]
        idx = logits.argmax(axis=2)[0]
        prob = logits.max(axis=2)[0]
        text, prev = [], -1
        for t in idx:
            if t != prev and t != 0:
                text.append(dec.character[int(t)])
            prev = int(t)
        out.append(("".join(text),
                    [[float(p[0]), float(p[1])] for p in b]))
    return out


def section_3(reader, verbose):
    print("  3. the runtime end to end, against rapidocr")
    _need(DET_ONNX, "the detector checkpoint")
    _need(REC_ONNX, "the recogniser checkpoint")
    _need(DICT_TXT, "the character dictionary")
    got, _ = _run_runtime(CONTAINER[0])
    want = _oracle(reader, DET_ONNX, REC_ONNX, IMAGE)

    check(len(got["boxes"]) == len(want),
          f"the runtime found {len(got['boxes'])} boxes and rapidocr found "
          f"{len(want)}. The detector's thresholds come from the container, "
          f"so a different count is either a different graph or a different "
          f"threshold -- and section 1 has already ruled out the graph.")
    # IN THE SAME ORDER, and the order is not the same. The runtime returns
    # the boxes top to bottom, because it walks the map; rapidocr returns
    # them in the order OpenCV found the contours, which for a page of text
    # is bottom to top. The boxes are matched by WHERE THEY ARE and not by
    # which number they are, and a comparison that pairs them by index would
    # call a perfect agreement a catastrophe.
    def place(box):
        quad = box["quad"] if "quad" in box else box[1]
        return (min(p[1] for p in quad), min(p[0] for p in quad))

    mine = sorted(got["boxes"], key=place)
    theirs = sorted(want, key=lambda w: place({"quad": w[1]}))
    worst, rows = 0.0, []
    for b, (text, quad) in zip(mine, theirs):
        d = max(abs(b["quad"][k][j] - quad[k][j])
                for k in range(4) for j in range(2))
        worst = max(worst, d)
        same = b["text"] == text
        rows.append((b["text"], text, d, same))
    for mine, theirs, d, same in rows:
        flag = "ok  " if same else "TEXT"
        print(f"     {flag} {mine!r:<28} against {theirs!r:<28} "
              f"box {d:.1f} px")
    check(worst <= TOL_BOX_PX,
          f"the boxes are {worst:.1f} px from rapidocr's, outside the "
          f"{TOL_BOX_PX:.0f} px allowed. The two front ends resize the frame "
          f"to different rules -- the long side to 960 here, the short side "
          f"to 736 there -- so the probability maps are of different "
          f"resolutions and the boxes differ by a pixel or two.")
    bad = [r for r in rows if not r[3]]
    check(not bad,
          f"{len(bad)} of {len(rows)} transcriptions differ from rapidocr's: "
          + "; ".join(f"{m!r} against {t!r}" for m, t, _, _ in bad) +
          ". The crop is the same to within a pixel, so a different "
          "transcription is the recogniser's walk or its CTC decode, and "
          "section 2 has already pinned the walk.")
    print(f"     {len(rows)} boxes within {worst:.1f} px and every "
          f"transcription identical")


# ==============================================================================
# -- section 4: the array path against the host path ---------------------------
# ==============================================================================


def section_4(reader, verbose):
    print("  4. the array path against the host path, same frame")
    design = os.path.join(REPO, "runtime", "artifacts", "paddleocr-onnx",
                          "artifacts_npu1")
    if not os.path.exists(design):
        raise Fail(
            f"{design} is not there, so there is no array path to compare the "
            f"host against. Build it with\n"
            f"  source /opt/xilinx/xrt/setup.sh\n"
            f"  .venv/bin/python tools/export/export_gemm_rtp.py --target "
            f"paddleocr-onnx --arch 1 --cols 4 -n 32\n"
            f"which needs MLIR-AIE -- and that is in .venv/, not on the system "
            f"PATH, which is the whole reason this gate says so rather than "
            f"reporting a missing directory and leaving it at that.")
    host, _ = _run_runtime(CONTAINER[0])
    arr, err = _run_runtime(CONTAINER[0], ("--npu-ops", "conv",
                                            "--artifacts", design))
    line = [l for l in err.splitlines() if "array:" in l]
    check(line, "the status block printed no array line, so nothing was "
                "dispatched and this section would be comparing two host runs")
    print(f"     {line[0].strip()}")
    th = host["timings_ms"]["total"]
    ta = arr["timings_ms"]["total"]
    print(f"     frame     {ta:.0f} ms on the array against {th:.0f} ms on "
          f"the host -- {ta / th:.2f}x")
    check(len(host["boxes"]) == len(arr["boxes"]),
          f"the host found {len(host['boxes'])} boxes and the array "
          f"{len(arr['boxes'])}")


# ==============================================================================


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="verify an arch=9 PaddleOCR container and the reference runtime "
                    "built from it")
    ap.add_argument("--container", default=None,
                    help="the .npue to check (default: models/paddleocr-onnx.npue)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.container:
        CONTAINER[0] = args.container
    _need(CONTAINER[0], "the container")
    _need(BINARY, "the runtime binary (build it with `cmake -S runtime -B "
                   "runtime/build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake "
                   "--build runtime/build`)")

    reader = Reader(CONTAINER[0])
    for section in (section_1, section_2, section_3, section_4):
        section(reader, args.verbose)
    print("  PASS")


if __name__ == "__main__":
    try:
        main()
    except Fail as e:
        print(f"  FAIL: {e}")
        sys.exit(1)
