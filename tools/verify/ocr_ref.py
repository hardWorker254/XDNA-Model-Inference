# NpuEmbeddings -- a NumPy reference for the arch=9 container.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS FILE IS
# =================
# An independent implementation of everything the arch=9 container carries: the
# packed op list, the front end, the DB post-process and the CTC greedy decode.
#
# WHY IT EXISTS, AND WHY IT IS NOT THE PACKER'S TRACER
# ---------------------------------------------------
# The packer resolves an ONNX graph into a list of ops; this file EXECUTES that
# list. If it were the tracer's own code it would agree with the tracer by
# construction, which is the one thing a check cannot do. So the arithmetic here
# is written out again, from the op list's own description, and the two are
# compared against onnxruntime -- the checkpoint itself -- in
# tools/verify/verify_ocr.py.
#
# WHAT IT IS NOT
# --------------
# It is not fast. It is a reference: it says what the container MEANS, and the
# C++ runtime says what it DOES, and the gate compares them. A reference that
# ran at the runtime's speed would be a second implementation to maintain
# instead of an independent one to check against.
#
# THE OP LIST'S SPELLINGS, ONCE
# ------------------------------
#   conv          inputs[0], w, b, stride[2], pad[4] (top,left,bottom,right),
#                 dilation[2], group, act (the fused epilogue)
#   convtranspose  inputs[0], w ([Cin, Cout, kh, kw]), b, stride, pad,
#                 dilation, outpad
#   relu sigmoid hardswish hardsigmoid
#   add sub mul div sqrt    -- one operand plus a constant ("c"), or two
#   concat        axis
#   maxpool avgpool   kernel[2], stride[2], pad[4], ceil
#   gapool        a global average pool
#   resize        scale_h, scale_w -- NEAREST, with ONNX's own "asymmetric"
#                 coordinate transform and "floor" nearest mode
#   reshape       shape
#   transpose     perm
#   slice         axes, starts, ends, steps
#   squeeze unsqueeze   axes
#   matmul        a projection onto a container matrix (w, and optionally b),
#                 or a batched product of two activations
#   softmax       axis
#   reducemean    axes, keepdims

import json

import numpy as np

# -- the activations ----------------------------------------------------------


def hardswish(x, alpha=1.0 / 6.0, beta=0.5):
    """x * clip(x * alpha + beta, 0, 1).

    ONNX's HardSwish is `x * Max(0, Min(1, alpha*x + beta))` with the
    defaults alpha = 1/6 and beta = 0.5 -- which is the same function as
    `x * clip(x + 3, 0, 6) / 6`.
    """
    return x * np.clip(x * alpha + beta, 0.0, 1.0)


def hardsigmoid(x, alpha=1.0 / 6.0, beta=0.5):
    """clip(alpha*x + beta, 0, 1).

    The alpha is NOT always the default: the detector's SCSE blocks use
    alpha = 0.2, and the recogniser's use 1/6.
    """
    return np.clip(x * alpha + beta, 0.0, 1.0)


def apply_act(x, kind, alpha=1.0 / 6.0, beta=0.5):
    if kind == "none" or kind is None:
        return x
    if kind == "relu":
        return np.maximum(x, 0.0)
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if kind == "hardswish":
        return hardswish(x, alpha, beta)
    if kind == "hardsigmoid":
        return hardsigmoid(x, alpha, beta)
    raise ValueError(f"unknown activation {kind!r}")


# -- the convolutions ---------------------------------------------------------


def conv_nchw(x, w, b, stride, pad, dil, group, act="none", alpha=1.0 / 6.0,
               beta=0.5):
    """One ONNX convolution, NCHW, with the epilogue applied.

    `pad` is ONNX's FOUR terms [top, left, bottom, right], and `w` is
    [Cout, Cin/group, kh, kw] -- the second extent is the input channels PER
    GROUP, which is what makes a depthwise convolution's weight [Cin, 1, kh, kw]
    rather than a one-channel convolution's.
    """
    n, c, h, wd = x.shape
    co, cpg, kh, kw = w.shape
    pt, pl, pb, pr = pad
    sh, sw = stride
    dh, dw = dil
    eh = (kh - 1) * dh + 1
    ew = (kw - 1) * dw + 1
    oh = (h + pt + pb - eh) // sh + 1
    ow = (wd + pl + pr - ew) // sw + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    out = np.zeros((n, co, oh, ow), dtype=np.float32)
    cin = cpg * group
    for g in range(group):
        cg = co // group
        for i in range(kh):
            for j in range(kw):
                # The window at kernel position (i, j), strided: (N, C, OH, OW)
                patch = xp[:, g * cpg:(g + 1) * cpg,
                           i * dh:i * dh + oh * sh:sh,
                           j * dw:j * dw + ow * sw:sw]
                # (C, OH*OW) -- the reduction axis is the channel axis, so the
                # product is one GEMM per kernel position rather than one per
                # output pixel.
                p = np.ascontiguousarray(
                    patch.transpose(0, 2, 3, 1)).reshape(-1, cpg)
                k = np.ascontiguousarray(
                    w[g * cg:(g + 1) * cg, :, i, j].T)      # (C, cg)
                r = p @ k                                      # (N*OH*OW, cg)
                out[:, g * cg:(g + 1) * cg] += r.reshape(
                    n, oh, ow, cg).transpose(0, 3, 1, 2)
    if b is not None:
        out += b.reshape(1, -1, 1, 1)
    return apply_act(out, act, alpha, beta)


def convtranspose_nchw(x, w, b, stride, pad, dil, outpad):
    """One ONNX ConvTranspose, NCHW.

    `w` is [Cin, Cout, kh, kw] -- the INPUT channels first, which is the
    opposite of a convolution's and the reason a reader that took the first
    extent for the output count would build a head of the wrong width.

    The arithmetic is a SUM OF SHIFTED CONTRIBUTIONS: every input pixel is
    scattered into the output wherever the kernel puts it, and the output
    accumulates what several of them wrote. It is not a GEMM, and the array
    backend cannot express it -- which is why the two transposed convolutions
    in the DB head stay on the host.
    """
    n, c, h, wd = x.shape
    cin, co, kh, kw = w.shape
    pt, pl, pb, pr = pad
    sh, sw = stride
    dh, dw = dil
    eh = (kh - 1) * dh + 1
    ew = (kw - 1) * dw + 1
    oh = sh * (h - 1) + outpad[0] + eh - pt - pb
    ow = sw * (wd - 1) + outpad[1] + ew - pl - pr
    out = np.zeros((n, co, oh, ow), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            # WHERE THIS KERNEL POSITION LANDS. The output rows it writes are
            # STRIDED by the stride -- input row iy writes output row
            # iy*sh - pt + i*dh -- so the rows it covers are NOT a contiguous
            # run, and treating them as one is what put 479 rows of a
            # convolution's output where 120 belong.
            oy = np.arange(h) * sh - pt + i * dh
            ox = np.arange(wd) * sw - pl + j * dw
            vy = np.nonzero((oy >= 0) & (oy < oh))[0]
            vx = np.nonzero((ox >= 0) & (ox < ow))[0]
            if len(vy) == 0 or len(vx) == 0:
                continue
            # (N, Cin, ny, nx) @ (Cin, Cout) -> (N, ny, nx, Cout)
            p = x[:, :, vy][:, :, :, vx]
            r = np.tensordot(p, w[:, :, i, j], axes=([1], [0]))
            # The destination is (N, Cout, OH, OW) and the contribution is
            # (N, ny, nx, Cout), so the channels move to the front. The indices
            # are unique -- one input pixel per output position for a fixed
            # kernel element -- so the += is not an accumulation over
            # duplicates.
            out[:, :, oy[vy][:, None], ox[vx][None, :]] += r.transpose(0, 3, 1, 2)
    if b is not None:
        out += b.reshape(1, -1, 1, 1)
    return out


# -- the rest of the vocabulary ------------------------------------------------


def pool_nchw(x, kind, kernel, stride, pad, ceil):
    n, c, h, wd = x.shape
    kh, kw = kernel
    sh, sw = stride
    pt, pl, pb, pr = pad
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    num_h = h + pt + pb - kh
    num_w = wd + pl + pr - kw
    if ceil:
        oh = (num_h + sh - 1) // sh + 1
        ow = (num_w + sw - 1) // sw + 1
    else:
        oh = num_h // sh + 1
        ow = num_w // sw + 1
    is_max = kind == "maxpool"
    out = np.zeros((n, c, oh, ow), dtype=np.float32)
    for y in range(oh):
        ys = y * sh
        for x in range(ow):
            xs = x * sw
            win = xp[:, :, ys:ys + kh, xs:xs + kw]
            if win.shape[2] == 0 or win.shape[3] == 0:
                # A window entirely in the padding has no samples. A max over
                # nothing is -inf; an average over nothing is NaN. ONNX leaves
                # the count out of the divisor, so the mean of what there is.
                if is_max:
                    out[:, :, y, x] = -np.inf
                continue
            if is_max:
                out[:, :, y, x] = win.max(axis=(2, 3))
            else:
                cnt = win.shape[2] * win.shape[3]
                out[:, :, y, x] = win.sum(axis=(2, 3)) / cnt
    return out


def gapool_nchw(x):
    return x.mean(axis=(2, 3), keepdims=True)


def resize_nearest(x, scale_h, scale_w):
    """ONNX Resize, nearest, "asymmetric" + "floor".

    The source index of output row `d` is floor(d * in / out), and `out` is the
    RESIZED extent -- not `in * scale`, which is not the same number when the
    product is not an integer and the container's own declared output extent is.
    """
    n, c, h, wd = x.shape
    oh = int(np.floor(h * scale_h))
    ow = int(np.floor(wd * scale_w))
    ys = np.minimum((np.arange(oh) * h) // max(oh, 1), h - 1)
    xs = np.minimum((np.arange(ow) * wd) // max(ow, 1), wd - 1)
    return x[:, :, ys][:, :, :, xs]


def matmul(a, b):
    """ONNX's MatMul: a batched product with broadcasting batch axes."""
    if a.ndim == 1 and b.ndim == 1:
        return np.dot(a, b)
    if a.ndim == 1:
        return np.tensordot(a, b, axes=([-1], [-2]))
    if b.ndim == 1:
        return np.tensordot(a, b, axes=([-1], [0]))
    return np.matmul(a, b)


# -- the walk -----------------------------------------------------------------


class Graph:
    """One packed graph: the op list, the constants and the weights."""

    def __init__(self, ops, consts, weights):
        self.ops = ops
        self.consts = consts
        self.weights = weights

    @classmethod
    def from_container(cls, reader, prefix):
        ops = json.loads(reader.config[prefix + "_graph"])
        names = json.loads(reader.config[prefix + "_graph_nodes"])
        if len(ops) != len(names):
            raise ValueError(f"{prefix}: {len(ops)} ops and {len(names)} names")
        consts = {}
        weights = {}
        for name in reader.entries:
            if not name.startswith(prefix + "."):
                continue
            if name.endswith(".w"):
                weights[name] = reader.tensor(name)
            elif name.endswith(".b"):
                weights[name] = reader.tensor(name)
            elif ".const." in name:
                consts[name] = reader.tensor(name)
        return cls(ops, consts, weights)

    def run(self, x, upto=None, vals=None):
        """The op list, executed. `x` is the graph's input.

        `upto` stops the walk after that op and `vals` carries the values from
        an earlier walk, so a caller can compare the reference against a
        checkpoint ONE OP AT A TIME and be told which op first differs -- which
        is the difference between "the detector disagrees" and "the resize
        indexes its source row wrongly".
        """
        if vals is None:
            vals = [None] * len(self.ops)
        last = len(self.ops) - 1 if upto is None else upto
        for i, op in enumerate(self.ops):
            if i > last:
                break
            vals[i] = self.run_one(op, i, vals, x)
        return vals

    def run_one(self, op, i, vals, x):
        kind = op["op"]
        ins = op["inputs"]

        def take(k):
            j = ins[k]
            return x if j == -1 else vals[j]

        if kind == "conv":
            w = self.weights[op["w"]]
            b = self.weights[op["b"]] if op.get("b") else None
            out = conv_nchw(take(0), w, b, op["stride"], op["pad"],
                            op["dilation"], int(op.get("group", 1)),
                            op.get("act", "none"),
                            op.get("alpha", 1.0 / 6.0), op.get("beta", 0.5))
        elif kind == "convtranspose":
            w = self.weights[op["w"]]
            b = self.weights[op["b"]] if op.get("b") else None
            out = convtranspose_nchw(
                take(0), w, b, op["stride"], op["pad"], op["dilation"],
                op.get("outpad", [0, 0]))
        elif kind in ("relu", "sigmoid", "hardswish", "hardsigmoid"):
            out = apply_act(take(0), kind, op.get("alpha", 1.0 / 6.0),
                             op.get("beta", 0.5))
        elif kind in ("add", "sub", "mul", "div"):
            if op.get("c") is not None:
                a = take(0)
                c = self.consts[op["c"]]
                if kind == "add":
                    out = a + c
                elif kind == "sub":
                    out = a - c
                elif kind == "mul":
                    out = a * c
                else:
                    out = a / c
            else:
                a, b = take(0), take(1)
                if kind == "add":
                    out = a + b
                elif kind == "sub":
                    out = a - b
                elif kind == "mul":
                    out = a * b
                else:
                    out = a / b
        elif kind == "sqrt":
            out = np.sqrt(take(0))
        elif kind == "concat":
            out = np.concatenate([take(k) for k in range(len(ins))],
                                  axis=op["axis"])
        elif kind in ("maxpool", "avgpool", "averagepool"):
            out = pool_nchw(take(0), kind, op["kernel"], op["stride"],
                            op["pad"], int(op.get("ceil", 0)))
        elif kind == "gapool":
            out = gapool_nchw(take(0))
        elif kind == "resize":
            out = resize_nearest(take(0), op["scale_h"], op["scale_w"])
        elif kind == "reshape":
            out = take(0).reshape(op["shape"])
        elif kind == "transpose":
            out = np.transpose(take(0), axes=op["perm"])
        elif kind == "slice":
            a = take(0)
            for k, ax in enumerate(op["axes"]):
                st = op["starts"][k]
                en = op["ends"][k]
                sp = op["steps"][k]
                dim = a.shape[ax]
                if sp > 0:
                    s = st + dim if st < 0 else st
                    e = en + dim if en < 0 else en
                    s = min(max(s, 0), dim)
                    e = min(max(e, 0), dim)
                else:
                    s = st + dim if st < 0 else st
                    e = en + dim if en < 0 else en
                    s = min(max(s, -1), dim - 1)
                    e = min(max(e, -1), dim - 1)
                idx = list(range(s, e, sp))
                a = np.take(a, idx, axis=ax)
            out = a
        elif kind in ("squeeze", "unsqueeze"):
            a = take(0)
            axes = op["axes"]
            if kind == "squeeze":
                out = np.squeeze(a, axis=tuple(axes))
            else:
                out = np.expand_dims(a, axis=tuple(axes))
        elif kind == "matmul":
            if op.get("w"):
                w = self.weights[op["w"]]
                a = take(0)
                out = matmul(a, w)
                if op.get("b"):
                    out = out + self.weights[op["b"]]
            else:
                out = matmul(take(0), take(1))
        elif kind == "softmax":
            a = take(0)
            ax = op["axis"]
            m = np.max(a, axis=ax, keepdims=True)
            e = np.exp(a - m)
            out = e / e.sum(axis=ax, keepdims=True)
        elif kind == "reducemean":
            a = take(0)
            axes = tuple(op["axes"])
            out = a.mean(axis=axes, keepdims=bool(op.get("keepdims", 1)))
        else:
            raise ValueError(f"op {i}: unknown kind {kind!r}")
        want = op.get("oshape")
        if want is not None and list(out.shape) != list(want):
            raise ValueError(
                f"op {i} ({kind}): produced {list(out.shape)} and the "
                f"container declares {list(want)}")
        return out


# -- the packed op list, as an ONNX graph -----------------------------------------
#
# WHY THIS EXISTS
# ---------------
# The numpy reference above is the ORACLE for the runtime: it is an independent
# reading of the container's own op list, and the C++ runtime is compared
# against it. But it is numpy, and the v5 detector is 20 M parameters over
# 960x960 maps -- which the numpy reference cannot walk in a reasonable time.
#
# So the packer is checked a second way, and this is that way: the packed op
# list is MATERIALISED back into an ONNX graph, with the container's own
# weights, and run by onnxruntime against the checkpoint itself. Every
# difference the packer could have introduced -- a normalisation folded into
# the wrong tensor, a padding transcribed symmetrically, an activation's alpha
# read as the default -- shows up as a number, and the comparison is exact to
# float32 rather than to the numpy reference's tolerance.
#
# It is also the check that says the CONTAINER is a faithful copy of the
# CHECKPOINT: the two graphs differ -- the container has no normalisation ops
# and no shape nodes -- and the point is that they are the same function.
def _hardswish_nodes(src, dst, op, name, init):
    """HardSwish as ONNX nodes.

    onnxruntime's HardSwish kernel does not take the alpha and beta
    attributes -- it refuses the graph rather than ignoring them -- so a
    HardSwish whose parameters are not the defaults is written out as the
    arithmetic it stands for. The checkpoints here use the defaults, so this
    path is normally not taken; it is here because a checkpoint that did not
    would otherwise be unverifiable rather than wrong.
    """
    from onnx import helper as oh
    alpha = float(op.get("alpha", 1.0 / 6.0))
    beta = float(op.get("beta", 0.5))
    if abs(alpha - 1.0 / 6.0) < 1e-12 and abs(beta - 0.5) < 1e-12:
        return [oh.make_node("HardSwish", [src], [dst], name=name)]
    a = init(f"{name}_a", np.float32(alpha))
    b = init(f"{name}_b", np.float32(beta))
    lo = init(f"{name}_lo", np.float32(0.0))
    hi = init(f"{name}_hi", np.float32(1.0))
    s = f"{name}_c"
    return [
        oh.make_node("Mul", [src, a], [s], name=f"{name}_mul"),
        oh.make_node("Add", [s, b], [f"{name}_s"], name=f"{name}_add"),
        oh.make_node("Clip", [f"{name}_s", lo, hi], [f"{name}_t"],
                     name=f"{name}_clip"),
        oh.make_node("Mul", [src, f"{name}_t"], [dst], name=f"{name}_out"),
    ]


def _hardsigmoid_nodes(src, dst, op, name, init):
    """HardSigmoid as ONNX nodes, with its alpha and beta."""
    from onnx import helper as oh
    alpha = float(op.get("alpha", 1.0 / 6.0))
    beta = float(op.get("beta", 0.5))
    if abs(alpha - 1.0 / 6.0) < 1e-12 and abs(beta - 0.5) < 1e-12:
        return [oh.make_node("HardSigmoid", [src], [dst], name=name)]
    a = init(f"{name}_a", np.float32(alpha))
    b = init(f"{name}_b", np.float32(beta))
    lo = init(f"{name}_lo", np.float32(0.0))
    hi = init(f"{name}_hi", np.float32(1.0))
    return [
        oh.make_node("Mul", [src, a], [f"{name}_s"], name=f"{name}_mul"),
        oh.make_node("Add", [f"{name}_s", b], [f"{name}_t"],
                     name=f"{name}_add"),
        oh.make_node("Clip", [f"{name}_t", lo, hi], [dst], name=f"{name}_clip"),
    ]


def materialise(g, input_shape, prefix="x"):
    """The packed op list as an (onnx.ModelProto, names) pair.

    `input_shape` is the graph's input extent, which the container fixes.
    `names[i]` is the tensor op i produces, so that a caller can ask
    onnxruntime for every op's output at once and compare them one at a time.
    """
    import onnx
    from onnx import helper as oh
    from onnx import numpy_helper as nh

    inits = []
    nodes = []

    def init(name, arr):
        a = np.ascontiguousarray(arr)
        inits.append(nh.from_array(a, name))
        return name

    def const_of(op, key):
        return g.consts[op[key]]

    outs = [prefix]
    names = []
    shapes = []
    for i, op in enumerate(g.ops):
        kind = op["op"]
        ins = op["inputs"]
        # `outs[0]` is the graph's input and `outs[j + 1]` is op j's output,
        # so an operand index of -1 -- the image -- is `prefix` and any other
        # one is one past the op that produced it.
        src = [prefix if j == -1 else outs[j + 1] for j in ins]
        t = f"t{i}"

        if kind == "conv":
            w = init(f"w{i}", g.weights[op["w"]])
            args = [src[0], w]
            if op.get("b"):
                args.append(init(f"b{i}", g.weights[op["b"]]))
            nodes.append(oh.make_node(
                "Conv", args, [t], name=f"op{i}",
                kernel_shape=list(g.weights[op["w"]].shape[2:]),
                strides=list(op["stride"]), dilations=list(op["dilation"]),
                pads=list(op["pad"]), group=int(op.get("group", 1))))
            act = op.get("act", "none")
            if act != "none":
                at = f"t{i}a"
                if act == "relu":
                    nodes.append(oh.make_node("Relu", [t], [at], name=f"op{i}a"))
                elif act == "sigmoid":
                    nodes.append(oh.make_node("Sigmoid", [t], [at], name=f"op{i}a"))
                elif act == "hardswish":
                    nodes.extend(_hardswish_nodes(t, at, op, f"op{i}a", init))
                else:
                    nodes.extend(_hardsigmoid_nodes(t, at, op, f"op{i}a", init))
                t = at
        elif kind == "convtranspose":
            w = init(f"w{i}", g.weights[op["w"]])
            args = [src[0], w]
            if op.get("b"):
                args.append(init(f"b{i}", g.weights[op["b"]]))
            nodes.append(oh.make_node(
                "ConvTranspose", args, [t], name=f"op{i}",
                kernel_shape=list(g.weights[op["w"]].shape[2:]),
                strides=list(op["stride"]), dilations=list(op["dilation"]),
                pads=list(op["pad"]),
                output_padding=list(op.get("outpad", [0, 0])),
                group=int(op.get("group", 1))))
        elif kind in ("relu", "sigmoid", "hardswish", "hardsigmoid"):
            if kind == "relu":
                nodes.append(oh.make_node("Relu", src, [t], name=f"op{i}"))
            elif kind == "sigmoid":
                nodes.append(oh.make_node("Sigmoid", src, [t], name=f"op{i}"))
            elif kind == "hardswish":
                nodes.extend(_hardswish_nodes(src[0], t, op, f"op{i}", init))
            else:
                nodes.extend(_hardsigmoid_nodes(src[0], t, op, f"op{i}", init))
        elif kind in ("add", "sub", "mul", "div"):
            if op.get("c") is not None:
                c = init(f"c{i}", const_of(op, "c"))
                nodes.append(oh.make_node(
                    {"add": "Add", "sub": "Sub", "mul": "Mul",
                     "div": "Div"}[kind], [src[0], c], [t], name=f"op{i}"))
            else:
                nodes.append(oh.make_node(
                    {"add": "Add", "sub": "Sub", "mul": "Mul",
                     "div": "Div"}[kind], src, [t], name=f"op{i}"))
        elif kind == "sqrt":
            nodes.append(oh.make_node("Sqrt", src, [t], name=f"op{i}"))
        elif kind == "concat":
            nodes.append(oh.make_node("Concat", src, [t], name=f"op{i}",
                                       axis=int(op["axis"])))
        elif kind in ("maxpool", "avgpool", "averagepool"):
            oname = "MaxPool" if kind == "maxpool" else "AveragePool"
            nodes.append(oh.make_node(
                oname, src, [t], name=f"op{i}",
                kernel_shape=list(op["kernel"]), strides=list(op["stride"]),
                pads=list(op["pad"]), ceil_mode=int(op.get("ceil", 0))))
        elif kind == "gapool":
            nodes.append(oh.make_node("GlobalAveragePool", src, [t],
                                       name=f"op{i}"))
        elif kind == "resize":
            # The scales, and the three spellings of the source index. The
            # container records the SCALE and not the output extent, because
            # the extent is the scale applied to the input -- and the runtime
            # derives it the same way.
            scales = np.array([1.0, 1.0, float(op["scale_h"]),
                                float(op["scale_w"])], dtype=np.float32)
            sc = init(f"s{i}", scales)
            nodes.append(oh.make_node(
                "Resize", [src[0], "", sc], [t], name=f"op{i}", mode="nearest",
                coordinate_transformation_mode="asymmetric",
                nearest_mode="floor"))
        elif kind == "reshape":
            sh = init(f"sh{i}", np.asarray(op["shape"], dtype=np.int64))
            nodes.append(oh.make_node("Reshape", [src[0], sh], [t],
                                       name=f"op{i}"))
        elif kind == "transpose":
            nodes.append(oh.make_node("Transpose", src, [t], name=f"op{i}",
                                       perm=list(op["perm"])))
        elif kind == "slice":
            st = init(f"st{i}", np.asarray(op["starts"], dtype=np.int64))
            en = init(f"en{i}", np.asarray(op["ends"], dtype=np.int64))
            ax = init(f"ax{i}", np.asarray(op["axes"], dtype=np.int64))
            sp = init(f"sp{i}", np.asarray(op["steps"], dtype=np.int64))
            nodes.append(oh.make_node("Slice", [src[0], st, en, ax, sp], [t],
                                       name=f"op{i}"))
        elif kind in ("squeeze", "unsqueeze"):
            ax = init(f"ax{i}", np.asarray(op["axes"], dtype=np.int64))
            nodes.append(oh.make_node(
                "Squeeze" if kind == "squeeze" else "Unsqueeze",
                [src[0], ax], [t], name=f"op{i}"))
        elif kind == "matmul":
            if op.get("w"):
                w = init(f"w{i}", g.weights[op["w"]])
                nodes.append(oh.make_node("MatMul", [src[0], w], [t],
                                           name=f"op{i}"))
                if op.get("b"):
                    b = init(f"b{i}", g.weights[op["b"]])
                    t2 = f"t{i}b"
                    nodes.append(oh.make_node("Add", [t, b], [t2], name=f"op{i}b"))
                    t = t2
            else:
                nodes.append(oh.make_node("MatMul", src, [t], name=f"op{i}"))
        elif kind == "softmax":
            nodes.append(oh.make_node("Softmax", src, [t], name=f"op{i}",
                                       axis=int(op["axis"])))
        elif kind == "reducemean":
            ax = init(f"ax{i}", np.asarray(op["axes"], dtype=np.int64))
            nodes.append(oh.make_node("ReduceMean", [src[0], ax], [t],
                                       name=f"op{i}",
                                       keepdims=int(op.get("keepdims", 1))))
        else:
            raise ValueError(f"op {i}: unknown kind {kind!r}")
        outs.append(t)
        names.append(t)
        shapes.append(op.get("oshape"))

    # Every op's output is DECLARED, with the extent the container recorded for
    # it. Two reasons: onnxruntime refuses a graph whose outputs have no shape
    # at all, and a declared extent is a check in itself -- if the packer's
    # arithmetic and onnxruntime's disagree about an op's output size, the
    # shape inference says so here rather than at the first tensor that does
    # not fit.
    graph = oh.make_graph(nodes, "packed",
                          [oh.make_tensor_value_info(prefix, 1,
                                                     list(input_shape))],
                          [oh.make_tensor_value_info(outs[-1], 1,
                                                     [int(d) for d in
                                                      (shapes[-1] or [])])])
    for n, sh in zip(names, shapes):
        if sh:
            graph.value_info.append(
                oh.make_tensor_value_info(n, 1, [int(d) for d in sh]))
    graph.initializer.extend(inits)
    # OPSET 18, and not 14, for ONE reason: onnxruntime's ReduceMean at
    # opset 13 and 14 takes a single input and refuses the axes input that the
    # schema has had since opset 13. The recogniser's LayerNorm reduces over
    # one axis, and an opset that cannot express that is not one to verify
    # against. Opset 18 takes the axes as an input and takes Resize's mode as
    # a string, which is what the checkpoint's own nodes carry.
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
    model.ir_version = 8
    return model, names

