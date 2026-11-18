# NpuEmbeddings -- pack a PaddleOCR ONNX pair (detector + recogniser) into an
# arch=9 .npue.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS FILE IS
# =================
# The OCR family, and the first container in this tree that holds a RECOGNISER
# next to a detector: a text detector cannot read the text it finds, and a text
# recogniser cannot find text. So one container carries both graphs, the
# character dictionary, and the two front ends' geometry -- for the same reason
# arch=7 carries a palm detector and a hand-landmark net.
#
# The upstream source is https://huggingface.co/monkt/paddleocr-onnx: PaddleOCR's
# PP-OCRv3 and PP-OCRv5 exports, in ONNX, with the per-language dictionaries.
# Apache-2.0.
#
# WHY THE PACKER RESOLVES THE SHAPE PLUMBING AND THE RUNTIME DOES NOT
# ==================================================================
# Both graphs are traced at a FIXED input size -- the detector at
# det_size x det_size and the recogniser at rec_height x rec_width -- which is
# what makes an ARRAY path possible at all: every convolution's output extent is
# then a constant, and a design set can be compiled for it.
#
# With the extents constant, every shape-only node in the ONNX (Shape, the
# Reshape that consumes it, Slice's starts/ends, Squeeze/Unsqueeze's axes,
# Transpose's perm) is CONSTANT at pack time, and this packer resolves it into
# a plain attribute. The runtime therefore walks a list of ops whose output
# shapes are all known, and the container carries each op's declared output
# shape so the runtime can CHECK its own arithmetic against the checkpoint's
# rather than trust it.
#
# The alternative -- a shape interpreter in C++ -- would be a second ONNX
# runtime, and the two would disagree about a corner somewhere.
#
# WHAT RUNS WHERE
# ===============
#   ON THE HOST, by default:
#     the front end (letterbox, crop, perspective warp, normalise), the
#     activations, the elementwise ops, the pooling, the FPN's resizes, the
#     transposed convolutions, the attention blocks' matmuls, and the two
#     decodes (the DB post-process and the CTC greedy decode).
#   ON THE ARRAY, only under --npu-ops conv:
#     the DENSE convolutions of both graphs, as im2col plus dispatched GEMMs,
#     with the B panels pre-tiled and a stream named per distinct padded
#     (K, N). Depthwise convolutions stay on the host: a depthwise filter is
#     not a GEMM, and pretending otherwise is the wrong answer rather than a
#     slow one.
#
# Env: numpy + onnx. Both build-time only, as this whole directory is.

import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from tools.lib import conv_quant                                        # noqa: E402
from tools.lib.npue import (ARCH_PADDLEOCR_DB_SVTR_CTC, Writer,   # noqa: E402
                            cols_for_device, gemm_b_layout, mac_for_device, tile_b,
                            to_bf16_bits)

# Kept as a literal and compared against the C++ header by
# tools/verify/verify_ocr_geometry.py, because the two must agree and a test
# that reads both is the only thing that can say they do.
ARCH_STRING = "paddleocr_db_svtr_ctc"

# The op vocabulary of the packed graph. The runtime walks exactly these and
# refuses anything else BY NAME -- a dropped op is a network that computes
# something else and returns a plausible transcription.
GRAPH_OPS = (
    "conv", "convtranspose", "relu", "sigmoid", "hardswish", "hardsigmoid",
    "add", "sub", "mul", "div", "sqrt", "concat", "maxpool", "avgpool",
    "gapool", "resize", "reshape", "transpose", "slice", "squeeze",
    "unsqueeze", "matmul", "softmax", "reducemean",
)

# The activations that may be FUSED into a convolution's epilogue. Chosen
# because they are elementwise and shape-preserving, so a fused convolution and
# an unfused pair compute the same tensor -- and because a fused one does not
# spend a whole pass over the tensor to do it.
FUSABLE_ACTS = ("relu", "sigmoid", "hardswish", "hardsigmoid")


def _die(msg):
    raise SystemExit(f"ocr packer: {msg}")


# -- attribute readers --------------------------------------------------------

# onnx.AttributeProto field types
_AT = {"float": 1, "int": 2, "string": 3, "tensor": 4, "graph": 5,
       "floats": 6, "ints": 7, "strings": 8}
_AT_KIND = {v: k for k, v in _AT.items()}


def _raw(a):
    """An attribute's value, read by the field it is actually stored in.

    ONNX changed the representation of several attributes between opsets -- and
    even WITHIN one opset, `Resize.mode` is an int enum at opset 11 and a
    string at opset 18 -- so reading a field that is not there returns a zero
    rather than an error. That is the failure worth guarding: a nearest-
    neighbour resample is mode 0, and so is an attribute that was read from the
    wrong field.
    """
    kind = _AT_KIND.get(a.type)
    if kind == "int":
        return a.i
    if kind == "float":
        return a.f
    if kind == "string":
        return a.s.decode("utf-8") if isinstance(a.s, bytes) else a.s
    if kind == "ints":
        return list(a.ints)
    if kind == "floats":
        return list(a.floats)
    return None


def _attribute(n, kind, name):
    """The attribute `name`, required to be stored as `kind` ("i" or "ints")."""
    for a in n.attribute:
        if a.name != name:
            continue
        got = _AT_KIND.get(a.type)
        if got != ("ints" if kind == "ints" else "int"):
            _die(f"{n.name} carries \"{name}\" as {got}, not {kind}. This "
                 f"reader does not coerce between representations, because a "
                 f"coercion that happens to work on one export will read a "
                 f"zero on the next.")
        return list(a.ints) if kind == "ints" else a.i
    return None


def _norm_dim(d, rank, what):
    """An ONNX axis, negative-counting from the end."""
    d = int(d)
    if d < 0:
        d += rank
    if d < 0 or d >= rank:
        _die(f"{what} names axis {d}, which is not an axis of a rank-{rank} "
             f"tensor.")
    return d


def _slice_indices(dim, start, end, step):
    """ONNX's Slice index list for one axis.

    The clamping is the spec's, and it matters: a negative step clamps the
    START to -1 and the END to -1 rather than to 0, so that a slice of
    [0, 4) with step -1 yields the LAST element and not an empty range. A
    reader that clamped both to 0 would return nothing, and the shape it
    computes would be 0 where the checkpoint's is 1.
    """
    if step > 0:
        s = start + dim if start < 0 else start
        e = end + dim if end < 0 else end
        s = min(max(s, 0), dim)
        e = min(max(e, 0), dim)
        return list(range(s, e, step))
    s = start + dim if start < 0 else start
    e = end + dim if end < 0 else end
    s = min(max(s, -1), dim - 1)
    e = min(max(e, -1), dim - 1)
    return list(range(s, e, step))


def _broadcast_shape(a, b):
    """The ONNX broadcast of two shapes, right-aligned."""
    out = []
    n = max(len(a), len(b))
    for i in range(n):
        da = a[len(a) - n + i] if i >= n - len(a) else 1
        db = b[len(b) - n + i] if i >= n - len(b) else 1
        if da == db or da == 1 or db == 1:
            out.append(max(da, db))
        else:
            _die(f"shapes {list(a)} and {list(b)} do not broadcast.")
    return tuple(out)


def _matmul_shape(a, b, what):
    """The output extent of an ONNX MatMul, and nothing else.

    ONNX's MatMul is a BATCHED matrix product, and the batch axes broadcast
    against each other. The three spellings:

      rank 1 x rank 1   (K,) @ (K,)          -> ()
      rank 1 x rank 2+  (K,) @ (..., K, N)    -> (..., N)
      rank 2+ x rank 1  (..., M, K) @ (K,)    -> (..., M)
      rank 2+ x rank 2+ (..., M, K) @ (..., K, N) -> (..., M, N)

    The recogniser's attention block is the last one: (1, 8, 40, 15) times
    (1, 8, 15, 40) gives (1, 8, 40, 40), with the leading axes broadcast. A
    reader that treated a MatMul as a plain 2-D product would either refuse the
    graph or multiply the wrong thing, and both would still produce a number.
    """
    ar, br = len(a), len(b)
    if ar == 0 or br == 0:
        _die(f"{what} multiplies a rank-{ar} tensor by a rank-{br} one.")
    if ar == 1 and br == 1:
        if a[0] != b[0]:
            _die(f"{what} multiplies a {a[0]}-vector by a {b[0]}-vector.")
        return ()
    if ar == 1:
        if a[0] != b[-2]:
            _die(f"{what} multiplies a {a[0]}-vector by a matrix whose rows are "
                 f"{b[-2]} long.")
        return tuple(b[:-2]) + (b[-1],)
    if br == 1:
        if a[-1] != b[0]:
            _die(f"{what} multiplies a matrix whose columns are {a[-1]} long by "
                 f"a {b[0]}-vector.")
        return tuple(a[:-2]) + (a[-2],)
    if a[-1] != b[-2]:
        _die(f"{what} multiplies a matrix whose columns are {a[-1]} long by one "
             f"whose rows are {b[-2]} long.")
    return tuple(_broadcast_shape(a[:-2], b[:-2])) + (a[-2], b[-1])


class Tracer:
    """One ONNX graph, traced into the runtime's op list.

    A class rather than a function so that the shape table, the constant table
    and the consumer counts are properties of the trace rather than of a call
    stack -- the consumer counts in particular are what decide whether a
    BatchNormalization may be folded into its convolution, and a fold that is
    wrong changes every number downstream of it.
    """

    def __init__(self, model, prefix, input_shape):
        self.model = model
        self.prefix = prefix
        self.input_shape = tuple(input_shape)
        self.by_name = {}
        self.by_out = {}
        self.consumers = {}
        self.vals = {}        # tensor name -> ndarray (constants only)
        self.shapes = {}      # tensor name -> shape tuple
        self.ops = []         # the emitted op list
        self.op_src = []      # the ONNX tensor each op's first input came from
        self.node_names = []  # the checkpoint node each op came from
        self.conv_w = {}      # conv index -> (weight, bias|None)
        self.linear_w = {}    # linear index -> (weight, bias|None)
        self.consts = {}      # container tensor name -> ndarray
        self.const_name_of = {}   # ONNX tensor name -> container tensor name
        self._fused = set()   # node names folded into an earlier op
        self._n_const = 0
        self._n_conv = 0
        self._n_linear = 0
        self.producer = {}    # tensor name -> the index of the op that made it
        for i, n in enumerate(model.graph.node):
            if not n.name:
                # ONNX allows an unnamed node, and paddle2onnx emits some. A
                # name-keyed table would collapse them into one, and the
                # consumer counts below are what decide whether a normalisation
                # may be folded -- so they are given a name of their own, out
                # of the node's position in the list.
                n.name = f"node.{i}"
            if n.name in self.by_name:
                _die(f"the graph has two nodes named {n.name!r}. Node names "
                     f"are how this packer reports which checkpoint node an op "
                     f"came from, and two nodes with one name make that report "
                     f"ambiguous.")
            self.by_name[n.name] = n
            for o in n.output:
                if o:
                    self.by_out[o] = n
        for n in model.graph.node:
            for i in n.input:
                if i:
                    self.consumers.setdefault(i, []).append(n)

    # -- helpers -------------------------------------------------------------

    def _const(self, name, what):
        v = self.vals.get(name)
        if v is None:
            _die(f"{what} reads {name}, which is not a constant this packer "
                 f"can see. Every shape-only input in these graphs is a "
                 f"Constant node; a graph that computes one is a different "
                 f"export.")
        return v

    def _shape(self, name, what):
        s = self.shapes.get(name)
        if s is None:
            _die(f"{what} reads {name}, whose extent this packer does not "
                 f"know. Every tensor's extent is propagated here from the "
                 f"fixed input size; a hole means an op this packer does not "
                 f"implement produced it.")
        return s

    def _store_const(self, arr):
        name = f"{self.prefix}.const.{self._n_const}"
        self._n_const += 1
        self.consts[name] = np.ascontiguousarray(arr, dtype=np.float32)
        return name

    def _const_name(self, tensor, prefer=None):
        """The container tensor name for a constant ONNX tensor.

        A constant that the graph DERIVES -- a Reshape of a bias, which is how
        paddle2onnx expresses a depthwise convolution's bias -- is computed
        here and stored like any other constant, so the runtime never has to
        know that a shape node ever existed.

        `prefer` is the name the caller would LIKE this constant to have, and
        it is used only when the constant has not been stored yet. The
        convolution and projection weights carry their op's index in their
        name, because the array's panels are named after it; a constant that
        is already stored keeps the name it has, so that one tensor is stored
        once and not twice under two names.
        """
        name = self.const_name_of.get(tensor)
        if name is None:
            if prefer is not None:
                name = prefer
            else:
                name = self._store_const(self.vals[tensor])
            self.consts[name] = np.ascontiguousarray(self.vals[tensor],
                                                     dtype=np.float32)
            self.const_name_of[tensor] = name
        return name

    def _emit(self, op, node_name, out=None, **kw):
        # The ONNX tensor this op PRODUCES goes INTO the op, because it is a
        # property of the op: the runtime does not read it, but a gate does --
        # it compares this op's value against the checkpoint's own tensor of
        # that name, one op at a time. The name is not always the node's output,
        # because a BatchNormalization and an activation are folded into the
        # convolution in front of them, and the folded op's value is the
        # NORMALISED, ACTIVATED one. Comparing a packed op against the ONNX
        # node it came from would compare an activated value against a raw one
        # -- a difference of about 98%, which says nothing at all.
        self.ops.append(dict(op=op, out=out, **kw))
        self.node_names.append(node_name)
        # The ONNX tensor an op's FIRST activation input came from, kept beside
        # the op list rather than inside it. It is what lets the normalisation
        # fold find the convolution behind an "add a constant" node.
        self.op_src.append(None)

    def _index(self, tensor, n):
        """The op index that produced `tensor`, or -1 for the graph's input."""
        if tensor == self.model.graph.input[0].name:
            return -1
        idx = self.producer.get(tensor)
        if idx is None:
            _die(f"{n.name} reads {tensor}, which no earlier node produced.")
        return idx

    def _produced(self, n, shape):
        self.shapes[n.output[0]] = tuple(shape)
        self.producer[n.output[0]] = len(self.ops) - 1

    # -- the trace -----------------------------------------------------------

    def trace(self):
        g = self.model.graph
        out_name = g.output[0].name
        # The graph's input, at the FIXED size this container was packed for.
        # Everything downstream is propagated from here, which is why the size
        # is a packing decision and not a run-time one.
        self.shapes[g.input[0].name] = self.input_shape
        for n in g.node:
            if n.name in self._fused:
                continue
            self._node(n)
        if out_name not in self.shapes:
            _die(f"the graph's output {out_name} was never produced. The "
                 f"trace dropped a node the output depends on.")
        self.output_shape = self.shapes[out_name]
        self._prune_consts()
        return self

    def _prune_consts(self):
        """Drop the constants no op reads.

        The trace resolves every shape-only node at pack time -- a Reshape's
        target, a Slice's starts, a Concat's axis -- and a constant that fed
        one of those is a pack-time fact, not a runtime tensor. Several of
        them are DERIVED constants (the recogniser's bias is a Reshape of an
        initializer to [1, C, 1, 1], and paddle2onnx adds that), so they were
        stored once and then used by the trace rather than by any op.

        What the runtime needs is exactly what an op NAMES: a convolution's
        weight and bias, a projection's weight and bias, an elementwise op's
        constant. Everything else is dropped, so the container does not carry
        a hundred kilobytes of shape vectors that nothing will read.
        """
        live = set()
        for op in self.ops:
            for k in ("w", "b", "c"):
                v = op.get(k)
                if v:
                    live.add(v)
        dead = [k for k in self.consts if k not in live]
        for k in dead:
            del self.consts[k]
        self.pruned_consts = dead

    def _node(self, n):
        t = n.op_type

        if t == "Constant":
            attr = n.attribute[0]
            if attr.type != onnx.AttributeProto.TENSOR:
                _die(f"{n.name} is a Constant that is not a tensor. This "
                     f"packer stores tensors.")
            v = numpy_helper.to_array(attr.t)
            self.vals[n.output[0]] = v
            self.shapes[n.output[0]] = tuple(v.shape)
            return

        if t == "Identity":
            self.shapes[n.output[0]] = self._shape(n.input[0], n.name)
            self.producer[n.output[0]] = self._index(n.input[0], n)
            return

        if t == "Shape":
            s = self._shape(n.input[0], n.name)
            self.vals[n.output[0]] = np.asarray(s, dtype=np.int64)
            self.shapes[n.output[0]] = (len(s),)
            return

        if t == "BatchNormalization":
            self._batch_norm(n)
            return

        if t == "Conv":
            self._conv(n)
            return

        if t == "ConvTranspose":
            self._conv_transpose(n)
            return

        if t in ("Relu", "Sigmoid", "HardSwish", "HardSigmoid"):
            self._activation(n, t.lower())
            return

        if t in ("Add", "Sub", "Mul", "Div"):
            self._binary(n, t.lower())
            return

        if t == "Sqrt":
            x = self._shape(n.input[0], n.name)
            self._emit("sqrt", n.name, out=n.output[0],
                       inputs=[self._index(n.input[0], n)], oshape=list(x))
            self._produced(n, x)
            return

        if t == "Pow":
            # Pow(x, 2.0) is the LayerNorm's variance, and x*x is it exactly.
            # The exponent is read and checked rather than assumed: a Pow with a
            # different exponent is a different function, and x**3 read as x*x
            # is a variance that is wrong in a way nothing downstream reports.
            e = float(self._const(n.input[1], n.name).reshape(-1)[0])
            if abs(e - 2.0) > 1e-9:
                _die(f"{n.name} raises to {e}. The LayerNorm this graph "
                     f"carries squares its deviation, and a different exponent "
                     f"is a different normalisation.")
            i = self._index(n.input[0], n)
            x = self._shape(n.input[0], n.name)
            self._emit("mul", n.name, out=n.output[0], inputs=[i, i], oshape=list(x))
            self._produced(n, x)
            return

        if t == "MatMul":
            self._matmul(n)
            return

        if t == "Softmax":
            x = self._shape(n.input[0], n.name)
            axis = _attribute(n, "i", "axis")
            if axis is None:
                axis = -1
            axis = _norm_dim(axis, len(x), f"{n.name}'s softmax")
            self._emit("softmax", n.name, out=n.output[0], axis=axis,
                       inputs=[self._index(n.input[0], n)], oshape=list(x))
            self._produced(n, x)
            return

        if t == "ReduceMean":
            self._reduce_mean(n)
            return

        if t == "Concat":
            self._concat(n)
            return

        if t in ("MaxPool", "AveragePool"):
            # The op's name in the container is the RUNTIME's name, not
            # ONNX's: "avgpool", and not "averagepool". The vocabulary above
            # lists one spelling per op and the runtime walks exactly that
            # list, so a second spelling here would be a name the runtime
            # refuses -- with a message that says the container is from a
            # different day, which it is not.
            self._pool(n, "maxpool" if t == "MaxPool" else "avgpool")
            return

        if t == "GlobalAveragePool":
            x = self._shape(n.input[0], n.name)
            if len(x) != 4:
                _die(f"{n.name} pools a rank-{len(x)} tensor globally. The "
                     f"runtime's global average pool is NCHW.")
            out = (x[0], x[1], 1, 1)
            self._emit("gapool", n.name, out=n.output[0],
                       inputs=[self._index(n.input[0], n)], oshape=list(out))
            self._produced(n, out)
            return

        if t == "Resize":
            self._resize(n)
            return

        if t == "Reshape":
            self._reshape(n)
            return

        if t == "Transpose":
            self._transpose(n)
            return

        if t == "Slice":
            self._slice(n)
            return

        if t in ("Squeeze", "Unsqueeze"):
            self._squeeze(n, t.lower())
            return

        _die(f"{n.name} is a {t}, which this packer does not implement. The "
             f"op vocabulary is {', '.join(GRAPH_OPS)}. Every node in the "
             f"graph is either one of those or something this trace dropped, "
             f"and a dropped node is a network that computes something else "
             f"and returns a plausible transcription.")

    # -- the individual ops ---------------------------------------------------

    def _conv(self, n):
        w = self._const(n.input[1], n.name)
        if w.ndim != 4:
            _die(f"{n.name}'s weight is {list(w.shape)}, not rank 4. A "
                 f"convolution's weight is [Cout, Cin, kh, kw] and every index "
                 f"in the runtime's im2col depends on that order.")
        cout, cpg, kh, kw = w.shape
        group = _attribute(n, "i", "group")
        if group is None:
            group = 1
        # ONNX's Conv weight is [Cout, Cin/group, kh, kw]: the SECOND extent is
        # the input channels PER GROUP. A depthwise convolution is group == Cin
        # with one channel per group, so its weight reads [Cin, 1, kh, kw] and a
        # reader that took the second extent for the channel count would see a
        # one-channel convolution.
        cin = cpg * group
        strides = _attribute(n, "ints", "strides") or [1, 1]
        dil = _attribute(n, "ints", "dilations") or [1, 1]
        pads = _attribute(n, "ints", "pads")
        auto = _auto_pad(n)
        if auto != "NOTSET" and pads is not None:
            _die(f"{n.name} carries both auto_pad={auto!r} and explicit pads. "
                 f"ONNX says the two are alternatives, and a node that has both "
                 f"is not a node this packer can read.")
        if pads is None:
            pads = [0, 0, 0, 0]
        if len(pads) != 4:
            _die(f"{n.name} carries {len(pads)} padding terms. ONNX's order is "
                 f"[top, left, bottom, right] and this packer does not guess "
                 f"which order an export used.")
        b = None
        if len(n.input) >= 3:
            b = np.asarray(self._const(n.input[2], n.name), dtype=np.float32)
            if b.size != cout:
                _die(f"{n.name}'s bias has {b.size} values and its weight has "
                     f"{cout} output channels.")

        x = self._shape(n.input[0], n.name)
        if len(x) != 4:
            _die(f"{n.name} convolves a rank-{len(x)} tensor. The runtime's "
                 f"convolutions are NCHW.")
        if group != 1 and group != cin:
            _die(f"{n.name} has group={group} with {cin} input channels. A "
                 f"grouped convolution is a different op and is not run as a "
                 f"dense one.")
        if x[1] != cin:
            _die(f"{n.name} convolves {x[1]} channels with a weight for "
                 f"{cin}. The convolution's own arithmetic does not add up, "
                 f"and the extent this packer propagates would be wrong from "
                 f"here on.")
        eff = [(kh - 1) * dil[0] + 1, (kw - 1) * dil[1] + 1]
        # THE PADDING, FROM WHICHEVER SPELLING THE NODE USES.
        #
        # `pads` is the explicit one, and it is SYMMETRIC in every convolution
        # of these graphs except the two the v5 detector spells SAME_UPPER --
        # where the padding is NOT symmetric, because "the extra pixel goes at
        # the end" is the whole of what SAME_UPPER means.
        #
        # The asymmetry is one pixel on the bottom and the right, and it is not
        # a rounding detail: a 2x2 kernel over a 480-wide map reads one row
        # further right under SAME_UPPER than under a symmetric padding of the
        # same width, and the map's own content shifts with it. A reader that
        # transcribed the padding symmetrically would produce a 479-wide map
        # where the checkpoint has a 480-wide one, and the graph would then
        # refuse to join them -- which is the good case, because the bad case is
        # a map of the right width with the wrong pixels in it.
        if auto in ("SAME_UPPER", "SAME_LOWER"):
            upper = auto == "SAME_UPPER"
            oh = (x[2] + strides[0] - 1) // strides[0]
            ow = (x[3] + strides[1] - 1) // strides[1]
            pt, pb = _same_pad(x[2], eff[0], strides[0], upper)
            pl, pr = _same_pad(x[3], eff[1], strides[1], upper)
            pads = [pt, pl, pb, pr]
        elif auto == "VALID":
            pads = [0, 0, 0, 0]
            oh = (x[2] - eff[0]) // strides[0] + 1
            ow = (x[3] - eff[1]) // strides[1] + 1
        elif auto == "NOTSET":
            oh = (x[2] + pads[0] + pads[2] - eff[0]) // strides[0] + 1
            ow = (x[3] + pads[1] + pads[3] - eff[1]) // strides[1] + 1
        else:
            _die(f"{n.name} carries auto_pad={auto!r}. The runtime implements "
                 f"NOTSET, VALID, SAME_UPPER and SAME_LOWER, and the other two "
                 f"are not paddings this runtime has a spelling for.")
        if oh <= 0 or ow <= 0:
            _die(f"{n.name} produces a {oh}x{ow} output from a {x[2]}x{x[3]} "
                 f"input.")
        out = (x[0], cout, oh, ow)

        idx = self._n_conv
        self._n_conv += 1
        wname = self._const_name(n.input[1], f"{self.prefix}.conv.{idx}.w")
        bname = None
        if b is not None:
            bname = self._const_name(n.input[2], f"{self.prefix}.conv.{idx}.b")
        self.conv_w[idx] = (self.consts[wname],
                            None if bname is None else self.consts[bname])
        self._emit("conv", n.name, out=n.output[0], conv=idx, w=wname,
                   b=(bname if b is not None else None),
                   stride=[int(strides[0]), int(strides[1])],
                   pad=[int(p) for p in pads],
                   dilation=[int(dil[0]), int(dil[1])],
                   group=int(group), act="none",
                   inputs=[self._index(n.input[0], n)], oshape=list(out))
        self._produced(n, out)
        # The activation that follows is fused only when the convolution's
        # output has no other reader; a BatchNormalization between them is
        # folded below and does the fusing there.
        self._maybe_fuse_act(n.output[0], n)

    def _maybe_fuse_act(self, tensor, n):
        """Fuse the activation that follows `tensor` into its convolution.

        Only when `tensor` has EXACTLY ONE consumer, and that consumer is an
        activation. Two consumers means one of them would see the activated
        value instead of the raw one, which is a different network.
        """
        cons = self.consumers.get(tensor, [])
        if len(cons) != 1:
            return
        a = cons[0]
        kind = a.op_type.lower()
        if kind not in FUSABLE_ACTS:
            return
        if len(a.input) != 1 or a.input[0] != tensor:
            return
        # The op that produced `tensor` must be the convolution.
        idx = self.producer.get(tensor)
        if idx is None or self.ops[idx]["op"] != "conv":
            return
        self.ops[idx]["act"] = kind
        if kind in ("hardswish", "hardsigmoid"):
            # The activation's OWN parameters travel with it. They are not the
            # defaults: the DETECTOR's HardSigmoid has alpha 0.2 and the
            # RECOGNISER's has alpha 1/6, and a fused activation that lost its
            # alpha would be a different function in the middle of the network.
            alpha, beta = _act_params(a)
            self.ops[idx]["alpha"] = alpha
            self.ops[idx]["beta"] = beta
        self.shapes[a.output[0]] = self.shapes[tensor]
        self.producer[a.output[0]] = idx
        self._fused.add(a.name)
        # The convolution now computes the ACTIVATED value, so the tensor it
        # produces is the activation's own.
        self.ops[idx]["out"] = a.output[0]

    def _batch_norm(self, n):
        """FOLD a BatchNormalization into the convolution that precedes it.

        In inference mode a BatchNormalization is
            y = gamma * (x - mean) / sqrt(var + eps) + beta
        which is an affine map of the convolution's output, and an affine map
        of a convolution's output is a convolution with different weights and a
        different bias. Folding it is EXACT (up to the floating point of the
        fold itself), and it is what lets this runtime have no normalisation
        op at all -- the same reason arch=6's packer can refuse a graph with a
        BatchNormalization in it.

        THE TWO SHAPES THIS GRAPH HAS
        ------------------------------
        paddle2onnx emits two spellings, and both occur here:

          BN(Conv(x))         the normalisation reads the convolution's output
                              directly. The fold scales the convolution's
                              output-channel weights and folds beta, mean and
                              the convolution's own bias into its bias.

          BN(Conv(x) + c)     the normalisation reads an ADD, because the
                              convolution carries no bias of its own and
                              paddle2onnx adds a reshaped bias constant
                              instead. The fold scales the convolution's
                              weights, moves the constant to
                              scale*(c - mean) + beta, and leaves the add
                              where it is.

        The fold is only legal when the convolution's output has exactly one
        consumer, this one. A second consumer would see the normalised value
        instead of the raw one, which is a different network.
        """
        src = n.input[0]
        producer_idx = self.producer.get(src)
        if producer_idx is None:
            _die(f"{n.name} normalises {src}, which no earlier node produced.")
        op = self.ops[producer_idx]

        gamma = np.asarray(self._const(n.input[1], n.name), dtype=np.float32)
        beta = np.asarray(self._const(n.input[2], n.name), dtype=np.float32)
        mean = np.asarray(self._const(n.input[3], n.name), dtype=np.float32)
        var = np.asarray(self._const(n.input[4], n.name), dtype=np.float32)
        eps = None
        for a in n.attribute:
            if a.name == "epsilon":
                eps = float(_raw(a))
        if eps is None:
            eps = 1e-5
        scale = gamma / np.sqrt(var + np.float32(eps))
        cout = gamma.size

        if op["op"] in ("conv", "convtranspose"):
            # BN(conv(x)): the convolution itself is the producer.
            if len(self.consumers.get(src, [])) != 1:
                _die(f"{n.name} normalises {src}, which "
                     f"{len(self.consumers.get(src, []))} nodes read. Folding "
                     f"it into the convolution would give one of them the "
                     f"normalised value and the others the raw one.")
            self._fold_into_conv(producer_idx, scale, beta, mean, None)
            self.shapes[n.output[0]] = self.shapes[src]
            self.producer[n.output[0]] = producer_idx
            self._fused.add(n.name)
            # The convolution now computes the NORMALISED value, so the tensor
            # it produces is the normalisation's own -- not the convolution's.
            self.ops[producer_idx]["out"] = n.output[0]
            self._maybe_fuse_act(n.output[0], producer_idx)
            return

        if op["op"] == "add" and op.get("c") is not None and \
                len(op["inputs"]) == 1:
            # BN(conv(x) + c): the constant is the convolution's bias, and the
            # add is the only thing between the convolution and the
            # normalisation.
            conv_idx = self.producer.get(self.op_src[producer_idx])
            if conv_idx is None or self.ops[conv_idx]["op"] not in (
                    "conv", "convtranspose"):
                _die(f"{n.name} normalises {src}, which is a sum of a "
                     f"constant and something this packer cannot fold a "
                     f"normalisation into.")
            if len(self.consumers.get(src, [])) != 1:
                _die(f"{n.name} normalises {src}, which "
                     f"{len(self.consumers.get(src, []))} nodes read.")
            cname = op["c"]
            c = np.asarray(self.consts[cname], dtype=np.float32)
            pscale = self._param_for(c.shape, scale, cout, n.name)
            pmean = self._param_for(c.shape, mean, cout, n.name)
            pbeta = self._param_for(c.shape, beta, cout, n.name)
            # c' = scale*(c - mean) + beta, and the convolution's weights are
            # scaled so that it produces scale*conv(x).
            self.consts[cname] = np.ascontiguousarray(
                pscale * (c - pmean) + pbeta, dtype=np.float32)
            self._fold_into_conv(conv_idx, scale, beta, mean, None,
                                 keep_bias_zero=True)
            self.shapes[n.output[0]] = self.shapes[src]
            self.producer[n.output[0]] = producer_idx
            self._fused.add(n.name)
            # TWO TENSORS CHANGE HANDS HERE, and the pair is what the gate
            # compares.
            #
            # The ADD now computes the NORMALISED value -- the convolution was
            # scaled and the constant was moved to scale*(c - mean) + beta -- so
            # the tensor it produces is the normalisation's own.
            #
            # The CONVOLUTION, on the other hand, now computes scale*conv(x),
            # and there is NO tensor in the checkpoint with that value: the
            # checkpoint's convolution output is unscaled. So its "out" is
            # empty, and the gate compares the pair through the add. A gate
            # that compared the convolution against the checkpoint's own
            # convolution output would see a per-channel factor of the
            # normalisation's scale and call it a divergence.
            self.ops[producer_idx]["out"] = n.output[0]
            self.ops[conv_idx]["out"] = None
            return

        _die(f"{n.name} normalises the output of a {op['op']}, which this "
             f"packer cannot fold a normalisation into. The only two shapes "
             f"these graphs have are a normalisation straight after a "
             f"convolution and one after a convolution's bias add.")

    @staticmethod
    def _param_for(shape, v, cout, what):
        """The per-channel vector `v` reshaped to broadcast over `shape`."""
        if len(shape) == 4 and shape[1] == cout:
            return v.reshape(1, cout, 1, 1)
        if len(shape) == 1 and shape[0] == cout:
            return v.reshape(cout)
        raise SystemExit(
            f"ocr packer: {what} folds a normalisation over {cout} channels "
            f"into a constant of shape {list(shape)}, which does not have a "
            f"channel axis to fold them along. The bias add would broadcast "
            f"them along the WIDTH, which is a different network.")

    def _fold_into_conv(self, conv_idx, scale, beta, mean, extra, 
                         keep_bias_zero=False):
        """Scale a convolution's output channels and set its bias.

        The folded weight and bias go into NEW tensors rather than into the
        ones the convolution had. The convolution's own weight might be a
        constant that some other op also reads -- paddle2onnx builds a
        convolution's bias as a Reshape of an initializer, and the reshaped
        array is a constant like any other -- and scaling it in place would
        change that other op's constant too. A folded convolution gets its
        own copy; the old one is dropped by the packer if nothing reads it.
        """
        op = self.ops[conv_idx]
        old_w = op["w"]
        w = self.consts[old_w]
        cout = w.shape[0] if op["op"] == "conv" else w.shape[1]
        if scale.size != cout:
            _die(f"{op.get('conv')} has {cout} output channels and the "
                 f"normalisation has {scale.size}.")
        if op["op"] == "conv":
            w2 = w * scale.reshape(-1, 1, 1, 1)
        else:
            # A ConvTranspose's weight is [Cin, Cout, kh, kw]: the OUTPUT
            # channel is the second axis, not the first.
            w2 = w * scale.reshape(1, -1, 1, 1)
        old_b = op["b"]
        b = self.consts[old_b] if old_b else np.zeros(cout, dtype=np.float32)
        if keep_bias_zero:
            # The bias lives in the add's constant; the convolution keeps none.
            b2 = b * scale
        else:
            b2 = b * scale + (beta - mean * scale)
        wname = f"{self.prefix}.conv.{op['conv']}.w"
        bname = f"{self.prefix}.conv.{op['conv']}.b"
        self.consts[wname] = np.ascontiguousarray(w2, dtype=np.float32)
        self.consts[bname] = np.ascontiguousarray(b2, dtype=np.float32)
        op["w"] = wname
        op["b"] = bname
        self.conv_w[op["conv"]] = (w2, b2)

    def _conv_transpose(self, n):
        w = self._const(n.input[1], n.name)
        if w.ndim != 4:
            _die(f"{n.name}'s weight is {list(w.shape)}, not rank 4.")
        cin, cout, kh, kw = w.shape
        strides = _attribute(n, "ints", "strides") or [1, 1]
        dil = _attribute(n, "ints", "dilations") or [1, 1]
        pads = _attribute(n, "ints", "pads") or [0, 0, 0, 0]
        out_pad = _attribute(n, "ints", "output_padding") or [0, 0]
        if len(pads) != 4:
            _die(f"{n.name} carries {len(pads)} padding terms.")
        if pads[0] != pads[2] or pads[1] != pads[3]:
            _die(f"{n.name} pads asymmetrically ({pads}). The runtime's "
                 f"transposed convolution pads symmetrically.")
        b = None
        if len(n.input) >= 3:
            b = np.asarray(self._const(n.input[2], n.name), dtype=np.float32)
            if b.size != cout:
                _die(f"{n.name}'s bias has {b.size} values and its weight has "
                     f"{cout} output channels.")
        x = self._shape(n.input[0], n.name)
        if len(x) != 4:
            _die(f"{n.name} convolves a rank-{len(x)} tensor.")
        eff = [(kh - 1) * dil[0] + 1, (kw - 1) * dil[1] + 1]
        oh = strides[0] * (x[2] - 1) + out_pad[0] + eff[0] - pads[0] - pads[2]
        ow = strides[1] * (x[3] - 1) + out_pad[1] + eff[1] - pads[1] - pads[3]
        out = (x[0], cout, oh, ow)
        idx = self._n_conv
        self._n_conv += 1
        wname = self._const_name(n.input[1], f"{self.prefix}.conv.{idx}.w")
        bname = None
        if b is not None:
            bname = self._const_name(n.input[2], f"{self.prefix}.conv.{idx}.b")
        self.conv_w[idx] = (self.consts[wname],
                            None if bname is None else self.consts[bname])
        self._emit("convtranspose", n.name, out=n.output[0], conv=idx, w=wname,
                   b=(bname if b is not None else None),
                   stride=[int(strides[0]), int(strides[1])],
                   pad=[int(p) for p in pads],
                   dilation=[int(dil[0]), int(dil[1])],
                   outpad=[int(out_pad[0]), int(out_pad[1])],
                   inputs=[self._index(n.input[0], n)], oshape=list(out))
        self._produced(n, out)

    def _activation(self, n, kind):
        x = self._shape(n.input[0], n.name)
        kw = {}
        if kind in ("hardswish", "hardsigmoid"):
            a, b = _act_params(n)
            kw["alpha"] = a
            kw["beta"] = b
        self._emit(kind, n.name, out=n.output[0],
                   inputs=[self._index(n.input[0], n)], oshape=list(x), **kw)
        self._produced(n, x)

    def _binary(self, n, kind):
        a = self._shape(n.input[0], n.name)
        bshape = self._shape(n.input[1], n.name)
        out = _broadcast_shape(a, bshape)
        if n.input[1] in self.vals:
            cname = self._const_name(n.input[1])
            self._emit(kind, n.name, out=n.output[0],
                       inputs=[self._index(n.input[0], n)],
                       c=cname, oshape=list(out))
            self.op_src[-1] = n.input[0]
        elif n.input[0] in self.vals:
            cname = self._const_name(n.input[0])
            self._emit(kind, n.name, out=n.output[0],
                       inputs=[self._index(n.input[1], n)],
                       c=cname, oshape=list(out))
            self.op_src[-1] = n.input[1]
        else:
            self._emit(kind, n.name, out=n.output[0],
                       inputs=[self._index(n.input[0], n),
                               self._index(n.input[1], n)], oshape=list(out))
        self._produced(n, out)

    def _matmul(self, n):
        """A MatMul, in ONNX's general sense rather than a narrower one.

        Three spellings occur in these graphs, and the runtime implements ONNX's
        rule for all of them:

          x @ W       a projection onto a matrix from the container. W is rank
                      2; the activation is rank 2 or more. This is the shape
                      the RECOGNISER's qkv and head projections have.

          A @ B       two activations, batched. The recogniser's attention
                      block multiplies (1, 8, 40, 15) by (1, 8, 15, 40) and
                      gets (1, 8, 40, 40) -- the leading axes are BATCH axes
                      and they broadcast.

          W @ x       a rank-2 matrix times a rank-1 activation. It does not
                      occur in these exports, and it is supported because the
                      runtime's rule is ONNX's rule and not a special case
                      fitted to one graph.
        """
        a_name, b_name = n.input[0], n.input[1]
        a_const = a_name in self.vals
        b_const = b_name in self.vals
        if a_const and b_const:
            _die(f"{n.name} multiplies two constants. That is a weight, and "
                 f"the runtime has no place to put it.")

        a = self._shape(a_name, n.name)
        b = self._shape(b_name, n.name)

        wname = None
        idx = None
        if b_const:
            # x @ W
            w = np.asarray(self.vals[b_name], dtype=np.float32)
            if w.ndim != 2:
                _die(f"{n.name}'s right operand is {list(w.shape)}, not rank 2. "
                     f"A projection matrix is rank 2, and the runtime's "
                     f"projection reads the weight as [K, N].")
            out = _matmul_shape(a, list(w.shape), n.name)
            idx = self._n_linear
            self._n_linear += 1
            wname = self._const_name(b_name, f"{self.prefix}.linear.{idx}.w")
            inputs = [self._index(a_name, n)]
        elif a_const:
            # W @ x
            w = np.asarray(self.vals[a_name], dtype=np.float32)
            if w.ndim != 2:
                _die(f"{n.name}'s left operand is {list(w.shape)}, not rank 2. "
                     f"A projection matrix is rank 2, and the runtime's "
                     f"projection reads the weight as [K, N].")
            out = _matmul_shape(list(w.shape), b, n.name)
            idx = self._n_linear
            self._n_linear += 1
            wname = self._const_name(a_name, f"{self.prefix}.linear.{idx}.w")
            inputs = [self._index(b_name, n)]
        else:
            out = _matmul_shape(a, b, n.name)
            inputs = [self._index(a_name, n), self._index(b_name, n)]

        out = tuple(out)
        # The bias, if the graph adds one right after: folded into the op. Only
        # for a PROJECTION -- the Add that follows a batched attention product
        # is a mask, not a bias, and folding a mask into the matrix product
        # would be a different network.
        bname = None
        fused_add = None
        if wname is not None and len(out) >= 1:
            cons = self.consumers.get(n.output[0], [])
            if len(cons) == 1:
                c = cons[0]
                if c.op_type == "Add" and len(c.input) == 2 and \
                        c.input[1] in self.vals:
                    bb = np.asarray(self.vals[c.input[1]], dtype=np.float32)
                    if bb.size == out[-1]:
                        bname = f"{self.prefix}.linear.{idx}.b"
                        self.consts[bname] = np.ascontiguousarray(bb)
                        fused_add = c
        if idx is not None:
            self.linear_w[idx] = (self.consts[wname],
                                  None if bname is None else self.consts[bname])
        self._emit("matmul", n.name, w=wname, b=bname,
                   out=(fused_add.output[0] if fused_add is not None
                        else n.output[0]),
                   inputs=inputs, oshape=list(out))
        # THE FOLDED ADD'S PRODUCER IS SET AFTER THE EMIT, and that ordering is
        # the whole of a bug that was here once. The Add's output now comes from
        # the MATMUL -- the bias is inside it -- so its producer is the matmul's
        # index. Setting it BEFORE the emit would name the op BEFORE the
        # matmul, and the recogniser's Reshape that reads the biased projection
        # would then be handed the projection's INPUT instead: a tensor of
        # 120 columns where it wanted 360. The shapes all stayed right -- the
        # reshape's own target was computed from the right extent -- and the
        # graph produced a recogniser that read the wrong tensor.
        if fused_add is not None:
            self._fused.add(fused_add.name)
            self.shapes[fused_add.output[0]] = out
            self.producer[fused_add.output[0]] = len(self.ops) - 1
        self._produced(n, out)

    def _reduce_mean(self, n):
        x = self._shape(n.input[0], n.name)
        axes = _attribute(n, "ints", "axes")
        if axes is None and len(n.input) > 1:
            axes = [int(v) for v in np.asarray(self._const(n.input[1], n.name))
                    .reshape(-1)]
        if axes is None:
            axes = list(range(len(x)))
        keepdims = _attribute(n, "i", "keepdims")
        if keepdims is None:
            keepdims = 1
        axes = sorted(_norm_dim(a, len(x), f"{n.name}'s reduction") for a in axes)
        out = list(x)
        if keepdims:
            for a in axes:
                out[a] = 1
        else:
            out = [d for i, d in enumerate(out) if i not in axes]
        self._emit("reducemean", n.name, out=n.output[0], axes=axes,
                   keepdims=int(keepdims),
                   inputs=[self._index(n.input[0], n)], oshape=out)
        self._produced(n, out)

    def _concat(self, n):
        shapes = [self._shape(i, n.name) for i in n.input]
        axis = _attribute(n, "i", "axis")
        if axis is None:
            axis = 0
        axis = _norm_dim(axis, len(shapes[0]), f"{n.name}'s join")
        out = list(shapes[0])
        out[axis] = sum(s[axis] for s in shapes)
        # A JOIN OF CONSTANTS IS A CONSTANT -- which is how the recogniser's
        # attention block builds the next Reshape's target extent out of a
        # Shape and a literal.
        if all(i in self.vals for i in n.input):
            arr = np.concatenate([np.asarray(self.vals[i]) for i in n.input],
                                 axis=axis)
            self.vals[n.output[0]] = arr
            self.shapes[n.output[0]] = tuple(arr.shape)
            self.const_name_of[n.output[0]] = self._store_const(arr)
            return
        self._emit("concat", n.name, out=n.output[0], axis=axis,
                   inputs=[self._index(i, n.name) for i in n.input],
                   oshape=out)
        self._produced(n, out)

    def _pool(self, n, kind):
        x = self._shape(n.input[0], n.name)
        k = _attribute(n, "ints", "kernel_shape")
        st = _attribute(n, "ints", "strides") or k
        pads = _attribute(n, "ints", "pads")
        ceil_mode = _attribute(n, "i", "ceil_mode") or 0
        auto = _auto_pad(n)
        if auto != "NOTSET" and pads is not None:
            _die(f"{n.name} carries both auto_pad={auto!r} and explicit pads.")
        if pads is None:
            pads = [0, 0, 0, 0]
        if len(pads) != 4:
            _die(f"{n.name} carries {len(pads)} padding terms.")
        out = list(x)
        for i, dim in ((0, 2), (1, 3)):
            num = x[dim] + pads[i] + pads[i + 2] - k[i]
            if ceil_mode:
                out[dim] = (num + st[i] - 1) // st[i] + 1
            else:
                out[dim] = num // st[i] + 1
        # SAME_UPPER AND SAME_LOWER, for the pooling. The v5 detector's stem
        # pools with them, and the pooling's output extent is then the input's
        # divided by the stride and rounded UP -- not the floor of the padded
        # window, which is what ceil_mode alone gives. The difference is one
        # row of the output, and the stem's two branches are JOINED afterwards,
        # so a pooling that came out one row short would be a join of a
        # 480-row map with a 479-row one.
        if auto in ("SAME_UPPER", "SAME_LOWER"):
            upper = auto == "SAME_UPPER"
            for i, dim in ((0, 2), (1, 3)):
                out[dim] = (x[dim] + st[i] - 1) // st[i]
                begin, end = _same_pad(x[dim], k[i], st[i], upper)
                pads[i] = begin
                pads[i + 2] = end
        elif auto == "VALID":
            pads = [0, 0, 0, 0]
            for i, dim in ((0, 2), (1, 3)):
                out[dim] = (x[dim] - k[i]) // st[i] + 1
        elif auto != "NOTSET":
            _die(f"{n.name} carries auto_pad={auto!r}.")
        self._emit(kind, n.name, out=n.output[0],
                   kernel=[int(k[0]), int(k[1])],
                   stride=[int(st[0]), int(st[1])],
                   pad=[int(p) for p in pads], ceil=int(ceil_mode),
                   inputs=[self._index(n.input[0], n)], oshape=out)
        self._produced(n, out)

    def _resize(self, n):
        x = self._shape(n.input[0], n.name)
        mode = _resize_mode(n)
        if "nearest" not in mode:
            _die(f"{n.name} resizes with mode {mode!r}, not nearest. The "
                 f"detector's FPN upsamples by point sampling; a linear or "
                 f"cubic resample is a different network.")
        # THE TWO SPELLINGS OF "WHERE IS THE SOURCE PIXEL". ONNX's nearest
        # resample has three coordinate transformations and two rounding modes,
        # and the two the runtime implements -- asymmetric and floor -- are the
        # ones these graphs use. They are CHECKED here rather than assumed,
        # because the difference between them is one pixel of shift in the
        # upsampled probability map, and one pixel is a quarter of a character
        # at the detector's output stride.
        for attr, want in (("coordinate_transformation_mode", "asymmetric"),
                           ("nearest_mode", "floor")):
            got = _resize_attr(n, attr)
            if got is not None and got != want:
                _die(f"{n.name} resizes with {attr} {got!r}, not {want!r}. The "
                     f"runtime's nearest resample is {want}, and the two "
                     f"disagree by a pixel of shift in the upsampled map.")
        scales = None
        if len(n.input) > 2 and n.input[2] in self.vals:
            scales = np.asarray(self.vals[n.input[2]], dtype=np.float64).ravel()
        elif len(n.input) > 3 and n.input[3] in self.vals:
            scales = np.asarray(self.vals[n.input[3]], dtype=np.float64).ravel()
        if scales is None:
            _die(f"{n.name} carries no scales this packer can read. A "
                 f"nearest-neighbour resample without a scale is a resize to "
                 f"an unknown size.")
        if len(scales) != 4:
            _die(f"{n.name} carries {len(scales)} scales. A four-axis NCHW "
                 f"scale vector is the only shape this reader takes.")
        oh = int(np.floor(x[2] * scales[2]))
        ow = int(np.floor(x[3] * scales[3]))
        out = (x[0], x[1], oh, ow)
        self._emit("resize", n.name, out=n.output[0], scale_h=float(scales[2]),
                   scale_w=float(scales[3]),
                   inputs=[self._index(n.input[0], n)], oshape=list(out))
        self._produced(n, out)

    def _reshape(self, n):
        x = self._shape(n.input[0], n.name)
        target = [int(v) for v in np.asarray(self._const(n.input[1], n.name))
                  .reshape(-1)]
        allowzero = _attribute(n, "i", "allowzero") or 0
        total = 1
        for d in x:
            total *= d
        # ONNX's Reshape: a 0 means "take the input's extent at this position",
        # unless allowzero says otherwise; a -1 means "infer from the rest".
        neg = [i for i, d in enumerate(target) if d == -1]
        if len(neg) > 1:
            _die(f"{n.name} reshapes to {target}, which has more than one "
                 f"-1. At most one extent can be inferred.")
        out = []
        known = 1
        for i, d in enumerate(target):
            if d == 0 and not allowzero:
                out.append(x[i])
            else:
                out.append(d)
            if d != -1:
                known *= max(d, 1)
        if neg:
            if known == 0 or total % known != 0:
                _die(f"{n.name} reshapes {list(x)} to {target}, which does not "
                     f"divide the element count.")
            out[neg[0]] = total // known
        prod = 1
        for d in out:
            prod *= d
        if prod != total:
            _die(f"{n.name} reshapes {list(x)} to {out}, which holds {prod} "
                 f"values against the input's {total}.")
        # A RESHAPE OF A CONSTANT IS A CONSTANT. This is not a corner case: it
        # is how paddle2onnx expresses a convolution's bias, which it emits as
        # a Reshape of the bias initializer to [1, C, 1, 1] and an Add of that.
        # Resolving it here means the runtime never sees a shape node at all.
        if n.input[0] in self.vals:
            arr = np.asarray(self.vals[n.input[0]])
            try:
                arr = arr.reshape(out)
            except ValueError as e:
                _die(f"{n.name} reshapes a constant of shape "
                     f"{list(arr.shape)} to {out}: {e}")
            self.vals[n.output[0]] = arr
            self.shapes[n.output[0]] = tuple(out)
            self.const_name_of[n.output[0]] = self._store_const(arr)
            return
        self._emit("reshape", n.name, out=n.output[0], shape=out,
                   inputs=[self._index(n.input[0], n)], oshape=list(out))
        self._produced(n, out)

    def _transpose(self, n):
        x = self._shape(n.input[0], n.name)
        perm = _attribute(n, "ints", "perm")
        if perm is None:
            perm = list(range(len(x) - 1, -1, -1))
        out = [x[p] for p in perm]
        # A TRANSPOSE OF A CONSTANT IS A CONSTANT. Same reason as the Reshape
        # above it: a weight that the graph transposes on the way to a
        # projection is a constant, and the runtime should never learn that a
        # shape node existed.
        if n.input[0] in self.vals:
            arr = np.asarray(self.vals[n.input[0]])
            arr = np.transpose(arr, axes=perm)
            self.vals[n.output[0]] = arr
            self.shapes[n.output[0]] = tuple(arr.shape)
            self.const_name_of[n.output[0]] = self._store_const(arr)
            return
        self._emit("transpose", n.name, out=n.output[0],
                   perm=[int(p) for p in perm],
                   inputs=[self._index(n.input[0], n)], oshape=out)
        self._produced(n, out)

    def _slice(self, n):
        x = self._shape(n.input[0], n.name)
        rank = len(x)
        starts = [int(v) for v in np.asarray(self._const(n.input[1], n.name))
                  .reshape(-1)]
        ends = [int(v) for v in np.asarray(self._const(n.input[2], n.name))
                .reshape(-1)]
        axes = None
        steps = None
        if len(n.input) > 3 and n.input[3] in self.vals:
            axes = [int(v) for v in np.asarray(self.vals[n.input[3]]).reshape(-1)]
        if len(n.input) > 4 and n.input[4] in self.vals:
            steps = [int(v) for v in np.asarray(self.vals[n.input[4]]).reshape(-1)]
        if axes is None:
            axes = list(range(len(starts)))
        if steps is None:
            steps = [1] * len(starts)
        axes = [_norm_dim(a, rank, f"{n.name}'s slice") for a in axes]
        out = list(x)
        picks = []
        for k, a in enumerate(axes):
            idx = _slice_indices(x[a], starts[k], ends[k], steps[k])
            picks.append((a, idx))
            out[a] = len(idx)
        # A SLICE OF A SHAPE VECTOR IS A SHAPE VECTOR. The recogniser's
        # attention block reads one extent out of a Shape and uses it to build
        # the next Reshape's target, and with a fixed input size every one of
        # those is a constant -- so it is computed here rather than by a shape
        # interpreter in the runtime.
        if n.input[0] in self.vals:
            arr = np.asarray(self.vals[n.input[0]])
            for a, idx in picks:
                arr = np.take(arr, idx, axis=a)
            self.vals[n.output[0]] = arr
            self.shapes[n.output[0]] = tuple(arr.shape)
            self.const_name_of[n.output[0]] = self._store_const(arr)
            return
        self._emit("slice", n.name, out=n.output[0], axes=[int(a) for a in axes],
                   starts=starts, ends=ends, steps=steps,
                   inputs=[self._index(n.input[0], n)], oshape=out)
        self._produced(n, out)

    def _squeeze(self, n, kind):
        x = self._shape(n.input[0], n.name)
        axes = None
        if len(n.input) > 1 and n.input[1] in self.vals:
            axes = [int(v) for v in np.asarray(self.vals[n.input[1]]).reshape(-1)]
        else:
            axes = _attribute(n, "ints", "axes")
        if kind == "squeeze":
            if axes is None:
                axes = [i for i, d in enumerate(x) if d == 1]
            axes = sorted(_norm_dim(a, len(x), f"{n.name}'s squeeze")
                          for a in axes)
            out = [d for i, d in enumerate(x) if i not in axes]
        else:
            if axes is None:
                _die(f"{n.name} unsqueezes without naming its axes. The "
                     f"runtime's unsqueeze is explicit, because the two "
                     f"spellings of the same tensor differ in rank and rank is "
                     f"what every later op reads.")
            rank = len(x) + len(axes)
            axes = sorted(_norm_dim(a, rank, f"{n.name}'s unsqueeze")
                          for a in axes)
            out = list(x)
            for a in axes:
                out.insert(a, 1)
        # A SQUEEZE OF A SHAPE VECTOR IS A SHAPE VECTOR: the recogniser's
        # attention block squeezes the extent it sliced out of a Shape into a
        # scalar and puts it back together with a literal.
        if n.input[0] in self.vals:
            arr = np.asarray(self.vals[n.input[0]])
            if kind == "squeeze":
                arr = np.squeeze(arr, axis=tuple(axes))
            else:
                arr = np.expand_dims(arr, axis=tuple(axes))
            self.vals[n.output[0]] = arr
            self.shapes[n.output[0]] = tuple(arr.shape)
            self.const_name_of[n.output[0]] = self._store_const(arr)
            return
        self._emit(kind, n.name, out=n.output[0], axes=[int(a) for a in axes],
                   inputs=[self._index(n.input[0], n)], oshape=list(out))
        self._produced(n, out)


def _auto_pad(n):
    """A node's auto_pad, as ONNX spells it.

    "NOTSET" is the schema's default and means "the `pads` attribute says".
    The other three are the paddings ONNX computes from the input's own size,
    and they are not decoration: SAME_UPPER's output is ceil(in/stride) and its
    padding is ASYMMETRIC, with the extra row and column at the END.
    """
    for a in n.attribute:
        if a.name != "auto_pad":
            continue
        kind = _AT_KIND.get(a.type)
        if kind == "string":
            return a.s.decode("utf-8") if isinstance(a.s, bytes) else a.s
        return "NOTSET"
    return "NOTSET"


def _same_pad(in_size, eff_k, stride, upper):
    """SAME_UPPER's (begin, end) padding for one axis.

    ONNX's rule: the output is ceil(in/stride), the total padding is
    (out-1)*stride + kernel - in, and SAME_UPPER puts the larger half at the
    END while SAME_LOWER puts it at the BEGIN. The total is odd whenever the
    kernel and the stride have different parities, and then the two differ by
    one pixel -- which is the difference between a 480-wide map and a 479-wide
    one, and between a stem that joins two branches of equal width and one that
    does not.
    """
    out = (in_size + stride - 1) // stride
    total = max((out - 1) * stride + eff_k - in_size, 0)
    half = total // 2
    if upper:
        return half, total - half
    return total - half, half


def _act_params(n):
    """A HardSwish's or a HardSigmoid's (alpha, beta).

    ONNX's defaults are alpha = 1/6 and beta = 0.5, and they are the values
    the RECOGNISER's HardSigmoid carries -- but the DETECTOR's carries alpha =
    0.2, which is the SCSE block's own parameter and not the default. An
    activation that is read with the default alpha where the checkpoint says
    otherwise is a different function in the middle of the network: the
    detector's attention saturates at a different point, and every box it
    draws afterwards is a little too tight or a little too loose.
    """
    alpha, beta = 1.0 / 6.0, 0.5
    for a in n.attribute:
        if a.name == "alpha":
            alpha = float(_raw(a))
        elif a.name == "beta":
            beta = float(_raw(a))
    return alpha, beta


def _resize_attr(n, name):
    """One of Resize's string attributes, or None when the node has not it."""
    for a in n.attribute:
        if a.name != name:
            continue
        kind = _AT_KIND.get(a.type)
        if kind == "string":
            return a.s.decode("utf-8") if isinstance(a.s, bytes) else a.s
        if kind == "int":
            return a.i
        return None
    return None


def _resize_mode(n):
    """Resize's mode, as an int enum (opset 11) or a string (opset 18)."""
    for a in n.attribute:
        if a.name != "mode":
            continue
        kind = _AT_KIND.get(a.type)
        if kind == "int":
            return {0: "nearest", 1: "linear", 2: "cubic"}.get(
                a.i, f"enum {a.i}")
        if kind == "string":
            return a.s.decode("utf-8") if isinstance(a.s, bytes) else a.s
        _die(f"{n.name} carries \"mode\" as {kind}, which this reader does "
             f"not interpret.")
    return "nearest"


# -- the front end's constants -------------------------------------------------
#
# These are the numbers PaddleOCR's own pipeline uses, and they are HERE, in the
# container, rather than in the C++ -- for the reason arch=6's packer writes its
# mean and std: the front end is a property of the CONTAINER, so a different
# checkpoint's preprocessing is two numbers in a file and not a function in C++.
#
# The DETECTOR's normalisation is the ImageNet one PaddleOCR's
# `NormalizeImage` carries for detection: the image is scaled by 1/255, the
# mean is subtracted and the standard deviation divided, in RGB order.
#
# The RECOGNISER's is a different pair, because it is a different model's
# preprocessing: PaddleOCR's recognition pipeline scales by 1/255 and centres on
# 0.5 with a standard deviation of 0.5, which puts the input in [-1, 1].
#
# Both are written with the RGB/mean/std triples in the order the runtime reads
# them, and the runtime REFUSES a container whose triples are not three long:
# a two-long triple would be read as three with the last one uninitialised, and
# the normalisation would be wrong by a factor of nothing at all in the first
# two channels and by the third channel's absence in the third.
DET_MEAN = (0.485, 0.456, 0.406)
DET_STD = (0.229, 0.224, 0.225)
REC_MEAN = (0.5, 0.5, 0.5)
REC_STD = (0.5, 0.5, 0.5)

# PaddleOCR's DB post-process defaults, which are also the values its
# PP-OCRv3 and PP-OCRv5 detection configs carry:
#
#   thresh        0.3   the probability a pixel needs to be part of a region
#   box_thresh    0.6   the mean probability a region's rectangle needs
#   unclip_ratio  1.5   how far the rectangle is expanded (see the note below)
#   max_candidates 1000 how many regions are kept at most
#
# THE UNCLIP IS APPLIED, and the distance it uses is the one everybody's
# implementation uses:  distance = area * ratio / perimeter  of the rectangle
# itself. For a 100x20 text box that is 2000 * 1.5 / 240 = 12.5 pixels, so the
# box grows by about twelve pixels on every side -- enough to include the
# ascenders and the inter-character gaps that the shrunk probability map does
# not cover, and not so much that neighbouring lines merge.
#
# This is PaddleOCR's own `DBPostProcess.boxes_from_bitmap` (the `box_type:
# quad` path, which is what the v3 and v5 configs select) and it is also what
# rapidocr's port of it does. An earlier reading of this file claimed the
# un-clip was skipped because the configs say `use_polygon: true`; that flag
# belongs to a different generation of the same file, and in the generation
# these checkpoints come from the quad path expands the rectangle.
#
# The container records the ratio rather than hard-coding it, so a reader can
# see WHICH spelling this is -- and so that a container packed with a different
# ratio says so instead of looking like a different detector.
DET_THRESH = 0.3
DET_BOX_THRESH = 0.6
DET_UNCLIP_RATIO = 1.5
DET_MAX_CANDIDATES = 1000
DET_MIN_SIZE = 3

# The detector's input side. PaddleOCR's `DetResizeForTest` scales the image so
# that its LONGEST side is at most `limit_side_len`, and then rounds both sides
# up to a multiple of 32 -- which is what makes the DB head's four strides of 8
# and 2 land on an integer feature map. The container is packed for ONE size,
# because the array needs a static shape: 960x960, letterboxed.
DET_LIMIT_SIDE_LEN = 960
DET_INPUT_SIZE = 960

# The recogniser's input. PaddleOCR's `RecResizeImg` with `keep_ratio=True`
# scales the crop so that its HEIGHT is `imgH` and its width is whatever keeps
# the aspect ratio, and then pads the rest of the width with zeros. The width is
# 320 here, which is the value PaddleOCR's own configs use for the mobile
# recogniser and the value the array can be compiled for.
REC_INPUT_H = 48
REC_INPUT_W = 320

# THE RECOGNISER'S PADDING VALUE, and where the pad happens.
#
# PaddleOCR's own `resize_norm_img` -- the function its inference pipeline calls,
# and the one rapidocr ports -- does this:
#
#     resized = cv2.resize(crop, (w, 48))
#     resized = resized / 255.0
#     resized = (resized - 0.5) / 0.5
#     padded  = zeros((3, 48, 320))
#     padded[:, :, 0:w] = resized
#
# so the padding is ZERO IN THE NORMALISED SPACE -- mid-grey, 127 in 0..255 --
# and not the -1 that padding BEFORE the normalisation would give. The two differ
# by one whole unit of the recogniser's input range, over the whole right-hand
# side of every crop, and the recogniser reads that side: it is where the blank
# frames of the CTC output live, and a border of -1 instead of 0 makes those
# frames argue for a character.
#
# It is recorded in the container rather than assumed, because it is a property
# of the CHECKPOINT's preprocessing and not of the CTC decoder.
REC_PAD_VALUE = 0.0


def _load_dict(path):
    """The character table, as the list of characters the CTC decoder indexes.

    PaddleOCR's `BaseRecLabelDecode` builds its character table as

        ['blank'] + the file's lines + [' ']

    -- the blank FIRST, the file's characters in the file's order, and a space
    LAST. The blank is the CTC null class and the space is the last class, and
    both are outside the file: a decoder that read the file alone would map
    every class index one too low and turn every digit into the character
    before it.

    The file's lines are read as UTF-8 and stripped of the line ending only.
    A line that is empty is a character (the space), and a line with trailing
    spaces is a character with trailing spaces, because a character table is a
    list of strings and trimming one would change the transcription.
    """
    raw = Path(path).read_bytes()
    # A UTF-8 BOM is stripped: it is an artefact of the file, not a character,
    # and it would become the first class.
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    chars = []
    for line in lines:
        if line.endswith(b"\r"):
            line = line[:-1]
        chars.append(line.decode("utf-8"))
    if not chars:
        _die(f"{path} holds no characters. A CTC decoder needs at least the "
             f"blank and the space, and a table with nothing in it would decode "
             f"every frame to an empty string.")
    return chars


def _det_and_rec_shapes(det_onnx, rec_onnx):
    """The two input shapes, read out of the graphs themselves.

    The DETECTOR's shape is this packer's decision -- the container is packed
    for one size, because the array needs a static one -- and the RECOGNISER's
    height is the graph's own, because PaddleOCR's recogniser has a fixed
    height and a variable width, and the graph says which.
    """
    rec = onnx.load(str(rec_onnx))
    rdim = rec.graph.input[0].type.tensor_type.shape.dim
    if len(rdim) != 4:
        _die(f"{rec_onnx} takes a rank-{len(rdim)} input. The recogniser's "
             f"input is [N, 3, H, W].")
    rh = rdim[2].dim_value
    if rh <= 0:
        _die(f"{rec_onnx} has a DYNAMIC height. The recogniser's height is "
             f"fixed by the checkpoint, and a container packed for one height "
             f"is the only kind the array can be compiled for.")
    return (1, 3, DET_INPUT_SIZE, DET_INPUT_SIZE), (1, 3, rh, REC_INPUT_W)


def _conv_groups(t):
    """conv index -> group, for every convolution in the trace.

    Read out of the OP LIST rather than out of the weight's shape, because the
    weight's shape cannot tell a depthwise convolution from a dense one that
    happens to have one input channel -- and a dense one dispatched against a
    depthwise panel is a different network rather than an error.
    """
    out = {}
    for op in t.ops:
        if op["op"] == "conv":
            out[op["conv"]] = int(op.get("group", 1))
    return out


def npu_panels(t, device):
    """Pre-tiled bf16 B panels for the array backend, one per DENSE convolution.

    Returns (panels, streams, (tile_k, tile_n)) where panels maps a conv INDEX
    to (ndarray, layout, padded_k, padded_n) and streams is the sorted list of
    distinct padded (K, N) shapes -- the design set the exporter builds.

    THE PANELS ARE BUILT FROM THE OP LIST, NOT FROM THE WEIGHT TABLE, and that is
    the whole of the difference between a panel and a mistake. The op list says
    which ops are CONVOLUTIONS and which are TRANSPOSED CONVOLUTIONS, and the two
    have weights of the same rank and opposite layouts: a convolution's is
    [Cout, Cin, kh, kw] and a transposed one's is [Cin, Cout, kh, kw]. A reader
    that walked the weight table would take the transposed convolution's [Cin,
    Cout, kh, kw] for a [Cout, Cin, kh, kw], transpose it, and stage a panel of
    the right shape and the wrong numbers -- and the DB head's two upsampling
    convolutions are the only transposed ones in either graph, so the mistake
    would be in the two layers that decide where the boxes are.

    WHY THE DENSE CONVOLUTIONS ONLY
    -------------------------------
    A depthwise convolution reduces WITHIN one channel: its K is kh*kw and its
    N is one, and there is no [M, N] GEMM in it to dispatch. Pretending
    otherwise is not a slow path, it is a wrong one. So the depthwise filters
    stay on the host and are not given a panel, and the slot table says -1
    for them.

    WHY THE TRANSPOSED CONVOLUTIONS ARE NOT HERE EITHER
    -------------------------------------------------
    The DB head ends in two transposed convolutions that upsample the feature
    map by 2 and by 2 again. A transposed convolution is not a GEMM over the
    checkpoint's weight: its weight is [Cin, Cout, kh, kw] and its output is a
    SUM of SHIFTED contributions, and the array's GEMM computes one matrix
    product. There is a way to express it as a GEMM with a scattered A panel,
    and it is not this file's. So the two stay on the host and the container
    says so by not giving them a panel.

    THE TILE SIZES
    --------------
    64x32, the same as arch=6's and arch=8's: K is the convolution's
    Cin*kh*kw and is nearly always already a multiple of 64 because these
    networks' channel counts are multiples of 8 and their kernels are 3x3 (9*8
    = 72, which pads to 128 -- so K DOES pad here, and the waste is real and
    is measured rather than assumed). N pads to a multiple of tile_n * the
    device's AIE columns, which is 128 on npu1.
    """
    TILE_K, TILE_N = 64, 32
    mac_s, mac_t = mac_for_device(device, "BF16")
    N_MULT = TILE_N * cols_for_device(device)
    layout = gemm_b_layout(TILE_K, TILE_N, mac_s, mac_t)
    panels, streams = {}, set()
    for op in t.ops:
        if op["op"] != "conv":
            continue
        if int(op.get("group", 1)) != 1:
            continue    # depthwise: no GEMM to dispatch
        idx = int(op["conv"])
        w = t.consts[op["w"]]
        if w.ndim != 4:
            continue
        cout, cin, kh, kw = w.shape
        K, N = cin * kh * kw, cout
        pk = ((K + TILE_K - 1) // TILE_K) * TILE_K
        pn = ((N + N_MULT - 1) // N_MULT) * N_MULT
        # The panel is padded HERE and the padding is ZERO, which is exact: a
        # padded K column is multiplied by nothing, and a padded N column is a
        # channel the runtime never reads back.
        b = np.zeros((pk, pn), dtype=np.float32)
        # A TRANSPOSE, NOT A RESHAPE. `w` is the checkpoint's [cout, cin, kh, kw]
        # and the panel is [K, N]; `w.reshape(N, K).T` is the transpose. The flat
        # order `w.reshape(K, N)` happens to have the right SIZE and is not the
        # same matrix -- it is a permutation that no arithmetic rejects, and the
        # array would multiply a transposed filter.
        b[:K, :N] = w.reshape(N, K).T
        panels[idx] = (tile_b(b, TILE_K, TILE_N, s=mac_s, t=mac_t), layout,
                       pk, pn)
        streams.add((pk, pn))
    return panels, sorted(streams), (TILE_K, TILE_N)


# -- the packer ----------------------------------------------------------------


def _trace_graph(path, prefix, shape):
    """One ONNX graph, traced at a FIXED input size.

    The shape is the point: with it fixed, every shape-only node in the graph is
    a pack-time fact, and the runtime gets a list of ops whose output extents
    are all known.
    """
    model = onnx.load(str(path))
    return Tracer(model, prefix, shape).trace()


def _graph_summary(t):
    counts = {}
    for op in t.ops:
        counts[op["op"]] = counts.get(op["op"], 0) + 1
    return counts


def pack_ocr(det_onnx, rec_onnx, dict_path, out_path, dry_run=False,
             device=None, npu=False, dtype="f32", int4_group=None,
             npue_args=None):
    """Pack a PaddleOCR detector + recogniser + dictionary into an arch=9 .npue.

    The pair is the model. A detector finds text and cannot read it; a
    recogniser reads a crop and cannot find one. So one container carries both
    graphs, the character dictionary and both front ends' geometry -- the same
    reason arch=7 carries a palm detector beside a hand-landmark network, and
    the same reason arch=8 carries a person detector beside a pose net.
    """
    wdtype = conv_quant.conv_weight_dtype(dtype)
    # THE QUANTISED DTYPES ARE REFUSED HERE, BY NAME, AND NOT BECAUSE THEY
    # WOULD BE SLOW.
    #
    # The quantisation path stores a convolution's weight as an [N, K] VIEW --
    # N output channels, K = Cin*kh*kw -- and the scale is per output channel.
    # A TRANSPOSED convolution's weight is not in that view: ONNX lays it out as
    # [Cin, Cout, kh, kw], the INPUT channels first, which is the opposite of
    # a dense convolution's [Cout, Cin, kh, kw]. So the DB head's two
    # upsampling convolutions would be quantised as if they had 24 output
    # channels when they have one, and the scale sidecar would carry 24
    # numbers for a 1-channel weight -- which the runtime refuses to load, by
    # name, with a message about the scale count.
    #
    # Two ways out, and both are worse than refusing: quantising them against
    # the transposed view would make the container's layout differ from the
    # checkpoint's for two convolutions and every reader of it, and leaving
    # them in f32 inside an int8 container would need a per-convolution dtype
    # that the format does not have (one `conv_weight_dtype` for the whole
    # container).
    #
    # So this says no. The FLOAT container is the one this architecture ships,
    # and the reason is a layout, not a preference.
    if wdtype in ("i8", "i4"):
        _die(f"--dtype {wdtype} is not supported for an ocr pack. The "
             "quantisation path stores a convolution's weight as an [N, K] view "
             "with one scale per output channel, and the DB head's two "
             "TRANSPOSED convolutions are laid out the other way round -- "
             "[Cin, Cout, kh, kw], the input channels first -- so their scale "
             "sidecar would carry 24 numbers for a 1-channel weight. The "
             "float container is the one this architecture ships, and the "
             "reason is the layout, not the precision: --dtype f32 (the "
             "default) and --dtype bf16 both work.")
    if wdtype == "i4":
        if int4_group is not None and int(int4_group) < 0:
            _die(f"--int4-group {int4_group} is negative. 0 means one group over "
                 f"the whole input axis, i.e. per-channel int4, and is the only "
                 f"value below 1 that has a meaning.")
    elif int4_group is not None:
        _die(f"--int4-group {int4_group} was given with --dtype {wdtype}. It "
             f"only means anything for i4, and ignoring it would leave a "
             f"container whose `conv_int4_group` says 0 while the operator "
             f"asked for {int4_group}.")

    det_shape, rec_shape = _det_and_rec_shapes(det_onnx, rec_onnx)
    if det_shape[2] != DET_INPUT_SIZE or det_shape[3] != DET_INPUT_SIZE:
        _die(f"the detector's input is {det_shape[2]}x{det_shape[3]} and this "
             f"packer is written for {DET_INPUT_SIZE}x{DET_INPUT_SIZE}.")

    td = _trace_graph(det_onnx, "det", det_shape)
    tr = _trace_graph(rec_onnx, "rec", rec_shape)

    # THE DICTIONARY, and the class count it has to agree with.
    chars = _load_dict(dict_path)
    n_classes = len(chars) + 2       # 'blank' first, ' ' last
    if n_classes != tr.output_shape[-1]:
        _die(f"the dictionary holds {len(chars)} characters, so the CTC table "
             f"is {n_classes} classes with the blank and the space, and the "
             f"recogniser's head emits {tr.output_shape[-1]}. A decoder that "
             f"reads one table and a network that learned another produce text "
             f"that is confidently wrong.")

    # The BYTES, as they go into the container: the file's characters joined by
    # newlines. The blank and the space are added by the DECODER, not stored
    # here, because they are not in the file and a dictionary that carried its
    # own blank would be a second place to keep the two in step.
    dict_bytes = "\n".join(chars).encode("utf-8")

    if dry_run:
        for tag, t, shape in (("det", td, det_shape), ("rec", tr, rec_shape)):
            counts = _graph_summary(t)
            print(f"{tag}: {len(t.ops)} ops: " + ", ".join(
                f"{counts[k]} {k}" for k in sorted(counts)))
            dense = sum(1 for o in t.ops if o["op"] == "conv" and
                        int(o.get("group", 1)) == 1)
            dwc = sum(1 for o in t.ops if o["op"] == "conv" and
                      int(o.get("group", 1)) != 1)
            tc = sum(1 for o in t.ops if o["op"] == "convtranspose")
            print(f"  convs {len(t.conv_w)}: {dense} dense, {dwc} depthwise, "
                  f"{tc} transposed")
            print(f"  input {shape}  output {t.output_shape}")
        print(f"dictionary: {len(chars)} characters -> {n_classes} classes")
        print(f"thresholds: thresh {DET_THRESH} box {DET_BOX_THRESH} unclip "
              f"{DET_UNCLIP_RATIO} max_candidates {DET_MAX_CANDIDATES}")
        return

    config = {
        "arch": ARCH_STRING,
        # Its OWN kind, and not "det" or "rec": both of those are ONE network,
        # and the registry's rows and the exporter's validation key on this
        # string, so a shared one would merge two unrelated rows.
        "kind": "ocr",
        "model": "paddleocr-db-svtr-ctc",
        "det_input_size": DET_INPUT_SIZE,
        "rec_input_height": rec_shape[2],
        "rec_input_width": rec_shape[3],
        # TWO NORMALISATIONS, and they are not the same. The detector is the
        # ImageNet pair PaddleOCR's `NormalizeImage` carries for detection, in
        # RGB order; the recogniser is the (x/255 - 0.5)/0.5 pair its own
        # `resize_norm_img` uses, which puts its input in [-1, 1]. One pair
        # would be a detector fed the wrong range or a recogniser fed a range it
        # was never trained on, and both would return text.
        "det_image_mean": list(DET_MEAN),
        "det_image_std": list(DET_STD),
        "rec_image_mean": list(REC_MEAN),
        "rec_image_std": list(REC_STD),
        # WHERE THE DETECTOR'S ZERO PAD LANDS. The image is scaled so its long
        # side is at most 960, padded to 960x960 with BLACK in the uint8
        # raster, and normalised afterwards -- so the border of the normalised
        # tensor is (0 - mean)/std and not 0.0. A reader that padded in the
        # normalised tensor would put mid-grey there, which to this stem is a
        # value the detector has seen and the black border is not.
        "det_pad_value": 0,
        # The recogniser's padding is ZERO IN THE NORMALISED SPACE, which is a
        # different fact and is recorded as such: see REC_PAD_VALUE above.
        "rec_pad_value": REC_PAD_VALUE,
        "det_thresh": DET_THRESH,
        "det_box_thresh": DET_BOX_THRESH,
        "det_unclip_ratio": DET_UNCLIP_RATIO,
        "det_max_candidates": DET_MAX_CANDIDATES,
        "det_min_size": DET_MIN_SIZE,
        "det_num_convs": len(td.conv_w),
        "rec_num_convs": len(tr.conv_w),
        "rec_num_classes": n_classes,
        "rec_dict_size": len(chars),
        "det_graph": json.dumps(td.ops, separators=(",", ":")),
        "rec_graph": json.dumps(tr.ops, separators=(",", ":")),
        "det_graph_nodes": json.dumps(td.node_names, separators=(",", ":")),
        "rec_graph_nodes": json.dumps(tr.node_names, separators=(",", ":")),
        # WHAT THE CONVOLUTION WEIGHTS ARE STORED AT. The recogniser's
        # PROJECTIONS are not convolution weights -- they are rank-2 matrices
        # with no kernel and no channel axis -- so they stay fp32 whatever this
        # says, and so do the elementwise constants. Both are named here so a
        # reader knows which tensors the precision applies to.
        "conv_weight_dtype": wdtype,
        "conv_int4_group": int(int4_group) if wdtype == "i4" else 0,
        "linear_weight_dtype": "f32",
    }

    w = Writer(config, arch=ARCH_PADDLEOCR_DB_SVTR_CTC)

    # -- the weights
    #
    # The two graphs' tensors are namespaced by their tracer's prefix, so the
    # detector's conv 0 and the recogniser's conv 0 are `det.conv.0.w` and
    # `rec.conv.0.w` and cannot be confused. That is the same reason arch=8's
    # two graphs are namespaced, and the same reason it matters: the two stems
    # are of similar width, so a mix-up is a panel of the right shape and the
    # wrong weights.
    for t in (td, tr):
        for name, arr in sorted(t.consts.items()):
            if name.endswith(".w") and ".conv." in name:
                # A CONVOLUTION WEIGHT: stored at `conv_weight_dtype`, with the
                # scale sidecars that dtype implies. The `.w` is checked as well
                # as the `.conv.` because the BIAS of every convolution is named
                # `*.conv.N.b`, contains ".conv." and is rank 1 -- routed into
                # add_conv_w it dies in `_view` with a message about
                # convolution shapes that does not name the tensor that arrived.
                conv_quant.add_conv_w(w, name[:-len(".w")], arr, wdtype,
                                       int4_group)
            elif name.endswith(".w") and ".linear." in name:
                # A PROJECTION, not a convolution weight. It stays fp32 and it
                # says so: the recogniser's attention block has nine of them and
                # the head one, and they are 120x438 and smaller -- 60 k
                # parameters against the detector's 20 M -- so the bytes are not
                # worth a scheme that would have to be taught to the reader.
                w.add(name, arr, "F32", "linear", list(arr.shape))
            else:
                # A BIAS or an elementwise constant. fp32, always: a bias is
                # four bytes per output channel and an elementwise constant is
                # a handful of numbers, and quantising either would move an
                # activation by a constant the network never saw in training.
                role = "const"
                w.add(name, arr, "F32", role, list(arr.shape))

    # -- the character table
    w.add("rec_dict", np.frombuffer(dict_bytes, dtype=np.uint8), "U8", "dict",
          [len(dict_bytes)])

    if npu:
        if device is None:
            _die("--npu needs a device: pass --device npu1 (or npu2). N pads to "
                 "tile_n*cols and cols is a property of the array, so a panel "
                 "built without one is not a panel the design can read.")
        # BOTH graphs' panels, because both graphs' dense convolutions dispatch.
        # A stream set for one network and not the other is a design whose slots
        # half match, which resolves for the detector and not for the text.
        #
        # Tiled from the DEQUANTISED weights, so that on a --dtype i8 container
        # the host's `q * s` and the array's bf16(q * s) are one network rather
        # than two that a performance flag selects between.
        panels, streams, tile = {}, set(), None
        for pfx, t in (("det", td), ("rec", tr)):
            p, s, tl = npu_panels(t, device)
            if tile is None:
                tile = tl
            elif tile != tl:
                _die(f"the two networks were tiled differently ({tile} and "
                     f"{tl}), so one container would carry two panel layouts "
                     f"under one design's layout_hash")
            for idx, (panel, layout, pk, pn) in p.items():
                panels[(pfx, idx)] = (panel, layout, pk, pn)
            streams |= set(s)
        allstreams = sorted(streams)
        name_of = {k: f"conv{k[0]}x{k[1]}" for k in allstreams}
        for pfx, t in (("det", td), ("rec", tr)):
            for o in t.ops:
                if o["op"] != "conv":
                    continue
                i = int(o["conv"])
                if (pfx, i) not in panels:
                    continue    # depthwise: no panel, and no stream
                panel, layout, pk, pn = panels[(pfx, i)]
                w.add(f"{pfx}.conv.{i}.btile", to_bf16_bits(panel), "BF16",
                      "gemm_b", [pk, pn], layout=layout, padded_shape=[pk, pn])
                o["stream"] = name_of[(pk, pn)]
        # Into w.config and NOT into the local `config` dict: Writer.__init__
        # took a copy of `config` when it was constructed above, so a key set on
        # the local dict now is a dict the file never sees.
        w.config["npu_streams"] = json.dumps(
            [{"op": name_of[k], "k": k[0], "n": k[1], "tile_k": tile[0],
              "tile_n": tile[1]} for k in allstreams],
            separators=(",", ":"))
        # Re-serialised because the stream names above went into `ops` AFTER the
        # config dict was built, so the strings the writer already holds do not
        # have them.
        w.config["det_graph"] = json.dumps(td.ops, separators=(",", ":"))
        w.config["rec_graph"] = json.dumps(tr.ops, separators=(",", ":"))

    _embed(w, "paddleocr-onnx", "bf16", npue_args, "ocr")
    w.write(out_path)
    size = os.path.getsize(out_path)
    print(f"wrote {out_path}  ({size / 1e6:.1f} MB, arch={ARCH_STRING})")
    print(f"  det {len(td.ops)} ops / {len(td.conv_w)} convs, "
          f"rec {len(tr.ops)} ops / {len(tr.conv_w)} convs, "
          f"{len(chars)} characters -> {n_classes} classes")
    if wdtype != "f32":
        print(f"  conv weights {wdtype.upper()}; the recogniser's projections "
              f"and every bias stay f32")
    if npu:
        print(f"  npu        {len(panels)} pre-tiled bf16 panels over "
              f"{len(allstreams)} padded (K, N) shapes, tile {tile[0]}x{tile[1]}, "
              f"device {device}")


def _embed(w, model_name, datapath, npue_args, where):
    if npue_args is None or getattr(npue_args, "no_embed_artifacts", False):
        return None
    from tools.lib.design_embed import (GEMM_SET, embed_design_sets, find_design_dir)
    explicit = getattr(npue_args, "embed_artifacts", None)
    here = os.path.abspath(__file__)
    for _ in range(4):
        here = os.path.dirname(here)
    root = os.path.join(here, "runtime", "artifacts")
    d = find_design_dir(model_name, datapath, root, explicit)
    if d is None:
        if explicit:
            raise SystemExit(
                f"--embed-artifacts {explicit} holds no {GEMM_SET}/final.xclbin. "
                f"Point it at the directory CONTAINING the sets, not at one of "
                f"them.")
        print(f"  design     none embedded: no set under {root} for "
              f"{model_name} ({datapath}). The container still runs; it needs "
              f"the design set beside it.")
        return None
    r = embed_design_sets(w, d,
                          device=getattr(npue_args, "device", "npu1") or "npu1",
                          datapath=datapath)
    sets = ", ".join(f"{k} {v['bytes'] / 1024:.0f} KB"
                     for k, v in sorted(r["sets"].items()))
    print(f"  design     {r['total_bytes'] / 1024:.0f} KB embedded from {d} "
          f"({getattr(npue_args, 'device', 'npu1')}, {datapath}): {sets}")
    return r


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("det_onnx")
    ap.add_argument("rec_onnx")
    ap.add_argument("dict_txt")
    ap.add_argument("out")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--npu", action="store_true",
                    help="also stage the pre-tiled bf16 B panel for every DENSE "
                         "convolution of both graphs and record the array stream "
                         "each one runs on. The depthwise convolutions and the "
                         "two transposed ones stay on the host: a depthwise "
                         "filter is not a GEMM, and a transposed convolution's "
                         "output is a sum of shifted contributions that the "
                         "array's one GEMM cannot express.")
    ap.add_argument("--device", default=None,
                    help="the array the panels are tiled for (npu1, npu2). "
                         "Required with --npu.")
    ap.add_argument("--dtype", default="f32",
                    choices=["f32", "bf16", "i8", "i4"],
                    help="the precision the CONVOLUTION weights are stored at. "
                         "f32 is the default because an unstated dtype has to "
                         "mean the precision the checkpoint itself holds.")
    ap.add_argument("--int4-group", type=int, default=None,
                    help="the group size for four-bit weights. 0 means one "
                         "group over the whole input axis.")
    args = ap.parse_args()
    return pack_ocr(args.det_onnx, args.rec_onnx, args.dict_txt, args.out,
                    dry_run=args.dry_run, device=args.device, npu=args.npu,
                    dtype=args.dtype, int4_group=args.int4_group)


if __name__ == "__main__":
    main()
