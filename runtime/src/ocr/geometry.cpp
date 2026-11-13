//===- geometry.cpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- read the arch=9 container's two graphs, their weights, the
// constants and the character dictionary. See geometry.hpp for what is in one
// of these and why.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "ocr/geometry.hpp"

#include "common/host_kernels.hpp"   // bf16_read

#include <algorithm>
#include <sstream>

namespace npue::ocr {

namespace {

int64_t jint(const npue::json::Value &o, const std::string &key,
             const std::string &label) {
  const npue::json::Value *const v = o.find(key);
  if (!v)
    throw std::runtime_error(label + ": no '" + key +
                             "'. The packed graph and this runtime have to "
                             "agree about the model's own fields, and a key "
                             "that is not there is a packer and a runtime from "
                             "different days.");
  return static_cast<int64_t>(v->as_number());
}

double jdouble_or(const npue::json::Value &o, const std::string &key,
                  double dflt) {
  const npue::json::Value *const v = o.find(key);
  return (v && !v->is_null()) ? v->as_number() : dflt;
}

std::string jstr(const npue::json::Value &o, const std::string &key,
                 const std::string &label) {
  const npue::json::Value *const v = o.find(key);
  if (!v)
    throw std::runtime_error(label + ": no '" + key + "'");
  return v->as_string();
}

std::string jopt_str(const npue::json::Value &o, const std::string &key,
                     const std::string &dflt) {
  const npue::json::Value *const v = o.find(key);
  return (v && !v->is_null()) ? v->as_string() : dflt;
}

void jpair(const npue::json::Value &o, const std::string &key,
           const std::string &label, int64_t *a, int64_t *b) {
  const npue::json::Value *const v = o.find(key);
  if (!v || v->as_array().size() != 2)
    throw std::runtime_error(label + ": '" + key +
                             "' is not a pair. Every stride, kernel and scale "
                             "in these graphs is per axis and both axes have to "
                             "be stated.");
  *a = static_cast<int64_t>((*v).as_array()[0].as_number());
  *b = static_cast<int64_t>((*v).as_array()[1].as_number());
}

void jpad4(const npue::json::Value &o, const std::string &label,
           int64_t *pad) {
  const npue::json::Value *const v = o.find("pad");
  if (!v || v->as_array().size() != 4)
    throw std::runtime_error(
        label + ": 'pad' is not four terms [top, left, bottom, right]. It is "
        "four and not two because the v5 detector spells two of its stem "
        "convolutions SAME_UPPER, which puts the extra row and column at the "
        "END -- a symmetric transcription of that is a 479-wide map where the "
        "checkpoint has a 480-wide one, and the stem's two branches then "
        "cannot be joined.");
  for (int i = 0; i < 4; ++i)
    pad[i] = static_cast<int64_t>((*v).as_array()[i].as_number());
}

std::vector<int64_t> jints(const npue::json::Value &o, const std::string &key,
                            const std::string &label) {
  const npue::json::Value *const v = o.find(key);
  if (!v || !v->is_array())
    throw std::runtime_error(label + ": '" + key + "' is not an array");
  std::vector<int64_t> out;
  for (const auto &e : v->as_array())
    out.push_back(static_cast<int64_t>(e.as_number()));
  return out;
}

// One convolution's weights, as read from the container. [Cout, Cin, kh, kw]
// row-major, which is what the host convolution consumes.
//
// `wdtype` is the container's conv_weight_dtype. Whatever it says, the byte
// count is checked against what that dtype implies for this shape BEFORE
// anything is read: a payload that contradicts its declared precision is not a
// weight at a different precision, it is a number of the wrong size.
//
// `w_storage` receives the widened/dequantised fp32 weight (bf16 and the two
// int dtypes) and must outlive `c`; on f32 it stays empty and `c.w` points
// straight into the mapping.
ConvW read_conv(npue::File &f, const std::string &pfx, int i,
                 const std::string &wdtype, int64_t i4_group,
                 std::vector<float> &w_storage, bool transposed,
                 const std::string &label) {
  ConvW c;
  c.index = i;
  const std::string base = pfx + ".conv." + std::to_string(i);
  const auto &wi = f.info(base + ".w");
  if (wi.logical_shape.size() != 4)
    throw std::runtime_error(label + ": " + base + ".w is not a 4-D filter");
  // THE TWO LAYOUTS, AND THEY ARE OPPOSITE.
  //
  // A CONVOLUTION's weight is [Cout, Cin, kh, kw] and a TRANSPOSED
  // CONVOLUTION's is [Cin, Cout, kh, kw] -- the INPUT channels first, which
  // is ONNX's own rule and the opposite of the other one. A reader that took
  // the first extent for the output count in both cases builds the DB head's
  // two upsampling convolutions with 24 output channels where the checkpoint
  // has one, and the probability map comes out 24 times too wide.
  //
  // So the LAYOUT IS READ FROM THE OP that calls the convolution, and the
  // two are never guessed apart.
  if (transposed) {
    c.cin = wi.logical_shape[0];
    c.cout = wi.logical_shape[1];
  } else {
    c.cout = wi.logical_shape[0];
    c.cin = wi.logical_shape[1];
  }
  c.kh = wi.logical_shape[2];
  c.kw = wi.logical_shape[3];
  const int64_t want = c.cout * c.cin * c.kh * c.kw;
  const int64_t N = c.cout, K = c.cin * c.kh * c.kw;
  if (wdtype == "f32") {
    if (static_cast<int64_t>(wi.nbytes) != want * 4)
      throw std::runtime_error(
          label + ": " + base + ".w is " + std::to_string(wi.nbytes) +
          " bytes, which is not " + std::to_string(want * 4) +
          " fp32 values for a [" + std::to_string(c.cout) + ", " +
          std::to_string(c.cin) + ", " + std::to_string(c.kh) + ", " +
          std::to_string(c.kw) + "] weight.");
    c.w = f.raw(base + ".w").as<float>();
  } else if (wdtype == "bf16") {
    if (static_cast<int64_t>(wi.nbytes) != want * 2)
      throw std::runtime_error(
          label + ": " + base + ".w is " + std::to_string(wi.nbytes) +
          " bytes, which is not " + std::to_string(want * 2) + " for " +
          std::to_string(want) + " bf16 values.");
    w_storage.resize(static_cast<size_t>(want));
    app::bf16_read(w_storage.data(), f.raw(base + ".w").data,
                   static_cast<size_t>(want));
    c.w = w_storage.data();
  } else {
    // int8: one byte per weight. int4: one byte per TWO weights, and K may be
    // ODD -- these kernels are 3x3xC -- so the payload is ceil(K/2) bytes per
    // row with a zero pad nibble at the end. ceil, not K/2: computing it the
    // other way makes the last nibble of the last row a phantom weight.
    const int64_t packed = wdtype == "i8" ? K : (K + 1) / 2;
    if (static_cast<int64_t>(wi.nbytes) != N * packed)
      throw std::runtime_error(
          label + ": " + base + ".w is " + std::to_string(wi.nbytes) +
          " bytes, which is not " + std::to_string(N * packed) + " for a [" +
          std::to_string(N) + ", " + std::to_string(K) + "] " + wdtype +
          " weight.");
    if (!f.has(base + ".wscale"))
      throw std::runtime_error(
          label + ": " + base + ".wscale is not in the container, and " +
          base + ".w says its weights are " + wdtype +
          ". An integer payload without its scale is bytes whose meaning is "
          "one multiply away and not present in the file.");
    const auto &si = f.info(base + ".wscale");
    if (static_cast<int64_t>(si.nbytes) != N * 4)
      throw std::runtime_error(
          label + ": " + base + ".wscale is " + std::to_string(si.nbytes) +
          " bytes for " + std::to_string(N) +
          " output channels. The scale is per output channel -- one per row of "
          "this weight -- and a scale count that disagrees with the row count "
          "puts every channel after the last one on another channel's factor.");
    const float *const s = f.raw(base + ".wscale").as<float>();
    w_storage.resize(static_cast<size_t>(want));
    if (wdtype == "i8") {
      const int8_t *const q = f.raw(base + ".w").as<int8_t>();
      for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
          w_storage[static_cast<size_t>(n * K + k)] =
              static_cast<float>(q[n * K + k]) * s[n];
    } else {
      const int64_t G = i4_group > 0 ? (K + i4_group - 1) / i4_group : 1;
      if (!f.has(base + ".gscale"))
        throw std::runtime_error(
            label + ": " + base + ".gscale is not in the container, and " +
            base + ".w says its weights are i4 with a group of " +
            std::to_string(i4_group) + ".");
      const auto &gi = f.info(base + ".gscale");
      if (static_cast<int64_t>(gi.nbytes) != G * N * 4)
        throw std::runtime_error(
            label + ": " + base + ".gscale is " + std::to_string(gi.nbytes) +
            " bytes, not " + std::to_string(G * N * 4) + " -- [" +
            std::to_string(G) + ", " + std::to_string(N) + "] for a group of " +
            std::to_string(i4_group ? i4_group : K) + " over K=" +
            std::to_string(K) + ".");
      const float *const gs = f.raw(base + ".gscale").as<float>();
      const uint8_t *const p = f.raw(base + ".w").as<uint8_t>();
      for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
          // Four bits, low nibble first, SIGN-EXTENDED. A nibble of 0xF is -1
          // and not 15: reading them unsigned maps the most negative weight
          // onto the most positive one, in every group.
          const uint8_t byte = p[n * ((K + 1) / 2) + k / 2];
          int v = (k & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
          if (v >= 8) v -= 16;
          const int64_t gsel = i4_group > 0 ? k / i4_group : 0;
          w_storage[static_cast<size_t>(n * K + k)] =
              static_cast<float>(v) * s[n] * gs[gsel * N + n];
        }
    }
    c.w = w_storage.data();
  }
  if (f.has(base + ".b"))
    c.b = f.raw(base + ".b").as<float>();
  return c;
}

struct GraphRead {
  std::vector<Layer> layers;
  std::vector<ConvW> convs;
  std::vector<std::string> nodes;
  std::vector<std::vector<float>> w_storage;
};

// Read one graph. `pfx` namespaces its tensors, and it is not decoration: both
// graphs number their convolutions from zero, so an unprefixed lookup reads
// the OTHER graph's weights.
GraphRead read_one_graph(npue::File &f, const std::string &pfx,
                          const std::string &graph_key,
                          const std::string &nodes_key, int64_t num_convs,
                          const std::string &wdtype, int64_t i4_group,
                          const std::string &label) {
  GraphRead g;
  const npue::json::Value ops = need_json(f, graph_key, label);
  const npue::json::Value names = need_json(f, nodes_key, label);
  if (ops.as_array().size() != names.as_array().size())
    throw std::runtime_error(
        label + ": " + graph_key + " has " +
        std::to_string(ops.as_array().size()) + " nodes and " + nodes_key +
        " has " + std::to_string(names.as_array().size()) +
        ". The second list is what a refusal quotes by name, and a refusal "
        "that quotes node 7 when node 7 does not exist is worse than none.");

  int64_t max_conv = -1;
  for (size_t n = 0; n < ops.as_array().size(); ++n) {
    const npue::json::Value &o = ops.as_array()[n];
    const std::string nl =
        label + "/" + graph_key + ": node '" +
        names.as_array()[n].as_string() + "' [" + std::to_string(n) + "]";
    Layer l;
    l.op = op_from_string(jstr(o, "op", nl), nl);
    l.act = jopt_str(o, "act", "");
    // "none" IS the absence of an activation, and the packer writes it as a
    // word rather than leaving the key out: a missing key and an empty
    // string are the same absence, and one spelling of it is easier to read
    // in a container than two.
    if (l.act == "none") l.act.clear();
    l.alpha = jdouble_or(o, "alpha", 1.0 / 6.0);
    l.beta = jdouble_or(o, "beta", 0.5);
    // The fused activation is only ever on a convolution, and it is only ever
    // one of four. An activation this runtime cannot compute is refused here,
    // where the node's name is known, rather than at the first tensor that
    // comes out of it wrong.
    if (!l.act.empty() && l.act != "relu" && l.act != "sigmoid" &&
        l.act != "hardswish" && l.act != "hardsigmoid")
      throw std::runtime_error(
          nl + ": the fused activation is '" + l.act +
          "'. This runtime computes relu, sigmoid, hardswish and hardsigmoid, "
          "and an activation it does not have is a network it cannot run.");
    if (l.op == Op::Conv || l.op == Op::ConvTranspose) {
      l.conv = static_cast<int>(jint(o, "conv", nl));
      if (l.conv < 0)
        throw std::runtime_error(nl + ": no 'conv' index");
      if (l.conv >= num_convs)
        throw std::runtime_error(nl + ": conv " + std::to_string(l.conv) +
                                 " is past the container's " +
                                 std::to_string(num_convs) +
                                 " convolutions for this graph");
      max_conv = std::max<int64_t>(max_conv, l.conv);
      jpad4(o, nl, l.pad);
      jpair(o, "stride", nl, &l.stride[0], &l.stride[1]);
      const npue::json::Value *const dil = o.find("dilation");
      if (dil && dil->is_array() && dil->as_array().size() == 2) {
        l.dilation[0] = static_cast<int64_t>(dil->as_array()[0].as_number());
        l.dilation[1] = static_cast<int64_t>(dil->as_array()[1].as_number());
      }
      const npue::json::Value *const grp = o.find("group");
      if (grp && !grp->is_null())
        l.group = static_cast<int64_t>(grp->as_number());
      const npue::json::Value *const op = o.find("outpad");
      if (op && op->is_array() && op->as_array().size() == 2) {
        l.outpad[0] = static_cast<int64_t>(op->as_array()[0].as_number());
        l.outpad[1] = static_cast<int64_t>(op->as_array()[1].as_number());
      }
      l.w = jopt_str(o, "w", "");
      l.b = jopt_str(o, "b", "");
      l.stream = jopt_str(o, "stream", "");
    } else if (l.op == Op::MaxPool || l.op == Op::AvgPool) {
      jpair(o, "kernel", nl, &l.kernel[0], &l.kernel[1]);
      jpair(o, "stride", nl, &l.stride[0], &l.stride[1]);
      jpad4(o, nl, l.pad);
    } else if (l.op == Op::Resize) {
      const npue::json::Value *const s = o.find("scale_h");
      const npue::json::Value *const t = o.find("scale_w");
      if (!s || !t)
        throw std::runtime_error(nl + ": no 'scale_h'/'scale_w'");
      l.scale_h = s->as_number();
      l.scale_w = t->as_number();
      if (l.scale_h <= 0.0 || l.scale_w <= 0.0)
        throw std::runtime_error(nl + ": resize has a non-positive scale");
    } else if (l.op == Op::Concat || l.op == Op::Softmax) {
      l.axis = jint(o, "axis", nl);
    } else if (l.op == Op::Reshape) {
      l.shape = jints(o, "shape", nl);
    } else if (l.op == Op::Transpose) {
      l.perm = jints(o, "perm", nl);
    } else if (l.op == Op::Slice) {
      l.axes = jints(o, "axes", nl);
      l.starts = jints(o, "starts", nl);
      l.ends = jints(o, "ends", nl);
      l.steps = jints(o, "steps", nl);
    } else if (l.op == Op::Squeeze || l.op == Op::Unsqueeze) {
      l.axes = jints(o, "axes", nl);
    } else if (l.op == Op::ReduceMean) {
      l.axes = jints(o, "axes", nl);
      const npue::json::Value *const kd = o.find("keepdims");
      if (kd && !kd->is_null())
        l.keepdims = static_cast<int64_t>(kd->as_number());
    } else if (l.op == Op::Add || l.op == Op::Sub || l.op == Op::Mul ||
               l.op == Op::Div) {
      l.c = jopt_str(o, "c", "");
    } else if (l.op == Op::MatMul) {
      l.w = jopt_str(o, "w", "");
      l.b = jopt_str(o, "b", "");
    }
    // The op's declared output extent, which the walk CHECKS.
    {
      const npue::json::Value *const out = o.find("oshape");
      if (!out || !out->is_array())
        throw std::runtime_error(nl + ": no 'oshape'. The packer resolved "
                                 "every extent at pack time, and a graph "
                                 "without them is a graph whose arithmetic "
                                 "this runtime would have to guess.");
      for (const auto &e : out->as_array())
        l.oshape.push_back(static_cast<int64_t>(e.as_number()));
    }
    for (const auto &e : o.at("inputs").as_array())
      l.inputs.push_back(static_cast<int>(e.as_number()));
    if (l.inputs.empty())
      throw std::runtime_error(nl + ": no 'inputs'");
    for (int in : l.inputs) {
      if (in != -1 && static_cast<size_t>(in) >= n)
        throw std::runtime_error(
            nl + ": takes operand " + std::to_string(in) +
            ", which is not an earlier op and not the image (-1). This graph "
            "is evaluated in order, so a forward reference is not a cycle the "
            "runtime could resolve -- it is a graph whose outputs it cannot "
            "compute.");
    }
    g.layers.push_back(std::move(l));
    g.nodes.push_back(names.as_array()[n].as_string());
  }

  // WHICH CONVOLUTIONS ARE TRANSPOSED, taken from the op list. The weight's
  // own shape cannot say: a transposed convolution's [Cin, Cout, kh, kw] and
  // a dense one's [Cout, Cin, kh, kw] are the same four numbers in a
  // different order, and reading them the same way is a head of the wrong
  // width.
  std::vector<bool> transposed(g.layers.size(), false);
  {
    std::vector<int> by_conv;
    by_conv.assign(static_cast<size_t>(std::max<int64_t>(num_convs, 0)), 0);
    for (const Layer &l : g.layers) {
      if (l.op != Op::Conv && l.op != Op::ConvTranspose) continue;
      if (l.conv >= 0 && static_cast<size_t>(l.conv) < by_conv.size())
        by_conv[static_cast<size_t>(l.conv)] = l.op == Op::ConvTranspose ? 1 : 0;
    }
    transposed.assign(by_conv.size(), false);
    for (size_t i = 0; i < by_conv.size(); ++i) transposed[i] = by_conv[i] != 0;
  }

  const size_t n_conv = static_cast<size_t>(std::max<int64_t>(num_convs, max_conv + 1));
  g.convs.resize(n_conv);
  g.w_storage.resize(n_conv);
  for (size_t i = 0; i < n_conv; ++i) {
    const std::string base = pfx + ".conv." + std::to_string(i);
    if (!f.has(base + ".w"))
      continue;   // a sparse index; conv_of() refuses an op that names one.
    g.convs[i] = read_conv(f, pfx, static_cast<int>(i), wdtype, i4_group,
                            g.w_storage[i],
                            i < transposed.size() && transposed[i], label);
    // The flag the slot table reads. It is set here, where the op list has
    // already said which convolution is transposed and the weight has been
    // read in the right layout for it.
    g.convs[i].transposed = i < transposed.size() && transposed[i];
  }
  // THE GROUP, TAKEN FROM THE OP AND NOT FROM THE WEIGHT.
  //
  // A depthwise convolution is group == Cin with one input channel per
  // group, and its weight reads [Cin, 1, kh, kw] -- which is the same shape
  // as a DENSE convolution with one input channel. The weight cannot tell
  // the two apart, and the difference is not a detail: a depthwise filter
  // reduces WITHIN one channel and has no [M, N] GEMM in it, so dispatching
  // it to the array is a different network rather than a slow one.
  //
  // So the group is read from the op that calls the convolution, and a
  // convolution that two ops call is refused: the array path keys panels by
  // which node calls a convolution, and two nodes sharing one would stage one
  // panel for two different layers.
  std::vector<bool> seen(g.convs.size(), false);
  for (const Layer &l : g.layers) {
    if (l.op != Op::Conv && l.op != Op::ConvTranspose) continue;
    ConvW &cw = g.convs[static_cast<size_t>(l.conv)];
    if (cw.w == nullptr)
      throw std::runtime_error(label + ": the graph names conv " +
                               std::to_string(l.conv) +
                               " and the container has no weights for it.");
    if (seen[static_cast<size_t>(l.conv)])
      throw std::runtime_error(
          label + ": conv " + std::to_string(l.conv) +
          " is called by two graph nodes. The array path keys panels by which "
          "node calls a convolution, and two nodes sharing one would stage one "
          "panel for two different layers.");
    seen[static_cast<size_t>(l.conv)] = true;
    cw.group = l.group;
  }
  return g;
}

// A named tensor the graph reads: an elementwise constant, a projection's
// weight or bias. F32 always -- a projection is not a convolution weight and
// an elementwise constant is a handful of numbers, and quantising either
// would move an activation by a constant the network never saw in training.
Named read_named(npue::File &f, const std::string &name) {
  Named n;
  const auto &i = f.info(name);
  for (int64_t s : i.logical_shape) n.shape.push_back(s);
  const int64_t count = n.numel();
  if (static_cast<int64_t>(i.nbytes) != count * 4)
    throw std::runtime_error("ocr: " + name + " is " +
                             std::to_string(i.nbytes) + " bytes, which is not " +
                             std::to_string(count * 4) + " fp32 values.");
  const float *const p = f.raw(name).as<float>();
  n.d.assign(p, p + count);
  return n;
}

}  // namespace

const ConvW &Geometry::conv_of(const std::vector<ConvW> &tab, const Layer &l,
                               const std::string &label) const {
  if (l.conv < 0 || static_cast<size_t>(l.conv) >= tab.size() ||
      tab[static_cast<size_t>(l.conv)].w == nullptr)
    throw std::runtime_error(label + ": op '" + op_name(l.op) +
                             "' names conv " + std::to_string(l.conv) +
                             ", which the container has no weights for. Both "
                             "lists are written by the same packer pass, so a "
                             "disagreement here means the container was edited "
                             "or half-written -- and a convolution with no "
                             "weights is a graph that would run on whatever was "
                             "in the buffer.");
  return tab[static_cast<size_t>(l.conv)];
}

const Named &Geometry::named(const std::string &name,
                             const std::string &label) const {
  const auto it = consts.find(name);
  if (it == consts.end())
    throw std::runtime_error(label + ": the graph names the tensor '" + name +
                             "', which is not in the container. The names come "
                             "from the same packer pass that wrote the op "
                             "list, so this is a container that was edited.");
  return it->second;
}

Geometry read_geometry(npue::File &f, const std::string &label) {
  const std::string arch = f.config_string("arch");
  if (arch != kArch)
    throw std::runtime_error(
        label + ": this is an arch=" + arch + " container and this build reads "
        "arch=" + kArch +
        ". The OCR family holds a text DETECTOR, a text RECOGNISER and the "
        "character dictionary between them, and none of those three is what "
        "another architecture's container holds.");
  Geometry g;
  g.det_input_size = need_int(f, "det_input_size", label);
  g.rec_input_height = need_int(f, "rec_input_height", label);
  g.rec_input_width = need_int(f, "rec_input_width", label);
  if (g.det_input_size <= 0 || g.rec_input_height <= 0 ||
      g.rec_input_width <= 0)
    throw std::runtime_error(label + ": the container declares a front end of " +
                             std::to_string(g.det_input_size) + "px and " +
                             std::to_string(g.rec_input_height) + "x" +
                             std::to_string(g.rec_input_width) +
                             ", and a size of zero is not a size.");

  // -- the two normalisations, and the two pad values
  {
    const auto norm = [&](const char *mkey, const char *skey,
                          std::vector<float> &mean, std::vector<float> &sd,
                          const char *who) {
      const std::vector<double> m = need_doubles(f, mkey, 3, label);
      const std::vector<double> s = need_doubles(f, skey, 3, label);
      mean.assign(m.begin(), m.end());
      sd.assign(s.begin(), s.end());
      for (double v : sd)
        if (v == 0.0)
          throw std::runtime_error(label + ": " + skey +
                                   " has a zero. Both front ends normalise by "
                                   "it, so that is a division by zero on every "
                                   "pixel of every frame. (" + who + ")");
    };
    norm("det_image_mean", "det_image_std", g.det_mean, g.det_std, "detector");
    norm("rec_image_mean", "rec_image_std", g.rec_mean, g.rec_std,
         "recogniser");
  }
  g.det_pad_value = need_double(f, "det_pad_value", label);
  g.rec_pad_value = need_double(f, "rec_pad_value", label);

  // -- the detector's post-process constants
  g.det_thresh = need_double(f, "det_thresh", label);
  g.det_box_thresh = need_double(f, "det_box_thresh", label);
  g.det_unclip_ratio = need_double(f, "det_unclip_ratio", label);
  g.det_max_candidates = need_int(f, "det_max_candidates", label);
  g.det_min_size = need_int(f, "det_min_size", label);
  if (g.det_thresh < 0.0 || g.det_thresh > 1.0 || g.det_box_thresh < 0.0 ||
      g.det_box_thresh > 1.0)
    throw std::runtime_error(label + ": the detector's thresholds are " +
                             std::to_string(g.det_thresh) + " and " +
                             std::to_string(g.det_box_thresh) +
                             ", and a threshold outside 0..1 is not one.");

  // -- the recogniser's head
  g.rec_num_classes = need_int(f, "rec_num_classes", label);
  g.rec_dict_size = need_int(f, "rec_dict_size", label);
  if (g.rec_num_classes != g.rec_dict_size + 2)
    throw std::runtime_error(
        label + ": the container declares " + std::to_string(g.rec_num_classes) +
        " classes over a dictionary of " + std::to_string(g.rec_dict_size) +
        " characters. The CTC table is the dictionary's characters plus the "
        "blank and the space, so the two must differ by exactly two: a "
        "decoder that reads one table and a network that learned another "
        "produce text that is confidently wrong.");

  // -- the character table
  //
  // The container stores the dictionary's BYTES -- the characters joined by
  // newlines -- and the blank and the space are added HERE, by the decoder,
  // because they are not in the file and a dictionary that carried its own
  // blank would be a second place to keep the two in step.
  if (!f.has("rec_dict"))
    throw std::runtime_error(label + ": rec_dict is not in the container. The "
                             "recogniser emits one logit per CLASS and the "
                             "classes are the dictionary's characters; a "
                             "recogniser without them transcribes noise.");
  {
    const auto &di = f.info("rec_dict");
    const uint8_t *const p = f.raw("rec_dict").as<uint8_t>();
    const std::string bytes(reinterpret_cast<const char *>(p),
                            static_cast<size_t>(di.nbytes));
    std::vector<std::string> lines;
    size_t start = 0;
    while (start <= bytes.size()) {
      const size_t nl = bytes.find('\n', start);
      if (nl == std::string::npos) {
        if (start < bytes.size())
          lines.push_back(bytes.substr(start));
        break;
      }
      std::string line = bytes.substr(start, nl - start);
      if (!line.empty() && line.back() == '\r') line.pop_back();
      lines.push_back(line);
      start = nl + 1;
    }
    g.dict.reserve(lines.size() + 2);
    g.dict.push_back("blank");   // the CTC null class, FIRST
    for (const std::string &l : lines) g.dict.push_back(l);
    g.dict.push_back(" ");        // the space, LAST
    if (static_cast<int64_t>(g.dict.size()) != g.rec_num_classes)
      throw std::runtime_error(
          label + ": the dictionary holds " + std::to_string(lines.size()) +
          " characters, so the CTC table is " + std::to_string(g.dict.size()) +
          " classes with the blank and the space, and the container declares " +
          std::to_string(g.rec_num_classes) + ".");
  }

  // -- the weight precision, and the int4 group
  std::string wdtype = "f32";
  try {
    wdtype = f.config_string("conv_weight_dtype");
  } catch (const std::exception &) {
    wdtype = "f32";
  }
  if (wdtype != "f32" && wdtype != "bf16" && wdtype != "i8" && wdtype != "i4")
    throw std::runtime_error(
        label + ": conv_weight_dtype is '" + wdtype +
        "', and this runtime reads f32, bf16, i8 and i4.");
  int64_t i4_group = 0;
  try {
    i4_group = f.config_int("conv_int4_group");
  } catch (const std::exception &) {
    if (wdtype == "i4")
      throw std::runtime_error(
          label + ": conv_weight_dtype is 'i4' and conv_int4_group is not in "
          "the container. The group size is what places each weight against "
          "its own group's scale.");
    i4_group = 0;
  }
  if (i4_group > 0 && wdtype != "i4")
    throw std::runtime_error(label + ": conv_int4_group is " +
                             std::to_string(i4_group) + " and "
                             "conv_weight_dtype is '" + wdtype + "'.");

  // -- the two graphs and their weights
  {
    GraphRead det = read_one_graph(f, "det", "det_graph", "det_graph_nodes",
                                   need_int(f, "det_num_convs", label),
                                   wdtype, i4_group, label + "/det");
    GraphRead rec = read_one_graph(f, "rec", "rec_graph", "rec_graph_nodes",
                                   need_int(f, "rec_num_convs", label),
                                   wdtype, i4_group, label + "/rec");
    g.det_graph = std::move(det.layers);
    g.det_nodes = std::move(det.nodes);
    g.det_convs = std::move(det.convs);
    g.rec_graph = std::move(rec.layers);
    g.rec_nodes = std::move(rec.nodes);
    g.rec_convs = std::move(rec.convs);
    g.w_storage.reserve(det.w_storage.size() + rec.w_storage.size());
    for (std::vector<float> &v : det.w_storage)
      g.w_storage.push_back(std::move(v));
    for (std::vector<float> &v : rec.w_storage)
      g.w_storage.push_back(std::move(v));
  }

  // -- the named tensors: the elementwise constants and the projections
  //
  // They are read by NAME, because the op list names them, and the names are
  // the packer's own and ALREADY carry the graph's prefix: "det.const.0" for
  // a constant and "rec.linear.0.w" for a projection. The prefix is not
  // decoration -- the detector's constant 0 and the recogniser's constant 0
  // are different numbers.
  {
    std::map<std::string, bool> want;
    for (const Layer &l : g.det_graph) {
      if (!l.c.empty()) want[l.c] = true;
      if (!l.w.empty() && l.op == Op::MatMul) want[l.w] = true;
      if (!l.b.empty() && l.op == Op::MatMul) want[l.b] = true;
    }
    for (const Layer &l : g.rec_graph) {
      if (!l.c.empty()) want[l.c] = true;
      if (!l.w.empty() && l.op == Op::MatMul) want[l.w] = true;
      if (!l.b.empty() && l.op == Op::MatMul) want[l.b] = true;
    }
    for (const auto &kv : want) {
      const std::string &name = kv.first;
      if (!f.has(name))
        throw std::runtime_error(label + ": the graph names the tensor '" +
                                 name + "', which is not in the container.");
      g.consts[name] = read_named(f, name);
    }
  }

  return g;
}

}  // namespace npue::ocr
