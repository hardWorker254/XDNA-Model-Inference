//===- geometry.hpp --------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- arch=9 geometry, read from the container.
//
// WHAT IS IN THIS CONTAINER
// -------------------------
// One .npue holds a TEXT DETECTOR, a TEXT RECOGNISER and the CHARACTER
// DICTIONARY, plus both front ends' geometry. The three are one model: a
// detector finds text and cannot read it, and a recogniser reads a crop and
// cannot find one, so a container holding only one of them answers "no text
// here" to every image.
//
//   det   PaddleOCR's DB detector. A probability map over the whole frame at
//         the input's own resolution, thresholded, turned into boxes.
//   rec   PaddleOCR's SVTR recogniser. One crop in, a CTC sequence out.
//   dict  The characters the recogniser's classes stand for. 436 lines for
//         English, 438 classes once the blank and the space are added.
//
// TWO NORMALISATIONS, AND THEY ARE NOT THE SAME
// ----------------------------------------------
// The detector is the ImageNet pair (0.485, 0.456, 0.406) / (0.229, 0.224,
// 0.225) and the recogniser is (0.5, 0.5, 0.5) / (0.5, 0.5, 0.5), which puts
// the recogniser's input in [-1, 1]. One pair would be a detector fed the
// wrong range or a recogniser fed a range it was never trained on, and both
// would return text.
//
// WHERE EACH ONE'S ZERO PAD LANDS
// --------------------------------
// The detector's image is scaled to fit 960x960 and padded with BLACK in the
// 8-BIT raster, and normalised afterwards -- so the border of the normalised
// tensor is (0 - mean) / std and not 0.0. The recogniser's crop is padded on
// the right with ZERO IN THE NORMALISED SPACE -- mid-grey, 127 in 0..255 --
// because PaddleOCR's own `resize_norm_img` normalises first and pads after,
// and a border of -1 instead of 0 makes the recogniser's blank frames argue
// for a character. The two are opposite facts and both are in the container.
//
// THE GRAPH IS A LIST, NOT A SHAPE
// -------------------------------
// Which convolutions there are, in which order, what feeds what, and where
// the activations are is the model. The container carries an explicit op list
// per graph and the runtime walks it.
//
// EVERY SHAPE IS ALREADY KNOWN
// -----------------------------
// The packer resolved every shape-only node -- Shape, the Reshape that
// consumes it, Slice's starts, Squeeze's axes, Transpose's perm -- at pack
// time, because the input size is fixed and every extent is then a constant.
// The runtime therefore walks a list of ops whose output extents it can CHECK
// rather than infer, and it checks them.
//
// FOUR PADDING TERMS, NOT TWO
// ---------------------------
// The v5 detector spells two of its stem convolutions SAME_UPPER, which is
// ASYMMETRIC padding: the extra row and column go at the end. A symmetric
// transcription of it is a 479-wide map where the checkpoint has a 480-wide
// one, and the stem's two branches then cannot be joined. The four terms are
// read, not assumed.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common/json_min.hpp"
#include "runtime/model.hpp"

namespace npue::ocr {

// The arch=9 container's `arch` string. Refused by name rather than run with
// the wrong front end, which is the whole point of carrying one.
inline constexpr const char *kArch = "paddleocr_db_svtr_ctc";

// The op vocabulary of a packed graph. The runtime walks exactly these and
// refuses anything else BY NAME: a dropped op is a network that computes
// something else and returns a plausible transcription.
enum class Op {
  Conv,
  ConvTranspose,
  Relu,
  Sigmoid,
  HardSwish,
  HardSigmoid,
  Add,
  Sub,
  Mul,
  Div,
  Sqrt,
  Concat,
  MaxPool,
  AvgPool,
  GAveragePool,
  Resize,
  Reshape,
  Transpose,
  Slice,
  Squeeze,
  Unsqueeze,
  MatMul,
  Softmax,
  ReduceMean,
};

inline const char *op_name(Op o) {
  switch (o) {
    case Op::Conv: return "conv";
    case Op::ConvTranspose: return "convtranspose";
    case Op::Relu: return "relu";
    case Op::Sigmoid: return "sigmoid";
    case Op::HardSwish: return "hardswish";
    case Op::HardSigmoid: return "hardsigmoid";
    case Op::Add: return "add";
    case Op::Sub: return "sub";
    case Op::Mul: return "mul";
    case Op::Div: return "div";
    case Op::Sqrt: return "sqrt";
    case Op::Concat: return "concat";
    case Op::MaxPool: return "maxpool";
    case Op::AvgPool: return "avgpool";
    case Op::GAveragePool: return "gapool";
    case Op::Resize: return "resize";
    case Op::Reshape: return "reshape";
    case Op::Transpose: return "transpose";
    case Op::Slice: return "slice";
    case Op::Squeeze: return "squeeze";
    case Op::Unsqueeze: return "unsqueeze";
    case Op::MatMul: return "matmul";
    case Op::Softmax: return "softmax";
    case Op::ReduceMean: return "reducemean";
  }
  return "?";
}

// A tensor of any rank: a shape and the values, row-major.
//
// NOT NCHW-ONLY, and that is the difference from arch=6's and arch=8's
// tensors. The recogniser's graph has rank-3 tensors (the CTC head's
// [1, 40, 438]) and rank-2 ones (the attention block's projections), and a
// runtime that only held [C, H, W] would have to fake the rest -- which is
// how a recogniser ends up reading a transposed matrix and transcribing
// confidently wrong characters.
struct Tensor {
  std::vector<int64_t> shape;
  std::vector<float> d;

  int64_t rank() const { return static_cast<int64_t>(shape.size()); }
  int64_t numel() const {
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    return n;
  }
  bool empty() const { return d.empty(); }
  // The NCHW view, for the ops that are NCHW. A tensor that is not rank 4 is
  // refused at the call site with the op's name, not here.
  int64_t channels() const { return shape.size() == 4 ? shape[1] : 0; }
  int64_t height() const { return shape.size() == 4 ? shape[2] : 0; }
  int64_t width() const { return shape.size() == 4 ? shape[3] : 0; }
  float *chw(int64_t c) {
    const int64_t plane = shape[2] * shape[3];
    return d.data() + c * plane;
  }
  const float *chw(int64_t c) const {
    const int64_t plane = shape[2] * shape[3];
    return d.data() + c * plane;
  }
};

// One op, as the container records it.
struct Layer {
  Op op = Op::Conv;
  // Which convolution, into Geometry::det_convs / rec_convs.
  int conv = -1;
  // The FUSED activation, for the convolutions only: "", or one of relu,
  // sigmoid, hardswish, hardsigmoid.
  std::string act;
  // The activation's OWN parameters. They are not the defaults: the
  // DETECTOR's HardSigmoid has alpha 0.2 and the RECOGNISER's has 1/6, and
  // an activation that lost its alpha is a different function in the middle
  // of the network.
  double alpha = 1.0 / 6.0;
  double beta = 0.5;
  int64_t stride[2] = {1, 1};
  int64_t pad[4] = {0, 0, 0, 0};   // top, left, bottom, right
  int64_t dilation[2] = {1, 1};
  int64_t kernel[2] = {1, 1};
  int64_t outpad[2] = {0, 0};
  int64_t group = 1;
  double scale_h = 1.0, scale_w = 1.0;   // resize
  int64_t axis = 0;                      // concat, softmax
  int64_t keepdims = 1;                  // reducemean
  std::vector<int64_t> shape;            // reshape
  std::vector<int64_t> perm;             // transpose
  std::vector<int64_t> axes;             // slice, squeeze, unsqueeze, reducemean
  std::vector<int64_t> starts, ends, steps;   // slice
  // The container tensor names: a convolution's weight and bias, a
  // projection's weight and bias, an elementwise op's constant.
  std::string w, b, c;
  // The operands: an index into the walk's output list, or -1 for the image.
  std::vector<int> inputs;
  // The op's declared output extent, which the runtime CHECKS against its own
  // arithmetic. A disagreement here is a bug in one of the two, and it is
  // caught at the op rather than at the end of the graph.
  std::vector<int64_t> oshape;
  // The array stream this convolution runs on, when the container was packed
  // with the panels.
  std::string stream;
};

// One convolution's weights, as read from the container.
//
// `group` is read from the OP LIST and not inferred from the weight's shape,
// because the weight cannot tell a depthwise convolution from a dense one
// that happens to have one input channel -- and a dense one dispatched
// against a depthwise panel is a different network rather than an error.
struct ConvW {
  int index = -1;
  int64_t cout = 0, cin = 0, kh = 0, kw = 0, group = 1;
  const float *w = nullptr;   // cout * cin * kh * kw
  const float *b = nullptr;   // cout, or null
  bool depthwise() const { return group > 1; }
  // A TRANSPOSED CONVOLUTION, and it stays on the host.
  //
  // The DB head ends in two transposed convolutions that upsample the
  // probability map, and a transposed convolution is a SUM OF SHIFTED
  // CONTRIBUTIONS -- each input pixel scatters into kh x kw output pixels --
  // rather than a matrix product. There is no [M, N] GEMM in it to dispatch,
  // exactly as there is none in a depthwise filter, and the packer stages no
  // panel for one: `npu_panels` filters on the op name, and a transposed
  // convolution's op name is not "conv".
  //
  // WITHOUT THIS FLAG the slot table handed one of them a slot, and the
  // array backend refused the whole run with "det conv 72 has no stream
  // name" -- a refusal about the SLOT rather than about the convolution,
  // which is why it read as a missing-panel problem and was not one.
  bool transposed = false;
  // Whether this convolution can go on the array at all. The two that
  // cannot are the depthwise ones and the transposed ones, and they are
  // both there for the same reason: neither is a [M, N] GEMM.
  bool arrayable() const { return !depthwise() && !transposed; }
};

// A named tensor the graph reads: an elementwise constant, a projection's
// weight or bias. Read by NAME, because the op list names them.
struct Named {
  std::vector<int64_t> shape;
  std::vector<float> d;
  int64_t numel() const {
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    return n;
  }
};

struct Geometry {
  // -- the two front ends
  int64_t det_input_size = 0;
  int64_t rec_input_height = 0;
  int64_t rec_input_width = 0;
  std::vector<float> det_mean, det_std, rec_mean, rec_std;
  // The pad values, in the space the pad happens in. The detector's is a
  // uint8 raster value (0 = black) and the recogniser's is a NORMALISED
  // value (0.0 = mid-grey), and they are not interchangeable.
  double det_pad_value = 0.0;
  double rec_pad_value = 0.0;

  // -- the detector's post-process
  double det_thresh = 0.3;
  double det_box_thresh = 0.6;
  double det_unclip_ratio = 1.5;
  int64_t det_max_candidates = 1000;
  int64_t det_min_size = 3;

  // -- the recogniser's head
  int64_t rec_num_classes = 0;
  int64_t rec_dict_size = 0;
  // The character table, in class order: 'blank' FIRST, the dictionary's
  // characters in the file's order, and ' ' LAST. The blank is the CTC null
  // class and the space is outside the file, and a decoder that read the file
  // alone would map every class index one too low and turn every digit into
  // the character before it.
  std::vector<std::string> dict;

  // -- the two graphs
  std::vector<Layer> det_graph, rec_graph;
  std::vector<ConvW> det_convs, rec_convs;
  std::vector<std::string> det_nodes, rec_nodes;
  // The named tensors the graphs read, by name.
  std::map<std::string, Named> consts;
  // Widened/dequantised weight storage, owned here for the lifetime of the
  // Geometry: ConvW::w is a `const float *`, and a bf16 or int8/int4
  // container holds bytes that do not exist as floats until someone widens
  // them, once, at load.
  std::vector<std::vector<float>> w_storage;

  const ConvW &conv_of(const std::vector<ConvW> &tab, const Layer &l,
                       const std::string &label) const;
  const Named &named(const std::string &name, const std::string &label) const;
};

// Read both graphs, their weights, the constants and the dictionary. `f` must
// outlive the returned Geometry, whose ConvW::w pointers point into it.
Geometry read_geometry(npue::File &f, const std::string &label = std::string());

// -- refusals, in the order they are cheap to check --------------------------------

inline int64_t need_int(const npue::File &f, const std::string &key,
                        const std::string &label) {
  try {
    return f.config_int(key);
  } catch (const std::exception &e) {
    throw std::runtime_error(
        label + ": " + e.what() +
        " -- not an OCR container, or one packed by an older packer (arch " +
        f.config_string("arch") + ")");
  }
}

inline double need_double(const npue::File &f, const std::string &key,
                          const std::string &label) {
  try {
    return f.config_double(key);
  } catch (const std::exception &e) {
    throw std::runtime_error(label + ": " + e.what() + " -- " + key +
                             " is not in the container");
  }
}

inline std::vector<double> need_doubles(const npue::File &f,
                                        const std::string &key, int64_t want,
                                        const std::string &label) {
  const std::string raw = f.config_string(key);
  npue::json::Value v;
  try {
    v = npue::json::parse(raw);
  } catch (const std::exception &) {
    throw std::runtime_error(label + ": " + key + " is not readable JSON: " + raw);
  }
  std::vector<double> out;
  for (const auto &e : v.as_array())
    out.push_back(static_cast<double>(e.as_number()));
  if (want > 0 && static_cast<int64_t>(out.size()) != want)
    throw std::runtime_error(label + ": " + key + " has " +
                             std::to_string(out.size()) + " entries, expected " +
                             std::to_string(want));
  return out;
}

inline npue::json::Value need_json(const npue::File &f, const std::string &key,
                                   const std::string &label) {
  const std::string raw = f.config_string(key);
  try {
    return npue::json::parse(raw);
  } catch (const std::exception &e) {
    throw std::runtime_error(label + ": " + key +
                             " is not readable JSON (" + e.what() + "): " + raw);
  }
}

inline Op op_from_string(const std::string &s, const std::string &label) {
  if (s == "conv") return Op::Conv;
  if (s == "convtranspose") return Op::ConvTranspose;
  if (s == "relu") return Op::Relu;
  if (s == "sigmoid") return Op::Sigmoid;
  if (s == "hardswish") return Op::HardSwish;
  if (s == "hardsigmoid") return Op::HardSigmoid;
  if (s == "add") return Op::Add;
  if (s == "sub") return Op::Sub;
  if (s == "mul") return Op::Mul;
  if (s == "div") return Op::Div;
  if (s == "sqrt") return Op::Sqrt;
  if (s == "concat") return Op::Concat;
  if (s == "maxpool") return Op::MaxPool;
  if (s == "avgpool") return Op::AvgPool;
  if (s == "gapool") return Op::GAveragePool;
  if (s == "resize") return Op::Resize;
  if (s == "reshape") return Op::Reshape;
  if (s == "transpose") return Op::Transpose;
  if (s == "slice") return Op::Slice;
  if (s == "squeeze") return Op::Squeeze;
  if (s == "unsqueeze") return Op::Unsqueeze;
  if (s == "matmul") return Op::MatMul;
  if (s == "softmax") return Op::Softmax;
  if (s == "reducemean") return Op::ReduceMean;
  throw std::runtime_error(
      label + ": unknown op '" + s +
      "' in the packed graph. This build walks conv, convtranspose, relu, "
      "sigmoid, hardswish, hardsigmoid, add, sub, mul, div, sqrt, concat, "
      "maxpool, avgpool, gapool, resize, reshape, transpose, slice, squeeze, "
      "unsqueeze, matmul, softmax and reducemean. An op that is not in that "
      "list is a network this runtime cannot compute, and a transcription "
      "that comes out of it would be a plausible wrong answer.");
}

}  // namespace npue::ocr
