//===- net.hpp -------------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=9 graph walk: twenty-four ops, two graphs.
//
// WHY THE VOCABULARY IS TWENTY-FOUR OPS AND NOT SIX
// ---------------------------------------------------
// arch=6 walks conv, concat, slice, add, maxpool, upsample and detect. arch=7
// walks conv, dwconv, add, maxpool, pad_c and resize. arch=8 walks conv,
// dwconv, add, maxpool, resize and d2s. This one walks everything the two
// PaddleOCR graphs contain, and the list is longer because the RECOGNISER is
// a different kind of network from the other three:
//
//   * the RECOGNISER's head is a LayerNorm and an attention block: a
//     ReduceMean, a Sub, a Sqrt, a Div, a MatMul and a Softmax. None of
//     those is a convolution or an elementwise pass over one, and a runtime
//     without them could not read a crop at all.
//   * the tensors are not all [C, H, W]. The CTC head's output is rank 3
//     ([1, 40, 438]) and the attention block's are rank 4, and a runtime
//     that only held four-dimensional tensors would have to fake the rest.
//   * there is a CONVTRANSPOSE -- the DB head's two upsampling convolutions
//     -- whose output is a sum of SHIFTED contributions and not a GEMM, so
//     it stays on the host whatever the array is asked to do.
//
// EVERY OUTPUT EXTENT IS CHECKED
// -------------------------------
// The packer resolved every shape at pack time and recorded each op's output
// extent in the container. The walk computes its own and compares. A
// disagreement is a bug in the packer or in here, and it is caught AT THE OP
// rather than at the end of the graph -- which is the difference between
// "the recogniser disagrees" and "the resize indexes its source row
// wrongly".
//
// THE ACTIVATIONS ARE THE MODEL'S
// -------------------------------
// relu, sigmoid, hardswish and hardsigmoid, and the last two carry their own
// alpha and beta: the DETECTOR's HardSigmoid has alpha 0.2 and the
// RECOGNISER's has 1/6. They are NOT fused into the convolution's epilogue,
// and the reason is arithmetic rather than taste: the host convolution's
// epilogue lives in common/conv_host.hpp and computes the OTHER
// architectures' activations, and this one's carry a parameter that enum
// cannot hold. So they are a separate pass over the finished tensor -- one
// read and one write of a megabyte or two, against a convolution that reads
// and writes that much a dozen times.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "common/conv_host.hpp"
#include "conv/array.hpp"
#include "ocr/geometry.hpp"
#include "runtime/pool.hpp"

namespace npue::ocr {

using Tensor = npue::ocr::Tensor;

// Which ops the array is asked to run. An EMPTY placement is the default and
// means everything is on the host, which is also what a run gets when the
// container was packed without --npu.
struct Placement {
  bool conv_on_array = false;
  std::string design_dir;
  // WHERE THIS GRAPH'S CONVOLUTIONS LAND in the array backend's panel table.
  //
  // This container packs TWO graphs with SEPARATE convolution index spaces --
  // both start at zero, both have a conv 0 -- while the backend has ONE
  // panel table. So slot numbers are handed out over the DENSE convolutions
  // of BOTH graphs in one run, and each graph's own table translates its
  // index into the right absolute slot. The table is PASSED rather than
  // computed, because `slot = base + conv` is wrong here: depthwise
  // convolutions are interleaved through both graphs, so the detector's conv
  // 8 is not slot 8.
  //
  // A -1 in the table means "this convolution has no panel", which is what a
  // depthwise convolution gets: its reduction is within one channel, so
  // there is no [M, N] GEMM in it to dispatch.
  const std::vector<int64_t> *slots = nullptr;
  std::string group = "det";
};

// What one run cost.
struct Cost {
  int64_t convs_host = 0;
  int64_t dwconvs_host = 0;
  int64_t convs_array = 0;
  int64_t dispatches = 0;
  int64_t elementwise_ops = 0;
  int64_t transposed_convs = 0;
  double t_host = 0.0;
  double t_array = 0.0;
  double t_dwconv = 0.0;
  double t_wmat = 0.0;
  double t_im2col = 0.0;
  double t_gemm = 0.0;
  double t_transpose = 0.0;
  // The two transposed convolutions' own time. They were counted in
  // t_elementwise, which is a place a reader looks for an Add or a Mul and
  // not for a convolution, and a breakdown that cannot say how long the
  // upsampling head took cannot say whether the array path is worth it.
  double t_transposed = 0.0;
  double t_array_gemm = 0.0;
  double t_array_repack = 0.0;
  double t_array_transpose = 0.0;
  double t_elementwise = 0.0;
  void reset() { *this = Cost{}; }
};

// Executes one packed graph. Owns nothing the caller has to keep alive except
// the container the geometry's ConvW pointers point into.
class Network {
public:
  // `array` is borrowed and may be null, which is the default and the only
  // thing a container packed without --npu can use.
  //
  // `convs` is THIS GRAPH's convolution table, and it is a parameter rather
  // than something read out of the Geometry because the two graphs have
  // SEPARATE index spaces -- both start at zero, both have a conv 0 -- and
  // the [K,N] weight table this builds is indexed by the same numbers the
  // graph uses.
  Network(const Geometry &g, const Placement &place, app::Pool &pool,
          const std::vector<ConvW> &convs, conv::Convs *array = nullptr);

  // One image in, every op's output out.
  //
  // Operand -1 in an op's `inputs` is the image, so the returned vector is
  // one longer than the graph and op i is at index i + 1 -- index 0 is the
  // image. Shifting by one rather than keeping a side map because an op's
  // operand list is read by array index at a dozen call sites and a special
  // case at each of them is a dozen chances to forget it.
  const std::vector<Tensor> &run(const Tensor &input,
                                 const std::vector<Layer> &graph,
                                 const std::vector<ConvW> &convs,
                                 const std::string &label);

  // Node i of the graph, whatever the operand bookkeeping did with the
  // indices.
  const Tensor &node(size_t i) const { return body_[i + 1]; }

  // Called once per op, in graph order, with the op's index and its output.
  // Nothing in the runtime sets it; it exists so the gate can say WHICH op
  // this walk and an independent reading of the ONNX graph first disagree
  // at.
  //
  // The index is -1 for the network's INPUT, which is reported before the
  // first op: a gate that compares this walk against another reading of the
  // same graph needs the same input on both sides, and the input is the one
  // thing the two sides cannot each derive and agree on -- it is a resample
  // of a photograph, and a resampler that rounds differently puts the two
  // sides a whole quantization step apart before the first convolution.
  std::function<void(int64_t, const Tensor &)> on_node;

  const Cost &cost() const { return cost_; }
  void reset_cost() { cost_.reset(); }
  bool on_array() const { return array_ != nullptr; }
  int64_t array_dispatches() const {
    return array_ ? array_->dispatches() : 0;
  }
  double array_seconds() const { return array_ ? array_->t_array() : 0.0; }
  size_t array_staged_bytes() const {
    return array_ ? array_->staged_bytes() : 0;
  }

private:
  Tensor conv(const Tensor &in, const Layer &l, const ConvW &w,
              const float *wkn, const std::string &label);
  Tensor conv_array(const Tensor &in, Tensor &out, const Layer &l,
                    const ConvW &w, int64_t M, int64_t K, int64_t N,
                    const std::string &label);
  Tensor dwconv(const Tensor &in, const Layer &l, const ConvW &w,
                const std::string &label);
  Tensor convtranspose(const Tensor &in, const Layer &l, const ConvW &w,
                       const std::string &label);
  Tensor unary(const Tensor &in, const Layer &l, const std::string &label);
  Tensor binary(const Tensor &in, const Layer &l, const Named *c,
                const std::string &label);
  Tensor concat(const Tensor &in, const Layer &l, const std::string &label);
  Tensor pool(const Tensor &in, const Layer &l, const std::string &label);
  Tensor gapool(const Tensor &in, const Layer &l, const std::string &label);
  Tensor resize(const Tensor &in, const Layer &l, const std::string &label);
  Tensor reshape(const Tensor &in, const Layer &l, const std::string &label);
  Tensor transpose(const Tensor &in, const Layer &l, const std::string &label);
  Tensor slice(const Tensor &in, const Layer &l, const std::string &label);
  Tensor squeeze(const Tensor &in, const Layer &l, bool unsq,
                 const std::string &label);
  Tensor matmul(const Tensor &in, const Layer &l, const Named *w, const Named *b,
                const std::string &label);
  Tensor softmax(const Tensor &in, const Layer &l, const std::string &label);
  Tensor reducemean(const Tensor &in, const Layer &l, const std::string &label);

  // The op's own output extent, checked against the container's.
  void check(const Tensor &out, const Layer &l, const std::string &label) const;

  const Geometry &g_;
  const Placement place_;
  app::Pool &pool_;
  Cost cost_;
  conv::Convs *array_ = nullptr;
  std::vector<float> a_;       // the im2col buffer, reused across convolutions
  std::vector<float> c_;       // the [M,N] GEMM output, reused
  std::vector<float> zeros_;   // a row of zeros, for a convolution with no bias
  std::vector<float> apad_;    // the array path's own buffers
  std::vector<float> cpad_;
  std::vector<Tensor> body_;   // every op's output; index 0 is the image
  // The [K,N] weight tables, one per DENSE convolution of the graph this
  // network runs. Storage first, then the pointers into it, so that the
  // pointers stay valid when the outer vectors are moved or resized.
  std::vector<std::vector<float>> wmat_;
  std::vector<const float *> wt_;
};

}  // namespace npue::ocr
