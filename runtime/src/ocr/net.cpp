//===- net.cpp -------------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=9 graph walk. See net.hpp for the vocabulary and
// for why it is twenty-four ops wide.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "ocr/net.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "common/host_kernels.hpp"   // app::now_s

namespace npue::ocr {

using hostconv::gemm_nt;
using hostconv::im2col;
using hostconv::par_items;
using hostconv::transpose_mn_to_nchw;

namespace {

// -- shapes -------------------------------------------------------------------

int64_t numel_of(const std::vector<int64_t> &shape) {
  int64_t n = 1;
  for (int64_t s : shape) n *= s;
  return n;
}

// The stride of one axis of a row-major shape: the product of the extents
// after it.
int64_t row_stride(const std::vector<int64_t> &shape, int64_t axis) {
  int64_t acc = 1;
  for (size_t i = shape.size(); i-- > static_cast<size_t>(axis) + 1;) acc *= shape[i];
  return acc;
}

std::string shape_str(const std::vector<int64_t> &shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) s += ",";
    s += std::to_string(shape[i]);
  }
  return s + "]";
}

// numpy's broadcasting: the shapes are right-aligned and an axis of extent 1
// stretches. The output extent is the larger of each pair.
std::vector<int64_t> broadcast_shape(const std::vector<int64_t> &a,
                                     const std::vector<int64_t> &b) {
  const size_t ra = a.size(), rb = b.size();
  const size_t r = ra > rb ? ra : rb;
  std::vector<int64_t> out(r, 1);
  for (size_t i = 0; i < r; ++i) {
    const int64_t av = i < r - ra ? 1 : a[i - (r - ra)];
    const int64_t bv = i < r - rb ? 1 : b[i - (r - rb)];
    if (av != bv && av != 1 && bv != 1)
      throw std::runtime_error("cannot broadcast " + shape_str(a) + " against " +
                               shape_str(b));
    out[i] = av > bv ? av : bv;
  }
  return out;
}

// The strides of ONE operand against an output shape: 0 on an axis the
// operand does not have or whose extent is 1, and the operand's own row
// stride otherwise.
std::vector<int64_t> bstride(const std::vector<int64_t> &out,
                             const std::vector<int64_t> &in) {
  const size_t r = out.size(), ri = in.size();
  std::vector<int64_t> s(r, 0);
  int64_t acc = 1;
  // THE AXES AND THE STRIDES ADVANCE TOGETHER, from the last one backwards:
  // the last axis of the operand has stride 1, the one before it has the
  // stride of the last axis's extent, and so on.
  //
  // They are two different counters and pairing them wrongly -- taking the
  // stride of the LAST axis and giving it to the FIRST -- produces a tensor
  // of the right shape whose elements are read from the wrong offsets. The
  // read then runs off the end of the buffer, which is a crash rather than a
  // wrong number, and a crash is at least honest about it.
  for (size_t i = ri; i-- > 0;) {
    const size_t axis = r - ri + i;
    if (in[i] != 1) {
      s[axis] = acc;
      acc *= in[i];
    } else {
      s[axis] = 0;
    }
  }
  return s;
}

// One elementwise pass over two operands, broadcast against each other.
//
// THE ODOMETER, NOT A MULTI-INDEX PER ELEMENT. The strides are worked out
// once and the offsets are advanced as the output's index runs, so the inner
// loop is two adds and a compare. Recomputing the index per element would be
// a division per axis per element, and the recogniser's graph has 334 ops
// over tensors of a few hundred thousand elements -- which is a few million
// divisions that buy nothing.
template <typename F>
void broadcast_apply(const std::vector<int64_t> &oshape,
                     const std::vector<int64_t> &ashape, const float *a,
                     const std::vector<int64_t> &bshape, const float *b,
                     std::vector<float> &out, F fn) {
  const std::vector<int64_t> as = bstride(oshape, ashape);
  const std::vector<int64_t> bs = bstride(oshape, bshape);
  const size_t r = oshape.size();
  std::vector<int64_t> idx(r, 0);
  int64_t aoff = 0, boff = 0;
  const int64_t total = numel_of(oshape);
  out.resize(static_cast<size_t>(total));
  for (int64_t n = 0; n < total; ++n) {
    out[static_cast<size_t>(n)] = fn(a[aoff], b[boff]);
    for (size_t k = r; k-- > 0;) {
      ++idx[k];
      aoff += as[k];
      boff += bs[k];
      if (idx[k] < oshape[k]) break;
      aoff -= as[k] * oshape[k];
      boff -= bs[k] * oshape[k];
      idx[k] = 0;
    }
  }
}

// -- the activations ----------------------------------------------------------

// ONE DEFINITION, and it is the arithmetic rather than the name: the four
// activations are functions of one element, and the two that carry a
// parameter carry THIS one's.
float act1(const std::string &act, double alpha, double beta, float v) {
  if (act == "relu") return v > 0.f ? v : 0.f;
  if (act == "sigmoid") return static_cast<float>(1.0 / (1.0 + std::exp(-v)));
  if (act == "hardswish") {
    const double t = v * alpha + beta;
    const double c = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
    return static_cast<float>(v * c);
  }
  if (act == "hardsigmoid") {
    const double t = v * alpha + beta;
    return static_cast<float>(t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t));
  }
  return v;
}

void apply_act(Tensor &t, const Layer &l) {
  if (l.act.empty() || l.act == "none") return;
  for (float &v : t.d) v = act1(l.act, l.alpha, l.beta, v);
}

}  // namespace

// -- the constructor ----------------------------------------------------------

Network::Network(const Geometry &g, const Placement &place, app::Pool &pool,
                 const std::vector<ConvW> &convs, conv::Convs *array)
    : g_(g), place_(place), pool_(pool), array_(array) {
  // THE [K,N] WEIGHT TABLES, BUILT ONCE.
  //
  // gemm_nt reads B[k*N + n], so the filter has to be [K, N] and the
  // container stores it as [Cout, Cin, kh, kw] = [N, K]. The transpose happens
  // ONCE, at load: the filter is the same on every frame, and paying for it
  // inside the convolution would be a cost the network does not have and one
  // that grows with the number of convolutions rather than with the number of
  // images.
  //
  // The index order is the whole of this and it is the thing that is easy to
  // get wrong: a [N,K] filter passed as a [K,N] one is a shape-compatible
  // mistake -- both are K*N floats -- and produces a convolution with a
  // TRANSPOSED filter, which is not a wrong answer but a different network.
  wt_.assign(convs.size(), nullptr);
  wmat_.assign(convs.size(), {});
  for (size_t i = 0; i < convs.size(); ++i) {
    const ConvW &cw = convs[i];
    if (!cw.w || cw.depthwise()) continue;
    const int64_t K = cw.cin * cw.kh * cw.kw, N = cw.cout;
    wmat_[i].assign(static_cast<size_t>(K * N), 0.f);
    hostconv::transpose_nk_to_kn(cw.w, N, K, wmat_[i].data());
    wt_[i] = wmat_[i].data();
  }
}

// -- the walk -----------------------------------------------------------------

const std::vector<Tensor> &Network::run(const Tensor &input,
                                        const std::vector<Layer> &graph,
                                        const std::vector<ConvW> &convs,
                                        const std::string &label) {
  body_.assign(graph.size() + 1, Tensor{});
  body_[0] = input;
  // THE INPUT, at index -1, before the first op. See net.hpp: the gate
  // compares this walk against an independent reading of the same graph, and
  // the two have to be fed the same tensor.
  if (on_node) on_node(-1, body_[0]);
  for (size_t i = 0; i < graph.size(); ++i) {
    const Layer &l = graph[i];
    const std::string nl = label + " node " + std::to_string(i);
    // The operands. -1 is the image, and every other index is an EARLIER op's
    // output -- the graph is evaluated in order, which the reader checked.
    const auto in = [&](int k) -> const Tensor & {
      const int j = l.inputs[static_cast<size_t>(k)];
      // -1 IS the image, and the image is body_[0]. The other indices are
      // EARLIER ops' outputs, which the reader checked when it read the
      // graph -- and a forward reference here would be a graph whose outputs
      // cannot be computed, not a cycle the walk could resolve.
      if (j == -1) return body_[0];
      if (j < 0 || static_cast<size_t>(j) >= i)
        throw std::runtime_error(nl + ": operand " + std::to_string(j) +
                                 " is not an earlier op and not the image (-1).");
      return body_[static_cast<size_t>(j) + 1];
    };
    Tensor out;
    switch (l.op) {
      case Op::Conv: {
        const ConvW &w = g_.conv_of(convs, l, nl);
        if (w.depthwise()) {
          out = dwconv(in(0), l, w, nl);
        } else {
          out = conv(in(0), l, w, wt_[static_cast<size_t>(l.conv)], nl);
        }
        break;
      }
      case Op::ConvTranspose: {
        const ConvW &w = g_.conv_of(convs, l, nl);
        out = convtranspose(in(0), l, w, nl);
        break;
      }
      case Op::Relu:
      case Op::Sigmoid:
      case Op::HardSwish:
      case Op::HardSigmoid: {
        // A STANDALONE activation. The container records its parameters on the
        // op, and they are not the defaults -- the detector's HardSigmoid has
        // alpha 0.2 -- so they are read here rather than assumed.
        out = in(0);
        Layer a = l;
        a.act = op_name(l.op);
        apply_act(out, a);
        break;
      }
      case Op::Add:
      case Op::Sub:
      case Op::Mul:
      case Op::Div: {
        const Named *c = nullptr;
        if (!l.c.empty()) {
          c = &g_.named(l.c, nl);
          out = binary(in(0), l, c, nl);
        } else {
          // TWO ACTIVATIONS. The recogniser's attention block multiplies
          // (1,8,40,15) by (1,8,15,40), and the batch axes BROADCAST: the two
          // operands are not the same shape and are not required to be.
          const Tensor &b = in(1);
          const std::vector<int64_t> os =
              broadcast_shape(in(0).shape, b.shape);
          std::vector<float> r;
          const bool add = l.op == Op::Add, sub = l.op == Op::Sub;
          const bool mul = l.op == Op::Mul, div = l.op == Op::Div;
          broadcast_apply(os, in(0).shape, in(0).d.data(), b.shape, b.d.data(),
                          r, [&](float x, float y) {
                            if (add) return x + y;
                            if (sub) return x - y;
                            if (mul) return x * y;
                            return x / y;
                          });
          out.shape = os;
          out.d = std::move(r);
        }
        break;
      }
      case Op::Sqrt: {
        out = in(0);
        for (float &v : out.d) v = std::sqrt(v);
        break;
      }
      case Op::Concat:
        out = concat(in(0), l, nl);
        break;
      case Op::MaxPool:
      case Op::AvgPool:
        out = pool(in(0), l, nl);
        break;
      case Op::GAveragePool:
        out = gapool(in(0), l, nl);
        break;
      case Op::Resize:
        out = resize(in(0), l, nl);
        break;
      case Op::Reshape:
        out = reshape(in(0), l, nl);
        break;
      case Op::Transpose:
        out = transpose(in(0), l, nl);
        break;
      case Op::Slice:
        out = slice(in(0), l, nl);
        break;
      case Op::Squeeze:
        out = squeeze(in(0), l, false, nl);
        break;
      case Op::Unsqueeze:
        out = squeeze(in(0), l, true, nl);
        break;
      case Op::MatMul: {
        const Named *w = l.w.empty() ? nullptr : &g_.named(l.w, nl);
        const Named *b = l.b.empty() ? nullptr : &g_.named(l.b, nl);
        out = matmul(in(0), l, w, b, nl);
        break;
      }
      case Op::Softmax:
        out = softmax(in(0), l, nl);
        break;
      case Op::ReduceMean:
        out = reducemean(in(0), l, nl);
        break;
    }
    check(out, l, nl);
    body_[i + 1] = std::move(out);
    if (on_node) on_node(static_cast<int64_t>(i), body_[i + 1]);
  }
  return body_;
}

// THE CHECK, AND WHY IT IS HERE.
//
// Every op's output extent was resolved at PACK time and recorded in the
// container, and the walk computes its own. The two must agree, and where they
// do not, one of the two is wrong.
//
// A disagreement is caught AT THE OP, and that is the whole point: "the
// recogniser disagrees with the checkpoint" and "the resize indexes its
// source row wrongly" are different sentences, and only the second one is
// actionable. A walk that trusted the container's extents would produce
// plausible numbers at every step and a wrong answer at the end.
void Network::check(const Tensor &out, const Layer &l,
                    const std::string &label) const {
  if (l.oshape.empty()) return;
  if (out.shape != l.oshape)
    throw std::runtime_error(label + ": op '" + op_name(l.op) +
                             "' produced " + shape_str(out.shape) +
                             " and the container declares " +
                             shape_str(l.oshape) + ". The packer resolved every "
                             "extent at pack time and recorded it, so one of "
                             "the two is wrong -- and this is the op where "
                             "they part company.");
}

// -- dense convolution --------------------------------------------------------

Tensor Network::conv(const Tensor &in, const Layer &l, const ConvW &w,
                     const float *wkn, const std::string &label) {
  if (in.rank() != 4)
    throw std::runtime_error(label + ": conv " + std::to_string(w.index) +
                             " reads a rank-" + std::to_string(in.rank()) +
                             " tensor. The convolutions are NCHW.");
  if (l.dilation[0] != 1 || l.dilation[1] != 1)
    throw std::runtime_error(
        label + ": conv " + std::to_string(w.index) + " is dilated (" +
        std::to_string(l.dilation[0]) + ", " + std::to_string(l.dilation[1]) +
        "). The shared im2col gathers the kernel's own positions and has no "
        "dilation term; these checkpoints do not use one.");
  if (w.cin != in.shape[1])
    throw std::runtime_error(label + ": conv " + std::to_string(w.index) +
                             " takes " + std::to_string(w.cin) +
                             " channels and the tensor has " +
                             std::to_string(in.shape[1]) +
                             ". Reading the other one would index past the end "
                             "of the im2col row.");
  const int64_t H = in.shape[2], W = in.shape[3];
  const int64_t eh = (w.kh - 1) * l.dilation[0] + 1;
  const int64_t ew = (w.kw - 1) * l.dilation[1] + 1;
  const int64_t OH = (H + l.pad[0] + l.pad[2] - eh) / l.stride[0] + 1;
  const int64_t OW = (W + l.pad[1] + l.pad[3] - ew) / l.stride[1] + 1;
  if (OH <= 0 || OW <= 0)
    throw std::runtime_error(label + ": conv " + std::to_string(w.index) +
                             " produces a " + std::to_string(OH) + "x" +
                             std::to_string(OW) + " output from a " +
                             std::to_string(H) + "x" + std::to_string(W) +
                             " input.");
  if (!wkn)
    throw std::runtime_error(
        label + ": conv " + std::to_string(w.index) +
        " has no [K,N] weight table. The transpose is built in the constructor "
        "for every DENSE filter, so a missing one means this convolution is "
        "depthwise and was routed here by name -- which would compute a "
        "different network rather than fail.");

  Tensor out;
  out.shape = {1, w.cout, OH, OW};
  out.d.assign(static_cast<size_t>(w.cout) * OH * OW, 0.f);
  const double t0 = app::now_s();
  im2col(in.d.data(), w.cin, H, W, w.kh, w.kw, l.pad[0], l.pad[1], l.stride[0],
         l.stride[1], a_, pool_, l.pad[2], l.pad[3]);
  const int64_t M = OH * OW, K = w.cin * w.kh * w.kw, N = w.cout;
  cost_.t_im2col += app::now_s() - t0;

  if (array_) return conv_array(in, out, l, w, M, K, N, label);

  // THE BIAS, OR A ROW OF ZEROS IN ITS PLACE. gemm_nt adds bias[n] to every
  // output column, so a null bias is a null dereference and not a zero
  // contribution -- and the detector's stem convolutions have no bias of
  // their own, because the normalisation that follows them was folded INTO
  // them and a convolution without a bias is not rare here.
  if (!w.b) zeros_.assign(static_cast<size_t>(N), 0.f);
  const float *const bias = w.b ? w.b : zeros_.data();
  const double t1 = app::now_s();
  c_.resize(static_cast<size_t>(M) * static_cast<size_t>(N));
  gemm_nt(a_.data(), M, wkn, K, N, bias, c_.data(), pool_);
  const double t2 = app::now_s();
  cost_.t_gemm += t2 - t1;
  transpose_mn_to_nchw(c_.data(), M, N, out.d.data(), pool_, 0, 0,
                       hostconv::Act::kNone, nullptr);
  cost_.t_transpose += app::now_s() - t2;
  // The host's share of a dense convolution: the GEMM and the transpose back
  // to NCHW. It is the number the array path is measured against, so it is
  // counted here rather than left to be reconstructed from two others.
  cost_.t_host += app::now_s() - t1;
  cost_.convs_host++;
  // The activation is a SEPARATE PASS, and the reason is arithmetic: the
  // host convolution's epilogue lives in common/conv_host.hpp and computes
  // the OTHER architectures' activations, and this one's hardswish and
  // hardsigmoid carry a parameter that enum cannot hold.
  const double t3 = app::now_s();
  apply_act(out, l);
  cost_.t_elementwise += app::now_s() - t3;
  return out;
}

// The array half of one dense convolution.
//
// The slot comes from a TABLE and never from arithmetic on the convolution
// index: the two disagree because slot numbers are handed out over the DENSE
// convolutions only, and the depthwise ones are interleaved through both
// graphs. `-1` means there is no panel, and that is the depthwise case.
Tensor Network::conv_array(const Tensor &in, Tensor &out, const Layer &l,
                            const ConvW &w, int64_t M, int64_t K, int64_t N,
                            const std::string &label) {
  if (!place_.slots || w.index < 0 ||
      static_cast<size_t>(w.index) >= place_.slots->size())
    throw std::runtime_error(label + ": conv " + std::to_string(w.index) +
                             " is outside this graph's slot table, which has " +
                             std::to_string(place_.slots ? place_.slots->size() : 0) +
                             " entries");
  const int64_t slot = (*place_.slots)[static_cast<size_t>(w.index)];
  if (slot < 0)
    throw std::runtime_error(
        label + ": conv " + std::to_string(w.index) +
        " has no array panel (slot -1). Only the DENSE convolutions are "
        "staged: a depthwise filter reduces within one channel, so it has no "
        "[M, N] GEMM to dispatch.");
  const double t_before = app::now_s();
  conv::conv_array_dispatch(*array_, slot, a_.data(), M, K, N, out.d.data(),
                             out.shape[2] * out.shape[3],
                             hostconv::Act::kNone, nullptr, pool_, apad_, cpad_,
                             &cost_.t_array_repack, &cost_.t_array_gemm,
                             &cost_.t_array_transpose, &cost_.dispatches);
  cost_.convs_array++;
  cost_.t_array += app::now_s() - t_before;
  apply_act(out, l);
  return out;
}

// -- depthwise convolution ----------------------------------------------------

Tensor Network::dwconv(const Tensor &in, const Layer &l, const ConvW &w,
                        const std::string &label) {
  if (in.rank() != 4)
    throw std::runtime_error(label + ": dwconv " + std::to_string(w.index) +
                             " reads a rank-" + std::to_string(in.rank()) +
                             " tensor.");
  const int64_t H = in.shape[2], W = in.shape[3];
  const int64_t KH = w.kh, KW = w.kw;
  const int64_t pt = l.pad[0], pl = l.pad[1];
  const int64_t OH = (H + l.pad[0] + l.pad[2] - KH) / l.stride[0] + 1;
  const int64_t OW = (W + l.pad[1] + l.pad[3] - KW) / l.stride[1] + 1;
  if (w.cout != in.shape[1])
    throw std::runtime_error(
        label + ": dwconv " + std::to_string(w.index) + " writes " +
        std::to_string(w.cout) + " channels and the tensor has " +
        std::to_string(in.shape[1]) +
        ". A depthwise convolution reads channel c and writes channel c, and "
        "the two counts have to be the same number or the kernel below reads "
        "past the end of the input.");
  Tensor out;
  out.shape = {1, w.cout, OH, OW};
  out.d.assign(static_cast<size_t>(w.cout) * OH * OW, 0.f);
  const double t0 = app::now_s();
  // One filter per channel. The weight is [Cout, 1, kh, kw] and the input
  // channel of filter co IS channel co -- a depthwise convolution reduces
  // WITHIN one channel, which is why the array cannot take it: there is no
  // [M, N] GEMM in it to dispatch.
  par_items(pool_, w.cout, OH * OW, [&](int64_t co) {
    const float *const wf =
        w.w + static_cast<size_t>(co) * KH * KW;
    const float *const src = in.chw(co);
    float *const dst = out.chw(co);
    const float b = w.b ? w.b[co] : 0.f;
    for (int64_t y = 0; y < OH; ++y) {
      for (int64_t x = 0; x < OW; ++x) {
        float acc = b;
        for (int64_t i = 0; i < KH; ++i) {
          const int64_t sy = y * l.stride[0] + i - pt;
          if (sy < 0 || sy >= H) continue;
          for (int64_t j = 0; j < KW; ++j) {
            const int64_t sx = x * l.stride[1] + j - pl;
            if (sx < 0 || sx >= W) continue;
            acc += wf[i * KW + j] * src[static_cast<size_t>(sy) * W + sx];
          }
        }
        dst[static_cast<size_t>(y) * OW + x] =
            act1(l.act, l.alpha, l.beta, acc);
      }
    }
  });
  cost_.t_dwconv += app::now_s() - t0;
  cost_.dwconvs_host++;
  return out;
}

// -- transposed convolution ---------------------------------------------------

// A transposed convolution's output is a SUM OF SHIFTED CONTRIBUTIONS: every
// input pixel is scattered into the output wherever the kernel puts it, and
// the output accumulates what several of them wrote.
//
// It is NOT a GEMM over the checkpoint's weight -- the weight is [Cin, Cout,
// kh, kw], the INPUT channels first, and the output is not one matrix
// product -- which is why the DB head's two transposed convolutions stay on
// the host whatever the array is asked to do.
Tensor Network::convtranspose(const Tensor &in, const Layer &l, const ConvW &w,
                               const std::string &label) {
  if (in.rank() != 4)
    throw std::runtime_error(label + ": convtranspose reads a rank-" +
                             std::to_string(in.rank()) + " tensor.");
  const int64_t H = in.shape[2], W = in.shape[3];
  const int64_t eh = (w.kh - 1) * l.dilation[0] + 1;
  const int64_t ew = (w.kw - 1) * l.dilation[1] + 1;
  const int64_t OH = l.stride[0] * (H - 1) + l.outpad[0] + eh - l.pad[0] - l.pad[2];
  const int64_t OW = l.stride[1] * (W - 1) + l.outpad[1] + ew - l.pad[1] - l.pad[3];
  Tensor out;
  out.shape = {1, w.cout, OH, OW};
  out.d.assign(static_cast<size_t>(w.cout) * OH * OW, 0.f);
  const double t0 = app::now_s();
  // The weight is [Cin, Cout, kh, kw] -- the SECOND extent is the OUTPUT
  // channel count, which is the opposite of a convolution's and the reason a
  // reader that took the first extent for the output count would build a
  // head of the wrong width.
  for (int64_t i = 0; i < w.kh; ++i) {
    for (int64_t j = 0; j < w.kw; ++j) {
      for (int64_t iy = 0; iy < H; ++iy) {
        const int64_t oy = iy * l.stride[0] - l.pad[0] + i * l.dilation[0];
        if (oy < 0 || oy >= OH) continue;
        for (int64_t ix = 0; ix < W; ++ix) {
          const int64_t ox = ix * l.stride[1] - l.pad[1] + j * l.dilation[1];
          if (ox < 0 || ox >= OW) continue;
          for (int64_t co = 0; co < w.cout; ++co) {
            float acc = 0.f;
            for (int64_t ci = 0; ci < w.cin; ++ci) {
              acc += in.chw(ci)[static_cast<size_t>(iy) * W + ix] *
                     w.w[(ci * w.cout + co) * w.kh * w.kw + i * w.kw + j];
            }
            out.chw(co)[static_cast<size_t>(oy) * OW + ox] += acc;
          }
        }
      }
    }
  }
  if (w.b)
    for (int64_t co = 0; co < w.cout; ++co)
      for (int64_t i = 0; i < OH * OW; ++i) out.chw(co)[i] += w.b[co];
  cost_.t_transposed += app::now_s() - t0;
  cost_.transposed_convs++;
  return out;
}

// -- elementwise with a constant ----------------------------------------------

Tensor Network::binary(const Tensor &in, const Layer &l, const Named *c,
                        const std::string &label) {
  if (!c)
    throw std::runtime_error(label + ": op '" + op_name(l.op) +
                             "' has no constant and no second operand.");
  // THE CONSTANT BROADCASTS. It is usually a [1, C, 1, 1] bias -- paddle2onnx
  // expresses a convolution's bias that way -- and sometimes a scalar.
  const std::vector<int64_t> os = broadcast_shape(in.shape, c->shape);
  Tensor out;
  out.shape = os;
  std::vector<float> r;
  const bool add = l.op == Op::Add, sub = l.op == Op::Sub;
  const bool mul = l.op == Op::Mul, div = l.op == Op::Div;
  broadcast_apply(os, in.shape, in.d.data(), c->shape, c->d.data(), r,
                  [&](float x, float y) {
                    if (add) return x + y;
                    if (sub) return x - y;
                    if (mul) return x * y;
                    return x / y;
                  });
  out.d = std::move(r);
  cost_.elementwise_ops++;
  return out;
}

// -- concat -------------------------------------------------------------------

Tensor Network::concat(const Tensor &in, const Layer &l,
                        const std::string &label) {
  // The first operand carries the shape; the others are joined along the
  // axis. The packer resolved the axis at pack time, and the walk checks the
  // result against the declared extent.
  const Tensor &first = in;
  const int64_t axis = l.axis;
  const int64_t rank = first.rank();
  if (axis < 0 || axis >= rank)
    throw std::runtime_error(label + ": concat's axis is " +
                             std::to_string(axis) + " and its operand has rank " +
                             std::to_string(rank) + ".");
  // The extents of every operand are the graph's; this walk is handed one
  // operand at a time, so the others are read from the body. The caller
  // stores them under the op's own index.
  const std::vector<Tensor> &b = body_;
  std::vector<int64_t> os = first.shape;
  int64_t total = 0;
  for (int k : l.inputs) {
    const Tensor &t = b[static_cast<size_t>(k) + 1];
    if (t.rank() != rank)
      throw std::runtime_error(label + ": concat joins a rank-" +
                               std::to_string(t.rank()) + " tensor with a "
                               "rank-" + std::to_string(rank) + " one.");
    total += t.shape[static_cast<size_t>(axis)];
  }
  os[static_cast<size_t>(axis)] = total;
  Tensor out;
  out.shape = os;
  out.d.assign(static_cast<size_t>(numel_of(os)), 0.f);

  // The strides of the output's axes, and of each operand's.
  std::vector<int64_t> ostride(rank, 1);
  for (int64_t k = rank - 1; k >= 0; --k) {
    ostride[static_cast<size_t>(k)] =
        (k == rank - 1) ? 1 : ostride[static_cast<size_t>(k) + 1] *
                              os[static_cast<size_t>(k) + 1];
  }
  int64_t offset = 0;
  for (int k : l.inputs) {
    const Tensor &t = b[static_cast<size_t>(k) + 1];
    // The operand's own strides.
    std::vector<int64_t> istride(rank, 1);
    for (int64_t a = rank - 1; a >= 0; --a) {
      istride[static_cast<size_t>(a)] =
          (a == rank - 1) ? 1 : istride[static_cast<size_t>(a) + 1] *
                                  t.shape[static_cast<size_t>(a) + 1];
    }
    std::vector<int64_t> idx(rank, 0);
    const int64_t total_elems = numel_of(t.shape);
    for (int64_t n = 0; n < total_elems; ++n) {
      int64_t o = offset;
      for (int64_t a = 0; a < rank; ++a)
        o += idx[static_cast<size_t>(a)] * ostride[static_cast<size_t>(a)];
      out.d[static_cast<size_t>(o)] = t.d[static_cast<size_t>(n)];
      for (int64_t a = rank; a-- > 0;) {
        ++idx[static_cast<size_t>(a)];
        if (idx[static_cast<size_t>(a)] < t.shape[static_cast<size_t>(a)]) break;
        idx[static_cast<size_t>(a)] = 0;
      }
    }
    offset += t.shape[static_cast<size_t>(axis)] *
              ostride[static_cast<size_t>(axis)];
  }
  cost_.elementwise_ops++;
  return out;
}

// -- pooling ------------------------------------------------------------------

Tensor Network::pool(const Tensor &in, const Layer &l,
                     const std::string &label) {
  if (in.rank() != 4)
    throw std::runtime_error(label + ": pooling reads a rank-" +
                             std::to_string(in.rank()) + " tensor.");
  const int64_t H = in.shape[2], W = in.shape[3];
  const int64_t KH = l.kernel[0], KW = l.kernel[1];
  const int64_t pt = l.pad[0], pl = l.pad[1];
  const int64_t OH = (H + l.pad[0] + l.pad[2] - KH) / l.stride[0] + 1;
  const int64_t OW = (W + l.pad[1] + l.pad[3] - KW) / l.stride[1] + 1;
  Tensor out;
  out.shape = {1, in.shape[1], OH, OW};
  out.d.assign(static_cast<size_t>(in.shape[1]) * OH * OW, 0.f);
  const bool is_max = l.op == Op::MaxPool;
  const double t0 = app::now_s();
  par_items(pool_, in.shape[1], OH * OW, [&](int64_t c) {
    const float *const src = in.chw(c);
    float *const dst = out.chw(c);
    for (int64_t y = 0; y < OH; ++y) {
      for (int64_t x = 0; x < OW; ++x) {
        float best = -std::numeric_limits<float>::infinity();
        float sum = 0.f;
        int64_t n = 0;
        for (int64_t i = 0; i < KH; ++i) {
          const int64_t sy = y * l.stride[0] + i - pt;
          if (sy < 0 || sy >= H) continue;
          for (int64_t j = 0; j < KW; ++j) {
            const int64_t sx = x * l.stride[1] + j - pl;
            if (sx < 0 || sx >= W) continue;
            const float v = src[static_cast<size_t>(sy) * W + sx];
            if (is_max) best = std::max(best, v);
            sum += v;
            ++n;
          }
        }
        // The AVERAGE divides by the elements that were IN the window, not by
        // the window's size: ONNX's count_include_pad is 0, which is what
        // PaddleOCR's own average pool carries, and these graphs' pools have
        // no padding at all -- so the two agree here and the distinction is
        // recorded rather than assumed.
        dst[static_cast<size_t>(y) * OW + x] =
            is_max ? best : (n > 0 ? sum / static_cast<float>(n) : 0.f);
      }
    }
  });
  cost_.t_elementwise += app::now_s() - t0;
  cost_.elementwise_ops++;
  return out;
}

Tensor Network::gapool(const Tensor &in, const Layer &l,
                       const std::string &label) {
  if (in.rank() != 4)
    throw std::runtime_error(label + ": the global average pool reads a rank-" +
                             std::to_string(in.rank()) + " tensor.");
  const int64_t C = in.shape[1], H = in.shape[2], W = in.shape[3];
  Tensor out;
  out.shape = {1, C, 1, 1};
  out.d.assign(static_cast<size_t>(C), 0.f);
  const double t0 = app::now_s();
  par_items(pool_, C, 1, [&](int64_t c) {
    const float *const src = in.chw(c);
    double acc = 0.0;
    for (int64_t i = 0; i < H * W; ++i) acc += src[i];
    out.d[static_cast<size_t>(c)] =
        static_cast<float>(acc / static_cast<double>(H * W));
  });
  cost_.t_elementwise += app::now_s() - t0;
  cost_.elementwise_ops++;
  return out;
}

// -- resize -------------------------------------------------------------------

// ONNX Resize, nearest, "asymmetric" + "floor".
//
// THE SOURCE INDEX IS floor(dst * in / out) -- NOT dst * scale and NOT the
// half-pixel grid. The container records the SCALE and the walk derives the
// output extent from the input and the scale, which is the same number the
// packer used, and the source index comes out of the two.
//
// The difference between the three spellings is a pixel of shift in the
// upsampled probability map, and one pixel is a quarter of a character at the
// detector's output stride.
Tensor Network::resize(const Tensor &in, const Layer &l,
                       const std::string &label) {
  if (in.rank() != 4)
    throw std::runtime_error(label + ": resize reads a rank-" +
                             std::to_string(in.rank()) + " tensor.");
  const int64_t C = in.shape[1], H = in.shape[2], W = in.shape[3];
  const int64_t OH = static_cast<int64_t>(std::floor(H * l.scale_h));
  const int64_t OW = static_cast<int64_t>(std::floor(W * l.scale_w));
  Tensor out;
  out.shape = {1, C, OH, OW};
  out.d.assign(static_cast<size_t>(C) * OH * OW, 0.f);
  std::vector<int64_t> ys(static_cast<size_t>(OH)), xs(static_cast<size_t>(OW));
  for (int64_t d = 0; d < OH; ++d)
    ys[static_cast<size_t>(d)] =
        OH > 0 ? std::min<int64_t>((d * H) / OH, H - 1) : 0;
  for (int64_t d = 0; d < OW; ++d)
    xs[static_cast<size_t>(d)] =
        OW > 0 ? std::min<int64_t>((d * W) / OW, W - 1) : 0;
  const double t0 = app::now_s();
  par_items(pool_, C, OH * OW, [&](int64_t c) {
    const float *const src = in.chw(c);
    float *const dst = out.chw(c);
    for (int64_t y = 0; y < OH; ++y) {
      const float *const srow = src + static_cast<size_t>(ys[static_cast<size_t>(y)]) * W;
      float *const drow = dst + static_cast<size_t>(y) * OW;
      for (int64_t x = 0; x < OW; ++x)
        drow[x] = srow[xs[static_cast<size_t>(x)]];
    }
  });
  cost_.t_elementwise += app::now_s() - t0;
  cost_.elementwise_ops++;
  return out;
}

// -- shape ops ----------------------------------------------------------------

Tensor Network::reshape(const Tensor &in, const Layer &l,
                        const std::string &label) {
  Tensor out;
  out.shape = l.shape;
  const int64_t n = numel_of(l.shape);
  if (n != static_cast<int64_t>(in.d.size()))
    throw std::runtime_error(label + ": reshape to " + shape_str(l.shape) +
                             " holds " + std::to_string(n) +
                             " values and the input holds " +
                             std::to_string(in.d.size()) +
                             ". A reshape preserves the element count.");
  out.d = in.d;
  return out;
}

Tensor Network::transpose(const Tensor &in, const Layer &l,
                          const std::string &label) {
  const int64_t rank = in.rank();
  std::vector<int64_t> perm = l.perm;
  if (static_cast<int64_t>(perm.size()) != rank)
    throw std::runtime_error(label + ": transpose's perm has " +
                             std::to_string(perm.size()) + " entries and the "
                             "tensor has rank " + std::to_string(rank) + ".");
  std::vector<int64_t> os(rank, 1);
  for (int64_t i = 0; i < rank; ++i)
    os[static_cast<size_t>(i)] = in.shape[static_cast<size_t>(perm[static_cast<size_t>(i)])];
  Tensor out;
  out.shape = os;
  out.d.assign(static_cast<size_t>(numel_of(os)), 0.f);
  // The output's strides, and the input's strides under the permutation.
  std::vector<int64_t> ostride(rank, 1), istride(rank, 1);
  for (int64_t i = rank - 1; i >= 0; --i) {
    ostride[static_cast<size_t>(i)] =
        i == rank - 1 ? 1 : ostride[static_cast<size_t>(i) + 1] *
                              os[static_cast<size_t>(i) + 1];
    istride[static_cast<size_t>(i)] =
        i == rank - 1 ? 1 : istride[static_cast<size_t>(i) + 1] *
                              in.shape[static_cast<size_t>(i) + 1];
  }
  std::vector<int64_t> idx(rank, 0);
  const int64_t total = numel_of(os);
  for (int64_t n = 0; n < total; ++n) {
    int64_t src = 0;
    for (int64_t i = 0; i < rank; ++i)
      src += idx[static_cast<size_t>(i)] *
             istride[static_cast<size_t>(perm[static_cast<size_t>(i)])];
    out.d[static_cast<size_t>(n)] = in.d[static_cast<size_t>(src)];
    for (int64_t i = rank; i-- > 0;) {
      ++idx[static_cast<size_t>(i)];
      if (idx[static_cast<size_t>(i)] < os[static_cast<size_t>(i)]) break;
      idx[static_cast<size_t>(i)] = 0;
    }
  }
  cost_.elementwise_ops++;
  return out;
}

Tensor Network::slice(const Tensor &in, const Layer &l,
                      const std::string &label) {
  const int64_t rank = in.rank();
  // The starts, ends and steps are PER AXIS, and the axes they apply to are
  // named -- ONNX's Slice takes one start, end and step per axis and says
  // which axis each is for.
  std::vector<int64_t> begin(rank, 0), stop(in.shape), step(rank, 1);
  for (size_t k = 0; k < l.axes.size(); ++k) {
    const int64_t a = l.axes[k];
    if (a < 0 || a >= rank)
      throw std::runtime_error(label + ": slice names axis " +
                               std::to_string(a) + " and the tensor has rank " +
                               std::to_string(rank) + ".");
    const int64_t dim = in.shape[static_cast<size_t>(a)];
    int64_t s = l.starts[k], e = l.ends[k];
    const int64_t sp = l.steps.empty() ? 1 : l.steps[k];
    // numpy's rules: a negative start counts from the end, a negative end
    // counts from the end, and both are clipped into the axis.
    if (s < 0) s += dim;
    if (e < 0) e += dim;
    s = std::min<int64_t>(std::max<int64_t>(s, 0), dim);
    e = std::min<int64_t>(std::max<int64_t>(e, 0), dim);
    if (sp > 0) {
      if (e < s) e = s;
    } else {
      s = std::min<int64_t>(std::max<int64_t>(s, -1), dim - 1);
      e = std::min<int64_t>(std::max<int64_t>(e, -1), dim - 1);
      if (e > s) e = s;
    }
    begin[static_cast<size_t>(a)] = s;
    stop[static_cast<size_t>(a)] = e;
    step[static_cast<size_t>(a)] = sp;
  }
  std::vector<int64_t> os(rank, 1);
  for (int64_t a = 0; a < rank; ++a) {
    const int64_t dim = in.shape[static_cast<size_t>(a)];
    const int64_t n = step[static_cast<size_t>(a)] > 0
                          ? (stop[static_cast<size_t>(a)] - begin[static_cast<size_t>(a)] +
                             step[static_cast<size_t>(a)] - 1) /
                                step[static_cast<size_t>(a)]
                          : (begin[static_cast<size_t>(a)] - stop[static_cast<size_t>(a)] +
                                 (-step[static_cast<size_t>(a)]) - 1) /
                                (-step[static_cast<size_t>(a)]);
    os[static_cast<size_t>(a)] = std::max<int64_t>(n, 0);
  }
  Tensor out;
  out.shape = os;
  out.d.assign(static_cast<size_t>(numel_of(os)), 0.f);
  std::vector<int64_t> idx(rank, 0);
  const int64_t total = numel_of(os);
  for (int64_t n = 0; n < total; ++n) {
    int64_t src = 0;
    for (int64_t a = 0; a < rank; ++a)
      src += (begin[static_cast<size_t>(a)] +
              idx[static_cast<size_t>(a)] * step[static_cast<size_t>(a)]) *
             row_stride(in.shape, a);
    out.d[static_cast<size_t>(n)] = in.d[static_cast<size_t>(src)];
    for (int64_t a = rank; a-- > 0;) {
      ++idx[static_cast<size_t>(a)];
      if (idx[static_cast<size_t>(a)] < os[static_cast<size_t>(a)]) break;
      idx[static_cast<size_t>(a)] = 0;
    }
  }
  cost_.elementwise_ops++;
  return out;
}

Tensor Network::squeeze(const Tensor &in, const Layer &l, bool unsq,
                        const std::string &label) {
  Tensor out;
  if (unsq) {
    // AN UNSQUEEZE INSERTS a 1 at each of the axes it names, and the axes
    // are positions in the OUTPUT's rank -- not in the input's. The
    // recogniser's attention block unsqueezes a scalar it squeezed out of a
    // Shape back into a rank-3 tensor, and reading the axes as input
    // positions would put the 1 in the wrong place: a tensor of the right
    // shape holding the right values, read one axis off.
    const int64_t rank = in.rank() + static_cast<int64_t>(l.axes.size());
    std::vector<int64_t> sorted = l.axes;
    std::sort(sorted.begin(), sorted.end());
    std::vector<int64_t> os;
    size_t next = 0;
    for (int64_t a = 0; a < rank; ++a) {
      if (std::binary_search(sorted.begin(), sorted.end(), a)) {
        os.push_back(1);
      } else {
        os.push_back(in.shape[next++]);
      }
    }
    if (next != in.shape.size())
      throw std::runtime_error(label + ": unsqueeze names " +
                               std::to_string(l.axes.size()) +
                               " axes and the tensor has rank " +
                               std::to_string(in.rank()) + ".");
    out.shape = os;
    out.d = in.d;
    return out;
  }
  // A SQUEEZE DROPS THE AXES IT NAMES, and the others keep their order. The
  // recogniser's attention block squeezes an extent it sliced out of a Shape
  // -- a rank-1 vector into a scalar -- and puts it back together with a
  // literal.
  std::vector<int64_t> os;
  for (int64_t a = 0; a < in.rank(); ++a) {
    const bool drop =
        std::find(l.axes.begin(), l.axes.end(), a) != l.axes.end();
    if (!drop) os.push_back(in.shape[static_cast<size_t>(a)]);
  }
  out.shape = os;
  out.d = in.d;
  cost_.elementwise_ops++;
  return out;
}

// -- matmul -------------------------------------------------------------------

// ONNX's MatMul, in ONNX's general sense.
//
// Three spellings occur in these graphs:
//
//   x @ W       a projection onto a matrix from the container. W is rank 2.
//   A @ B       two activations, batched. The recogniser's attention block
//               multiplies (1,8,40,15) by (1,8,15,40) and gets (1,8,40,40) --
//               the leading axes are BATCH axes and they broadcast.
//   W @ x       a rank-2 matrix times a rank-1 activation.
Tensor Network::matmul(const Tensor &in, const Layer &l, const Named *w,
                       const Named *b, const std::string &label) {
  Tensor a = in;
  const Named *rhs = w;
  std::vector<int64_t> ashape = a.shape;
  std::vector<int64_t> bshape;
  std::vector<float> bdata;
  if (w) {
    bshape = w->shape;
    bdata = w->d;
    if (a.rank() == 1) ashape.insert(ashape.begin(), 1);
  } else {
    // A @ B, both activations. The second operand is the graph's other
    // operand, and the walk is handed one operand at a time -- so it is read
    // back from the body.
    const Tensor &rhs_t = body_[static_cast<size_t>(l.inputs[1]) + 1];
    bshape = rhs_t.shape;
    bdata = rhs_t.d;
    if (a.rank() == 1) ashape.insert(ashape.begin(), 1);
    if (bshape.size() == 1) bshape.push_back(1);
  }
  if (bshape.size() == 1) bshape.push_back(1);
  const int64_t M = ashape[ashape.size() - 2];
  const int64_t K = ashape[ashape.size() - 1];
  const int64_t N = bshape[bshape.size() - 1];
  if (bshape[bshape.size() - 2] != K)
    throw std::runtime_error(label + ": matmul's operands are " +
                             shape_str(ashape) + " and " + shape_str(bshape) +
                             ", and the inner extents do not agree.");
  // The batch axes, broadcast.
  const size_t ab = ashape.size() - 2, bb = bshape.size() - 2;
  const size_t nb = ab > bb ? ab : bb;
  std::vector<int64_t> batch(nb, 1);
  for (size_t i = 0; i < nb; ++i) {
    const int64_t av = i < nb - ab ? 1 : ashape[i - (nb - ab)];
    const int64_t bv = i < nb - bb ? 1 : bshape[i - (nb - bb)];
    if (av != bv && av != 1 && bv != 1)
      throw std::runtime_error(label + ": matmul's batch axes do not "
                               "broadcast: " + shape_str(ashape) + " and " +
                               shape_str(bshape) + ".");
    batch[i] = av > bv ? av : bv;
  }
  std::vector<int64_t> os = batch;
  os.push_back(M);
  os.push_back(N);
  Tensor out;
  out.shape = os;
  out.d.assign(static_cast<size_t>(numel_of(os)), 0.f);
  // The strides of the operands' batch axes, in units of ONE BATCH ELEMENT
  // (M*K for the left operand and K*N for the right one, which is what the
  // offsets below multiply).
  //
  // The axes and the strides advance TOGETHER, from the last one backwards:
  // the last batch axis has stride 1 and each earlier one is multiplied by
  // the extents of the ones after it. Pairing them the other way round --
  // taking the stride of the last axis and giving it to the first -- leaves
  // every batch reading the FIRST batch's slice of the operand, and the
  // attention's eight heads then all read the same keys.
  //
  // An operand's batch axes are the RIGHTMOST ones: that is what the
  // broadcasting above aligned them to.
  auto batch_stride = [](const std::vector<int64_t> &shape, size_t nb) {
    const size_t nbatch = shape.size() - 2;
    std::vector<int64_t> s(nb, 0);
    int64_t acc = 1;
    for (size_t k = 0; k < nbatch; ++k) {
      const size_t i = nbatch - 1 - k;              // the axis, from the last
      const size_t out_axis = nb - nbatch + i;      // where its stride goes
      if (shape[i] != 1) {
        s[out_axis] = acc;
        acc *= shape[i];
      }
    }
    return s;
  };
  const std::vector<int64_t> astr = batch_stride(ashape, nb);
  const std::vector<int64_t> bstr = batch_stride(bshape, nb);
  int64_t nbatches = 1;
  for (int64_t v : batch) nbatches *= v;
  const double t0 = app::now_s();
  // One pass over the batch, and one GEMM per batch element.
  std::vector<int64_t> bidx(nb, 0);
  for (int64_t bi = 0; bi < nbatches; ++bi) {
    int64_t aoff = 0, boff = 0;
    for (size_t i = 0; i < nb; ++i) {
      aoff += bidx[i] * astr[i];
      boff += bidx[i] * bstr[i];
    }
    const float *const A = a.d.data() + aoff * M * K;
    const float *const B = bdata.data() + boff * K * N;
    float *const C = out.d.data() + bi * M * N;
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float acc = 0.f;
        for (int64_t k = 0; k < K; ++k)
          acc += A[m * K + k] * B[k * N + n];
        C[m * N + n] = acc;
      }
    }
    for (size_t i = nb; i-- > 0;) {
      ++bidx[i];
      if (bidx[i] < batch[i]) break;
      bidx[i] = 0;
    }
  }
  cost_.t_gemm += app::now_s() - t0;
  // The bias, if the graph adds one: it was FOLDED into the projection by the
  // packer, so it is added here rather than being a second op.
  if (b) {
    const int64_t count = b->numel();
    for (int64_t n = 0; n < count; ++n) {
      const float bias = b->d[static_cast<size_t>(n)];
      for (int64_t bi = 0; bi < nbatches; ++bi)
        for (int64_t m = 0; m < M; ++m)
          out.d[static_cast<size_t>((bi * M + m) * N + n)] += bias;
    }
  }
  cost_.elementwise_ops++;
  return out;
}

// -- softmax and the reduction ------------------------------------------------

Tensor Network::softmax(const Tensor &in, const Layer &l,
                        const std::string &label) {
  Tensor out = in;
  const int64_t rank = in.rank();
  int64_t axis = l.axis;
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank)
    throw std::runtime_error(label + ": softmax's axis is " +
                             std::to_string(l.axis) + " and the tensor has "
                             "rank " + std::to_string(rank) + ".");
  // The axis's extent, and the strides of the other axes.
  const int64_t dim = in.shape[static_cast<size_t>(axis)];
  int64_t outer = 1, inner = 1;
  for (int64_t i = 0; i < axis; ++i) outer *= in.shape[static_cast<size_t>(i)];
  for (int64_t i = axis + 1; i < rank; ++i) inner *= in.shape[static_cast<size_t>(i)];
  const double t0 = app::now_s();
  par_items(pool_, outer, inner, [&](int64_t o) {
    for (int64_t i = 0; i < inner; ++i) {
      float *const row = out.d.data() + (o * dim + 0) * inner + i;
      float m = -std::numeric_limits<float>::infinity();
      for (int64_t k = 0; k < dim; ++k)
        m = std::max(m, row[k * inner]);
      float sum = 0.f;
      for (int64_t k = 0; k < dim; ++k) {
        const float e = std::exp(row[k * inner] - m);
        row[k * inner] = e;
        sum += e;
      }
      for (int64_t k = 0; k < dim; ++k) row[k * inner] /= sum;
    }
  });
  cost_.t_elementwise += app::now_s() - t0;
  cost_.elementwise_ops++;
  return out;
}

Tensor Network::reducemean(const Tensor &in, const Layer &l,
                           const std::string &label) {
  const int64_t rank = in.rank();
  std::vector<bool> reduced(static_cast<size_t>(rank), false);
  for (int64_t a : l.axes) {
    const int64_t ax = a < 0 ? a + rank : a;
    if (ax < 0 || ax >= rank)
      throw std::runtime_error(label + ": reducemean names axis " +
                               std::to_string(a) + " and the tensor has rank " +
                               std::to_string(rank) + ".");
    reduced[static_cast<size_t>(ax)] = true;
  }
  std::vector<int64_t> os;
  for (int64_t a = 0; a < rank; ++a) {
    if (l.keepdims) {
      os.push_back(reduced[static_cast<size_t>(a)] ? 1 : in.shape[static_cast<size_t>(a)]);
    } else if (!reduced[static_cast<size_t>(a)]) {
      os.push_back(in.shape[static_cast<size_t>(a)]);
    }
  }
  Tensor out;
  out.shape = os;
  out.d.assign(static_cast<size_t>(numel_of(os)), 0.f);
  // The output's axes map to the input's: the reduced ones are gone (or are
  // 1), and the others keep their extent.
  std::vector<int64_t> ostride(rank, 1);
  {
    // THE INPUT'S OWN STRIDES, and not the output's sizes: a reduced axis
    // still separates the elements on either side of it. The input here is
    // [1, 40, 120] and the reduction is over the LAST axis, so the input's
    // strides are [4800, 120, 1] -- and the strides of a tensor whose last
    // axis is 1 are the strides of a tensor that is 120 times shorter, so
    // reading the input through the output's sizes reads the wrong elements
    // and the mean comes out of a column that is not there.
    int64_t acc = 1;
    for (int64_t a = rank - 1; a >= 0; --a) {
      ostride[static_cast<size_t>(a)] = acc;
      acc *= in.shape[static_cast<size_t>(a)];
    }
  }
  const int64_t n_red = static_cast<int64_t>(
      std::count(reduced.begin(), reduced.end(), true));
  int64_t count = 1;
  for (int64_t a = 0; a < rank; ++a)
    if (reduced[static_cast<size_t>(a)]) count *= in.shape[static_cast<size_t>(a)];
  // The output index and the reduced index, walked together.
  const double t0 = app::now_s();
  std::vector<int64_t> idx(rank, 0);
  const int64_t total = numel_of(os);
  // The output's own strides over the kept axes.
  std::vector<int64_t> out_stride(rank, 0);
  {
    int64_t acc = 1;
    for (int64_t a = rank - 1; a >= 0; --a) {
      if (!reduced[static_cast<size_t>(a)]) {
        out_stride[static_cast<size_t>(a)] = acc;
        acc *= in.shape[static_cast<size_t>(a)];
      }
    }
  }
  for (int64_t n = 0; n < total; ++n) {
    double acc = 0.0;
    // Every combination of the reduced axes.
    std::vector<int64_t> ridx(static_cast<size_t>(n_red), 0);
    for (int64_t k = 0; k < count; ++k) {
      int64_t src = 0;
      size_t r = 0;
      for (int64_t a = 0; a < rank; ++a) {
        const int64_t v = reduced[static_cast<size_t>(a)]
                              ? ridx[r++]
                              : idx[static_cast<size_t>(a)];
        src += v * ostride[static_cast<size_t>(a)];
      }
      acc += in.d[static_cast<size_t>(src)];
      for (size_t a = n_red; a-- > 0;) {
        ++ridx[a];
        int64_t dim = 1;
        size_t r2 = 0;
        for (int64_t q = 0; q < rank; ++q)
          if (reduced[static_cast<size_t>(q)]) {
            if (r2 == a) dim = in.shape[static_cast<size_t>(q)];
            ++r2;
          }
        if (ridx[a] < dim) break;
        ridx[a] = 0;
      }
    }
    int64_t dst = 0;
    for (int64_t a = 0; a < rank; ++a)
      if (!reduced[static_cast<size_t>(a)])
        dst += idx[static_cast<size_t>(a)] * out_stride[static_cast<size_t>(a)];
    out.d[static_cast<size_t>(dst)] = static_cast<float>(acc / count);
    for (int64_t a = rank; a-- > 0;) {
      if (reduced[static_cast<size_t>(a)]) continue;
      ++idx[static_cast<size_t>(a)];
      if (idx[static_cast<size_t>(a)] < in.shape[static_cast<size_t>(a)]) break;
      idx[static_cast<size_t>(a)] = 0;
    }
  }
  cost_.t_elementwise += app::now_s() - t0;
  cost_.elementwise_ops++;
  return out;
}

}  // namespace npue::ocr
