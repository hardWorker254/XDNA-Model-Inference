//===- session.cpp -----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=9 session. See session.hpp for the contracts.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "ocr/session.hpp"

#include <algorithm>
#include <stdexcept>

#include "common/host_kernels.hpp"   // app::now_s
#include "ocr/ctc.hpp"

namespace npue::ocr {

namespace {

// The design stream one convolution runs on, or "" if the graph does not say.
//
// The graph is SEARCHED for the node that calls this convolution rather than
// assumed to have one entry per convolution: the two are equal in every
// container in this repository, and if that stops being true the wrong answer
// here is a panel staged for another layer -- right shape, wrong weights, no
// error.
std::string stream_of(const std::vector<Layer> &layers, int64_t conv,
                      const std::string &name) {
  const Layer *found = nullptr;
  for (const Layer &l : layers)
    if (l.op == Op::Conv && l.conv == conv) {
      if (found)
        throw std::runtime_error(
            name + ": convolution " + std::to_string(conv) +
            " is called by two graph nodes. The array path keys panels by which "
            "node calls a convolution, and two nodes sharing one would stage "
            "one panel for two different layers.");
      found = &l;
    }
  return found ? found->stream : std::string();
}

}  // namespace

Session::Session(npue::File &model, const std::string &model_name,
                 const std::string &artifacts, int threads, int64_t max_boxes,
                 bool want_array)
    : model_(model), name_(model_name), art_(artifacts),
      max_boxes_(max_boxes) {
  geom_ = read_geometry(model_, name_);
  pool_ = std::make_unique<app::Pool>(threads);
  Placement place;   // the default: everything on the host
  if (want_array) {
    place.conv_on_array = true;
    place.design_dir = art_;
    std::vector<conv::ConvSlot> slots;
    // THE SLOT RANGES, ASSIGNED HERE AND FROM ONE VECTOR.
    //
    // The detector's convolutions and the recogniser's get DISJOINT slot
    // ranges, in that order, and the same vector is what stages the panels --
    // there is one number per convolution and it is written once.
    //
    // The alternative -- keying panels by the bare convolution index -- puts
    // the recogniser's conv 0 on the detector's conv 0 panel. Both are the
    // stem of a network of similar width, so the panel is the right shape,
    // the layout_hash matches, every load succeeds, and the boxes come out
    // of a recogniser's weights.
    //
    // THE SLOT TABLES ARE FILLED AT THE SAME TIME AND FROM THE SAME LOOP,
    // which is the whole point: slot numbers are handed out over the DENSE
    // convolutions only, so `slot = base + conv` is WRONG here -- the
    // depthwise ones are interleaved through both graphs, and the detector's
    // conv 8 is not slot 8.
    //
    // AND THE TRANSPOSED ONES ARE LEFT OUT TOO, for the same reason as the
    // depthwise ones: the detector's DB head ends in two transposed
    // convolutions that upsample the probability map, and a transposed
    // convolution is a sum of shifted contributions rather than a matrix
    // product, so there is no [M, N] GEMM in it to dispatch. The packer
    // stages no panel for one, and a slot with no panel behind it is a run
    // the array backend refuses -- with a message about the SLOT ("has no
    // stream name"), which reads like a missing panel and is not one.
    det_slots_.assign(geom_.det_convs.size(), -1);
    for (size_t i = 0; i < geom_.det_convs.size(); ++i) {
      const ConvW &cw = geom_.det_convs[i];
      if (!cw.arrayable()) continue;
      det_slots_[i] = static_cast<int64_t>(slots.size());
      slots.push_back(conv::ConvSlot{static_cast<int64_t>(slots.size()), "det",
                                     "det.", static_cast<int64_t>(i),
                                     stream_of(geom_.det_graph,
                                               static_cast<int64_t>(i), name_),
                                     cw.cin, cw.cout, cw.kh, cw.kw, cw.b});
    }
    rec_slots_.assign(geom_.rec_convs.size(), -1);
    for (size_t i = 0; i < geom_.rec_convs.size(); ++i) {
      const ConvW &cw = geom_.rec_convs[i];
      if (!cw.arrayable()) continue;
      rec_slots_[i] = static_cast<int64_t>(slots.size());
      slots.push_back(conv::ConvSlot{static_cast<int64_t>(slots.size()), "rec",
                                     "rec.", static_cast<int64_t>(i),
                                     stream_of(geom_.rec_graph,
                                               static_cast<int64_t>(i), name_),
                                     cw.cin, cw.cout, cw.kh, cw.kw, cw.b});
    }
    if (slots.empty())
      throw std::runtime_error(
          name_ + ": --npu-ops conv was asked for and neither graph has a dense "
          "convolution. A design set with no slots behind it is not a design "
          "set.");
    array_ = conv::make_array_backend(model_, slots, name_, art_, *pool_);
  }
  Placement det_place = place, rec_place = place;
  det_place.group = "det";
  det_place.slots = &det_slots_;
  rec_place.group = "rec";
  rec_place.slots = &rec_slots_;
  det_net_ = std::make_unique<Network>(geom_, det_place, *pool_,
                                       geom_.det_convs,
                                       array_ ? array_.get() : nullptr);
  rec_net_ = std::make_unique<Network>(geom_, rec_place, *pool_,
                                       geom_.rec_convs,
                                       array_ ? array_.get() : nullptr);
}

Result Session::read_file(const std::string &path) {
  return read(vit::decode_image(path));
}

Result Session::read(const vit::Image &src) {
  Result r;
  if (src.empty())
    throw std::runtime_error(name_ + ": the image decoded to nothing");
  r.width = src.width;
  r.height = src.height;
  det_net_->reset_cost();
  rec_net_->reset_cost();

  // -- stage 1: the detector
  const double t0 = app::now_s();
  const DetInput din = det_input(src, geom_);
  r.front_end_s = app::now_s() - t0;

  Tensor det_tensor;
  det_tensor.shape = {1, 3, geom_.det_input_size, geom_.det_input_size};
  det_tensor.d = din.blob;

  const double t1 = app::now_s();
  const std::vector<Tensor> &dnodes =
      det_net_->run(det_tensor, geom_.det_graph, geom_.det_convs, "det");
  r.det_s = app::now_s() - t1;

  // THE PROBABILITY MAP. The detector's output is [1, 1, H, W] -- one channel
  // of probabilities over the whole frame -- and the post-process reads it as
  // a map.
  const Tensor &prob = dnodes.back();
  if (prob.rank() != 4 || prob.shape[0] != 1 || prob.shape[1] != 1)
    throw std::runtime_error(
        name_ + ": the detector's output is " +
        std::to_string(prob.shape.size()) + " dimensions and the first two are " +
        std::to_string(prob.shape[0]) + " and " +
        std::to_string(prob.shape.size() > 1 ? prob.shape[1] : 0) +
        ". The DB post-process reads a [1, 1, H, W] probability map.");
  const int64_t ph = prob.shape[2], pw = prob.shape[3];

  const double t2 = app::now_s();
  const std::vector<DbBox> found =
      db_postprocess(prob.d.data(), ph, pw, geom_.det_thresh,
                     geom_.det_box_thresh, geom_.det_unclip_ratio,
                     geom_.det_max_candidates, geom_.det_min_size,
                     din.scale_x, din.scale_y,
                     static_cast<double>(src.width),
                     static_cast<double>(src.height));
  r.post_s = app::now_s() - t2;
  r.boxes_found = static_cast<int64_t>(found.size());

  // -- stage 2: one recogniser per box
  const int64_t n = static_cast<int64_t>(found.size());
  const int64_t run = (max_boxes_ < 0 || n < max_boxes_) ? n : max_boxes_;
  r.boxes_read = run;
  for (int64_t i = 0; i < run; ++i) {
    const DbBox &b = found[static_cast<size_t>(i)];
    const double t3 = app::now_s();
    const RecInput rin = rec_input(src, b.q, geom_);
    r.crop_s += app::now_s() - t3;

    Tensor rec_tensor;
    rec_tensor.shape = {1, 3, geom_.rec_input_height, geom_.rec_input_width};
    rec_tensor.d = rin.blob;

    const double t4 = app::now_s();
    const std::vector<Tensor> &rnodes =
        rec_net_->run(rec_tensor, geom_.rec_graph, geom_.rec_convs, "rec");
    r.rec_s += app::now_s() - t4;

    // THE CTC HEAD'S OUTPUT: [1, frames, classes]. The frames are the
    // recogniser's time steps -- one per eight columns of the crop -- and
    // the classes are the dictionary's characters plus the blank and the
    // space.
    const Tensor &logits = rnodes.back();
    if (logits.rank() != 3)
      throw std::runtime_error(name_ + ": the recogniser's output is rank " +
                               std::to_string(logits.rank()) +
                               " and the CTC decoder reads a rank-3 "
                               "[1, frames, classes] tensor.");
    const int64_t frames = logits.shape[1];
    const int64_t classes = logits.shape[2];
    const CtcResult cr =
        ctc_decode(logits.d.data(), frames, classes, geom_.dict);

    TextBox t;
    t.q = b.q;
    t.box_score = b.score;
    t.text = cr.text;
    t.conf = cr.conf;
    r.boxes.push_back(std::move(t));
  }
  r.det_cost = det_net_->cost();
  r.rec_cost = rec_net_->cost();
  r.total_s = app::now_s() - t0;
  return r;
}

}  // namespace npue::ocr
