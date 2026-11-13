//===- session.hpp -------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=9 session: one detector, then one recogniser per
// box.
//
// THE TWO STAGES
// --------------
// The DETECTOR runs once per frame and returns boxes; the RECOGNISER runs
// once per box, on a crop of that frame, and returns text. The two graphs
// have separate convolution index spaces -- both start at zero, both have a
// conv 0 -- and the array backend has one panel table, so the slot numbers
// are handed out over the DENSE convolutions of BOTH graphs in one run and
// each graph's own table translates its index into the right absolute slot.
//
// THE ORDER OF THE BOXES
// ----------------------
// The boxes come out in the detector's own order and are NOT sorted here.
// PaddleOCR sorts them top to bottom and left to right before reading them,
// and the sort is a READING convenience rather than part of the model -- so
// the caller that wants the reading order sorts them.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ocr/db.hpp"
#include "ocr/frontend.hpp"
#include "ocr/geometry.hpp"
#include "ocr/net.hpp"
#include "runtime/model.hpp"
#include "runtime/pool.hpp"
#include "vit/image.hpp"

namespace npue::ocr {

// One box and the text read out of it.
struct TextBox {
  Quad q;            // the four corners, in the image's coordinates
  double box_score = 0.0;   // the DB score: the mean probability inside
  std::string text;
  double conf = 0.0;        // the CTC confidence
};

struct Result {
  int64_t width = 0, height = 0;
  std::vector<TextBox> boxes;
  int64_t boxes_found = 0;   // before the cap
  int64_t boxes_read = 0;    // after it
  double total_s = 0.0;
  double front_end_s = 0.0;
  double det_s = 0.0;
  double post_s = 0.0;
  double crop_s = 0.0;
  double rec_s = 0.0;
  Cost det_cost;
  Cost rec_cost;
};

class Session {
public:
  Session(npue::File &model, const std::string &model_name,
          const std::string &artifacts, int threads, int64_t max_boxes,
          bool want_array);

  Result read_file(const std::string &path);
  Result read(const vit::Image &src);

  const Geometry &geometry() const { return geom_; }
  bool array_placement() const { return array_ != nullptr; }
  int64_t array_dispatches() const {
    return array_ ? array_->dispatches() : 0;
  }
  double array_seconds() const { return array_ ? array_->t_array() : 0.0; }
  size_t array_staged_bytes() const {
    return array_ ? array_->staged_bytes() : 0;
  }
  Network &det_network() { return *det_net_; }
  Network &rec_network() { return *rec_net_; }

private:
  npue::File &model_;
  std::string name_;
  std::string art_;
  int64_t max_boxes_ = -1;
  Geometry geom_;
  std::unique_ptr<app::Pool> pool_;
  std::unique_ptr<conv::Convs> array_;
  std::unique_ptr<Network> det_net_;
  std::unique_ptr<Network> rec_net_;
  std::vector<int64_t> det_slots_, rec_slots_;
};

}  // namespace npue::ocr
