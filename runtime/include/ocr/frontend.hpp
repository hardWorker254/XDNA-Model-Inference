//===- frontend.hpp ----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=9 image front ends.
//
// TWO FRONT ENDS, AND THEY ARE NOT THE SAME PIPELINE
// --------------------------------------------------
// The DETECTOR sees the whole frame, letterboxed to a fixed square; the
// RECOGNISER sees one crop, straightened and squashed to a fixed height.
//
// THE DETECTOR'S LETTERBOX
// ------------------------
// The image is scaled so that its LONG side is at most the container's
// detector size, and the remainder of the square is padded with BLACK in the
// 8-BIT raster -- and normalised AFTERWARDS. So the border of the normalised
// tensor is (0 - mean) / std and not 0.0, and that is a fact about the
// checkpoint's own preprocessing rather than a taste.
//
// The scale is computed in INTEGER arithmetic -- the long side becomes
// exactly the container's size and the short side is that times the ratio,
// truncated -- so that a 2000-pixel-wide image scales to exactly 960 and not
// to 959 because of a rounding of one part in 10^12. The last pixel column
// of the image lands one pixel short of the canvas's edge, which is a
// fraction of a pixel of error in every box.
//
// THE RECOGNISER'S CROP
// ---------------------
// The box's four corners are straightened by a PERSPECTIVE WARP -- the text
// is rarely square to the frame, and a rectangular crop of a slanted line is
// a line of half its height -- and the result is squashed to the container's
// height and padded on the right to its width.
//
// THE PADDING IS ZERO IN THE NORMALISED SPACE
// --------------------------------------------
// which is mid-grey, 127 in 0..255, and not the -1 that padding BEFORE the
// normalisation would give. PaddleOCR's own `resize_norm_img` normalises
// first and pads after, and the difference is one whole unit of the
// recogniser's input range over the whole right-hand side of every crop --
// which is where the blank frames of the CTC output live, and a border of -1
// instead of 0 makes those frames argue for a character.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <vector>

#include "ocr/db.hpp"
#include "ocr/geometry.hpp"
#include "vit/image.hpp"

namespace npue::ocr {

// The detector's input, and the two scales that undo the letterbox.
struct DetInput {
  std::vector<float> blob;    // [3, S, S], normalised
  double scale_x = 1.0;       // the image's pixels -> the canvas's
  double scale_y = 1.0;
  int64_t scaled_w = 0;       // the image's size inside the canvas
  int64_t scaled_h = 0;
};

DetInput det_input(const vit::Image &src, const Geometry &g);

// The recogniser's input, and the geometry of the crop it came from.
struct RecInput {
  std::vector<float> blob;    // [3, H, W], normalised and padded
  int64_t crop_w = 0;         // the warped crop's size
  int64_t crop_h = 0;
  int64_t content_w = 0;      // the content's width inside the padded frame
  bool rotated = false;       // the crop was turned a quarter turn
};

RecInput rec_input(const vit::Image &src, const Quad &q, const Geometry &g);

}  // namespace npue::ocr
