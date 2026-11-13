//===- db.hpp -------------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the DB post-process: a probability map in, boxes out.
//
// WHAT THIS IS
// ------------
// PaddleOCR's detector ends in a DIFFERENTIABLE BINARIZATION head: a
// probability map over the whole frame, and the text is wherever the map is
// above a threshold. Turning that into boxes is four steps, and each of them
// is a decision somebody could get wrong in a way that still returns boxes:
//
//   1. THRESHOLD.   pred > thresh (0.3) is the mask.
//   2. COMPONENTS.  Each connected blob of the mask is one candidate, and
//                   each is replaced by its MINIMUM-AREA RECTANGLE -- not its
//                   bounding box. A line of text is a long thin blob, and its
//                   bounding box is the same rectangle for a slanted line
//                   only if the line is horizontal.
//   3. SCORE.       The mean probability inside the rectangle, and a
//                   rectangle whose mean is below box_thresh (0.6) is
//                   dropped. This is what removes the half-text that the
//                   threshold let through.
//   4. UNCLIP.      The rectangle is grown by area * ratio / perimeter --
//                   about twelve pixels on every side of a 100x20 box -- so
//                   that it covers the ascenders and the gaps between the
//                   characters that the shrunk probability map does not.
//
// THE COMPONENTS ARE FOUND WITH 8-CONNECTIVITY
// --------------------------------------------
// which is what cv2.findContours does, and the two are the same thing here
// because OpenCV's contour is the boundary of one 8-connected blob and the
// minimum-area rectangle of a blob is the minimum-area rectangle of its
// boundary. A blob's HOLES are not candidates: they are regions where the
// probability is BELOW the threshold, so their mean score is below
// box_thresh and they are dropped by the score test.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <vector>

namespace npue::ocr {

// A box, as FOUR CORNERS in the order the recogniser wants them:
// top-left, top-right, bottom-right, bottom-left.
//
// The order is not cosmetic: the perspective warp that straightens the crop
// maps the corners to the corners of a rectangle in that order, and the
// crop's width comes from the two horizontal edges and its height from the
// two vertical ones. A different order is a crop turned a quarter turn,
// which the recogniser answers with a confident transcription of nothing.
struct Quad {
  double x[4] = {0, 0, 0, 0};
  double y[4] = {0, 0, 0, 0};
};

struct DbBox {
  Quad q;
  double score = 0.0;
};

// The DB post-process.
//
// `pred` is the probability map, [H, W] row-major -- the detector's output
// after the sigmoid, which the graph has already applied.
//
// `scale_x` and `scale_y` are how many map pixels there are in one pixel of
// the image: the detector saw the image letterboxed into the frame, with its
// proportions kept, so the map is the image blown up by that factor and the
// boxes are scaled back down by it. `dest_w` and `dest_h` are the size of
// the image the boxes are returned in, and they are what the boxes are
// clipped to.
//
// The two scales are not the same number, and that is not a detail: the map
// is the WHOLE frame and the image is only part of it, so a box's distance
// from the frame's edge says how far it is from the image's edge and not how
// far it is from the text.
std::vector<DbBox> db_postprocess(const float *pred, int64_t h, int64_t w,
                                  double thresh, double box_thresh,
                                  double unclip_ratio, int64_t max_candidates,
                                  int64_t min_size, double scale_x,
                                  double scale_y, double dest_w, double dest_h);

}  // namespace npue::ocr
