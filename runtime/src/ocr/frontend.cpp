//===- frontend.cpp ----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=9 image front ends. See frontend.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "ocr/frontend.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/raster.hpp"

namespace npue::ocr {

namespace {

// A crop turned a quarter turn, COUNTER-CLOCKWISE, which is what numpy's
// rot90 does: B[j][i] = A[i][W-1-j].
//
// The crop is turned when it is TALL -- a height at least one and a half
// times its width -- because that is a line of text going UP the frame, and
// the recogniser was trained on lines that go across. PaddleOCR's own
// pipeline does the same, and the threshold is its 1.5.
vit::Image rot90(const vit::Image &src) {
  vit::Image out;
  out.width = src.height;
  out.height = src.width;
  out.rgb.resize(static_cast<size_t>(src.height) * static_cast<size_t>(src.width) * 3);
  for (int64_t j = 0; j < out.height; ++j) {
    for (int64_t i = 0; i < out.width; ++i) {
      const int64_t sy = i;
      const int64_t sx = src.width - 1 - j;
      const uint8_t *const s = src.at(sy, sx);
      uint8_t *const d = out.at(j, i);
      d[0] = s[0];
      d[1] = s[1];
      d[2] = s[2];
    }
  }
  return out;
}

}  // namespace

DetInput det_input(const vit::Image &src, const Geometry &g) {
  DetInput out;
  if (src.empty())
    throw std::runtime_error("the detector's front end: the image is empty");
  const int64_t S = g.det_input_size;
  const int64_t W = src.width, H = src.height;
  // THE SCALE, IN INTEGER ARITHMETIC. The long side becomes exactly S and
  // the short side is that times the ratio, truncated -- so a 2000-pixel
  // image scales to exactly 960 and not to 959 because of a rounding.
  int64_t sw, sh;
  if (W >= H) {
    sw = S;
    sh = std::max<int64_t>(1, (H * S) / W);
  } else {
    sh = S;
    sw = std::max<int64_t>(1, (W * S) / H);
  }
  out.scaled_w = sw;
  out.scaled_h = sh;
  out.scale_x = static_cast<double>(sw) / static_cast<double>(W);
  out.scale_y = static_cast<double>(sh) / static_cast<double>(H);

  // THE RESIZE, then the PAD, then the NORMALISATION -- in that order. The
  // pad is BLACK in the 8-BIT raster and the normalisation comes after, so
  // the border of the tensor is (0 - mean) / std.
  const vit::Image scaled = npue::raster::resize_bilinear(src, sh, sw);
  vit::Image canvas = npue::raster::like(S, S);
  for (int64_t y = 0; y < sh; ++y)
    for (int64_t x = 0; x < sw; ++x) {
      const uint8_t *const s = scaled.at(y, x);
      uint8_t *const d = canvas.at(y, x);
      d[0] = s[0];
      d[1] = s[1];
      d[2] = s[2];
    }
  out.blob = npue::raster::to_nchw_normalised(canvas, g.det_mean, g.det_std);
  return out;
}

RecInput rec_input(const vit::Image &src, const Quad &q, const Geometry &g) {
  RecInput out;
  if (src.empty())
    throw std::runtime_error("the recogniser's front end: the image is empty");
  // THE CROP'S SIZE: the longer of the two horizontal edges and the longer
  // of the two vertical ones. The corners are in the order TOP-LEFT,
  // TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT, so the horizontal edges are
  // (0,1) and (2,3) and the vertical ones are (0,3) and (1,2).
  const double e01 = std::hypot(q.x[1] - q.x[0], q.y[1] - q.y[0]);
  const double e23 = std::hypot(q.x[3] - q.x[2], q.y[3] - q.y[2]);
  const double e03 = std::hypot(q.x[3] - q.x[0], q.y[3] - q.y[0]);
  const double e12 = std::hypot(q.x[2] - q.x[1], q.y[2] - q.y[1]);
  const int64_t cw = std::max<int64_t>(1, static_cast<int64_t>(std::max(e01, e23)));
  const int64_t ch = std::max<int64_t>(1, static_cast<int64_t>(std::max(e03, e12)));
  out.crop_w = cw;
  out.crop_h = ch;

  // THE PERSPECTIVE WARP. The homography is asked for in the direction the
  // resampling wants it -- from the destination rectangle to the source
  // quad -- rather than computed forwards and inverted, which is the same
  // thing without a matrix inversion.
  const double rect[8] = {0, 0, static_cast<double>(cw), 0,
                          static_cast<double>(cw), static_cast<double>(ch), 0,
                          static_cast<double>(ch)};
  const double quad[8] = {q.x[0], q.y[0], q.x[1], q.y[1],
                          q.x[2], q.y[2], q.x[3], q.y[3]};
  double m[9];
  if (!npue::raster::get_perspective_transform(rect, quad, m))
    throw std::runtime_error(
        "the recogniser's front end: the box's four corners do not determine "
        "a homography. Three of them are collinear or two of them are the "
        "same point, and the crop of such a box is not an image.");
  vit::Image crop = npue::raster::warp_perspective(src, m, ch, cw);

  // THE TURN, for a line of text that goes UP the frame.
  if (static_cast<double>(ch) / static_cast<double>(cw) >= 1.5) {
    crop = rot90(crop);
    out.rotated = true;
  }

  // THE RESIZE: the content is squashed to the container's height, and its
  // width follows the crop's own aspect ratio -- FLOORED, which is what
  // PaddleOCR's own `resize_norm_img` does: it computes the padded width as
  // int(height * ratio) and the resized width as the smaller of that and
  // ceil(height * ratio), so the content is never wider than the frame it is
  // padded into.
  const int64_t H = g.rec_input_height;
  const int64_t W = g.rec_input_width;
  const double ratio = static_cast<double>(crop.width) /
                       static_cast<double>(std::max<int64_t>(1, crop.height));
  int64_t rw = static_cast<int64_t>(static_cast<double>(H) * ratio);
  rw = std::min<int64_t>(std::max<int64_t>(rw, 1), W);
  out.content_w = rw;

  const vit::Image scaled = npue::raster::resize_bilinear(crop, H, rw);
  // THE NORMALISATION, then the PAD -- in that order, and the pad is in the
  // NORMALISED space.
  std::vector<float> blob = npue::raster::to_nchw_normalised(scaled, g.rec_mean,
                                                             g.rec_std);
  const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);
  out.blob.assign(3 * plane, static_cast<float>(g.rec_pad_value));
  for (int64_t c = 0; c < 3; ++c)
    for (int64_t y = 0; y < H; ++y)
      for (int64_t x = 0; x < rw; ++x)
        out.blob[static_cast<size_t>(c) * plane +
                 static_cast<size_t>(y) * static_cast<size_t>(W) + x] =
            blob[static_cast<size_t>(c) * static_cast<size_t>(H) *
                     static_cast<size_t>(rw) +
                 static_cast<size_t>(y) * static_cast<size_t>(rw) + x];
  return out;
}

}  // namespace npue::ocr
