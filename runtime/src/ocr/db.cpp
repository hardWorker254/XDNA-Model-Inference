//===- db.cpp -------------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the DB post-process. See db.hpp for the four steps and for
// why each of them is a decision.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "ocr/db.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace npue::ocr {

namespace {

struct Pt {
  double x = 0, y = 0;
};

// The CONVEX HULL, by Andrew's monotone chain. The minimum-area rectangle is
// determined by the hull -- a point inside the hull cannot be a corner of it
// -- so the hull is what the rotating calipers below walks.
std::vector<Pt> convex_hull(std::vector<Pt> p) {
  std::sort(p.begin(), p.end(), [](const Pt &a, const Pt &b) {
    return a.x != b.x ? a.x < b.x : a.y < b.y;
  });
  p.erase(std::unique(p.begin(), p.end(),
                       [](const Pt &a, const Pt &b) {
                         return a.x == b.x && a.y == b.y;
                       }),
          p.end());
  const size_t n = p.size();
  if (n < 3) return p;
  std::vector<Pt> h(2 * n);
  size_t k = 0;
  for (size_t i = 0; i < n; ++i) {
    while (k >= 2 &&
           (h[k - 1].x - h[k - 2].x) * (p[i].y - h[k - 2].y) -
               (h[k - 1].y - h[k - 2].y) * (p[i].x - h[k - 2].x) <=
               0)
      --k;
    h[k++] = p[i];
  }
  // THE UPPER HULL, walking the points from the right back to the left.
  //
  // The point that goes into the CROSS PRODUCT is the one that is about to be
  // pushed, and not its neighbour: the test asks whether the last two points
  // of the chain and THIS one make a left turn, and "this one" is the point
  // being added. Reading the neighbour instead is an off-by-one that costs the
  // hull its lower edge -- the hull then describes the top of a text line
  // only, the minimum-area rectangle is half as tall as it should be, and the
  // box that comes out of it is a strip through the middle of the glyphs.
  for (size_t i = n - 1, t = k + 1; i > 0; --i) {
    const Pt &q = p[i - 1];
    while (k >= t &&
           (h[k - 1].x - h[k - 2].x) * (q.y - h[k - 2].y) -
               (h[k - 1].y - h[k - 2].y) * (q.x - h[k - 2].x) <=
               0)
      --k;
    h[k++] = q;
  }
  h.resize(k > 0 ? k - 1 : 0);
  return h;
}

// THE MINIMUM-AREA RECTANGLE, by rotating calipers over the hull.
//
// For each hull EDGE, the rectangle that has that edge as one side: the
// extent along the edge is the projection's range, and the extent across it
// is the largest distance any point reaches from the edge's line. The
// smallest such rectangle is the minimum-area one.
//
// The corners are returned in the order TOP-LEFT, BOTTOM-LEFT, BOTTOM-RIGHT,
// TOP-RIGHT -- the order the recogniser's crop wants -- and they are ordered
// the way PaddleOCR orders them: sort the four by x, take the lower of the
// two left-most as the top-left, and the lower of the two right-most as the
// bottom-right.
struct Rect {
  double cx = 0, cy = 0;      // the centre
  double ux = 1, uy = 0;       // the unit vector along the width
  double vx = 0, vy = 1;       // the unit vector along the height
  double hw = 0, hh = 0;       // the half extents
  double area = 0;
};

Rect min_area_rect(const std::vector<Pt> &pts) {
  Rect best;
  std::vector<Pt> hull = convex_hull(pts);
  if (hull.size() < 3) {
    // ONE POINT or a segment: the rectangle is degenerate, and it is zero
    // wide or zero tall -- which the min_size test below rejects.
    if (!hull.empty()) {
      best.cx = hull[0].x;
      best.cy = hull[0].y;
    }
    if (hull.size() == 2) {
      best.hw = 0.5 * std::hypot(hull[1].x - hull[0].x, hull[1].y - hull[0].y);
      best.area = 0;
    }
    return best;
  }
  double best_area = 1e300;
  const size_t n = hull.size();
  for (size_t i = 0; i < n; ++i) {
    const Pt &a = hull[i];
    const Pt &b = hull[(i + 1) % n];
    const double ex = b.x - a.x, ey = b.y - a.y;
    const double len = std::hypot(ex, ey);
    if (len <= 0.0) continue;
    const double ux = ex / len, uy = ey / len;   // along the edge
    const double vx = -uy, vy = ux;              // across it
    double tmin = 1e300, tmax = -1e300, smin = 1e300, smax = -1e300;
    for (const Pt &p : hull) {
      const double t = p.x * ux + p.y * uy;
      const double s = p.x * vx + p.y * vy;
      tmin = std::min(tmin, t);
      tmax = std::max(tmax, t);
      smin = std::min(smin, s);
      smax = std::max(smax, s);
    }
    const double area = (tmax - tmin) * (smax - smin);
    if (area < best_area) {
      best_area = area;
      const double tm = 0.5 * (tmin + tmax), sm = 0.5 * (smin + smax);
      best.cx = tm * ux + sm * vx;
      best.cy = tm * uy + sm * vy;
      best.ux = ux;
      best.uy = uy;
      best.vx = vx;
      best.vy = vy;
      best.hw = 0.5 * (tmax - tmin);
      best.hh = 0.5 * (smax - smin);
      best.area = area;
    }
  }
  return best;
}

// THE ORDER OF THE FOUR CORNERS, and it is PaddleOCR's.
//
// Sort the corners by x; the two left-most are the LEFT pair and the two
// right-most the RIGHT pair; within each pair the one with the SMALLER y is
// the upper one -- and in an IMAGE, y grows downwards, so the smaller y is
// the top.
//
// The result is [TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT], and the
// order is not cosmetic: the recogniser's crop takes the width from the two
// horizontal edges and the height from the two vertical ones, and reads the
// corners in that order. A different order is a crop turned a quarter turn,
// which the recogniser answers with a confident transcription of nothing.
void order_corners(const Pt c[4], int pick[4]) {
  int order[4] = {0, 1, 2, 3};
  std::sort(order, order + 4, [&](int a, int b) {
    return c[a].x != c[b].x ? c[a].x < c[b].x : c[a].y < c[b].y;
  });
  const int l0 = order[0], l1 = order[1], r0 = order[2], r1 = order[3];
  // The left pair's upper corner is the top-left; the right pair's upper
  // corner is the top-right.
  const int tl = (c[l0].y <= c[l1].y) ? l0 : l1;
  const int bl = (tl == l0) ? l1 : l0;
  const int tr = (c[r0].y <= c[r1].y) ? r0 : r1;
  const int br = (tr == r0) ? r1 : r0;
  pick[0] = tl;
  pick[1] = tr;
  pick[2] = br;
  pick[3] = bl;
}

// The rectangle's four corners, in the order the recogniser wants.
Quad rect_corners(const Rect &r) {
  // The four corners in the rectangle's own frame.
  const double cu[4] = {-1, -1, 1, 1};
  const double cv[4] = {-1, 1, 1, -1};
  Pt c[4];
  for (int k = 0; k < 4; ++k) {
    const double t = cu[k] * r.hw, s = cv[k] * r.hh;
    c[k].x = r.cx + t * r.ux + s * r.vx;
    c[k].y = r.cy + t * r.uy + s * r.vy;
  }
  int pick[4];
  order_corners(c, pick);
  Quad q;
  for (int k = 0; k < 4; ++k) {
    q.x[k] = c[pick[k]].x;
    q.y[k] = c[pick[k]].y;
  }
  return q;
}

// The mean probability INSIDE a convex quad, over the quad's axis-aligned
// bounding box.
//
// The pixel centres are tested against the quad's four edges, which is
// exact for a convex polygon and is what a scanline fill computes. The
// polygon here is always the minimum-area rectangle, so it is convex.
double box_score(const float *pred, int64_t h, int64_t w, const Quad &q) {
  double xmin = q.x[0], xmax = q.x[0], ymin = q.y[0], ymax = q.y[0];
  for (int k = 1; k < 4; ++k) {
    xmin = std::min(xmin, q.x[k]);
    xmax = std::max(xmax, q.x[k]);
    ymin = std::min(ymin, q.y[k]);
    ymax = std::max(ymax, q.y[k]);
  }
  int64_t x0 = static_cast<int64_t>(std::floor(xmin));
  int64_t x1 = static_cast<int64_t>(std::ceil(xmax));
  int64_t y0 = static_cast<int64_t>(std::floor(ymin));
  int64_t y1 = static_cast<int64_t>(std::ceil(ymax));
  x0 = std::min<int64_t>(std::max<int64_t>(x0, 0), w - 1);
  x1 = std::min<int64_t>(std::max<int64_t>(x1, 0), w - 1);
  y0 = std::min<int64_t>(std::max<int64_t>(y0, 0), h - 1);
  y1 = std::min<int64_t>(std::max<int64_t>(y1, 0), h - 1);
  double sum = 0.0;
  int64_t n = 0;
  for (int64_t y = y0; y <= y1; ++y) {
    for (int64_t x = x0; x <= x1; ++x) {
      const double px = x + 0.5, py = y + 0.5;
      // INSIDE, by the sign of the four cross products. The polygon is
      // convex and wound the same way, so all four have the same sign inside
      // and the test is a comparison of products.
      double sign = 0.0;
      bool inside = true;
      for (int k = 0; k < 4; ++k) {
        const int k2 = (k + 1) & 3;
        const double cr = (q.x[k2] - q.x[k]) * (py - q.y[k]) -
                          (q.y[k2] - q.y[k]) * (px - q.x[k]);
        if (k == 0) sign = cr;
        else if ((cr > 0) != (sign > 0) && cr != 0.0) {
          inside = false;
          break;
        }
      }
      if (!inside) continue;
      sum += pred[y * w + x];
      ++n;
    }
  }
  if (n == 0) return 0.0;
  return sum / static_cast<double>(n);
}

}  // namespace

std::vector<DbBox> db_postprocess(const float *pred, int64_t h, int64_t w,
                                  double thresh, double box_thresh,
                                  double unclip_ratio, int64_t max_candidates,
                                  int64_t min_size, double scale_x,
                                  double scale_y, double dest_w,
                                  double dest_h) {
  std::vector<DbBox> out;
  if (h <= 0 || w <= 0) return out;

  // -- 1. the mask
  std::vector<uint8_t> mask(static_cast<size_t>(h) * static_cast<size_t>(w), 0);
  for (int64_t i = 0; i < h * w; ++i)
    mask[static_cast<size_t>(i)] = pred[i] > thresh ? 1 : 0;

  // -- 2. the connected components, 8-connected, in raster order
  //
  // The raster order is the order cv2.findContours returns them in, closely
  // enough: it matters only when there are more than max_candidates blobs,
  // and then it decides WHICH of them is kept.
  std::vector<int32_t> label(static_cast<size_t>(h) * static_cast<size_t>(w), -1);
  std::vector<std::vector<Pt>> blobs;
  std::vector<int32_t> stack;
  for (int64_t y0 = 0; y0 < h; ++y0) {
    for (int64_t x0 = 0; x0 < w; ++x0) {
      const size_t seed = static_cast<size_t>(y0 * w + x0);
      if (!mask[seed] || label[seed] >= 0) continue;
      const int32_t id = static_cast<int32_t>(blobs.size());
      blobs.push_back({});
      stack.clear();
      stack.push_back(static_cast<int32_t>(seed));
      label[seed] = id;
      while (!stack.empty()) {
        const int32_t cur = stack.back();
        stack.pop_back();
        const int64_t cy = cur / w, cx = cur % w;
        blobs[static_cast<size_t>(id)].push_back(Pt{static_cast<double>(cx),
                                                    static_cast<double>(cy)});
        for (int64_t dy = -1; dy <= 1; ++dy) {
          for (int64_t dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            const int64_t ny = cy + dy, nx = cx + dx;
            if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
            const size_t ni = static_cast<size_t>(ny * w + nx);
            if (!mask[ni] || label[ni] >= 0) continue;
            label[ni] = id;
            stack.push_back(static_cast<int32_t>(ni));
          }
        }
      }
    }
  }

  // -- 3. each blob's rectangle, its score, and the un-clip
  const int64_t take =
      std::min<int64_t>(static_cast<int64_t>(blobs.size()), max_candidates);
  for (int64_t b = 0; b < take; ++b) {
    const Rect r = min_area_rect(blobs[static_cast<size_t>(b)]);
    // THE SMALLEST SIDE, and it is the smaller of the two: a blob that is
    // three pixels tall is noise, and a blob that is three pixels wide is a
    // single character -- which is the same test PaddleOCR makes.
    const double min_side = std::min(2.0 * r.hw, 2.0 * r.hh);
    if (min_side < static_cast<double>(min_size)) continue;
    const Quad q = rect_corners(r);
    const double score = box_score(pred, h, w, q);
    if (box_thresh > score) continue;

    // THE UN-CLIP. The distance is the rectangle's own area times the ratio
    // over its perimeter -- PaddleOCR's `distance = poly.area * unclip_ratio
    // / poly.length`, and the rectangle's area is w*h and its length is
    // 2*(w+h). For a 100x20 box and a ratio of 1.5 that is 12.5 pixels, so
    // the box grows by about twelve on every side: enough for the ascenders
    // and the gaps between the characters, and not so much that neighbouring
    // lines merge.
    //
    // The expansion of a RECTANGLE by a distance d, with round joins, is the
    // rectangle grown by d on every side -- the rounded corners do not reach
    // past the grown edges -- so the minimum-area rectangle of the expanded
    // polygon is the grown rectangle. Which is what is computed here, rather
    // than a polygon offset: the offset's polygon has a hundred points whose
    // only use is to be measured again.
    const double pw = 2.0 * r.hw, ph = 2.0 * r.hh;
    const double per = 2.0 * (pw + ph);
    const double d = per > 0.0 ? (pw * ph * unclip_ratio) / per : 0.0;
    const double nhw = r.hw + d, nhh = r.hh + d;
    if (std::min(2.0 * nhw, 2.0 * nhh) < static_cast<double>(min_size) + 2.0)
      continue;

    // The grown rectangle's corners, in the same order.
    const double cu[4] = {-1, -1, 1, 1};
    const double cvv[4] = {-1, 1, 1, -1};
    Pt c[4];
    for (int k = 0; k < 4; ++k) {
      const double t = cu[k] * nhw, s = cvv[k] * nhh;
      c[k].x = r.cx + t * r.ux + s * r.vx;
      c[k].y = r.cy + t * r.uy + s * r.vy;
    }
    int pick[4];
    order_corners(c, pick);

    DbBox box;
    box.score = score;
    for (int k = 0; k < 4; ++k) {
      // INTO THE IMAGE'S COORDINATES: the map's pixels to the image's,
      // divided by the scale the front end scaled the image by, rounded half
      // to even and clipped to the image.
      //
      // The scale is the front end's, and not the map's size over the
      // image's: the image was letterboxed into the frame with its
      // proportions kept, so the map's frame is BIGGER than the image and
      // the text sits in a corner of it. Dividing the map's size by the
      // image's size instead -- the scale of a frame that was STRETCHED to
      // fit -- puts every box in the wrong place, and wrong by the amount
      // the frame was stretched by: on a 900x360 image in a 960x960 frame
      // that is a factor of two and a half, and the boxes land on the blank
      // paper above the text.
      double bx = c[pick[k]].x / scale_x;
      double by = c[pick[k]].y / scale_y;
      bx = std::nearbyint(bx);
      by = std::nearbyint(by);
      box.q.x[k] = std::min(std::max(bx, 0.0), dest_w);
      box.q.y[k] = std::min(std::max(by, 0.0), dest_h);
    }
    out.push_back(box);
  }
  return out;
}

}  // namespace npue::ocr
