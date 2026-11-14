//! The arch=9 front ends, the DB post-process and the CTC decode.
//!
//! # TWO FRONT ENDS, AND THEY ARE NOT THE SAME PIPELINE
//!
//! The DETECTOR sees the whole frame, letterboxed to a fixed square; the
//! RECOGNISER sees one crop, straightened and squashed to a fixed height.
//!
//! # THE DETECTOR'S LETTERBOX
//!
//! The image is scaled so that its LONG side is at most the container's
//! detector size, and the remainder of the square is padded with BLACK in the
//! 8-BIT raster -- and normalised AFTERWARDS. So the border of the normalised
//! tensor is `(0 - mean) / std` and not 0.0, and that is a fact about the
//! checkpoint's own preprocessing rather than a taste.
//!
//! The scale is computed in INTEGER arithmetic -- the long side becomes
//! exactly the container's size and the short side is that times the ratio,
//! truncated -- so that a 2000-pixel-wide image scales to exactly 960 and not
//! to 959 because of a rounding of one part in 10^12. The last pixel column
//! of the image lands one pixel short of the canvas's edge, which is a
//! fraction of a pixel of error in every box.
//!
//! # THE RECOGNISER'S CROP
//!
//! The box's four corners are straightened by a PERSPECTIVE WARP -- the text
//! is rarely square to the frame, and a rectangular crop of a slanted line is
//! a line of half its height -- and the result is squashed to the container's
//! height and padded on the right to its width.
//!
//! # THE PADDING IS ZERO IN THE NORMALISED SPACE
//!
//! which is mid-grey, 127 in 0..255, and not the -1 that padding BEFORE the
//! normalisation would give. PaddleOCR's own `resize_norm_img` normalises
//! first and pads after, and the difference is one whole unit of the
//! recogniser's input range over the whole right-hand side of every crop --
//! which is where the blank frames of the CTC output live, and a border of -1
//! instead of 0 makes those frames argue for a character.
//!
//! # THE DB POST-PROCESS IS FOUR DECISIONS
//!
//! 1. THRESHOLD. `pred > thresh (0.3)` is the mask.
//! 2. COMPONENTS. Each connected blob of the mask is one candidate, and each
//!    is replaced by its MINIMUM-AREA RECTANGLE -- not its bounding box. A
//!    line of text is a long thin blob, and its bounding box is the same
//!    rectangle for a slanted line only if the line is horizontal.
//! 3. SCORE. The mean probability inside the rectangle, and a rectangle whose
//!    mean is below `box_thresh (0.6)` is dropped.
//! 4. UNCLIP. The rectangle is grown by `area * ratio / perimeter` -- about
//!    twelve pixels on every side of a 100x20 box.
//!
//! The components are found with 8-CONNECTIVITY, which is what
//! `cv2.findContours` does.

use crate::hands_decode::{resize_bilinear, to_nchw_normalised_values, to_u8};
use crate::ocr::{Geometry, Tensor};
use crate::vit_image::Image;

// ---------------------------------------------------------------------------
// The detector's front end
// ---------------------------------------------------------------------------

/// The detector's input, and the two scales that undo the letterbox.
#[derive(Debug, Clone, Default)]
pub struct DetInput {
    /// `[3, S, S]`, normalised.
    pub blob: Vec<f32>,
    /// The image's pixels -> the canvas's.
    pub scale_x: f64,
    /// The image's pixels -> the canvas's.
    pub scale_y: f64,
    /// The image's size inside the canvas.
    pub scaled_w: i64,
    /// The image's size inside the canvas.
    pub scaled_h: i64,
}

/// The detector's input: the frame, letterboxed and normalised.
pub fn det_input(src: &Image, g: &Geometry) -> Result<DetInput, String> {
    if src.empty() {
        return Err("the detector's front end: the image is empty".into());
    }
    let s = g.det_input_size;
    let w = src.width;
    let h = src.height;
    // THE SCALE, IN INTEGER ARITHMETIC. The long side becomes exactly S and
    // the short side is that times the ratio, truncated -- so a 2000-pixel
    // image scales to exactly 960 and not to 959 because of a rounding.
    let (sw, sh) = if w >= h {
        (s, ((h * s) / w).max(1))
    } else {
        (((w * s) / h).max(1), s)
    };
    // THE RESIZE, then the PAD, then the NORMALISATION -- in that order. The
    // pad is BLACK in the 8-BIT raster and the normalisation comes after, so
    // the border of the tensor is (0 - mean) / std.
    let scaled = resize_bilinear(src, sh, sw)?;
    let mut canvas = Image {
        width: s,
        height: s,
        rgb: vec![0u8; (s * s * 3) as usize],
    };
    for y in 0..sh {
        let from = (y * sw * 3) as usize;
        let to = (y * s * 3) as usize;
        let len = (sw * 3) as usize;
        canvas.rgb[to..to + len].copy_from_slice(&scaled.rgb[from..from + len]);
    }
    let blob = to_nchw_normalised_values(&canvas, &g.det_mean, &g.det_std)?;
    Ok(DetInput {
        blob,
        scale_x: sw as f64 / w as f64,
        scale_y: sh as f64 / h as f64,
        scaled_w: sw,
        scaled_h: sh,
    })
}

// ---------------------------------------------------------------------------
// The recogniser's front end
// ---------------------------------------------------------------------------

/// The recogniser's input, and the geometry of the crop it came from.
#[derive(Debug, Clone, Default)]
pub struct RecInput {
    /// `[3, H, W]`, normalised and padded.
    pub blob: Vec<f32>,
    /// The warped crop's size.
    pub crop_w: i64,
    /// The warped crop's size.
    pub crop_h: i64,
    /// The content's width inside the padded frame.
    pub content_w: i64,
    /// Whether the crop was turned a quarter turn.
    pub rotated: bool,
}

/// A box, as FOUR CORNERS in the order the recogniser wants them:
/// top-left, top-right, bottom-right, bottom-left.
///
/// The order is not cosmetic: the perspective warp that straightens the crop
/// maps the corners to the corners of a rectangle in that order, and the
/// crop's width comes from the two horizontal edges and its height from the
/// two vertical ones. A different order is a crop turned a quarter turn,
/// which the recogniser answers with a confident transcription of nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Quad {
    /// The corners' x.
    pub x: [f64; 4],
    /// The corners' y.
    pub y: [f64; 4],
}

/// The recogniser's input: the crop, straightened, squashed and padded.
pub fn rec_input(src: &Image, q: &Quad, g: &Geometry) -> Result<RecInput, String> {
    let mut out = RecInput::default();
    if src.empty() {
        return Err("the recogniser's front end: the image is empty".into());
    }
    // THE CROP'S SIZE: the longer of the two horizontal edges and the longer
    // of the two vertical ones. The corners are in the order TOP-LEFT,
    // TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT, so the horizontal edges are (0,1)
    // and (2,3) and the vertical ones are (0,3) and (1,2).
    let e01 = (q.x[1] - q.x[0]).hypot(q.y[1] - q.y[0]);
    let e23 = (q.x[3] - q.x[2]).hypot(q.y[3] - q.y[2]);
    let e03 = (q.x[3] - q.x[0]).hypot(q.y[3] - q.y[0]);
    let e12 = (q.x[2] - q.x[1]).hypot(q.y[2] - q.y[1]);
    let cw = (e01.max(e23) as i64).max(1);
    let ch = (e03.max(e12) as i64).max(1);
    out.crop_w = cw;
    out.crop_h = ch;

    // THE PERSPECTIVE WARP. The homography is asked for in the direction the
    // resampling wants it -- from the destination rectangle to the source
    // quad -- rather than computed forwards and inverted, which is the same
    // thing without a matrix inversion.
    let rect = [0.0, 0.0, cw as f64, 0.0, cw as f64, ch as f64, 0.0, ch as f64];
    let quad = [
        q.x[0], q.y[0], q.x[1], q.y[1], q.x[2], q.y[2], q.x[3], q.y[3],
    ];
    let mut m = [0.0f64; 9];
    if !get_perspective_transform(&rect, &quad, &mut m) {
        return Err(
            "the recogniser's front end: the box's four corners do not determine a \
             homography. Three of them are collinear or two of them are the same point, and \
             the crop of such a box is not an image."
                .into(),
        );
    }
    let mut crop = warp_perspective(src, &m, ch, cw)?;

    // THE TURN, for a line of text that goes UP the frame.
    if ch as f64 / cw as f64 >= 1.5 {
        crop = rot90(&crop);
        out.rotated = true;
    }

    // THE RESIZE: the content is squashed to the container's height, and its
    // width follows the crop's own aspect ratio -- FLOORED, which is what
    // PaddleOCR's own `resize_norm_img` does.
    let height = g.rec_input_height;
    let width = g.rec_input_width;
    let ratio = crop.width as f64 / (crop.height.max(1)) as f64;
    let mut rw = (height as f64 * ratio) as i64;
    rw = rw.max(1).min(width);
    out.content_w = rw;

    let scaled = resize_bilinear(&crop, height, rw)?;
    // THE NORMALISATION, then the PAD -- in that order, and the pad is in the
    // NORMALISED space.
    let blob = to_nchw_normalised_values(&scaled, &g.rec_mean, &g.rec_std)?;
    let plane = (height * width) as usize;
    let mut padded = vec![g.rec_pad_value as f32; 3 * plane];
    for c in 0..3usize {
        for y in 0..height {
            for x in 0..rw {
                padded[c * plane + (y * width + x) as usize] =
                    blob[c * (height * rw) as usize + (y * rw + x) as usize];
            }
        }
    }
    out.blob = padded;
    Ok(out)
}

/// A crop turned a quarter turn, COUNTER-CLOCKWISE, which is what numpy's
/// rot90 does: `B[j][i] = A[i][W-1-j]`.
///
/// The crop is turned when it is TALL -- a height at least one and a half
/// times its width -- because that is a line of text going UP the frame, and
/// the recogniser was trained on lines that go across.
fn rot90(src: &Image) -> Image {
    let mut out = Image {
        width: src.height,
        height: src.width,
        rgb: vec![0u8; (src.height * src.width * 3) as usize],
    };
    for j in 0..out.height {
        for i in 0..out.width {
            let sy = i;
            let sx = src.width - 1 - j;
            let s = (sy * src.width + sx) as usize;
            let d = (j * out.width + i) as usize;
            out.rgb[d] = src.rgb[s];
            out.rgb[d + 1] = src.rgb[s + 1];
            out.rgb[d + 2] = src.rgb[s + 2];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The resamplers
// ---------------------------------------------------------------------------

/// `cv2.getPerspectiveTransform(src, dst)`: the homography that maps the four
/// points `src` to the four points `dst`, as a 3x3 row-major matrix with
/// `m[8] == 1`.
///
/// `False` when the four points do not determine a homography -- three of them
/// collinear, or two of them the same -- which is a degenerate box and not a
/// rounding accident.
pub fn get_perspective_transform(src: &[f64; 8], dst: &[f64; 8], m: &mut [f64; 9]) -> bool {
    // The homography H maps (u, v) to
    //
    //     x = (a*u + b*v + c) / (g*u + h*v + 1)
    //     y = (d*u + e*v + f) / (g*u + h*v + 1)
    //
    // and the eight unknowns come from the four point pairs, each of which
    // gives two linear equations once the denominators are cleared.
    let mut a = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let u = src[2 * i];
        let v = src[2 * i + 1];
        let x = dst[2 * i];
        let y = dst[2 * i + 1];
        a[2 * i][0] = u;
        a[2 * i][1] = v;
        a[2 * i][2] = 1.0;
        a[2 * i][6] = -u * x;
        a[2 * i][7] = -v * x;
        a[2 * i][8] = x;
        a[2 * i + 1][3] = u;
        a[2 * i + 1][4] = v;
        a[2 * i + 1][5] = 1.0;
        a[2 * i + 1][6] = -u * y;
        a[2 * i + 1][7] = -v * y;
        a[2 * i + 1][8] = y;
    }
    // GAUSSIAN ELIMINATION WITH PARTIAL PIVOTING. The pivots are found
    // rather than assumed: a degenerate box -- three corners collinear, which
    // a very thin rectangle is -- makes a pivot zero, and dividing by it is
    // how a crop becomes a picture of NaNs.
    for col in 0..8 {
        let mut piv = col;
        for r in col + 1..8 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return false;
        }
        if piv != col {
            a.swap(col, piv);
        }
        for r in 0..8 {
            if r == col {
                continue;
            }
            let f = a[r][col] / a[col][col];
            if f == 0.0 {
                continue;
            }
            for k in col..9 {
                a[r][k] -= f * a[col][k];
            }
        }
    }
    for i in 0..8 {
        m[i] = a[i][8] / a[i][i];
    }
    m[8] = 1.0;
    true
}

/// `cv2.warpPerspective(src, m, (w, h), INTER_LINEAR, border=REPLICATE)`, where
/// `m` maps the DESTINATION to the SOURCE.
///
/// THE BORDER IS REPLICATED, not zero: the crop's corners reach outside the
/// image when the box sits at the edge of the frame, and a black border there
/// is a black stripe across the first character.
pub fn warp_perspective(src: &Image, m: &[f64; 9], dh: i64, dw: i64) -> Result<Image, String> {
    if src.empty() {
        return Err("warp_perspective: the image is empty".into());
    }
    let mut out = Image {
        width: dw,
        height: dh,
        rgb: vec![0u8; (dh * dw * 3) as usize],
    };
    let (a, b, c) = (m[0], m[1], m[2]);
    let (d, e, f) = (m[3], m[4], m[5]);
    let (g, h) = (m[6], m[7]);
    for y in 0..dh {
        for x in 0..dw {
            let den = g * x as f64 + h * y as f64 + 1.0;
            if den == 0.0 {
                continue;
            }
            let fx = (a * x as f64 + b * y as f64 + c) / den;
            let fy = (d * x as f64 + e * y as f64 + f) / den;
            let at = ((y * dw + x) * 3) as usize;
            blend_at(src, fy, fx, &mut out.rgb[at..at + 3]);
        }
    }
    Ok(out)
}

/// A bilinear sample at float coordinates, CLAMPING at the border.
///
/// The four taps are gathered first and blended once per channel.
fn blend_at(src: &Image, fy: f64, fx: f64, dst: &mut [u8]) {
    let (h, w) = (src.height, src.width);
    let fy0f = fy.floor();
    let fx0f = fx.floor();
    let (y0, x0) = (fy0f as i64, fx0f as i64);
    let wy = fy - fy0f;
    let wx = fx - fx0f;
    let y0c = y0.clamp(0, h - 1);
    let y1c = (y0 + 1).clamp(0, h - 1);
    let x0c = x0.clamp(0, w - 1);
    let x1c = (x0 + 1).clamp(0, w - 1);
    let at = |y: i64, x: i64| -> [f64; 3] {
        let s = ((y * w + x) * 3) as usize;
        [
            src.rgb[s] as f64,
            src.rgb[s + 1] as f64,
            src.rgb[s + 2] as f64,
        ]
    };
    let a = at(y0c, x0c);
    let b = at(y0c, x1c);
    let c = at(y1c, x0c);
    let d = at(y1c, x1c);
    let owx = 1.0 - wx;
    let owy = 1.0 - wy;
    for k in 0..3 {
        let top = a[k].mul_add(owx, b[k] * wx);
        let bot = c[k].mul_add(owx, d[k] * wx);
        dst[k] = to_u8(top.mul_add(owy, bot * wy));
    }
}

// ---------------------------------------------------------------------------
// The DB post-process
// ---------------------------------------------------------------------------

/// A box and its score.
#[derive(Debug, Clone, Copy, Default)]
pub struct DbBox {
    /// The four corners.
    pub q: Quad,
    /// The mean probability inside the box.
    pub score: f64,
}

/// A point.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Pt {
    x: f64,
    y: f64,
}

/// The minimum-area rectangle's frame.
#[derive(Debug, Clone, Copy)]
struct Rect {
    /// The centre.
    cx: f64,
    cy: f64,
    /// The unit vector along the width.
    ux: f64,
    uy: f64,
    /// The unit vector along the height.
    vx: f64,
    vy: f64,
    /// The half extents.
    hw: f64,
    hh: f64,
}

impl Default for Rect {
    fn default() -> Self {
        Rect {
            cx: 0.0,
            cy: 0.0,
            ux: 1.0,
            uy: 0.0,
            vx: 0.0,
            vy: 1.0,
            hw: 0.0,
            hh: 0.0,
        }
    }
}

/// The CONVEX HULL, by Andrew's monotone chain.
///
/// The minimum-area rectangle is determined by the hull -- a point inside the
/// hull cannot be a corner of it -- so the hull is what the rotating calipers
/// below walks.
fn convex_hull(mut p: Vec<Pt>) -> Vec<Pt> {
    p.sort_by(|a, b| {
        if a.x != b.x {
            a.x.partial_cmp(&b.x).unwrap()
        } else {
            a.y.partial_cmp(&b.y).unwrap()
        }
    });
    p.dedup_by(|a, b| a.x == b.x && a.y == b.y);
    let n = p.len();
    if n < 3 {
        return p;
    }
    let mut h: Vec<Pt> = vec![Pt::default(); 2 * n];
    let mut k = 0usize;
    for i in 0..n {
        while k >= 2
            && (h[k - 1].x - h[k - 2].x) * (p[i].y - h[k - 2].y)
                - (h[k - 1].y - h[k - 2].y) * (p[i].x - h[k - 2].x)
                <= 0.0
        {
            k -= 1;
        }
        h[k] = p[i];
        k += 1;
    }
    // THE UPPER HULL, walking the points from the right back to the left.
    //
    // The point that goes into the CROSS PRODUCT is the one that is about to
    // be pushed, and not its neighbour: the test asks whether the last two
    // points of the chain and THIS one make a left turn, and "this one" is the
    // point being added. Reading the neighbour instead is an off-by-one that
    // costs the hull its lower edge -- the hull then describes the top of a
    // text line only, the minimum-area rectangle is half as tall as it should
    // be, and the box that comes out of it is a strip through the middle of
    // the glyphs.
    let mut i = n - 1;
    let t = k + 1;
    while i > 0 {
        let q = p[i - 1];
        while k >= t
            && (h[k - 1].x - h[k - 2].x) * (q.y - h[k - 2].y)
                - (h[k - 1].y - h[k - 2].y) * (q.x - h[k - 2].x)
                <= 0.0
        {
            k -= 1;
        }
        h[k] = q;
        k += 1;
        i -= 1;
    }
    h.truncate(k.saturating_sub(1));
    h
}

/// THE MINIMUM-AREA RECTANGLE, by rotating calipers over the hull.
///
/// For each hull EDGE, the rectangle that has that edge as one side: the
/// extent along the edge is the projection's range, and the extent across it
/// is the largest distance any point reaches from the edge's line.
fn min_area_rect(pts: &[Pt]) -> Rect {
    let mut best = Rect::default();
    let hull = convex_hull(pts.to_vec());
    if hull.len() < 3 {
        // ONE POINT or a segment: the rectangle is degenerate, and it is zero
        // wide or zero tall -- which the min_size test below rejects.
        if !hull.is_empty() {
            best.cx = hull[0].x;
            best.cy = hull[0].y;
        }
        if hull.len() == 2 {
            best.hw = 0.5 * (hull[1].x - hull[0].x).hypot(hull[1].y - hull[0].y);
        }
        return best;
    }
    let mut best_area = 1e300f64;
    let n = hull.len();
    for i in 0..n {
        let a = hull[i];
        let b = hull[(i + 1) % n];
        let ex = b.x - a.x;
        let ey = b.y - a.y;
        let len = ex.hypot(ey);
        if len <= 0.0 {
            continue;
        }
        let ux = ex / len;
        let uy = ey / len; // along the edge
        let vx = -uy;
        let vy = ux; // across it
        let mut tmin = 1e300f64;
        let mut tmax = -1e300f64;
        let mut smin = 1e300f64;
        let mut smax = -1e300f64;
        for p in &hull {
            let t = p.x * ux + p.y * uy;
            let s = p.x * vx + p.y * vy;
            tmin = tmin.min(t);
            tmax = tmax.max(t);
            smin = smin.min(s);
            smax = smax.max(s);
        }
        let area = (tmax - tmin) * (smax - smin);
        if area < best_area {
            best_area = area;
            let tm = 0.5 * (tmin + tmax);
            let sm = 0.5 * (smin + smax);
            best.cx = tm * ux + sm * vx;
            best.cy = tm * uy + sm * vy;
            best.ux = ux;
            best.uy = uy;
            best.vx = vx;
            best.vy = vy;
            best.hw = 0.5 * (tmax - tmin);
            best.hh = 0.5 * (smax - smin);
        }
    }
    best
}

/// THE ORDER OF THE FOUR CORNERS, and it is PaddleOCR's.
///
/// Sort the corners by x; the two left-most are the LEFT pair and the two
/// right-most the RIGHT pair; within each pair the one with the SMALLER y is
/// the upper one -- and in an IMAGE, y grows downwards, so the smaller y is
/// the top.
///
/// The result is `[TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT]`.
fn order_corners(c: &[Pt; 4]) -> [usize; 4] {
    let mut order = [0usize, 1, 2, 3];
    order.sort_by(|a, b| {
        let (a, b) = (&c[*a], &c[*b]);
        if a.x != b.x {
            a.x.partial_cmp(&b.x).unwrap()
        } else {
            a.y.partial_cmp(&b.y).unwrap()
        }
    });
    let (l0, l1, r0, r1) = (order[0], order[1], order[2], order[3]);
    let tl = if c[l0].y <= c[l1].y { l0 } else { l1 };
    let bl = if tl == l0 { l1 } else { l0 };
    let tr = if c[r0].y <= c[r1].y { r0 } else { r1 };
    let br = if tr == r0 { r1 } else { r0 };
    [tl, tr, br, bl]
}

/// The rectangle's four corners, in the order the recogniser wants.
fn rect_corners(r: &Rect) -> Quad {
    let cu = [-1.0, -1.0, 1.0, 1.0];
    let cv = [-1.0, 1.0, 1.0, -1.0];
    let mut c = [Pt::default(); 4];
    for k in 0..4 {
        let t = cu[k] * r.hw;
        let s = cv[k] * r.hh;
        c[k].x = r.cx + t * r.ux + s * r.vx;
        c[k].y = r.cy + t * r.uy + s * r.vy;
    }
    let pick = order_corners(&c);
    let mut q = Quad::default();
    for k in 0..4 {
        q.x[k] = c[pick[k]].x;
        q.y[k] = c[pick[k]].y;
    }
    q
}

/// The mean probability INSIDE a convex quad, over the quad's axis-aligned
/// bounding box.
///
/// The pixel centres are tested against the quad's four edges, which is exact
/// for a convex polygon and is what a scanline fill computes.
fn box_score(pred: &[f32], h: i64, w: i64, q: &Quad) -> f64 {
    let mut xmin = q.x[0];
    let mut xmax = q.x[0];
    let mut ymin = q.y[0];
    let mut ymax = q.y[0];
    for k in 1..4 {
        xmin = xmin.min(q.x[k]);
        xmax = xmax.max(q.x[k]);
        ymin = ymin.min(q.y[k]);
        ymax = ymax.max(q.y[k]);
    }
    let mut x0 = xmin.floor() as i64;
    let mut x1 = xmax.ceil() as i64;
    let mut y0 = ymin.floor() as i64;
    let mut y1 = ymax.ceil() as i64;
    x0 = x0.clamp(0, w - 1);
    x1 = x1.clamp(0, w - 1);
    y0 = y0.clamp(0, h - 1);
    y1 = y1.clamp(0, h - 1);
    let mut sum = 0.0f64;
    let mut n = 0i64;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let px = x as f64 + 0.5;
            let py = y as f64 + 0.5;
            // INSIDE, by the sign of the four cross products. The polygon is
            // convex and wound the same way, so all four have the same sign
            // inside and the test is a comparison of products.
            let mut sign = 0.0f64;
            let mut inside = true;
            for k in 0..4 {
                let k2 = (k + 1) & 3;
                let cr = (q.x[k2] - q.x[k]) * (py - q.y[k])
                    - (q.y[k2] - q.y[k]) * (px - q.x[k]);
                if k == 0 {
                    sign = cr;
                } else if (cr > 0.0) != (sign > 0.0) && cr != 0.0 {
                    inside = false;
                    break;
                }
            }
            if !inside {
                continue;
            }
            sum += pred[(y * w + x) as usize] as f64;
            n += 1;
        }
    }
    if n == 0 {
        return 0.0;
    }
    sum / n as f64
}

/// The DB post-process: a probability map in, boxes out.
///
/// `pred` is the probability map, `[H, W]` row-major -- the detector's output
/// after the sigmoid, which the graph has already applied.
///
/// `scale_x` and `scale_y` are how many map pixels there are in one pixel of
/// the image: the detector saw the image letterboxed into the frame, with its
/// proportions kept, so the map is the image blown up by that factor and the
/// boxes are scaled back down by it.
pub fn db_postprocess(
    pred: &[f32],
    h: i64,
    w: i64,
    thresh: f64,
    box_thresh: f64,
    unclip_ratio: f64,
    max_candidates: i64,
    min_size: i64,
    scale_x: f64,
    scale_y: f64,
    dest_w: f64,
    dest_h: f64,
) -> Vec<DbBox> {
    let mut out: Vec<DbBox> = Vec::new();
    if h <= 0 || w <= 0 {
        return out;
    }

    // -- 1. the mask
    let mut mask = vec![0u8; (h * w) as usize];
    for i in 0..(h * w) as usize {
        mask[i] = if pred[i] as f64 > thresh { 1 } else { 0 };
    }

    // -- 2. the connected components, 8-connected, in raster order
    //
    // The raster order is the order cv2.findContours returns them in, closely
    // enough: it matters only when there are more than max_candidates blobs,
    // and then it decides WHICH of them is kept.
    let mut label = vec![-1i32; (h * w) as usize];
    let mut blobs: Vec<Vec<Pt>> = Vec::new();
    let mut stack: Vec<i32> = Vec::new();
    for y0 in 0..h {
        for x0 in 0..w {
            let seed = (y0 * w + x0) as usize;
            if mask[seed] == 0 || label[seed] >= 0 {
                continue;
            }
            let id = blobs.len() as i32;
            blobs.push(Vec::new());
            stack.clear();
            stack.push(seed as i32);
            label[seed] = id;
            while let Some(cur) = stack.pop() {
                let cy = cur / w as i32;
                let cx = cur % w as i32;
                blobs[id as usize].push(Pt {
                    x: cx as f64,
                    y: cy as f64,
                });
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let ny = cy + dy;
                        let nx = cx + dx;
                        if ny < 0 || ny >= h as i32 || nx < 0 || nx >= w as i32 {
                            continue;
                        }
                        let ni = (ny * w as i32 + nx) as usize;
                        if mask[ni] == 0 || label[ni] >= 0 {
                            continue;
                        }
                        label[ni] = id;
                        stack.push(ni as i32);
                    }
                }
            }
        }
    }

    // -- 3. each blob's rectangle, its score, and the un-clip
    let take = (blobs.len() as i64).min(max_candidates);
    for b in 0..take {
        let r = min_area_rect(&blobs[b as usize]);
        // THE SMALLEST SIDE, and it is the smaller of the two: a blob that is
        // three pixels tall is noise, and a blob that is three pixels wide is a
        // single character -- which is the same test PaddleOCR makes.
        let min_side = (2.0 * r.hw).min(2.0 * r.hh);
        if min_side < min_size as f64 {
            continue;
        }
        let q = rect_corners(&r);
        let score = box_score(pred, h, w, &q);
        if box_thresh > score {
            continue;
        }

        // THE UN-CLIP. The distance is the rectangle's own area times the ratio
        // over its perimeter -- PaddleOCR's `distance = poly.area * unclip_ratio
        // / poly.length`, and the rectangle's area is w*h and its length is
        // 2*(w+h). For a 100x20 box and a ratio of 1.5 that is 12.5 pixels.
        //
        // The expansion of a RECTANGLE by a distance d, with round joins, is
        // the rectangle grown by d on every side -- the rounded corners do not
        // reach past the grown edges -- so the minimum-area rectangle of the
        // expanded polygon is the grown rectangle.
        let pw = 2.0 * r.hw;
        let ph = 2.0 * r.hh;
        let per = 2.0 * (pw + ph);
        let d = if per > 0.0 {
            (pw * ph * unclip_ratio) / per
        } else {
            0.0
        };
        let nhw = r.hw + d;
        let nhh = r.hh + d;
        if (2.0 * nhw).min(2.0 * nhh) < min_size as f64 + 2.0 {
            continue;
        }

        // The grown rectangle's corners, in the same order.
        let cu = [-1.0, -1.0, 1.0, 1.0];
        let cv = [-1.0, 1.0, 1.0, -1.0];
        let mut c = [Pt::default(); 4];
        for k in 0..4 {
            let t = cu[k] * nhw;
            let s = cv[k] * nhh;
            c[k].x = r.cx + t * r.ux + s * r.vx;
            c[k].y = r.cy + t * r.uy + s * r.vy;
        }
        let pick = order_corners(&c);

        let mut box_ = DbBox {
            q: Quad::default(),
            score,
        };
        for k in 0..4 {
            // INTO THE IMAGE'S COORDINATES: the map's pixels to the image's,
            // divided by the scale the front end scaled the image by, rounded
            // half to even and clipped to the image.
            //
            // The scale is the front end's, and not the map's size over the
            // image's: the image was letterboxed into the frame with its
            // proportions kept, so the map's frame is BIGGER than the image
            // and the text sits in a corner of it.
            let mut bx = c[pick[k]].x / scale_x;
            let mut by = c[pick[k]].y / scale_y;
            bx = bx.round_ties_even();
            by = by.round_ties_even();
            box_.q.x[k] = bx.clamp(0.0, dest_w);
            box_.q.y[k] = by.clamp(0.0, dest_h);
        }
        out.push(box_);
    }
    out
}

// ---------------------------------------------------------------------------
// The CTC decode
// ---------------------------------------------------------------------------

/// What the CTC decode found.
#[derive(Debug, Clone, Default)]
pub struct CtcResult {
    /// The text.
    pub text: String,
    /// The mean confidence over the kept frames.
    pub conf: f64,
}

/// The CTC greedy decode.
///
/// The blank is class 0, the repeats are collapsed against the previous
/// frame's RAW index, and the confidence is the mean over the KEPT frames.
pub fn ctc_decode(logits: &[f32], frames: i64, classes: i64, dict: &[String]) -> Result<CtcResult, String> {
    let mut r = CtcResult::default();
    if classes <= 0 || frames <= 0 {
        return Ok(r);
    }
    if dict.len() as i64 != classes {
        return Err(format!(
            "ctc: the character table holds {} classes and the recogniser emitted {classes} \
             logits per frame. The table is the dictionary's characters plus the blank and \
             the space, and a decoder that reads one and a network that learned another \
             produce text that is confidently wrong.",
            dict.len()
        ));
    }
    let mut text = String::new();
    let mut sum = 0.0f64;
    let mut n = 0i64;
    let mut prev: i64 = -1;
    for t in 0..frames {
        let row = &logits[(t * classes) as usize..][..classes as usize];
        let mut best = 0i64;
        let mut bv = row[0];
        for k in 1..classes as usize {
            if row[k] > bv {
                bv = row[k];
                best = k as i64;
            }
        }
        // THE BLANK IS CLASS 0, and it is not a character.
        if best == 0 {
            // The previous index is still remembered: a repeat is judged
            // against the frame BEFORE, blank or not.
            prev = best;
            continue;
        }
        // THE REPEATS ARE COLLAPSED against the previous frame's RAW index. A
        // character repeated twice is one character, and the second one is
        // dropped.
        if best == prev {
            prev = best;
            continue;
        }
        prev = best;
        text.push_str(&dict[best as usize]);
        sum += bv as f64;
        n += 1;
    }
    // The confidence is the mean over the KEPT frames, and an empty text has a
    // confidence of zero rather than a division by zero.
    r.text = text;
    r.conf = if n > 0 { sum / n as f64 } else { 0.0 };
    Ok(r)
}

/// One box and the text read out of it.
#[derive(Debug, Clone, Default)]
pub struct TextBox {
    /// The four corners, in the image's coordinates.
    pub q: Quad,
    /// The DB score: the mean probability inside.
    pub box_score: f64,
    /// The text.
    pub text: String,
    /// The CTC confidence.
    pub conf: f64,
}

/// The result of one frame.
#[derive(Debug, Clone, Default)]
pub struct OcrResult {
    /// The frame's width.
    pub width: i64,
    /// The frame's height.
    pub height: i64,
    /// The boxes that were read.
    pub boxes: Vec<TextBox>,
    /// How many boxes the detector found.
    pub boxes_found: i64,
    /// How many boxes reached the recogniser.
    pub boxes_read: i64,
    /// The whole run.
    pub total_s: f64,
    /// The detector's front end.
    pub front_end_s: f64,
    /// The detector's graph.
    pub det_s: f64,
    /// The DB post-process.
    pub post_s: f64,
    /// The crops.
    pub crop_s: f64,
    /// The recogniser's graph.
    pub rec_s: f64,
}

// ---------------------------------------------------------------------------
// The two answers
// ---------------------------------------------------------------------------

/// The human summary: one line per box.
///
/// The oracle's line is `"%s: box %zu  score %.3f  conf %.3f  \"%s\""` -- two
/// spaces before the score and two before the confidence -- and a frame with
/// no boxes in it says so instead of printing nothing.
pub fn text_summary(image: &str, r: &OcrResult) -> String {
    if r.boxes.is_empty() {
        return format!("{image}: no text\n");
    }
    let mut s = String::new();
    for (i, b) in r.boxes.iter().enumerate() {
        s.push_str(&format!(
            "{image}: box {i}  score {:.3}  conf {:.3}  \"{}\"\n",
            b.box_score, b.conf, b.text
        ));
    }
    s
}

/// The whole answer, as the JSON object on stdout.
///
/// The path is printed RAW, because the oracle prints it raw: an escaping that
/// the oracle does not do is a difference, and a difference is what this gate
/// exists to catch. The TEXT, on the other hand, is escaped -- it comes out of
/// a dictionary of characters, and a quote in it is a quote in the output.
pub fn result_json(r: &OcrResult, image: &str) -> String {
    let mut out = String::new();
    out.push_str("{\n  \"image\": \"");
    out.push_str(image);
    out.push_str(&format!(
        "\",\n  \"width\": {},\n  \"height\": {},\n",
        r.width, r.height
    ));
    out.push_str(&format!(
        "  \"timings_ms\": {{\"total\": {:.3}, \"front_end\": {:.3}, \"detector\": {:.3}, \
         \"db\": {:.3}, \"crop\": {:.3}, \"recogniser\": {:.3}}},\n",
        r.total_s * 1e3,
        r.front_end_s * 1e3,
        r.det_s * 1e3,
        r.post_s * 1e3,
        r.crop_s * 1e3,
        r.rec_s * 1e3
    ));
    out.push_str("  \"boxes\": [");
    for (i, b) in r.boxes.iter().enumerate() {
        out.push_str(if i > 0 { ",\n    " } else { "\n    " });
        out.push_str(&format!(
            "{{\"score\": {:.6}, \"conf\": {:.6}, \"text\": \"",
            b.box_score, b.conf
        ));
        for ch in b.text.chars() {
            match ch {
                '"' | '\\' => {
                    out.push('\\');
                    out.push(ch);
                }
                '\n' => out.push_str("\\n"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => {
                    out.push_str(&format!("\\u{:04x}", c as u32));
                }
                c => out.push(c),
            }
        }
        out.push_str("\", \"quad\": [");
        for k in 0..4 {
            out.push_str(if k > 0 { ", " } else { "" });
            out.push_str(&format!("[{:.3}, {:.3}]", b.q.x[k], b.q.y[k]));
        }
        out.push_str("]}");
    }
    out.push_str(if r.boxes.is_empty() {
        "]}\n"
    } else {
        "\n  ]}\n"
    });
    out
}

/// The recogniser's output, as the CTC head leaves it.
pub fn ctc_logits(output: &Tensor) -> Result<(usize, usize), String> {
    if output.rank() != 3 {
        return Err(format!(
            "the recogniser's output is rank {} and the CTC decoder reads a rank-3 \
             [1, frames, classes] tensor.",
            output.rank()
        ));
    }
    Ok((output.shape[1] as usize, output.shape[2] as usize))
}
