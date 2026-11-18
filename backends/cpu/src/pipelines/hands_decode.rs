//! The arch=7 front end and decode: the four cv2 resamplers, the letterbox,
//! the 2016-row palm decode, one rotated landmark crop, and the inverse.
//!
//! # FOUR RESAMPLERS, AND CV2 NAMES ALL FOUR
//!
//! * **the letterbox** is `INTER_LINEAR` with no antialiasing. Using the area
//!   average for it is the tempting "better" choice and it is a different
//!   detector: the zoo shipped with a bilinear letterbox and this checkpoint's
//!   edge responses were measured against it.
//! * **the landmark crop's resize** is `INTER_AREA`, an exact box average.
//!   Bilinear there moves the fingertips by a pixel and a half on a downscaled
//!   crop.
//! * **the rotation** is `warpAffine`, also `INTER_LINEAR`, border 0, and its
//!   matrix maps SOURCE to DESTINATION so the resampling uses the INVERSE.
//!   Using M directly rotates the hand the wrong way and the landmark network
//!   absorbs it well enough to still return a plausible hand.
//! * **the crop** is a half-open rectangle copy, and it is the caller's job to
//!   have clipped first.
//!
//! # THE RASTER IS 8-BIT AND THE ACCUMULATION IS NOT
//!
//! Every resampler returns an 8-bit raster, so the letterboxed palm input is
//! rounded back to 8 bits before the network sees it -- which is what cv2 does
//! and what the zoo does. The rounding is `nearbyint`: ties to EVEN, not away
//! from zero. It differs from the other tie rule by one 8-bit level on exact
//! halves and by nothing elsewhere, and it is stated because the front end is
//! measured in PIXELS rather than at tensor scale.
//!
//! The box average accumulates in f64 and converts ONCE, at the end: rounding
//! each tap's partial sum to 8 bits as it arrives quantises a box average to 8
//! bits per tap rather than per pixel, and with four taps that is a brightness
//! bias -- on exactly the downscaled crops every landmark stage sees.
//!
//! # WHERE THE MULTIPLY-ADD IS FUSED, AND WHERE IT IS NOT
//!
//! Read off the oracle's object file, because the difference is one rounding
//! per tap and this front end feeds 192x192 pixels into 87 nodes:
//!
//! | site | the oracle's instructions |
//! |---|---|
//! | `blend_at` | `vmulsd b*wx`, then `vfmadd a*(1-wx) + that` |
//! | `resize_area` | `vfmadd` on both passes |
//! | `letterbox` | no arithmetic of its own: it is the bilinear plus a zero pad |
//!
//! So every blend below is `a.mul_add(1 - w, b * w)`: the SECOND product is the
//! rounded one, which is what the compiler chose for the source's
//! `a*(1-wx) + b*wx` and what this port reproduces.
//!
//! # THE ROTATION IS PREDICTED, NOT CHOSEN
//!
//! MediaPipe does not try every angle. The palm detector emits seven keypoints,
//! two of which (recorded in the container as `palm_lm_wrist` and
//! `palm_lm_middle_base`) name the wrist and the base of the middle finger, and
//! the rotation is the one that puts THAT PAIR vertical. WHICH two matters: the
//! wrong pair rotates by some other angle -- usually not zero -- and the network
//! still returns 21 joints and a plausible hand, because a rotated hand is
//! still a hand.
//!
//! # THE CROP'S SHIFT MOVES BOTH CORNERS
//!
//! The demo writes the shift as `bbox + shift * wh` on a (2, 2) array, and a
//! (2,) broadcast along the LAST axis moves x1 and x2 together -- so the box's
//! SIZE is unchanged and only its centre moves. Applying it to the low corner
//! alone GROWS the box instead of moving it, and the symptom is uniform: no
//! landmark is wild, the whole hand is small.

use crate::hands::{Geometry, Hands, LmOut, PalmOut};
use crate::pose::Tensor;
use crate::vit_image::Image;

/// What one palm detection is, in the CALLER's pixels.
#[derive(Debug, Clone, Default)]
pub struct Detection {
    /// `x1, y1, x2, y2`.
    pub bbox: [f64; 4],
    /// The seven palm keypoints, x/y.
    pub kps: [[f64; 2]; 7],
    /// The sigmoid score.
    pub score: f64,
}

/// One hand, in the CALLER's pixels.
///
/// `landmarks` x and y are absolute pixels and z is a DEPTH RELATIVE TO THE
/// WRIST with the same pixel scale: a z of 30 is thirty pixels nearer than the
/// wrist, not thirty pixels of real distance. `world` is in METRES and
/// wrist-relative -- the metric frame the checkpoint regresses, which has no
/// relation to the frame's pixels at all.
#[derive(Debug, Clone, Default)]
pub struct Hand {
    /// The reported box.
    pub bbox: [f64; 4],
    /// 21 landmarks, `[x, y, z]`.
    pub landmarks: [[f64; 3]; 21],
    /// 21 world landmarks, `[x, y, z]` in metres.
    pub world: [[f64; 3]; 21],
    /// The sigmoid handedness; above 0.5 is a right hand.
    pub handedness: f64,
    /// The sigmoid presence.
    pub presence: f64,
}

/// The letterbox's transform, so the decode can take a detection back.
#[derive(Debug, Clone)]
pub struct Letterbox {
    /// Model pixels per source pixel.
    pub scale: f64,
    /// The left offset, in canvas pixels.
    pub left: i64,
    /// The top offset, in canvas pixels.
    pub top: i64,
    /// The same left offset in ORIGINAL pixels.
    pub pad_orig_x: i64,
    /// The same top offset in ORIGINAL pixels.
    pub pad_orig_y: i64,
}

/// The rotation crop's geometry, kept so the postprocess can undo it.
#[derive(Debug, Clone, Default)]
pub struct LandmarkStage {
    /// The `lm_size x lm_size` raster the landmark network sees.
    pub blob: Image,
    /// The SECOND crop's clipped box.
    pub rot_bbox: [f64; 4],
    /// The rotated palm keypoints' bounding box BEFORE the second crop grows
    /// it. Kept because the second crop's box is clipped to the raster and a
    /// clip is silent.
    pub palm_kp_box: [f64; 4],
    /// The rotation applied, in degrees.
    pub angle: f64,
    /// The same rotation as a 2x3, source -> destination.
    pub rot_m: [f64; 6],
    /// What the first crop's padding cost, in frame pixels.
    pub pad_bias: [f64; 2],
}

// ---------------------------------------------------------------------------
// The raster
// ---------------------------------------------------------------------------

/// An all-zero raster of this size.
fn like(h: i64, w: i64) -> Image {
    Image {
        width: w,
        height: h,
        rgb: vec![0u8; (h * w * 3).max(0) as usize],
    }
}

/// One pixel's three bytes.
fn px(im: &Image, y: i64, x: i64) -> &[u8] {
    let at = ((y * im.width + x) * 3) as usize;
    &im.rgb[at..at + 3]
}

/// One pixel, mutably.
fn px_mut(im: &mut Image, y: i64, x: i64) -> &mut [u8] {
    let at = ((y * im.width + x) * 3) as usize;
    &mut im.rgb[at..at + 3]
}

/// cv2's `saturate_cast<uchar>(double)`: round to nearest, ties to EVEN, then
/// clamp.
pub fn to_u8(v: f64) -> u8 {
    let r = v.round_ties_even();
    if r <= 0.0 {
        0
    } else if r >= 255.0 {
        255
    } else {
        r as u8
    }
}

/// A bilinear sample at float coordinates, CLAMPING at the border.
///
/// `warp_affine` masks the outside itself, because clamping and zeroing are
/// different answers and the zoo's border is 0.
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
    let a = px(src, y0c, x0c);
    let b = px(src, y0c, x1c);
    let c = px(src, y1c, x0c);
    let d = px(src, y1c, x1c);
    let owx = 1.0 - wx;
    let owy = 1.0 - wy;
    for k in 0..3 {
        let top = a[k].mul_add_f64(owx, b[k] as f64 * wx);
        let bot = c[k].mul_add_f64(owx, d[k] as f64 * wx);
        dst[k] = to_u8(top.mul_add_f64(owy, bot * wy));
    }
}

/// The multiply-add on two f64s.
///
/// A free function rather than a bare method call so that the FUSION is at the
/// call site and visible: `a.mul_add_f64(w, b)` is one rounding for the product
/// of `a` and `w`, exactly as the oracle's `vfmadd`.
trait MulAddF64 {
    /// `self * w + other`, fused.
    fn mul_add_f64(self, w: f64, other: f64) -> f64;
}

impl MulAddF64 for u8 {
    fn mul_add_f64(self, w: f64, other: f64) -> f64 {
        (self as f64).mul_add(w, other)
    }
}

impl MulAddF64 for f64 {
    fn mul_add_f64(self, w: f64, other: f64) -> f64 {
        self.mul_add(w, other)
    }
}

/// `cv2.INTER_LINEAR`, no antialiasing.
///
/// `dst(x)` reads `src` at `(x + 0.5) * scale - 0.5`, clamping at the border.
/// Off by half a source pixel it is not, and the result looks like a bad model
/// rather than a bad resample.
pub fn resize_bilinear(src: &Image, nh: i64, nw: i64) -> Result<Image, String> {
    if src.empty() || nh <= 0 || nw <= 0 {
        return Err(format!(
            "resize_bilinear: {nh}x{nw} from an empty or degenerate source"
        ));
    }
    let sy = src.height as f64 / nh as f64;
    let sx = src.width as f64 / nw as f64;
    let mut out = like(nh, nw);
    for y in 0..nh {
        let fy = (y as f64 + 0.5) * sy - 0.5;
        for x in 0..nw {
            let fx = (x as f64 + 0.5) * sx - 0.5;
            let d = px_mut(&mut out, y, x);
            blend_at(src, fy, fx, d);
        }
    }
    Ok(out)
}

/// Exact `INTER_AREA` weights for ONE axis, as (dst, src, weight) triples.
///
/// The coverage sum is corrected back onto the last tap: without that the
/// weights carry an f64 residue of a few 1e-17, which on a box of 49 pixels is
/// far below the 8-bit output. It is corrected anyway because the alternative
/// is a resampler whose weights provably do not sum to one.
struct AreaAxis {
    dst: Vec<i64>,
    src: Vec<i64>,
    w: Vec<f64>,
}

fn area_axis(n_src: i64, n_dst: i64) -> AreaAxis {
    let mut ax = AreaAxis {
        dst: Vec::new(),
        src: Vec::new(),
        w: Vec::new(),
    };
    let s = n_src as f64 / n_dst as f64;
    for i in 0..n_dst {
        let y0 = i as f64 * s;
        let y1 = (i + 1) as f64 * s;
        let mut acc = 0.0f64;
        let j_start = y0.floor() as i64;
        let j_end = (y1.ceil() as i64).min(n_src);
        for j in j_start..j_end {
            let lo = (j as f64).max(y0);
            let hi = ((j + 1) as f64).min(y1);
            if hi <= lo {
                continue;
            }
            ax.dst.push(i);
            ax.src.push(j);
            ax.w.push((hi - lo) / s);
            acc += (hi - lo) / s;
        }
        if let Some(last) = ax.w.last_mut() {
            if (acc - 1.0).abs() > 1e-12 {
                *last += 1.0 - acc;
            }
        }
    }
    ax
}

/// `cv2.INTER_AREA` -- an exact, separable box average.
///
/// NO SPECIAL CASE FOR UPSCALING, and that is deliberate: when the axis grows
/// the box is narrower than one source cell and the same loop degrades on its
/// own into a linear blend of the two cells it straddles -- which is what cv2
/// does, and not the nearest-neighbour its documentation claims.
pub fn resize_area(src: &Image, nh: i64, nw: i64) -> Result<Image, String> {
    if src.empty() || nh <= 0 || nw <= 0 {
        return Err(format!(
            "resize_area: {nh}x{nw} from an empty or degenerate source"
        ));
    }
    let ay = area_axis(src.height, nh);
    let ax = area_axis(src.width, nw);
    let sw = src.width;
    // Separable, with the intermediate kept in f64 at the DESTINATION height
    // and the SOURCE width: the tap list along the columns is what the second
    // pass walks.
    let mut rows = vec![0.0f64; (nh * sw * 3) as usize];
    for t in 0..ay.w.len() {
        let (di, sj, wgt) = (ay.dst[t], ay.src[t], ay.w[t]);
        let s = &src.rgb[(sj * sw * 3) as usize..][..(sw * 3) as usize];
        let d = &mut rows[(di * sw * 3) as usize..][..(sw * 3) as usize];
        for x in 0..(sw * 3) as usize {
            // Fused, as the oracle's `vfmadd` in both passes.
            d[x] = (s[x] as f64).mul_add(wgt, d[x]);
        }
    }
    let mut acc = vec![0.0f64; (nh * nw * 3) as usize];
    for y in 0..nh {
        let r = &rows[(y * sw * 3) as usize..][..(sw * 3) as usize];
        for t in 0..ax.w.len() {
            let (di, sj, wgt) = (ax.dst[t], ax.src[t], ax.w[t]);
            let s = &r[(sj * 3) as usize..][..3];
            let d = &mut acc[((y * nw + di) * 3) as usize..][..3];
            for k in 0..3 {
                d[k] = s[k].mul_add_f64(wgt, d[k]);
            }
        }
    }
    let mut out = like(nh, nw);
    for y in 0..nh {
        for x in 0..nw {
            let s = &acc[((y * nw + x) * 3) as usize..][..3];
            let d = px_mut(&mut out, y, x);
            for k in 0..3 {
                d[k] = to_u8(s[k]);
            }
        }
    }
    Ok(out)
}

/// `cv2.warpAffine(src, m, INTER_LINEAR, border=0)`.
///
/// `m` is 2x3 row-major and maps SOURCE to DESTINATION, so the resampling uses
/// the INVERSE. Destination pixels sampling outside the source are ZERO, not
/// clamped: clamping fills the corners with edge pixels and the network sees a
/// frame that is not there.
pub fn warp_affine(src: &Image, m: &[f64; 6]) -> Result<Image, String> {
    if src.empty() {
        return Err("warp_affine: the image is empty".into());
    }
    let (h, w) = (src.height, src.width);
    let (a, b, tx) = (m[0], m[1], m[2]);
    let (c, d, ty) = (m[3], m[4], m[5]);
    let det = a * d - b * c;
    if det == 0.0 {
        return Err(
            "warp_affine: the affine is singular, so it has no inverse and no \
             destination pixel has a source. The rotation here is built from a point \
             pair and a box, and a degenerate one is a zero-length box, not a rounding \
             accident."
                .into(),
        );
    }
    let ia = d / det;
    let ib = -b / det;
    let ic = -c / det;
    let id = a / det;
    // The inverse translation pairs ROW 0 of M against the ROW-0 inverse
    // entries; pairing it the other way round is a translation in the wrong
    // direction, which at a rotation of about pi/2 returns a black crop.
    let itx = -(ia * tx + ib * ty);
    let ity = -(ic * tx + id * ty);
    let mut out = like(h, w);
    for y in 0..h {
        for x in 0..w {
            let fx = ia * x as f64 + ib * y as f64 + itx;
            let fy = ic * x as f64 + id * y as f64 + ity;
            if fx < 0.0 || fx > (w - 1) as f64 || fy < 0.0 || fy > (h - 1) as f64 {
                continue; // border = 0
            }
            let dst = px_mut(&mut out, y, x);
            blend_at(src, fy, fx, dst);
        }
    }
    Ok(out)
}

/// A copy of the half-open rectangle. The clip is the CALLER's job.
pub fn crop(src: &Image, x0: i64, y0: i64, x1: i64, y1: i64) -> Result<Image, String> {
    if x0 < 0 || y0 < 0 || x1 <= x0 || y1 <= y0 {
        return Err(format!(
            "crop: {x0},{y0} to {x1},{y1} is not a non-empty rectangle inside an image"
        ));
    }
    if x1 > src.width || y1 > src.height {
        return Err(format!(
            "crop: the rectangle runs past the image's {}x{}",
            src.width, src.height
        ));
    }
    let mut out = like(y1 - y0, x1 - x0);
    for y in 0..out.height {
        let at = (((y0 + y) * src.width + x0) * 3) as usize;
        let len = (out.width * 3) as usize;
        let dst = &mut out.rgb[(y * out.width * 3) as usize..][..len];
        dst.copy_from_slice(&src.rgb[at..at + len]);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// The letterbox and the normalisation
// ---------------------------------------------------------------------------

/// Short side to `side x side`, uniform scale, then pad with ZEROS.
///
/// The pad value is 0 and not the grey 114 the pose front end uses, because
/// this is not the same model: MediaPipe's detectors were trained on
/// letterboxes filled with black, and a grey border moves the pyramid's edge
/// responses and with them the keypoints nearest the margin.
pub fn letterbox(src: &Image, side: i64) -> Result<(Image, Letterbox), String> {
    if src.empty() {
        return Err("letterbox: the image is empty".into());
    }
    let (h, w) = (src.height, src.width);
    let r = (side as f64 / h as f64).min(side as f64 / w as f64);
    // Truncated toward zero, which for these positive values is a floor and is
    // what cv2 and numpy both do when they take an int of a float. Rounding
    // instead makes the padded canvas one row taller on some inputs and shifts
    // every detection by a fraction of its own size.
    let rs_h = (h as f64 * r) as i64;
    let rs_w = (w as f64 * r) as i64;
    if rs_h < 1 || rs_w < 1 {
        return Err(format!(
            "letterbox: a {h}x{w} image scales to {rs_h}x{rs_w} at side {side}"
        ));
    }
    let small = resize_bilinear(src, rs_h, rs_w)?;
    let ph = side - rs_h;
    let pw = side - rs_w;
    let left = pw / 2;
    let top = ph / 2;
    let mut out = like(side, side);
    for y in 0..side {
        let sy = y - top;
        if sy < 0 || sy >= rs_h {
            continue;
        }
        for x in 0..side {
            let sx = x - left;
            if sx < 0 || sx >= rs_w {
                continue;
            }
            let s = px(&small, sy, sx);
            let d = px_mut(&mut out, y, x);
            d.copy_from_slice(s);
        }
    }
    let lb = Letterbox {
        scale: r,
        left,
        top,
        // The same offset in ORIGINAL pixels, which is the unit a detection is
        // corrected in: the head's outputs are fractions of the input side and
        // are scaled back by the ORIGINAL long side.
        pad_orig_x: (left as f64 / r) as i64,
        pad_orig_y: (top as f64 / r) as i64,
    };
    Ok((out, lb))
}

/// HWC u8 -> `[3, H, W]` f32 as `(u8/255 - mean) / std`.
///
/// The `std` check is here, once, rather than inside the loop: a divide by zero
/// is a channel that reaches the first convolution as a NaN rather than as an
/// error.
pub fn to_nchw_normalised(src: &Image, mean: &[f32], std_dev: &[f32]) -> Result<Tensor, String> {
    let d = to_nchw_normalised_values(src, mean, std_dev)?;
    Ok(Tensor {
        c: 3,
        h: src.height,
        w: src.width,
        d,
    })
}

/// The same normalisation, as a flat buffer.
///
/// The OCR family's tensors are not all NCHW -- the recogniser's attention
/// block has rank-5 ones -- so this returns the VALUES and lets the caller
/// wrap them in whatever shape it needs.
pub fn to_nchw_normalised_values(
    src: &Image,
    mean: &[f32],
    std_dev: &[f32],
) -> Result<Vec<f32>, String> {
    if src.empty() {
        return Err("to_nchw_normalised: the image is empty".into());
    }
    if mean.len() != 3 || std_dev.len() != 3 {
        return Err(format!(
            "to_nchw_normalised: the container declares {} means and {} standard \
             deviations, and this architecture has three channels",
            mean.len(),
            std_dev.len()
        ));
    }
    for c in 0..3 {
        if std_dev[c] == 0.0 {
            return Err(format!(
                "to_nchw_normalised: image_std[{c}] is zero, so the normalisation is \
                 not invertible and no input can be recovered"
            ));
        }
    }
    let plane = src.height * src.width;
    let mut out = vec![0.0f32; (3 * plane) as usize];
    for c in 0..3 {
        let mu = mean[c] as f64;
        let sd = std_dev[c] as f64;
        let base = (c as i64 * plane) as usize;
        for p in 0..plane as usize {
            // The double arithmetic is the oracle's: it divides the byte by
            // 255.0 in f64, subtracts the f64 mean and divides by the f64 std,
            // and converts ONCE to f32.
            let v = src.rgb[p * 3 + c] as f64 / 255.0;
            out[base + p] = ((v - mu) / sd) as f32;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// The stages
// ---------------------------------------------------------------------------

/// A cropped and squared raster, and what the crop did to the coordinates.
#[derive(Debug, Clone)]
struct CropPad {
    img: Image,
    /// The CLIPPED rectangle, unmoved by the pad.
    bbox: [f64; 4],
    /// Frame pixels the padding and crop moved by.
    bias: [f64; 2],
}

/// numpy's `astype(np.int32)` on a float: TRUNCATION toward zero.
fn trunc_i32(v: f64) -> i64 {
    if v < 0.0 {
        v.ceil() as i64
    } else {
        v.floor() as i64
    }
}

/// The demo's crop: shift by a fraction of the box, enlarge about the shifted
/// centre, truncate and clip to the frame, then square.
///
/// SIDE IS THE DIAGONAL FOR THE ROTATION CROP AND THE LONG SIDE FOR THE OTHER,
/// and that asymmetry is not a simplification: the rotation crop is squared by
/// its diagonal because a hand is rotated about its centre inside it and a
/// long-side square clips the fingertips off a wide palm. The second crop is a
/// tight box around the already-rotated keypoints, where the long side is the
/// right square.
fn crop_pad(
    src: &Image,
    bbox_in: &[f64; 4],
    for_rotation: bool,
    geom: &crate::hands::Crop,
) -> Result<CropPad, String> {
    let mut bbox = *bbox_in;
    let wh0 = [bbox[2] - bbox[0], bbox[3] - bbox[1]];
    // The shift moves BOTH CORNERS of the box, not just the low one: the demo
    // writes it as `bbox + shift * wh` on a (2, 2) array and a (2,) broadcast
    // along the last axis moves x1 and x2 together, so the box's size is
    // unchanged and only its centre moves.
    for j in 0..2 {
        for k in 0..2 {
            bbox[2 * j + k] += geom.shift[k] * wh0[k];
        }
    }
    let ctr = [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5];
    let wh = [bbox[2] - bbox[0], bbox[3] - bbox[1]];
    for k in 0..2 {
        let half = wh[k] * geom.enlarge * 0.5;
        bbox[k] = ctr[k] - half;
        bbox[2 + k] = ctr[k] + half;
    }
    // x clips to the WIDTH and y to the HEIGHT, which is what the demo's
    // [0, 0] .. [shape[1], shape[0]] bound says. Swapping the two is invisible
    // on a square frame and clips the whole right edge of a landscape one.
    let hi = [src.width, src.height];
    let mut q = [0i64; 4];
    for k in 0..2 {
        for j in 0..2 {
            q[2 * k + j] = trunc_i32(bbox[2 * k + j]).clamp(0, hi[j]);
        }
    }
    let cw = q[2] - q[0];
    let ch = q[3] - q[1];
    if cw <= 0 || ch <= 0 {
        return Err(format!(
            "the palm crop came out {cw}x{ch} after enlarging by {} and clipping to \
             the {}x{} frame. That is a DETECTION problem rather than a crop one: the \
             palm detector's largest anchor is 6 cells across at level 1, and a box \
             that survives NMS is not supposed to be bigger than the picture or to sit \
             entirely outside it.",
            geom.enlarge, src.width, src.height
        ));
    }
    let sub = crop(src, q[0], q[1], q[2], q[3])?;
    let side = if for_rotation {
        (((ch * ch + cw * cw) as f64).sqrt()) as i64
    } else {
        ch.max(cw)
    };
    let ph = side - ch;
    let pw = side - cw;
    let left = pw / 2;
    let top = ph / 2;
    let mut out = like(side, side);
    for y in 0..ch {
        for x in 0..cw {
            let s = px(&sub, y, x).to_vec();
            let d = px_mut(&mut out, y + top, x + left);
            d.copy_from_slice(&s);
        }
    }
    // THE RETURNED BOX IS THE CLIPPED ONE, UNMOVED BY THE PADDING. The first
    // crop subtracts `bias` to get into padded-crop coordinates and the second
    // discards the bias entirely and reports the box as it stands in the
    // rotated crop -- the demo's asymmetry, and it is visible: it makes the
    // second crop's reported box SMALLER than the square it was padded into.
    Ok(CropPad {
        img: out,
        bbox: [q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64],
        bias: [q[0] as f64 - left as f64, q[1] as f64 - top as f64],
    })
}

/// cv2's 2x3 rotation about a centre: `[[cos, sin, tx], [-sin, cos, ty]]`.
///
/// THE SIGN OF THE SECOND ROW IS NOT A CONVENTION: MediaPipe's rotation puts
/// the wrist BELOW the middle-finger base, which in a frame whose y grows
/// downward is the matrix written here. Its transpose is also a rotation, also
/// by the same angle, and also produces 21 joints -- so the choice is caught by
/// the landmarks' pixel positions and by nothing else about the output.
fn rotation_matrix(cx: f64, cy: f64, angle_deg: f64, m: &mut [f64; 6]) {
    let a = angle_deg * std::f64::consts::PI / 180.0;
    let ca = a.cos();
    let sa = a.sin();
    m[0] = ca;
    m[1] = sa;
    m[2] = cx - ca * cx - sa * cy;
    m[3] = -sa;
    m[4] = ca;
    m[5] = cy + sa * cx - ca * cy;
}

/// The palm stage: the letterbox, and the pad the decode has to undo.
pub fn palm_stage(img: &Image, g: &Geometry) -> Result<(Tensor, Letterbox), String> {
    let (padded, lb) = letterbox(img, g.palm_input_size)?;
    let input = to_nchw_normalised(&padded, &g.mean, &g.std_dev)?;
    Ok((input, lb))
}

/// The palm decode: anchors, sigmoid, NMS, inverse letterbox.
///
/// The detections come back in DESCENDING SCORE order, which is the order NMS
/// produces them in. Every hand is an independent crop, so there is no "best"
/// argument here.
pub fn decode_palm(
    g: &Geometry,
    head: &PalmOut,
    lb: &Letterbox,
    orig_long_side: f64,
) -> Result<Vec<Detection>, String> {
    if head.n != g.palm_num_anchors {
        return Err(format!(
            "decode_palm: the head produced {} rows and the container declares {}",
            head.n, g.palm_num_anchors
        ));
    }
    let n = head.n;
    let inv = 1.0 / g.palm_input_size as f64;
    let mut bx = vec![0.0f64; (n * 4) as usize];
    let mut sc = vec![0.0f64; n as usize];
    for i in 0..n {
        let row = &head.boxes[(i * g.palm_row_terms) as usize..][..g.palm_row_terms as usize];
        let a = &g.anchors[(i * 2) as usize..][..2];
        // The score is a LOGIT: the graph's last node is the sigmoid's input
        // and the zoo applies 1/(1+exp(-x)) HERE. A head that folded the
        // sigmoid in would then be double-activated, and 0.89 becomes 0.71.
        sc[i as usize] = 1.0 / (1.0 + (-head.scores[i as usize]).exp());
        // The anchor is the box's CENTRE, so it is added after the half-width
        // is subtracted. Treating it as a corner is the other common SSD
        // convention and it doubles the box's position error.
        let cx = row[0] * inv * orig_long_side + a[0] * orig_long_side;
        let cy = row[1] * inv * orig_long_side + a[1] * orig_long_side;
        let bw = row[2] * inv * orig_long_side;
        let bh = row[3] * inv * orig_long_side;
        bx[(i * 4) as usize] = cx - bw * 0.5 - lb.pad_orig_x as f64;
        bx[(i * 4 + 1) as usize] = cy - bh * 0.5 - lb.pad_orig_y as f64;
        bx[(i * 4 + 2) as usize] = cx + bw * 0.5 - lb.pad_orig_x as f64;
        bx[(i * 4 + 3) as usize] = cy + bh * 0.5 - lb.pad_orig_y as f64;
    }

    // NMS: greedy, on the score order, dropping anything at IoU above the
    // threshold with the survivor. ONE box is kept unconditionally even at the
    // end of the list -- that is the demo's loop and it matters only in the
    // single-detection case, where dropping it would return nothing.
    let mut order: Vec<i64> = (0..n).collect();
    // Stable on the score, so two anchors with the same score suppress each
    // other in ANCHOR order rather than in whatever order the sort produced.
    order.sort_by(|a, b| {
        sc[*b as usize]
            .partial_cmp(&sc[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep: Vec<i64> = Vec::new();
    while !order.is_empty() && (keep.len() as i64) < g.top_k {
        let i = order[0];
        if sc[i as usize] < g.score_threshold {
            break;
        }
        keep.push(i);
        order.remove(0);
        if order.is_empty() {
            break;
        }
        let bi = &bx[(i * 4) as usize..][..4];
        let ai = (bi[2] - bi[0]) * (bi[3] - bi[1]);
        let mut rest: Vec<i64> = Vec::with_capacity(order.len());
        for j in order {
            let bj = &bx[(j * 4) as usize..][..4];
            let ix = 0.0f64.max(bi[2].min(bj[2]) - bi[0].max(bj[0]));
            let iy = 0.0f64.max(bi[3].min(bj[3]) - bi[1].max(bj[1]));
            let inter = ix * iy;
            let aj = (bj[2] - bj[0]) * (bj[3] - bj[1]);
            let den = ai + aj - inter;
            // A zero-area box against another gives 0/0, and the demo's
            // comparison is then FALSE, so the candidate is DROPPED rather
            // than kept. Mirroring that rather than substituting an IoU of zero
            // matters only for degenerate boxes, and the two differ in WHICH of
            // them survives -- so it is a choice, and it is the demo's.
            if den > 0.0 && inter / den <= g.nms_threshold {
                rest.push(j);
            }
        }
        order = rest;
    }

    let mut out = Vec::with_capacity(keep.len());
    for i in keep {
        let mut d = Detection {
            score: sc[i as usize],
            ..Detection::default()
        };
        for k in 0..4 {
            d.bbox[k] = bx[(i * 4 + k as i64) as usize];
        }
        let row = &head.boxes[(i * g.palm_row_terms) as usize..][..g.palm_row_terms as usize];
        let a = &g.anchors[(i * 2) as usize..][..2];
        for k in 0..7i64 {
            for j in 0..2usize {
                d.kps[k as usize][j] = (row[(4 + 2 * k + j as i64) as usize] * inv + a[j])
                    * orig_long_side
                    - if j == 0 {
                        lb.pad_orig_x as f64
                    } else {
                        lb.pad_orig_y as f64
                    };
            }
        }
        out.push(d);
    }
    Ok(out)
}

/// The landmark stage: crop, rotate, resize.
pub fn landmark_stage(img: &Image, g: &Geometry, det: &Detection) -> Result<LandmarkStage, String> {
    let mut s = LandmarkStage::default();
    let a = crop_pad(img, &det.bbox, true, &g.palm_pre)?;
    s.pad_bias = a.bias;

    // Everything below is in the padded rotation crop's own pixels. The
    // keypoints move by the crop's bias along with the raster: a landmark left
    // in frame pixels would put the rotation pivot off the hand by however far
    // the crop was shifted.
    let mut pl = [[0.0f64; 2]; 7];
    for k in 0..7 {
        for j in 0..2 {
            pl[k][j] = det.kps[k][j] - s.pad_bias[j];
        }
    }

    // The angle that puts the wrist BELOW the middle-finger base, wrapped into
    // [-pi, pi). The wrap is not optional: without it the same angle near pi
    // from the atan2 branch and near 0 from the other differs by a full turn,
    // and the landmark network is shown a hand upside down relative to itself.
    let p1 = pl[g.palm_lm_wrist as usize];
    let p2 = pl[g.palm_lm_middle_base as usize];
    let mut rad = std::f64::consts::FRAC_PI_2 - (-(p2[1] - p1[1])).atan2(p2[0] - p1[0]);
    rad -= 2.0
        * std::f64::consts::PI
        * ((rad + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)).floor();
    s.angle = rad * 180.0 / std::f64::consts::PI;

    let b2 = [
        a.bbox[0] - s.pad_bias[0],
        a.bbox[1] - s.pad_bias[1],
        a.bbox[2] - s.pad_bias[0],
        a.bbox[3] - s.pad_bias[1],
    ];
    let mut rot_m = [0.0f64; 6];
    rotation_matrix(
        (b2[0] + b2[2]) * 0.5,
        (b2[1] + b2[3]) * 0.5,
        s.angle,
        &mut rot_m,
    );
    s.rot_m = rot_m;
    let rot = warp_affine(&a.img, &rot_m)?;

    // The palm keypoints under the SAME rotation -- not re-predicted. The
    // detector already said where they are and the rotation is a rigid motion,
    // so transforming them is exact.
    let mut rb = [0.0f64; 4];
    let mut first = true;
    for k in 0..7 {
        let rx = pl[k][0] * rot_m[0] + pl[k][1] * rot_m[1] + rot_m[2];
        let ry = pl[k][0] * rot_m[3] + pl[k][1] * rot_m[4] + rot_m[5];
        if first {
            rb = [rx, ry, rx, ry];
            first = false;
        } else {
            rb[0] = rb[0].min(rx);
            rb[1] = rb[1].min(ry);
            rb[2] = rb[2].max(rx);
            rb[3] = rb[3].max(ry);
        }
    }
    s.palm_kp_box = rb;
    let b = crop_pad(&rot, &rb, false, &g.palm_post)?;
    s.blob = resize_area(&b.img, g.lm_input_size, g.lm_input_size)?;
    s.rot_bbox = b.bbox;
    Ok(s)
}

/// The inverse: the rotation undone, and the hand's reported box.
pub fn postprocess(g: &Geometry, lm_out: &LmOut, s: &LandmarkStage) -> Result<Hand, String> {
    // The container lists four outputs and the landmark network is not a
    // heatmap net: the argmax is already folded into the graph, so the screen
    // landmarks arrive as 63 plain numbers. Reshaping is to 21 x 3 and taking
    // the roles by POSITION, which is why the container records the count per
    // output and refuses a set that does not add up.
    if lm_out.outs.len() != 4 {
        return Err(format!(
            "postprocess: the landmark network returned {} outputs, this architecture \
             has four (63 screen, 1 presence, 1 handedness, 63 world)",
            lm_out.outs.len()
        ));
    }
    let want = 21usize * 3;
    if lm_out.outs[0].len() != want || lm_out.outs[3].len() != want {
        return Err(format!(
            "postprocess: the screen output is {} and the world output is {}, both \
             expected {want} (21 landmarks x 3)",
            lm_out.outs[0].len(),
            lm_out.outs[3].len()
        ));
    }
    if lm_out.outs[1].len() != 1 || lm_out.outs[2].len() != 1 {
        return Err("postprocess: presence and handedness are scalars, not vectors".to_string());
    }

    // One scale for x, y AND z, and it is the LARGER of the crop's two axis
    // scales. The crop is square only when the detector's keypoints are, and
    // taking the mean would shrink the depth by a different fraction than the
    // two pixel axes, which comes back as a hand slightly flattened in depth
    // with no individual number looking wrong.
    let sc_x = (s.rot_bbox[2] - s.rot_bbox[0]) / g.lm_input_size as f64;
    let sc_y = (s.rot_bbox[3] - s.rot_bbox[1]) / g.lm_input_size as f64;
    let sc = sc_x.max(sc_y);
    let half = g.lm_input_size as f64 / 2.0;

    let a = s.angle * std::f64::consts::PI / 180.0;
    let ca = a.cos();
    let sa = a.sin();
    // The rotation applied to the CROP, about the origin, which is the rotation
    // to undo. Its translation is undone separately below: composing the two
    // into one matrix would hide which of the two is the detector's and which
    // is the postprocess's.
    let r00 = ca;
    let r01 = sa;
    let r10 = -sa;
    let r11 = ca;
    // The inverse affine of the crop rotation, assembled by hand rather than
    // inverted numerically: the matrix is orthonormal up to f64 error.
    let i00 = r00;
    let i01 = r10;
    let i10 = r01;
    let i11 = r11;
    let it0 = -(i00 * s.rot_m[2] + i01 * s.rot_m[5]);
    let it1 = -(i10 * s.rot_m[2] + i11 * s.rot_m[5]);
    let cx = (s.rot_bbox[0] + s.rot_bbox[2]) * 0.5;
    let cy = (s.rot_bbox[1] + s.rot_bbox[3]) * 0.5;
    let ox = cx * i00 + cy * i01 + it0;
    let oy = cx * i10 + cy * i11 + it1;

    let mut h = Hand::default();
    for k in 0..21usize {
        let l = &lm_out.outs[0][k * 3..][..3];
        let w = &lm_out.outs[3][k * 3..][..3];
        let lx = (l[0] - half) * sc;
        let ly = (l[1] - half) * sc;
        let lz = l[2] * sc;
        // Rotate the crop's own coordinates by the same angle, and then take
        // the rotation's inverse plus the frame offset. The world points take
        // the ROTATION but not the translation: their origin is the wrist, in
        // metres, and adding a pixel offset would put a hand-sized translation
        // on a measurement whose scale is metres.
        h.landmarks[k][0] = lx * r00 + ly * r10 + ox + s.pad_bias[0];
        h.landmarks[k][1] = lx * r01 + ly * r11 + oy + s.pad_bias[1];
        h.landmarks[k][2] = lz;
        h.world[k][0] = w[0] * r00 + w[1] * r10;
        h.world[k][1] = w[0] * r01 + w[1] * r11;
        h.world[k][2] = w[2];
    }
    h.presence = lm_out.outs[1][0];
    h.handedness = lm_out.outs[2][0];

    // The reported box is fitted to the LANDMARKS, not to the detection: it is
    // the hand as the network sees it, which is a different rectangle from the
    // palm detector's -- larger, and shifted down the wrist.
    let mut lo = [h.landmarks[0][0], h.landmarks[0][1]];
    let mut hi = [lo[0], lo[1]];
    for k in 1..21usize {
        for j in 0..2 {
            lo[j] = lo[j].min(h.landmarks[k][j]);
            hi[j] = hi[j].max(h.landmarks[k][j]);
        }
    }
    let mut b = [lo[0], lo[1]];
    let mut e = [hi[0], hi[1]];
    let wh = [e[0] - b[0], e[1] - b[1]];
    for j in 0..2 {
        b[j] += g.hand.shift[j] * wh[j];
    }
    for j in 0..2 {
        e[j] = b[j] + wh[j];
    }
    let en = [
        (e[0] - b[0]) * g.hand.enlarge * 0.5,
        (e[1] - b[1]) * g.hand.enlarge * 0.5,
    ];
    let ctr = [(b[0] + e[0]) * 0.5, (b[1] + e[1]) * 0.5];
    for j in 0..2 {
        h.bbox[j] = ctr[j] - en[j];
        h.bbox[2 + j] = ctr[j] + en[j];
    }
    Ok(h)
}

// ---------------------------------------------------------------------------
// One frame
// ---------------------------------------------------------------------------

/// One image's answer, with the timings split so a client can see where the
/// frame went.
#[derive(Debug, Clone, Default)]
pub struct Run {
    /// What the palm stage found, in score order.
    pub detections: Vec<Detection>,
    /// The same subset, run through stage 2.
    pub hands: Vec<Hand>,
    /// Seconds in the front end: decode + letterbox + normalise.
    pub front_end_s: f64,
    /// Seconds in the palm network.
    pub palm_s: f64,
    /// Seconds in the anchor decode, the sigmoid and NMS.
    pub nms_s: f64,
    /// Seconds in every crop, rotate and area resize.
    pub crop_s: f64,
    /// Seconds in the landmark network, summed over the hands.
    pub lm_s: f64,
    /// Seconds in the postprocess.
    pub post_s: f64,
    /// Seconds for the whole image.
    pub total_s: f64,
    /// The SOURCE image's width.
    pub width: i64,
    /// The SOURCE image's height.
    pub height: i64,
    /// The front end's uniform scale.
    pub scale: f64,
    /// The letterbox's left pad, in original pixels.
    pub pad_x: i64,
    /// The letterbox's top pad, in original pixels.
    pub pad_y: i64,
    /// How many detections reached stage 2.
    pub hands_run: i64,
    /// The SECOND stage's geometry for the FIRST hand, in the caller's pixels.
    /// Not part of a hand's answer -- the landmarks are the answer -- but it is
    /// the one intermediate whose being wrong produces plausible landmarks.
    pub angle_deg: f64,
    /// The second crop's clipped box.
    pub rot_bbox: [f64; 4],
    /// The rotated palm keypoints' box before the second crop grew it.
    pub palm_kp_box: [f64; 4],
    /// What the first crop's padding cost.
    pub crop_pad_bias: [f64; 2],
    /// The crop's mean luminance in [0, 1].
    pub crop_mean: f64,
    /// How many dispatches the array saw, over both networks.
    ///
    /// It is the BACKEND's count and not the sum of the two networks': one
    /// backend serves both graphs, so adding their readings would count every
    /// dispatch twice.
    pub array_dispatches: i64,
    /// Seconds the array spent on the device, over both networks.
    pub array_s: f64,
    /// Seconds the host spent turning its activations into the device's dtype.
    pub array_repack_s: f64,
    /// Seconds the host spent transposing the array's result back into NCHW.
    pub array_transpose_s: f64,
    /// How many convolutions ran on the array, over both networks.
    pub array_convs: i64,
}

/// Where a dump hook wants the pipeline to report.
///
/// `net` is "palm" or "lm"; `node` is the graph node, and `usize::MAX` is the
/// network's INPUT (the photograph after the front end), which is what the
/// reader looks for as `-1`.
pub type Hook<'a> = dyn FnMut(&str, usize, &Tensor) + 'a;

/// One frame through both networks, with the whole decode between them.
///
/// `array` is the array's backend, when the run asked for the array: the
/// palm detector and the landmark network then dispatch their DENSE
/// convolutions to the board, and the depthwise ones stay on the host -- a
/// depthwise filter reduces WITHIN one channel and has no `[M, N]` product
/// in it to dispatch.
pub fn detect(
    model: &Hands,
    im: &Image,
    max_hands: i64,
    hook: &mut Hook<'_>,
    mut array: npum_model::array::ArrayRef<'_, '_>,
) -> Result<Run, String> {
    if im.empty() {
        return Err("the image decoded to nothing".into());
    }
    let g = model.geometry();
    let mut r = Run {
        width: im.width,
        height: im.height,
        ..Run::default()
    };

    // -- stage 1 ------------------------------------------------------------
    let t0 = std::time::Instant::now();
    let (palm_in, lb) = palm_stage(im, g)?;
    r.scale = lb.scale;
    r.pad_x = lb.pad_orig_x;
    r.pad_y = lb.pad_orig_y;
    r.front_end_s = t0.elapsed().as_secs_f64();

    // The INPUT is dumped too, as node -1, and it is the first thing a gate
    // should look at: a front-end difference shows up here as a difference in
    // 150k pixels rather than as an amplified one in 58 nodes of convolutions.
    hook("palm", usize::MAX, &palm_in);

    let t1 = std::time::Instant::now();
    let mut cost = crate::hands::Cost::default();
    let mut on_node = |i: usize, t: &Tensor| hook("palm", i, t);
    let pnodes =
        model.run_palm_costed_on_array(&palm_in, &mut on_node, array.as_deref_mut(), &mut cost)?;
    r.palm_s = t1.elapsed().as_secs_f64();
    // THE ARRAY'S COUNTERS, over both networks.
    //
    // The host-side spans ARE per network, so they are the ones that are
    // summed; the device's own time and the dispatch count are the BACKEND's,
    // and one backend serves both graphs -- so adding the two networks'
    // readings would count every dispatch twice. They are read once, at the
    // end of the frame.
    r.array_s = cost.t_array_gemm_s;
    r.array_repack_s = cost.t_array_repack_s;
    r.array_transpose_s = cost.t_array_transpose_s;
    r.array_convs = cost.convs_array;
    let head = model.palm_head(&pnodes)?;

    let t2 = std::time::Instant::now();
    let long_side = im.height.max(im.width) as f64;
    r.detections = decode_palm(g, &head, &lb, long_side)?;
    r.nms_s = t2.elapsed().as_secs_f64();

    // -- stage 2, once per surviving detection ------------------------------
    let n = r.detections.len() as i64;
    let run = if max_hands < 0 || n < max_hands {
        n
    } else {
        max_hands
    };
    r.hands_run = run;
    for i in 0..run {
        let t3 = std::time::Instant::now();
        let st = landmark_stage(im, g, &r.detections[i as usize])?;
        let lm_in = to_nchw_normalised(&st.blob, &g.mean, &g.std_dev)?;
        r.crop_s += t3.elapsed().as_secs_f64();

        hook("lm", usize::MAX, &lm_in);

        let t4 = std::time::Instant::now();
        let mut cost = crate::hands::Cost::default();
        let mut on_node = |j: usize, t: &Tensor| hook("lm", j, t);
        let lnodes =
            model.run_lm_costed_on_array(&lm_in, &mut on_node, array.as_deref_mut(), &mut cost)?;
        let lo = model.lm_head(&lnodes, &mut cost)?;
        r.lm_s += t4.elapsed().as_secs_f64();
        // THE LANDMARK NETWORK'S SHARE OF THE ARRAY'S TIME.
        //
        // The host-side spans are per network, so they add up; the device's
        // own time is the backend's and is read once, at the end.
        r.array_s += cost.t_array_gemm_s;
        r.array_repack_s += cost.t_array_repack_s;
        r.array_transpose_s += cost.t_array_transpose_s;
        r.array_convs += cost.convs_array;

        if i == 0 {
            r.angle_deg = st.angle;
            r.rot_bbox = st.rot_bbox;
            r.palm_kp_box = st.palm_kp_box;
            r.crop_pad_bias = st.pad_bias;
            let mut acc = 0.0f64;
            for v in &st.blob.rgb {
                acc += *v as f64;
            }
            r.crop_mean = if st.blob.rgb.is_empty() {
                0.0
            } else {
                acc / (255.0 * st.blob.rgb.len() as f64)
            };
        }
        let t5 = std::time::Instant::now();
        r.hands.push(postprocess(g, &lo, &st)?);
        r.post_s += t5.elapsed().as_secs_f64();
    }
    r.total_s = t0.elapsed().as_secs_f64();
    // THE BACKEND'S DISPATCH COUNT, read at the very end: the landmark
    // network runs once per hand, so the count is only final here.
    r.array_dispatches = array.as_deref().map(|a| a.dispatches() as i64).unwrap_or(0);
    Ok(r)
}

// ---------------------------------------------------------------------------
// The answers
// ---------------------------------------------------------------------------

/// `printf("%.9g")`: nine significant digits, trailing zeros stripped.
///
/// Rust's formatter has no `%g`, and the JSON's `letterbox.scale` was recorded
/// through one -- so the choice between the fixed and the exponential form is
/// made here, by C's rule: fixed when the exponent is in [-4, P).
pub fn g9(v: f64) -> String {
    if !v.is_finite() {
        return format!("{v}");
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let sci = format!("{:.8e}", v.abs());
    let (_, exp) = sci.split_once('e').unwrap_or(("", "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    if exp < -4 || exp >= 9 {
        // The exponential form, with C's two-digit exponent and no trailing
        // zeros in the mantissa.
        let (mant, _) = sci.split_once('e').unwrap_or(("0", ""));
        let m = mant.trim_end_matches('0').trim_end_matches('.');
        return format!("{m}e{}{:02}", if exp < 0 { "-" } else { "+" }, exp.abs());
    }
    let decimals = (8 - exp).max(0) as usize;
    let s = format!("{:.*}", decimals, v);
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// Escape a file name the way the oracle's JSON does.
///
/// The C++ prints the path through `%s`, so the quote and the backslash are the
/// only two an editor would notice; the newline is folded to `\n` because a
/// path with one would break the object it is in.
fn json_path(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' | '\\' => {
                out.push('\\');
                out.push(c);
            }
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    out
}

/// The whole answer, as the MediaPipe-shaped JSON object on stdout.
pub fn result_json(r: &Run, image_label: &str) -> String {
    let mut out = String::new();
    out.push_str("{\n  \"image\": \"");
    out.push_str(&json_path(image_label));
    out.push_str(&format!(
        "\",\n  \"width\": {},\n  \"height\": {},\n",
        r.width, r.height
    ));
    out.push_str(&format!(
        "  \"letterbox\": {{\"scale\": {}, \"pad_x\": {}, \"pad_y\": {}}},\n",
        g9(r.scale),
        r.pad_x,
        r.pad_y
    ));
    out.push_str(&format!(
        "  \"timings_ms\": {{\"total\": {:.3}, \"front_end\": {:.3}, \"palm\": {:.3}, \
         \"nms\": {:.3}, \"crop\": {:.3}, \"landmarks\": {:.3}, \"postprocess\": {:.3}}},\n",
        r.total_s * 1e3,
        r.front_end_s * 1e3,
        r.palm_s * 1e3,
        r.nms_s * 1e3,
        r.crop_s * 1e3,
        r.lm_s * 1e3,
        r.post_s * 1e3
    ));
    out.push_str("  \"detections\": [");
    for (i, d) in r.detections.iter().enumerate() {
        out.push_str(if i > 0 { ",\n    " } else { "\n    " });
        out.push_str(&format!(
            "{{\"score\": {:.6}, \"box\": [{:.3}, {:.3}, {:.3}, {:.3}], \"keypoints\": [",
            d.score, d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]
        ));
        for k in 0..7usize {
            out.push_str(if k > 0 { ", " } else { "" });
            out.push_str(&format!("[{:.3}, {:.3}]", d.kps[k][0], d.kps[k][1]));
        }
        out.push_str("]}");
    }
    out.push_str(if r.detections.is_empty() {
        "],\n  \"hands\": ["
    } else {
        "\n  ],\n  \"hands\": ["
    });
    for (i, h) in r.hands.iter().enumerate() {
        out.push_str(if i > 0 { ",\n    " } else { "\n    " });
        out.push_str(&format!(
            "{{\"score\": {:.6}, \"handedness\": {:.6}, \"presence\": {:.6}, \
             \"handedness_category\": \"{}\", \"bbox\": [{:.3}, {:.3}, {:.3}, {:.3}],\n     \
             \"landmarks\": [",
            r.detections[i].score,
            h.handedness,
            h.presence,
            if h.handedness > 0.5 { "Right" } else { "Left" },
            h.bbox[0],
            h.bbox[1],
            h.bbox[2],
            h.bbox[3]
        ));
        for k in 0..21usize {
            out.push_str(if k > 0 { ", " } else { "" });
            out.push_str(&format!(
                "[{:.3}, {:.3}, {:.3}]",
                h.landmarks[k][0], h.landmarks[k][1], h.landmarks[k][2]
            ));
        }
        out.push_str("],\n     \"world_landmarks\": [");
        for k in 0..21usize {
            out.push_str(if k > 0 { ", " } else { "" });
            out.push_str(&format!(
                "[{:.5}, {:.5}, {:.5}]",
                h.world[k][0], h.world[k][1], h.world[k][2]
            ));
        }
        out.push_str("]}");
    }
    out.push_str(if r.hands.is_empty() {
        "]}\n"
    } else {
        "\n  ]}\n"
    });
    out
}

/// The human summary `--text` prints.
pub fn text_summary(image: &str, r: &Run) -> String {
    if r.hands.is_empty() {
        return format!("{image}: no hands\n");
    }
    let mut s = String::new();
    for (i, h) in r.hands.iter().enumerate() {
        s.push_str(&format!(
            "{image}: hand {i}  score {:.3}  {}  presence {:.3}  box {:.0},{:.0} \
             {:.0},{:.0}  wrist {:.1},{:.1}  middle tip {:.1},{:.1}\n",
            r.detections[i].score,
            if h.handedness > 0.5 { "right" } else { "left" },
            h.presence,
            h.bbox[0],
            h.bbox[1],
            h.bbox[2],
            h.bbox[3],
            h.landmarks[0][0],
            h.landmarks[0][1],
            h.landmarks[12][0],
            h.landmarks[12][1]
        ));
    }
    s
}
