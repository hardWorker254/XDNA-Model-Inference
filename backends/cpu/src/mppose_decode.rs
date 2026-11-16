//! arch=8 decode: 2254 anchor rows become boxes, then ONE box plus TWO of its
//! four keypoints become a rotated 256x256 crop, then the landmark network's
//! 195 + 1 + 117 numbers become 39 rows in the caller's pixels and a frame-sized
//! mask.
//!
//! # FOUR FUNCTIONS, AND THE SPLITS ARE THE NETWORKS
//!
//! ```text
//!   det_stage     letterbox, into the detector's own normalisation
//!   decode_det    anchors, sigmoid, NMS, inverse letterbox
//!   pose_stage    crop, square, rotate, area-resize
//!   postprocess   the rotation's inverse, the reported box, and the mask
//! ```
//!
//! Between `det_stage` and `decode_det` the DETECTOR runs; between `pose_stage`
//! and `postprocess` the LANDMARK NETWORK runs. So no function here hands a
//! network in -- they take its outputs, which is what lets a caller compare the
//! two stages separately.
//!
//! # THE DETECTOR'S ORDER IS NORMALISE, THEN RESIZE
//!
//! The person detector divides by 255, rescales to [-1,1] and only THEN resizes
//! and pads, so its border is 0.0 IN THE [-1,1] TENSOR -- the middle of the
//! range, not the bottom. A front end that letterboxes a uint8 raster with 0 and
//! normalises afterwards puts (0 - 0.5)/0.5 = -1.0 there, which to this stem is
//! the most saturated value the input can take, all the way round the frame.
//!
//! The landmark net's own front end pads in the 8-bit raster and divides at the
//! very end, so ITS border is 0.0 too -- for the opposite reason. Both are 0.0
//! and only one of them is a bug, which is why neither is shared with the other.
//!
//! # THE ROTATION IS BETWEEN TWO PREDICTED POINTS
//!
//! The detector's row carries four keypoints; the region of interest is the
//! SQUARE centred on the hip whose half-side is the distance from the hip to the
//! full-body point. The crop is therefore scaled by the person's own anatomy
//! rather than by a detection box, and a loose box does not produce a loose crop.
//! WHICH two keypoints is read from the container: the wrong pair rotates by some
//! other angle, the network still returns 39 rows, and every one of them is
//! somewhere plausible.
//!
//! # EVERY INTEGER CONVERSION HERE IS numpy's
//!
//! The zoo's region-of-interest arithmetic is four `astype(np.int32)` calls and
//! numpy's float-to-int cast truncates TOWARD ZERO -- not floor: the square's
//! left edge is negative on a person near the margin, and floor there moves the
//! crop a whole pixel further out than the zoo does.

use crate::hands_decode::{crop, resize_area, to_nchw_normalised, warp_affine};
use crate::mppose::{DetOut, Geometry, MpPose, PoseOuts, Cost};
use crate::pose::Tensor;
use crate::vit::format_g;
use crate::vit_image::Image;

/// The uniform scale and symmetric zero padding the detector's letterbox applied.
#[derive(Debug, Clone, Copy)]
pub struct DetLetterbox {
    /// Model pixels per source pixel: `min(side/h, side/w)`.
    pub scale: f64,
    /// The left offset, in canvas pixels.
    pub left: i64,
    /// The top offset, in canvas pixels.
    pub top: i64,
    /// The same left offset in ORIGINAL pixels, truncated -- `(left / r).astype(int)`.
    pub pad_bias_x: i64,
    /// The same top offset in ORIGINAL pixels, truncated.
    pub pad_bias_y: i64,
}

/// One person detection, in the CALLER's pixels.
#[derive(Debug, Clone, Copy, Default)]
pub struct Detection {
    /// `x1, y1, x2, y2`.
    pub bounds: [f64; 4],
    /// The four keypoints the region of interest and the rotation are read from:
    /// hip centre, full body, shoulder centre, upper body.
    pub kps: [[f64; 2]; 4],
    /// The sigmoid'd score.
    pub score: f64,
}

/// One person, fully decoded, in the CALLER's pixels.
#[derive(Debug, Clone, Default)]
pub struct Person {
    /// The box fitted to the LANDMARKS and then enlarged. Not the detector's box.
    pub bbox: [f64; 4],
    /// 39 rows of `x, y, z, visibility, presence`.
    pub landmarks: Vec<[f64; 5]>,
    /// 39 rows of `x, y, z`, metres, hip-relative.
    pub world: Vec<[f64; 3]>,
    /// The network's own confidence, already a probability.
    pub conf: f64,
    /// The mask's width -- the frame's.
    pub mask_w: i64,
    /// The mask's height -- the frame's.
    pub mask_h: i64,
    /// The frame-sized binary silhouette: 255 for body, 0 for background.
    pub mask: Vec<u8>,
}

/// The crop and the four things the postprocess needs to undo the geometry.
#[derive(Debug, Clone)]
pub struct PoseStage {
    /// The `pose_input_size` square, 8-bit RGB.
    pub blob: Image,
    /// The padded square's width.
    pub square_w: i64,
    /// The padded square's height.
    pub square_h: i64,
    /// The rotation applied, in degrees.
    pub angle: f64,
    /// The same rotation as a 2x3, source -> destination.
    pub rot_m: [f64; 6],
    /// Where the square sits in the frame. MAY BE NEGATIVE.
    pub pad_bias: [f64; 2],
}

/// One image's answer, with the timings split so a client can see where the
/// frame went.
#[derive(Debug, Clone, Default)]
pub struct Run {
    /// What the detector found, in descending score order.
    pub detections: Vec<Detection>,
    /// The subset that reached stage two, in the same order.
    pub people: Vec<Person>,
    /// The letterbox and normalisation.
    pub front_end_s: f64,
    /// The detector network only.
    pub det_s: f64,
    /// The anchor decode, the sigmoid and NMS.
    pub nms_s: f64,
    /// Every crop, square, rotate and area-resize.
    pub crop_s: f64,
    /// The landmark network, summed over the people.
    pub pose_s: f64,
    /// The rotation's inverse, the box and the mask.
    pub post_s: f64,
    /// The whole frame.
    pub total_s: f64,
    /// The SOURCE image's width.
    pub width: i64,
    /// The SOURCE image's height.
    pub height: i64,
    /// The letterbox's scale.
    pub scale: f64,
    /// The letterbox's x offset, in original pixels.
    pub pad_x: i64,
    /// The letterbox's y offset, in original pixels.
    pub pad_y: i64,
    /// How many of `detections` reached stage two.
    pub people_run: i64,
    /// The detector's cost.
    pub det_cost: Cost,
    /// The landmark network's cost.
    pub pose_cost: Cost,
    /// The SECOND stage's geometry for the FIRST person, in degrees.
    pub angle_deg: f64,
    /// The padded square the first crop became.
    pub rot_bbox: [f64; 4],
    /// Where that square sits in the frame.
    pub crop_pad_bias: [f64; 2],
    /// The first crop's mean luminance in [0, 1].
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

// ---------------------------------------------------------------------------
// The raster helpers this architecture adds
// ---------------------------------------------------------------------------

/// An all-zero RGB raster.
fn like(h: i64, w: i64) -> Image {
    Image {
        width: w,
        height: h,
        rgb: vec![0u8; (h * w * 3).max(0) as usize],
    }
}

/// `cv2.copyMakeBorder(..., BORDER_CONSTANT, 0)` on an 8-bit RGB raster.
///
/// Not `letterbox`: that one decides the scale, this one pads a rectangle the
/// caller has already cut. `top`/`bottom` are ROWS and `left`/`right` are
/// COLUMNS -- transposing the last pair pads the wrong axis, which on a square
/// crop is invisible and on a rectangular one moves the subject.
fn pad_zeros(src: &Image, top: i64, bottom: i64, left: i64, right: i64) -> Result<Image, String> {
    if top < 0 || bottom < 0 || left < 0 || right < 0 {
        return Err(format!(
            "pad_zeros: negative border {top},{bottom},{left},{right}"
        ));
    }
    let mut out = like(src.height + top + bottom, src.width + left + right);
    for y in 0..src.height {
        let at = ((y * src.width) * 3) as usize;
        let len = (src.width * 3) as usize;
        let dst = (((y + top) * out.width + left) * 3) as usize;
        out.rgb[dst..dst + len].copy_from_slice(&src.rgb[at..at + len]);
    }
    Ok(out)
}

/// Bilinear at cv2's half-pixel grid on a CHANNEL-INTERLEAVED float buffer, with
/// the index clamp at the border and NO 8-bit rounding.
///
/// This is a separate function from `hands_decode::resize_bilinear` because the
/// ORDER of the two operations differs by architecture: arch=7 resizes an 8-bit
/// raster and normalises after, arch=8's DETECTOR normalises first and then
/// resizes in float. Resizing a uint8 raster and normalising afterwards is not
/// the same pipeline: it rounds to 8 bits at every tap rather than accumulating
/// in float.
///
/// The blend is `fma(a, 1-w, b*w)` -- one rounded multiply and one fused
/// multiply-add, which is the shape the oracle's `vmulss`/`vfmadd` pair has and
/// the same shape the u8 twin in `hands_decode` was verified bit-exact with.
fn resize_bilinear_f32(
    src: &[f32],
    sh: i64,
    sw: i64,
    dh: i64,
    dw: i64,
    ch: i64,
) -> Result<Vec<f32>, String> {
    if sh <= 0 || sw <= 0 || dh <= 0 || dw <= 0 || ch <= 0 {
        return Err(format!(
            "resize_bilinear_f32: {dh}x{dw} from a {sh}x{sw} source"
        ));
    }
    let fy_scale = sh as f64 / dh as f64;
    let fx_scale = sw as f64 / dw as f64;
    let cn = ch;
    let mut dst = vec![0.0f32; (dh * dw * cn) as usize];
    for y in 0..dh {
        let fy = (y as f64 + 0.5) * fy_scale - 0.5;
        let fy0f = fy.floor();
        let wy = fy - fy0f;
        let y0c = ((fy0f as i64).clamp(0, sh - 1)) as usize;
        let y1c = (((fy0f as i64) + 1).clamp(0, sh - 1)) as usize;
        let ra = &src[y0c * (sw * cn) as usize..][..(sw * cn) as usize];
        let rb = &src[y1c * (sw * cn) as usize..][..(sw * cn) as usize];
        for x in 0..dw {
            let fx = (x as f64 + 0.5) * fx_scale - 0.5;
            let fx0f = fx.floor();
            let wx = fx - fx0f;
            let x0c = ((fx0f as i64).clamp(0, sw - 1)) as usize;
            let x1c = (((fx0f as i64) + 1).clamp(0, sw - 1)) as usize;
            for k in 0..cn as usize {
                let a0 = ra[x0c * cn as usize + k] as f64;
                let a1 = ra[x1c * cn as usize + k] as f64;
                let b0 = rb[x0c * cn as usize + k] as f64;
                let b1 = rb[x1c * cn as usize + k] as f64;
                let top = a0.mul_add(1.0 - wx, a1 * wx);
                let bot = b0.mul_add(1.0 - wx, b1 * wx);
                dst[(y * dw * cn + x * cn) as usize + k] = top.mul_add(1.0 - wy, bot * wy) as f32;
            }
        }
    }
    Ok(dst)
}

/// `cv2.warpAffine` on a float buffer, at cv2's own grid, border 0.
///
/// The float twin of `hands_decode::warp_affine`, and it exists because the
/// segmentation mask is a float32 tensor with values well past 255: rounding it
/// to 8 bits -- or clamping it -- would throw away exactly the part of the signal
/// the threshold is applied to.
fn warp_affine_f32(
    src: &[f32],
    sh: i64,
    sw: i64,
    m: &[f64; 6],
    ch: i64,
) -> Result<Vec<f32>, String> {
    if sh <= 0 || sw <= 0 || ch <= 0 {
        return Err("warp_affine_f32: degenerate sizes".into());
    }
    let (a, b, tx) = (m[0], m[1], m[2]);
    let (c, d, ty) = (m[3], m[4], m[5]);
    let det = a * d - b * c;
    if det == 0.0 {
        return Err(
            "warp_affine_f32: the affine is singular, so it has no inverse and no destination \
             pixel has a source."
                .into(),
        );
    }
    let ia = d / det;
    let ib = -b / det;
    let ic = -c / det;
    let id = a / det;
    let itx = -(ia * tx + ib * ty);
    let ity = -(ic * tx + id * ty);
    let cn = ch;
    let mut dst = vec![0.0f32; (sh * sw * cn) as usize];
    for y in 0..sh {
        for x in 0..sw {
            let fx = ia * x as f64 + ib * y as f64 + itx;
            let fy = ic * x as f64 + id * y as f64 + ity;
            let o = ((y * sw + x) * cn) as usize;
            if fx < 0.0 || fx > (sw - 1) as f64 || fy < 0.0 || fy > (sh - 1) as f64 {
                // border = 0, and clamping instead would carry edge values of the
                // rotated square out to the corners.
                continue;
            }
            let fy0f = fy.floor();
            let fx0f = fx.floor();
            let wy = fy - fy0f;
            let wx = fx - fx0f;
            let y0c = (fy0f as i64).clamp(0, sh - 1) as usize;
            let y1c = ((fy0f as i64) + 1).clamp(0, sh - 1) as usize;
            let x0c = (fx0f as i64).clamp(0, sw - 1) as usize;
            let x1c = ((fx0f as i64) + 1).clamp(0, sw - 1) as usize;
            for k in 0..cn as usize {
                let pa = src[y0c * (sw * cn) as usize + x0c * cn as usize + k] as f64;
                let pb = src[y0c * (sw * cn) as usize + x1c * cn as usize + k] as f64;
                let pc = src[y1c * (sw * cn) as usize + x0c * cn as usize + k] as f64;
                let pd = src[y1c * (sw * cn) as usize + x1c * cn as usize + k] as f64;
                let top = pa.mul_add(1.0 - wx, pb * wx);
                let bot = pc.mul_add(1.0 - wx, pd * wx);
                dst[o + k] = top.mul_add(1.0 - wy, bot * wy) as f32;
            }
        }
    }
    Ok(dst)
}

/// numpy's `astype(np.int32)` on a float: truncation toward zero.
fn trunc_i32(v: f64) -> i64 {
    if v < 0.0 {
        v.ceil() as i64
    } else {
        v.floor() as i64
    }
}

/// cv2's 2x3 rotation about a centre: `[[ca, sa, tx], [-sa, ca, ty]]`.
///
/// THE SIGN OF THE SECOND ROW IS NOT A CONVENTION. MediaPipe's rotation puts the
/// full-body point ABOVE the hip, which in a frame whose y grows downward is the
/// matrix written here. Its transpose is also a rotation, also by the same angle,
/// and also produces 39 rows -- so the choice is caught by the landmarks' pixel
/// positions and by nothing else about the output.
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

// ---------------------------------------------------------------------------
// Stage 1
// ---------------------------------------------------------------------------

/// The detector's input, `[3, S, S]` float32, and the letterbox it came from.
///
/// `src` is the caller's frame in RGB. Doing a BGR2RGB here would swap the red
/// and blue channels, which on this architecture's own earlier demo moved the
/// pose joints by up to 110.8 px.
pub fn det_stage(src: &Image, g: &Geometry) -> Result<(Tensor, DetLetterbox), String> {
    if src.empty() {
        return Err("det_input: the image is empty".into());
    }
    let side = g.det_input_size;
    let (h, w) = (src.height, src.width);
    if side <= 0 {
        return Err(format!(
            "det_input: the container declares a detector input size of {side}"
        ));
    }
    if g.det_mean.len() != 3 || g.det_std.len() != 3 {
        return Err(format!(
            "det_input: the container declares {} means and {} standard deviations for the \
             detector; three of each are needed.",
            g.det_mean.len(),
            g.det_std.len()
        ));
    }
    // r = min(side/h, side/w), the LONG side onto the canvas.
    let r = (side as f64 / h as f64).min(side as f64 / w as f64);
    // Truncated toward zero, which for these positive values is a floor, and is
    // what `(h * ratio).astype(np.int32)` does.
    let rs_h = (h as f64 * r) as i64;
    let rs_w = (w as f64 * r) as i64;
    if rs_h < 1 || rs_w < 1 || rs_h > side || rs_w > side {
        return Err(format!(
            "det_input: a {h}x{w} image scales to {rs_h}x{rs_w} at side {side}"
        ));
    }

    // STEP 1: NORMALISE THE WHOLE FRAME, into an interleaved float buffer. This is
    // the zoo's `/255.0` and `(x - 0.5) * 2`, over the ENTIRE image and before any
    // resampling -- which is the whole reason this is not hands' letterbox.
    // The arithmetic is f32 throughout, as the oracle's is, AND the multiply and
    // the subtract are FUSED. The oracle's source reads `x * (1/255) - mean`, but
    // its object code is `vfmsub213ss` -- one rounding, not two -- and the two
    // differ by an ulp on roughly half the pixels of a frame. That single ulp at
    // the network's input is what moved the detector's score in its sixth
    // decimal, the pose confidence in its fifth and two mask pixels across the
    // threshold; nothing inside either graph was wrong. Read the contraction off
    // the object (`objdump -d npuimage` at npue::mppose::det_input), never off
    // the source.
    let mut lin = vec![0.0f32; (h * w * 3) as usize];
    let k = 1.0f32 / 255.0f32;
    for y in 0..h {
        let src_row = (y * w * 3) as usize;
        let dst_row = (y * w * 3) as usize;
        for p in 0..(w * 3) as usize {
            let c = p % 3;
            lin[dst_row + p] =
                (src.rgb[src_row + p] as f32).mul_add(k, -g.det_mean[c]) / g.det_std[c];
        }
    }

    // STEP 2: cv2.resize to (rs_w, rs_h), which is a REAL reduction -- a 810x1080
    // frame becomes 168x224 here, a factor of about 4.8.
    let resized = resize_bilinear_f32(&lin, h, w, rs_h, rs_w, 3)?;

    // STEP 3: the canvas, with the border written as 0.0 IN TENSOR SPACE. With
    // this container's mean 0.5 and std 0.5, -1.0 is the most saturated value the
    // stem can see, and a frame ringed with it moves the pyramid's edge responses.
    let plane = side * side;
    let mut out = vec![0.0f32; (3 * plane) as usize];
    let left = (side - rs_w) / 2;
    let top = (side - rs_h) / 2;
    for y in 0..rs_h {
        let srow = (y * rs_w * 3) as usize;
        let dy = y + top;
        for x in 0..rs_w {
            let d = (dy * side + x + left) as usize;
            for c in 0..3usize {
                out[c * plane as usize + d] = resized[srow + x as usize * 3 + c];
            }
        }
    }

    let lb = DetLetterbox {
        scale: r,
        left,
        top,
        // `(pad_bias / ratio).astype(np.int32)` -- a truncation, and an INTEGER
        // one. Both offsets are non-negative, so truncation is a floor here.
        pad_bias_x: (left as f64 / r) as i64,
        pad_bias_y: (top as f64 / r) as i64,
    };
    Ok((
        Tensor {
            c: 3,
            h: side,
            w: side,
            d: out,
        },
        lb,
    ))
}

/// 2254 anchor rows into detections, in DESCENDING SCORE order.
pub fn decode_det(
    g: &Geometry,
    head: &DetOut,
    lb: &DetLetterbox,
    img_w: i64,
    img_h: i64,
) -> Result<Vec<Detection>, String> {
    if head.n != g.det_num_anchors {
        return Err(format!(
            "decode_det: the head produced {} rows and the container declares {}",
            head.n, g.det_num_anchors
        ));
    }
    let n = head.n;
    let inv = 1.0 / g.det_input_size as f64;
    // The zoo's `scale = max(original_shape)` where original_shape is [w, h]: the
    // LONG side, not the width. On a portrait frame that is the height, and using
    // the width instead scales every box by 1080/810.
    let scale = img_w.max(img_h) as f64;
    let mut bx = vec![0.0f64; (n * 4) as usize];
    let mut sc = vec![0.0f64; n as usize];
    for i in 0..n {
        let row = &head.boxes[(i * g.det_row_terms) as usize..];
        let a = &g.anchors[(i * 2) as usize..];
        // The score is a LOGIT: the graph's last node before this head is the
        // classifier's 1x1 convolution and the zoo applies 1/(1+exp(-x)) HERE. The
        // clip to [-100, 100] is the zoo's too: exp(100) overflows to infinity.
        let raw = head.scores[i as usize].clamp(-100.0, 100.0);
        sc[i as usize] = 1.0 / (1.0 + (-raw).exp());
        // cx,cy,w,h against an anchor CENTRE -- not xy1,xy2.
        let cx = row[0] * inv * scale + a[0] * scale;
        let cy = row[1] * inv * scale + a[1] * scale;
        let bw = row[2] * inv * scale;
        let bh = row[3] * inv * scale;
        bx[(i * 4) as usize] = cx - bw * 0.5 - lb.pad_bias_x as f64;
        bx[(i * 4 + 1) as usize] = cy - bh * 0.5 - lb.pad_bias_y as f64;
        bx[(i * 4 + 2) as usize] = cx + bw * 0.5 - lb.pad_bias_x as f64;
        bx[(i * 4 + 3) as usize] = cy + bh * 0.5 - lb.pad_bias_y as f64;
    }

    // NMS: greedy, on the score order, dropping anything at IoU above the
    // threshold with the survivor, stopping at top_k. A STABLE sort, because two
    // anchors with the same score are ordered by index in the zoo too.
    let mut order: Vec<i64> = (0..n).collect();
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
        let bi = [bx[(i * 4) as usize], bx[(i * 4 + 1) as usize], bx[(i * 4 + 2) as usize], bx[(i * 4 + 3) as usize]];
        let ai = (bi[2] - bi[0]) * (bi[3] - bi[1]);
        let mut rest = Vec::with_capacity(order.len());
        for j in &order {
            let bj = [
                bx[(*j * 4) as usize],
                bx[(*j * 4 + 1) as usize],
                bx[(*j * 4 + 2) as usize],
                bx[(*j * 4 + 3) as usize],
            ];
            let ix = 0.0f64.max(bi[2].min(bj[2]) - bi[0].max(bj[0]));
            let iy = 0.0f64.max(bi[3].min(bj[3]) - bi[1].max(bj[1]));
            let inter = ix * iy;
            let aj = (bj[2] - bj[0]) * (bj[3] - bj[1]);
            let den = ai + aj - inter;
            if den > 0.0 && inter / den <= g.nms_threshold {
                rest.push(*j);
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
            d.bounds[k] = bx[(i * 4) as usize + k];
        }
        let row = &head.boxes[(i * g.det_row_terms) as usize..];
        let a = &g.anchors[(i * 2) as usize..];
        // The four keypoints are OFFSETS from the anchor, in units of the input
        // side, and they get the same anchor and the same long-side scale the box
        // does. They are NOT corners and NOT absolute.
        for k in 0..g.det_landmarks as usize {
            for j in 0..2usize {
                d.kps[k][j] = (row[4 + 2 * k + j] * inv + a[j]) * scale
                    - if j == 0 {
                        lb.pad_bias_x as f64
                    } else {
                        lb.pad_bias_y as f64
                    };
            }
        }
        out.push(d);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Stage 2
// ---------------------------------------------------------------------------

/// The rotated, area-resampled crop the landmark network sees.
pub fn pose_stage(img: &Image, g: &Geometry, det: &Detection) -> Result<PoseStage, String> {
    let hip = det.kps[g.person_lm_mid_hip as usize];
    let body = det.kps[g.person_lm_full_body as usize];
    let dist = (body[0] - hip[0]).hypot(body[1] - hip[1]);

    // FIRST TRUNCATION. `np.array([hip - d, hip + d], np.int32)`.
    let mut f0 = [trunc_i32(hip[0] - dist) as f64, trunc_i32(hip[1] - dist) as f64];
    let mut f1 = [trunc_i32(hip[0] + dist) as f64, trunc_i32(hip[1] + dist) as f64];
    // Enlarged about its own centre BY THE TRUNCATED WIDTH, which is an integer
    // number of pixels rather than the float 2d.
    let ctr = [(f0[0] + f1[0]) * 0.5, (f0[1] + f1[1]) * 0.5];
    let wh = [f1[0] - f0[0], f1[1] - f0[1]];
    let half = [
        wh[0] * g.person_box_pre_enlarge * 0.5,
        wh[1] * g.person_box_pre_enlarge * 0.5,
    ];
    // SECOND TRUNCATION, and a fresh pair: from here on the square is the integer
    // one and every coordinate below is measured against it.
    f0[0] = trunc_i32(ctr[0] - half[0]) as f64;
    f0[1] = trunc_i32(ctr[1] - half[1]) as f64;
    f1[0] = trunc_i32(ctr[0] + half[0]) as f64;
    f1[1] = trunc_i32(ctr[1] + half[1]) as f64;

    // Refine: clip to the frame, PER AXIS. x clips to the WIDTH and y to the
    // HEIGHT; swapping the two is invisible on a square frame and clips a whole
    // edge of a landscape one.
    let p0 = [
        0.0f64.max(f0[0].min(img.width as f64)),
        0.0f64.max(f0[1].min(img.height as f64)),
    ];
    let p1 = [
        0.0f64.max(f1[0].min(img.width as f64)),
        0.0f64.max(f1[1].min(img.height as f64)),
    ];
    if p1[0] <= p0[0] || p1[1] <= p0[1] {
        return Err(format!(
            "pose_stage: the region of interest clipped to nothing -- {},{} to {},{} in a \
             {}x{} frame. That is a DETECTION problem rather than a crop one: the square is \
             centred on the detector's hip point and has the distance from the hip to the \
             full-body point as its half-side, so a square that lands entirely outside the \
             frame means the box's four keypoints do not describe a body.",
            p0[0], p0[1], p1[0], p1[1], img.width, img.height
        ));
    }
    let sub = crop(
        img,
        p0[0] as i64,
        p0[1] as i64,
        p1[0] as i64,
        p1[1] as i64,
    )?;
    // Pad back to the unclipped square, so the result is exactly f0..f1's extent --
    // which is what the rest of the geometry is measured against.
    let pad_left = (p0[0] - f0[0]) as i64;
    let pad_top = (p0[1] - f0[1]) as i64;
    let pad_right = (f1[0] - p1[0]) as i64;
    let pad_bottom = (f1[1] - p1[1]) as i64;
    let square = pad_zeros(&sub, pad_top, pad_bottom, pad_left, pad_right)?;

    // `pad_bias = full_bbox[0]`, an INTEGER that may be NEGATIVE when the square
    // reaches past the frame's edge. The landmarks are offset by it in frame pixels
    // and the mask is placed at it.
    let pad_bias = [f0[0], f0[1]];

    // The rotation reads the two points in the SQUARE's own coordinates.
    let hx = hip[0] - pad_bias[0];
    let hy = hip[1] - pad_bias[1];
    let bx = body[0] - pad_bias[0];
    let by = body[1] - pad_bias[1];

    // The angle that puts the full-body point ABOVE the hip, wrapped into
    // (-pi, pi]. The wrap is not optional: without it the same angle near pi from
    // one atan2 branch and near 0 from the other differs by a full turn.
    let mut rad = std::f64::consts::FRAC_PI_2 - (-(by - hy)).atan2(bx - hx);
    rad -= 2.0 * std::f64::consts::PI * ((rad + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)).floor();
    let angle = rad * 180.0 / std::f64::consts::PI;

    let mut rot_m = [0.0f64; 6];
    rotation_matrix(hx, hy, angle, &mut rot_m);
    let rotated = warp_affine(&square, &rot_m)?;
    // INTER_AREA, an exact box average, and not bilinear: the square is normally
    // several times larger than 256 and bilinear there moves every landmark by a
    // pixel and a half.
    let blob = resize_area(&rotated, g.pose_input_size, g.pose_input_size)?;
    Ok(PoseStage {
        blob,
        square_w: square.width,
        square_h: square.height,
        angle,
        rot_m,
        pad_bias,
    })
}

/// The rotation's inverse, the reported box, and the mask.
pub fn postprocess(
    g: &Geometry,
    out: &PoseOuts,
    s: &PoseStage,
    img_w: i64,
    img_h: i64,
) -> Result<Person, String> {
    let nl = g.num_landmarks as usize;
    let want_lm = nl * g.lm_cols as usize;
    let want_w = nl * g.world_cols as usize;
    if out.landmarks.len() != want_lm || out.world.len() != want_w {
        return Err(format!(
            "postprocess: the screen output is {} and the world output is {}, expected {want_lm} \
             ({nl}x{}) and {want_w} ({nl}x{})",
            out.landmarks.len(),
            out.world.len(),
            g.lm_cols,
            g.world_cols
        ));
    }

    let in_sz = g.pose_input_size as f64;
    let half = in_sz / 2.0;
    // One scale PER AXIS for x and y, and the LARGER of the two for z. The square
    // is not exactly square -- two truncations are between it and a square.
    let sf_x = s.square_w as f64 / in_sz;
    let sf_y = s.square_h as f64 / in_sz;
    let sf_z = sf_x.max(sf_y);

    let mut p = Person {
        conf: out.conf,
        ..Person::default()
    };

    // The 2x2 of the crop rotation, applied to the landmarks to UNDO it:
    // `landmarks[:, :2] @ [[cos, sin], [-sin, cos]]`, i.e. x*cos - y*sin and
    // x*sin + y*cos.
    let a = s.angle * std::f64::consts::PI / 180.0;
    let ca = a.cos();
    let sa = a.sin();
    let (m00, m01, m10, m11) = (ca, -sa, sa, ca);

    // The inverse of the crop's rotation, assembled by hand: the matrix is
    // orthonormal up to f64 error, so an analytic inverse is exact.
    let i00 = s.rot_m[0];
    let i01 = s.rot_m[3];
    let i10 = s.rot_m[1];
    let i11 = s.rot_m[4];
    let it0 = -(i00 * s.rot_m[2] + i01 * s.rot_m[5]);
    let it1 = -(i10 * s.rot_m[2] + i11 * s.rot_m[5]);
    // The square's centre, mapped through that inverse. This is the translation the
    // landmarks need and the one number here easiest to get wrong by a plausible
    // amount.
    let ccx = s.square_w as f64 * 0.5;
    let ccy = s.square_h as f64 * 0.5;
    let ocx = ccx * i00 + ccy * i01 + it0;
    let ocy = ccx * i10 + ccy * i11 + it1;

    p.landmarks = Vec::with_capacity(nl);
    p.world = Vec::with_capacity(nl);
    for k in 0..nl {
        let l = &out.landmarks[k * g.lm_cols as usize..];
        let w = &out.world[k * g.world_cols as usize..];
        let lx = (l[0] - half) * sf_x;
        let ly = (l[1] - half) * sf_y;
        let lz = l[2] * sf_z;
        // The visibility and presence columns ARE squashed, and only those two.
        // Squashing the coordinates as well would turn a pose into noise with
        // every shape right.
        p.landmarks.push([
            lx * m00 + ly * m01 + ocx + s.pad_bias[0],
            lx * m10 + ly * m11 + ocy + s.pad_bias[1],
            lz,
            1.0 / (1.0 + (-l[3]).exp()),
            1.0 / (1.0 + (-l[4]).exp()),
        ]);
        // The world points take the ROTATION and not the translation: their origin
        // is the hip, in metres.
        p.world.push([
            w[0] * m00 + w[1] * m01,
            w[0] * m10 + w[1] * m11,
            w[2],
        ]);
    }

    // The reported box is fitted to the LANDMARKS -- all thirty-nine, the six
    // auxiliary rows included -- then enlarged about its own centre. It is NOT the
    // detector's box.
    let mut lo = [p.landmarks[0][0], p.landmarks[0][1]];
    let mut hi = [lo[0], lo[1]];
    for row in p.landmarks.iter().skip(1) {
        for j in 0..2 {
            lo[j] = lo[j].min(row[j]);
            hi[j] = hi[j].max(row[j]);
        }
    }
    let bctr = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5];
    let bhalf = [
        (hi[0] - lo[0]) * g.person_box_enlarge * 0.5,
        (hi[1] - lo[1]) * g.person_box_enlarge * 0.5,
    ];
    for j in 0..2 {
        p.bbox[j] = bctr[j] - bhalf[j];
        p.bbox[2 + j] = bctr[j] + bhalf[j];
    }

    // -- the mask, back into the frame
    //
    // Four steps, all of them silent: a rotation by -angle about the 256 square's
    // centre, a resize to the square's size, a crop of whatever hangs outside the
    // frame, and a pad that places the result at the square's offset.
    let ms = g.pose_input_size;
    if out.mask.len() as i64 != ms * ms {
        return Err(format!(
            "postprocess: the mask is {} values and a {ms}x{ms} mask is {}",
            out.mask.len(),
            ms * ms
        ));
    }
    let mut inv = [0.0f64; 6];
    rotation_matrix(ms as f64 / 2.0, ms as f64 / 2.0, -s.angle, &mut inv);
    let warped = warp_affine_f32(&out.mask, ms, ms, &inv, 1)?;
    // cv2.resize with NO interpolation argument, which is INTER_LINEAR -- not the
    // INTER_AREA the crop itself used.
    let placed = resize_bilinear_f32(&warped, ms, ms, s.square_h, s.square_w, 1)?;

    // The crop is whatever lies inside the frame and the pad is the rest.
    // pad_bias may be NEGATIVE, in which case there is no left pad and the square
    // is cropped there instead -- which is why the low and high bounds are
    // separate terms rather than one offset.
    let over = [
        img_w as f64 - s.square_w as f64 - s.pad_bias[0],
        img_h as f64 - s.square_h as f64 - s.pad_bias[1],
    ];
    let lo_x = (-s.pad_bias[0].min(0.0)) as i64;
    let lo_y = (-s.pad_bias[1].min(0.0)) as i64;
    let hi_x = s.square_w + over[0].min(0.0) as i64;
    let hi_y = s.square_h + over[1].min(0.0) as i64;
    let pad_l = s.pad_bias[0].max(0.0) as i64;
    let pad_t = s.pad_bias[1].max(0.0) as i64;
    let pad_r = over[0].max(0.0) as i64;
    let pad_b = over[1].max(0.0) as i64;
    // THE COMPOSED EXTENT MUST BE THE FRAME'S, and this is where that is checked
    // rather than assumed: a pad_bias computed with a rounding instead of a
    // truncation makes the two disagree by a row or a column.
    if (hi_x - lo_x) + pad_l + pad_r != img_w || (hi_y - lo_y) + pad_t + pad_b != img_h {
        return Err(format!(
            "postprocess: the mask's crop and pad compose to {}x{} and the frame is {img_w}x\
             {img_h}. pad_bias is ({}, {}), the square is {}x{}. One of those three is not \
             what the zoo's arithmetic produces.",
            (hi_x - lo_x) + pad_l + pad_r,
            (hi_y - lo_y) + pad_t + pad_b,
            s.pad_bias[0],
            s.pad_bias[1],
            s.square_w,
            s.square_h
        ));
    }
    if hi_x <= lo_x || hi_y <= lo_y {
        return Err(format!(
            "postprocess: the mask's square lies entirely outside the frame -- {lo_x},{lo_y} to \
             {hi_x},{hi_y} in a {img_w}x{img_h} frame. The landmarks would be reported anyway, \
             so without this the failure is an empty mask with nothing saying why."
        ));
    }

    p.mask_w = img_w;
    p.mask_h = img_h;
    p.mask = vec![0u8; (img_w * img_h) as usize];
    for y in 0..(hi_y - lo_y) {
        for x in 0..(hi_x - lo_x) {
            // `np.where(mask > 0, 255, 0)` -- a BINARY mask, thresholded at zero.
            if placed[(y * s.square_w + x) as usize] > 0.0 {
                p.mask[((y + pad_t) * img_w + (x + pad_l)) as usize] = 255;
            }
        }
    }
    Ok(p)
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

/// One image in, N people out.
///
/// `hook` is called once per graph node with `(net, node, tensor)`, and once per
/// network with node `usize::MAX` for the INPUT -- which is the photograph after
/// the front end, and the first thing a gate should look at.
///
/// `array` is the array's backend, when the run asked for the array: the person
/// detector and the landmark network then dispatch their DENSE convolutions to
/// the board, and the depthwise ones stay on the host -- a depthwise filter
/// reduces WITHIN one channel and has no `[M, N]` product in it to dispatch.
pub fn detect(
    m: &MpPose,
    img: &Image,
    max_people: i64,
    hook: &mut dyn FnMut(&str, usize, &Tensor),
    mut array: npum_model::array::ArrayRef<'_, '_>,
) -> Result<Run, String> {
    let g = m.geometry();
    if img.empty() {
        return Err("the image decoded to nothing".into());
    }
    let mut r = Run {
        width: img.width,
        height: img.height,
        ..Run::default()
    };

    // -- stage 1 ------------------------------------------------------------
    let t0 = std::time::Instant::now();
    let (det_in, lb) = det_stage(img, g)?;
    r.scale = lb.scale;
    r.pad_x = lb.pad_bias_x;
    r.pad_y = lb.pad_bias_y;
    r.front_end_s = t0.elapsed().as_secs_f64();
    hook("det", usize::MAX, &det_in);

    let t1 = std::time::Instant::now();
    let dnodes = m.run_det(
        &det_in,
        &mut |i, t| hook("det", i, t),
        &mut r.det_cost,
        array.as_deref_mut(),
    )?;
    r.det_s = t1.elapsed().as_secs_f64();
    let dh = m.det_head(&dnodes)?;

    let t2 = std::time::Instant::now();
    r.detections = decode_det(g, &dh, &lb, img.width, img.height)?;
    r.nms_s = t2.elapsed().as_secs_f64();

    // -- stage 2, once per surviving detection -------------------------------
    let n_det = r.detections.len() as i64;
    let run = if max_people < 0 || n_det < max_people {
        n_det
    } else {
        max_people
    };
    r.people_run = run;
    for i in 0..run {
        let t3 = std::time::Instant::now();
        let st = pose_stage(img, g, &r.detections[i as usize])?;
        // The landmark net's own normalisation, which is u8/255 and NOT the
        // detector's: its two front ends are different models' worth of
        // preprocessing and the container carries both.
        let pose_in = to_nchw_normalised(&st.blob, &g.pose_mean, &g.pose_std)?;
        r.crop_s += t3.elapsed().as_secs_f64();
        hook("pose", usize::MAX, &pose_in);

        let t4 = std::time::Instant::now();
        let pnodes = m.run_pose(
            &pose_in,
            &mut |n, t| hook("pose", n, t),
            &mut r.pose_cost,
            array.as_deref_mut(),
        )?;
        let po = m.pose_head(&pnodes)?;
        r.pose_s += t4.elapsed().as_secs_f64();

        // The confidence gate is HERE, after the network, exactly as the demo
        // places it: `if conf < confThreshold: return None`. A person it drops is
        // not reported, and a person after it is still tried -- the demo continues
        // rather than breaking.
        if po.conf < g.pose_conf_threshold {
            r.people_run = i;
            continue;
        }

        if i == 0 {
            r.angle_deg = st.angle;
            r.rot_bbox = [0.0, 0.0, st.square_w as f64, st.square_h as f64];
            r.crop_pad_bias = st.pad_bias;
            let acc: f64 = st.blob.rgb.iter().map(|v| *v as f64).sum();
            r.crop_mean = if st.blob.rgb.is_empty() {
                0.0
            } else {
                acc / (255.0 * st.blob.rgb.len() as f64)
            };
        }
        let t5 = std::time::Instant::now();
        r.people
            .push(postprocess(g, &po, &st, img.width, img.height)?);
        r.post_s += t5.elapsed().as_secs_f64();
    }
    r.total_s = t0.elapsed().as_secs_f64();
    // THE ARRAY'S COUNTERS, over both networks.
    //
    // The host-side spans ARE per network, so they are the ones that are
    // summed -- and the landmark network's are summed over the people it ran
    // for, because its cost is one accumulator for the whole frame. The
    // device's own time and the dispatch count are the BACKEND's, and one
    // backend serves both graphs -- so adding the two networks' readings
    // would count every dispatch twice. They are read once, at the end.
    r.array_s = r.det_cost.t_array_gemm_s + r.pose_cost.t_array_gemm_s;
    r.array_repack_s = r.det_cost.t_array_repack_s + r.pose_cost.t_array_repack_s;
    r.array_transpose_s = r.det_cost.t_array_transpose_s + r.pose_cost.t_array_transpose_s;
    r.array_convs = r.det_cost.convs_array + r.pose_cost.convs_array;
    r.array_dispatches = array.as_deref().map(|a| a.dispatches() as i64).unwrap_or(0);
    Ok(r)
}

/// The answer, shaped like MediaPipe's `PoseLandmarker` output.
///
/// The field order and width are a contract: the golden transcript holds these
/// digits, and a caller porting from the zoo reads them by name and by position.
pub fn result_json(r: &Run, image: &str) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "{{\n  \"image\": \"{}\",\n  \"width\": {},\n  \"height\": {},\n",
        crate::vit::json_escape(image),
        r.width,
        r.height
    ));
    s.push_str(&format!(
        "  \"letterbox\": {{\"scale\": {}, \"pad_x\": {}, \"pad_y\": {}}},\n",
        format_g(r.scale, 9),
        r.pad_x,
        r.pad_y
    ));
    s.push_str(&format!(
        "  \"timings_ms\": {{\"total\": {:.3}, \"front_end\": {:.3}, \"detector\": {:.3}, \
         \"nms\": {:.3}, \"crop\": {:.3}, \"landmarks\": {:.3}, \"postprocess\": {:.3}}},\n",
        r.total_s * 1e3,
        r.front_end_s * 1e3,
        r.det_s * 1e3,
        r.nms_s * 1e3,
        r.crop_s * 1e3,
        r.pose_s * 1e3,
        r.post_s * 1e3
    ));
    s.push_str("  \"detections\": [");
    for (i, d) in r.detections.iter().enumerate() {
        s.push_str(&format!(
            "{}\n    {{\"score\": {:.6}, \"box\": [{:.3}, {:.3}, {:.3}, {:.3}], \"keypoints\": [",
            if i == 0 { "" } else { "," },
            d.score,
            d.bounds[0],
            d.bounds[1],
            d.bounds[2],
            d.bounds[3]
        ));
        for (k, kp) in d.kps.iter().enumerate() {
            s.push_str(&format!(
                "{}[{:.3}, {:.3}]",
                if k == 0 { "" } else { ", " },
                kp[0],
                kp[1]
            ));
        }
        s.push_str("]}");
    }
    s.push_str(&format!(
        "{}],\n  \"poses\": [",
        if r.detections.is_empty() { "" } else { "\n  " }
    ));
    for (i, p) in r.people.iter().enumerate() {
        let score = r.detections[i].score;
        s.push_str(&format!(
            "{}\n    {{\"score\": {:.6}, \"pose_confidence\": {:.6}, \"bbox\": [{:.3}, {:.3}, \
             {:.3}, {:.3}],\n     \"landmarks\": [",
            if i == 0 { "" } else { "," },
            score,
            p.conf,
            p.bbox[0],
            p.bbox[1],
            p.bbox[2],
            p.bbox[3]
        ));
        for (k, l) in p.landmarks.iter().enumerate() {
            s.push_str(&format!(
                "{}[{:.3}, {:.3}, {:.3}, {:.5}, {:.5}]",
                if k == 0 { "" } else { ", " },
                l[0],
                l[1],
                l[2],
                l[3],
                l[4]
            ));
        }
        s.push_str("],\n     \"world_landmarks\": [");
        for (k, w) in p.world.iter().enumerate() {
            s.push_str(&format!(
                "{}[{:.5}, {:.5}, {:.5}]",
                if k == 0 { "" } else { ", " },
                w[0],
                w[1],
                w[2]
            ));
        }
        let nonzero = p.mask.iter().filter(|v| **v != 0).count();
        s.push_str(&format!(
            "],\n     \"segmentation_mask\": {{\"width\": {}, \"height\": {}, \"nonzero\": \
             {nonzero}}}}}",
            p.mask_w, p.mask_h
        ));
    }
    s.push_str(&format!(
        "{}]}}\n",
        if r.people.is_empty() { "" } else { "\n  " }
    ));
    s
}

/// A human summary instead of the JSON.
///
/// It reports the nose and the LEFT WRIST by index, which is MediaPipe's own
/// numbering: landmark 0 and landmark 15. Both are printed so that a reader can
/// see one joint near the face and one at the far end of the skeleton without
/// reading thirty-nine rows.
pub fn text_summary(image: &str, r: &Run) -> String {
    if r.people.is_empty() {
        return format!("{image}: no person\n");
    }
    let mut s = String::new();
    for (i, p) in r.people.iter().enumerate() {
        let nose = p.landmarks.first().copied().unwrap_or([0.0; 5]);
        let wrist = p.landmarks.get(15).copied().unwrap_or([0.0; 5]);
        s.push_str(&format!(
            "{}: pose {}  score {:.3}  conf {:.3}  box {:.0},{:.0} {:.0},{:.0}  nose \
             {:.1},{:.1}  left wrist {:.1},{:.1}\n",
            image,
            i,
            r.detections[i].score,
            p.conf,
            p.bbox[0],
            p.bbox[1],
            p.bbox[2],
            p.bbox[3],
            nose[0],
            nose[1],
            wrist[0],
            wrist[1]
        ));
    }
    s
}
