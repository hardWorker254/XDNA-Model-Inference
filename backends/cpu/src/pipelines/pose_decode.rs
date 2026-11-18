//! The pose front end and the head decode: letterbox, threshold, NMS and
//! the letterbox inverse.
//!
//! # WHY LETTERBOX AND NOT THE ViT's SQUARE RESIZE
//!
//! The ViT resizes to a square because classification does not care where
//! the person is. Pose is not classification: every keypoint is a PIXEL
//! COORDINATE in the original frame, so the front end has to remember
//! exactly how it moved them. Squashing a 16:9 photograph into 640x640 does
//! not scale uniformly, and a decode that assumes it did would place a nose
//! at 1.78x its true offset -- a pose that looks almost right, which is the
//! worst kind of wrong.
//!
//! So this letterboxes: one uniform scale, then padding, and both are
//! recorded and inverted in the decode.
//!
//! # THE PAD VALUE IS PART OF THE MODEL
//!
//! Ultralytics fills with the grey 114, and a front end that filled with 0
//! would feed the network a border it was never trained on, which moves the
//! pyramid's edge responses and with them the keypoints nearest the margin.
//! The pad is applied as a PIXEL VALUE and then normalised like any other,
//! so it becomes `(114/255 - mean)/std`.
//!
//! # THE LETTERBOX INVERSE'S ORDER IS THE WHOLE FUNCTION
//!
//! The front end built the canvas as `canvas = source * scale + pad`, so the
//! inverse subtracts the pad in CANVAS pixels and only then undoes the
//! scale. Dividing first and subtracting after gives `v / scale - pad`
//! instead of `(v - pad) / scale`, which differs by exactly
//! `pad * (1/scale - 1)` -- 55 pixels on an 810x1080 source letterboxed to
//! 640.
//!
//! And the clip is in SOURCE coordinates against the source's own width:
//! a model coordinate that falls inside the letterbox PAD maps to a
//! negative source pixel, and a joint reported at x = -83 is the pad
//! showing through, not an estimate of a limb.

use crate::pose::{Geometry, Tensor};
use crate::vit_image::{self, Image, Resample};

/// The COCO 17-point skeleton, in the head tensor's keypoint order.
///
/// These seventeen names in this order are what every COCO pose checkpoint
/// was trained to emit, which is why the geometry refuses a different
/// number of keypoints rather than accepting one and naming it with these.
pub const SKELETON_NAMES: [&str; 17] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
];

/// The 19 bones, as COCO defines them.
pub const SKELETON_EDGES: [&str; 12] = [
    "left_shoulder-left_elbow",
    "left_elbow-left_wrist",
    "left_shoulder-left_hip",
    "left_hip-left_knee",
    "left_knee-left_ankle",
    "right_shoulder-right_elbow",
    "right_elbow-right_wrist",
    "right_shoulder-right_hip",
    "right_hip-right_knee",
    "right_knee-right_ankle",
    "left_shoulder-right_shoulder",
    "left_hip-right_hip",
];

/// The grey Ultralytics pads with, as a 0..255 value.
pub const PAD_VALUE: f32 = 114.0;

/// The uniform scale and padding that letterbox applied, so the decode can
/// invert it.
#[derive(Debug, Clone)]
pub struct Letterbox {
    /// Model pixels per source pixel.
    pub scale: f64,
    /// The left offset, in model pixels.
    pub pad_x: i64,
    /// The top offset, in model pixels.
    pub pad_y: i64,
    /// The source's width.
    pub src_w: i64,
    /// The source's height.
    pub src_h: i64,
    /// The padded canvas's side.
    pub dst: i64,
}

/// The knobs. Their defaults are Ultralytics' predict defaults, and a pose
/// client that reports a person count is reporting the product of these
/// numbers.
#[derive(Debug, Clone)]
pub struct DecodeParams {
    /// The class score threshold.
    pub conf: f64,
    /// The NMS overlap threshold.
    pub iou: f64,
    /// Below this a joint is reported not visible.
    pub keypoint: f64,
    /// The cap on the returned list.
    pub max_det: i64,
    /// The cap on the pre-NMS candidate count.
    pub max_nms: i64,
}

impl Default for DecodeParams {
    fn default() -> DecodeParams {
        DecodeParams {
            conf: 0.25,
            iou: 0.70,
            keypoint: 0.50,
            max_det: 300,
            max_nms: 30000,
        }
    }
}

/// One joint, in the caller's pixel coordinates.
#[derive(Debug, Clone)]
pub struct Keypoint {
    /// The x coordinate.
    pub x: f64,
    /// The y coordinate.
    pub y: f64,
    /// The head's visibility score, as the model gave it.
    pub v: f64,
    /// Whether the score is at or above the threshold.
    pub visible: bool,
    /// The index, 0..16.
    pub index: i32,
}

/// One person, in the caller's pixel coordinates.
#[derive(Debug, Clone, Default)]
pub struct Person {
    /// The box's left edge.
    pub x1: f64,
    /// The box's top edge.
    pub y1: f64,
    /// The box's right edge.
    pub x2: f64,
    /// The box's bottom edge.
    pub y2: f64,
    /// The class score.
    pub score: f64,
    /// Which of the 8400 anchors produced it.
    pub anchor: i64,
    /// Which stride.
    pub level: i64,
    /// The keypoints.
    pub keypoints: Vec<Keypoint>,
}

/// The letterbox front end: resize, pad and normalise.
///
/// The resize is PIL-equivalent BILINEAR, and it is NOT cv2.resize: those
/// differ by up to one 8-bit level on a downscale, and after /255 and 200
/// convolutions that is a pixel of keypoint drift nobody would attribute to
/// the resampler.
pub fn letterbox_normalise(src: &Image, g: &Geometry) -> Result<(Tensor, Letterbox), String> {
    if src.empty() {
        return Err("pose front end: the image decoded to nothing".into());
    }
    let side = g.input_size;
    let mut lb = Letterbox {
        scale: 1.0,
        pad_x: 0,
        pad_y: 0,
        src_w: src.width,
        src_h: src.height,
        dst: side,
    };
    // ONE uniform scale, from the LONG side, so the aspect ratio survives.
    let longest = src.width.max(src.height);
    let scale = side as f64 / longest as f64;
    lb.scale = scale;
    // The resized extent, rounded. Round-half-away-from-zero so that an
    // image which is exactly twice the side comes out at exactly half, and
    // the padded region is symmetric to the pixel.
    let round = |v: f64| -> i64 { (if v >= 0.0 { v + 0.5 } else { v - 0.5 }) as i64 };
    let mut rw = round(src.width as f64 * scale);
    let mut rh = round(src.height as f64 * scale);
    if rw < 1 {
        rw = 1;
    }
    if rh < 1 {
        rh = 1;
    }
    // A rounding artefact can push the long side one pixel past the canvas.
    if rw > side {
        rw = side;
    }
    if rh > side {
        rh = side;
    }
    lb.pad_x = (side - rw) / 2;
    lb.pad_y = (side - rh) / 2;

    let scaled = if src.width == rw && src.height == rh {
        src.clone()
    } else {
        // BILINEAR, and only BILINEAR: the pose pipeline's letterbox is a
        // bilinear resize, and the ViT's choices were made for a classifier
        // that does not care about coordinates.
        vit_image::resize_to(src, rw, rh, Resample::Bilinear)?
    };

    let plane = (side * side) as usize;
    let mut out = Tensor {
        c: 3,
        h: side,
        w: side,
        d: vec![0.0f32; 3 * plane],
    };
    for c in 0..3usize {
        if g.std_dev[c] == 0.0 {
            return Err(format!("pose front end: image_std[{c}] is zero"));
        }
    }
    // The pad is applied as a PIXEL VALUE and then normalised like any
    // other, so it becomes (114/255 - mean)/std -- which is 0 for YOLOv8's
    // mean 0 and std 1. Filling the normalised tensor with zeros instead
    // would be a different pad for any checkpoint whose mean is not zero.
    let pad_n = (PAD_VALUE as f64 / 255.0 - g.mean[0] as f64) / g.std_dev[0] as f64;
    for i in 0..(3 * plane) {
        out.d[i] = pad_n as f32;
    }
    for c in 0..3usize {
        let mu = g.mean[c] as f64;
        let sd = g.std_dev[c] as f64;
        let base = c * plane;
        for y in 0..rh {
            let py = (y + lb.pad_y) as usize;
            for x in 0..rw as usize {
                // The SCALED image's width, not the source's: the row being
                // read is the scaled raster's.
                let v = scaled.rgb[(y as usize * rw as usize + x) * 3 + c] as f64 / 255.0;
                out.d[base + py * side as usize + x + lb.pad_x as usize] = ((v - mu) / sd) as f32;
            }
        }
    }
    Ok((out, lb))
}

/// The letterbox inverse, for the x axis.
fn invert(v: f64, lb: &Letterbox, src_len: i64) -> f64 {
    let s = (v - lb.pad_x as f64) / lb.scale;
    s.min((src_len - 1) as f64).max(0.0)
}

/// The letterbox inverse, for the y axis.
fn invert_y(v: f64, lb: &Letterbox, src_len: i64) -> f64 {
    let s = (v - lb.pad_y as f64) / lb.scale;
    s.min((src_len - 1) as f64).max(0.0)
}

/// Intersection over union of two xyxy boxes.
fn iou(a: &Person, b: &Person) -> f64 {
    let w = a.x2.min(b.x2) - a.x1.max(b.x1);
    let h = a.y2.min(b.y2) - a.y1.max(b.y1);
    if w <= 0.0 || h <= 0.0 {
        return 0.0;
    }
    let inter = w * h;
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    let uni = area_a + area_b - inter;
    // Two degenerate boxes give uni == 0. IoU is undefined there and the
    // only non-NaN answer that does not depend on the order of the two
    // boxes is 0, i.e. "they do not suppress each other".
    if uni > 0.0 { inter / uni } else { 0.0 }
}

/// The head decode: the threshold, the keypoints and the NMS.
///
/// Everything read here is CANONICAL: xyxy corners in letterbox pixels, the
/// score and the visibility already probabilities, keypoints in letterbox
/// pixels. The head is what makes it so, and the point of putting the
/// exporter's conventions there and not here is that this function has
/// nothing to be wrong about.
pub fn decode(
    head: &Tensor,
    g: &Geometry,
    lb: &Letterbox,
    p: &DecodeParams,
) -> Result<Vec<Person>, String> {
    let c = g.out_channels();
    let cells = g.n_anchors();
    if head.c != c || head.h * head.w != cells {
        return Err(format!(
            "pose decode: the head tensor is [{}, {}, {}] and the geometry \
             describes {c} values for {cells} cells.",
            head.c, head.h, head.w
        ));
    }
    let k_cls = 4i64;
    let k_kpt0 = 4 + g.num_classes;
    let mut cand: Vec<Person> = Vec::new();
    let mut offset = 0i64;
    for lv in 0..g.n_levels() {
        let side = g.grid[lv];
        for _ in 0..(side * side) {
            let cell = offset as usize;
            offset += 1;
            // The canonical tensor is [C, cells] -- channels first -- so one
            // cell's C values are STRIDED by `cells`, not adjacent.
            let at = |ch: i64| -> f64 { head.chw(ch)[cell] as f64 };
            let score = at(k_cls);
            if score < p.conf {
                continue;
            }
            if cand.len() as i64 >= p.max_nms {
                break;
            }
            let mut per = Person {
                score,
                anchor: offset - 1,
                level: lv as i64,
                ..Person::default()
            };
            per.x1 = invert(at(0), lb, lb.src_w);
            per.y1 = invert_y(at(1), lb, lb.src_h);
            per.x2 = invert(at(2), lb, lb.src_w);
            per.y2 = invert_y(at(3), lb, lb.src_h);
            for k in 0..g.num_keypoints {
                let v = at(k_kpt0 + k * 3 + 2);
                per.keypoints.push(Keypoint {
                    x: invert(at(k_kpt0 + k * 3), lb, lb.src_w),
                    y: invert_y(at(k_kpt0 + k * 3 + 1), lb, lb.src_h),
                    v,
                    visible: v >= p.keypoint,
                    index: k as i32,
                });
            }
            cand.push(per);
        }
    }
    // NMS, greedy by score. The sort is STABLE on the score, so that two
    // anchors with the same score suppress each other in ANCHOR order rather
    // than in whatever order the sort happened to produce.
    cand.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<Person> = Vec::new();
    let mut dead = vec![false; cand.len()];
    let mut i = 0usize;
    while i < cand.len() && (kept.len() as i64) < p.max_det {
        if dead[i] {
            i += 1;
            continue;
        }
        kept.push(cand[i].clone());
        for j in i + 1..cand.len() {
            if dead[j] {
                continue;
            }
            if iou(&cand[i], &cand[j]) > p.iou {
                dead[j] = true;
            }
        }
        i += 1;
    }
    Ok(kept)
}

/// What one detection produced.
#[derive(Debug, Clone, Default)]
pub struct Detection {
    /// The people, in the caller's pixels.
    pub people: Vec<Person>,
    /// Where the time went, per request.
    pub front_end_s: f64,
    pub network_s: f64,
    pub decode_s: f64,
    pub total_s: f64,
    /// The SOURCE image's size, so a client can scale.
    pub width: i64,
    pub height: i64,
    /// The front end's transform.
    pub scale: f64,
    pub pad_x: i64,
    pub pad_y: i64,
    /// How many array dispatches there were.
    pub dispatches: i64,
    /// How many convolutions ran on the host.
    pub convs_host: i64,
}

/// The whole answer, as the JSON object.
///
/// The C++'s shape, in the C++'s order. The image's label is escaped the
/// way the C++'s escapes it -- the quote, the backslash and the newline --
/// and the numbers are the C++'s own widths.
pub fn result_json(
    r: &Detection,
    g: &Geometry,
    p: &DecodeParams,
    image_label: &str,
    array: bool,
) -> String {
    let mut out = String::from("{\n  \"image\": \"");
    for c in image_label.chars() {
        if c == '"' || c == '\\' {
            out.push('\\');
        }
        if c == '\n' {
            out.push_str("\\n");
            continue;
        }
        out.push(c);
    }
    out.push_str(&format!(
        "\",\n  \"width\": {},\n  \"height\": {},\n  \"input_size\": {},\n  \
         \"letterbox\": {{\"scale\": {:.6}, \"pad_x\": {}, \"pad_y\": {}}},\n  \
         \"thresholds\": {{\"conf\": {:.3}, \"iou\": {:.3}, \"kpt\": {:.3}, \
         \"max_det\": {}}},\n  \
         \"backend\": {{\"conv\": \"{}\", \"dispatches\": {}}},\n  \
         \"timing_s\": {{\"front_end\": {:.4}, \"network\": {:.4}, \"decode\": \
         {:.4}, \"total\": {:.4}}},\n  \"landmarks\": [",
        r.width,
        r.height,
        g.input_size,
        r.scale,
        r.pad_x,
        r.pad_y,
        p.conf,
        p.iou,
        p.keypoint,
        p.max_det,
        if array { "npu" } else { "host" },
        r.dispatches,
        r.front_end_s,
        r.network_s,
        r.decode_s,
        r.total_s,
    ));
    for (i, person) in r.people.iter().enumerate() {
        out.push_str(if i > 0 { ",\n    " } else { "\n    " });
        out.push_str(&person_json(person, i as i64));
    }
    out.push_str(if r.people.is_empty() {
        "],\n"
    } else {
        "\n  ],\n"
    });
    // The 17 names as 17 STRINGS, not one comma-joined line: the array form
    // is the one a client can index, and the CSV exists for humans reading
    // stderr. Both come from the same table, so the two cannot spell the
    // order differently.
    out.push_str("  \"skeleton\": {\"keypoints\": [");
    for (i, name) in SKELETON_NAMES.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("\"{name}\""));
    }
    out.push_str("], \"edges\": [");
    for (i, edge) in SKELETON_EDGES.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("\"{edge}\""));
    }
    out.push_str("]}\n}\n");
    out
}

/// One person as one JSON object, no enclosing array.
///
/// A pixel coordinate is `%.1f`: at 640px the letterbox's own quantisation
/// is already one model pixel, so a tenth of a source pixel is finer than
/// the input and saying more of it would be a claim about a precision the
/// model does not have. The score keeps three decimals.
pub fn person_json(p: &Person, id: i64) -> String {
    let mut s = format!(
        "{{\"id\":{id},\"score\":{:.3},\"box\":{{\"x1\":{:.1},\"y1\":{:.1},\"x2\":{:.1},\"y2\":{:.1}}},\"level\":{},\"anchor\":{},\"keypoints\":[",
        p.score, p.x1, p.y1, p.x2, p.y2, p.level, p.anchor
    );
    for (i, k) in p.keypoints.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!(
            "{{\"x\":{:.1},\"y\":{:.1},\"score\":{:.3},\"visible\":{},\"index\":{},\"name\":\"{}\"}}",
            k.x,
            k.y,
            k.v,
            if k.visible { "true" } else { "false" },
            k.index,
            SKELETON_NAMES[k.index as usize]
        ));
    }
    s.push_str("]}");
    s
}
