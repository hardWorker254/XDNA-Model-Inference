//! The image front end: decode, resize, normalise, im2col.
//!
//! It produces the `[n_patches, patch_dim]` matrix that the patch
//! embedding GEMM consumes.
//!
//! # THE FOUR STEPS ARE SEPARATE FUNCTIONS ON PURPOSE
//!
//! Each one is a place where a "reasonable" shortcut produces a correctly
//! shaped wrong answer, so each is separately callable and separately
//! testable:
//!
//! * `decode_image_bytes` -- the format from the BYTES, not the extension;
//! * `resize_square` -- PIL's resample, downscale antialiased;
//! * `normalise` -- `(x/255 - mean) / std`, per channel, from the container;
//! * `im2col_patches` -- `[3,S,S]` to `[n_patches, 3*patch*patch]`.
//!
//! # THE RESIZE IS PIL'S, MEASURED AGAINST PIL
//!
//! PIL's filter is a SEQUENTIAL resample with a scale-adaptive support --
//! when downscaling, the kernel is widened by the shrink factor so it
//! low-passes instead of point-sampling. That widening is the whole
//! difference between a correct downscale and a moire of aliased pixels,
//! and 1/std then multiplies the error by 2.
//!
//! The index arithmetic is transcribed exactly:
//!
//! ```text
//! scale       = in_len / out_len
//! filterscale = max(1, scale)          <- the antialias widening
//! support     = kernel.support * filterscale
//! centre      = (i + 0.5) * scale
//! xmin        = int(centre - support + 0.5), clamped to [0, in_len)
//! xmax        = int(centre + support + 0.5), clamped to [0, in_len)
//! weight      = kernel.eval((x - centre + 0.5) / filterscale)
//! ```
//!
//! and the output is `sum(w*v)/sum(w)`, not a plain mean, so a pixel that
//! straddles an edge keeps its level.
//!
//! The accumulation is an FMA, and not a multiply and an add: the oracle
//! is built with `-march=x86-64-v3` and its compiler contracts
//! `acc += w * x` into one rounding. The tap count is small, but the
//! difference is one ulp per tap and a downscale has hundreds.
//!
//! # THE FILTERS ARE FIVE DIFFERENT FUNCTIONS
//!
//! The only thing they share is a name:
//!
//! ```text
//! BOX       1.0 on |x| <= 0.5                       (a RECTANGLE, not a cubic)
//! BILINEAR  1 - |x| on |x| < 1                      (a triangle)
//! BICUBIC   Keys a = -0.5 on |x| < 2                (the only cubic)
//! LANCZOS   sinc(x)*sinc(x/3) on |x| < 3             (a windowed sinc)
//! ```
//!
//! HAMMING and NEAREST are refused by name: this function claims
//! PIL-EQUIVALENCE, not "a smooth interpolation", and a claim that is not
//! measured is a guess about the model's input.

use crate::vit::Geometry;

/// An 8-bit interleaved RGB raster. `rgb` is width*height*3, row-major.
#[derive(Debug, Clone, Default)]
pub struct Image {
    /// The raster's width, in pixels.
    pub width: i64,
    /// The raster's height, in pixels.
    pub height: i64,
    /// The bytes, row-major, three per pixel.
    pub rgb: Vec<u8>,
}

impl Image {
    /// Whether the raster holds nothing.
    pub fn empty(&self) -> bool {
        self.width <= 0 || self.height <= 0 || self.rgb.is_empty()
    }
}

/// PIL's resample codes, as the container carries them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resample {
    /// 0: point-sampling. Refused.
    Nearest,
    /// 1: a windowed sinc.
    Lanczos,
    /// 2: a triangle.
    Bilinear,
    /// 3: Keys' cubic. What every shipped ViT checkpoint asks for.
    Bicubic,
    /// 4: a rectangle.
    Box,
    /// 5: refused.
    Hamming,
}

impl Resample {
    /// A code out of the container.
    pub fn from_code(code: i64) -> Option<Resample> {
        Some(match code {
            0 => Resample::Nearest,
            1 => Resample::Lanczos,
            2 => Resample::Bilinear,
            3 => Resample::Bicubic,
            4 => Resample::Box,
            5 => Resample::Hamming,
            _ => return None,
        })
    }
}

// ---------------------------------------------------------------------------
// The decode
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// libjpeg, through the shim. See `jpeg_shim.c`.
    fn npue_jpeg_decode(
        data: *const u8,
        length: std::ffi::c_ulong,
        out: *mut *mut u8,
        width: *mut i32,
        height: *mut i32,
        errbuf: *mut u8,
        errlen: std::ffi::c_ulong,
    ) -> i32;
    fn npue_free(p: *mut std::ffi::c_void);
}

/// Decode a JPEG into RGB.
///
/// The same libjpeg the C++ runtime calls, with the same one setting --
/// `out_color_space = JCS_RGB`, so a grayscale JPEG becomes three
/// channels. Everything else is libjpeg's defaults: the "fancy"
/// upsampling, the slow integer IDCT, libjpeg's own YCbCr -> RGB.
pub fn decode_jpeg(bytes: &[u8], name: &str) -> Result<Image, String> {
    let mut out: *mut u8 = std::ptr::null_mut();
    let mut width: i32 = 0;
    let mut height: i32 = 0;
    let mut err = [0u8; 256];
    let ok = unsafe {
        npue_jpeg_decode(
            bytes.as_ptr(),
            bytes.len() as std::ffi::c_ulong,
            &mut out,
            &mut width,
            &mut height,
            err.as_mut_ptr(),
            err.len() as std::ffi::c_ulong,
        )
    };
    if ok == 0 {
        let message = String::from_utf8_lossy(&err)
            .trim_end_matches('\0')
            .to_string();
        return Err(format!(
            "{name}: {}",
            if message.is_empty() {
                "JPEG decode failed"
            } else {
                &message
            }
        ));
    }
    let count = (width as usize) * (height as usize) * 3;
    let rgb = unsafe { std::slice::from_raw_parts(out, count) }.to_vec();
    unsafe { npue_free(out as *mut std::ffi::c_void) };
    Ok(Image {
        width: width as i64,
        height: height as i64,
        rgb,
    })
}

/// The PNG signature.
const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

/// Decode a PNG or a JPEG into RGB.
///
/// PNG and JPEG only, and both are DECIDED BY MAGIC BYTES: a `.png` that
/// holds a JPEG is common enough in the wild (a browser's `<img src>`
/// renamed, an exporter's bug) that a decoder keyed on the extension
/// returns an error the user cannot act on, or worse, succeeds on the wrong
/// bytes.
///
/// A PNG is REFUSED here, by name, and not decoded: the PNG reader is
/// libpng's, and the OCR family is where it is wired. Refusing is the
/// honest answer -- a decoder that guesses at a format reports a label for
/// a picture it may never have seen.
pub fn decode_image_bytes(bytes: &[u8], name: &str) -> Result<Image, String> {
    if bytes.len() < 8 {
        return Err(format!(
            "{name}: {} bytes, too short to be an image",
            bytes.len()
        ));
    }
    if bytes[..8] == PNG_MAGIC {
        return Err(format!(
            "{name}: a PNG. This runtime's image front end decodes JPEG; the PNG \
             reader is not wired here yet, and it will not guess at one."
        ));
    }
    if bytes[0] == 0xFF && bytes[1] == 0xD8 {
        return decode_jpeg(bytes, name);
    }
    Err(format!(
        "{name}: not a PNG (signature) and not a JPEG (SOI marker). This runtime \
         reads those two and refuses anything else rather than guessing -- \
         re-save the image as PNG or JPEG."
    ))
}

/// Decode an image file.
pub fn decode_image(path: &std::path::Path) -> Result<Image, String> {
    let bytes = std::fs::read(path)
        .map_err(|_| format!("cannot open image file {}", path.display()))?;
    decode_image_bytes(&bytes, &path.display().to_string())
}

// ---------------------------------------------------------------------------
// The resize
// ---------------------------------------------------------------------------

/// A filter kernel: the half-width in SOURCE pixels, and the kernel.
struct Kernel {
    support: f64,
    eval: fn(f64) -> f64,
}

fn box_eval(x: f64) -> f64 {
    let x = x.abs();
    if x <= 0.5 { 1.0 } else { 0.0 }
}

fn bilinear_eval(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 { 1.0 - x } else { 0.0 }
}

fn bicubic_eval(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((-0.5 + 2.0) * x - (-0.5 + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * (-0.5)
    } else {
        0.0
    }
}

fn lanczos_eval(x: f64) -> f64 {
    let x = x.abs();
    if x < 3.0 {
        let pix = std::f64::consts::PI * x;
        (pix.sin() * (pix / 3.0).sin()) / (pix * (pix / 3.0))
    } else {
        0.0
    }
}

/// The kernel for a code.
///
/// NEAREST is the identity filter at zero width, which the generic path
/// below handles correctly with support 0.5 and the rectangle kernel -- see
/// the refusal in `resize_to`; NEAREST is refused there rather than
/// approximated here.
fn kernel_for(how: Resample) -> Kernel {
    match how {
        Resample::Nearest | Resample::Box => Kernel {
            support: 0.5,
            eval: box_eval,
        },
        Resample::Bilinear | Resample::Hamming => Kernel {
            support: 1.0,
            eval: bilinear_eval,
        },
        Resample::Bicubic => Kernel {
            support: 2.0,
            eval: bicubic_eval,
        },
        Resample::Lanczos => Kernel {
            support: 3.0,
            eval: lanczos_eval,
        },
    }
}

/// The filter taps for every output position along ONE axis.
struct AxisPlan {
    /// The first source index, per output index.
    xmin: Vec<i64>,
    /// Where this output index's taps start in `w`.
    off: Vec<usize>,
    /// The tap count.
    n: Vec<i64>,
    /// Every output index's taps, concatenated.
    w: Vec<f64>,
    /// The sum of that index's taps.
    ww: Vec<f64>,
}

/// Build the axis plan: PIL's index arithmetic, exactly.
fn make_axis_plan(in_len: i64, out_len: i64, kern: &Kernel, filterscale: f64) -> AxisPlan {
    let scale = in_len as f64 / out_len as f64;
    let support = kern.support * filterscale;
    let inv_filter = 1.0 / filterscale;
    // Worst case one source pixel contributes to every output pixel inside a
    // support of `support`. It is computed from kern.support rather than
    // written as "2.0 * filterscale", which was BICUBIC's number and would
    // have silently truncated a LANCZOS window (support 3.0) by a third.
    let kmax = in_len.min((support * 2.0).ceil() as i64 + 2);
    let mut plan = AxisPlan {
        xmin: vec![0; out_len as usize],
        off: vec![0; out_len as usize],
        n: vec![0; out_len as usize],
        w: Vec::new(),
        ww: vec![1.0; out_len as usize],
    };
    plan.w.reserve(out_len as usize * kmax as usize);
    for i in 0..out_len {
        let centre = (i as f64 + 0.5) * scale;
        let mut xmin = (centre - support + 0.5) as i64;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (centre + support + 0.5) as i64;
        if xmax > in_len {
            xmax = in_len;
        }
        let n = xmax - xmin;
        plan.xmin[i as usize] = xmin;
        plan.off[i as usize] = plan.w.len();
        if n <= 0 {
            // Cannot happen for support >= 0.5 and scale > 0, but a
            // zero-weight row would divide by zero below, and a NaN pixel is
            // the worst thing this function can produce.
            plan.n[i as usize] = 0;
            plan.ww[i as usize] = 1.0;
            continue;
        }
        if n > kmax {
            return plan;
        }
        let mut ww = 0.0f64;
        for x in 0..n {
            let v = (kern.eval)(((x + xmin) as f64 - centre + 0.5) * inv_filter);
            plan.w.push(v);
            ww += v;
        }
        plan.n[i as usize] = n;
        plan.ww[i as usize] = ww;
        if plan.ww[i as usize] == 0.0 {
            plan.ww[i as usize] = 1.0;
        }
    }
    plan
}

/// The 8-bit write, shared by both passes so the rounding is stated once.
///
/// Round half away from zero, then clamp.
fn resample_round(acc: f64, ww: f64) -> u8 {
    let v = acc / ww;
    let r = if v < 0.0 {
        v - 0.5
    } else if v > 0.0 {
        v + 0.5
    } else {
        v
    };
    let q = r as i64;
    q.clamp(0, 255) as u8
}

/// ONE SOURCE LINE, all output positions along the axis.
fn apply_axis_line(
    input: &[u8],
    output: &mut [u8],
    stride_in: i64,
    stride_out: i64,
    nchan: i64,
    plan: &AxisPlan,
    out_len: i64,
) {
    for i in 0..out_len {
        let oi = i as usize;
        let n = plan.n[oi];
        let xmin = plan.xmin[oi];
        if n <= 0 {
            for c in 0..nchan {
                output[(i * stride_out + c) as usize] = 0;
            }
            continue;
        }
        let w = &plan.w[plan.off[oi]..plan.off[oi] + n as usize];
        let ww = plan.ww[oi];
        for c in 0..nchan {
            let mut acc = 0.0f64;
            for x in 0..n {
                let value = input[((xmin + x) * stride_in + c) as usize] as f64;
                // An FMA, and not a multiply and an add: the oracle's
                // compiler contracts `acc += w * x` into one rounding.
                acc = w[x as usize].mul_add(value, acc);
            }
            output[(i * stride_out + c) as usize] = resample_round(acc, ww);
        }
    }
}

/// ONE OUTPUT POSITION, across every source line -- the vertical pass.
fn apply_axis_across(
    input: &[u8],
    output: &mut [u8],
    tap_stride: i64,
    nchan: i64,
    nlines: i64,
    plan: &AxisPlan,
    i: i64,
) {
    let oi = i as usize;
    let n = plan.n[oi];
    let xmin = plan.xmin[oi];
    if n <= 0 {
        for l in 0..nlines {
            for c in 0..nchan {
                output[((i * nlines + l) * nchan + c) as usize] = 0;
            }
        }
        return;
    }
    let w = &plan.w[plan.off[oi]..plan.off[oi] + n as usize];
    let ww = plan.ww[oi];
    for l in 0..nlines {
        for c in 0..nchan {
            let mut acc = 0.0f64;
            for x in 0..n {
                let value = input[(l * nchan + c + (xmin + x) * tap_stride) as usize] as f64;
                acc = w[x as usize].mul_add(value, acc);
            }
            output[((i * nlines + l) * nchan + c) as usize] = resample_round(acc, ww);
        }
    }
}

/// Resize to a size, with PIL's resample.
///
/// Horizontal first, into an `h x out_w` buffer, then vertical: the
/// horizontal pass is the wider of the two and doing it first keeps the
/// intermediate at `h*out_w` rather than `out_w*out_h`.
pub fn resize_to(src: &Image, out_w: i64, out_h: i64, how: Resample) -> Result<Image, String> {
    if out_w <= 0 || out_h <= 0 {
        return Err(format!("resize to {out_w}x{out_h} pixels"));
    }
    if src.empty() {
        return Err("resize: the source image decoded to nothing".into());
    }
    // PIL applies no filter when the size already matches, and neither do
    // we: a filter here would be a half-LSB change to a pipeline that did
    // not ask for one.
    if src.width == out_w && src.height == out_h {
        return Ok(src.clone());
    }
    if how == Resample::Nearest {
        return Err(
            "resize: NEAREST is not implemented. It point-samples a downscale (no \
             filter widening) and the aliasing it produces is amplified by 1/std \
             before the classifier sees it. LANCZOS, BILINEAR, BICUBIC and BOX are \
             implemented and measured against PIL.Image.resize on the same bytes; \
             BICUBIC is what this model was trained with."
                .into(),
        );
    }
    if how == Resample::Hamming {
        return Err(
            "resize: HAMMING (PIL.Image code 5) is refused. LANCZOS, BILINEAR, \
             BICUBIC and BOX are implemented and measured against PIL.Image.resize \
             on the same bytes -- within one 8-bit LSB, exactly on a constant \
             raster. HAMMING was measured against PIL the same way and does not \
             agree: worst channel difference 65 on a random raster. This build \
             will not ship a resampler that is close to PIL rather than equal to \
             it, because a smooth filter that is wrong is a confident label about \
             a picture the model never saw."
                .into(),
        );
    }

    let w = src.width;
    let h = src.height;
    let kern = kernel_for(how);
    let mut tmp = vec![0u8; (h * out_w * 3) as usize];
    let fscale_x = (w as f64 / out_w as f64).max(1.0);
    let px = make_axis_plan(w, out_w, &kern, fscale_x);
    for y in 0..h {
        let line_in = &src.rgb[(y * w * 3) as usize..];
        let line_out = &mut tmp[(y * out_w * 3) as usize..];
        apply_axis_line(line_in, line_out, 3, 3, 3, &px, out_w);
    }

    let mut out = Image {
        width: out_w,
        height: out_h,
        rgb: vec![0u8; (out_w * out_h * 3) as usize],
    };
    let fscale_y = (h as f64 / out_h as f64).max(1.0);
    let py = make_axis_plan(h, out_h, &kern, fscale_y);
    for i in 0..out_h {
        apply_axis_across(&tmp, &mut out.rgb, out_w * 3, 3, out_w, &py, i);
    }
    Ok(out)
}

/// The square case, which is all a ViT asks for: `resize_to` with one size
/// twice.
pub fn resize_square(src: &Image, side: i64, how: Resample) -> Result<Image, String> {
    resize_to(src, side, side, how)
}

// ---------------------------------------------------------------------------
// The normalise and the im2col
// ---------------------------------------------------------------------------

/// `(x/255 - mean) / std`, per channel, from the container.
///
/// The values are laid out `[3, S, S]`, channel slowest.
pub fn normalise(im: &Image, g: &Geometry) -> Result<Vec<f32>, String> {
    if im.width != g.image_size || im.height != g.image_size {
        return Err(format!(
            "normalise: image is {}x{}px, the container's image_size is {}. \
             Resize first; the mean/std are per channel of a grid this size.",
            im.width, im.height, g.image_size
        ));
    }
    if g.mean.len() != 3 || g.std_dev.len() != 3 {
        return Err(format!(
            "normalise: this build's image front end is three channels (RGB) and \
             the container carries {} means.",
            g.mean.len()
        ));
    }
    let n = g.image_size * g.image_size;
    let mut out = vec![0.0f32; (g.num_channels * g.image_size * g.image_size) as usize];
    for i in 0..n {
        for c in 0..3usize {
            // A real DIVIDE by 255, not a multiply by (1/255f): the two
            // disagree for 176 of the 256 pixel values, by exactly 1 ULP,
            // and the reference (numpy, and therefore transformers) does
            // `a / np.float32(255.0)`.
            let v = im.rgb[(i * 3 + c as i64) as usize] as f32 / 255.0f32;
            out[(c as i64 * n + i) as usize] = (v - g.mean[c]) / g.std_dev[c];
        }
    }
    Ok(out)
}

/// `[3,S,S]` to `[n_patches, 3*patch*patch]`.
///
/// The layout is `(c, i, j) -> (c*patch + i)*patch + j`, and the rows are
/// in `(patch_row, patch_col)` order.
pub fn im2col_patches(chw: &[f32], g: &Geometry) -> Vec<f32> {
    let n = g.image_size / g.patch_size;
    let mut out = vec![0.0f32; (g.n_patches * g.patch_dim) as usize];
    for p in 0..n {
        for q in 0..n {
            let base = (p * n + q) * g.patch_dim;
            for c in 0..g.num_channels {
                for i in 0..g.patch_size {
                    for j in 0..g.patch_size {
                        let dst = base + (c * g.patch_size + i) * g.patch_size + j;
                        let src = (c * g.image_size + p * g.patch_size + i) * g.image_size
                            + q * g.patch_size
                            + j;
                        out[dst as usize] = chw[src as usize];
                    }
                }
            }
        }
    }
    out
}

/// The whole front end: decode, resize, normalise, im2col.
pub fn preprocess(im: &Image, g: &Geometry) -> Result<Vec<f32>, String> {
    let sq = resize_square(im, g.image_size, g.resample)?;
    let chw = normalise(&sq, g)?;
    Ok(im2col_patches(&chw, g))
}
