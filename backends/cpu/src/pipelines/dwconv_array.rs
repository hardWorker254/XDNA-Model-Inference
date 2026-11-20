//! Host-side gather for the array's 3x3 depthwise kernel.
//!
//! The AIE core computes, for one output row y and every channel,
//!
//!   out[x] = bias + sum_{dy=0..2} sum_{j=0..2} w[dy,j] * in[dy][x+j]
//!
//! where `in[dy][p]` is the input at `(y*stride_h+dy-pad_top,
//! p*stride_w-pad_left)`. This module builds those three padded rows for all
//! channels and feeds them to `ArrayConvs::dwconv`, one output row per
//! dispatch. The kernel itself has no notion of padding or stride: the host
//! gather is where the model's geometry is applied, exactly as it is on the
//! host path.
//!
//! Activation is NOT applied here: the kernel adds the bias, and the caller
//! applies the fused activation to the returned pre-activation values.

use npum_model::array::{ArrayConvs, ArrayError};

/// The design's row stride: 1024 outputs plus 32 elements of load slack.
pub const ROW_STRIDE: usize = 1056;
pub const MAX_OUTPUTS: usize = 1024;

/// Run one depthwise layer on the array, returning pre-activation f32 in
/// `[c, oh, ow]` order. `input` is `[c, in_h, in_w]`.
pub fn run(
    array: &mut dyn ArrayConvs,
    weight: &str,
    input: &[f32],
    in_h: usize,
    in_w: usize,
    channels: usize,
    oh: usize,
    ow: usize,
    stride_h: i64,
    stride_w: i64,
    pad: [i64; 4],
) -> Result<Vec<f32>, ArrayError> {
    if ow == 0 || oh == 0 || ow > MAX_OUTPUTS {
        return Err(ArrayError(format!(
            "{weight}: {ow} outputs per row, and the dwconv design holds at most {MAX_OUTPUTS}"
        )));
    }
    if channels == 0 {
        return Err(ArrayError(format!("{weight}: zero channels")));
    }
    let plane = in_h * in_w;
    if input.len() < channels * plane {
        return Err(ArrayError(format!(
            "{weight}: input has {} values, and {channels}x{in_h}x{in_w} need {}",
            input.len(),
            channels * plane
        )));
    }
    let mut result = vec![0.0f32; channels * oh * ow];
    let mut gathered = vec![0.0f32; channels * 3 * ROW_STRIDE];
    let mut row_out = vec![0.0f32; channels * ow];
    for y in 0..oh {
        gathered.fill(0.0);
        for c in 0..channels {
            for dy in 0..3usize {
                let sy = y as i64 * stride_h + dy as i64 - pad[0];
                if sy < 0 || sy >= in_h as i64 {
                    continue;
                }
                let base = c * 3 * ROW_STRIDE + dy * ROW_STRIDE;
                for p in 0..ROW_STRIDE {
                    let sx = p as i64 * stride_w - pad[1];
                    if sx >= 0 && sx < in_w as i64 {
                        gathered[base + p] =
                            input[c * plane + sy as usize * in_w + sx as usize];
                    }
                }
            }
        }
        row_out.fill(0.0);
        array.dwconv(weight, &gathered, channels, ow, ROW_STRIDE, &mut row_out)?;
        for c in 0..channels {
            let dst = &mut result[c * oh * ow + y * ow..][..ow];
            dst.copy_from_slice(&row_out[c * ow..][..ow]);
        }
    }
    Ok(result)
}
