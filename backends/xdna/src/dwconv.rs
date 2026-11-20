//! Depthwise 3x3 conv on the array.
//!
//! A separate design set from `gemm_rtp`: the core program is
//! `kernels/dwconv9x3_bf16.cc`, not the matrix-multiply kernel, so it cannot
//! live as another stream in the GEMM xclbin. One dispatch computes 16
//! channels x 1024 outputs; larger layers are chunked here.
//!
//! The weights are NOT staged as a panel in the container. Every depthwise
//! layer's `[C,1,3,3]` weights and `[C]` bias are already in the container as
//! ordinary tensors (the dense-conv packer stages panels only for GEMMs), so
//! this module builds the kernel's `[C,10]` bf16 tap+bias table at open time.

use std::collections::HashMap;
use std::path::Path;

use npum_format::{Container, bf16_to_f32, f32_to_bf16};
use npum_model::array::{ArrayError, Result};

use crate::array::ArrayDesign;
use crate::transport::Transport;

/// The design's fixed geometry. Must match `tools/export/export_dwconv.py`.
pub const CHANNELS: usize = 16;
pub const WIDTH: usize = 1024;
const STRIDE_EXTRA: usize = 32;
const ROW_STRIDE: usize = WIDTH + STRIDE_EXTRA;
/// bf16 elements per channel in the design's input object: three padded
/// rows, ten tap+bias values, then alignment padding.
pub const CHAN_IN: usize = ((3 * ROW_STRIDE + 10 + 15) / 16) * 16;
const BUFFER_IN: usize = 0;
const BUFFER_OUT: usize = 1;
const SLOT: usize = 1;

pub struct Dwconv<T: Transport> {
    design: ArrayDesign<T>,
    /// `[C,10]` bf16 tap+bias tables, keyed by the container's weight name.
    panels: HashMap<String, Vec<u8>>,
    dispatches: usize,
}

impl<T: Transport> Dwconv<T> {
    pub fn open(container: &Path, design: ArrayDesign<T>) -> std::result::Result<Self, String> {
        let file = Container::open(container)
            .map_err(|e| format!("{}: {e}", container.display()))?;
        let mut panels = HashMap::new();
        for entry in file.directory().tensors() {
            if !entry.name.ends_with(".w") || entry.logical_shape.len() != 4 {
                continue;
            }
            let s = &entry.logical_shape;
            if s[1] != 1 || s[2] != 3 || s[3] != 3 {
                continue;
            }
            let c = s[0] as usize;
            let bias_name = format!("{}.b", &entry.name[..entry.name.len() - 2]);
            let wbytes = file.tensor_bytes(&entry.name).map_err(|e| e.to_string())?;
            // Some recognisers have no bias at all (the container simply
            // has no tensor for it). Zero is the mathematical value there.
            let zeros = vec![0u8; c * 4];
            let bbytes = file.tensor_bytes(&bias_name).unwrap_or(&zeros);
            if wbytes.len() != c * 9 * 4 || bbytes.len() != c * 4 {
                return Err(format!(
                    "{}: depthwise weights are {} bytes and bias {}; expected {} and {}",
                    entry.name, wbytes.len(), bbytes.len(), c * 9 * 4, c * 4
                ));
            }
            let mut panel = vec![0u8; c * 10 * 2];
            for ch in 0..c {
                for k in 0..9 {
                    let bits = u32::from_le_bytes(
                        wbytes[(ch * 9 + k) * 4..][..4].try_into().unwrap());
                    panel[(ch * 10 + k) * 2..][..2]
                        .copy_from_slice(&f32_to_bf16(f32::from_bits(bits)).to_le_bytes());
                }
                let bits = u32::from_le_bytes(bbytes[ch * 4..][..4].try_into().unwrap());
                panel[(ch * 10 + 9) * 2..][..2]
                    .copy_from_slice(&f32_to_bf16(f32::from_bits(bits)).to_le_bytes());
            }
            panels.insert(entry.name.clone(), panel);
        }
        if panels.is_empty() {
            return Err(format!(
                "{}: --npu-ops dwconv needs 3x3 depthwise weights in the container, \
                 and this container has none",
                container.display()
            ));
        }
        if design.info().buffers.len() < 2 {
            return Err("dwconv design set needs A and C buffers".to_string());
        }
        Ok(Self { design, panels, dispatches: 0 })
    }

    pub fn dispatches(&self) -> usize {
        self.dispatches
    }

    pub fn has(&self, weight: &str) -> bool {
        self.panels.contains_key(weight)
    }

    /// Run one output row. `a` is `[channels, 3, ROW_STRIDE]` f32, already
    /// gathered by the caller; `out` receives `[channels, outputs]` f32.
    pub fn run(
        &mut self,
        weight: &str,
        a: &[f32],
        channels: usize,
        outputs: usize,
        out: &mut [f32],
    ) -> Result<usize> {
        let panel = self.panels.get(weight).ok_or_else(|| {
            ArrayError(format!("the array has no depthwise weights named {weight}"))
        })?;
        if channels == 0 {
            return Err(ArrayError(format!("{weight}: zero channels")));
        }
        // The core grid computes 16 channels at a time. Real models have
        // 24, 48, ... so pad the LAST block with zero weights and ignore its
        // outputs rather than refuse a layer the design can compute.
        let padded = ((channels + CHANNELS - 1) / CHANNELS) * CHANNELS;
        if outputs > WIDTH || outputs == 0 {
            return Err(ArrayError(format!(
                "{weight}: {outputs} outputs, and the design holds at most {WIDTH}"
            )));
        }
        let need_a = channels * 3 * ROW_STRIDE;
        if a.len() < need_a {
            return Err(ArrayError(format!(
                "{weight}: gathered input has {} values, need {need_a}", a.len()
            )));
        }
        if out.len() < channels * outputs {
            return Err(ArrayError(format!(
                "{weight}: output has {} values, need {}", out.len(), channels * outputs
            )));
        }

        let mut chunks = 0;
        for base in (0..padded).step_by(CHANNELS) {
            let mut x = vec![0u8; CHANNELS * CHAN_IN * 2];
            for c in 0..CHANNELS {
                let channel = base + c;
                if channel >= channels {
                    continue;
                }
                let src = &a[channel * 3 * ROW_STRIDE..][..3 * ROW_STRIDE];
                let dst = &mut x[c * CHAN_IN * 2..][..3 * ROW_STRIDE * 2];
                for (i, v) in src.iter().enumerate() {
                    dst[i * 2..][..2].copy_from_slice(&f32_to_bf16(*v).to_le_bytes());
                }
                let w = &panel[channel * 10 * 2..][..10 * 2];
                x[(c * CHAN_IN + 3 * ROW_STRIDE) * 2..][..10 * 2]
                    .copy_from_slice(w);
            }
            self.design
                .write_bound(BUFFER_IN, &x)
                .map_err(|e| ArrayError(e.to_string()))?;
            self.design
                .dispatch(SLOT)
                .map_err(|e| ArrayError(e.to_string()))?;
            let mut y = vec![0u8; CHANNELS * WIDTH * 2];
            self.design
                .read(BUFFER_OUT, &mut y)
                .map_err(|e| ArrayError(e.to_string()))?;
            for c in 0..CHANNELS {
                let channel = base + c;
                if channel >= channels {
                    break;
                }
                for o in 0..outputs {
                    let bits = u16::from_le_bytes(y[(c * WIDTH + o) * 2..][..2].try_into().unwrap());
                    out[channel * outputs + o] = bf16_to_f32(bits);
                }
            }
            self.dispatches += 1;
            chunks += 1;
        }
        Ok(chunks)
    }
}
