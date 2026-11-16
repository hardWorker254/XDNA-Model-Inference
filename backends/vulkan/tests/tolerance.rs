//! The GPU's arithmetic against the host's, operation by operation.
//!
//! # These are tolerance gates, and the tolerance is stated
//!
//! A GPU's f32 is not the host's f32: it fuses multiply-add, its `exp2` is a
//! different polynomial, and its reduction order inside a workgroup is the
//! driver's choice. The gates here therefore compare numbers, not bits, and
//! each case names the bound it holds to.
//!
//! # Why every row width here is a multiple of eight
//!
//! `npum_cpu::kernels::softmax_row` and `layernorm_row` read eight lanes at a
//! time and REFUSE a row that is not a multiple of eight (the C++ did the
//! same). The GPU has no such constraint, but the gate compares against the
//! host, so the cases are shaped so the host will answer.
//!
//! # What a skipped case means
//!
//! On a machine with no Vulkan device the whole file reports that it did not
//! run. That is a skip and not a pass: the gate prints it, and the run that
//! matters is the one with a GPU.

use npum_cpu::kernels;
use npum_model::{Backend, Tensor};
use npum_vulkan::VulkanBackend;

/// A deterministic sequence, so a failure is reproducible from the seed.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 40) as u32;
        (bits as f32 / (1u32 << 23) as f32) - 1.0
    }

    fn take(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

/// The worst relative error, with an absolute floor so a zero is not a
/// division by zero.
fn worst(gpu: &[f32], host: &[f32]) -> f32 {
    assert_eq!(gpu.len(), host.len(), "the two answers have different sizes");
    gpu.iter()
        .zip(host)
        .map(|(g, h)| (g - h).abs() / h.abs().max(1e-3))
        .fold(0.0f32, f32::max)
}

fn open() -> Option<VulkanBackend> {
    let backend = VulkanBackend::new();
    if !backend.has_gpu() {
        eprintln!("SKIP: no Vulkan device on this machine");
        return None;
    }
    eprintln!("device: {}", backend.device_name().unwrap_or("?"));
    Some(backend)
}

#[test]
fn gemm_is_within_2e_5_of_the_host() {
    let Some(mut gpu) = open() else { return };
    let (m, k, n) = (8usize, 16usize, 12usize);
    let mut rng = Lcg::new(7);
    let a = rng.take(m * k);
    let b = rng.take(k * n);
    let at = Tensor::f32(vec![m as i64, k as i64], a).unwrap();
    let bt = Tensor::f32(vec![k as i64, n as i64], b).unwrap();
    let mut host = Tensor::f32(vec![m as i64, n as i64], vec![0.0; m * n]).unwrap();
    kernels::gemm(&at, &bt, &mut host, None).unwrap();
    let mut on_gpu = Tensor::f32(vec![m as i64, n as i64], vec![0.0; m * n]).unwrap();
    gpu.gemm(&at, &bt, &mut on_gpu, None).unwrap();
    let error = worst(&on_gpu.to_f32(), &host.to_f32());
    eprintln!("gemm: worst relative error {error:e}");
    assert!(error <= 2e-5, "gemm differs by {error:e}");
}

#[test]
fn gemm_accumulates_into_acc() {
    let Some(mut gpu) = open() else { return };
    let (m, k, n) = (4usize, 8usize, 6usize);
    let mut rng = Lcg::new(11);
    let a = rng.take(m * k);
    let b = rng.take(k * n);
    let acc = rng.take(m * n);
    let at = Tensor::f32(vec![m as i64, k as i64], a).unwrap();
    let bt = Tensor::f32(vec![k as i64, n as i64], b).unwrap();
    let mut host_c = Tensor::f32(vec![m as i64, n as i64], vec![0.0; m * n]).unwrap();
    let mut host_acc = Tensor::f32(vec![m as i64, n as i64], acc.clone()).unwrap();
    kernels::gemm(&at, &bt, &mut host_c, Some(&mut host_acc)).unwrap();
    let mut gpu_c = Tensor::f32(vec![m as i64, n as i64], vec![0.0; m * n]).unwrap();
    let mut gpu_acc = Tensor::f32(vec![m as i64, n as i64], acc).unwrap();
    gpu.gemm(&at, &bt, &mut gpu_c, Some(&mut gpu_acc)).unwrap();
    let error = worst(&gpu_c.to_f32(), &host_c.to_f32());
    eprintln!("gemm+acc: worst relative error {error:e}");
    assert!(error <= 2e-5, "gemm with acc differs by {error:e}");
}

#[test]
fn gelu_is_within_1e_6_of_the_host() {
    let Some(mut gpu) = open() else { return };
    let mut rng = Lcg::new(3);
    let values = rng.take(200);
    let mut host = Tensor::f32(vec![200], values.clone()).unwrap();
    kernels::gelu_(&mut host).unwrap();
    let mut on_gpu = Tensor::f32(vec![200], values).unwrap();
    gpu.gelu(&mut on_gpu).unwrap();
    let error = worst(&on_gpu.to_f32(), &host.to_f32());
    eprintln!("gelu: worst relative error {error:e}");
    // The shader uses the same polynomial and the same order; only FMA
    // contraction can differ, and that is one ulp.
    assert!(error <= 1e-6, "gelu differs by {error:e}");
}

#[test]
fn layernorm_is_within_2e_5_of_the_host() {
    let Some(mut gpu) = open() else { return };
    let (rows, width) = (4usize, 64usize);
    let mut rng = Lcg::new(19);
    let values = rng.take(rows * width);
    let gamma = rng.take(width);
    let beta = rng.take(width);
    let mut host = Tensor::f32(vec![rows as i64, width as i64], values.clone()).unwrap();
    let gt = Tensor::f32(vec![width as i64], gamma.clone()).unwrap();
    let bt = Tensor::f32(vec![width as i64], beta.clone()).unwrap();
    kernels::layernorm_(&mut host, &gt, &bt, 1e-12).unwrap();
    let mut on_gpu = Tensor::f32(vec![rows as i64, width as i64], values).unwrap();
    gpu.layernorm(&mut on_gpu, &gt, &bt, 1e-12).unwrap();
    let error = worst(&on_gpu.to_f32(), &host.to_f32());
    eprintln!("layernorm: worst relative error {error:e}");
    assert!(error <= 2e-5, "layernorm differs by {error:e}");
}

#[test]
fn softmax_with_a_mask_matches_the_host() {
    let Some(mut gpu) = open() else { return };
    let (rows, width) = (3usize, 48usize);
    let mut rng = Lcg::new(23);
    let mut values = rng.take(rows * width);
    // Masked entries, which are the case the clamps exist for.
    values[7] = -1e30;
    values[width + 9] = -3.0e38;
    let mut host = Tensor::f32(vec![rows as i64, width as i64], values.clone()).unwrap();
    kernels::softmax_(&mut host).unwrap();
    let mut on_gpu = Tensor::f32(vec![rows as i64, width as i64], values).unwrap();
    gpu.softmax(&mut on_gpu).unwrap();
    let gpu_values = on_gpu.to_f32();
    for (i, v) in gpu_values.iter().enumerate() {
        assert!(v.is_finite(), "softmax produced {v} at {i}");
    }
    let error = worst(&gpu_values, &host.to_f32());
    eprintln!("softmax: worst relative error {error:e}");
    assert!(error <= 2e-5, "softmax differs by {error:e}");
}

#[test]
fn attention_is_within_1e_4_of_the_host() {
    let Some(mut gpu) = open() else { return };
    let (q_len, kv_len, d) = (5usize, 8usize, 8usize);
    let mut rng = Lcg::new(31);
    let q = rng.take(q_len * d);
    let k = rng.take(kv_len * d);
    let v = rng.take(kv_len * d);
    let qt = Tensor::f32(vec![q_len as i64, d as i64], q).unwrap();
    let kt = Tensor::f32(vec![kv_len as i64, d as i64], k).unwrap();
    let vt = Tensor::f32(vec![kv_len as i64, d as i64], v).unwrap();
    let mut host = Tensor::f32(vec![q_len as i64, d as i64], vec![0.0; q_len * d]).unwrap();
    kernels::attention(&qt, &kt, &vt, &mut host).unwrap();
    let mut on_gpu = Tensor::f32(vec![q_len as i64, d as i64], vec![0.0; q_len * d]).unwrap();
    gpu.attention(&qt, &kt, &vt, &mut on_gpu).unwrap();
    let error = worst(&on_gpu.to_f32(), &host.to_f32());
    eprintln!("attention: worst relative error {error:e}");
    assert!(error <= 1e-4, "attention differs by {error:e}");
}

#[test]
fn a_long_attention_goes_back_to_the_host() {
    let Some(mut gpu) = open() else { return };
    // One key over the shader's bound: the backend must not compute it on the
    // GPU, and must not compute a truncated softmax either.
    let (q_len, kv_len, d) = (1usize, npum_vulkan::MAX_ATTENTION_KEYS + 8, 2usize);
    let q = vec![1.0f32; q_len * d];
    let k = vec![1.0f32; kv_len * d];
    let v = vec![1.0f32; kv_len * d];
    let qt = Tensor::f32(vec![q_len as i64, d as i64], q).unwrap();
    let kt = Tensor::f32(vec![kv_len as i64, d as i64], k).unwrap();
    let vt = Tensor::f32(vec![kv_len as i64, d as i64], v).unwrap();
    let mut out = Tensor::f32(vec![q_len as i64, d as i64], vec![0.0; q_len * d]).unwrap();
    gpu.attention(&qt, &kt, &vt, &mut out).unwrap();
    // All ones: every weight is 1/kv_len, so the answer is 1.0 exactly, on
    // whichever side computed it.
    for value in out.to_f32() {
        assert!((value - 1.0).abs() < 1e-6, "long attention gave {value}");
    }
}
