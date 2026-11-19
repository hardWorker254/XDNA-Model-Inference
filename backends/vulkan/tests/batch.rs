//! One submission for a chain of operations, with no host round-trip in the
//! middle. This is the gate for the batching path that the text family uses
//! for the tail of a transformer layer.
//!
//! The chain deliberately reuses the SAME host slice as the output of every
//! step: the device has to forward that slice's pending buffer from one
//! dispatch to the next instead of reading stale host memory.

use npum_cpu::ops::CpuOps;
use npum_model::Ops;
use npum_vulkan::ops::VulkanOps;

fn values(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn worst(gpu: &[f32], host: &[f32]) -> f32 {
    gpu.iter()
        .zip(host)
        .map(|(g, h)| (g - h).abs() / h.abs().max(1e-3))
        .fold(0.0f32, f32::max)
}

#[test]
fn batch_stages() {
    let mut gpu = VulkanOps::new();
    if !gpu.has_gpu() { return; }
    let (m, k, n) = (8usize, 16usize, 12usize);
    let a = values(m * k, 1);
    let b = values(k * n, 2);
    let bias = values(n, 3);
    let ones = vec![1.0f32; m * n];
    let gamma = values(m * n, 4);
    let beta = values(m * n, 5);
    let mut host = vec![0.0f32; m * n];
    let mut on_gpu = vec![0.0f32; m * n];
    let mut cpu = CpuOps::best();

    macro_rules! stage {
        ($host:expr, $gpuop:expr) => {{
            $host;
            assert!(gpu.begin_batch());
            $gpuop;
            gpu.end_batch().unwrap();
            eprintln!("stage worst={:e}", worst(&on_gpu, &host));
        }};
    }
    stage!(cpu.gemm_nt(&a, m, &b, k, n, &bias, &mut host).unwrap(),
           gpu.gemm_nt(&a, m, &b, k, n, &bias, &mut on_gpu).unwrap());
    stage!(cpu.add_inplace(&mut host, &ones).unwrap(),
           gpu.add_inplace(&mut on_gpu, &ones).unwrap());
    stage!(cpu.gelu(&mut host).unwrap(), gpu.gelu(&mut on_gpu).unwrap());
    stage!(cpu.layernorm_rows(&mut host, 1, m * n, &gamma, &beta).unwrap(),
           gpu.layernorm_rows(&mut on_gpu, 1, m * n, &gamma, &beta).unwrap());
    stage!(cpu.softmax_rows(&mut host, 1, m * n).unwrap(),
           gpu.softmax_rows(&mut on_gpu, 1, m * n).unwrap());
}

#[test]
fn a_batched_layer_tail_matches_the_host() {
    let mut gpu = VulkanOps::new();
    if !gpu.has_gpu() {
        eprintln!("SKIP: no Vulkan device on this machine");
        return;
    }
    let mut host_ops = CpuOps::best();

    let (m, k, n) = (8usize, 16usize, 12usize);
    let a = values(m * k, 1);
    let b = values(k * n, 2);
    let bias = values(n, 3);
    let ones = vec![1.0f32; m * n];
    let gamma = values(m * n, 4);
    let beta = values(m * n, 5);

    let mut host = vec![0.0f32; m * n];
    host_ops.gemm_nt(&a, m, &b, k, n, &bias, &mut host).unwrap();
    host_ops.add_inplace(&mut host, &ones).unwrap();
    host_ops.gelu(&mut host).unwrap();
    host_ops
        .layernorm_rows(&mut host, 1, m * n, &gamma, &beta)
        .unwrap();
    host_ops.softmax_rows(&mut host, 1, m * n).unwrap();

    let mut on_gpu = vec![0.0f32; m * n];
    assert!(gpu.begin_batch(), "the GPU backend batches");
    gpu.gemm_nt(&a, m, &b, k, n, &bias, &mut on_gpu).unwrap();
    gpu.add_inplace(&mut on_gpu, &ones).unwrap();
    gpu.gelu(&mut on_gpu).unwrap();
    gpu.layernorm_rows(&mut on_gpu, 1, m * n, &gamma, &beta)
        .unwrap();
    gpu.softmax_rows(&mut on_gpu, 1, m * n).unwrap();
    gpu.end_batch().unwrap();

    let error = worst(&on_gpu, &host);
    eprintln!("batched tail: worst relative error {error:e}, {} dispatches", gpu.dispatches());
    assert!(error <= 2e-4, "the batched tail differs by {error:e}");
    assert_eq!(gpu.dispatches(), 5, "five operations, five counted dispatches");
}

#[test]
fn a_batched_gemm_alone_matches_the_host() {
    let mut gpu = VulkanOps::new();
    if !gpu.has_gpu() {
        return;
    }
    let (m, k, n) = (4usize, 8usize, 6usize);
    let a = values(m * k, 7);
    let b = values(k * n, 8);
    let bias = values(n, 9);
    let mut host = vec![0.0f32; m * n];
    CpuOps::best().gemm_nt(&a, m, &b, k, n, &bias, &mut host).unwrap();
    let mut on_gpu = vec![0.0f32; m * n];
    assert!(gpu.begin_batch());
    gpu.gemm_nt(&a, m, &b, k, n, &bias, &mut on_gpu).unwrap();
    gpu.end_batch().unwrap();
    assert!(worst(&on_gpu, &host) <= 2e-5);
}

#[test]
fn a_batched_gemm_can_be_consumed_as_an_input() {
    let mut gpu = VulkanOps::new();
    if !gpu.has_gpu() {
        return;
    }
    let (m, k, n) = (4usize, 8usize, 6usize);
    let a = values(m * k, 7);
    let b = values(k * n, 8);
    let bias = values(n, 9);
    let y = values(m * n, 10);

    let mut host = vec![0.0f32; m * n];
    CpuOps::best().gemm_nt(&a, m, &b, k, n, &bias, &mut host).unwrap();
    CpuOps::best().add_inplace(&mut host, &y).unwrap();

    let mut on_gpu = vec![0.0f32; m * n];
    assert!(gpu.begin_batch());
    gpu.gemm_nt(&a, m, &b, k, n, &bias, &mut on_gpu).unwrap();
    gpu.add_inplace(&mut on_gpu, &y).unwrap();
    gpu.end_batch().unwrap();

    let error = worst(&on_gpu, &host);
    assert!(error <= 2e-5, "batched gemm + add differs by {error:e}");
}
