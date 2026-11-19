//! GEMM microbenchmarks for the shapes that dominate CPU inference.
//!
//! Run with `cargo bench -p npum-cpu --bench gemm`. The numbers are printed,
//! not asserted: the point is to see whether a kernel change helps on THIS
//! host, before and after the change. No RNG crate is needed; the data is a
//! cheap deterministic xorshift, and every kernel reads it in the same order.
//!
//! Shapes:
//!   * transformer projections (M = batch * sequence tokens, K = hidden)
//!   * transformer feed-forward (K = hidden, N = 4 * hidden)
//!   * single-token decode (M = 1)
//!   * small detector convolution (few output channels, long reduction)

use std::hint::black_box;
use std::time::Instant;

use npum_cpu::kernels::{gemm_nt, gemm_nt_rows};
use npum_cpu::parallel;

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

/// How many independent timing trials each shape gets. The machine is
/// shared and frequencies move; a median over trials is far steadier than
/// one long run.
const TRIALS: usize = 5;

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

fn time_ms(mut f: impl FnMut()) -> f64 {
    let t0 = Instant::now();
    f();
    t0.elapsed().as_secs_f64() * 1e3
}

fn fmt_ms(ms: f64) -> String {
    if ms < 0.01 {
        format!("{:.1} us", ms * 1e3)
    } else {
        format!("{ms:8.3} ms")
    }
}

fn bench(name: &str, m: usize, k: usize, n: usize, reps: usize) {
    let a = values(m * k, 0x1234_5678);
    let b = values(k * n, 0x9abc_def0);
    let bias = values(n, 0x0f0f_0f0f);
    let mut out = vec![0.0f32; m * n];

    // Warm the caches and the ISA-detection once-lock.
    for _ in 0..2 {
        gemm_nt(&a, m, &b, k, n, &bias, &mut out);
    }

    let mut serial_trials = Vec::with_capacity(TRIALS);
    let mut pooled_trials = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        serial_trials.push(time_ms(|| {
            for _ in 0..reps {
                black_box(gemm_nt(
                    black_box(&a),
                    m,
                    black_box(&b),
                    k,
                    n,
                    &bias,
                    black_box(&mut out),
                ));
            }
        }) / reps as f64);
        pooled_trials.push(time_ms(|| {
            for _ in 0..reps {
                gemm_nt_rows(
                    black_box(&a),
                    m,
                    black_box(&b),
                    k,
                    n,
                    &bias,
                    black_box(&mut out),
                    parallel::threads(),
                );
            }
        }) / reps as f64);
    }
    let serial = median(serial_trials);
    let pooled = median(pooled_trials);

    // 2 FLOPs per multiply-add, in GFLOP; time is in ms.
    let gflops = 2.0 * (m * k * n) as f64 / 1e6;
    println!(
        "{name:<22} M={m:<5} K={k:<5} N={n:<5}  serial {:>12} ({:>6.1} GF/s)   pooled {:>12} ({:>6.1} GF/s)  x{:.2}",
        fmt_ms(serial),
        gflops / serial,
        fmt_ms(pooled),
        gflops / pooled,
        serial / pooled.max(1e-12),
    );
}

fn main() {
    println!(
        "npum-cpu GEMM: threads = {}, PAR_MIN_ITEMS = {}, rows/worker = {}",
        parallel::threads(),
        npum_cpu::kernels::PAR_MIN_ITEMS,
        npum_cpu::parallel::PAR_MIN_ROWS_PER_WORKER,
    );
    bench("decode qkv", 1, 768, 2304, 200);
    bench("encode qkv", 128, 768, 2304, 20);
    bench("encode attn out", 128, 768, 768, 40);
    bench("feed-forward up", 128, 768, 3072, 20);
    bench("feed-forward down", 128, 3072, 768, 10);
    bench("large batch qkv", 512, 768, 2304, 8);
    bench("detector conv", 64, 864, 24, 30);
    bench("narrow tail", 128, 768, 23, 30);
}
