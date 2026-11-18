//! The GEMM accumulator is now a reused thread-local scratch buffer instead
//! of a `vec![0.0; n]` built on every call. That is only safe if every kernel
//! zeroes it before reducing and if a wide call cannot leave stale floats
//! that a narrower call then reads. These tests pin that, and pin the
//! per-worker slices carved out in `gemm_nt_rows` against the serial path.

use npum_cpu::kernels::{gemm_nt, gemm_nt_rows};

/// A deterministic pseudo-random f32, so the test needs no random seed and
/// repeats exactly.
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

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Reuse must be invisible: a call after a wider one and a call after a
/// narrower one must produce exactly the bytes the very first call did.
#[test]
fn reusing_the_accumulator_matches_a_fresh_one() {
    let (m, k, n) = (9usize, 37usize, 23usize);

    let run = |n: usize, seed: u32| {
        let a = values(m * k, seed);
        let b = values(k * n, seed ^ 0x55aa_55aa);
        let bias = values(n, seed ^ 0x0f0f_0f0f);
        let mut out = vec![0.0f32; m * n];
        gemm_nt(&a, m, &b, k, n, &bias, &mut out);
        bytes(&out)
    };

    let fresh = run(n, 0x1234_5678);

    // Grow the scratch well past `n`, then shrink it, then ask for `n` again.
    let _ = run(n + 2048, 0xdead_beef);
    let _ = run(n / 2, 0xcafe_babe);
    let reused = run(n, 0x1234_5678);

    assert_eq!(fresh, reused, "scratch reuse changed the GEMM bytes");
}

/// The row-parallel path hands each worker its own slice of one scratch
/// buffer. Splitting rows changes which thread computes a value, never the
/// value itself -- so it must be bit-identical to the serial call.
#[test]
fn row_parallel_matches_serial_with_a_reused_scratch() {
    for n in [23usize, 1024, 2048] {
        let (m, k) = (40usize, 61usize);
        let a = values(m * k, 0x1234_5678);
        let b = values(k * n, 0x9abc_def0);
        let bias = values(n, 0x0f0f_0f0f);

        let mut serial = vec![0.0f32; m * n];
        gemm_nt(&a, m, &b, k, n, &bias, &mut serial);

        // Call it twice: the second call reuses both the parent's buffer and
        // the workers' per-task slices.
        for _ in 0..2 {
            let mut parallel = vec![0.0f32; m * n];
            gemm_nt_rows(&a, m, &b, k, n, &bias, &mut parallel, 4);
            assert_eq!(
                bytes(&parallel),
                bytes(&serial),
                "row-parallel GEMM differs at n = {n}"
            );
        }
    }
}
