//! Gates for the vector tails and the wider register blocking.
//!
//! A masked vector tail replaces a scalar loop, which is a change of INSTRUCTION
//! and must not be a change of NUMBER. These tests compare every narrow and
//! unaligned width against the portable kernel bit for bit, including the row
//! counts that choose the four-row and the eight-row blocking paths.

use npum_cpu::kernels::{Isa, available_isas, gemm_nt, gemm_nt_on, gemm_nt_rows};

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

#[test]
fn every_column_tail_is_bit_identical_to_portable() {
    let isas = available_isas();
    if !isas.contains(&Isa::Avx2) {
        return; // a host without AVX2 has no masked tails to check
    }
    let k = 13usize;
    for m in [1usize, 3, 4, 5, 7, 8, 9, 12, 17] {
        for n in 1..=40usize {
            let a = values(m * k, (m * 131 + n) as u32);
            let b = values(k * n, (m * 977 + n * 31) as u32);
            let bias = values(n, (n * 7919) as u32);

            let mut want = vec![0.0f32; m * n];
            assert!(gemm_nt_on(Isa::Portable, &a, m, &b, k, n, &bias, &mut want));

            for isa in &isas {
                let mut got = vec![0.0f32; m * n];
                assert!(gemm_nt_on(*isa, &a, m, &b, k, n, &bias, &mut got));
                assert_eq!(bytes(&got), bytes(&want), "{isa:?}: m={m}, n={n}");
            }

            // The dispatch itself: it picks the row kernel, the blocked
            // kernel, or an ISA based on width, and all three must agree.
            let mut got = vec![0.0f32; m * n];
            gemm_nt(&a, m, &b, k, n, &bias, &mut got);
            assert_eq!(bytes(&got), bytes(&want), "dispatch: m={m}, n={n}");
        }
    }
}

#[test]
fn the_packed_panel_path_is_bit_identical_to_portable() {
    // The pack pass changes ADDRESSES, not values. It is used for k >= 16,
    // m >= 8, n >= 8 -- including the page-multiple widths that used to run
    // the row-wise kernel because of cache-set conflicts.
    if !available_isas().contains(&Isa::Avx2) {
        return;
    }
    for (m, k, n) in [
        (8usize, 16usize, 8usize),
        (9, 17, 23),
        (16, 32, 1024),
        (24, 64, 2048),
        (12, 129, 1000),
    ] {
        let a = values(m * k, (m + n) as u32);
        let b = values(k * n, (k * n) as u32);
        let bias = values(n, n as u32);
        let mut want = vec![0.0f32; m * n];
        assert!(gemm_nt_on(Isa::Portable, &a, m, &b, k, n, &bias, &mut want));

        // A wide call first, then the shape under test, exercises the
        // thread-local panel scratch being grown and then reused.
        let wide_n = n.max(2048);
        let wide_b = values(k * wide_n, 0xbeef);
        let wide_bias = values(wide_n, 0xcafe);
        let mut wide_out = vec![0.0f32; m * wide_n];
        gemm_nt(&a, m, &wide_b, k, wide_n, &wide_bias, &mut wide_out);

        let mut got = vec![0.0f32; m * n];
        gemm_nt(&a, m, &b, k, n, &bias, &mut got);
        assert_eq!(bytes(&got), bytes(&want), "packed m={m}, k={k}, n={n}");

        let mut parallel = vec![0.0f32; m * n];
        gemm_nt_rows(&a, m, &b, k, n, &bias, &mut parallel, 4);
        assert_eq!(bytes(&parallel), bytes(&want), "packed rows m={m}, n={n}");
    }
}

#[test]
fn row_parallel_tails_are_bit_identical_to_serial() {
    let k = 64usize;
    let m = 256usize;
    for n in [1usize, 3, 7, 8, 9, 15, 17, 23, 24, 31, 33] {
        let a = values(m * k, n as u32 * 17);
        let b = values(k * n, n as u32 * 31);
        let bias = values(n, n as u32 * 47);
        let mut serial = vec![0.0f32; m * n];
        gemm_nt(&a, m, &b, k, n, &bias, &mut serial);
        let mut parallel = vec![0.0f32; m * n];
        gemm_nt_rows(&a, m, &b, k, n, &bias, &mut parallel, 4);
        assert_eq!(bytes(&parallel), bytes(&serial), "n={n}");
    }
}
