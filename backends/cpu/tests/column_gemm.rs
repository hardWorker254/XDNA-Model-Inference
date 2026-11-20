//! The column-parallel GEMM crosses no reduction.
//!
//! A decode GEMM has `m = 1`, so the row-split policy has to leave it serial.
//! Its output COLUMNS are independent, though: column `j` is reduced over K
//! in the same order no matter which worker computes it. These tests make
//! the pool take that column split on shapes that actually clear the work
//! threshold, and compare the resulting bytes with the serial reference.

use npum_cpu::kernels::{
    Isa, available_isas, gemm_nt, gemm_nt_cols_on, gemm_nt_rows, gemm_nt_rows_on,
};

fn values(n: usize) -> Vec<f32> {
    let mut x = 0x9e37_79b9u32;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x as i32 as f32) / 1_000_000.0
        })
        .collect()
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn check(m: usize, k: usize, n: usize, threads: usize, chunk: usize) {
    let a = values(m * k);
    let b = values(k * n);
    let bias = values(n);

    let mut serial = vec![0.0f32; m * n];
    gemm_nt(&a, m, &b, k, n, &bias, &mut serial);
    let want = bytes(&serial);

    // The public entry point picks the split itself.
    let mut pooled = vec![0.0f32; m * n];
    gemm_nt_rows(&a, m, &b, k, n, &bias, &mut pooled, threads);
    assert_eq!(bytes(&pooled), want, "gemm_nt_rows {m}x{k}x{n}");

    let mut named = vec![0.0f32; m * n];
    for isa in available_isas() {
        named.fill(0.0);
        assert!(
            gemm_nt_rows_on(isa, &a, m, &b, k, n, &bias, &mut named, threads),
            "host should run {isa:?}"
        );
        assert_eq!(bytes(&named), want, "gemm_nt_rows_on {isa:?} {m}x{k}x{n}");

        named.fill(0.0);
        assert!(gemm_nt_cols_on(
            isa, &a, m, &b, k, n, &bias, &mut named, chunk
        ));
        assert_eq!(bytes(&named), want, "gemm_nt_cols_on {isa:?} chunk {chunk}");
    }
}

#[test]
fn decode_shapes_are_bit_identical_to_the_serial_gemm() {
    // The bench's qkv projection, plus a short-sequence variant, a batch of
    // two, and a narrow width whose chunks end in a masked vector tail.
    for (m, k, n) in [(1, 768, 2304), (1, 129, 1024), (2, 256, 512), (7, 64, 1024)] {
        check(m, k, n, 4, n.div_ceil(7));
    }
}

#[test]
fn every_column_tail_is_bit_identical_to_the_serial_gemm() {
    // 137 columns over four workers leaves windows of 35, 35, 35 and 32 --
    // none of them a multiple of eight, so both masked tails run.
    check(1, 1024, 137, 4, 35);
    check(3, 512, 137, 4, 9);
    check(1, 512, 17, 4, 8);
    // A chunk wider than the matrix still produces one window.
    check(1, 512, 129, 4, 1024);
}

#[test]
fn a_named_isa_that_the_host_cannot_run_falls_back() {
    // Portable always runs; the named paths return `true` after a portable
    // fallback, and the answer still has to be the reference's bytes.
    let (m, k, n) = (1, 512, 256);
    let a = values(m * k);
    let b = values(k * n);
    let bias = values(n);
    let mut serial = vec![0.0f32; m * n];
    gemm_nt(&a, m, &b, k, n, &bias, &mut serial);
    let mut got = vec![0.0f32; m * n];
    for isa in [Isa::Portable, Isa::Avx2, Isa::Avx512] {
        got.fill(0.0);
        assert!(gemm_nt_cols_on(isa, &a, m, &b, k, n, &bias, &mut got, 32));
        assert_eq!(bytes(&got), bytes(&serial), "{isa:?}");
    }
}
