#![cfg(target_arch = "x86_64")]

//! The packed-B cache is a pure address optimisation.
//!
//! A cached panel holds the SAME floats as B in a different order, so a hit
//! must produce exactly the bytes a fresh pack produces. Two tensors at
//! different addresses must never share an entry, even with the same shape.

use npum_cpu::kernels::{clear_b_pack_cache, gemm_nt, gemm_nt_rows};

fn values(n: usize, salt: u32) -> Vec<f32> {
    let mut x = 0x9e37_79b9u32 ^ salt.wrapping_mul(0x85eb_ca6b);
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

#[test]
fn a_cached_panel_computes_the_same_bytes_as_a_fresh_one() {
    // 1024 columns is a whole page: the width that takes the packed dispatch.
    let (m, k, n) = (32usize, 64usize, 1024usize);
    let a = values(m * k, 1);
    let b = values(k * n, 2);
    let bias = values(n, 3);

    clear_b_pack_cache();
    let mut want = vec![0.0f32; m * n];
    gemm_nt(&a, m, &b, k, n, &bias, &mut want);
    let want = bytes(&want);

    // First call fills the cache; the rest are hits through the Arc.
    for _ in 0..4 {
        let mut got = vec![0.0f32; m * n];
        gemm_nt(&a, m, &b, k, n, &bias, &mut got);
        assert_eq!(bytes(&got), want, "cache hit changed the GEMM bytes");
    }

    // A row-parallel GEMM shares the calling thread's cached panel too.
    let mut pooled = vec![0.0f32; m * n];
    gemm_nt_rows(&a, m, &b, k, n, &bias, &mut pooled, 4);
    assert_eq!(bytes(&pooled), want);
}

#[test]
fn different_weights_at_different_addresses_do_not_share_a_panel() {
    let (m, k, n) = (16usize, 64usize, 1024usize);
    let a = values(m * k, 5);
    let bias = values(n, 6);
    let b1 = values(k * n, 7);
    let b2 = values(k * n, 8);

    // Compute both references with the cache cleared between them.
    clear_b_pack_cache();
    let mut want1 = vec![0.0f32; m * n];
    gemm_nt(&a, m, &b1, k, n, &bias, &mut want1);
    clear_b_pack_cache();
    let mut want2 = vec![0.0f32; m * n];
    gemm_nt(&a, m, &b2, k, n, &bias, &mut want2);
    assert_ne!(bytes(&want1), bytes(&want2));

    // Now interleave them: both panels have to stay live at once.
    clear_b_pack_cache();
    for round in 0..3 {
        let mut got1 = vec![0.0f32; m * n];
        gemm_nt(&a, m, &b1, k, n, &bias, &mut got1);
        let mut got2 = vec![0.0f32; m * n];
        gemm_nt(&a, m, &b2, k, n, &bias, &mut got2);
        assert_eq!(bytes(&got1), bytes(&want1), "round {round}, first matrix");
        assert_eq!(bytes(&got2), bytes(&want2), "round {round}, second matrix");
    }
}
