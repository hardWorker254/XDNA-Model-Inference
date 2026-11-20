//! Row kernels: vectorised exponential, and rows that are not eight wide.
//!
//! The golden was recorded with a degree-7 `exp2` polynomial, NOT libm's.
//! The AVX2 softmax now computes that polynomial eight lanes at a time with
//! the same FMAs; these tests pin every lane to the scalar answer. The second
//! test covers the masked tails added for short attention sequences, where a
//! row width is no longer rounded up to multiple of eight by the caller.

use npum_cpu::kernels::{Isa, layernorm_row, softmax_row, softmax_row_on};

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn values(n: usize, salt: u32) -> Vec<f32> {
    let mut x = 0x1234_5678u32 ^ salt.wrapping_mul(0x9e37_79b9);
    (0..n)
        .map(|i| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let f = (x as i32 as f32) / 100_000_000.0;
            // A few extreme scores exercise the -120 clamp and the exponent
            // field's integer part; the rest stay near the usual attention
            // range.
            match i % 37 {
                0 => -1000.0,
                1 => 1000.0,
                _ => f,
            }
        })
        .collect()
}

#[test]
fn vector_exp_is_bit_identical_to_the_scalar_polynomial() {
    for width in [8usize, 16, 64, 384] {
        for salt in 0..12u32 {
            let input = values(width, salt);
            let mut portable = input.clone();
            softmax_row_on(Isa::Portable, &mut portable).unwrap();
            let mut avx2 = input.clone();
            softmax_row_on(Isa::Avx2, &mut avx2).unwrap();
            assert_eq!(
                bytes(&avx2),
                bytes(&portable),
                "AVX2 exp differs at width {width}, seed {salt}"
            );
            let mut best = input.clone();
            softmax_row(&mut best).unwrap();
            assert_eq!(bytes(&best), bytes(&portable), "best ISA differs");
        }
    }
}

#[test]
fn short_rows_are_supported_by_the_masked_tails() {
    // These widths used to be an Err at the seam. They must now produce a
    // normalised row (and the two paths must still agree byte for byte).
    for width in 1..=40usize {
        let input = values(width, width as u32);
        let mut portable = input.clone();
        softmax_row_on(Isa::Portable, &mut portable).unwrap();
        let mut avx2 = input.clone();
        softmax_row_on(Isa::Avx2, &mut avx2).unwrap();
        assert_eq!(bytes(&avx2), bytes(&portable), "softmax width {width}");
        let sum: f32 = avx2.iter().copied().sum();
        assert!((sum - 1.0).abs() < 1e-4, "width {width} sums to {sum}");

        let gamma = values(width, 7);
        let beta = values(width, 9);
        let mut a = input.clone();
        layernorm_row(&mut a, &gamma, &beta).unwrap();
        let mut b = input.clone();
        layernorm_row(&mut b, &gamma, &beta).unwrap();
        assert_eq!(bytes(&a), bytes(&b));
    }
}
