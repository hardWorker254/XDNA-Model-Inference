//! Perf paths must not change the reference arithmetic.
//! SPDX-License-Identifier: Apache-2.0

use npum_cpu::kernels;
use npum_model::Tensor;

fn values(seed: f32, n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + seed) * 0.123).sin()).collect()
}

#[test]
fn fft_keeps_the_reference_angle_expression() {
    for n in [16usize, 400] {
        let x = Tensor::f32(vec![n as i64], values(0.3, n)).unwrap();
        let mut out = Tensor::f32(vec![n as i64], vec![0.0; n]).unwrap();
        kernels::fft(&x, &mut out).unwrap();
        let xv = x.to_f32();
        let mut want = vec![0.0f32; n];
        for k in 0..n {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (t, &v) in xv.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * (k as f32) * (t as f32) / (n as f32);
                re += v * angle.cos();
                im += v * angle.sin();
            }
            want[k] = (re * re + im * im).sqrt();
        }
        assert_eq!(out.to_f32(), want, "fft n={n}");
    }
}

#[test]
fn attention_keeps_the_scalar_qk_and_v_accumulation() {
    for d in [4usize, 8, 16] {
        let (nq, nk) = (3usize, 5usize);
        let q = Tensor::f32(vec![nq as i64, d as i64], values(1.0, nq * d)).unwrap();
        let k = Tensor::f32(vec![nk as i64, d as i64], values(2.0, nk * d)).unwrap();
        let v = Tensor::f32(vec![nk as i64, d as i64], values(3.0, nk * d)).unwrap();
        let mut out = Tensor::f32(vec![nq as i64, d as i64], vec![0.0; nq * d]).unwrap();
        kernels::attention(&q, &k, &v, &mut out).unwrap();

        let (qv, kv, vv) = (q.to_f32(), k.to_f32(), v.to_f32());
        let scale = 1.0 / (d as f32).sqrt();
        let mut scores = vec![0.0f32; nk];
        let mut want = vec![0.0f32; nq * d];
        for i in 0..nq {
            for j in 0..nk {
                let mut sum = 0.0f32;
                for p in 0..d {
                    sum += qv[i * d + p] * kv[j * d + p];
                }
                scores[j] = sum * scale;
            }
            kernels::softmax_row(&mut scores).unwrap();
            for j in 0..nk {
                for p in 0..d {
                    want[i * d + p] += scores[j] * vv[j * d + p];
                }
            }
        }
        assert_eq!(out.to_f32(), want, "attention d={d}");
    }
}

#[test]
fn the_packed_weight_cache_notices_in_place_mutation() {
    let (m, k, n) = (32usize, 16usize, 1024usize);
    let a = values(0.5, m * k);
    let mut b = values(0.7, k * n);
    let bias = vec![0.0f32; n];

    kernels::clear_b_pack_cache();
    let mut first = vec![0.0f32; m * n];
    kernels::gemm_nt_rows(&a, m, &b, k, n, &bias, &mut first, 4);
    b[5] = 12345.0; // an index the old 13-sample fingerprint never probed
    let mut mutated = vec![0.0f32; m * n];
    kernels::gemm_nt_rows(&a, m, &b, k, n, &bias, &mut mutated, 4);
    let copy = b.clone(); // a different address cannot hit the cached panel
    let mut fresh = vec![0.0f32; m * n];
    kernels::gemm_nt_rows(&a, m, &copy, k, n, &bias, &mut fresh, 4);
    assert_eq!(mutated, fresh);
}
