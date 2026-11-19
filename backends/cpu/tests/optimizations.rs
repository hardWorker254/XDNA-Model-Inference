//! Gates for the shared work pool and the im2col convolution paths.
//!
//! The byte-exactness gates live in `isa.rs` and `gemm_scratch.rs`. These
//! tests cover the two things the optimisation pass added that those gates
//! do not see: the generic tensor convolutions now go through im2col + GEMM,
//! and `CpuOps` can split independent rows over the pool.

use npum_cpu::kernels::{
    Isa, add_inplace, dot8, fused_attention, gelu_f32, gelu_row_on, gemm_nt, gemm_nt_on,
    gemm_nt_rows, gemm_nt_rows_on, softmax_row, weighted_accumulate,
};
use npum_cpu::ops::CpuOps;
use npum_model::{Ops, Tensor};

/// A deterministic f32, so the tests need no RNG.
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

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 2e-5,
            "{what}: element {i}: got {g}, want {w}"
        );
    }
}

/// The old direct-loop convolution, as the oracle's arithmetic was written:
/// a plain `sum += a * b` with the reduction in `(channel, kernel)` order.
fn reference_conv1d(
    x: &[f32],
    n: usize,
    cin: usize,
    l: usize,
    w: &[f32],
    cout: usize,
    k: usize,
    b: &[f32],
    stride: i64,
    pad: i64,
) -> Vec<f32> {
    let l_out = ((l as i64 + 2 * pad - k as i64) / stride + 1).max(0) as usize;
    let mut out = vec![0.0f32; n * cout * l_out];
    for bi in 0..n {
        for co in 0..cout {
            for oi in 0..l_out {
                let mut sum = b.get(co).copied().unwrap_or(0.0);
                let base = oi as i64 * stride - pad;
                for ci in 0..cin {
                    for ki in 0..k {
                        let pos = base + ki as i64;
                        if (0..l as i64).contains(&pos) {
                            sum +=
                                x[(bi * cin + ci) * l + pos as usize] * w[(co * cin + ci) * k + ki];
                        }
                    }
                }
                out[(bi * cout + co) * l_out + oi] = sum;
            }
        }
    }
    out
}

#[test]
fn conv1d_im2col_matches_the_direct_loops() {
    // The generic tensor conv writes its result into the SAME tensor, so the
    // input and output byte counts have to agree: 2*4*6 == 2*6*4.
    let (n, cin, l, cout, k) = (2usize, 4, 6, 6, 3);
    let (stride, pad) = (1i64, 0i64);
    let xv = values(n * cin * l, 0x1234);
    let wv = values(cout * cin * k, 0x9abc);
    let bv = values(cout, 0x0f0f);

    let w = Tensor::f32(vec![cout as i64, cin as i64, k as i64], wv.clone()).unwrap();
    let b = Tensor::f32(vec![cout as i64], bv.clone()).unwrap();

    let mut out = Tensor::f32(vec![n as i64, cin as i64, l as i64], xv.clone()).unwrap();
    npum_cpu::kernels::conv1d(&mut out, &w, Some(&b), stride, pad).unwrap();

    let got: Vec<f32> = out
        .bytes()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let want = reference_conv1d(&xv, n, cin, l, &wv, cout, k, &bv, stride, pad);
    assert_close(&got, &want, "conv1d");
}

#[test]
fn conv2d_im2col_matches_the_direct_loops() {
    // Same in-place byte-count constraint as conv1d: 1*2*4*4 == 1*2*4*4.
    let (n, cin, h, width, cout, kh, kw) = (1usize, 2, 4, 4, 2, 3, 3);
    let stride = (1i64, 1i64);
    let pad = (1i64, 1i64);
    let xv = values(n * cin * h * width, 0x5678);
    let wv = values(cout * cin * kh * kw, 0xcdef);
    let bv = values(cout, 0x3322);

    let w = Tensor::f32(
        vec![cout as i64, cin as i64, kh as i64, kw as i64],
        wv.clone(),
    )
    .unwrap();
    let b = Tensor::f32(vec![cout as i64], bv.clone()).unwrap();

    let h_out = (h as i64 + 2 * pad.0 - kh as i64) / stride.0 + 1;
    let w_out = (width as i64 + 2 * pad.1 - kw as i64) / stride.1 + 1;
    let (h_out, w_out) = (h_out as usize, w_out as usize);
    let mut out = Tensor::f32(
        vec![n as i64, cin as i64, h as i64, width as i64],
        xv.clone(),
    )
    .unwrap();
    npum_cpu::kernels::conv2d(&mut out, &w, Some(&b), stride, pad).unwrap();

    let got: Vec<f32> = out
        .bytes()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let mut want = vec![0.0f32; n * cout * h_out * w_out];
    for bi in 0..n {
        for co in 0..cout {
            for oi in 0..h_out {
                for oj in 0..w_out {
                    let mut sum = bv[co];
                    let bh = oi as i64 * stride.0 - pad.0;
                    let bw = oj as i64 * stride.1 - pad.1;
                    for ci in 0..cin {
                        for ki in 0..kh {
                            let ih = bh + ki as i64;
                            if !(0..h as i64).contains(&ih) {
                                continue;
                            }
                            for kwi in 0..kw {
                                let iw = bw + kwi as i64;
                                if (0..width as i64).contains(&iw) {
                                    sum += xv
                                        [((bi * cin + ci) * h + ih as usize) * width + iw as usize]
                                        * wv[((co * cin + ci) * kh + ki) * kw + kwi];
                                }
                            }
                        }
                    }
                    want[((bi * cout + co) * h_out + oi) * w_out + oj] = sum;
                }
            }
        }
    }
    assert_close(&got, &want, "conv2d");
}

#[test]
fn add_inplace_matches_the_scalar_loop() {
    for n in [1usize, 7, 8, 9, 65] {
        let mut got = values(n, 0x1111);
        let src = values(n, 0x2222);
        let mut want = got.clone();
        for (d, &s) in want.iter_mut().zip(&src) {
            *d += s;
        }
        add_inplace(&mut got, &src);
        assert_eq!(bytes(&got), bytes(&want), "add_inplace at n = {n}");
    }
}

#[test]
fn row_parallel_gemm_agrees_with_the_serial_one_above_the_threshold() {
    // Well above PAR_MIN_ITEMS, so the split really happens.
    let (m, k, n) = (256usize, 64, 64);
    let a = values(m * k, 0xa1a1);
    let b = values(k * n, 0xb2b2);
    let bias = values(n, 0xc3c3);
    let mut serial = vec![0.0f32; m * n];
    gemm_nt(&a, m, &b, k, n, &bias, &mut serial);

    for threads in [2usize, 4, 16] {
        let mut parallel = vec![0.0f32; m * n];
        gemm_nt_rows(&a, m, &b, k, n, &bias, &mut parallel, threads);
        assert_eq!(
            bytes(&parallel),
            bytes(&serial),
            "row-parallel GEMM differs at {threads} workers"
        );
        if let Some(isa) = npum_cpu::kernels::available_isas().first().copied() {
            let mut named_serial = vec![0.0f32; m * n];
            let mut named_parallel = vec![0.0f32; m * n];
            assert!(gemm_nt_on(isa, &a, m, &b, k, n, &bias, &mut named_serial));
            assert!(gemm_nt_rows_on(
                isa,
                &a,
                m,
                &b,
                k,
                n,
                &bias,
                &mut named_parallel,
                threads
            ));
            assert_eq!(
                bytes(&named_parallel),
                bytes(&named_serial),
                "named-ISA row-parallel GEMM differs at {threads} workers"
            );
        }
    }
}

#[test]
fn reduction_helpers_agree_with_the_scalar_loops_bit_for_bit() {
    // Eight-lane dot product: the first group is plain multiplies, the rest
    // are FMAs, and the tree is hsum8. Both paths must produce the same bytes.
    for n in [8usize, 16, 32, 64, 132] {
        let a = values(n, 0x1000 + n as u32);
        let b = values(n, 0x2000 + n as u32);
        let nv = n / 8;
        let mut lanes = [0.0f32; 8];
        for l in 0..8 {
            lanes[l] = a[l] * b[l];
        }
        for v in 1..nv {
            for l in 0..8 {
                lanes[l] = a[v * 8 + l].mul_add(b[v * 8 + l], lanes[l]);
            }
        }
        let mut want = npum_cpu::kernels::hsum8(lanes);
        let mut j = nv * 8;
        while j < n {
            want += a[j] * b[j];
            j += 1;
        }
        assert_eq!(
            dot8(&a, &b).to_bits(),
            want.to_bits(),
            "dot8 differs at n = {n}"
        );
    }

    // Weighted accumulation (the attention V path).
    for n in [3usize, 8, 9, 65] {
        let v = values(n, 0x3000 + n as u32);
        let mut got = values(n, 0x4000 + n as u32);
        let mut want = got.clone();
        for (d, &x) in want.iter_mut().zip(&v) {
            *d = 0.375f32.mul_add(x, *d);
        }
        weighted_accumulate(&mut got, &v, 0.375);
        assert_eq!(bytes(&got), bytes(&want), "weighted_accumulate at n = {n}");
    }
}

#[test]
fn gelu_avx2_row_is_bit_identical_to_the_scalar_polynomial() {
    for n in [1usize, 7, 8, 9, 32, 257] {
        let x = values(n, 0x5000 + n as u32);
        let mut scalar = x.clone();
        for v in scalar.iter_mut() {
            *v = gelu_f32(*v);
        }
        for isa in npum_cpu::kernels::available_isas() {
            let mut got = x.clone();
            gelu_row_on(isa, &mut got);
            assert_eq!(
                bytes(&got),
                bytes(&scalar),
                "GELU differs on {isa:?} at n = {n}"
            );
        }
        // Keep the unused import honest on non-x86 hosts, where the gate
        // above can only ever run the portable path.
        let _ = Isa::Portable;
    }
}

#[test]
fn ops_row_kernels_agree_between_serial_and_pooled() {
    let (rows, width) = (512usize, 64usize);
    let gamma = values(width, 0x4747);
    let beta = values(width, 0x4242);
    for isa in npum_cpu::kernels::available_isas() {
        let mut a = values(rows * width, 0x5a5a);
        let mut b = a.clone();
        let mut serial = CpuOps::on(isa);
        let mut pooled = CpuOps::with_threads(isa, 4);
        serial
            .layernorm_rows(&mut a, rows, width, &gamma, &beta)
            .unwrap();
        pooled
            .layernorm_rows(&mut b, rows, width, &gamma, &beta)
            .unwrap();
        assert_eq!(bytes(&a), bytes(&b), "LayerNorm rows differ on {isa:?}");

        let mut a = values(rows * width, 0x6b6b);
        let mut b = a.clone();
        serial.softmax_rows(&mut a, rows, width).unwrap();
        pooled.softmax_rows(&mut b, rows, width).unwrap();
        assert_eq!(bytes(&a), bytes(&b), "softmax rows differ on {isa:?}");
    }
}

#[test]
fn fused_attention_is_bit_identical_to_the_three_separate_passes() {
    const BATCH: usize = 2;
    const SEQ: usize = 16;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 8;
    const HIDDEN: usize = HEADS * HEAD_DIM;

    let qkv = values(BATCH * SEQ * 3 * HIDDEN, 0xa11e);
    let mut add_mask = vec![0.0f32; BATCH * SEQ];
    for (i, slot) in add_mask.iter_mut().enumerate() {
        if i % 7 == 3 || i % 11 == 5 {
            *slot = -1.0e30;
        }
    }

    // The three passes, as GTE runs them: QK with the mask folded in, a
    // softmax over the whole row, then the weighted V accumulation.
    let mut scores = vec![0.0f32; BATCH * HEADS * SEQ * SEQ];
    for row in 0..BATCH * HEADS * SEQ {
        let p = row / SEQ;
        let i = row % SEQ;
        let b = p / HEADS;
        let h = p % HEADS;
        let q = &qkv[(b * SEQ + i) * 3 * HIDDEN + h * HEAD_DIM..][..HEAD_DIM];
        for (j, slot) in scores[row * SEQ..(row + 1) * SEQ].iter_mut().enumerate() {
            let k = &qkv[(b * SEQ + j) * 3 * HIDDEN + HIDDEN + h * HEAD_DIM..][..HEAD_DIM];
            *slot = dot8(q, k) + add_mask[b * SEQ + j];
        }
    }
    for row in scores.chunks_exact_mut(SEQ) {
        softmax_row(row).unwrap();
    }
    let mut want = vec![0.0f32; BATCH * SEQ * HIDDEN];
    for row in 0..BATCH * SEQ {
        let (b, i) = (row / SEQ, row % SEQ);
        for h in 0..HEADS {
            let a = &scores[((b * HEADS + h) * SEQ + i) * SEQ..][..SEQ];
            let o = &mut want[row * HIDDEN + h * HEAD_DIM..][..HEAD_DIM];
            for (j, &weight) in a.iter().enumerate() {
                let v = &qkv[(b * SEQ + j) * 3 * HIDDEN + 2 * HIDDEN + h * HEAD_DIM..][..HEAD_DIM];
                weighted_accumulate(o, v, weight);
            }
        }
    }

    let mut got = vec![0.0f32; BATCH * SEQ * HIDDEN];
    fused_attention(
        &qkv, &mut got, &add_mask, BATCH, SEQ, HEADS, HEAD_DIM, HIDDEN,
    )
    .unwrap();
    assert_eq!(
        bytes(&got),
        bytes(&want),
        "fused attention changed the bytes"
    );
}
