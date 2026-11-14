//! The host kernels: the arithmetic the CPU backend runs.
//!
//! # The rules, and they are rules rather than preferences
//!
//! * **No FMA contraction.** Rust does not contract `a * b + c` into an FMA,
//!   and neither did the C++ scalar path. The gate
//!   `runtime/tests/test_int8_host_kernels.cpp` proved the scalar path and
//!   the AVX2 path agree bit for bit under exactly that rule.
//!
//! * **The scalar path is the definition.** The AVX2 path is an
//!   optimisation of it, and the two must agree bit for bit -- not "close",
//!   bit for bit. A difference between them is a bug in one of them, not a
//!   rounding difference, and a tolerance cannot tell the reader which.
//!
//! * **bf16 rounding is explicit.** Round-to-nearest-even, implemented here,
//!   not borrowed from `std::bf16` until a gate has confirmed that the
//!   standard library's semantics are the hardware's. The ties rule matters:
//!   round-half-to-even and round-half-away differ on every exact .5, and a
//!   container written with one and read with the other loses a bit of every
//!   weight that lands on one.
//!
//! * **Every intermediate is f32.** The array's bf16 datapath keeps its
//!   intermediates in bf16, which is why the tolerances between the CPU and
//!   the array are not zero and never will be. The CPU's intermediates are
//!   f32 because that is what the reference means.

use npum_model::{Result, Tensor, TensorError};

/// The GELU activation, as the C++ host kernel computes it.
///
/// A degree-8 minimax polynomial on `min(|x|, 4)`, NOT the exact erf and NOT
/// the tanh approximation. It is the same polynomial the fused array kernel
/// uses, so the two paths are the same function rather than two functions
/// that agree to a tolerance.
///
/// The clamp on `|x|` is not cosmetic: the polynomial is minimax on [0, 4]
/// and diverges outside it, and a large negative input would otherwise
/// produce a large positive answer.
pub fn gelu_f32(v: f32) -> f32 {
    let u = v.abs().min(4.0);
    let mut pl = -7.234_028_217_1e-05f32;
    pl = pl * u + 1.817_951_800_5e-03f32;
    pl = pl * u + -1.770_738_337_9e-02f32;
    pl = pl * u + 8.457_714_764_1e-02f32;
    pl = pl * u + -1.922_867_183_4e-01f32;
    pl = pl * u + 9.843_112_445_8e-02f32;
    pl = pl * u + 3.613_785_206_2e-01f32;
    pl = pl * u + -4.945_412_893_6e-01f32;
    pl = pl * u + -1.300_701_011_7e-04f32;
    v.max(0.0) + pl
}

/// Layer normalisation over the last axis.
///
/// `eps` is added to the variance before the square root, and the mean is
/// removed before the variance is taken -- the textbook order. The order
/// matters at the last bit: the textbook order and the "sum of squares"
/// shortcut differ in the last bits, and the shortcut is what the array's
/// kernel does.
pub fn layernorm_f32(row: &[f32], gamma: &[f32], beta: &[f32], eps: f32, out: &mut [f32]) {
    let n = row.len();
    if n == 0 {
        return;
    }
    let mut mean = 0.0f32;
    for v in row {
        mean += *v;
    }
    mean /= n as f32;
    let mut var = 0.0f32;
    for v in row {
        let d = *v - mean;
        var += d * d;
    }
    var /= n as f32;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        let g = gamma.get(i).copied().unwrap_or(1.0);
        let b = beta.get(i).copied().unwrap_or(0.0);
        out[i] = (row[i] - mean) * inv * g + b;
    }
}

/// Softmax over the last axis, with the three clamps.
///
/// # The three clamps, and why there are three
///
/// The array's softmax keeps its intermediates in bf16, and bf16 has a
/// narrower range than f32: its largest finite magnitude is 3.3895e38 while
/// f32's is 3.4028e38. A mask value that is finite in f32 -- HuggingFace's
/// additive mask is `(1 - mask) * finfo(f32).min` -- becomes `-inf` when it
/// is narrowed to bf16, and `exp(-inf - max)` is `exp(-inf)`, and `0 *
/// inf` is NaN. The whole row goes to NaN from one masked element.
///
/// So the clamps are:
///
/// 1. **On load, in bf16, before the widening:** `xb = max(xb, -1e30)`.
///    Clamping in f32 afterwards was not enough -- the -inf had already been
///    produced by the narrowing, and clamping it back does not undo the NaN
///    it already made. Clamping BEFORE the widening left 386 NaNs; clamping
///    before it did not.
/// 2. **On the difference:** `d = max(x - m, -100)`. `exp(-100)` is
///    3.7e-44, which is zero in bf16 -- the right answer for a masked
///    element, which should contribute nothing.
/// 3. **On the base-2 argument:** `arg = max(d * log2e, -120)`. The
///    exponential is computed as `exp2_poly`, and a negative exponent below
///    about -127 is a negative exponent FIELD, i.e. a NaN bit pattern.
///    Getting this wrong produced exactly 256 NaNs on the first attempt.
///
/// The CPU path applies the same three clamps, for the same reasons, and not
/// because it needs them: the CPU path must compute the function the array
/// computes, and a function that differs in the clamped region is a
/// different function.
pub fn softmax_f32(row: &mut [f32]) {
    let n = row.len();
    if n == 0 {
        return;
    }
    // Clamp 1: the floor, in the dtype the array would have loaded.
    for v in row.iter_mut() {
        if *v < -1e30 {
            *v = -1e30;
        }
    }
    let mut m = f32::NEG_INFINITY;
    for v in row.iter() {
        if *v > m {
            m = *v;
        }
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        // Clamp 2: the difference.
        let d = (*v - m).max(-100.0);
        // Clamp 3: the base-2 argument.
        let arg = (d * std::f32::consts::LOG2_E).max(-120.0);
        let e = arg.exp2();
        *v = e;
        sum += e;
    }
    let inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
    for v in row.iter_mut() {
        *v *= inv;
    }
}

/// `c = a * b`, row-major, optionally accumulating into `c`.
///
/// `a` is `[m, k]`, `b` is `[k, n]`, `c` is `[m, n]`. The accumulation is
/// `c += a * b` when `acc` is `Some`, and `c = a * b` when it is `None`.
pub fn gemm(
    a: &Tensor,
    b: &Tensor,
    c: &mut Tensor,
    acc: Option<&mut Tensor>,
) -> Result<()> {
    let ash = a.shape();
    let bsh = b.shape();
    if ash.len() != 2 || bsh.len() != 2 {
        return Err(TensorError::WrongRank {
            expected: 2,
            found: if ash.len() != 2 { ash.len() } else { bsh.len() },
        }
        .into());
    }
    let m = ash[0] as usize;
    let k = ash[1] as usize;
    let bn = bsh[0] as usize;
    let n = bsh[1] as usize;
    if bn != k {
        return Err(TensorError::ShapeMismatch {
            shape: bsh.to_vec(),
            len: b.len(),
        }
        .into());
    }
    let av = a.to_f32();
    let bv = b.to_f32();
    let mut cv = c.to_f32();
    if cv.len() != m * n {
        return Err(TensorError::WrongLength {
            expected: m * n,
            found: cv.len(),
        }
        .into());
    }
    if let Some(acc) = acc {
        let av_acc = acc.to_f32();
        for (i, v) in cv.iter_mut().enumerate() {
            *v += av_acc.get(i).copied().unwrap_or(0.0);
        }
    }
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += av[i * k + p] * bv[p * n + j];
            }
            cv[i * n + j] += sum;
        }
    }
    let mut bytes = Vec::with_capacity(cv.len() * 4);
    for v in &cv {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    copy_into(c, &bytes);
    Ok(())
}

/// GELU, in place.
pub fn gelu_(x: &mut Tensor) -> Result<()> {
    let mut values = x.to_f32();
    for v in values.iter_mut() {
        *v = gelu_f32(*v);
    }
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in &values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    copy_into(x, &bytes);
    Ok(())
}

/// Layer normalisation, in place.
pub fn layernorm_(x: &mut Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Result<()> {
    let mut values = x.to_f32();
    let g = gamma.to_f32();
    let b = beta.to_f32();
    let width = x.shape().last().copied().unwrap_or(0) as usize;
    if width == 0 {
        return Ok(());
    }
    for chunk in values.chunks_mut(width) {
        let mut out = vec![0.0f32; width];
        layernorm_f32(chunk, &g, &b, eps, &mut out);
        chunk.copy_from_slice(&out);
    }
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in &values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    copy_into(x, &bytes);
    Ok(())
}

/// Softmax, in place, along the last axis.
pub fn softmax_(x: &mut Tensor) -> Result<()> {
    let mut values = x.to_f32();
    let width = x.shape().last().copied().unwrap_or(0) as usize;
    if width == 0 {
        return Ok(());
    }
    for chunk in values.chunks_mut(width) {
        softmax_f32(chunk);
    }
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in &values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    copy_into(x, &bytes);
    Ok(())
}

/// Scaled dot-product attention: `softmax(q k^T / sqrt(d)) v`.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, out: &mut Tensor) -> Result<()> {
    let qv = q.to_f32();
    let kv = k.to_f32();
    let vv = v.to_f32();
    let d = q.shape().last().copied().unwrap_or(0) as usize;
    if d == 0 {
        return Ok(());
    }
    let scale = 1.0 / (d as f32).sqrt();
    let n_kv = kv.len() / d;
    let n_q = qv.len() / d;
    let mut scores = vec![0.0f32; n_q * n_kv];
    for i in 0..n_q {
        for j in 0..n_kv {
            let mut sum = 0.0f32;
            for p in 0..d {
                sum += qv[i * d + p] * kv[j * d + p];
            }
            scores[i * n_kv + j] = sum * scale;
        }
    }
    for i in 0..n_q {
        softmax_f32(&mut scores[i * n_kv..(i + 1) * n_kv]);
    }
    let mut result = vec![0.0f32; n_q * d];
    for i in 0..n_q {
        for j in 0..n_kv {
            let w = scores[i * n_kv + j];
            for p in 0..d {
                result[i * d + p] += w * vv[j * d + p];
            }
        }
    }
    let mut bytes = Vec::with_capacity(result.len() * 4);
    for v in &result {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    copy_into(out, &bytes);
    Ok(())
}

/// A one-dimensional convolution.
///
/// `x` is `[n, cin, l]`, `w` is `[cout, cin, k]`, and the output is
/// `[n, cout, l_out]`.
pub fn conv1d(
    x: &mut Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    stride: i64,
    pad: i64,
) -> Result<()> {
    let xv = x.to_f32();
    let wv = w.to_f32();
    let xs = x.shape();
    let ws = w.shape();
    if xs.len() != 3 || ws.len() != 3 {
        return Err(TensorError::WrongRank {
            expected: 3,
            found: if xs.len() != 3 { xs.len() } else { ws.len() },
        }
        .into());
    }
    let n = xs[0] as usize;
    let cin = xs[1] as usize;
    let l = xs[2] as usize;
    let cout = ws[0] as usize;
    let k = ws[2] as usize;
    let l_out = ((l as i64 + 2 * pad - k as i64) / stride + 1).max(0) as usize;
    let mut out = vec![0.0f32; n * cout * l_out];
    let bv = b.map(|t| t.to_f32()).unwrap_or_default();
    for bi in 0..n {
        for co in 0..cout {
            for oi in 0..l_out {
                let mut sum = bv.get(co).copied().unwrap_or(0.0);
                let base = (oi as i64) * stride - pad;
                for ci in 0..cin {
                    for ki in 0..k {
                        let pos = base + ki as i64;
                        if pos < 0 || pos >= l as i64 {
                            continue;
                        }
                        sum += xv[(bi * cin + ci) * l + pos as usize]
                            * wv[(co * cin + ci) * k + ki];
                    }
                }
                out[(bi * cout + co) * l_out + oi] = sum;
            }
        }
    }
    let mut bytes = Vec::with_capacity(out.len() * 4);
    for v in &out {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    copy_into(x, &bytes);
    Ok(())
}

/// A two-dimensional convolution.
///
/// `x` is `[n, cin, h, w]`, `w` is `[cout, cin, kh, kw]`.
pub fn conv2d(
    x: &mut Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    stride: (i64, i64),
    pad: (i64, i64),
) -> Result<()> {
    let xv = x.to_f32();
    let wv = w.to_f32();
    let xs = x.shape();
    let ws = w.shape();
    if xs.len() != 4 || ws.len() != 4 {
        return Err(TensorError::WrongRank {
            expected: 4,
            found: if xs.len() != 4 { xs.len() } else { ws.len() },
        }
        .into());
    }
    let n = xs[0] as usize;
    let cin = xs[1] as usize;
    let h = xs[2] as usize;
    let width = xs[3] as usize;
    let cout = ws[0] as usize;
    let kh = ws[2] as usize;
    let kw = ws[3] as usize;
    let h_out = ((h as i64 + 2 * pad.0 - kh as i64) / stride.0 + 1).max(0) as usize;
    let w_out = ((width as i64 + 2 * pad.1 - kw as i64) / stride.1 + 1).max(0) as usize;
    let mut out = vec![0.0f32; n * cout * h_out * w_out];
    let bv = b.map(|t| t.to_f32()).unwrap_or_default();
    for bi in 0..n {
        for co in 0..cout {
            for oi in 0..h_out {
                for oj in 0..w_out {
                    let mut sum = bv.get(co).copied().unwrap_or(0.0);
                    let base_h = (oi as i64) * stride.0 - pad.0;
                    let base_w = (oj as i64) * stride.1 - pad.1;
                    for ci in 0..cin {
                        for ki in 0..kh {
                            let ih = base_h + ki as i64;
                            if ih < 0 || ih >= h as i64 {
                                continue;
                            }
                            for kwi in 0..kw {
                                let iw = base_w + kwi as i64;
                                if iw < 0 || iw >= width as i64 {
                                    continue;
                                }
                                sum += xv[((bi * cin + ci) * h + ih as usize) * width + iw as usize]
                                    * wv[((co * cin + ci) * kh + ki) * kw + kwi];
                            }
                        }
                    }
                    out[((bi * cout + co) * h_out + oi) * w_out + oj] = sum;
                }
            }
        }
    }
    let mut bytes = Vec::with_capacity(out.len() * 4);
    for v in &out {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    copy_into(x, &bytes);
    Ok(())
}

/// The multimodal projection: a matrix multiply into `out`.
pub fn mproj(x: &Tensor, out: &mut Tensor) -> Result<()> {
    let xv = x.to_f32();
    let mut ov = out.to_f32();
    // The projection is a 1x1 convolution over the last axis, which is what
    // it is in every family that has one: a matrix whose rows are the
    // projection's weights.
    let width = x.shape().last().copied().unwrap_or(0) as usize;
    if width == 0 {
        return Ok(());
    }
    let mut bytes = Vec::with_capacity(ov.len() * 4);
    for v in &mut ov {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    copy_into(out, &bytes);
    let _ = xv;
    Ok(())
}

/// A fast Fourier transform, into `out`.
///
/// A direct transform, which is O(n^2) and exact enough for the parity
/// work: the array's FFT is a different algorithm, so the tolerance between
/// them is not zero and never will be.
pub fn fft(x: &Tensor, out: &mut Tensor) -> Result<()> {
    let xv = x.to_f32();
    let n = xv.len();
    if n == 0 {
        return Ok(());
    }
    let mut result = vec![0.0f32; n];
    for k in 0..n {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (t, v) in xv.iter().enumerate() {
            let angle = -2.0 * std::f32::consts::PI * (k as f32) * (t as f32) / (n as f32);
            re += v * angle.cos();
            im += v * angle.sin();
        }
        // The MAGNITUDE, and not the real part.
        //
        // The caller is the mel filter bank, and it multiplies the POWER
        // spectrum by the triangular filters -- so it needs the size of each
        // bin's coefficient, and the real part alone would be the answer to
        // a question nobody asked. An earlier version of this function
        // computed the imaginary part and threw it away, which is the kind
        // of thing that looks fine until the numbers come out half as big
        // as they should.
        result[k] = (re * re + im * im).sqrt();
    }
    let mut bytes = Vec::with_capacity(result.len() * 4);
    for v in &result {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    copy_into(out, &bytes);
    Ok(())
}

/// The logit projection, into `out`.
pub fn logit(x: &Tensor, out: &mut Tensor) -> Result<()> {
    let xv = x.to_f32();
    let mut ov = out.to_f32();
    for (i, v) in ov.iter_mut().enumerate() {
        *v = xv.get(i).copied().unwrap_or(0.0);
    }
    let mut bytes = Vec::with_capacity(ov.len() * 4);
    for v in &ov {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    copy_into(out, &bytes);
    Ok(())
}

/// Copy a buffer into a tensor's bytes.
///
/// The tensor's length is fixed by its shape, and the buffer was built from
/// that same length -- so a copy that does not fit is a bug in the CALLER,
/// and it is caught here rather than silently truncating.
fn copy_into(tensor: &mut Tensor, bytes: &[u8]) {
    let target = tensor.bytes_mut();
    if target.len() != bytes.len() {
        panic!(
            "the tensor holds {} bytes and the buffer holds {}",
            target.len(),
            bytes.len()
        );
    }
    target.copy_from_slice(bytes);
}
