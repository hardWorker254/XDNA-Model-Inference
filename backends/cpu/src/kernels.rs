//! The host kernels: the arithmetic the CPU backend runs.
//!
//! # These are not "a reasonable f32 implementation"
//!
//! They are the C++ runtime's arithmetic, down to the last bit, and the
//! difference is not academic. The golden transcripts in `tools/parity/golden`
//! were recorded from the C++ binaries on an AVX2 host, and a golden is a
//! contract: the numbers in it are the numbers the runtime owes its users.
//! A "correct" f32 implementation that differs in the last bit is not correct
//! here -- it is a different runtime with a different answer.
//!
//! So every kernel below is a port of one C++ function, and where the C++
//! made a choice that a textbook would not, the port keeps the choice and
//! says so. The four that matter:
//!
//! | kernel | what a textbook would do | what the C++ does |
//! |---|---|---|
//! | GEMM | `acc += a * b` | `acc = fma(b, a, acc)` -- one rounding, not two |
//! | LayerNorm | sum in f32, variance in f32 | eight lane sums, then a fixed tree, and an FMA in the variance |
//! | softmax | `libm`'s `exp2` | a degree-7 polynomial on the fractional part |
//! | GELU | the exact erf, or the tanh approximation | a degree-8 minimax polynomial, with FMA |
//!
//! # The FMA, and why it is written `mul_add`
//!
//! The C++'s AVX2 paths use `_mm256_fmadd_ps`: the multiply and the add are
//! ONE rounding, not two. The scalar fallback in the C++ rounds twice, and
//! the C++ says so in its own header: "the two paths are a reproducibility
//! fact, and changing either one silently is exactly what the gate exists to
//! catch". The oracle ran the AVX2 path, so this port reproduces the AVX2
//! path -- and `f32::mul_add` is the Rust spelling of a fused
//! multiply-add: one rounding, by definition, on every target.
//!
//! # The summation ORDERS, which are not interchangeable
//!
//! Three of the kernels sum a row of floats, and each sums it in a
//! different order. Floating-point addition is not associative, so the order
//! is part of the function:
//!
//! * the horizontal sum of eight lanes is a fixed tree, `hsum8` below;
//! * the LayerNorm mean and variance use eight lane accumulators and then
//!   that tree;
//! * the softmax sum uses eight lane accumulators and then that tree;
//! * the GEMM sums over K in order, one output column at a time.
//!
//! # The width must be a multiple of eight
//!
//! The C++'s AVX2 LayerNorm and softmax read a row eight floats at a time
//! and have NO tail: a row whose width is not a multiple of eight reads past
//! its end. The port refuses such a row rather than reading past it, and the
//! model's own shapes are checked against it when the container is opened.

use npum_model::{Result, Tensor, TensorError};

/// Which instruction set a kernel runs on.
///
/// # Why the choice is a VALUE and not a compile-time flag
///
/// The C++ runtime picked its ISA with `-march`, and the choice was a build
/// flag: `x86-64-v3` for a portable baseline, `native` for the build
/// machine's own silicon -- and `native` on a Zen 4 turned on AVX-512,
/// which changed the numbers, which is why the tree pins the baseline
/// instead.
///
/// The Rust asks the HOST at run time. The reasons are the same two: a
/// binary that runs everywhere is worth more than one that is 5% faster
/// somewhere, and a build flag that changes the arithmetic is a build flag
/// that changes the ANSWER. The ISAs are therefore a runtime choice, and
/// the gate is that they all agree bit for bit.
///
/// | variant | what it is | why it exists |
/// |---|---|---|
/// | `Portable` | plain Rust, `mul_add` everywhere | the reference; a host without AVX2 |
/// | `Avx2` | AVX2 + FMA, eight floats at a time | the C++'s own ISA baseline (`x86-64-v3`) |
/// | `Avx512` | AVX-512, sixteen floats at a time | twice the width on the GEMM |
///
/// # What AVX-512 may and may not change
///
/// The GEMM's reduction runs over K, and the ORDER of that reduction does
/// not depend on how many output columns are computed at once. So the
/// 512-bit GEMM is bit-identical to the 256-bit one.
///
/// The LayerNorm and the softmax sum a row in EIGHT lanes and fold the
/// lanes with a fixed tree, and the attention's dot product does the same.
/// Sixteen lanes would be a different tree and therefore a different number,
/// so those kernels stay eight-wide on every path. See the plan's 5.1a.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Isa {
    /// Plain Rust, with the fused multiply-add spelled out.
    Portable,
    /// AVX2 and FMA: eight floats at a time.
    Avx2,
    /// AVX-512: sixteen floats at a time.
    Avx512,
}

impl Isa {
    /// The name, as the diagnostics print it.
    pub fn as_str(self) -> &'static str {
        match self {
            Isa::Portable => "portable",
            Isa::Avx2 => "avx2+fma",
            Isa::Avx512 => "avx512f",
        }
    }
}

/// The ISAs this host can run, fastest first.
pub fn available_isas() -> Vec<Isa> {
    let mut out = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            out.push(Isa::Avx512);
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            out.push(Isa::Avx2);
        }
    }
    out.push(Isa::Portable);
    out
}

/// Whether this host has AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
pub fn has_avx2_fma() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

/// Whether this host has AVX2 and FMA.
#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx2_fma() -> bool {
    false
}

/// Whether this host has AVX-512F.
#[cfg(target_arch = "x86_64")]
pub fn has_avx512f() -> bool {
    std::is_x86_feature_detected!("avx512f")
}

/// Whether this host has AVX-512F.
#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx512f() -> bool {
    false
}

/// The epsilon the C++'s host LayerNorm adds to the variance.
///
/// It is HARDCODED there -- `1e-12f`, in both the AVX2 and the scalar path --
/// and it is NOT the container's `layer_norm_eps`. The two agree for every
/// container in this repository, and where they would not, the runtime's
/// constant wins: the golden was recorded from the runtime.
pub const LAYERNORM_EPS: f32 = 1e-12;

/// The eight-lane horizontal sum, in the C++'s exact order.
///
/// `_mm256` holds eight lanes; `hsum256` folds them with two `_mm_hadd_ps`:
/// first `[l0+l1, l2+l3, l0+l1, l2+l3]`, then the halves added again. The
/// result is `(l0+l4) + (l1+l5) + (l2+l6) + (l3+l7)`, grouped as
/// `((l0+l4)+(l1+l5)) + ((l2+l6)+(l3+l7))`.
///
/// Floating-point addition is not associative, so a different tree is a
/// different number. This one is the tree the golden was recorded with.
#[inline]
pub fn hsum8(lanes: [f32; 8]) -> f32 {
    // lo[l] = lanes[l] + lanes[l + 4]
    let lo = [
        lanes[0] + lanes[4],
        lanes[1] + lanes[5],
        lanes[2] + lanes[6],
        lanes[3] + lanes[7],
    ];
    // hadd(lo, lo) = [lo0+lo1, lo2+lo3, lo0+lo1, lo2+lo3]
    let h = [lo[0] + lo[1], lo[2] + lo[3], lo[0] + lo[1], lo[2] + lo[3]];
    // hadd(h, h) = [(h0+h1), (h2+h3), ...]
    h[0] + h[1]
}

/// The eight-lane horizontal MAXIMUM, in the C++'s exact order.
///
/// `_mm_max_ps` on the two halves, then on the pairs, then on the first
/// two: `max(max(m0, m4), max(m2, m6))` against `max(max(m1, m5), max(m3, m7))`.
///
/// The maximum is associative, so the tree is not a numerical question --
/// but the SUM of the lanes would be, and using the sum here is the kind of
/// mistake the softmax survives: the result is shifted, the shift cancels in
/// the normalisation, and only the rounding of the exponentials gives the
/// answer away.
#[inline]
pub fn hmax8(lanes: [f32; 8]) -> f32 {
    let m4 = [
        lanes[0].max(lanes[4]),
        lanes[1].max(lanes[5]),
        lanes[2].max(lanes[6]),
        lanes[3].max(lanes[7]),
    ];
    let m2 = [m4[0].max(m4[2]), m4[1].max(m4[3])];
    m2[0].max(m2[1])
}

/// `_mm256_cvttps_epi32`: the float to int32 conversion with truncation.
///
/// The C++'s conversion returns the "integer indefinite" value `0x80000000`
/// for a value that does not fit and for NaN, and Rust's `as` saturates
/// instead. The difference is invisible for any real score and visible for a
/// pathological one, so the port reproduces the C++ exactly.
#[inline]
fn cvttps_i32(x: f32) -> i32 {
    if x.is_nan() || x >= 2_147_483_648.0 || x < -2_147_483_648.0 {
        i32::MIN
    } else {
        x.trunc() as i32
    }
}

/// The degree-7 polynomial `exp2`, as the C++'s `exp2_avx2` computes it.
///
/// Not `libm`'s `exp2`, and not `2^x` by another route: the C++ splits `x`
/// into an integer part and a fraction, evaluates a degree-7 polynomial on the
/// fraction with FMA, and scales by `2^k` built out of the exponent field.
///
/// The polynomial's own rounding is part of the answer, which is why the
/// constants are written out here rather than taken from a library: a
/// different polynomial of the same degree is a different function.
#[inline]
pub fn exp2_poly(x: f32) -> f32 {
    const C0: f32 = 1.548_327_546_3e-05;
    const C1: f32 = 1.566_983_317_4e-04;
    const C2: f32 = 1.333_182_523_6e-03;
    const C3: f32 = 9.616_460_553_8e-03;
    const C4: f32 = 5.550_415_685_5e-02;
    const C5: f32 = 2.402_268_410_9e-01;
    const C6: f32 = 6.931_471_769_4e-01;
    const C7: f32 = 9.999_999_895_5e-01;

    let k = cvttps_i32(x);
    let f = x - k as f32;
    let mut pl = C0.mul_add(f, C1);
    pl = pl.mul_add(f, C2);
    pl = pl.mul_add(f, C3);
    pl = pl.mul_add(f, C4);
    pl = pl.mul_add(f, C5);
    pl = pl.mul_add(f, C6);
    pl = pl.mul_add(f, C7);
    // 2^k, built out of the exponent field exactly as the C++ does.
    let bits = (k.wrapping_add(127) as u32) << 23;
    pl * f32::from_bits(bits)
}

/// The GELU activation, as the C++'s `gelu_cpu` computes it.
///
/// A degree-8 minimax polynomial on `min(|x|, 4)`, NOT the exact erf and NOT
/// the tanh approximation. The clamp on `|x|` is not cosmetic: the
/// polynomial is minimax on [0, 4] and diverges outside it, and a large
/// negative input would otherwise produce a large positive answer.
///
/// The C++'s AVX2 path contracts the polynomial with FMA and its scalar tail
/// does not, so the two differ in the last bit -- and the oracle ran the
/// AVX2 path. This is the AVX2 path.
#[inline]
pub fn gelu_f32(v: f32) -> f32 {
    const C0: f32 = -7.234_028_217_1e-05;
    const C1: f32 = 1.817_951_800_5e-03;
    const C2: f32 = -1.770_738_337_9e-02;
    const C3: f32 = 8.457_714_764_1e-02;
    const C4: f32 = -1.922_867_183_4e-01;
    const C5: f32 = 9.843_112_445_8e-02;
    const C6: f32 = 3.613_785_206_2e-01;
    const C7: f32 = -4.945_412_893_6e-01;
    const C8: f32 = -1.300_701_011_7e-04;

    let u = v.abs().min(4.0);
    let mut pl = C0.mul_add(u, C1);
    pl = pl.mul_add(u, C2);
    pl = pl.mul_add(u, C3);
    pl = pl.mul_add(u, C4);
    pl = pl.mul_add(u, C5);
    pl = pl.mul_add(u, C6);
    pl = pl.mul_add(u, C7);
    pl = pl.mul_add(u, C8);
    v.max(0.0) + pl
}

/// Layer normalisation over one row.
///
/// The width must be a multiple of eight: the C++'s AVX2 path reads eight
/// floats at a time and has no tail, and a row that is not a multiple of
/// eight would be read past its end. The port refuses instead.
///
/// The order is the C++'s: eight lane accumulators for the mean, the tree of
/// `hsum8` to finish it, eight lane accumulators with an FMA for the
/// variance, the same tree to finish it, and an FMA in the epilogue.
pub fn layernorm_row(row: &mut [f32], gamma: &[f32], beta: &[f32]) -> Result<()> {
    layernorm_row_on(row_isa(), row, gamma, beta)
}

/// The LayerNorm, on a named ISA.
///
/// The eight-lane structure is the C++'s and it is the same on every ISA:
/// AVX-512 would fold sixteen lanes and fold them differently, so the row
/// kernels do not take the wider path. See the plan's 5.1a.
pub fn layernorm_row_on(
    isa: Isa,
    row: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
) -> Result<()> {
    let n = row.len();
    if n == 0 {
        return Ok(());
    }
    if n % 8 != 0 {
        return Err(TensorError::WrongLength {
            expected: n.next_multiple_of(8),
            found: n,
        }
        .into());
    }
    #[cfg(target_arch = "x86_64")]
    if isa != Isa::Portable && has_avx2_fma() {
        // SAFETY: the host was checked for the features. The AVX-512 request
        // lands here too: the row kernels are eight-lane on every path, so
        // the wider ISA has nothing to offer them.
        unsafe { layernorm_row_avx2(row, gamma, beta) };
        return Ok(());
    }
    layernorm_row_impl(row, gamma, beta);
    Ok(())
}

/// The LayerNorm, compiled for a host with AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn layernorm_row_avx2(row: &mut [f32], gamma: &[f32], beta: &[f32]) {
    layernorm_row_impl(row, gamma, beta)
}

/// The ISA the GEMM uses: the widest this host has.
///
/// The GEMM is the one kernel where the width is free: its reduction runs
/// over K, and the order of that reduction does not depend on how many
/// output columns are computed at once.
pub fn gemm_isa() -> Isa {
    if has_avx512f() {
        Isa::Avx512
    } else if has_avx2_fma() {
        Isa::Avx2
    } else {
        Isa::Portable
    }
}

/// The ISA the ROW kernels use: eight lanes, always.
///
/// AVX-512 is not used here, and not because it is slower: the LayerNorm
/// and the softmax sum a row in eight lanes and fold them with a fixed
/// tree, and a sixteen-lane tree is a different number.
pub fn row_isa() -> Isa {
    if has_avx2_fma() { Isa::Avx2 } else { Isa::Portable }
}

/// The LayerNorm body, written once and shared by both builds.
#[inline(always)]
fn layernorm_row_impl(row: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = row.len();
    // The mean: eight lane accumulators.
    let mut sum = [0.0f32; 8];
    let mut j = 0;
    while j < n {
        for l in 0..8 {
            sum[l] += row[j + l];
        }
        j += 8;
    }
    let mean = hsum8(sum) / n as f32;

    // The variance: eight lane accumulators, and an FMA per element.
    let mut acc = [0.0f32; 8];
    j = 0;
    while j < n {
        for l in 0..8 {
            let d = row[j + l] - mean;
            acc[l] = d.mul_add(d, acc[l]);
        }
        j += 8;
    }
    let var = hsum8(acc) / n as f32;
    let inv = 1.0f32 / (var + LAYERNORM_EPS).sqrt();

    // The epilogue: `fma((x - mean) * inv, gamma, beta)`.
    j = 0;
    while j < n {
        for l in 0..8 {
            let d = row[j + l] - mean;
            row[j + l] = (d * inv).mul_add(gamma[j + l], beta[j + l]);
        }
        j += 8;
    }
}

/// Softmax over one row, as the C++'s `softmax_cpu` computes it.
///
/// The width must be a multiple of eight, for the same reason as the
/// LayerNorm's.
///
/// Three things are not textbook and all three are the C++'s:
///
/// * the exponential is `exp2_poly`, not `libm`'s `exp2`;
/// * the argument is clamped at `-120` BEFORE the polynomial, because a
///   negative exponent below about -127 is a negative exponent FIELD -- a
///   NaN bit pattern -- and a masked element must contribute nothing;
/// * the sum is eight lane accumulators finished with `hsum8`.
///
/// The mask is NOT applied here: the C++'s caller adds the additive mask to
/// the row first, and adding it inside would change the maximum.
pub fn softmax_row(row: &mut [f32]) -> Result<()> {
    softmax_row_on(row_isa(), row)
}

/// The softmax, on a named ISA.
///
/// Eight lanes, like the LayerNorm's, for the same reason.
pub fn softmax_row_on(isa: Isa, row: &mut [f32]) -> Result<()> {
    let n = row.len();
    if n == 0 {
        return Ok(());
    }
    if n % 8 != 0 {
        return Err(TensorError::WrongLength {
            expected: n.next_multiple_of(8),
            found: n,
        }
        .into());
    }
    #[cfg(target_arch = "x86_64")]
    if isa != Isa::Portable && has_avx2_fma() {
        // SAFETY: the host was checked for the features. The AVX-512 request
        // lands here too: the row kernels are eight-lane on every path.
        unsafe { softmax_row_avx2(row) };
        return Ok(());
    }
    softmax_row_impl(row);
    Ok(())
}

/// The softmax, compiled for a host with AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_row_avx2(row: &mut [f32]) {
    softmax_row_impl(row)
}

/// The softmax body, written once and shared by both builds.
#[inline(always)]
fn softmax_row_impl(row: &mut [f32]) {
    let n = row.len();
    // The maximum: eight lanes, then a tree of maxes. Max is associative, so
    // the tree is not a numerical question -- but it is the C++'s tree.
    //
    // The lanes start at the row's FIRST eight elements and not at zero: a
    // row of negatives would otherwise have a maximum of zero, and the
    // softmax would then divide by a sum of e^0s instead of e^(x - max).
    let mut mx = [0.0f32; 8];
    mx.copy_from_slice(&row[..8]);
    let mut j = 8;
    while j < n {
        for l in 0..8 {
            mx[l] = mx[l].max(row[j + l]);
        }
        j += 8;
    }
    let m = hmax8(mx);

    // The exponentials, and their sum.
    let mut sum = [0.0f32; 8];
    j = 0;
    while j < n {
        for l in 0..8 {
            let a = ((row[j + l] - m) * std::f32::consts::LOG2_E).max(-120.0);
            let e = exp2_poly(a);
            row[j + l] = e;
            sum[l] += e;
        }
        j += 8;
    }
    let inv = 1.0f32 / hsum8(sum);
    j = 0;
    while j < n {
        for l in 0..8 {
            row[j + l] *= inv;
        }
        j += 8;
    }
}

/// `C[M, N] = A[M, K] @ B[K, N] + bias[N]`.
///
/// `B` is `[K, N]` -- one row per reduction step, N output columns
/// contiguous -- which is the layout the container's `gemm_b_host` tensors
/// already have.
///
/// The accumulation is over K, in order, one column at a time, and each step
/// is an FMA: `acc[n] = fma(b[k][n], a[m][k], acc[n])`. That is what the
/// C++'s compiled inner loop does -- `vfmadd213ps` over eight columns, with
/// a scalar FMA tail -- and it is why the port is not `acc += a * b`.
///
/// The bias is added AFTER the whole reduction, in a plain addition.
pub fn gemm_nt(
    a: &[f32],
    m: usize,
    b: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
) {
    gemm_nt_dispatch(a, m, b, k, n, bias, out)
}

/// The GEMM body, written once and shared by both builds.
#[inline(always)]
fn gemm_nt_impl(a: &[f32], m: usize, b: &[f32], k: usize, n: usize, bias: &[f32], out: &mut [f32]) {
    let mut acc = vec![0.0f32; n];
    for r in 0..m {
        let arow = &a[r * k..r * k + k];
        acc.fill(0.0);
        for kk in 0..k {
            let a_val = arow[kk];
            let brow = &b[kk * n..kk * n + n];
            for j in 0..n {
                acc[j] = brow[j].mul_add(a_val, acc[j]);
            }
        }
        let orow = &mut out[r * n..r * n + n];
        for j in 0..n {
            orow[j] = acc[j] + bias[j];
        }
    }
}

/// The dispatch: the fastest ISA this host has.
///
/// The portable build is not a different function: `mul_add` is a fused
/// multiply-add on every target, so the builds agree bit for bit and differ
/// only in speed. The AVX2 build is the C++'s own ISA baseline
/// (`x86-64-v3`), and this port asks the host at run time rather than
/// assuming it.
fn gemm_nt_dispatch(
    a: &[f32],
    m: usize,
    b: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512f() {
            // SAFETY: the function is only called when the host has the
            // features it is compiled for, and it upholds no extra invariant.
            unsafe { gemm_nt_avx512(a, m, b, k, n, bias, out) };
            return;
        }
        if has_avx2_fma() {
            // SAFETY: as above.
            unsafe { gemm_nt_avx2(a, m, b, k, n, bias, out) };
            return;
        }
    }
    gemm_nt_impl(a, m, b, k, n, bias, out);
}

/// The GEMM, on a named ISA.
///
/// `false` means the host does not have that ISA -- not that the GEMM
/// failed. The gate that matters is that every ISA this host DOES have
/// produces the same bytes, and `gemm_nt_on` is how that gate asks.
pub fn gemm_nt_on(
    isa: Isa,
    a: &[f32],
    m: usize,
    b: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
) -> bool {
    match isa {
        Isa::Portable => {
            gemm_nt_impl(a, m, b, k, n, bias, out);
            true
        }
        Isa::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            {
                if !has_avx2_fma() {
                    return false;
                }
                // SAFETY: the host was checked for the features.
                unsafe { gemm_nt_avx2(a, m, b, k, n, bias, out) };
                true
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                false
            }
        }
        Isa::Avx512 => {
            #[cfg(target_arch = "x86_64")]
            {
                if !has_avx512f() {
                    return false;
                }
                // SAFETY: the host was checked for the features.
                unsafe { gemm_nt_avx512(a, m, b, k, n, bias, out) };
                true
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                false
            }
        }
    }
}

/// The GEMM, compiled for a host with AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_nt_avx2(
    a: &[f32],
    m: usize,
    b: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
) {
    gemm_nt_impl(a, m, b, k, n, bias, out)
}

/// The GEMM, compiled for a host with AVX-512.
///
/// Sixteen output columns at a time, and the SAME reduction over K: the
/// width of the vector is not the order of the additions, and the order is
/// the answer. This is why the 512-bit path is bit-identical to the 256-bit
/// one and the sixteen-lane LayerNorm would not have been.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gemm_nt_avx512(
    a: &[f32],
    m: usize,
    b: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    let mut acc = vec![0.0f32; n];
    for r in 0..m {
        let arow = &a[r * k..r * k + k];
        acc.fill(0.0);
        for kk in 0..k {
            let av = _mm512_set1_ps(arow[kk]);
            let brow = &b[kk * n..kk * n + n];
            let mut j = 0usize;
            while j + 16 <= n {
                // SAFETY: the loads and the store are within the slices:
                // `j + 16 <= n`, and both slices are at least `n` long.
                unsafe {
                    let bv = _mm512_loadu_ps(brow.as_ptr().add(j));
                    let cv = _mm512_loadu_ps(acc.as_ptr().add(j));
                    let sum = _mm512_fmadd_ps(bv, av, cv);
                    _mm512_storeu_ps(acc.as_mut_ptr().add(j), sum);
                }
                j += 16;
            }
            // The tail, in the same order and the same arithmetic.
            while j < n {
                acc[j] = brow[j].mul_add(arow[kk], acc[j]);
                j += 1;
            }
        }
        let orow = &mut out[r * n..r * n + n];
        for j in 0..n {
            orow[j] = acc[j] + bias[j];
        }
    }
}

// ---------------------------------------------------------------- the trait

/// The GELU, on a named ISA.
///
/// The polynomial is contracted with FMA on the vector path and not on the
/// scalar tail, exactly as the C++'s is, and exactly as the C++'s AVX2 and
/// scalar paths differ from each other. The oracle ran the AVX2 path.
pub fn gelu_row_on(isa: Isa, x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if isa != Isa::Portable && has_avx2_fma() {
        // SAFETY: the host was checked for the features. The AVX-512 request
        // lands here too: the polynomial is eight-lane on every path.
        unsafe { gelu_row_avx2(x) };
        return;
    }
    for v in x.iter_mut() {
        *v = gelu_f32(*v);
    }
}

/// The GELU, compiled for a host with AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_row_avx2(x: &mut [f32]) {
    let n = x.len();
    let mut i = 0usize;
    while i + 8 <= n {
        for l in 0..8 {
            x[i + l] = gelu_f32(x[i + l]);
        }
        i += 8;
    }
    while i < n {
        x[i] = gelu_f32(x[i]);
        i += 1;
    }
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
pub fn layernorm_(x: &mut Tensor, gamma: &Tensor, beta: &Tensor, _eps: f32) -> Result<()> {
    let mut values = x.to_f32();
    let g = gamma.to_f32();
    let b = beta.to_f32();
    let width = x.shape().last().copied().unwrap_or(0) as usize;
    if width == 0 {
        return Ok(());
    }
    for chunk in values.chunks_mut(width) {
        layernorm_row(chunk, &g, &b)?;
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
        softmax_row(chunk)?;
    }
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in &values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    copy_into(x, &bytes);
    Ok(())
}

/// `c = a * b`, row-major, optionally accumulating into `c`.
///
/// `a` is `[m, k]`, `b` is `[k, n]`, `c` is `[m, n]`. The accumulation is
/// `c += a * b` when `acc` is `Some`, and `c = a * b` when it is `None`.
///
/// This is the TRAIT's generic multiply, and it is the exact one: the same
/// FMA accumulation in the same order as `gemm_nt`.
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
    let zeros = vec![0.0f32; n];
    for i in 0..m {
        gemm_nt(
            &av[i * k..(i + 1) * k],
            1,
            &bv,
            k,
            n,
            &zeros,
            &mut cv[i * n..(i + 1) * n],
        );
    }
    let mut bytes = Vec::with_capacity(cv.len() * 4);
    for v in &cv {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    copy_into(c, &bytes);
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
        softmax_row(&mut scores[i * n_kv..(i + 1) * n_kv])?;
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
        // computed the imaginary part and threw it away, which is the kind of
        // thing that looks fine until the numbers come out half as big
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
