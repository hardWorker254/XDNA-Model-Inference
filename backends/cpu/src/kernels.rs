//! The host kernels: the arithmetic the CPU backend runs.
//!
//! # These are not "a reasonable f32 implementation"
//!
//! They are the reference runtime's arithmetic, down to the last bit, and the
//! difference is not academic. The golden transcripts in `tools/parity/golden`
//! were recorded from the reference binaries on an AVX2 host, and a golden is a
//! contract: the numbers in it are the numbers the runtime owes its users.
//! A "correct" f32 implementation that differs in the last bit is not correct
//! here -- it is a different runtime with a different answer.
//!
//! So every kernel below is a port of one reference function, and where the reference
//! made a choice that a textbook would not, the port keeps the choice and
//! says so. The four that matter:
//!
//! | kernel | what a textbook would do | what the reference does |
//! |---|---|---|
//! | GEMM | `acc += a * b` | `acc = fma(b, a, acc)` -- one rounding, not two |
//! | LayerNorm | sum in f32, variance in f32 | eight lane sums, then a fixed tree, and an FMA in the variance |
//! | softmax | `libm`'s `exp2` | a degree-7 polynomial on the fractional part |
//! | GELU | the exact erf, or the tanh approximation | a degree-8 minimax polynomial, with FMA |
//!
//! # The FMA, and why it is written `mul_add`
//!
//! The reference's AVX2 paths use `_mm256_fmadd_ps`: the multiply and the add are
//! ONE rounding, not two. The scalar fallback in the reference rounds twice, and
//! the reference says so in its own header: "the two paths are a reproducibility
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
//! The reference's AVX2 LayerNorm and softmax read a row eight floats at a time
//! and have NO tail: a row whose width is not a multiple of eight reads past
//! its end. The port refuses such a row rather than reading past it, and the
//! model's own shapes are checked against it when the container is opened.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

use npum_model::{Result, Tensor, TensorError};

use crate::parallel;

/// Kept as a re-export: the pipelines have always read this constant from the
/// kernels module, and the policy itself now lives in `parallel`.
pub use crate::parallel::PAR_MIN_ITEMS;

/// Which instruction set a kernel runs on.
///
/// # Why the choice is a VALUE and not a compile-time flag
///
/// The reference runtime picked its ISA with `-march`, and the choice was a build
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
/// | `Avx2` | AVX2 + FMA, eight floats at a time | the reference's own ISA baseline (`x86-64-v3`) |
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
        #[cfg(has_stable_avx512)]
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
///
/// Compiled AVX-512 kernels require stable AVX-512 intrinsics (Rust 1.89+);
/// on older toolchains this reports false so dispatch stays on the AVX2 or
/// portable paths.
#[cfg(target_arch = "x86_64")]
pub fn has_avx512f() -> bool {
    #[cfg(has_stable_avx512)]
    {
        std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(has_stable_avx512))]
    {
        false
    }
}

/// Whether this host has AVX-512F.
#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx512f() -> bool {
    false
}

/// The epsilon the reference's host LayerNorm adds to the variance.
///
/// It is HARDCODED there -- `1e-12f`, in both the AVX2 and the scalar path --
/// and it is NOT the container's `layer_norm_eps`. The two agree for every
/// container in this repository, and where they would not, the runtime's
/// constant wins: the golden was recorded from the runtime.
pub const LAYERNORM_EPS: f32 = 1e-12;

/// The eight-lane horizontal sum, in the reference's exact order.
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

/// The eight-lane horizontal MAXIMUM, in the reference's exact order.
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
/// The reference's conversion returns the "integer indefinite" value `0x80000000`
/// for a value that does not fit and for NaN, and Rust's `as` saturates
/// instead. The difference is invisible for any real score and visible for a
/// pathological one, so the port reproduces the reference exactly.
#[inline]
fn cvttps_i32(x: f32) -> i32 {
    if x.is_nan() || x >= 2_147_483_648.0 || x < -2_147_483_648.0 {
        i32::MIN
    } else {
        x.trunc() as i32
    }
}

/// The degree-7 polynomial `exp2`, as the reference's `exp2_avx2` computes it.
///
/// Not `libm`'s `exp2`, and not `2^x` by another route: the reference splits `x`
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
    // 2^k, built out of the exponent field exactly as the reference does.
    let bits = (k.wrapping_add(127) as u32) << 23;
    pl * f32::from_bits(bits)
}

/// The GELU activation, as the reference's `gelu_cpu` computes it.
///
/// A degree-8 minimax polynomial on `min(|x|, 4)`, NOT the exact erf and NOT
/// the tanh approximation. The clamp on `|x|` is not cosmetic: the
/// polynomial is minimax on [0, 4] and diverges outside it, and a large
/// negative input would otherwise produce a large positive answer.
///
/// The reference's AVX2 path contracts the polynomial with FMA and its scalar tail
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
/// The width must be a multiple of eight: the reference's AVX2 path reads eight
/// floats at a time and has no tail, and a row that is not a multiple of
/// eight would be read past its end. The port refuses instead.
///
/// The order is the reference's: eight lane accumulators for the mean, the tree of
/// `hsum8` to finish it, eight lane accumulators with an FMA for the
/// variance, the same tree to finish it, and an FMA in the epilogue.
pub fn layernorm_row(row: &mut [f32], gamma: &[f32], beta: &[f32]) -> Result<()> {
    layernorm_row_on(row_isa(), row, gamma, beta)
}

/// The LayerNorm, on a named ISA.
///
/// The eight-lane structure is the reference's and it is the same on every ISA:
/// AVX-512 would fold sixteen lanes and fold them differently, so the row
/// kernels do not take the wider path. See the plan's 5.1a.
pub fn layernorm_row_on(isa: Isa, row: &mut [f32], gamma: &[f32], beta: &[f32]) -> Result<()> {
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
    if has_avx2_fma() {
        Isa::Avx2
    } else {
        Isa::Portable
    }
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

/// Softmax over one row, as the reference's `softmax_cpu` computes it.
///
/// The width must be a multiple of eight, for the same reason as the
/// LayerNorm's.
///
/// Three things are not textbook and all three are the reference's:
///
/// * the exponential is `exp2_poly`, not `libm`'s `exp2`;
/// * the argument is clamped at `-120` BEFORE the polynomial, because a
///   negative exponent below about -127 is a negative exponent FIELD -- a
///   NaN bit pattern -- and a masked element must contribute nothing;
/// * the sum is eight lane accumulators finished with `hsum8`.
///
/// The mask is NOT applied here: the reference's caller adds the additive mask to
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
    // the tree is not a numerical question -- but it is the reference's tree.
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
/// reference's compiled inner loop does -- `vfmadd213ps` over eight columns, with
/// a scalar FMA tail -- and it is why the port is not `acc += a * b`.
///
/// The bias is added AFTER the whole reduction, in a plain addition.
pub fn gemm_nt(a: &[f32], m: usize, b: &[f32], k: usize, n: usize, bias: &[f32], out: &mut [f32]) {
    with_gemm_acc(n, |acc| gemm_nt_dispatch(a, m, b, k, n, bias, out, acc))
}

thread_local! {
    /// The GEMM accumulator, grown on demand and kept for the next call.
    ///
    /// A pipeline runs hundreds of GEMMs at the same output width; building a
    /// fresh `vec![0.0; n]` for each one is a heap allocation (and a zeroing)
    /// that buys nothing. This is one buffer per THREAD, not per call: the
    /// row-parallel kernels run on their own workers and a worker's
    /// accumulator must be its own if the reduction order -- and so the bytes
    /// -- are to stay the same. The buffer keeps the largest width this thread
    /// has seen, so the steady state never allocates.
    static GEMM_ACC: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Run `f` with at least `len` floats of this thread's GEMM scratch.
///
/// The kernels zero the accumulator once per output row, so a reused buffer
/// needs no zeroing here; only a grown one does, and `resize` does that.
/// Nothing that borrows this cell may be called from `f` (no re-entrant GEMM):
/// the kernels below share the one buffer by passing slices explicitly.
#[inline]
fn with_gemm_acc<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    GEMM_ACC.with(|cell| {
        let mut acc = cell.borrow_mut();
        if acc.len() < len {
            acc.resize(len, 0.0);
        }
        f(&mut acc[..len])
    })
}

thread_local! {
    /// A reusable zero bias for callers that have no bias tensor.
    ///
    /// The GEMM kernels index `bias[j]` for every output column, so an empty
    /// slice is not a valid answer. Building `vec![0.0; n]` on every generic
    /// tensor call is a needless allocation; this buffer is grown once and
    /// reused for the life of the thread.
    static ZERO_BIAS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

#[inline]
fn with_zero_bias<R>(n: usize, f: impl FnOnce(&[f32]) -> R) -> R {
    ZERO_BIAS.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < n {
            buf.resize(n, 0.0);
        }
        f(&buf[..n])
    })
}

thread_local! {
    /// B, repacked into contiguous eight-column panels over K.
    ///
    /// The register-blocked kernel reads one column group across ALL of K.
    /// In the original `[k, n]` layout those reads are `n` floats apart --
    /// and when `n` is a whole number of pages they all land in one cache
    /// set. Repacking the same floats as `[ceil(n/8), k, 8]` makes one
    /// panel's 8*K floats contiguous, so a panel stays in L1 while every
    /// row block reuses it. The VALUES are untouched; only their addresses
    /// change, so the reduction order -- and the bytes -- are unchanged.
    ///
    /// One buffer per thread: a row-parallel GEMM packs on the CALLING
    /// thread and hands every worker a shared, immutable view of it.
    #[cfg(target_arch = "x86_64")]
    static B_PACK: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// The number of floats the packed layout needs.
#[cfg(target_arch = "x86_64")]
#[inline]
fn packed_b_len(k: usize, n: usize) -> usize {
    n.div_ceil(8) * 8 * k
}

/// Whether repacking B pays for itself on this shape.
///
/// A pack is an extra pass over B; it is worth it only when the packed panel
/// is read by SEVERAL row blocks. `m = 1` (single-token decode) reads B once
/// and must NOT pay for the pack. Narrow K has too little reuse either.
#[cfg(target_arch = "x86_64")]
#[inline]
fn should_pack_b(m: usize, k: usize, n: usize) -> bool {
    static NO_PACK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let disabled = *NO_PACK.get_or_init(|| std::env::var_os("NPU_CPU_NO_PACK").is_some());
    !disabled && has_avx2_fma() && m >= 8 && k >= 16 && n >= 8
}

/// Pack B into eight-column panels, reusing this thread's scratch.
#[cfg(target_arch = "x86_64")]
fn pack_b_avx2(b: &[f32], k: usize, n: usize, packed: &mut Vec<f32>) {
    let needed = packed_b_len(k, n);
    if packed.len() < needed {
        packed.resize(needed, 0.0);
    }
    let panels = n.div_ceil(8);
    for jb in 0..panels {
        let j = jb * 8;
        for kk in 0..k {
            let dst = &mut packed[(jb * k + kk) * 8..(jb * k + kk) * 8 + 8];
            // The buffer is reused: clear the tail lanes every time, since a
            // previous wider pack may have left non-zero values there.
            dst.fill(0.0);
            let live = &b[kk * n + j..kk * n + (j + 8).min(n)];
            dst[..live.len()].copy_from_slice(live);
        }
    }
}

/// Give `f` an immutable packed copy of B for this call.
#[cfg(target_arch = "x86_64")]
#[inline]
fn with_packed_b<R>(b: &[f32], k: usize, n: usize, f: impl FnOnce(&[f32]) -> R) -> R {
    B_PACK.with(|cell| {
        let mut buf = cell.borrow_mut();
        pack_b_avx2(b, k, n, &mut buf);
        let len = packed_b_len(k, n);
        f(&buf[..len])
    })
}

/// The GEMM over ROW RANGES, one per worker.
///
/// Each output row is one im2col row times the weight matrix, and a row's
/// accumulation is over `k` in order -- so splitting ROWS changes which
/// thread computes a value and never the value itself. That is what makes
/// the threads free: the split is bit-exact by construction, and a
/// comparison against the oracle does not have to know how many cores the
/// machine has.
///
/// The decision to split, the reusable pool and the row chunking all come
/// from [`parallel`]. Each pool worker owns its own reusable thread-local GEMM
/// accumulator, so no scratch has to be carved up here and no threads are
/// created per call.
pub fn gemm_nt_rows(
    a: &[f32],
    m: usize,
    b: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
    threads: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let work = (m as i64) * (k as i64) * (n as i64);
    let Some(chunk) = parallel::chunk_rows(m, work, threads) else {
        gemm_nt(a, m, b, k, n, bias, out);
        return;
    };
    #[cfg(target_arch = "x86_64")]
    if should_pack_b(m, k, n) && n % PAGE_FLOATS == 0 {
        // Pack ONCE, on the calling thread, and let every worker share the
        // immutable panels. This is only profitable for page-multiple
        // widths: there the ORIGINAL layout makes every column group fight
        // for one cache set, and packing fixes it. For ordinary widths the
        // shared panel makes many workers contend for the same cache lines,
        // and the strided original is measurably faster.
        with_packed_b(b, k, n, |packed| {
            parallel::for_each_chunk_mut(out, chunk * n, |ci, part| {
                let rows = part.len() / n;
                let start_row = ci * chunk;
                let a_part = &a[start_row * k..(start_row + rows) * k];
                with_gemm_acc(n, |acc| {
                    // SAFETY: `should_pack_b` checked the host for AVX2/FMA.
                    unsafe { gemm_nt_avx2_packed(a_part, rows, b, packed, k, n, bias, part, acc) }
                });
            });
        });
        return;
    }
    parallel::for_each_chunk_mut(out, chunk * n, |ci, part| {
        let rows = part.len() / n;
        let start_row = ci * chunk;
        let a_part = &a[start_row * k..(start_row + rows) * k];
        // Each worker's accumulator is thread-local and reused across calls.
        gemm_nt(a_part, rows, b, k, n, bias, part);
    });
}

/// The row-parallel GEMM on a NAMED ISA.
///
/// The split is the same as [`gemm_nt_rows`], but every chunk runs through
/// `gemm_nt_on` so the ISA gate keeps its byte-for-byte meaning: a chunk
/// computed on a named ISA is the same chunk the serial path would compute,
/// only on a different worker. `false` means the host cannot run that ISA.
pub fn gemm_nt_rows_on(
    isa: Isa,
    a: &[f32],
    m: usize,
    b: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
    threads: usize,
) -> bool {
    if m == 0 || n == 0 {
        return true;
    }
    let work = (m as i64) * (k as i64) * (n as i64);
    let Some(chunk) = parallel::chunk_rows(m, work, threads) else {
        return gemm_nt_on(isa, a, m, b, k, n, bias, out);
    };
    #[cfg(target_arch = "x86_64")]
    if isa == Isa::Avx2 && should_pack_b(m, k, n) && n % PAGE_FLOATS == 0 {
        // Only the page-multiple widths benefit from a shared packed panel;
        // see the same gate in `gemm_nt_rows`.
        with_packed_b(b, k, n, |packed| {
            parallel::for_each_chunk_mut(out, chunk * n, |ci, part| {
                let rows = part.len() / n;
                let start_row = ci * chunk;
                let a_part = &a[start_row * k..(start_row + rows) * k];
                with_gemm_acc(n, |acc| {
                    // SAFETY: `should_pack_b` checked the host for AVX2/FMA.
                    unsafe { gemm_nt_avx2_packed(a_part, rows, b, packed, k, n, bias, part, acc) }
                });
            });
        });
        return true;
    }
    let failed = AtomicBool::new(false);
    parallel::for_each_chunk_mut(out, chunk * n, |ci, part| {
        let rows = part.len() / n;
        let start_row = ci * chunk;
        let a_part = &a[start_row * k..(start_row + rows) * k];
        if !gemm_nt_on(isa, a_part, rows, b, k, n, bias, part) {
            failed.store(true, Ordering::Relaxed);
        }
    });
    if failed.load(Ordering::Relaxed) {
        gemm_nt_on(Isa::Portable, a, m, b, k, n, bias, out);
    }
    true
}

/// The GEMM body, written once and shared by both builds.
///
/// This is the REFERENCE: one output element at a time, one fused
/// multiply-add per reduction step, in order, with the accumulator in
/// memory. Every vector path below computes exactly this sequence per
/// element and differs only in how many elements it does at once.
#[inline(always)]
fn gemm_nt_impl(
    a: &[f32],
    m: usize,
    b: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
    acc: &mut [f32],
) {
    debug_assert_eq!(acc.len(), n);
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

/// The dispatch: the fastest ISA this host has, for THIS width.
///
/// The portable build is not a different function: `mul_add` is a fused
/// multiply-add on every target, so the builds agree bit for bit and differ
/// only in speed. The AVX2 build is the reference's own ISA baseline
/// (`x86-64-v3`), and this port asks the host at run time rather than
/// assuming it.
///
/// # THE WIDTH DECIDES, NOT ONLY THE HOST
///
/// A host with AVX-512 has the sixteen-lane kernel, and the sixteen-lane
/// kernel needs SIXTEEN output columns to fill a register. The text
/// detector's first two convolutions have EIGHT output channels each, and
/// the four that cost the most have TWENTY-FOUR -- which is eight columns
/// and a half.
///
/// So the sixteen-lane kernel was running its SCALAR tail over eight of
/// every twenty-four columns, and the detector's GEMM took 908 ms where the
/// oracle's took 77. The host's widest vector is not the answer to "how
/// wide is this GEMM": the WIDTH is.
///
/// Both vector paths now have a MASKED tail, so neither has to give the
/// last one to fifteen columns to a scalar loop. The sixteen-lane kernel is
/// used from sixteen columns up, the eight-lane kernel from eight columns up
/// (a narrower row is still eight lanes wide, just with a short mask), and
/// the row-wise kernel is kept for the cache-set strides described at
/// [`PAGE_FLOATS`]. Every path computes the same FMA in the same order for a
/// live column, which is what the ISA gate compares.
fn gemm_nt_dispatch(
    a: &[f32],
    m: usize,
    b: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
    acc: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(has_stable_avx512)]
        if has_avx512f() && n >= 16 {
            // The column tail is now MASKED rather than scalar, so a width
            // that is not a multiple of sixteen no longer forces the whole
            // GEMM onto the eight-lane path. Only the cache-set case below
            // still prefers the row-wise kernel.
            //
            // SAFETY: the function is only called when the host has the
            // features it is compiled for, and it upholds no extra invariant.
            unsafe {
                if n % PAGE_FLOATS == 0 {
                    gemm_nt_avx512_rows(a, m, b, k, n, bias, out, acc);
                } else {
                    gemm_nt_avx512(a, m, b, k, n, bias, out, acc);
                }
            }
            return;
        }
        if has_avx2_fma() {
            // The eight-lane path also has a masked tail, so widths below
            // eight are vectorised too; there is no scalar cut-off left.
            // SAFETY: as above.
            unsafe {
                if should_pack_b(m, k, n) {
                    // Packing makes the column-panel reads contiguous, so
                    // even the page-multiple widths that used to force the
                    // row-wise kernel can use the register-blocked one.
                    with_packed_b(b, k, n, |packed| {
                        gemm_nt_avx2_packed(a, m, b, packed, k, n, bias, out, acc)
                    });
                } else if n % PAGE_FLOATS == 0 {
                    gemm_nt_avx2_rows(a, m, b, k, n, bias, out, acc);
                } else {
                    gemm_nt_avx2(a, m, b, k, n, bias, out, acc);
                }
            }
            return;
        }
    }
    gemm_nt_impl(a, m, b, k, n, bias, out, acc);
}

/// The stride at which B's rows land in the same cache set.
///
/// The register-blocked kernel reads `B[k][j]` for one group of sixteen
/// columns and for every `k`, so the stride between its reads is the row
/// length of B. When that stride is a WHOLE NUMBER OF PAGES, every one of
/// those reads has the same offset inside its page -- and therefore the
/// same cache set. A column group's working set is one line per `k`, and
/// all of them fight for the same eight ways of that set: the ViT's
/// feed-forward width is 3072 columns, three pages exactly, and the
/// sixteen-lane kernel ran at a tenth of its speed there.
///
/// The row-at-a-time kernel reads a whole row of B at a time, contiguously,
/// and the prefetcher follows it. So for those widths the OLD kernel is
/// the faster one, and the dispatch picks it.
///
/// Both kernels accumulate over K in the same order with the same fused
/// multiply-add, so they produce the same bytes; only the speed differs.
const PAGE_FLOATS: usize = 4096 / 4;

/// The sixteen-lane mask for a tail of `1..=16` columns.
#[cfg(all(target_arch = "x86_64", has_stable_avx512))]
#[inline(always)]
fn avx512_tail_mask(tail: usize) -> u16 {
    debug_assert!((1..=16).contains(&tail));
    // A full sixteen-lane vector cannot be formed by shifting a u16 by 16:
    // that overflows in debug and wraps to zero in release, which would mask
    // EVERY lane out. The full width is simply "all sixteen bits set".
    if tail >= 16 {
        u16::MAX
    } else {
        (1u16 << tail) - 1
    }
}

/// The eight-lane mask for a tail of `1..=8` columns.
///
/// `_mm256_maskload_ps` / `_mm256_maskstore_ps` look at the SIGN BIT of each
/// lane index, so a compare that yields `-1` for a live lane is the mask.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_tail_mask(tail: usize) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    debug_assert!((1..=8).contains(&tail));
    _mm256_cmpgt_epi32(
        _mm256_set1_epi32(tail as i32),
        _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0),
    )
}

/// The GEMM, one row at a time, with the accumulator in memory.
///
/// This is the kernel the runtime used before the register-blocked one, and
/// it is still the right one when B's rows are a whole number of pages
/// apart: it reads a row of B at a time, contiguously, and the prefetcher
/// follows it. The register-blocked kernel reads sixteen columns at a time
/// across all of K, and at those strides every read lands in one cache
/// set.
///
/// The reduction is over K, in order, one column at a time, with an FMA per
/// step -- the same sequence, and so the same bytes.
#[cfg(all(target_arch = "x86_64", has_stable_avx512))]
#[target_feature(enable = "avx512f")]
unsafe fn gemm_nt_avx512_rows(
    a: &[f32],
    m: usize,
    b: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
    acc: &mut [f32],
) {
    use std::arch::x86_64::*;

    debug_assert_eq!(acc.len(), n);
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
            // The tail is MASKED, not scalar: a short row still runs the
            // sixteen-lane FMA. The masked lanes read zero and are never
            // stored, so the valid lanes get exactly the same reduction over
            // K, in the same order, as the full-vector loop above.
            if j < n {
                let mask = avx512_tail_mask(n - j);
                unsafe {
                    let bv = _mm512_maskz_loadu_ps(mask, brow.as_ptr().add(j));
                    let cv = _mm512_maskz_loadu_ps(mask, acc.as_ptr().add(j));
                    let sum = _mm512_fmadd_ps(bv, av, cv);
                    _mm512_mask_storeu_ps(acc.as_mut_ptr().add(j), mask, sum);
                }
            }
        }
        let orow = &mut out[r * n..r * n + n];
        for j in 0..n {
            orow[j] = acc[j] + bias[j];
        }
    }
}

/// The same kernel, eight lanes at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_nt_avx2_rows(
    a: &[f32],
    m: usize,
    b: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
    acc: &mut [f32],
) {
    use std::arch::x86_64::*;

    debug_assert_eq!(acc.len(), n);
    for r in 0..m {
        let arow = &a[r * k..r * k + k];
        acc.fill(0.0);
        for kk in 0..k {
            let av = _mm256_set1_ps(arow[kk]);
            let brow = &b[kk * n..kk * n + n];
            let mut j = 0usize;
            while j + 8 <= n {
                // SAFETY: as above.
                unsafe {
                    let bv = _mm256_loadu_ps(brow.as_ptr().add(j));
                    let cv = _mm256_loadu_ps(acc.as_ptr().add(j));
                    let sum = _mm256_fmadd_ps(bv, av, cv);
                    _mm256_storeu_ps(acc.as_mut_ptr().add(j), sum);
                }
                j += 8;
            }
            // A masked vector instead of a scalar loop on the last one to
            // seven columns. `_mm256_maskload_ps` keeps the invalid lanes out
            // of the FMA's way, and `_mm256_maskstore_ps` never touches them.
            if j < n {
                let mask = unsafe { avx2_tail_mask(n - j) };
                unsafe {
                    let bv = _mm256_maskload_ps(brow.as_ptr().add(j), mask);
                    let cv = _mm256_maskload_ps(acc.as_ptr().add(j), mask);
                    let sum = _mm256_fmadd_ps(bv, av, cv);
                    _mm256_maskstore_ps(acc.as_mut_ptr().add(j), mask, sum);
                }
            }
        }
        let orow = &mut out[r * n..r * n + n];
        for j in 0..n {
            orow[j] = acc[j] + bias[j];
        }
    }
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
            with_gemm_acc(n, |acc| gemm_nt_impl(a, m, b, k, n, bias, out, acc));
            true
        }
        Isa::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            {
                if !has_avx2_fma() {
                    return false;
                }
                // SAFETY: the host was checked for the features.
                with_gemm_acc(n, |acc| unsafe {
                    gemm_nt_avx2(a, m, b, k, n, bias, out, acc)
                });
                true
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                false
            }
        }
        Isa::Avx512 => {
            #[cfg(all(target_arch = "x86_64", has_stable_avx512))]
            {
                if !has_avx512f() {
                    return false;
                }
                // SAFETY: the host was checked for the features.
                with_gemm_acc(n, |acc| unsafe {
                    gemm_nt_avx512(a, m, b, k, n, bias, out, acc)
                });
                true
            }
            #[cfg(not(all(target_arch = "x86_64", has_stable_avx512)))]
            {
                false
            }
        }
    }
}

/// How many K steps ahead to prefetch B's next row.
///
/// A weight row is `n` floats from the previous one, so a column block walks
/// down memory one cache line per row across a large stride. Eight steps is
/// far enough ahead to hide that latency without running past the matrix.
const GEMM_PREFETCH_K: usize = 8;

/// The GEMM, compiled for a host with AVX2 and FMA.
///
/// `MR` output rows are in flight at once, each with its accumulator in a
/// register, and `W = 8` columns are computed per FMA. With `MR = 8` the
/// weight matrix is read once for eight output rows instead of four; the
/// smaller `MR = 4` variant is kept for the shapes that do not fill eight
/// rows. AVX2 has sixteen YMM registers, so eight accumulators plus the
/// broadcast and the weight vector fit without spilling.
///
/// The arithmetic is UNTOUCHED: for one output element the sequence is
/// `fma(b[k][j], a[r][k], acc)` for k in order, the same as the portable
/// kernel. Row blocking changes which rows share a weight load, never the
/// order of a row's reduction; the masked column tail changes which lanes
/// are live, never the FMA in a live lane. This is why the byte gate across
/// ISAs still passes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_nt_avx2_blocked<const MR: usize>(
    a: &[f32],
    m: usize,
    b: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
    acc: &mut [f32],
) {
    use std::arch::x86_64::*;

    /// The columns one vector holds.
    const W: usize = 8;

    debug_assert!((1..=8).contains(&MR));
    let full = m - m % MR;
    let mut r = 0usize;
    while r < full {
        let mut j = 0usize;
        while j < n {
            let tail = (n - j).min(W);
            // SAFETY: `j + W <= n` and `r + MR <= m`, so every live load is
            // inside `b` (k*n) or `bias` (n) and every live store is inside
            // `out` (m*n). A masked load/store touches only the live lanes.
            unsafe {
                let mask = if tail == W {
                    None
                } else {
                    Some(avx2_tail_mask(tail))
                };
                let mut c: [__m256; MR] = [_mm256_setzero_ps(); MR];
                for kk in 0..k {
                    if kk + GEMM_PREFETCH_K < k {
                        _mm_prefetch(
                            b.as_ptr().add((kk + GEMM_PREFETCH_K) * n + j).cast::<i8>(),
                            _MM_HINT_T1,
                        );
                    }
                    let bv = match mask {
                        None => _mm256_loadu_ps(b.as_ptr().add(kk * n + j)),
                        Some(mask) => _mm256_maskload_ps(b.as_ptr().add(kk * n + j), mask),
                    };
                    for i in 0..MR {
                        let av = _mm256_set1_ps(*a.get_unchecked((r + i) * k + kk));
                        c[i] = _mm256_fmadd_ps(bv, av, c[i]);
                    }
                }
                let bv = match mask {
                    None => _mm256_loadu_ps(bias.as_ptr().add(j)),
                    Some(mask) => _mm256_maskload_ps(bias.as_ptr().add(j), mask),
                };
                for i in 0..MR {
                    let sum = _mm256_add_ps(c[i], bv);
                    let dst = out.as_mut_ptr().add((r + i) * n + j);
                    match mask {
                        None => _mm256_storeu_ps(dst, sum),
                        Some(mask) => _mm256_maskstore_ps(dst, mask, sum),
                    }
                }
            }
            j += W;
        }
        r += MR;
    }
    // Rows that do not fill a block: the row-wise kernel, now with its own
    // masked column tail. A one-pixel convolution has exactly one such row.
    if full < m {
        unsafe {
            gemm_nt_avx2_rows(
                &a[full * k..],
                m - full,
                b,
                k,
                n,
                bias,
                &mut out[full * n..],
                acc,
            )
        };
    }
}

/// The AVX2 kernel at the reference's original row blocking.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn gemm_nt_avx2(
    a: &[f32],
    m: usize,
    b: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
    acc: &mut [f32],
) {
    // SAFETY: the host was checked for AVX2/FMA before dispatch.
    unsafe {
        if m >= 8 {
            gemm_nt_avx2_blocked::<8>(a, m, b, k, n, bias, out, acc)
        } else {
            gemm_nt_avx2_blocked::<4>(a, m, b, k, n, bias, out, acc)
        }
    }
}

/// The register-blocked kernel reading B from the packed panels.
///
/// Identical arithmetic to [`gemm_nt_avx2_blocked`]: same FMAs, same K order,
/// same bias addition. The only difference is the address the weight vector
/// is loaded from -- `packed[(panel*K + kk)*8]` instead of `b[kk*n + j]`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_nt_avx2_packed_blocked<const MR: usize>(
    a: &[f32],
    m: usize,
    b: &[f32],
    packed: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
    acc: &mut [f32],
) {
    use std::arch::x86_64::*;

    const W: usize = 8;

    debug_assert_eq!(packed.len(), packed_b_len(k, n));
    let panels = n.div_ceil(W);
    let full = m - m % MR;
    let mut r = 0usize;
    while r < full {
        let mut jb = 0usize;
        while jb < panels {
            let j = jb * W;
            let tail = (n - j).min(W);
            // SAFETY: every live lane is inside the caller's slices; a packed
            // panel always has all W lanes (the tail lanes were zeroed).
            unsafe {
                let mask = if tail == W {
                    None
                } else {
                    Some(avx2_tail_mask(tail))
                };
                let mut c: [__m256; MR] = [_mm256_setzero_ps(); MR];
                for kk in 0..k {
                    let bv = _mm256_loadu_ps(packed.as_ptr().add((jb * k + kk) * W));
                    for i in 0..MR {
                        let av = _mm256_set1_ps(*a.get_unchecked((r + i) * k + kk));
                        c[i] = _mm256_fmadd_ps(bv, av, c[i]);
                    }
                }
                let bv = match mask {
                    None => _mm256_loadu_ps(bias.as_ptr().add(j)),
                    Some(mask) => _mm256_maskload_ps(bias.as_ptr().add(j), mask),
                };
                for i in 0..MR {
                    let sum = _mm256_add_ps(c[i], bv);
                    let dst = out.as_mut_ptr().add((r + i) * n + j);
                    match mask {
                        None => _mm256_storeu_ps(dst, sum),
                        Some(mask) => _mm256_maskstore_ps(dst, mask, sum),
                    }
                }
            }
            jb += 1;
        }
        r += MR;
    }
    if full < m {
        // The row tail has too few rows to amortise a panel walk; the
        // row-wise kernel on the ORIGINAL B is the cheaper path.
        unsafe {
            gemm_nt_avx2_rows(
                &a[full * k..],
                m - full,
                b,
                k,
                n,
                bias,
                &mut out[full * n..],
                acc,
            )
        };
    }
}

/// The packed AVX2 kernel, with the row blocking chosen for this shape.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn gemm_nt_avx2_packed(
    a: &[f32],
    m: usize,
    b: &[f32],
    packed: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
    acc: &mut [f32],
) {
    // SAFETY: the host was checked for AVX2/FMA before dispatch.
    unsafe {
        if m >= 8 {
            gemm_nt_avx2_packed_blocked::<8>(a, m, b, packed, k, n, bias, out, acc)
        } else {
            gemm_nt_avx2_packed_blocked::<4>(a, m, b, packed, k, n, bias, out, acc)
        }
    }
}

/// The GEMM, compiled for a host with AVX-512.
///
/// Sixteen output columns at a time, and the SAME reduction over K: the
/// width of the vector is not the order of the additions, and the order is
/// the answer. This is why the 512-bit path is bit-identical to the 256-bit
/// one and the sixteen-lane LayerNorm would not have been.
#[cfg(all(target_arch = "x86_64", has_stable_avx512))]
#[target_feature(enable = "avx512f")]
unsafe fn gemm_nt_avx512_blocked<const MR: usize>(
    a: &[f32],
    m: usize,
    b: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
    acc: &mut [f32],
) {
    use std::arch::x86_64::*;

    /// The columns one vector holds.
    const W: usize = 16;

    debug_assert!((1..=8).contains(&MR));
    let full = m - m % MR;
    let mut r = 0usize;
    while r < full {
        let mut j = 0usize;
        while j < n {
            let tail = (n - j).min(W);
            // SAFETY: the live lanes are inside `b`, `bias` and `out`; the
            // masked lanes are neither read nor written.
            unsafe {
                let mask = avx512_tail_mask(tail);
                let full_mask = tail == W;
                let mut c: [__m512; MR] = [_mm512_setzero_ps(); MR];
                for kk in 0..k {
                    if kk + GEMM_PREFETCH_K < k {
                        _mm_prefetch(
                            b.as_ptr().add((kk + GEMM_PREFETCH_K) * n + j).cast::<i8>(),
                            _MM_HINT_T1,
                        );
                    }
                    let bptr = b.as_ptr().add(kk * n + j);
                    let bv = if full_mask {
                        _mm512_loadu_ps(bptr)
                    } else {
                        _mm512_maskz_loadu_ps(mask, bptr)
                    };
                    for i in 0..MR {
                        let av = _mm512_set1_ps(*a.get_unchecked((r + i) * k + kk));
                        c[i] = _mm512_fmadd_ps(bv, av, c[i]);
                    }
                }
                let bptr = bias.as_ptr().add(j);
                let bv = if full_mask {
                    _mm512_loadu_ps(bptr)
                } else {
                    _mm512_maskz_loadu_ps(mask, bptr)
                };
                for i in 0..MR {
                    let sum = _mm512_add_ps(c[i], bv);
                    let dst = out.as_mut_ptr().add((r + i) * n + j);
                    if full_mask {
                        _mm512_storeu_ps(dst, sum);
                    } else {
                        _mm512_mask_storeu_ps(dst, mask, sum);
                    }
                }
            }
            j += W;
        }
        r += MR;
    }
    if full < m {
        unsafe {
            gemm_nt_avx512_rows(
                &a[full * k..],
                m - full,
                b,
                k,
                n,
                bias,
                &mut out[full * n..],
                acc,
            )
        };
    }
}

/// The AVX-512 kernel at the reference's original row blocking.
#[cfg(all(target_arch = "x86_64", has_stable_avx512))]
#[inline(always)]
unsafe fn gemm_nt_avx512(
    a: &[f32],
    m: usize,
    b: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
    out: &mut [f32],
    acc: &mut [f32],
) {
    // SAFETY: the host was checked for AVX-512F before dispatch.
    unsafe {
        if m >= 8 {
            gemm_nt_avx512_blocked::<8>(a, m, b, k, n, bias, out, acc)
        } else {
            gemm_nt_avx512_blocked::<4>(a, m, b, k, n, bias, out, acc)
        }
    }
}

// ---------------------------------------------------------- elementwise helpers

/// `dst[i] += src[i]`, vectorised when the host has AVX2.
///
/// This crosses no reduction: each output element depends only on the two
/// elements at the same index, so every path computes the same sum. The BERT
/// residual adds and the additive attention mask used to be scalar loops over
/// `rows * hidden` on every layer of every inference; they now go through one
/// place that can use the full width of the machine.
#[inline]
pub fn add_inplace(dst: &mut [f32], src: &[f32]) {
    let n = dst.len().min(src.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: the host has the feature; the helper touches only indices
        // below `n`, which is within both slices.
        unsafe { add_inplace_avx2(&mut dst[..n], &src[..n]) };
        return;
    }
    for (d, &s) in dst[..n].iter_mut().zip(src.iter()) {
        *d += s;
    }
}

/// Eight lanes at a time, with a scalar tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_inplace_avx2(dst: &mut [f32], src: &[f32]) {
    use std::arch::x86_64::*;

    let n = dst.len().min(src.len());
    let mut i = 0usize;
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` and both slices are at least `n` long.
        unsafe {
            let d = _mm256_loadu_ps(dst.as_ptr().add(i));
            let s = _mm256_loadu_ps(src.as_ptr().add(i));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_add_ps(d, s));
        }
        i += 8;
    }
    while i < n {
        // SAFETY: `i < n`, inside both slices.
        unsafe {
            *dst.get_unchecked_mut(i) += *src.get_unchecked(i);
        }
        i += 1;
    }
}

// ------------------------------------------------------ reduction helpers

/// The reference's eight-lane dot product, bit-exact on every path.
///
/// The first eight products are plain multiplies, each subsequent group of
/// eight is one FMA per lane, and the lanes are folded with [`hsum8`]. That is
/// the order the attention score and the vocabulary projection use; changing
/// it changes the last bit of the goldens. Splitting a dot product into eight
/// vector lanes keeps the per-lane order intact, so the AVX2 path computes the
/// same number as the scalar one, eight elements at a time.
#[inline]
pub fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let nv = n / 8;
    if nv == 0 {
        return 0.0;
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: the host has AVX2/FMA, and `nv = n / 8` groups of eight are
        // all inside both slices.
        let lanes = unsafe { dot8_lanes_avx2(a, b, nv) };
        let mut acc = hsum8(lanes);
        let mut j = nv * 8;
        while j < n {
            acc += a[j] * b[j];
            j += 1;
        }
        return acc;
    }
    let mut lanes = [0.0f32; 8];
    for l in 0..8 {
        lanes[l] = a[l] * b[l];
    }
    for v in 1..nv {
        for l in 0..8 {
            lanes[l] = a[v * 8 + l].mul_add(b[v * 8 + l], lanes[l]);
        }
    }
    let mut acc = hsum8(lanes);
    let mut j = nv * 8;
    while j < n {
        acc += a[j] * b[j];
        j += 1;
    }
    acc
}

/// The vector lanes of [`dot8`], before the horizontal sum.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot8_lanes_avx2(a: &[f32], b: &[f32], nv: usize) -> [f32; 8] {
    use std::arch::x86_64::*;

    // SAFETY: `nv` groups of eight were bounds-checked by the caller.
    unsafe {
        let mut acc = _mm256_mul_ps(_mm256_loadu_ps(a.as_ptr()), _mm256_loadu_ps(b.as_ptr()));
        for v in 1..nv {
            acc = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(v * 8)),
                _mm256_loadu_ps(b.as_ptr().add(v * 8)),
                acc,
            );
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        lanes
    }
}

/// `acc[i] = weight * v[i] + acc[i]`, eight lanes at a time when possible.
///
/// The softmax-weighted value accumulation in attention. Each element is
/// independent, so the vector path computes the same FMA as the scalar loop.
#[inline]
pub fn weighted_accumulate(acc: &mut [f32], v: &[f32], weight: f32) {
    let n = acc.len().min(v.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: the host has AVX2/FMA; the helper touches only `..n`.
        unsafe { weighted_accumulate_avx2(&mut acc[..n], &v[..n], weight) };
        return;
    }
    for (d, &x) in acc[..n].iter_mut().zip(v.iter()) {
        *d = weight.mul_add(x, *d);
    }
}

/// Eight lanes at a time, with a scalar tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn weighted_accumulate_avx2(acc: &mut [f32], v: &[f32], weight: f32) {
    use std::arch::x86_64::*;

    let n = acc.len().min(v.len());
    // SAFETY: `n` is within both slices and the loop bounds keep every load
    // and store in range.
    unsafe {
        let w = _mm256_set1_ps(weight);
        let mut i = 0usize;
        while i + 8 <= n {
            let a = _mm256_loadu_ps(acc.as_ptr().add(i));
            let x = _mm256_loadu_ps(v.as_ptr().add(i));
            _mm256_storeu_ps(acc.as_mut_ptr().add(i), _mm256_fmadd_ps(x, w, a));
            i += 8;
        }
        while i < n {
            *acc.get_unchecked_mut(i) = weight.mul_add(*v.get_unchecked(i), *acc.get_unchecked(i));
            i += 1;
        }
    }
}

// ---------------------------------------------------------- fused attention

thread_local! {
    /// One attention score row per worker.
    ///
    /// The fused kernel normalises a row in place instead of materialising a
    /// `batch * heads * seq * seq` score matrix. A row is `seq` floats -- for
    /// a long sequence the difference between that and the matrix is the
    /// difference between kilobytes and tens of megabytes.
    static ATTN_ROW: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// The fused host attention: QK^T, the softmax and softmax.V in one pass over
/// one query row at a time.
///
/// WHY THE BYTES ARE THE SAME
/// --------------------------
/// Every arithmetic operation is the one the separate kernels use, in the
/// same order: `dot8` for each score, the additive mask folded in with the
/// same `+`, `softmax_row_impl` (the exact body behind `softmax_row_on` on
/// every ISA) for the row, and `weighted_accumulate` over the keys from first
/// to last. The bytes are therefore bit-identical to `qk` + `softmax_rows` +
/// `av`; only the memory traffic changes.
///
/// WHY IT IS FASTER
/// ----------------
/// The three passes keep a full `batch * heads * seq * seq` score matrix
/// alive between them. This kernel keeps ONE row: it writes that row's
/// scores, normalises and consumes them, then overwrites it for the next
/// query. For a long sequence that matrix is the largest temporary in the
/// forward pass. Rows are independent (a query reads only K and V), so the
/// work splits over the same row chunks as the separate kernels, with the
/// same byte result.
///
/// `qkv` is the model's `[3 * hidden]`-per-row layout (Q, K, V concatenated),
/// `ctx` is a row-major `batch * seq` by `hidden` output, and `add_mask` has
/// one entry per token.
pub fn fused_attention(
    qkv: &[f32],
    ctx: &mut [f32],
    add_mask: &[f32],
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    hidden: usize,
) -> Result<()> {
    if seq % 8 != 0 {
        return Err(TensorError::WrongLength {
            expected: seq.next_multiple_of(8),
            found: seq,
        }
        .into());
    }
    let rows = batch * seq;
    let span = rows * hidden;
    if ctx.len() < span {
        return Err(TensorError::WrongLength {
            expected: span,
            found: ctx.len(),
        }
        .into());
    }
    if add_mask.len() < rows {
        return Err(TensorError::WrongLength {
            expected: rows,
            found: add_mask.len(),
        }
        .into());
    }
    let qkv_span = rows * 3 * hidden;
    if qkv.len() < qkv_span || hidden != heads * head_dim {
        return Err(TensorError::WrongLength {
            expected: qkv_span,
            found: qkv.len(),
        }
        .into());
    }

    let one_row = |row: usize, out: &mut [f32]| {
        let b = row / seq;
        let i = row % seq;
        ATTN_ROW.with(|cell| {
            let mut scratch = cell.borrow_mut();
            if scratch.len() < seq {
                scratch.resize(seq, 0.0);
            }
            for h in 0..heads {
                let q_base = (b * seq + i) * 3 * hidden + h * head_dim;
                let q = &qkv[q_base..q_base + head_dim];
                let scores = &mut scratch[..seq];
                // QK^T, with the mask added where the separate QK pass adds it.
                for (j, slot) in scores.iter_mut().enumerate() {
                    let k_base = (b * seq + j) * 3 * hidden + hidden + h * head_dim;
                    *slot = dot8(q, &qkv[k_base..k_base + head_dim]) + add_mask[b * seq + j];
                }
                softmax_row_impl(scores);
                let o = &mut out[h * head_dim..][..head_dim];
                o.fill(0.0);
                for (j, &weight) in scores.iter().enumerate() {
                    let v_base = (b * seq + j) * 3 * hidden + 2 * hidden + h * head_dim;
                    weighted_accumulate(o, &qkv[v_base..v_base + head_dim], weight);
                }
            }
        });
    };

    let work = (rows as i64) * (heads as i64) * (seq as i64) * (head_dim as i64) * 2;
    match parallel::chunk_rows(rows, work, parallel::threads()) {
        Some(chunk) => {
            parallel::for_each_chunk_mut(&mut ctx[..span], chunk * hidden, |ci, part| {
                let first = ci * chunk;
                for (i, row) in part.chunks_exact_mut(hidden).enumerate() {
                    one_row(first + i, row);
                }
            });
        }
        None => {
            for (i, row) in ctx[..span].chunks_exact_mut(hidden).enumerate() {
                one_row(i, row);
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- the trait

/// Allocate `len` floats for a buffer the caller will OVERWRITE COMPLETELY.
///
/// `vec![0.0f32; len]` pays for a zero pass on every allocation. The layers'
/// scratch buffers are overwritten in full by a GEMM, a score row or an
/// activation before anything reads them, so that pass is pure overhead.
/// `f32` has no drop glue and every bit pattern is valid, so `set_len` can
/// publish reserved capacity without initialising it.
///
/// # Safety contract
///
/// The caller MUST write every element before reading any of them. Reading
/// an unwritten element is undefined behaviour, not a zero. Buffers whose
/// kernel ACCUMULATES into them (a bias add, a convtranspose's tap sum) must
/// keep using `vec![0.0; len]`.
#[inline]
pub fn scratch_f32(len: usize) -> Vec<f32> {
    let mut buf = Vec::with_capacity(len);
    // SAFETY: `with_capacity(len)` reserved room for `len` f32s; f32 has no
    // invalid bit patterns and no drop glue. The caller's contract above
    // guarantees every published element is written before it is read.
    unsafe { buf.set_len(len) };
    buf
}

/// The GELU, on a named ISA.
///
/// The polynomial is contracted with FMA on the vector path and not on the
/// scalar tail, exactly as the reference's is, and exactly as the reference's AVX2 and
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
///
/// Eight lanes at a time, and the SAME polynomial in the SAME FMA order as
/// [`gelu_f32`]: `pl = C0`, then `pl = fma(pl, u, C1)` ... `C8`, then
/// `max(v, 0) + pl`. The earlier body had the target feature but none of the
/// intrinsics -- it called the scalar polynomial eight times in a loop, so it
/// paid for the wide build and got none of the width. This one is genuinely
/// eight-wide, and bit-identical to the scalar path for every finite input.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_row_avx2(x: &mut [f32]) {
    use std::arch::x86_64::*;

    const C0: f32 = -7.234_028_217_1e-05;
    const C1: f32 = 1.817_951_800_5e-03;
    const C2: f32 = -1.770_738_337_9e-02;
    const C3: f32 = 8.457_714_764_1e-02;
    const C4: f32 = -1.922_867_183_4e-01;
    const C5: f32 = 9.843_112_445_8e-02;
    const C6: f32 = 3.613_785_206_2e-01;
    const C7: f32 = -4.945_412_893_6e-01;
    const C8: f32 = -1.300_701_011_7e-04;

    let n = x.len();
    // SAFETY: the caller checked the host for AVX2/FMA. All loads/stores below
    // are within `x` because of the loop bounds.
    unsafe {
        let c0 = _mm256_set1_ps(C0);
        let c1 = _mm256_set1_ps(C1);
        let c2 = _mm256_set1_ps(C2);
        let c3 = _mm256_set1_ps(C3);
        let c4 = _mm256_set1_ps(C4);
        let c5 = _mm256_set1_ps(C5);
        let c6 = _mm256_set1_ps(C6);
        let c7 = _mm256_set1_ps(C7);
        let c8 = _mm256_set1_ps(C8);
        let zero = _mm256_setzero_ps();
        let four = _mm256_set1_ps(4.0);
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));

        let mut i = 0usize;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(x.as_ptr().add(i));
            let u = _mm256_min_ps(_mm256_and_ps(v, abs_mask), four);
            let mut pl = c0;
            pl = _mm256_fmadd_ps(pl, u, c1);
            pl = _mm256_fmadd_ps(pl, u, c2);
            pl = _mm256_fmadd_ps(pl, u, c3);
            pl = _mm256_fmadd_ps(pl, u, c4);
            pl = _mm256_fmadd_ps(pl, u, c5);
            pl = _mm256_fmadd_ps(pl, u, c6);
            pl = _mm256_fmadd_ps(pl, u, c7);
            pl = _mm256_fmadd_ps(pl, u, c8);
            let r = _mm256_add_ps(_mm256_max_ps(v, zero), pl);
            _mm256_storeu_ps(x.as_mut_ptr().add(i), r);
            i += 8;
        }
        while i < n {
            *x.get_unchecked_mut(i) = gelu_f32(*x.get_unchecked(i));
            i += 1;
        }
    }
}

/// GELU, in place.
pub fn gelu_(x: &mut Tensor) -> Result<()> {
    let mut values = x.to_f32();
    for v in values.iter_mut() {
        *v = gelu_f32(*v);
    }
    copy_into(x, &values);
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
    copy_into(x, &values);
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
    copy_into(x, &values);
    Ok(())
}

/// `c = a * b`, row-major, optionally accumulating into `c`.
///
/// `a` is `[m, k]`, `b` is `[k, n]`, `c` is `[m, n]`. The accumulation is
/// `c += a * b` when `acc` is `Some`, and `c = a * b` when it is `None`.
///
/// This is the TRAIT's generic multiply, and it is the exact one: the same
/// FMA accumulation in the same order as `gemm_nt`.
pub fn gemm(a: &Tensor, b: &Tensor, c: &mut Tensor, acc: Option<&mut Tensor>) -> Result<()> {
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
    if c.len() != m * n {
        return Err(TensorError::WrongLength {
            expected: m * n,
            found: c.len(),
        }
        .into());
    }
    let av = a.to_f32();
    let bv = b.to_f32();
    let mut cv = vec![0.0f32; m * n];
    with_zero_bias(n, |zeros| gemm_nt(&av, m, &bv, k, n, zeros, &mut cv));
    // AFTER the multiply, and not before it: `gemm_nt` WRITES its output
    // (`out = a * b + bias`), it does not accumulate into it, so adding the
    // accumulator first would have had it overwritten and `acc` silently
    // ignored. Nothing in the tree passed `Some` until the GPU backend's gate
    // did, which is exactly how a dead branch survives a gate.
    if let Some(acc) = acc {
        let av_acc = acc.to_f32();
        for (i, v) in cv.iter_mut().enumerate() {
            *v += av_acc.get(i).copied().unwrap_or(0.0);
        }
    }
    copy_into(c, &cv);
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
    copy_into(out, &result);
    Ok(())
}

/// Pad an optional bias tensor to exactly `cout` entries.
///
/// `None` returns `None` so the caller can use the shared zero-bias buffer;
/// a short tensor is padded with zeros, as the old direct loops did with
/// `get(co).unwrap_or(0.0)`.
fn conv_bias(b: Option<&Tensor>, cout: usize) -> Option<Vec<f32>> {
    let values = b?.to_f32();
    let mut out = vec![0.0f32; cout];
    let n = values.len().min(cout);
    out[..n].copy_from_slice(&values[..n]);
    Some(out)
}

/// The shared convolution GEMM: `columns` is `[rows, k]`, `weights` is
/// `[k, cout]`, and the result is written row-major as `[rows, cout]`.
///
/// The im2col gather below is the only place that knows the image layout;
/// the arithmetic is the same reference GEMM every other path uses, with the
/// same reduction order over the kernel's flattened `(channel, ky, kx)`
/// index, and with the same row-split policy.
#[inline]
fn conv_gemm(
    columns: &[f32],
    rows: usize,
    weights: &[f32],
    k: usize,
    cout: usize,
    bias: &[f32],
    out: &mut [f32],
) {
    gemm_nt_rows(
        columns,
        rows,
        weights,
        k,
        cout,
        bias,
        out,
        parallel::threads(),
    );
}

/// A one-dimensional convolution.
///
/// `x` is `[n, cin, l]`, `w` is `[cout, cin, k]`, and the output is
/// `[n, cout, l_out]`.
///
/// The old direct loops did a gather and a dot product for every output
/// channel and position, re-reading the same input patches for each filter.
/// The gather is now done once into an im2col matrix and the dot products are
/// one GEMM, so the cost of the gather is paid once per convolution and the
/// arithmetic runs through the reference SIMD kernels.
pub fn conv1d(x: &mut Tensor, w: &Tensor, b: Option<&Tensor>, stride: i64, pad: i64) -> Result<()> {
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
    let kdim = cin * k;
    let rows = n * l_out;

    let mut columns = vec![0.0f32; rows * kdim];
    for bi in 0..n {
        for oi in 0..l_out {
            let base = (oi as i64) * stride - pad;
            let row = &mut columns[(bi * l_out + oi) * kdim..][..kdim];
            for ci in 0..cin {
                for ki in 0..k {
                    let pos = base + ki as i64;
                    if (0..l as i64).contains(&pos) {
                        row[ci * k + ki] = xv[(bi * cin + ci) * l + pos as usize];
                    }
                }
            }
        }
    }
    // The weight is stored `[cout, cin, k]`; the GEMM wants `[cin*k, cout]`.
    // Flattening the filter's `(channel, tap)` index keeps the reduction in
    // exactly the order the direct loops used.
    let mut weights = vec![0.0f32; kdim * cout];
    for co in 0..cout {
        for ci in 0..cin {
            for ki in 0..k {
                weights[(ci * k + ki) * cout + co] = wv[(co * cin + ci) * k + ki];
            }
        }
    }

    let bias_owned = conv_bias(b, cout);
    let mut gemm_out = vec![0.0f32; rows * cout];
    with_zero_bias(cout, |zeros| {
        conv_gemm(
            &columns,
            rows,
            &weights,
            kdim,
            cout,
            bias_owned.as_deref().unwrap_or(zeros),
            &mut gemm_out,
        );
    });

    let mut out = vec![0.0f32; n * cout * l_out];
    for bi in 0..n {
        for oi in 0..l_out {
            let src = &gemm_out[(bi * l_out + oi) * cout..][..cout];
            for co in 0..cout {
                out[(bi * cout + co) * l_out + oi] = src[co];
            }
        }
    }
    copy_into(x, &out);
    Ok(())
}

/// A two-dimensional convolution.
///
/// `x` is `[n, cin, h, w]`, `w` is `[cout, cin, kh, kw]`.
/// The layout is the same im2col + GEMM as [`conv1d`], with the flattened
/// kernel index `(channel, ky, kx)`.
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
    let kdim = cin * kh * kw;
    let plane_out = h_out * w_out;
    let rows = n * plane_out;

    let mut columns = vec![0.0f32; rows * kdim];
    for bi in 0..n {
        for oi in 0..h_out {
            let base_h = (oi as i64) * stride.0 - pad.0;
            for oj in 0..w_out {
                let base_w = (oj as i64) * stride.1 - pad.1;
                let row = &mut columns[(bi * plane_out + oi * w_out + oj) * kdim..][..kdim];
                for ci in 0..cin {
                    for ki in 0..kh {
                        let ih = base_h + ki as i64;
                        if !(0..h as i64).contains(&ih) {
                            continue;
                        }
                        for kwi in 0..kw {
                            let iw = base_w + kwi as i64;
                            if (0..width as i64).contains(&iw) {
                                row[(ci * kh + ki) * kw + kwi] =
                                    xv[((bi * cin + ci) * h + ih as usize) * width + iw as usize];
                            }
                        }
                    }
                }
            }
        }
    }
    let mut weights = vec![0.0f32; kdim * cout];
    for co in 0..cout {
        for ci in 0..cin {
            for ki in 0..kh {
                for kwi in 0..kw {
                    weights[((ci * kh + ki) * kw + kwi) * cout + co] =
                        wv[((co * cin + ci) * kh + ki) * kw + kwi];
                }
            }
        }
    }

    let bias_owned = conv_bias(b, cout);
    let mut gemm_out = vec![0.0f32; rows * cout];
    with_zero_bias(cout, |zeros| {
        conv_gemm(
            &columns,
            rows,
            &weights,
            kdim,
            cout,
            bias_owned.as_deref().unwrap_or(zeros),
            &mut gemm_out,
        );
    });

    let mut out = vec![0.0f32; n * cout * plane_out];
    for bi in 0..n {
        for oi in 0..h_out {
            for oj in 0..w_out {
                let src = &gemm_out[(bi * plane_out + oi * w_out + oj) * cout..][..cout];
                for co in 0..cout {
                    out[((bi * cout + co) * h_out + oi) * w_out + oj] = src[co];
                }
            }
        }
    }
    copy_into(x, &out);
    Ok(())
}

/// The multimodal projection, into `out`.
///
/// The backend trait hands this operation only the input and the output: the
/// projection's weights are not an argument, and no family keeps them behind
/// this method. What the operation owes the graph is therefore the input's
/// values in `out` (the same identity projection `logit` performs) -- NOT a
/// copy of `out` into itself. The old body read `out`, ignored `x`, and wrote
/// those same bytes back, so a caller that moved a tensor through `mproj`
/// silently lost its input. That was a correctness bug, not a slow path.
pub fn mproj(x: &Tensor, out: &mut Tensor) -> Result<()> {
    let xv = x.to_f32();
    let mut ov = out.to_f32();
    for (i, slot) in ov.iter_mut().enumerate() {
        *slot = xv.get(i).copied().unwrap_or(0.0);
    }
    copy_into(out, &ov);
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
    copy_into(out, &result);
    Ok(())
}

/// The logit projection, into `out`.
pub fn logit(x: &Tensor, out: &mut Tensor) -> Result<()> {
    let xv = x.to_f32();
    let mut ov = out.to_f32();
    for (i, v) in ov.iter_mut().enumerate() {
        *v = xv.get(i).copied().unwrap_or(0.0);
    }
    copy_into(out, &ov);
    Ok(())
}

/// Write f32 values into a tensor's byte storage.
///
/// The tensor's length is fixed by its shape, and the values were produced
/// for that same length -- so a size mismatch is a bug in the CALLER, and it
/// is caught here rather than silently truncating.
fn copy_into(tensor: &mut Tensor, values: &[f32]) {
    let target = tensor.bytes_mut();
    let bytes = values.len() * std::mem::size_of::<f32>();
    assert_eq!(
        target.len(),
        bytes,
        "the tensor holds {} bytes and the values hold {}",
        target.len(),
        bytes
    );

    #[cfg(target_endian = "little")]
    {
        // `to_le_bytes()` on a little-endian host is the identity map on the
        // f32's four bytes. f32 has no padding bits and no drop glue, so the
        // slice is just `4 * len` readable bytes: one bulk copy instead of a
        // per-element `to_le_bytes()` and a 4-byte store.
        //
        // SAFETY: the byte length was checked above, `f32` is four bytes with
        // no padding, and the source slice outlives the copy.
        unsafe {
            let src = std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), bytes);
            target.copy_from_slice(src);
        }
    }

    #[cfg(target_endian = "big")]
    {
        for (dst, &value) in target.chunks_exact_mut(4).zip(values) {
            dst.copy_from_slice(&value.to_le_bytes());
        }
    }
}
