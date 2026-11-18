//! The bf16 conversions, in blocks.
//!
//! # WHY A BLOCK API
//!
//! The array's operands are bf16 and the host's are f32, and the
//! conversion between them IS the boundary: every activation crosses it
//! once on its way in, and every output crosses it back. At the array's
//! volume that is hundreds of millions of values, and a scalar conversion
//! is one value at a time.
//!
//! So the conversion comes in blocks, and the blocks are SIMD on x86.
//!
//! # THE ROUNDING IS THE SAME ONE
//!
//! The vector path computes the SAME bits as `f32_to_bf16`, and the check
//! is in the tests: the two are compared over random values and over the
//! values where they are most likely to differ -- the exact ties, the
//! subnormals, the infinities and the NaNs. A vector path that rounded
//! differently would move every weight by a bit, and the container's
//! digest would still match: the failure this project keeps meeting is not
//! a crash, it is a plausible number.
//!
//! # THE BYTES ARE LITTLE-ENDIAN, AND THAT IS THE FORMAT'S
//!
//! The container and the device both spell bf16 little-endian, so the
//! block API speaks bytes. Writing the values as bytes rather than as
//! `u16`s is what lets the caller hand the result straight to the
//! transport, with no second copy and no unsafe reinterpretation.

/// A block of f32 values, packed into bf16, little-endian.
///
/// `dst` holds two bytes per value; the values past `src`'s length are
/// left alone.
pub fn f32_to_bf16_bytes(src: &[f32], dst: &mut [u8]) {
    let n = src.len().min(dst.len() / 2);
    let mut i = 0usize;

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the feature was just detected, and the function below
            // is compiled for it.
            unsafe { f32_to_bf16_avx2(src, dst, n) };
            i = n & !7;
        }
    }

    // THE SCALAR, for what is left and for every other machine.
    for (index, value) in src[i..n].iter().enumerate() {
        let bits = crate::f32_to_bf16(*value).to_le_bytes();
        let at = (i + index) * 2;
        dst[at] = bits[0];
        dst[at + 1] = bits[1];
    }
}

/// A row of f32 values, packed into bf16, followed by a zeroed padding tail.
///
/// This is the shape the array's A operand has: a real row of `src.len()`
/// values, then `pad_values` more columns whose bf16 bytes are zero. The
/// caller used to write those zeros in a separate pass before the conversion;
/// doing both here means each design row is VISITED ONCE rather than twice,
/// and the bytes are the same -- the real values go through
/// [`f32_to_bf16_bytes`] unchanged and the tail is zero.
///
/// `dst` may be longer than the row; the bytes past it are left alone. A
/// shorter `dst` still converts as many values as fit and zeroes the rest of
/// what fits, matching [`f32_to_bf16_bytes`]'s short-destination rule.
pub fn f32_to_bf16_bytes_padded(src: &[f32], dst: &mut [u8], pad_values: usize) {
    let real = src.len() * 2;
    let end = real + pad_values * 2;
    let fit = end.min(dst.len());
    let dst = &mut dst[..fit];
    let converted = real.min(fit);
    f32_to_bf16_bytes(src, &mut dst[..converted]);
    if fit > converted {
        dst[converted..].fill(0);
    }
}

/// A block of bf16 bytes, widened into f32.
///
/// `src` holds two bytes per value, little-endian; the values past `dst`'s
/// length are not read.
pub fn bf16_to_f32_bytes(src: &[u8], dst: &mut [f32]) {
    let n = dst.len().min(src.len() / 2);
    let mut i = 0usize;

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the feature was just detected, and the function below
            // is compiled for it.
            unsafe { bf16_to_f32_avx2(src, dst, n) };
            i = n & !7;
        }
    }

    // THE SCALAR, for what is left and for every other machine.
    for index in i..n {
        let at = index * 2;
        let bits = u16::from_le_bytes([src[at], src[at + 1]]);
        dst[index] = crate::bf16_to_f32(bits);
    }
}

/// A block of bf16 bytes, widened into f32 and added to a bias vector.
///
/// This is the read-back half of a biased dispatch fused into ONE pass. A plain
/// [`bf16_to_f32_bytes`] followed by a bias add decodes into a scratch row and
/// then reads that row again for the add -- two visits to every output value,
/// plus a buffer the size of a row. At the array's volume (hundreds of millions
/// of values per frame on the vision families) the second visit is the sort of
/// overhead that shows up in a measurement as "host shuffling". Here the widening
/// writes straight into its final place with the bias already added.
///
/// `dst[i] = bf16_to_f32(src[2i..2i+2]) + bias[i]`, with a missing bias entry
/// contributing zero. That matches the padded-bias convention the dispatch uses:
/// the convolution's own bias is followed by zeros out to the design's width.
pub fn bf16_to_f32_add_bias_bytes(src: &[u8], dst: &mut [f32], bias: &[f32]) {
    let n = dst.len().min(src.len() / 2);
    let mut i = 0usize;

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && bias.len() >= n {
            // SAFETY: the feature was just detected, and the function below
            // reads 16 bytes and 8 floats and writes 8 floats only while all
            // eight values are inside the slices it was given.
            unsafe { bf16_to_f32_add_bias_avx2(src, dst, bias, n) };
            i = n & !7;
        }
    }

    // THE SCALAR, for what is left and for every other machine.
    for index in i..n {
        let at = index * 2;
        let bits = u16::from_le_bytes([src[at], src[at + 1]]);
        dst[index] = crate::bf16_to_f32(bits) + bias.get(index).copied().unwrap_or(0.0);
    }
}

/// The AVX2 path: eight bf16 widened, biased and written in one visit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bf16_to_f32_add_bias_avx2(src: &[u8], dst: &mut [f32], bias: &[f32], n: usize) {
    use std::arch::x86_64::*;

    // SAFETY: the caller checked the feature and that `bias` holds at least
    // `n` entries. Every pointer below stays inside its slice: the loop reads
    // 16 source bytes and 8 bias floats and writes 8 floats only while eight
    // values remain in all three.
    unsafe {
        let mut index = 0usize;
        while index + 8 <= n {
            let packed = _mm_loadu_si128(src.as_ptr().add(index * 2) as *const __m128i);
            let widened = _mm256_cvtepu16_epi32(packed);
            let shifted = _mm256_slli_epi32(widened, 16);
            let values = _mm256_castsi256_ps(shifted);
            let bias_values = _mm256_loadu_ps(bias.as_ptr().add(index));
            _mm256_storeu_ps(
                dst.as_mut_ptr().add(index),
                _mm256_add_ps(values, bias_values),
            );
            index += 8;
        }
    }
}

/// The AVX2 path: eight f32 in, eight bf16 out.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn f32_to_bf16_avx2(src: &[f32], dst: &mut [u8], n: usize) {
    use std::arch::x86_64::*;

    // SAFETY: the caller checked the feature, and every pointer below is
    // inside the slice it was given: the loop reads 8 values and writes 16
    // bytes only while 8 values are left in both.
    unsafe {
        let rounding = _mm256_set1_epi32(0x7fff);
        let one = _mm256_set1_epi32(1);
        let mut index = 0usize;
        while index + 8 <= n {
            let values = _mm256_loadu_ps(src.as_ptr().add(index));
            let bits = _mm256_castps_si256(values);
            // Round to nearest even: add 0x7FFF plus the low bit of the
            // mantissa, then truncate. The same three steps as the scalar's,
            // eight lanes at a time.
            let lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), one);
            let bias = _mm256_add_epi32(rounding, lsb);
            let rounded = _mm256_add_epi32(bits, bias);
            let high = _mm256_srli_epi32(rounded, 16);

            // THE PACK, ACROSS THE LANES.
            //
            // `_mm256_packus_epi32` packs WITHIN each 128-bit lane, so the low
            // four values would land in the low lane and the high four in the
            // high one -- and the two 128-bit halves are then packed into the
            // eight u16 the caller asked for. The SSE2 pack takes one operand
            // from each, which is the order the caller wants.
            let low = _mm256_castsi256_si128(high);
            let high_half = _mm256_extracti128_si256(high, 1);
            let packed = _mm_packus_epi32(low, high_half);
            _mm_storeu_si128(dst.as_mut_ptr().add(index * 2) as *mut __m128i, packed);

            index += 8;
        }
    }
}

/// The AVX2 path: eight bf16 in, eight f32 out.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bf16_to_f32_avx2(src: &[u8], dst: &mut [f32], n: usize) {
    use std::arch::x86_64::*;

    // SAFETY: the caller checked the feature, and every pointer below is
    // inside the slice it was given: the loop reads 16 bytes and writes 8
    // floats only while 8 values are left in both.
    unsafe {
        let mut index = 0usize;
        while index + 8 <= n {
            let packed = _mm_loadu_si128(src.as_ptr().add(index * 2) as *const __m128i);
            // The widening is a zero-extend and a shift left by sixteen: a bf16
            // is the high half of an f32.
            let widened = _mm256_cvtepu16_epi32(packed);
            let shifted = _mm256_slli_epi32(widened, 16);
            _mm256_storeu_ps(dst.as_mut_ptr().add(index), _mm256_castsi256_ps(shifted));

            index += 8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic pseudo-random f32 in [-1, 1).
    fn draw(state: &mut u64) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        let unit = ((*state >> 40) as f32) / (1u32 << 24) as f32;
        unit * 2.0 - 1.0
    }

    #[test]
    fn the_block_conversion_is_the_scalar_one() {
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut values: Vec<f32> = (0..1000).map(|_| draw(&mut state)).collect();
        // THE VALUES WHERE THE TWO ROUNDINGS ARE MOST LIKELY TO DIFFER.
        //
        // The exact ties -- a mantissa's low bit set and the rest zero --
        // are the ones round-to-nearest-even and round-away-from-zero
        // disagree about, and every one of them is a value a trained
        // network can hold.
        for exponent in 0..=254u32 {
            let base = (exponent << 23) | 0x0080_0000;
            values.push(f32::from_bits(base));
            values.push(f32::from_bits(base | 0x0000_0001));
            values.push(f32::from_bits(base | 0x007f_ffff));
            values.push(f32::from_bits(base | 0x0080_0001));
        }
        values.push(f32::from_bits(0x7f80_0000)); // +inf
        values.push(f32::from_bits(0xff80_0000)); // -inf
        values.push(f32::from_bits(0x7fc0_0000)); // a quiet NaN
        values.push(f32::from_bits(0x0000_0001)); // the smallest subnormal
        values.push(f32::from_bits(0x007f_ffff)); // the largest subnormal

        let mut packed = vec![0u8; values.len() * 2];
        f32_to_bf16_bytes(&values, &mut packed);

        for (index, value) in values.iter().enumerate() {
            let want = crate::f32_to_bf16(*value).to_le_bytes();
            let at = index * 2;
            assert_eq!(
                &packed[at..at + 2],
                &want[..],
                "value {index} ({value:?}) packed differently"
            );
        }

        // AND BACK.
        let mut widened = vec![0f32; values.len()];
        bf16_to_f32_bytes(&packed, &mut widened);
        for (index, value) in values.iter().enumerate() {
            let want = crate::bf16_to_f32(crate::f32_to_bf16(*value));
            // THE BITS, NOT THE VALUES: a NaN is not equal to itself, and
            // the NaNs are half of what this test is about.
            assert_eq!(
                widened[index].to_bits(),
                want.to_bits(),
                "value {index} widened differently"
            );
        }
    }

    #[test]
    fn the_block_conversion_leaves_the_tail_alone() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut packed = vec![0u8; values.len() * 2];
        f32_to_bf16_bytes(&values, &mut packed);
        // The scalar's answer for the ninth value, written by hand.
        let want = crate::f32_to_bf16(9.0).to_le_bytes();
        assert_eq!(&packed[16..18], &want[..]);

        // A SHORT destination stops at its own length.
        let mut short = vec![0u8; 6];
        f32_to_bf16_bytes(&values, &mut short);
        assert_eq!(short[4], packed[4]);
        assert_eq!(short[5], packed[5]);
    }

    #[test]
    fn widening_with_bias_is_the_widen_then_add() {
        // The numbers the array's output holds are bf16-exact, so comparing the
        // fused helper with the two-pass original is an exact bit comparison.
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 31.0) / 9.0).collect();
        let packed = {
            let mut p = vec![0u8; values.len() * 2];
            f32_to_bf16_bytes(&values, &mut p);
            p
        };
        let bias: Vec<f32> = (0..values.len())
            .map(|i| ((i % 7) as f32 - 3.0) / 4.0)
            .collect();

        let mut want = vec![0f32; values.len()];
        bf16_to_f32_bytes(&packed, &mut want);
        for (w, b) in want.iter_mut().zip(&bias) {
            *w += *b;
        }

        let mut got = vec![0f32; values.len()];
        bf16_to_f32_add_bias_bytes(&packed, &mut got, &bias);
        assert_eq!(
            got, want,
            "the fused widen+add must be the two passes' result"
        );

        // A short bias contributes zero past its end, like the padded bias.
        let short = &bias[..5];
        let mut want_short = vec![0f32; values.len()];
        bf16_to_f32_bytes(&packed, &mut want_short);
        for (i, w) in want_short.iter_mut().enumerate() {
            *w += short.get(i).copied().unwrap_or(0.0);
        }
        let mut got_short = vec![0f32; values.len()];
        bf16_to_f32_add_bias_bytes(&packed, &mut got_short, short);
        assert_eq!(
            got_short, want_short,
            "a short bias must contribute zero past its end"
        );
    }

    #[test]
    fn the_padded_row_is_the_conversion_then_zeros() {
        let values = vec![1.0f32, -2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        const PAD: usize = 7;
        let row_len = values.len() + PAD;
        let mut got = vec![0xAAu8; row_len * 2 + 4];
        f32_to_bf16_bytes_padded(&values, &mut got, PAD);

        let mut want = vec![0xAAu8; row_len * 2 + 4];
        f32_to_bf16_bytes(&values, &mut want[..values.len() * 2]);
        want[values.len() * 2..row_len * 2].fill(0);
        assert_eq!(got, want, "one pass must write the old two passes' bytes");

        // A destination shorter than the row stops at its own length, and
        // still converts the same leading values.
        let mut short = vec![0xFFu8; 6];
        f32_to_bf16_bytes_padded(&values, &mut short, PAD);
        assert_eq!(&short, &want[..6]);
    }
}
