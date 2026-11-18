//===- rmsnorm.cc -------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- RMSNorm for Gemma-family models (e.g. embeddinggemma-300m).
// SPDX-License-Identifier: Apache-2.0
//
// WHY THIS FILE EXISTS
// --------------------
// Gemma normalises with RMSNorm, not LayerNorm: there is no mean subtraction,
// no beta, and the scale is a learned weight applied to the RMS-normalised
// value. kernels/layernorm.cc implements LayerNorm (mean, variance, gamma,
// beta) and cannot express RMSNorm without changing its definition, so this is
// a separate kernel body.
//
// Semantics (matching Hugging Face Gemma / Gemma3 RMSNorm):
//   rms      = sqrt(mean(x^2) + eps)
//   y        = x / rms * weight
//
// `eps` is inside the sqrt, like LayerNorm. The reduction is the mean of
// squares over the last dimension (the hidden width). Accumulation and the
// scale are in fp32; input/output are bf16. The packed weight is provided as
// fp32 (the exporter/runtime supplies the per-channel scale values; for
// Gemma3's (1 + w) form that folding is done when the parameters are staged).
//
// This follows the same AIE API patterns and constraints as layernorm.cc:
// two passes (sum of squares, then scale), no scalar float ops in the inner
// vector work, widening/narrowing through accumulator paths, and fixed
// row-chunk sizes bounded by L1.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef RN_COLS
#define RN_COLS 768
#endif
#ifndef RN_EPS
#define RN_EPS 1e-6f
#endif
#ifndef RN_ROWS
#define RN_ROWS 16
#endif
#define RN_VECS (RN_COLS / 16)
#define RN_INV_COLS (1.0f / (float)RN_COLS)

static inline float fast_invsqrt(float x) {
  float xhalf = 0.5f * x;
  int i = *reinterpret_cast<int *>(&x);
  i = 0x5f3759df - (i >> 1);
  float y = *reinterpret_cast<float *>(&i);
  y = y * (1.5f - xhalf * y * y);
  y = y * (1.5f - xhalf * y * y);
  return y;
}

#ifndef NPUE_ELTWISE_IMPL_ONLY
void rmsnorm_impl(bfloat16 *restrict input, float *restrict weight,
                  bfloat16 *restrict output, const int32_t rows) {
  event0();

  const aie::vector<bfloat16, 16> vone_bf =
      aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  const aie::vector<float, 16> vone_f = aie::broadcast<float, 16>(1.0f);

  for (int r = 0; r < rows; r++) {
    bfloat16 *in = input + r * RN_COLS;
    bfloat16 *out = output + r * RN_COLS;

    // -- pass 1: mean of squares -------------------------------------------
    aie::vector<float, 16> acc2 = aie::zeros<float, 16>();
    for (int i = 0; i < RN_VECS; i++) {
      aie::vector<bfloat16, 16> xb = aie::load_v<16>(in + i * 16);
      aie::vector<float, 16> x = aie::mul(xb, vone_bf).to_vector<float>();
      acc2 = aie::add(acc2, aie::mul(x, x).to_vector<float>());
    }
    float mean_sq = aie::reduce_add(acc2) * RN_INV_COLS;
    float inv_rms = fast_invsqrt(mean_sq + RN_EPS);
    aie::vector<float, 16> inv_rms_v = aie::broadcast<float, 16>(inv_rms);

    // -- pass 2: normalize and scale ---------------------------------------
    for (int i = 0; i < RN_VECS; i++) {
      aie::vector<bfloat16, 16> xb = aie::load_v<16>(in + i * 16);
      aie::vector<float, 16> x = aie::mul(xb, vone_bf).to_vector<float>();
      aie::vector<float, 16> nrm = aie::mul(x, inv_rms_v).to_vector<float>();
      aie::vector<float, 16> w = aie::load_v<16>(weight + i * 16);
      aie::vector<float, 16> y = aie::mul(nrm, w).to_vector<float>();
      aie::store_v(out + i * 16, aie::mul(y, vone_f).to_vector<bfloat16>());
    }
  }

  event1();
  return;
}

extern "C" {

void rmsnorm_bf16(bfloat16 *restrict input, float *restrict weight,
                  bfloat16 *restrict output) {
  rmsnorm_impl(input, weight, output, RN_ROWS);
}

} // extern "C"
#endif // NPUE_ELTWISE_IMPL_ONLY

// 4-row interleaved variant, same rationale as layernorm_il4: hide serial
// dependency latency by keeping independent rows in flight. Numerics are
// bit-identical per row.
void rmsnorm_il4_impl(bfloat16 *restrict input, float *restrict weight,
                      bfloat16 *restrict output, const int32_t rows) {
  event0();

  const aie::vector<bfloat16, 16> vone_bf =
      aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  const aie::vector<float, 16> vone_f = aie::broadcast<float, 16>(1.0f);

  for (int r = 0; r < rows; r += 4) {
    bfloat16 *in0 = input + (r + 0) * RN_COLS;
    bfloat16 *in1 = input + (r + 1) * RN_COLS;
    bfloat16 *in2 = input + (r + 2) * RN_COLS;
    bfloat16 *in3 = input + (r + 3) * RN_COLS;
    bfloat16 *out0 = output + (r + 0) * RN_COLS;
    bfloat16 *out1 = output + (r + 1) * RN_COLS;
    bfloat16 *out2 = output + (r + 2) * RN_COLS;
    bfloat16 *out3 = output + (r + 3) * RN_COLS;

    aie::vector<float, 16> a0 = aie::zeros<float, 16>();
    aie::vector<float, 16> a1 = aie::zeros<float, 16>();
    aie::vector<float, 16> a2 = aie::zeros<float, 16>();
    aie::vector<float, 16> a3 = aie::zeros<float, 16>();
    for (int i = 0; i < RN_VECS; i++) {
      aie::vector<float, 16> x0 =
          aie::mul(aie::load_v<16>(in0 + i * 16), vone_bf).to_vector<float>();
      aie::vector<float, 16> x1 =
          aie::mul(aie::load_v<16>(in1 + i * 16), vone_bf).to_vector<float>();
      aie::vector<float, 16> x2 =
          aie::mul(aie::load_v<16>(in2 + i * 16), vone_bf).to_vector<float>();
      aie::vector<float, 16> x3 =
          aie::mul(aie::load_v<16>(in3 + i * 16), vone_bf).to_vector<float>();
      a0 = aie::add(a0, aie::mul(x0, x0).to_vector<float>());
      a1 = aie::add(a1, aie::mul(x1, x1).to_vector<float>());
      a2 = aie::add(a2, aie::mul(x2, x2).to_vector<float>());
      a3 = aie::add(a3, aie::mul(x3, x3).to_vector<float>());
    }
    const aie::vector<float, 16> is0 = aie::broadcast<float, 16>(
        fast_invsqrt(aie::reduce_add(a0) * RN_INV_COLS + RN_EPS));
    const aie::vector<float, 16> is1 = aie::broadcast<float, 16>(
        fast_invsqrt(aie::reduce_add(a1) * RN_INV_COLS + RN_EPS));
    const aie::vector<float, 16> is2 = aie::broadcast<float, 16>(
        fast_invsqrt(aie::reduce_add(a2) * RN_INV_COLS + RN_EPS));
    const aie::vector<float, 16> is3 = aie::broadcast<float, 16>(
        fast_invsqrt(aie::reduce_add(a3) * RN_INV_COLS + RN_EPS));

    for (int i = 0; i < RN_VECS; i++) {
      const aie::vector<float, 16> w = aie::load_v<16>(weight + i * 16);
      aie::vector<float, 16> x0 =
          aie::mul(aie::load_v<16>(in0 + i * 16), vone_bf).to_vector<float>();
      aie::vector<float, 16> x1 =
          aie::mul(aie::load_v<16>(in1 + i * 16), vone_bf).to_vector<float>();
      aie::vector<float, 16> x2 =
          aie::mul(aie::load_v<16>(in2 + i * 16), vone_bf).to_vector<float>();
      aie::vector<float, 16> x3 =
          aie::mul(aie::load_v<16>(in3 + i * 16), vone_bf).to_vector<float>();
      aie::store_v(out0 + i * 16,
                   aie::mul(aie::mul(aie::mul(x0, is0).to_vector<float>(), w)
                                .to_vector<float>(),
                            vone_f)
                       .to_vector<bfloat16>());
      aie::store_v(out1 + i * 16,
                   aie::mul(aie::mul(aie::mul(x1, is1).to_vector<float>(), w)
                                .to_vector<float>(),
                            vone_f)
                       .to_vector<bfloat16>());
      aie::store_v(out2 + i * 16,
                   aie::mul(aie::mul(aie::mul(x2, is2).to_vector<float>(), w)
                                .to_vector<float>(),
                            vone_f)
                       .to_vector<bfloat16>());
      aie::store_v(out3 + i * 16,
                   aie::mul(aie::mul(aie::mul(x3, is3).to_vector<float>(), w)
                                .to_vector<float>(),
                            vone_f)
                       .to_vector<bfloat16>());
    }
  }

  event1();
}

#ifndef NPUE_ELTWISE_IMPL_ONLY
extern "C" {

void rmsnorm_il4_bf16(bfloat16 *restrict input, float *restrict weight,
                      bfloat16 *restrict output) {
  rmsnorm_il4_impl(input, weight, output, RN_ROWS);
}

} // extern "C"
#endif // NPUE_ELTWISE_IMPL_ONLY
