//===- silu.cc ----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- SiLU (Swish) activation: x * sigmoid(x).
// SPDX-License-Identifier: Apache-2.0
//
// WHY THIS FILE EXISTS
// --------------------
// Several conv-only models in this tree (e.g. YOLOv8-pose) use SiLU as their
// activation, but today it is fused into convolution epilogues in the packed
// graphs and there is no standalone eltwise design/kernel for it. This kernel
// provides a standalone bf16 SiLU pass so that, in principle, an unfused
// activation pass could be dispatched as its own array operation.
//
// Definition:
//   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Implementation notes:
//   - exp is computed as 2^(z * log2(e)) via exp2_poly.h, with clamping to
//     avoid negative exponent-field bit patterns (same rationale as
//     gelu_erf.cc / softmax.cc).
//   - fp32 accumulation/math, bf16 input/output.
//   - RNE narrowing is provided by kernels/silu_rne.cc.

#include "aie_kernel_utils.h"
#include "exp2_poly.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#define SI_ARG_FLOOR -125.0f

static inline void silu_impl(bfloat16 *restrict in, bfloat16 *restrict out,
                             int n) {
  const aie::vector<float, 16> vone = aie::broadcast<float, 16>(1.0f);
  const aie::vector<float, 16> vzero = aie::broadcast<float, 16>(0.0f);
  const aie::vector<float, 16> vlog2e =
      aie::broadcast<float, 16>(1.4426950408889634f);
  const aie::vector<float, 16> vfloor =
      aie::broadcast<float, 16>(SI_ARG_FLOOR);

  for (int i = 0; i < n; i += 16) {
    aie::accum<accfloat, 16> acc;
    acc.from_vector(aie::load_v<16>(in + i));
    const aie::vector<float, 16> x = acc.to_vector<float>();

    // z = -x * log2(e), clamped
    const aie::vector<float, 16> z =
        aie::mul(aie::sub(vzero, x), vlog2e).to_vector<float>();
    const aie::vector<float, 16> ex = exp2_poly(aie::max(z, vfloor));

    // sigmoid(x) = 1 / (1 + exp(-x))
    const aie::vector<float, 16> denom =
        aie::add(vone, ex);
    const aie::vector<float, 16> sig = aie::div(vone, denom).to_vector<float>();

    // silu(x) = x * sigmoid(x)
    const aie::vector<float, 16> y = aie::mul(x, sig).to_vector<float>();

    aie::accum<accfloat, 16> out_acc;
    out_acc.from_vector(y);
    aie::store_v(out + i, out_acc.to_vector<bfloat16>());
  }
}

#ifndef NPUE_ELTWISE_IMPL_ONLY
extern "C" {

void silu_bf16(bfloat16 *restrict in, bfloat16 *restrict out) {
  silu_impl(in, out, 1024);
}

void silu_bf16_4k(bfloat16 *restrict in, bfloat16 *restrict out) {
  silu_impl(in, out, 4096);
}

} // extern "C"
#endif // NPUE_ELTWISE_IMPL_ONLY
