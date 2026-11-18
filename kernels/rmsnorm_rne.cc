//===- rmsnorm_rne.cc ---------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- RMSNorm with conv_even rounding on bf16 stores.
// SPDX-License-Identifier: Apache-2.0
//
// Companion to rmsnorm.cc, matching the *_rne.cc pattern used for other
// eltwise kernels: the AIE default rounding mode is floor, so narrowing
// fp32 -> bf16 carries a downward bias unless set explicitly. Separate file
// to keep A/B builds unambiguous.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#define NPUE_ELTWISE_IMPL_ONLY
#include "rmsnorm.cc"
#undef NPUE_ELTWISE_IMPL_ONLY

extern "C" {

void rmsnorm_rne_bf16(bfloat16 *restrict input, float *restrict weight,
                      bfloat16 *restrict output) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  rmsnorm_impl(input, weight, output, RN_ROWS);
}

void rmsnorm_il4_rne_bf16(bfloat16 *restrict input, float *restrict weight,
                          bfloat16 *restrict output) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  rmsnorm_il4_impl(input, weight, output, RN_ROWS);
}

} // extern "C"
