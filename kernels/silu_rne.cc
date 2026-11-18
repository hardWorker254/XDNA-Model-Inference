//===- silu_rne.cc ------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- SiLU with conv_even rounding on bf16 stores.
// SPDX-License-Identifier: Apache-2.0

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#define NPUE_ELTWISE_IMPL_ONLY
#include "silu.cc"
#undef NPUE_ELTWISE_IMPL_ONLY

extern "C" {

void silu_rne_bf16(bfloat16 *restrict in, bfloat16 *restrict out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  silu_impl(in, out, 1024);
}

void silu_rne_bf16_4k(bfloat16 *restrict in, bfloat16 *restrict out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  silu_impl(in, out, 4096);
}

} // extern "C"
