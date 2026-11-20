//===- dwconv9x3_bf16.cc ---------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- research prototype: depthwise 3x3 on AIE-ML / XDNA1.
// SPDX-License-Identifier: Apache-2.0
//
// A depthwise 3x3 is THREE 3-tap 1-D convolutions accumulated, not a GEMM:
//
//   out[x] = bias + sum_{dy=0..2} sum_{dx=0..2} w[dy,dx] * in[dy][x+dx]
//
// Each dy is a `sliding_mul_ops<16,3,1,1,1,bfloat16,bfloat16>` mac, so every
// multiply is useful -- no block-diagonal B, no 100-250x MAC overhead.
//
// Layout: `in` holds three rows of `STRIDE` bf16 each for ONE channel. Row dy
// is padded on the right so `in[dy][x]` is the image at x, and taps
// w[dy][0..2] read x, x+1, x+2. STRIDE has 32 elements of slack so the
// 32-lane loads never run past a row. `w` is nine taps followed by the bias;
// `out` holds T outputs. T must be a multiple of 16 (one sliding_mul vector).
//
// TWO entry points:
//   dwconv9x3_bf16        -- the original one-channel call (baseline);
//   dwconv9x3_bf16_tiled  -- n_chan channels per core, each channel a block of
//                            `chan_stride` elements in `in` and T in `out`.
//                            The second is what the tiled/ping-pong IRON
//                            design uses; the first is kept so measurements
//                            stay comparable and nothing else can break.

#include <aie_api/aie.hpp>
#include <cstdint>

static inline void dwconv9x3_one(const bfloat16 *restrict in,
                                 bfloat16 *restrict out, int32_t T,
                                 int32_t stride) {
  // The 9 taps + bias are appended to the three padded rows in the same
  // L2 object, so there is only ONE input DMA channel per core column.
  const bfloat16 *w = in + 3 * stride;

  const float bias = static_cast<float>(w[9]);
  const ::aie::vector<float, 16> bias_v = ::aie::broadcast<float, 16>(bias);

  using conv = ::aie::sliding_mul_ops<16, 3, 1, 1, 1, bfloat16, bfloat16>;

  ::aie::vector<bfloat16, 16> taps[3];
  for (int dy = 0; dy < 3; ++dy) {
    taps[dy] = ::aie::zeros<bfloat16, 16>();
    for (int dx = 0; dx < 3; ++dx)
      taps[dy].set(w[dy * 3 + dx], dx);
  }

  for (int32_t x = 0; x < T; x += 16) {
    ::aie::accum<accfloat, 16> acc;
    acc.from_vector(bias_v);
    for (int dy = 0; dy < 3; ++dy) {
      const bfloat16 *row = in + dy * stride + x;
      // 16 outputs need 16 + 3 - 1 = 18 taps; load 32 and let
      // sliding_mul take the first 18.
      const ::aie::vector<bfloat16, 32> window = ::aie::concat(
          ::aie::load_v<16>(row), ::aie::load_v<16>(row + 16));
      acc = conv::mac(acc, taps[dy], 0, window, 0);
    }
    ::aie::store_v(out + x, acc.to_vector<bfloat16>());
  }
}

extern "C" {

void dwconv9x3_bf16(bfloat16 *in, bfloat16 *out, int32_t T, int32_t stride) {
  event0();
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  dwconv9x3_one(in, out, T, stride);
  ::aie::set_rounding(saved_rounding);
  event1();
}

void dwconv9x3_bf16_tiled(bfloat16 *in, bfloat16 *out, int32_t T,
                          int32_t stride, int32_t n_chan,
                          int32_t chan_stride) {
  event0();
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  // Straight-line calls for the npu1 cases (1, 2 or 4 channels per core).
  // NOTE: chan_stride must be a multiple of 16 bf16 -- load_v<16> needs a
  // 32-byte-aligned address, and the host pads the block accordingly.
  dwconv9x3_one(in, out, T, stride);
  for (int c = 1; c < n_chan; ++c)
    dwconv9x3_one(in + c * chan_stride, out + c * T, T, stride);
  ::aie::set_rounding(saved_rounding);
  event1();
}

} // extern "C"
