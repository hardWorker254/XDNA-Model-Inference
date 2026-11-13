//===- ctc.cpp -------------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the CTC greedy decode. See ctc.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "ocr/ctc.hpp"

#include <stdexcept>

namespace npue::ocr {

CtcResult ctc_decode(const float *logits, int64_t frames, int64_t classes,
                      const std::vector<std::string> &dict) {
  CtcResult r;
  if (classes <= 0 || frames <= 0) return r;
  if (static_cast<int64_t>(dict.size()) != classes)
    throw std::runtime_error(
        "ctc: the character table holds " + std::to_string(dict.size()) +
        " classes and the recogniser emitted " + std::to_string(classes) +
        " logits per frame. The table is the dictionary's characters plus the "
        "blank and the space, and a decoder that reads one and a network that "
        "learned another produce text that is confidently wrong.");
  std::string text;
  double sum = 0.0;
  int64_t n = 0;
  int64_t prev = -1;
  for (int64_t t = 0; t < frames; ++t) {
    const float *const row = logits + t * classes;
    int64_t best = 0;
    float bv = row[0];
    for (int64_t k = 1; k < classes; ++k) {
      if (row[k] > bv) {
        bv = row[k];
        best = k;
      }
    }
    // THE BLANK IS CLASS 0, and it is not a character.
    if (best == 0) {
      // The previous index is still remembered: a repeat is judged against
      // the frame BEFORE, blank or not.
      prev = best;
      continue;
    }
    // THE REPEATS ARE COLLAPSED against the previous frame's RAW index. A
    // character repeated twice is one character, and the second one is
    // dropped -- which is how a doubled letter ("ll") survives as one "l"
    // and how a blob of the same character does not become a word of them.
    if (best == prev) {
      prev = best;
      continue;
    }
    prev = best;
    text += dict[static_cast<size_t>(best)];
    sum += bv;
    ++n;
  }
  // The confidence is the mean over the KEPT frames, and an empty text has a
  // confidence of zero rather than a division by zero.
  r.text = std::move(text);
  r.conf = n > 0 ? sum / static_cast<double>(n) : 0.0;
  return r;
}

}  // namespace npue::ocr
