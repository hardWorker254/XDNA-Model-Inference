//===- ctc.hpp -------------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the CTC greedy decode.
//
// WHAT A CTC DECODE IS
// -------------------
// The recogniser emits, for every frame of the crop, one logit per CLASS --
// the classes are the dictionary's characters plus the blank and the space --
// and the text is the sequence of the frame's best classes with the BLANK
// frames removed and the REPEATS collapsed.
//
// The blank is the CTC null class and it is class 0. The repeats are
// collapsed by comparing each frame with the one before it, and the
// comparison is with the RAW index: a frame that repeats the class before a
// blank is still a repeat and is dropped, and one that repeats the class
// after a blank is a new character.
//
// The confidence is the MEAN of the kept frames' probabilities, which is
// what PaddleOCR reports and what a caller compares against a threshold.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace npue::ocr {

struct CtcResult {
  std::string text;
  double conf = 0.0;
};

// `logits` is the recogniser's output, [frames, classes] row-major, already
// softmax'd by the graph. `dict` is the character table in class order:
// 'blank' first, the dictionary's characters, and the space last.
CtcResult ctc_decode(const float *logits, int64_t frames, int64_t classes,
                      const std::vector<std::string> &dict);

}  // namespace npue::ocr
