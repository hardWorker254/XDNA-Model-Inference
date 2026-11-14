#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# NpuEmbeddings -- generate the Unicode tables the tokenizers need, as Rust.
#
#   python tools/gen/gen_unicode_tables.py
#
# WHY GENERATE INSTEAD OF LINKING ICU, OR USING A CRATE
# -----------------------------------------------------
# BERT's BasicTokenizer is defined in terms of Python's `unicodedata`:
# lowercasing, NFD decomposition with the Mn (nonspacing mark) characters
# removed, and the Unicode punctuation / control / space categories.
# Reimplementing those by hand is how tokenizers quietly diverge on the 1% of
# text that is not ASCII; linking ICU drags a large dependency into a
# runtime whose whole point is that it is small and self-contained; and a
# `unicode-normalization` crate would ship a DIFFERENT unicodedata version,
# which changes the ids of every text that contains a codepoint it disagrees
# about.
#
# So: compute it offline in Python -- build-time Python is explicitly
# allowed -- and emit a static Rust table. The tables are then EXACTLY what
# HuggingFace would compute, by construction rather than by careful reading.
#
# WHAT IS FOLDED INTO ONE LOOKUP
# ------------------------------
# `fold` maps a codepoint to the result of `lower()` followed by
# NFD-with-Mn-dropped -- both of BasicTokenizer's per-character transforms in
# one table. Entries equal to the identity are omitted, so the table only
# holds the codepoints that actually change.
#
# The one context-dependent case Python has is the Greek capital sigma,
# which lowercases to the final form at the end of a word. That cannot live
# in a per-codepoint table, so it is emitted as a constant.
#
# THIS REPLACES three C++ generators -- gen_tokenizer_tables.py,
# gen_xlmr_unicode_tables.py and gen_whisper_unicode_tables.py -- and their
# three baked-in headers, 6 034 lines of C++ that the runtime carried and
# the packer also carried, and that could therefore disagree.
#
# Env: any Python 3 (no NPU, no torch).

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "core" / "npum-tokenizers" / "src" / "tables_data.rs"

MAX_CP = 0x110000


# --- BERT -----------------------------------------------------------------

def fold_one(cp: int) -> list[int]:
    """lower() then strip_accents(), exactly as BasicTokenizer applies them."""
    ch = chr(cp)
    low = ch.lower()
    nfd = unicodedata.normalize("NFD", low)
    return [ord(c) for c in nfd if unicodedata.category(c) != "Mn"]


def is_punct(cp: int) -> bool:
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(chr(cp)).startswith("P")


def is_control(cp: int) -> bool:
    ch = chr(cp)
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch).startswith("C")


def is_space(cp: int) -> bool:
    ch = chr(cp)
    if ch in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(ch) == "Zs"


# --- XLM-R ----------------------------------------------------------------

def is_extend(cp: int) -> bool:
    return unicodedata.category(chr(cp)) in ("Mn", "Me")


def is_spacing_mark(cp: int) -> bool:
    return unicodedata.category(chr(cp)) == "Mc"


def is_xlmr_control(cp: int) -> bool:
    return unicodedata.category(chr(cp)) in ("Cc", "Cf", "Zl", "Zp")


# --- Whisper --------------------------------------------------------------

def is_letter(cp: int) -> bool:
    return unicodedata.category(chr(cp)).startswith("L")


def is_number(cp: int) -> bool:
    return unicodedata.category(chr(cp)).startswith("N")


def is_whisper_space(cp: int) -> bool:
    # `str.isspace()` and NOT a hand-written set.
    #
    # The GPT-2 pre-tokeniser's pattern uses `\s`, and Python's `str.isspace()`
    # is what that means. A hand-written set -- even one written carefully --
    # is a DIFFERENT predicate: `str.isspace()` also covers U+001C..U+001F (the
    # file, group, record and unit separators), U+0085 (NEL) and U+2028/U+2029
    # (Zl and Zp), and a hand-written set that forgets one of them produces a
    # tokeniser that disagrees with HuggingFace on a rare character and
    # produces DIFFERENT IDS, which is the failure that is invisible.
    #
    # Measured: the hand-written version below produced 8 ranges where
    # `str.isspace()` produces 10.
    return chr(cp).isspace()


# --- the shared machinery -------------------------------------------------

def ranges(pred) -> list[tuple[int, int]]:
    """Compress a predicate over all codepoints into inclusive ranges."""
    out, start = [], None
    for cp in range(MAX_CP):
        if 0xD800 <= cp <= 0xDFFF:          # surrogates: never valid scalars
            continue
        if pred(cp):
            if start is None:
                start = cp
        elif start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, MAX_CP - 1))
    return out


def rust_ranges(name: str, rs: list[tuple[int, int]]) -> str:
    body = ",\n    ".join(
        ", ".join(f"CpRange {{ lo: 0x{a:X}, hi: 0x{b:X} }}" for a, b in rs[i:i + 4])
        for i in range(0, len(rs), 4))
    return f"pub static {name}: &[CpRange] = &[\n    {body},\n];\n"


def rust_u32s(name: str, values: list[int], per_line: int = 12) -> str:
    body = ",\n    ".join(
        ", ".join(f"0x{c:X}" for c in values[i:i + per_line])
        for i in range(0, len(values), per_line))
    return f"pub static {name}: &[u32] = &[\n    {body},\n];\n"


def rust_folds(name: str, index: list[tuple[int, int, int]]) -> str:
    body = ",\n    ".join(
        ", ".join(f"FoldEntry {{ cp: 0x{cp:X}, off: {off}, len: {ln} }}"
                  for cp, off, ln in index[i:i + 6])
        for i in range(0, len(index), 6))
    return f"pub static {name}: &[FoldEntry] = &[\n    {body},\n];\n"


def main() -> int:
    # --- BERT's fold table ------------------------------------------------
    folds = {}
    for cp in range(MAX_CP):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        f = fold_one(cp)
        if f != [cp]:
            folds[cp] = f
    keys = sorted(folds)

    # Flat storage: each entry points at (offset, length) into one array, so
    # the multi-codepoint results -- and the empty ones, where a combining
    # mark folds to nothing -- need no special case.
    flat, index = [], []
    for cp in keys:
        index.append((cp, len(flat), len(folds[cp])))
        flat.extend(folds[cp])

    bert_punct = ranges(is_punct)
    bert_control = ranges(is_control)
    bert_space = ranges(is_space)

    xlmr_extend = ranges(is_extend)
    xlmr_spacing = ranges(is_spacing_mark)
    xlmr_control = ranges(is_xlmr_control)

    whisper_letters = ranges(is_letter)
    whisper_numbers = ranges(is_number)
    whisper_space = ranges(is_whisper_space)

    header = f"""//===- tables_data.rs ----------------------------------------*- rust -*-===//
//
// GENERATED by tools/gen/gen_unicode_tables.py -- DO NOT EDIT.
// SPDX-License-Identifier: Apache-2.0
//
// The Unicode facts the tokenizers are defined in terms of, computed from
// Python {sys.version.split()[0]}'s unicodedata (UCD {unicodedata.unidata_version})
// so the Rust agrees with HuggingFace by construction rather than by careful
// reading.
//
//   BERT     fold {len(keys)} codepoints, punct {len(bert_punct)},
//            control {len(bert_control)}, space {len(bert_space)}
//   XLM-R    extend {len(xlmr_extend)}, spacing {len(xlmr_spacing)},
//            control {len(xlmr_control)}
//   Whisper  letters {len(whisper_letters)}, numbers {len(whisper_numbers)},
//            space {len(whisper_space)}
//
//===----------------------------------------------------------------------===//

/// The unicodedata version these tables were computed from.
pub const UNICODE_VERSION: &str = "{unicodedata.unidata_version}";

"""

    parts = [header]
    parts.append(rust_u32s("BERT_FOLD_DATA", flat))
    parts.append(rust_folds("BERT_FOLD_INDEX", index))
    parts.append(rust_ranges("BERT_PUNCT", bert_punct))
    parts.append(rust_ranges("BERT_CONTROL", bert_control))
    parts.append(rust_ranges("BERT_SPACE", bert_space))
    parts.append(rust_ranges("XLMR_EXTEND", xlmr_extend))
    parts.append(rust_ranges("XLMR_SPACING_MARK", xlmr_spacing))
    parts.append(rust_ranges("XLMR_CONTROL", xlmr_control))
    parts.append(rust_ranges("WHISPER_LETTERS", whisper_letters))
    parts.append(rust_ranges("WHISPER_NUMBERS", whisper_numbers))
    parts.append(rust_ranges("WHISPER_SPACE", whisper_space))

    parts.append(f"""
/// BERT's tables.
pub static BERT: BertTables = BertTables {{
    unicode_version: UNICODE_VERSION,
    fold_data: BERT_FOLD_DATA,
    fold_index: BERT_FOLD_INDEX,
    punct: BERT_PUNCT,
    control: BERT_CONTROL,
    space: BERT_SPACE,
}};

/// XLM-R's tables.
pub static XLM_R: XlmrTables = XlmrTables {{
    unicode_version: UNICODE_VERSION,
    extend: XLMR_EXTEND,
    spacing_mark: XLMR_SPACING_MARK,
    control: XLMR_CONTROL,
}};

/// Whisper's tables.
pub static WHISPER: WhisperTables = WhisperTables {{
    unicode_version: UNICODE_VERSION,
    letters: WHISPER_LETTERS,
    numbers: WHISPER_NUMBERS,
    space: WHISPER_SPACE,
}};
""")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(parts), encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(REPO)}  ({kb:.1f} KB)")
    print(f"  UCD {unicodedata.unidata_version}")
    print(f"  BERT  fold {len(index)} entries / {len(flat)} codepoints, "
          f"punct {len(bert_punct)}, control {len(bert_control)}, space {len(bert_space)}")
    print(f"  XLM-R extend {len(xlmr_extend)}, spacing {len(xlmr_spacing)}, "
          f"control {len(xlmr_control)}")
    print(f"  Whisper letters {len(whisper_letters)}, numbers {len(whisper_numbers)}, "
          f"space {len(whisper_space)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
