#!/usr/bin/env python3
#===----------------------------------------------------------------------===//
# Gates the XLMRTOK1 builder: the blob it writes must tokenize like the
# checkpoint's own tokenizer.
#
# THE CLAIM BEING CHECKED
#
# tools/pack/xlmr_tokenizer_table.py compiles a tokenizer.json into the table a
# container carries, and tools/lib/xlmr_tokenizer_ref.py is the executable spec
# of what that table means (it CONSUMES the blob and mirrors tokenizers 0.22.2
# quirk for quirk). Two things must therefore hold, and they fail for different
# reasons:
#
#   1. THE BYTES PARSE. The blob must load through the reference reader -- a
#      builder that wrote a field in the wrong order, or truncated the trailing
#      vocabulary, still produces a file, and the reader is what notices.
#
#   2. THE IDS MATCH HUGGINGFACE. The reader over THAT blob must reproduce
#      `tokenizers`' ids on a corpus that exercises every stage of the
#      pipeline: the Precompiled normaliser's grapheme rules and multi-byte
#      replacements, WhitespaceSplit's Rust notion of whitespace, Metaspace's
#      prepend scheme, and Unigram's Viterbi ties and <unk> fusing. Ids that
#      differ here are ids the embedding would be computed over, so this is the
#      only comparison worth making.
#
# The corpus is deliberately multilingual and adversarial rather than long:
# a stage either disagrees on a script, a combining mark or a full-width form,
# or it does not. Empty and whitespace-only inputs are included because those
# are where the prepend scheme and the empty-Viterbi path show up.
#
# Usage:
#   .venv/bin/python tools/verify/verify_xlmr_tokenizer.py [model_dir ...]
#   (default: every models/* that has both tokenizer.json and xlmr_tokenizer.bin)
#===----------------------------------------------------------------------===//

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "lib"))
sys.path.insert(0, str(REPO / "tools" / "pack"))

from xlmr_tokenizer_table import generate  # noqa: E402
from xlmr_tokenizer_ref import XlmrTokenizer  # noqa: E402

CORPUS = [
    "Hello, world!",
    "Привет, мир! Как дела?",
    "你好，世界！这是一个测试。",
    "مرحبا بالعالم",
    "שלום עולם",
    "नमस्ते दुनिया",
    "สวัสดีชาวโลก",
    "こんにちは世界",
    "한국어 테스트입니다",
    "Καλημέρα κόσμε",
    "Здравей, свят",
    "ünïcödé — “quotes” and dashes…",
    "e\u0301 combining acute then \u1e9b\u0323 weird",
    "  leading and trailing  spaces  ",
    "tabs\tand\nnewlines\r\nmixed",
    "a_b_c and snake_case plus-dash/slash",
    "1234567890 3.14159 -42 1e-9",
    "emoji 😀👨‍👩‍👧‍👦🇺🇦🎉",
    "ｆｕｌｌｗｉｄｔｈ ｔｅｘｔ",
    "ﬁ ligature Ⅻ roman ① circled",
    "آن‌ها کتاب‌ها می‌خوانند",
    "ជំរាបសួរពិភពលោក",
    "ᚠᚢᚦᚨᚱᚲ runic",
    "𝕳𝖊𝖑𝖑𝖔 𝓂𝒶𝓉𝒽",
    "H₂O CO₂ NaHCO₃",
    "mixed—em—dash—and—ellipsis…",
    "www.example.com/user?id=42&x=y",
    "",
    " ",
    "\u00a0nbsp\u2003emspace\u3000ideo",
]


def check(model_dir: Path) -> list[str]:
    tokenizer_json = model_dir / "tokenizer.json"
    try:
        from tokenizers import Tokenizer
    except ImportError:
        # NOT a skip-with-success: without the reference tokenizer there is
        # nothing to compare against, and a green line here would mean "nobody
        # looked".
        return ["the `tokenizers` package is not installed, so the ids cannot "
                "be compared against HuggingFace -- run under .venv "
                "(python -m venv .venv && .venv/bin/pip install tokenizers)"]

    blob = generate(tokenizer_json)
    # Parsed from the BYTES the builder returned, through the reference reader
    # -- not from the cache file, which may be a stale build of an older
    # tokenizer.json.
    tmp = Path(tempfile.mkdtemp()) / "xlmr_tokenizer.bin"
    tmp.write_bytes(blob)
    try:
        parsed = XlmrTokenizer(str(tmp))
    except Exception as exc:  # noqa: BLE001 -- the reader's own diagnosis
        return [f"the builder's blob does not load: {exc}"]
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)

    # No re-assertion of the writer's own guards (vocab ids, special spellings):
    # a blob that broke those would have been refused before it was written,
    # and one that slipped through shows up below as ids that disagree.
    problems = []
    hf = Tokenizer.from_file(str(tokenizer_json))
    mismatches = []
    for text in CORPUS:
        want = hf.encode(text).ids
        got = parsed.encode(text)
        if want != got:
            mismatches.append((text, want, got))
    if mismatches:
        detail = "; ".join(
            f"{t[:28]!r}: hf {w[:8]} vs table {g[:8]}"
            for t, w, g in mismatches[:4])
        problems.append(f"{len(mismatches)} of {len(CORPUS)} texts tokenize "
                        f"differently: {detail}")
    if not problems:
        print(f"  PASS  {len(blob)} bytes, {parsed.vocab_size} pieces, "
              f"{len(CORPUS)}/{len(CORPUS)} texts identical to tokenizers")
    return problems


def main(argv: list[str]) -> int:
    dirs = ([Path(p) for p in argv[1:]] if len(argv) > 1 else
            sorted(p for p in (REPO / "models").iterdir() if p.is_dir()))
    checked = failed = 0
    for model_dir in dirs:
        if not (model_dir / "tokenizer.json").is_file():
            continue
        if not (model_dir / "xlmr_tokenizer.bin").is_file():
            print(f"# skip        {model_dir.name}: no xlmr_tokenizer.bin "
                  f"(not an XLM-R tokenizer)")
            continue
        print(f"== {model_dir.name}")
        problems = check(model_dir)
        if problems:
            failed += 1
            for p in problems:
                print(f"  FAIL  {p}")
        else:
            checked += 1
    print(f"\nchecked {checked}, failed {failed}")
    if checked == 0:
        print("FAIL: no model had an XLM-R table, so nothing was verified")
        return 1
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
