#!/usr/bin/env python3
#===----------------------------------------------------------------------===//
# The XLMRTOK1 table builder: tokenizer.json -> the blob a container carries.
#
# WHY THIS FILE EXISTS
#
# The runtime never parses tokenizer.json. `core/npum-tokenizers` reads a
# compact blob of precomputed tables (Darts trie, normalisation strings, vocab
# pieces and their log-scores) that the PACKER bakes into the container -- one
# file, one digest, nothing read from the model directory at run time, and no
# Unicode or regex logic on the hot path that could drift from the packer's.
#
# That blob used to be built by the reference implementation only (`runtime/src/tokenizers/
# xlmr_tokenizer_gen.cpp` in the pre-port tree), which left the Python packer
# with no way to produce one: it refused gte packouts with "generate it first
# with the reference packer". This is the same builder in Python, so a gte container
# is buildable from a checkpoint and this tree alone.
#
# THE SPECIFICATION IT IS CHECKED AGAINST
#
# tools/lib/xlmr_tokenizer_ref.py is the executable spec: it CONSUMES the blob
# and reproduces tokenizers 0.22.2 byte-exactly over a multilingual corpus
# (tools/verify/verify_tokenizer_xlmr.py holds it to HuggingFace's ids). The
# layout below is written to match that reader field for field, so the two can
# be diffed against each other and against this file's writer.
#
# BLOB LAYOUT (little-endian, every field exactly what the reader unpacks)
#
#   "XLMRTOK1"                                    8 bytes, magic
#   u32 version, vocab_size, unk_id, bos_id, eos_id, pad_id, mask_id,
#       add_bos, add_eos, metaspace_cp, prepend_scheme, metaspace_split,
#       max_piece_bytes, max_piece_chars          14 u32
#   f64 min_score, f64 unk_score                  Viterbi floor and <unk> score
#   u32 trie_size, u32 norm_size
#   trie bytes                                   darts_clone double-array
#   normalisation strings                        NUL-terminated, indexed by the
#                                                trie's first-match value
#   f64 scores[vocab_size]
#   per piece: u16 length + UTF-8 bytes          in id order
#
# WHY EVERY CHECK BELOW IS A REFUSAL AND NOT A REPAIR
#
# The blob is a compiled form of a tokenizer, and a tokenizer that disagrees
# with the checkpoint's own is not a smaller error than a wrong weight: the ids
# feed the embedding, so a silent fix produces a correctly-shaped vector of the
# wrong text. Each guard names the assumption the runtime makes and stops.
#
# Usage:
#   python tools/pack/xlmr_tokenizer_table.py <tokenizer.json> [out.bin]
#   (default out: xlmr_tokenizer.bin beside the tokenizer.json, which is the
#   name and place the packer looks for)
#===----------------------------------------------------------------------===//

from __future__ import annotations

import base64
import json
import struct
import sys
from pathlib import Path

MAGIC = b"XLMRTOK1"
VERSION = 1

#: Offset of `vocab_size` in the header, for callers that want to report it
#: without parsing the whole blob: the magic, then the version u32.
VOCAB_SIZE_AT = len(MAGIC) + 4

#: The metaspace replacement, U+2581. Named rather than inlined so the u32
#: written into the table and the string compared against tokenizer.json
#: cannot drift apart.
METASPACE = "\u2581"

#: The five specials the runtime needs ids for, as XLM-R spells them.
EXPECTED_SPECIALS = ("<s>", "<pad>", "</s>", "<unk>", "<mask>")

#: `tokenizers` spells the Metaspace prepend schemes this way; the runtime
#: stores an enum because a strcmp on the hot path is not worth it.
PREPEND_SCHEMES = {"never": 0, "first": 1, "always": 2}


def _refuse(message: str):
    raise SystemExit(message)


def _u32(value: int) -> bytes:
    return struct.pack("<I", value & 0xFFFFFFFF)


def _f64(value: float) -> bytes:
    return struct.pack("<d", value)


def _str16(text: str) -> bytes:
    raw = text.encode("utf-8")
    if len(raw) > 0xFFFF:
        _refuse(f"a vocab piece exceeds 65535 bytes: {text[:50]!r}...")
    return struct.pack("<H", len(raw)) + raw


def generate(tokenizer_json: Path) -> bytes:
    """Build the XLMRTOK1 blob for one tokenizer.json.

    Every guard here is the reference generator's guard, in the same order and
    against the same field, so the two can be compared line by line.
    """
    tokenizer_json = Path(tokenizer_json)
    try:
        tok = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    except OSError as exc:
        _refuse(f"cannot read {tokenizer_json}: {exc}")

    # --- model: must be Unigram, and shaped the way this builder expects -----
    model = tok.get("model")
    if not isinstance(model, dict):
        _refuse(f"{tokenizer_json}: no model object")
    if model.get("type") != "Unigram":
        _refuse(
            f"expected model.type == 'Unigram' -- this generator implements "
            f"SentencePiece Unigram specifically (the BPE builder is "
            f"tools/gen/gen_tokenizer_tables.py for gemma); {tokenizer_json} "
            f"has {model.get('type')!r}")
    if bool(model.get("byte_fallback")):
        _refuse(
            "expected byte_fallback: false -- the Viterbi <unk> path assumes "
            "no byte pieces; a checkpoint with byte_fallback needs new code, "
            "not this table")

    vocab = model.get("vocab")
    if not isinstance(vocab, list):
        _refuse(f"{tokenizer_json}: model.vocab is not a list")
    vocab_size = len(vocab)
    unk_id = model.get("unk_id")
    if not isinstance(unk_id, int) or isinstance(unk_id, bool):
        _refuse(f"unk_id {unk_id!r} is not an integer")
    if not 0 <= unk_id < vocab_size:
        _refuse(f"unk_id {unk_id} not an id in [0, {vocab_size})")

    pieces: list[str] = []
    scores: list[float] = []
    seen: set[str] = set()
    for row in vocab:
        if not (isinstance(row, list) and len(row) == 2):
            _refuse(f"vocab row {row!r} is not a [piece, score] pair")
        piece, score = row
        if not isinstance(piece, str):
            _refuse(f"vocab piece {piece!r} is not a string")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            _refuse(f"vocab score {score!r} for {piece!r} is not a number")
        if piece in seen:
            # The runtime looks pieces up in a hash map keyed by the string, so
            # one of the two scores would be silently unreachable.
            _refuse(f"duplicate vocab piece {piece!r} -- the runtime's hash "
                    f"lookup would silently keep one of the two scores")
        seen.add(piece)
        pieces.append(piece)
        scores.append(float(score))

    min_score = min(scores)
    # tokenizers' kUnkPenalty: an <unk> node must lose to every real piece, and
    # by a margin that makes it a last resort rather than a tie.
    unk_score = min_score - 10.0

    # --- normalizer: Precompiled charsmap, decoded and sanity-checked --------
    normalizer = tok.get("normalizer")
    if not isinstance(normalizer, dict) or normalizer.get("type") != "Precompiled":
        _refuse(
            "expected normalizer type 'Precompiled' with a precompiled_charsmap "
            "-- the charsmap trie below would not apply")
    charsmap_b64 = normalizer.get("precompiled_charsmap")
    if not isinstance(charsmap_b64, str) or not charsmap_b64:
        _refuse("normalizer has no precompiled_charsmap")
    charsmap = base64.b64decode(charsmap_b64)
    if len(charsmap) < 4:
        _refuse("precompiled_charsmap shorter than its own size field")
    (trie_size,) = struct.unpack_from("<I", charsmap, 0)
    if trie_size % 4 != 0 or 4 + trie_size > len(charsmap):
        _refuse(
            f"charsmap trie size {trie_size} inconsistent with blob length "
            f"{len(charsmap)} -- the u32-LE-size-then-trie-then-strings layout "
            f"assumption is wrong for this checkpoint")
    trie = charsmap[4:4 + trie_size]
    normalized = charsmap[4 + trie_size:]
    norm_size = len(normalized)
    if norm_size and normalized[-1] != 0:
        _refuse(
            "charsmap normalized-strings blob does not end in NUL -- the "
            "'replacement runs to the next NUL' walk would read past the end")

    # --- pre_tokenizer: exactly [WhitespaceSplit, Metaspace] ----------------
    pre = tok.get("pre_tokenizer")
    if not isinstance(pre, dict) or pre.get("type") != "Sequence":
        _refuse("expected pre_tokenizer Sequence")
    subs = pre.get("pretokenizers")
    if not isinstance(subs, list) or len(subs) != 2:
        _refuse("expected pre_tokenizers [WhitespaceSplit, Metaspace] -- the "
                "pre-tokenization in the runtime tokenizer would be wrong")
    kinds = [s.get("type") if isinstance(s, dict) else None for s in subs]
    if kinds != ["WhitespaceSplit", "Metaspace"]:
        _refuse(f"expected pre_tokenizers [WhitespaceSplit, Metaspace], got "
                f"{kinds} -- the pre-tokenization in the runtime tokenizer "
                f"would be wrong")
    meta = subs[1]
    if meta.get("replacement") != METASPACE:
        _refuse(f"expected Metaspace replacement U+2581, got "
                f"{meta.get('replacement')!r}")

    scheme_name = meta.get("prepend_scheme")
    if scheme_name is None:
        # Legacy serialization: an add_prefix_space bool and nothing else.
        scheme_name = "always" if bool(meta.get("add_prefix_space")) else "never"
    if scheme_name not in PREPEND_SCHEMES:
        _refuse(f"unknown Metaspace prepend_scheme {scheme_name!r}")
    if "add_prefix_space" in meta and \
            bool(meta["add_prefix_space"]) != (scheme_name != "never"):
        _refuse(f"Metaspace add_prefix_space contradicts prepend_scheme "
                f"{scheme_name!r}")
    prepend_scheme = PREPEND_SCHEMES[scheme_name]
    metaspace_split = bool(meta["split"]) if "split" in meta else True

    # --- specials: read from added_tokens, cross-checked against the vocab --
    added = {}
    for entry in tok.get("added_tokens") or []:
        if isinstance(entry, dict) and "content" in entry:
            added[entry["content"]] = entry
    ids: dict[str, int] = {}
    for name in EXPECTED_SPECIALS:
        entry = added.get(name)
        if entry is None:
            _refuse(f"special token {name!r} missing from added_tokens")
        if not bool(entry.get("special")):
            _refuse(f"{name!r} present but not marked special")
        if bool(entry.get("normalized")):
            _refuse(f"{name!r} is normalized:true -- unexpected for XLM-R")
        tid = entry.get("id")
        if not isinstance(tid, int) or isinstance(tid, bool):
            _refuse(f"added token id for {name!r} is not an integer: {tid!r}")
        ids[name] = tid
        # Every special must also be a vocab row at the same id, or the
        # runtime's id->piece table would disagree with added_tokens' ids.
        row = pieces[tid] if 0 <= tid < vocab_size else "<out of range>"
        if row != name:
            _refuse(f"added token {name!r} id {tid} does not match vocab row "
                    f"{row!r}")
    if ids["<unk>"] != unk_id:
        _refuse(f"added <unk> id {ids['<unk>']} != model unk_id {unk_id}")

    # --- post_processor: single-sequence template must be <s> A </s> --------
    post = tok.get("post_processor")
    if not isinstance(post, dict) or post.get("type") != "TemplateProcessing":
        _refuse("expected TemplateProcessing")

    def _is_step(step, kind: str, ident: str) -> bool:
        if not isinstance(step, dict) or len(step) != 1 or kind not in step:
            return False
        inner = step[kind]
        if not isinstance(inner, dict) or len(inner) != 2:
            return False
        return inner.get("id") == ident and inner.get("type_id") == 0

    single = post.get("single")
    if not (isinstance(single, list) and len(single) == 3
            and _is_step(single[0], "SpecialToken", "<s>")
            and _is_step(single[1], "Sequence", "A")
            and _is_step(single[2], "SpecialToken", "</s>")):
        _refuse("expected single template [<s>, A, </s>] -- the fixed "
                "add_bos/add_eos flags below would lie")
    special_tokens = post.get("special_tokens")
    for name in ("<s>", "</s>"):
        entry = (special_tokens or {}).get(name)
        got = entry.get("ids") if isinstance(entry, dict) else None
        if got != [ids[name]]:
            _refuse(f"post_processor maps {name!r} to {got!r}, not "
                    f"added_tokens' [{ids[name]}]")
    # add_bos / add_eos are fixed by the template check above: a tokenizer whose
    # single-sequence template is not exactly [<s>, A, </s>] was refused there,
    # so these are the two flags that template means and not a guess.

    if tok.get("truncation") is not None or tok.get("padding") is not None:
        _refuse(
            "tokenizer.json carries baked-in truncation/padding -- this "
            "project refuses silent truncation (tasks/0110); re-check the "
            "checkpoint")

    # Header limits only: a piece that does not fit its u16 length is refused by
    # `_str16` when it is written, with the same message.
    max_piece_bytes = max((len(p.encode("utf-8")) for p in pieces), default=0)
    max_piece_chars = max((len(p) for p in pieces), default=0)

    # --- write the binary table --------------------------------------------
    out = bytearray()
    out += MAGIC
    out += _u32(VERSION)
    out += _u32(vocab_size)
    out += _u32(unk_id)
    out += _u32(ids["<s>"])
    out += _u32(ids["</s>"])
    out += _u32(ids["<pad>"])
    out += _u32(ids["<mask>"])
    out += _u32(1)  # add_bos
    out += _u32(1)  # add_eos
    out += _u32(ord(METASPACE))
    out += _u32(prepend_scheme)
    out += _u32(1 if metaspace_split else 0)
    out += _u32(max_piece_bytes)
    out += _u32(max_piece_chars)
    out += _f64(min_score)
    out += _f64(unk_score)
    out += _u32(len(trie))
    out += _u32(norm_size)
    out += trie
    out += normalized
    for score in scores:
        out += _f64(score)
    for piece in pieces:
        out += _str16(piece)
    return bytes(out)


def write_table(tokenizer_json: Path, out: Path | None = None) -> Path:
    """Build the blob, write it, and say where it went.

    The default destination is `xlmr_tokenizer.bin` beside the tokenizer.json,
    because that is the name and place `pack_gte` reads (and the name the
    container builder documents), so a caller never has to know the convention.
    """
    tokenizer_json = Path(tokenizer_json)
    blob = generate(tokenizer_json)
    dest = Path(out) if out is not None else tokenizer_json.parent / "xlmr_tokenizer.bin"
    dest.write_bytes(blob)
    print(f"wrote {dest} ({len(blob)} bytes, "
          f"{struct.unpack_from('<I', blob, VOCAB_SIZE_AT)[0]} pieces)")
    return dest


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print("usage: xlmr_tokenizer_table.py <tokenizer.json> [out.bin]",
              file=sys.stderr)
        return 2
    write_table(Path(argv[1]), Path(argv[2]) if len(argv) == 3 else None)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
