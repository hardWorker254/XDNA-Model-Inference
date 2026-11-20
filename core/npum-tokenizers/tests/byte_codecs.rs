//! Byte fallback and byte-level codecs must round-trip.
//! SPDX-License-Identifier: Apache-2.0

use npum_tokenizers::gemma::Gemma;
use npum_tokenizers::whisper::Whisper;
use npum_tokenizers::Tokenizer;
use std::collections::HashMap;

#[test]
fn gemma_decodes_a_contiguous_byte_fallback_run() {
    let mut vocab = HashMap::new();
    for byte in [0x20u8, 0xF0, 0x9F, 0x98, 0x80] {
        vocab.insert(Gemma::byte_piece(byte), 100 + byte as u32);
    }
    vocab.insert("😀".to_string(), 1);
    let tokenizer = Gemma::new(vocab).with_unk(0);
    let ids = tokenizer.encode("😀").unwrap().ids;
    assert_eq!(tokenizer.decode(&ids).unwrap(), " 😀");
}

/// GPT-2's `bytes_to_unicode`, in byte order: 256 distinct printable chars.
fn byte_alphabet() -> Vec<char> {
    let mut out = Vec::with_capacity(256);
    let mut shifted = 0u32;
    for byte in 0u32..256 {
        if (33..=126).contains(&byte) || (161..=172).contains(&byte) || (174..=255).contains(&byte) {
            out.push(char::from_u32(byte).unwrap());
        } else {
            out.push(char::from_u32(256 + shifted).unwrap());
            shifted += 1;
        }
    }
    out
}

#[test]
fn whisper_byte_alphabet_is_a_bijection_and_round_trips() {
    let alphabet = byte_alphabet();
    assert_eq!(alphabet.len(), 256);
    let mut seen = std::collections::HashSet::new();
    assert!(alphabet.iter().all(|c| seen.insert(*c)));

    let vocab = alphabet
        .iter()
        .enumerate()
        .map(|(byte, c)| (c.to_string(), byte as u32 + 1))
        .collect::<HashMap<_, _>>();
    let tokenizer = Whisper::new(vocab);
    for text in ["Hi", "hello world", "😀", "a b"] {
        let ids = tokenizer.encode(text).unwrap().ids;
        assert_eq!(tokenizer.decode(&ids).unwrap(), text, "text {text:?}");
    }
}
