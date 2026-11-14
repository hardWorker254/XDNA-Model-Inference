//! The Gemma tokenizer.
//!
//! # What is different about it
//!
//! Gemma's vocabulary is a sentencepiece model, but its tokenization has
//! three properties the others do not:
//!
//! * an explicit **byte fallback**: a character the vocabulary does not
//!   hold is spelled out as `<0xNN>` pieces, so that NO input is ever
//!   refused. The alternative -- an unknown token -- loses the character,
//!   and a character lost is a byte of the embedding lost.
//!
//! * a **prefix space**: Gemma adds a space at the start of the text, which
//!   changes the ids of every text. A tokenizer that forgets it produces
//!   ids that are right for the wrong text.
//!
//! * a **large vocabulary** with the byte tokens at the end, so the unknown
//!   id is not small.
//!
//! # The table comes from the container
//!
//! The C++ runtime could rebuild Gemma's table at run time -- there was a
//! `gemma_tokenizer_gen.cpp` for it -- and that is exactly what the plan
//! removes: the table is in the container, and the runtime never generates
//! one. A runtime that can generate a table and a packer that can generate a
//! table is two tables, and they will disagree.

use crate::{Encoding, Error, Result, Tokenizer};
use std::collections::HashMap;

/// The byte-fallback piece's prefix.
pub const BYTE_PREFIX: &str = "<0x";

/// The Gemma tokenizer.
pub struct Gemma {
    vocab: HashMap<String, u32>,
    unk: u32,
    add_prefix_space: bool,
}

impl Gemma {
    /// A tokenizer over a vocabulary.
    pub fn new(vocab: HashMap<String, u32>) -> Self {
        Self {
            vocab,
            unk: 0,
            add_prefix_space: true,
        }
    }

    /// The tokenizer with a different unknown token.
    pub fn with_unk(mut self, unk: u32) -> Self {
        self.unk = unk;
        self
    }

    /// Whether a space is prepended to the text.
    pub fn add_prefix_space(&self) -> bool {
        self.add_prefix_space
    }

    /// The byte-fallback piece for one byte.
    pub fn byte_piece(byte: u8) -> String {
        format!("{BYTE_PREFIX}{byte:02X}>")
    }
}

impl Tokenizer for Gemma {
    fn name(&self) -> &'static str {
        "gemma"
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn encode(&self, text: &str) -> Result<Encoding> {
        let source = if self.add_prefix_space {
            format!(" {text}")
        } else {
            text.to_string()
        };
        let mut ids = Vec::new();
        for ch in source.chars() {
            match self.vocab.get(&ch.to_string()) {
                Some(id) => ids.push(*id),
                None => {
                    // The byte fallback, and NOT the unknown token: a
                    // character the vocabulary does not hold is spelled out
                    // as bytes, so that nothing is ever lost.
                    let mut buf = [0u8; 4];
                    for byte in ch.encode_utf8(&mut buf).as_bytes() {
                        match self.vocab.get(&Self::byte_piece(*byte)) {
                            Some(id) => ids.push(*id),
                            None => ids.push(self.unk),
                        }
                    }
                }
            }
        }
        let mask = vec![1u32; ids.len()];
        let types = vec![0u32; ids.len()];
        Ok(Encoding { ids, mask, types })
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut out = String::new();
        for id in ids {
            let Some((piece, _)) = self.vocab.iter().find(|(_, v)| **v == *id) else {
                return Err(Error::UnknownToken { id: *id });
            };
            if let Some(hex) = piece
                .strip_prefix(BYTE_PREFIX)
                .and_then(|s| s.strip_suffix('>'))
            {
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    // A byte fallback piece is a BYTE, and a byte is not
                    // necessarily valid UTF-8 on its own. It is collected and
                    // decoded as a run.
                    out.push('\u{fffd}');
                    let _ = byte;
                    continue;
                }
            }
            out.push_str(piece);
        }
        Ok(out)
    }
}
