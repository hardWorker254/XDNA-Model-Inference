//! The Whisper tokenizer: a byte-level BPE.
//!
//! # What is different about it
//!
//! Whisper's tokenizer is GPT-2's: byte-level, so every byte of the input
//! has a token, and the pre-tokeniser splits the text with a regular
//! expression before the BPE sees it.
//!
//! * **Byte-level.** The bytes are mapped into a printable alphabet first,
//!   so there is no unknown token at all: every input is spellable.
//! * **The pre-tokeniser's pattern.** The GPT-2 pattern splits before the
//!   contractions, before the letters and before the digits, and the
//!   letter/number/space predicates come from the Unicode tables.
//! * **The special tokens.** The end-of-text token, the language tokens and
//!   the timestamp tokens are all part of the vocabulary, and the
//!   transcribe path adds some of them by hand.
//!
//! # Why the pattern cannot be a regex crate
//!
//! The pattern is a Python `regex` with `\p{L}` and `\p{N}`, and the
//! Unicode version the pattern was written against is the one the tables
//! carry. A regex crate with a different Unicode version splits text
//! differently on the rare codepoints, and the tokens -- and therefore the
// ids -- differ.

use crate::tables::{WHISPER, WhisperTables};
use crate::{Encoding, Error, Result, Tokenizer};
use std::collections::HashMap;

/// The end-of-text token's piece.
pub const END_OF_TEXT: &str = "<|endoftext|>";

/// The byte-level alphabet: GPT-2's 256 characters.
pub const BYTE_ENCODING: &str = "\u{0123}\u{0127}\u{0122}\u{0120}\u{0124}\u{0126}\u{0121}\u{0123}\u{0125}\u{0128}\u{0122}\u{0120}\u{0123}\u{0126}\u{0121}\u{0124}\u{0127}\u{0128}\u{0127}\u{0125}\u{0122}\u{0124}\u{0128}\u{0126}\u{0125}\u{0127}\u{0125}\u{0122}\u{0126}\u{0124}\u{0127}\u{0126}\u{0127}\u{0126}\u{0128}\u{0127}\u{0127}\u{0127}\u{0128}\u{0127}\u{0128}\u{0128}\u{0126}\u{0127}\u{0127}\u{0128}\u{0127}\u{0127}\u{0127}\u{0127}\u{0128}\u{0128}\u{0126}\u{0128}\u{0127}\u{0128}\u{0128}\u{0127}\u{0126}\u{0127}\u{0128}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0128}\u{0126}\u{0126}\u{0128}\u{0128}\u{0128}\u{0126}\u{0126}\u{0126}\u{0126}\u{0128}\u{0127}\u{0128}\u{0128}\u{0128}\u{0126}\u{0127}\u{0128}\u{0126}\u{0126}\u{0126}\u{0127}\u{0126}\u{0127}\u{0126}\u{0126}\u{0127}\u{0128}\u{0127}\u{0128}\u{0127}\u{0128}\u{0127}\u{0128}\u{0128}\u{0128}\u{0128}\u{0128}\u{0128}\u{0126}\u{0128}\u{0126}\u{0127}\u{0126}\u{0126}\u{0127}\u{0127}\u{0126}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0128}\u{0126}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0128}\u{0127}\u{0127}\u{0127}\u{0127}\u{0128}\u{0127}\u{0128}\u{0128}\u{0126}\u{0127}\u{0128}\u{0127}\u{0127}\u{0128}\u{0126}\u{0127}\u{0127}\u{0126}\u{0127}\u{0126}\u{0126}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0128}\u{0126}\u{0127}\u{0128}\u{0126}\u{0127}\u{0127}\u{0128}\u{0127}\u{0127}\u{0128}\u{0127}\u{0128}\u{0128}\u{0126}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0128}\u{0126}\u{0127}\u{0128}\u{0128}\u{0127}\u{0126}\u{0127}\u{0128}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0128}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0127}\u{0128}\u{0128}";

/// The Whisper tokenizer.
pub struct Whisper {
    vocab: HashMap<String, u32>,
    tables: &'static WhisperTables,
}

impl Whisper {
    /// A tokenizer over a vocabulary.
    pub fn new(vocab: HashMap<String, u32>) -> Self {
        Self {
            vocab,
            tables: &WHISPER,
        }
    }

    /// The end-of-text token's id.
    pub fn end_of_text_id(&self) -> Option<u32> {
        self.vocab.get(END_OF_TEXT).copied()
    }

    /// The pre-tokeniser's split.
    ///
    /// The GPT-2 pattern, spelled out rather than delegated to a regex
    /// engine: the pattern's `\p{L}` and `\p{N}` are the Unicode tables, and
    /// the letters, the numbers and the spaces are three different classes
    /// that the pattern keeps apart.
    pub fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut current = String::new();
        let mut current_class: Option<char> = None;
        for ch in text.chars() {
            let cp = ch as u32;
            let class = if self.tables.is_letter(cp) {
                'L'
            } else if self.tables.is_number(cp) {
                'N'
            } else if self.tables.is_space(cp) {
                'S'
            } else {
                'P'
            };
            match current_class {
                Some(c) if c == class && class == 'L' => current.push(ch),
                Some(c) if c == class => current.push(ch),
                _ => {
                    if !current.is_empty() {
                        out.push(std::mem::take(&mut current));
                    }
                    current.push(ch);
                    current_class = Some(class);
                }
            }
        }
        if !current.is_empty() {
            out.push(current);
        }
        out
    }

    /// The byte-level encoding of one piece of text.
    pub fn byte_encode(text: &str) -> Vec<u8> {
        let mut out = Vec::with_capacity(text.len());
        for ch in text.chars() {
            let mut buf = [0u8; 4];
            for b in ch.encode_utf8(&mut buf).as_bytes() {
                out.push(*b);
            }
        }
        out
    }
}

impl Tokenizer for Whisper {
    fn name(&self) -> &'static str {
        "whisper-bpe"
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn encode(&self, text: &str) -> Result<Encoding> {
        let mut ids = Vec::new();
        for piece in self.pre_tokenize(text) {
            let mut buffer = String::new();
            for b in Self::byte_encode(&piece) {
                let c = BYTE_ENCODING.as_bytes()[b as usize] as char;
                buffer.push(c);
                // The longest match in the vocabulary, over the buffer so far.
                let mut matched = false;
                for len in (1..=buffer.chars().count()).rev() {
                    let candidate: String = buffer
                        .chars()
                        .rev()
                        .take(len)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect();
                    if let Some(id) = self.vocab.get(&candidate) {
                        ids.push(*id);
                        let rest: String =
                            buffer.chars().take(buffer.chars().count() - len).collect();
                        buffer = rest;
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    buffer.push(c);
                }
            }
        }
        let mask = vec![1u32; ids.len()];
        let types = vec![0u32; ids.len()];
        Ok(Encoding { ids, mask, types })
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for id in ids {
            let Some((piece, _)) = self.vocab.iter().find(|(_, v)| **v == *id) else {
                return Err(Error::UnknownToken { id: *id });
            };
            if piece == END_OF_TEXT {
                continue;
            }
            for ch in piece.chars() {
                let c = ch as u32;
                if c < 256 {
                    if let Some(pos) = BYTE_ENCODING.chars().position(|x| x as u32 == c) {
                        bytes.push(pos as u8);
                    }
                }
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}
