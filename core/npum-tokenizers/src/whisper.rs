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
pub const BYTE_ENCODING: &str = "\u{100}\u{101}\u{102}\u{103}\u{104}\u{105}\u{106}\u{107}\u{108}\u{109}\u{10a}\u{10b}\u{10c}\u{10d}\u{10e}\u{10f}\u{110}\u{111}\u{112}\u{113}\u{114}\u{115}\u{116}\u{117}\u{118}\u{119}\u{11a}\u{11b}\u{11c}\u{11d}\u{11e}\u{11f}\u{120}\u{21}\u{22}\u{23}\u{24}\u{25}\u{26}\u{27}\u{28}\u{29}\u{2a}\u{2b}\u{2c}\u{2d}\u{2e}\u{2f}\u{30}\u{31}\u{32}\u{33}\u{34}\u{35}\u{36}\u{37}\u{38}\u{39}\u{3a}\u{3b}\u{3c}\u{3d}\u{3e}\u{3f}\u{40}\u{41}\u{42}\u{43}\u{44}\u{45}\u{46}\u{47}\u{48}\u{49}\u{4a}\u{4b}\u{4c}\u{4d}\u{4e}\u{4f}\u{50}\u{51}\u{52}\u{53}\u{54}\u{55}\u{56}\u{57}\u{58}\u{59}\u{5a}\u{5b}\u{5c}\u{5d}\u{5e}\u{5f}\u{60}\u{61}\u{62}\u{63}\u{64}\u{65}\u{66}\u{67}\u{68}\u{69}\u{6a}\u{6b}\u{6c}\u{6d}\u{6e}\u{6f}\u{70}\u{71}\u{72}\u{73}\u{74}\u{75}\u{76}\u{77}\u{78}\u{79}\u{7a}\u{7b}\u{7c}\u{7d}\u{7e}\u{121}\u{122}\u{123}\u{124}\u{125}\u{126}\u{127}\u{128}\u{129}\u{12a}\u{12b}\u{12c}\u{12d}\u{12e}\u{12f}\u{130}\u{131}\u{132}\u{133}\u{134}\u{135}\u{136}\u{137}\u{138}\u{139}\u{13a}\u{13b}\u{13c}\u{13d}\u{13e}\u{13f}\u{140}\u{141}\u{142}\u{a1}\u{a2}\u{a3}\u{a4}\u{a5}\u{a6}\u{a7}\u{a8}\u{a9}\u{aa}\u{ab}\u{ac}\u{143}\u{ae}\u{af}\u{b0}\u{b1}\u{b2}\u{b3}\u{b4}\u{b5}\u{b6}\u{b7}\u{b8}\u{b9}\u{ba}\u{bb}\u{bc}\u{bd}\u{be}\u{bf}\u{c0}\u{c1}\u{c2}\u{c3}\u{c4}\u{c5}\u{c6}\u{c7}\u{c8}\u{c9}\u{ca}\u{cb}\u{cc}\u{cd}\u{ce}\u{cf}\u{d0}\u{d1}\u{d2}\u{d3}\u{d4}\u{d5}\u{d6}\u{d7}\u{d8}\u{d9}\u{da}\u{db}\u{dc}\u{dd}\u{de}\u{df}\u{e0}\u{e1}\u{e2}\u{e3}\u{e4}\u{e5}\u{e6}\u{e7}\u{e8}\u{e9}\u{ea}\u{eb}\u{ec}\u{ed}\u{ee}\u{ef}\u{f0}\u{f1}\u{f2}\u{f3}\u{f4}\u{f5}\u{f6}\u{f7}\u{f8}\u{f9}\u{fa}\u{fb}\u{fc}\u{fd}\u{fe}\u{ff}";

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
                // The alphabet has 256 Unicode CHARACTERS, most of them
                // outside ASCII. Indexing its UTF-8 bytes collapsed the byte
                // values onto nine symbols and made decoding impossible.
                let c = BYTE_ENCODING
                    .chars()
                    .nth(b as usize)
                    .expect("the byte alphabet has one character per byte");
                buffer.push(c);
                // The longest match in the vocabulary, over the buffer so far.
                // With no match the character STAYS in the buffer: pushing it
                // a second time here duplicated every unmapped byte.
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
                        break;
                    }
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
            // Special tokens are not text. Their characters ARE in the byte
            // alphabet (ASCII maps to itself), so they would otherwise be
            // spelled out into the decoded transcript.
            if piece.starts_with("<|") && piece.ends_with("|>") {
                continue;
            }
            for ch in piece.chars() {
                if let Some(pos) = BYTE_ENCODING.chars().position(|x| x == ch) {
                    bytes.push(pos as u8);
                }
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}
