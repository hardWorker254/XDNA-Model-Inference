//! The unigram tokenizer: XLM-R's.
//!
//! # What it does
//!
//! A unigram model is a vocabulary of pieces with a likelihood each, and
//! tokenizing is the Viterbi segmentation of the string into pieces that
//! maximises the total likelihood. It is not a greedy longest-match -- and
//! that is the whole difference: a greedy unigram tokenizer produces
//! different ids from a Viterbi one on ordinary text, and the ids are what
//! the embedding is computed over.
//!
//! # The pieces
//!
//! XLM-R's vocabulary is a sentencepiece model: it holds the pieces, their
//! likelihoods, and a flag saying whether a piece may begin a word. The
//! pieces are prefixed with U+2581 (the "lower one eighth block", which
//! sentencepiece uses to mark a word boundary) and that marker is part of
//! the piece's identity, not decoration.

use crate::tables::{XLM_R, XlmrTables};
use crate::{Encoding, Error, Result, Tokenizer};
use std::collections::HashMap;

/// The sentencepiece word-boundary marker.
pub const WORD_BOUNDARY: char = '\u{2581}';

/// One piece of the vocabulary.
#[derive(Debug, Clone)]
pub struct Piece {
    /// The piece's text.
    pub text: String,
    /// The piece's log-likelihood.
    pub score: f32,
    /// Whether the piece may begin a word.
    pub begins_word: bool,
}

/// The unigram tokenizer.
pub struct Unigram {
    pieces: Vec<Piece>,
    by_text: HashMap<String, u32>,
    tables: &'static XlmrTables,
    unk: u32,
}

impl Unigram {
    /// A tokenizer over a vocabulary.
    pub fn new(pieces: Vec<Piece>) -> Self {
        let by_text = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (p.text.clone(), i as u32))
            .collect();
        Self {
            pieces,
            by_text,
            tables: &XLM_R,
            unk: 0,
        }
    }

    /// The tokenizer with a different unknown token.
    pub fn with_unk(mut self, unk: u32) -> Self {
        self.unk = unk;
        self
    }

    /// The piece at an id.
    pub fn piece(&self, id: u32) -> Option<&Piece> {
        self.pieces.get(id as usize)
    }

    /// The id of a piece, by its text.
    ///
    /// The lookup is by EXACT text, and that is what a unigram vocabulary
    /// is for: the pieces are the strings, and the probabilities decide how
    /// the string is cut. A lookup that normalised the text first would be
    /// answering a different question -- the one the normalisation step
    /// already answered, before this one is reached.
    pub fn id_of(&self, text: &str) -> Option<u32> {
        self.by_text.get(text).copied()
    }

    /// Step 1: the grapheme-aware normalisation.
    ///
    /// The classifier branches on the general category, and the tables say
    /// which: `extend` is Mn+Me, `spacing_mark` is Mc, `control` is
    /// Cc+Cf+Zl+Zp. A control character is dropped and a space becomes the
    /// word-boundary marker.
    fn normalise(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        for ch in text.chars() {
            let cp = ch as u32;
            if self.tables.is_control(cp) {
                continue;
            }
            if self.tables.is_extend(cp) || self.tables.is_spacing_mark(cp) {
                // A mark is kept: it belongs to the character it follows, and
                // dropping it would change the piece the text spells.
                out.push(ch);
                continue;
            }
            if ch.is_whitespace() {
                out.push(WORD_BOUNDARY);
                continue;
            }
            out.push(ch);
        }
        out
    }
}

impl Tokenizer for Unigram {
    fn name(&self) -> &'static str {
        "unigram"
    }

    fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    fn encode(&self, text: &str) -> Result<Encoding> {
        let normalised = self.normalise(text);
        let mut ids = Vec::new();
        // The Viterbi segmentation, over the characters of the normalised
        // string. The inner loop walks the vocabulary and looks for the
        // pieces that start at this position; the outer loop keeps the best
        // path to every position.
        let chars: Vec<char> = normalised.chars().collect();
        let n = chars.len();
        let mut best = vec![f32::NEG_INFINITY; n + 1];
        let mut back = vec![usize::MAX; n + 1];
        best[0] = 0.0;
        for i in 0..n {
            if best[i] == f32::NEG_INFINITY {
                continue;
            }
            for (id, piece) in self.pieces.iter().enumerate() {
                let piece_chars: Vec<char> = piece.text.chars().collect();
                if piece_chars.is_empty() || piece_chars.len() > n - i {
                    continue;
                }
                if chars[i..i + piece_chars.len()] != piece_chars[..] {
                    continue;
                }
                let candidate = best[i] + piece.score;
                let end = i + piece_chars.len();
                if candidate > best[end] {
                    best[end] = candidate;
                    back[end] = id;
                }
            }
        }
        if best[n] == f32::NEG_INFINITY {
            ids.push(self.unk);
        } else {
            let mut at = n;
            while at > 0 {
                let id = back[at];
                if id == usize::MAX {
                    ids.push(self.unk);
                    break;
                }
                ids.push(id as u32);
                let piece_len = self.pieces[id].text.chars().count();
                at -= piece_len;
            }
            ids.reverse();
        }
        let mask = vec![1u32; ids.len()];
        let types = vec![0u32; ids.len()];
        Ok(Encoding { ids, mask, types })
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut out = String::new();
        for id in ids {
            let Some(piece) = self.piece(*id) else {
                return Err(Error::UnknownToken { id: *id });
            };
            out.push_str(&piece.text);
        }
        Ok(out.replace(WORD_BOUNDARY, " "))
    }
}
