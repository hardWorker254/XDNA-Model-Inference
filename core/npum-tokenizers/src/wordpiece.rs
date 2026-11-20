//! The wordpiece tokenizer: BERT's.
//!
//! # What it does
//!
//! Five steps, in this order, and the order is the contract:
//!
//! 1. **Clean.** Drop the control characters, replace the whitespace with
//!    U+0020, and pad the CJK ideographs with spaces so each is its own
//!    token.
//! 2. **Split** on the whitespace.
//! 3. **Fold.** Lowercase and drop the accents -- NFD, then remove the Mn
//!    characters -- PER CHARACTER.
//! 4. **Split on punctuation**, which happens AFTER the fold and not before.
//! 5. **Wordpiece.** Greedy longest-match-first, with the `##` continuation
//!    prefix for the pieces after the first.
//!
//! # Why step 4 is after step 3 and not before
//!
//! Because the fold can turn a character into a punctuation character, and a
//! splitter that ran first would have missed it. It is a rare case and it is
//! still the order the reference implementation uses.
//!
//! # Why the whitespace is the "space" PREDICATE and not U+0020
//!
//! Because BERT's BasicTokenizer treats every codepoint Python calls a space
//! as a separator, and the whitespace that reaches step 2 is whatever
//! survived step 1. A tab and a non-breaking space are both separators -- and
//! the reference runtime's `is_space` table was 9 ranges wide for exactly that
//! reason.
//!
//! # The fold is PER CHARACTER
//!
//! Not with the context-dependent final-sigma rule. The reference
//! implementation's documented behaviour and the shipped behaviour differ
//! here: `token.lower()` in Python applies the final-sigma rule, and the
//! FAST tokenizer this checkpoint loads does not. What matters is matching
//! the tokenizer the embeddings were produced with, and only a test that
//! compares every id could have caught it.

use crate::tables::{BERT, BertTables};
use crate::{Encoding, Error, Result, Tokenizer};
use std::collections::HashMap;

/// The special tokens' ids, in the order the vocabulary lists them.
///
/// These are the BERT defaults. A vocabulary that says otherwise wins: the
/// vocabulary is the authority, and these are only what a vocabulary that
/// says nothing would get.
pub const UNKNOWN_ID: u32 = 100;
/// The separator token.
pub const SEP_ID: u32 = 101;
/// The padding token.
pub const PAD_ID: u32 = 0;

/// The wordpiece tokenizer.
#[derive(Debug, Clone)]
pub struct WordPiece {
    /// The token to id map.
    vocab: HashMap<String, u32>,
    /// The id to token list, for the decode.
    tokens: Vec<String>,
    /// BERT's tables.
    tables: &'static BertTables,
    /// The id of `[CLS]`.
    cls_id: u32,
    /// The id of `[SEP]`.
    sep_id: u32,
    /// The id of `[PAD]`.
    pad_id: u32,
    /// The id of `[UNK]`.
    unk: u32,
    /// The added tokens, which are matched LITERALLY in the raw text.
    ///
    /// Longest first, because `[CLS]` and `[CLS]` in a longer string have to
    /// resolve to the same answer, and a shorter special found first would
    /// win over a longer one that starts at the same place.
    specials: Vec<String>,
    /// A word longer than this is `[UNK]` without even trying.
    ///
    /// It keeps a pathological input O(1), and it is what the reference
    /// implementation does.
    max_chars_per_word: usize,
}

impl WordPiece {
    /// A tokenizer over a vocabulary.
    pub fn new(vocab: HashMap<String, u32>) -> Self {
        let mut tokens: Vec<String> = vec![String::new(); vocab.len()];
        for (token, id) in &vocab {
            if let Some(slot) = tokens.get_mut(*id as usize) {
                *slot = token.clone();
            }
        }
        let cls_id = vocab.get("[CLS]").copied().unwrap_or(SEP_ID);
        let sep_id = vocab.get("[SEP]").copied().unwrap_or(SEP_ID);
        let pad_id = vocab.get("[PAD]").copied().unwrap_or(PAD_ID);
        let unk = vocab.get("[UNK]").copied().unwrap_or(UNKNOWN_ID);
        let mut specials: Vec<String> = ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"]
            .iter()
            .filter(|s| vocab.contains_key(**s))
            .map(|s| s.to_string())
            .collect();
        specials.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        Self {
            vocab,
            tokens,
            tables: &BERT,
            cls_id,
            sep_id,
            pad_id,
            unk,
            specials,
            max_chars_per_word: 100,
        }
    }

    /// A tokenizer over a vocabulary blob: one token per line, the id being
    /// the line number.
    ///
    /// This is the shape the container stores it in, and it is the shape the
    /// reference vocabulary file has. A trailing newline at the end of the
    /// file does NOT create an empty token -- an empty token would be an id
    /// that no text could ever produce, and a vocabulary with one is a
    /// vocabulary with a hole in it.
    pub fn from_lines(blob: &str) -> Result<Self> {
        let mut vocab = HashMap::new();
        for (id, line) in blob.lines().enumerate() {
            let line = line.strip_suffix('\r').unwrap_or(line);
            if line.is_empty() {
                continue;
            }
            vocab.entry(line.to_string()).or_insert(id as u32);
        }
        if vocab.is_empty() {
            return Err(Error::NoTable);
        }
        Ok(Self::new(vocab))
    }

    /// The id of `[CLS]`.
    pub fn cls_id(&self) -> u32 {
        self.cls_id
    }

    /// The id of `[SEP]`.
    pub fn sep_id(&self) -> u32 {
        self.sep_id
    }

    /// The id of `[PAD]`.
    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    /// The id of `[UNK]`.
    pub fn unk_id(&self) -> u32 {
        self.unk
    }

    /// The token for an id, if the vocabulary has one.
    pub fn token_of(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    /// Encode with the special tokens, the truncation and the padding.
    ///
    /// This is what the runtime's `encode` does, and it is a DIFFERENT thing
    /// from the trait's `encode`, which is the raw encoding. The trait's
    /// contract is that the caller decides what to pad and what to truncate
    /// to; this method is the one decision, made once, that everybody who
    /// wants the reference behaviour wants.
    ///
    /// The semantics, exactly:
    ///
    /// * the room is `max_len - 2`, because `[CLS]` and `[SEP]` are not
    ///   optional;
    /// * the truncation is measured BEFORE the cap, so the caller learns the
    ///   size of the input it gave rather than the size of the buffer it hit;
    /// * the padding is `[PAD]` up to `max_len`, with the mask 0 there.
    ///
    /// The returned flag says whether anything was cut.
    pub fn encode_padded(&self, text: &str, max_len: usize) -> Result<(Encoding, bool)> {
        if max_len < 2 {
            // [CLS] and [SEP] are mandatory; a shorter buffer cannot hold a
            // valid encoding, and silently returning two tokens would break
            // the length contract for every caller.
            return Err(Error::TooLong {
                tokens: 2,
                limit: max_len,
            });
        }
        let toks = self.tokenize(text);
        let room = max_len.saturating_sub(2);
        let take = toks.len().min(room);
        let truncated = take < toks.len();

        let mut ids = Vec::with_capacity(max_len);
        let mut mask = Vec::with_capacity(max_len);
        let mut types = Vec::with_capacity(max_len);

        ids.push(self.cls_id);
        mask.push(1);
        types.push(0);
        for token in toks.iter().take(take) {
            ids.push(self.id_of(token));
            mask.push(1);
            types.push(0);
        }
        ids.push(self.sep_id);
        mask.push(1);
        types.push(0);
        while ids.len() < max_len {
            ids.push(self.pad_id);
            mask.push(0);
            types.push(0);
        }

        Ok((Encoding { ids, mask, types }, truncated))
    }

    /// The id of a token, or `[UNK]`'s.
    fn id_of(&self, token: &str) -> u32 {
        self.vocab.get(token).copied().unwrap_or(self.unk)
    }

    /// Step 1: clean the text.
    ///
    /// The control characters go, the whitespace becomes U+0020, and the CJK
    /// ideographs are padded with spaces so that each is a word of its own.
    fn clean(&self, text: &str) -> Vec<char> {
        let mut out = Vec::with_capacity(text.len());
        for ch in text.chars() {
            let cp = ch as u32;
            if cp == 0 || cp == 0xFFFD || self.tables.is_control(cp) {
                continue;
            }
            if self.tables.is_space(cp) {
                out.push(' ');
                continue;
            }
            if is_cjk(cp) {
                out.push(' ');
                out.push(ch);
                out.push(' ');
                continue;
            }
            out.push(ch);
        }
        out
    }

    /// The whole pipeline, without the special tokens.
    ///
    /// The added-token pre-pass comes first: a special token is matched
    /// LITERALLY in the raw text and passed through whole, because without
    /// that "[CLS]" in a user string tokenizes to '[', 'cl', '##s', ']'.
    fn tokenize(&self, text: &str) -> Vec<String> {
        if self.specials.is_empty() {
            return self.tokenize_plain(text);
        }
        let mut out = Vec::new();
        let mut rest = text;
        loop {
            let mut best: Option<(usize, &String)> = None;
            for special in &self.specials {
                if let Some(at) = rest.find(special.as_str()) {
                    let better = match best {
                        None => true,
                        Some((best_at, best_which)) => {
                            at < best_at || (at == best_at && special.len() > best_which.len())
                        }
                    };
                    if better {
                        best = Some((at, special));
                    }
                }
            }
            let Some((at, special)) = best else {
                break;
            };
            let (head, tail) = rest.split_at(at);
            out.extend(self.tokenize_plain(head));
            out.push(special.clone());
            rest = &tail[special.len()..];
        }
        out.extend(self.tokenize_plain(rest));
        out
    }

    /// The pipeline for text with no special token in it.
    fn tokenize_plain(&self, text: &str) -> Vec<String> {
        let clean = self.clean(text);
        let mut out = Vec::new();
        let mut word: Vec<char> = Vec::new();
        for ch in clean {
            if ch == ' ' {
                self.flush_word(&mut word, &mut out);
            } else {
                word.push(ch);
            }
        }
        self.flush_word(&mut word, &mut out);
        out
    }

    /// One whitespace-separated word: fold it, split it on the punctuation,
    /// and wordpiece each piece.
    ///
    /// The word is EMPTIED, and that is not a detail: the caller keeps one
    /// buffer for the whole line and hands it over at every space, so a
    /// flush that did not clear it would emit the whole line so far again
    /// at every space.
    fn flush_word(&self, word: &mut Vec<char>, out: &mut Vec<String>) {
        if word.is_empty() {
            return;
        }
        let folded: Vec<char> = word
            .iter()
            .flat_map(|ch| {
                let cp = *ch as u32;
                match self.tables.fold(cp) {
                    // A fold that yields more than one codepoint cannot be
                    // one `char`: a capital ligature decomposes. The whole
                    // sequence is kept, because dropping the rest would
                    // change the word.
                    Some(folded) => folded
                        .iter()
                        .filter_map(|c| char::from_u32(*c))
                        .collect::<Vec<_>>(),
                    None => vec![*ch],
                }
            })
            .collect();

        // The punctuation split, which happens AFTER the fold.
        let mut pieces: Vec<Vec<char>> = Vec::new();
        let mut current: Vec<char> = Vec::new();
        for ch in folded {
            if self.tables.is_punct(ch as u32) {
                if !current.is_empty() {
                    pieces.push(std::mem::take(&mut current));
                }
                pieces.push(vec![ch]);
            } else {
                current.push(ch);
            }
        }
        if !current.is_empty() {
            pieces.push(current);
        }

        for piece in pieces {
            if piece.is_empty() {
                continue;
            }
            // Longer than the limit is `[UNK]` without even trying.
            if piece.len() > self.max_chars_per_word {
                out.push("[UNK]".to_string());
                continue;
            }
            self.encode_piece(&piece, out);
        }
        word.clear();
    }

    /// Step 5: the greedy longest-match-first, for one piece.
    fn encode_piece(&self, piece: &[char], out: &mut Vec<String>) {
        let mut start = 0usize;
        while start < piece.len() {
            let mut end = piece.len();
            let mut found = None;
            while start < end {
                let mut text = String::new();
                if start > 0 {
                    text.push_str("##");
                }
                text.extend(&piece[start..end]);
                if self.vocab.contains_key(&text) {
                    found = Some((end, text));
                    break;
                }
                end -= 1;
            }
            match found {
                Some((next_end, text)) => {
                    out.push(text);
                    start = next_end;
                }
                None => {
                    // The word cannot be spelled out: the unknown token, and
                    // the REST OF THE WORD IS DROPPED rather than retried.
                    // That is BasicTokenizer's behaviour, and it is why an
                    // unknown word produces ONE token and not one per
                    // character.
                    out.push("[UNK]".to_string());
                    return;
                }
            }
        }
    }
}

/// Whether the codepoint is a CJK ideograph.
///
/// The eight ranges the reference implementation uses. Hiragana, Katakana
/// and Hangul are NOT in them, and that is deliberate: they are not
/// ideographs, and a token per character for them would be a different
/// tokenizer.
fn is_cjk(cp: u32) -> bool {
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B820..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

impl Tokenizer for WordPiece {
    fn name(&self) -> &'static str {
        "wordpiece"
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// The RAW encoding: no special tokens, no padding, no truncation.
    ///
    /// The trait's contract, and it is a contract: a tokenizer that made
    /// those decisions for the caller would make them differently for
    /// different callers. `encode_padded` is the other thing, and it is
    /// named for what it does.
    fn encode(&self, text: &str) -> Result<Encoding> {
        let toks = self.tokenize(text);
        let ids: Vec<u32> = toks.iter().map(|t| self.id_of(t)).collect();
        let mask = vec![1u32; ids.len()];
        let types = vec![0u32; ids.len()];
        Ok(Encoding { ids, mask, types })
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut out = String::new();
        for id in ids {
            let Some(piece) = self.token_of(*id) else {
                return Err(Error::UnknownToken { id: *id });
            };
            if let Some(rest) = piece.strip_prefix("##") {
                out.push_str(rest);
            } else {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(piece);
            }
        }
        Ok(out)
    }
}
