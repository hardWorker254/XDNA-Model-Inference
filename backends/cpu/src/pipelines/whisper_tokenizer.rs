//! Whisper's tokenizer: GPT-2's byte-level BPE, packed into one blob.
//!
//! # The table, not the checkpoint
//!
//! Whisper's vocabulary is NOT in `vocab.json` alone. `vocab.json` is
//! GPT-2's 50258 entries and nothing else; every control token that makes
//! a transcription possible -- `<|startoftranscript|>`, `<|notimestamps|>`,
//! `<|transcribe|>`, `<|endoftext|>`, and the ~1600 timestamp and language
//! tokens -- lives in `added_tokens.json` with ids from 50258 up. Both are
//! packed into ONE binary table at container build time, and this is its
//! only reader.
//!
//! The blob's layout, from `tools/lib/whisper_bpe.py`:
//!
//! ```text
//! 0   4   magic "WBP1"
//! 4   4   version (1)
//! 8   4   n_vocab
//! 12  4   n_merges
//! 16  4   pool_bytes
//! 24  ..  the pool: every token's bytes, concatenated
//! ..  ..  the offsets: n_vocab u32
//! ..  ..  the lengths: n_vocab u32
//! ..  ..  the ids: n_vocab u32
//! ..  ..  the merges: n_merges pairs of u32
//! ```
//!
//! A wrong magic or a version this build does not know is a refusal and
//! not a guess: guessing at the layout is how a transcript comes out as
//! plausible garbage.
//!
//! # Only the decode direction
//!
//! `decode` is the direction an STT container needs: it is handed ids and
//! owes the caller a string. The encode direction -- the GPT-2 pretokenizer
//! with its `\p{L}` classes and its contraction list -- is not ported, and
//! nothing in the transcription path calls it.

use std::collections::HashMap;

/// The magic of a table this build reads.
const MAGIC: &[u8; 4] = b"WBP1";

/// The version of a table this build reads.
const VERSION: u32 = 1;

/// The header's size in bytes.
const HEADER_SIZE: usize = 24;

/// The alphabet's codepoints top out just above U+0130, so a flat array
/// beats a map: the decode side indexes it with the codepoint it decoded.
const MAX_BYTE_CP: usize = 0x200;

/// A Whisper tokenizer, read out of a container.
pub struct Tokenizer {
    /// Every token's bytes, in entry order.
    tokens: Vec<Vec<u8>>,
    /// The id of every entry, dense by construction.
    ids: Vec<i32>,
    /// The entry of every id, for the name lookup.
    index: HashMap<Vec<u8>, usize>,
    /// Codepoint to byte, -1 when the codepoint is not in the alphabet.
    byte_decoder: [i32; MAX_BYTE_CP],
}

impl Tokenizer {
    /// Read the tokenizer out of a container's `tokenizer.whisper_table`.
    pub fn from_table(table: &[u8]) -> Result<Tokenizer, String> {
        if table.len() < HEADER_SIZE || &table[0..4] != MAGIC {
            return Err("tokenizer.whisper_table: not a Whisper tokenizer table (magic)".into());
        }
        let version = rd_u32(table, 4);
        if version != VERSION {
            return Err(format!(
                "tokenizer.whisper_table: version {version}, this build reads {VERSION}"
            ));
        }
        let n_vocab = rd_u32(table, 8) as usize;
        let n_merges = rd_u32(table, 12) as usize;
        let pool_bytes = rd_u32(table, 16) as usize;

        // The size the header claims must fit the blob. Every field below is
        // a consequence of these offsets, so this one check is what makes the
        // rest safe to read.
        let need = HEADER_SIZE + pool_bytes + 12 * n_vocab + 8 * n_merges;
        if need > table.len() {
            return Err(format!(
                "tokenizer.whisper_table: header describes {need} bytes, tensor holds {}",
                table.len()
            ));
        }

        let pool = &table[HEADER_SIZE..HEADER_SIZE + pool_bytes];
        let offs = HEADER_SIZE + pool_bytes;
        let lens = offs + 4 * n_vocab;
        let ids_at = lens + 4 * n_vocab;
        let merges = ids_at + 4 * n_vocab;

        let mut tokens = Vec::with_capacity(n_vocab);
        let mut ids = Vec::with_capacity(n_vocab);
        let mut index: HashMap<Vec<u8>, usize> = HashMap::with_capacity(n_vocab);
        for i in 0..n_vocab {
            let o = rd_u32(table, offs + 4 * i) as usize;
            let l = rd_u32(table, lens + 4 * i) as usize;
            if o + l > pool_bytes {
                return Err(format!(
                    "tokenizer.whisper_table: entry {i} runs past the pool"
                ));
            }
            let token = pool[o..o + l].to_vec();
            let id = rd_u32(table, ids_at + 4 * i) as i32;
            index.insert(token.clone(), i);
            tokens.push(token);
            ids.push(id);
        }
        if index.len() != n_vocab {
            return Err(format!(
                "tokenizer.whisper_table: {} duplicate token strings; the table cannot represent this vocabulary",
                n_vocab - index.len()
            ));
        }
        for r in 0..n_merges {
            let a = rd_u32(table, merges + 8 * r) as usize;
            let b = rd_u32(table, merges + 8 * r + 4) as usize;
            if a >= n_vocab || b >= n_vocab {
                return Err(format!(
                    "tokenizer.whisper_table: merge {r} names a missing entry"
                ));
            }
        }

        // GPT-2's byte to printable-codepoint map, built the way
        // `whisper_bpe.bytes_to_unicode` builds it: the bytes that are
        // already printable map to themselves, the rest are pushed above
        // 0xFF so no two bytes collide.
        let mut bs: Vec<i32> = Vec::with_capacity(256);
        for b in b'!'..=b'~' {
            bs.push(b as i32);
        }
        for b in 0xA1..=0xAC {
            bs.push(b);
        }
        for b in 0xAE..=0xFF {
            bs.push(b);
        }
        let mut cs = bs.clone();
        let mut n = 0;
        for b in 0..256i32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let mut byte_decoder = [-1i32; MAX_BYTE_CP];
        for i in 0..256 {
            byte_decoder[cs[i] as usize] = bs[i];
        }

        Ok(Tokenizer {
            tokens,
            ids,
            index,
            byte_decoder,
        })
    }

    /// The vocabulary's size, in entries.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// A control token's id, looked up BY NAME.
    ///
    /// Refused when absent: a Whisper that cannot emit `<|endoftext|>` has
    /// no stopping condition, and one that cannot find `<|notimestamps|>`
    /// produces timestamp soup -- so the id is not invented here, because a
    /// wrong control token is a wrong transcript rather than an error.
    pub fn token_id(&self, token: &str) -> Result<i32, String> {
        match self.index.get(token.as_bytes()) {
            Some(entry) => Ok(self.ids[*entry]),
            None => Err(format!(
                "this checkpoint's tokenizer has no {token}. A Whisper container without it cannot start or finish a transcription, and the id is not invented here because a wrong control token is a wrong transcript rather than an error."
            )),
        }
    }

    /// Ids to text, with the byte-level alphabet undone.
    ///
    /// The result is BYTES and not a `String`: the byte-level alphabet can
    /// decode to a byte sequence that is not valid UTF-8, and a decoder that
    /// repaired it would be a decoder that silently changed the transcript.
    pub fn decode(&self, ids: &[i32]) -> Result<Vec<u8>, String> {
        // The ids are dense by construction -- the packer refuses a
        // vocabulary with a gap, because the decoder's logit indexing assumes
        // one -- so the id IS the entry index.
        let mut raw = Vec::new();
        for &id in ids {
            if id < 0 || id as usize >= self.ids.len() {
                return Err(format!("decode: id {id} is outside the vocabulary"));
            }
            let token = &self.tokens[id as usize];
            let mut i = 0;
            while i < token.len() {
                let (cp, len) = next_cp(token, i);
                if cp >= MAX_BYTE_CP as u32 || self.byte_decoder[cp as usize] < 0 {
                    return Err(format!(
                        "tokenizer.whisper_table: token '{}' contains a character outside the byte-level alphabet",
                        String::from_utf8_lossy(token)
                    ));
                }
                raw.push(self.byte_decoder[cp as usize] as u8);
                i += len;
            }
        }
        Ok(raw)
    }
}

/// One code point of the input, decoded from UTF-8, and the byte count it
/// consumed.
///
/// An invalid byte decodes as U+FFFD with len 1: it lands outside the
/// byte-level alphabet, which is what a decoder that emitted a stray byte
/// wants.
fn next_cp(s: &[u8], i: usize) -> (u32, usize) {
    let b0 = s[i];
    let left = s.len() - i;
    if b0 < 0x80 {
        return (b0 as u32, 1);
    }
    let is_cont = |c: u8| (c & 0xC0) == 0x80;
    if (b0 & 0xE0) == 0xC0 && left >= 2 && is_cont(s[i + 1]) {
        return ((((b0 & 0x1F) as u32) << 6) | ((s[i + 1] & 0x3F) as u32), 2);
    }
    if (b0 & 0xF0) == 0xE0 && left >= 3 && is_cont(s[i + 1]) && is_cont(s[i + 2]) {
        return (
            (((b0 & 0x0F) as u32) << 12)
                | (((s[i + 1] & 0x3F) as u32) << 6)
                | ((s[i + 2] & 0x3F) as u32),
            3,
        );
    }
    if (b0 & 0xF8) == 0xF0
        && left >= 4
        && is_cont(s[i + 1])
        && is_cont(s[i + 2])
        && is_cont(s[i + 3])
    {
        return (
            (((b0 & 0x07) as u32) << 18)
                | (((s[i + 1] & 0x3F) as u32) << 12)
                | (((s[i + 2] & 0x3F) as u32) << 6)
                | ((s[i + 3] & 0x3F) as u32),
            4,
        );
    }
    (0xFFFD, 1)
}

/// A little-endian u32.
fn rd_u32(table: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([table[at], table[at + 1], table[at + 2], table[at + 3]])
}
