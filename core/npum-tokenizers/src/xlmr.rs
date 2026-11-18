//! The XLM-R SentencePiece Unigram tokenizer.
//!
//! Line-for-line port of the reference baseline (`runtime/src/tokenizers/xlmr.cpp`)
//! and its executable spec (`tools/lib/xlmr_tokenizer_ref.py`). The blob
//! format is XLMRTOK1. Scores are f64 end-to-end.

use crate::tables::{XLM_R, XlmrTables};
use crate::{Encoding, Error, Result, Tokenizer};

/// A loaded XLM-R tokenizer.
#[derive(Clone)]
pub struct XlmrTokenizer {
    trie_units: Vec<u32>,
    normalized_blob: Vec<u8>,
    scores: Vec<f64>,
    pieces: Vec<String>,
    piece_to_id: std::collections::HashMap<String, i32>,
    _min_score: f64,
    unk_score: f64,
    metaspace_cp: u32,
    prepend_scheme: u32,
    metaspace_split: u32,
    max_piece_chars: u32,
    add_bos: u32,
    add_eos: u32,
    pub bos_id: i32,
    pub eos_id: i32,
    pub pad_id: i32,
    pub unk_id: i32,
    pub mask_id: i32,
    _tables: &'static XlmrTables,
}

fn read_u16(p: &[u8]) -> u16 {
    u16::from_le_bytes([p[0], p[1]])
}
fn read_u32_le(p: &[u8]) -> u32 {
    u32::from_le_bytes([p[0], p[1], p[2], p[3]])
}
fn read_f64_le(p: &[u8]) -> f64 {
    f64::from_le_bytes([p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]])
}

fn utf8_decode(s: &str) -> Vec<u32> {
    s.chars().map(|c| c as u32).collect()
}
fn utf8_append(s: &mut String, cp: u32) {
    if let Some(c) = char::from_u32(cp) {
        s.push(c);
    } else {
        s.push('\u{FFFD}');
    }
}
fn utf8_len(cp: u32) -> usize {
    if cp < 0x80 {
        1
    } else if cp < 0x800 {
        2
    } else if cp < 0x10000 {
        3
    } else {
        4
    }
}

fn in_ranges(cp: u32, rs: &[crate::tables::CpRange]) -> bool {
    let mut lo = 0i32;
    let mut hi = rs.len() as i32 - 1;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let r = rs[mid as usize];
        if cp < r.lo {
            hi = mid - 1;
        } else if cp > r.hi {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    false
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Gcb {
    CR,
    LF,
    ZWJ,
    Extend,
    Prepend,
    RI,
    L,
    V,
    T,
    LV,
    LVT,
    SpacingMark,
    Control,
    Other,
}

fn is_prepend(cp: u32) -> bool {
    (0x0600..=0x0605).contains(&cp)
        || cp == 0x06DD
        || cp == 0x070F
        || cp == 0x0890
        || cp == 0x0891
        || cp == 0x08E2
        || cp == 0x0D4E
        || cp == 0x110BD
        || cp == 0x110CD
}

fn gcb_class(cp: u32) -> Gcb {
    if cp == 0x000D {
        return Gcb::CR;
    }
    if cp == 0x000A {
        return Gcb::LF;
    }
    if cp == 0x200D {
        return Gcb::ZWJ;
    }
    if cp == 0x200C {
        return Gcb::Extend;
    }
    if is_prepend(cp) {
        return Gcb::Prepend;
    }
    if (0x1F1E6..=0x1F1FF).contains(&cp) {
        return Gcb::RI;
    }
    if (0x1100..=0x115F).contains(&cp) || (0xA960..=0xA97C).contains(&cp) {
        return Gcb::L;
    }
    if (0x1160..=0x11A7).contains(&cp) || (0xD7B0..=0xD7C6).contains(&cp) {
        return Gcb::V;
    }
    if (0x11A8..=0x11FF).contains(&cp) || (0xD7CB..=0xD7FB).contains(&cp) {
        return Gcb::T;
    }
    if (0xAC00..=0xD7A3).contains(&cp) {
        return if (cp - 0xAC00) % 28 == 0 {
            Gcb::LV
        } else {
            Gcb::LVT
        };
    }
    if in_ranges(cp, XLM_R.extend) {
        return Gcb::Extend;
    }
    if in_ranges(cp, XLM_R.spacing_mark) {
        return Gcb::SpacingMark;
    }
    if in_ranges(cp, XLM_R.control) {
        return Gcb::Control;
    }
    Gcb::Other
}

fn is_ext_pict(cp: u32) -> bool {
    (0x1F000..=0x1FFFD).contains(&cp)
        || (0x2600..=0x27BF).contains(&cp)
        || (0x2B00..=0x2BFF).contains(&cp)
        || cp == 0x00A9
        || cp == 0x00AE
        || cp == 0x2122
        || cp == 0x203C
        || cp == 0x2049
        || cp == 0x2139
        || (0x2190..=0x21FF).contains(&cp)
        || (0x2300..=0x23FF).contains(&cp)
        || (0x25A0..=0x25FF).contains(&cp)
        || (0x2900..=0x297F).contains(&cp)
        || (0x3030..=0x303D).contains(&cp)
        || cp == 0x3297
        || cp == 0x3299
        || (0xFE00..=0xFE0F).contains(&cp)
}

fn grapheme_starts(cps: &[u32]) -> Vec<usize> {
    let mut starts = Vec::new();
    if cps.is_empty() {
        return starts;
    }
    let cls: Vec<Gcb> = cps.iter().map(|&cp| gcb_class(cp)).collect();
    starts.push(0);
    let mut ri_run = 0i32;
    for i in 1..cps.len() {
        let left = cls[i - 1];
        let right = cls[i];
        if left == Gcb::RI {
            ri_run += 1;
        } else {
            ri_run = 0;
        }
        let brk = if left == Gcb::CR && right == Gcb::LF {
            false
        } else if left == Gcb::CR || left == Gcb::LF || left == Gcb::Control {
            true
        } else if right == Gcb::CR || right == Gcb::LF || right == Gcb::Control {
            true
        } else if left == Gcb::L
            && (right == Gcb::L || right == Gcb::V || right == Gcb::LV || right == Gcb::LVT)
        {
            false
        } else if (left == Gcb::LV || left == Gcb::V) && (right == Gcb::V || right == Gcb::T) {
            false
        } else if (left == Gcb::LVT || left == Gcb::T) && right == Gcb::T {
            false
        } else if right == Gcb::Extend || right == Gcb::ZWJ {
            false
        } else if right == Gcb::SpacingMark {
            false
        } else if left == Gcb::Prepend {
            false
        } else if left == Gcb::ZWJ && is_ext_pict(cps[i]) {
            false
        } else if left == Gcb::RI && right == Gcb::RI && ri_run % 2 == 1 {
            false
        } else {
            true
        };
        if brk {
            starts.push(i);
        }
    }
    starts.push(cps.len());
    starts
}

fn is_white_space(cp: u32) -> bool {
    cp == 0x09
        || cp == 0x0A
        || cp == 0x0B
        || cp == 0x0C
        || cp == 0x0D
        || cp == 0x20
        || cp == 0x85
        || cp == 0xA0
        || cp == 0x1680
        || (0x2000..=0x200A).contains(&cp)
        || cp == 0x2028
        || cp == 0x2029
        || cp == 0x202F
        || cp == 0x205F
        || cp == 0x3000
}

impl XlmrTokenizer {
    /// Load from an XLMRTOK1 blob.
    pub fn from_table_bytes(data: &[u8]) -> std::result::Result<Self, String> {
        let mut p = 0usize;
        let need = |p: usize, n: usize, data: &[u8]| -> std::result::Result<(), String> {
            if data.len().saturating_sub(p) < n {
                Err("tokenizer_xlmr: truncated table".to_string())
            } else {
                Ok(())
            }
        };
        need(p, 8, data)?;
        if &data[p..p + 8] != b"XLMRTOK1" {
            return Err("tokenizer_xlmr: bad magic".to_string());
        }
        p += 8;
        need(p, 14 * 4, data)?;
        let version = read_u32_le(&data[p..]);
        p += 4;
        if version != 1 {
            return Err(format!(
                "tokenizer_xlmr: unsupported table version {version}"
            ));
        }
        let vocab_size = read_u32_le(&data[p..]) as usize;
        p += 4;
        let unk_id = read_u32_le(&data[p..]) as i32;
        p += 4;
        let bos_id = read_u32_le(&data[p..]) as i32;
        p += 4;
        let eos_id = read_u32_le(&data[p..]) as i32;
        p += 4;
        let pad_id = read_u32_le(&data[p..]) as i32;
        p += 4;
        let mask_id = read_u32_le(&data[p..]) as i32;
        p += 4;
        let add_bos = read_u32_le(&data[p..]);
        p += 4;
        let add_eos = read_u32_le(&data[p..]);
        p += 4;
        let metaspace_cp = read_u32_le(&data[p..]);
        p += 4;
        let prepend_scheme = read_u32_le(&data[p..]);
        p += 4;
        let metaspace_split = read_u32_le(&data[p..]);
        p += 4;
        p += 4; // max_piece_bytes
        let max_piece_chars = read_u32_le(&data[p..]);
        p += 4;
        need(p, 16, data)?;
        let min_score = read_f64_le(&data[p..]);
        p += 8;
        let unk_score = read_f64_le(&data[p..]);
        p += 8;
        need(p, 8, data)?;
        let trie_size = read_u32_le(&data[p..]) as usize;
        p += 4;
        let norm_size = read_u32_le(&data[p..]) as usize;
        p += 4;
        if trie_size % 4 != 0 {
            return Err("tokenizer_xlmr: trie blob not a multiple of 4 bytes".to_string());
        }
        need(p, trie_size, data)?;
        let mut trie_units = vec![0u32; trie_size / 4];
        for (i, chunk) in data[p..p + trie_size].chunks_exact(4).enumerate() {
            trie_units[i] = read_u32_le(chunk);
        }
        p += trie_size;
        need(p, norm_size, data)?;
        let normalized_blob = data[p..p + norm_size].to_vec();
        p += norm_size;
        need(p, vocab_size * 8, data)?;
        let mut scores = vec![0f64; vocab_size];
        for i in 0..vocab_size {
            scores[i] = read_f64_le(&data[p..]);
            p += 8;
        }
        let mut pieces = Vec::with_capacity(vocab_size);
        let mut piece_to_id = std::collections::HashMap::with_capacity(vocab_size * 2);
        for i in 0..vocab_size {
            need(p, 2, data)?;
            let len = read_u16(&data[p..]) as usize;
            p += 2;
            need(p, len, data)?;
            let s = String::from_utf8(data[p..p + len].to_vec())
                .map_err(|_| "tokenizer_xlmr: piece is not valid utf-8".to_string())?;
            p += len;
            piece_to_id.insert(s.clone(), i as i32);
            pieces.push(s);
        }
        if p != data.len() {
            return Err(format!(
                "tokenizer_xlmr: trailing bytes: read {p} of {}",
                data.len()
            ));
        }
        if pieces.is_empty() {
            return Err("tokenizer_xlmr: empty vocabulary".to_string());
        }
        if bos_id < 0 || eos_id < 0 || pad_id < 0 || unk_id < 0 || (unk_id as usize) >= vocab_size {
            return Err("tokenizer_xlmr: missing a required special id".to_string());
        }
        if trie_units.is_empty() {
            return Err("tokenizer_xlmr: empty charsmap trie".to_string());
        }
        if normalized_blob.last().copied().unwrap_or(0) != 0 {
            return Err("tokenizer_xlmr: normalized-strings blob does not end in NUL".to_string());
        }
        if max_piece_chars == 0 {
            return Err("tokenizer_xlmr: max_piece_chars is 0".to_string());
        }
        Ok(Self {
            trie_units,
            normalized_blob,
            scores,
            pieces,
            piece_to_id,
            _min_score: min_score,
            unk_score,
            metaspace_cp,
            prepend_scheme,
            metaspace_split,
            max_piece_chars,
            add_bos,
            add_eos,
            bos_id,
            eos_id,
            pad_id,
            unk_id,
            mask_id,
            _tables: &XLM_R,
        })
    }

    /// Load from a file path.
    pub fn from_table_file(path: &std::path::Path) -> std::result::Result<Self, String> {
        let data = std::fs::read(path)
            .map_err(|e| format!("tokenizer_xlmr: cannot open {}: {e}", path.display()))?;
        Self::from_table_bytes(&data)
    }

    pub fn piece_of(&self, id: i32) -> Option<&str> {
        if id < 0 {
            return None;
        }
        self.pieces.get(id as usize).map(|s| s.as_str())
    }

    fn trie_first_match(&self, key: &[u8]) -> Option<i64> {
        let units = &self.trie_units;
        let n_units = units.len();
        if n_units == 0 {
            return None;
        }
        let mut node_pos: usize = 0;
        let mut unit = units[node_pos];
        let offset = ((unit >> 10) << ((unit & 0x200) >> 6)) as usize;
        node_pos ^= offset;
        for &b in key {
            let c = b as u32;
            node_pos ^= c as usize;
            if node_pos >= n_units {
                return None;
            }
            unit = units[node_pos];
            if (unit & 0x800000FF) != c {
                return None;
            }
            let offset = ((unit >> 10) << ((unit & 0x200) >> 6)) as usize;
            node_pos ^= offset;
            if ((unit >> 8) & 1) != 0 {
                if node_pos >= n_units {
                    return None;
                }
                return Some((units[node_pos] & 0x7FFFFFFF) as i64);
            }
        }
        None
    }

    fn transform(&self, chunk: &[u8], out: &mut String) -> bool {
        let Some(index) = self.trie_first_match(chunk) else {
            return false;
        };
        let idx = index as usize;
        if idx >= self.normalized_blob.len() {
            return false;
        }
        let rest = &self.normalized_blob[idx..];
        let nul = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
        if let Ok(s) = std::str::from_utf8(&rest[..nul]) {
            out.push_str(s);
        } else {
            // lossy fallback, should not happen
            out.push_str(&String::from_utf8_lossy(&rest[..nul]));
        }
        true
    }

    /// Stage 1: Precompiled normalizer.
    pub fn normalize(&self, text: &str) -> String {
        let cps = utf8_decode(text);
        let starts = grapheme_starts(&cps);
        let mut out = String::with_capacity(text.len());
        let mut chunk = String::new();
        let mut i = 0;
        while i + 1 < starts.len() {
            let lo = starts[i];
            let hi = starts[i + 1];
            let mut g_bytes = 0usize;
            for k in lo..hi {
                g_bytes += utf8_len(cps[k]);
            }
            if g_bytes < 6 {
                chunk.clear();
                for k in lo..hi {
                    utf8_append(&mut chunk, cps[k]);
                }
                if self.transform(chunk.as_bytes(), &mut out) {
                    i += 1;
                    continue;
                }
            }
            for k in lo..hi {
                chunk.clear();
                utf8_append(&mut chunk, cps[k]);
                if !self.transform(chunk.as_bytes(), &mut out) {
                    out.push_str(&chunk);
                }
            }
            i += 1;
        }
        out
    }

    /// Stage 2+3: WhitespaceSplit then Metaspace.
    pub fn pre_tokenize(&self, normalized: &str) -> Vec<String> {
        let cps = utf8_decode(normalized);
        let mut words: Vec<Vec<u32>> = Vec::new();
        let mut cur: Vec<u32> = Vec::new();
        for cp in cps {
            if is_white_space(cp) {
                if !cur.is_empty() {
                    words.push(cur);
                    cur = Vec::new();
                }
            } else {
                cur.push(cp);
            }
        }
        if !cur.is_empty() {
            words.push(cur);
        }
        let rep = self.metaspace_cp;
        let mut pretokens: Vec<String> = Vec::new();
        for mut word in words {
            if self.prepend_scheme == 2 && !(!word.is_empty() && word[0] == rep) {
                word.insert(0, rep);
            } else if self.prepend_scheme == 1
                && pretokens.is_empty()
                && !(!word.is_empty() && word[0] == rep)
            {
                word.insert(0, rep);
            }
            if self.metaspace_split != 0 {
                let mut cur_part: Vec<u32> = Vec::new();
                for cp in word {
                    if cp == rep && !cur_part.is_empty() {
                        let mut s = String::new();
                        for c in &cur_part {
                            utf8_append(&mut s, *c);
                        }
                        pretokens.push(s);
                        cur_part = vec![cp];
                    } else {
                        cur_part.push(cp);
                    }
                }
                if !cur_part.is_empty() {
                    let mut s = String::new();
                    for c in &cur_part {
                        utf8_append(&mut s, *c);
                    }
                    pretokens.push(s);
                }
            } else {
                let mut s = String::new();
                for c in &word {
                    utf8_append(&mut s, *c);
                }
                pretokens.push(s);
            }
        }
        pretokens
    }

    /// Stage 4: Unigram Viterbi per pretoken.
    pub fn viterbi(&self, pretoken: &str) -> Vec<i32> {
        let bytes = pretoken.as_bytes();
        let mut offs: Vec<usize> = Vec::new();
        for (i, b) in bytes.iter().enumerate() {
            if (b & 0xC0) != 0x80 {
                offs.push(i);
            }
        }
        offs.push(bytes.len());
        let n = if offs.is_empty() { 0 } else { offs.len() - 1 };
        if n == 0 {
            return Vec::new();
        }
        let neg = -1.0e300f64;
        let mut best_score = vec![neg; n + 1];
        let mut best_start = vec![-1i32; n + 1];
        let mut best_id = vec![-1i32; n + 1];
        best_score[0] = 0.0;
        let maxc = self.max_piece_chars as usize;
        for start in 0..n {
            let here = best_score[start];
            if best_start[start] == -1 && start != 0 {
                continue;
            }
            let mut has_single = false;
            let top = std::cmp::min(n, start + maxc);
            for end in (start + 1)..=top {
                let piece = &pretoken[offs[start]..offs[end]];
                if let Some(&tid) = self.piece_to_id.get(piece) {
                    let cand = here + self.scores[tid as usize];
                    if best_start[end] == -1 || cand > best_score[end] {
                        best_score[end] = cand;
                        best_start[end] = start as i32;
                        best_id[end] = tid;
                    }
                    if end - start == 1 {
                        has_single = true;
                    }
                }
            }
            if !has_single {
                let end = start + 1;
                let cand = here + self.unk_score;
                if best_start[end] == -1 || cand > best_score[end] {
                    best_score[end] = cand;
                    best_start[end] = start as i32;
                    best_id[end] = self.unk_id;
                }
            }
        }
        let mut ids: Vec<i32> = Vec::new();
        let mut pos = n;
        while pos > 0 {
            ids.push(best_id[pos]);
            pos = best_start[pos] as usize;
        }
        ids.reverse();
        let mut fused: Vec<i32> = Vec::with_capacity(ids.len());
        for tid in ids {
            if tid == self.unk_id && fused.last().copied() == Some(self.unk_id) {
                continue;
            }
            fused.push(tid);
        }
        fused
    }

    /// Full pipeline: normalize -> pre-tokenize -> Viterbi -> BOS/EOS.
    pub fn encode_raw(&self, text: &str) -> Vec<i32> {
        let mut ids: Vec<i32> = Vec::new();
        for pretoken in self.pre_tokenize(&self.normalize(text)) {
            ids.extend(self.viterbi(&pretoken));
        }
        if self.add_bos != 0 {
            ids.insert(0, self.bos_id);
        }
        if self.add_eos != 0 {
            ids.push(self.eos_id);
        }
        ids
    }

    /// Encode with truncation and padding to `max_len`, like the runtime.
    ///
    /// Returns (encoding, truncated). The encoding's arrays are `max_len`
    /// long (padded with `pad_id`, mask 0). If truncated, the last token
    /// of the kept prefix becomes `eos_id`.
    pub fn encode_padded(&self, text: &str, max_len: usize) -> Result<(Encoding, bool)> {
        let mut full = self.encode_raw(text);
        let n_tokens_full = full.len();
        let truncated = n_tokens_full > max_len;
        if truncated {
            if max_len == 0 {
                full.clear();
            } else {
                full.truncate(max_len);
                if let Some(last) = full.last_mut() {
                    *last = self.eos_id;
                }
            }
        }
        let n_tokens = full.len();
        let mut ids: Vec<u32> = full.into_iter().map(|x| x as u32).collect();
        ids.resize(max_len, self.pad_id as u32);
        let mut mask = vec![0u32; max_len];
        for m in mask.iter_mut().take(n_tokens.min(max_len)) {
            *m = 1;
        }
        let types = vec![0u32; max_len];
        Ok((Encoding { ids, mask, types }, truncated))
    }
}

impl Tokenizer for XlmrTokenizer {
    fn name(&self) -> &'static str {
        "xlmr"
    }
    fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
    fn encode(&self, text: &str) -> Result<Encoding> {
        let ids_i32 = self.encode_raw(text);
        let ids: Vec<u32> = ids_i32.into_iter().map(|x| x as u32).collect();
        let mask = vec![1u32; ids.len()];
        let types = vec![0u32; ids.len()];
        Ok(Encoding { ids, mask, types })
    }
    fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut out = String::new();
        for &id in ids {
            let Some(piece) = self.piece_of(id as i32) else {
                return Err(Error::UnknownToken { id });
            };
            // XLM-R pieces use '▁' (U+2581) as word boundary marker.
            let replaced = piece.replace('\u{2581}', " ");
            out.push_str(&replaced);
        }
        Ok(out.trim_start().to_string())
    }
}
