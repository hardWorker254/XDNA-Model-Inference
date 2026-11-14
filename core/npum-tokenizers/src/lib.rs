//! The tokenizers: wordpiece, unigram, gemma and whisper-BPE.
//!
//! # One source of truth for the tables
//!
//! The C++ tree had FOUR tokenizers and the tables in THREE places: the
//! Python packer wrote them into the container, a C++ generator in the binary
//! could rebuild them, and `whisper_unicode_tables.hpp`,
//! `xlmr_unicode_tables.hpp` and `bert_unicode_tables.hpp` -- 6 034 lines of
//! baked-in tables -- sat in the include tree as a third copy.
//!
//! Two copies of a table is one too many, because they drift, and a drifted
//! table is a tokenizer that disagrees with itself: the same text tokenizes
//! differently depending on which code path asks.
//!
//! So: the packer ALWAYS puts the table in the container, and the runtime
//! NEVER generates one. The tables come from the container, and the C++
//! generators -- `gemma_tokenizer_gen.cpp`, `xlmr_tokenizer_gen.cpp` -- and
//! the hand-written JSON parser -- `json_min.cpp` -- go away, which is 1 150
//! lines of C++ and one whole class of bug.
//!
//! # Why there is no Python-compatible crate
//!
//! The tokenizers live in the container, not in a Python-compatible form.
//! A crate that reproduced Python's `regex` module exactly would be a large
//! dependency bought for a property nothing needs.

pub mod gemma;
pub mod tables;
pub mod unigram;
pub mod whisper;
pub mod wordpiece;

pub use tables::{BertTables, CpRange, Error as TableError, FoldEntry, WhisperTables, XlmrTables};

/// What a tokenizer produces.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Encoding {
    /// The token ids.
    pub ids: Vec<u32>,
    /// The attention mask: 1 for a real token, 0 for padding.
    pub mask: Vec<u32>,
    /// The token type ids: 0 for a normal token, 1 for a special one.
    pub types: Vec<u32>,
}

impl Encoding {
    /// How many tokens the encoding holds.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the encoding holds no tokens.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// What can go wrong while tokenizing.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("the text is {tokens} tokens long, and the model accepts {limit}")]
    TooLong { tokens: usize, limit: usize },

    #[error("the tokenizer has no table for this container")]
    NoTable,

    #[error("the tokenizer's table is malformed: {0}")]
    Table(#[from] TableError),

    #[error("the token {id} is not in the vocabulary")]
    UnknownToken { id: u32 },
}

/// The result type of this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// A tokenizer.
///
/// # The contract
///
/// `encode` turns text into ids. It does NOT pad, does NOT truncate and does
/// NOT add special tokens: those are the caller's decisions, and a tokenizer
/// that made them for the caller would make them differently for different
/// callers -- which is the bug this trait exists to prevent.
pub trait Tokenizer {
    /// The tokenizer's name.
    fn name(&self) -> &'static str;

    /// The vocabulary's size.
    fn vocab_size(&self) -> usize;

    /// Turn text into tokens.
    fn encode(&self, text: &str) -> Result<Encoding>;

    /// Turn tokens back into text.
    fn decode(&self, ids: &[u32]) -> Result<String>;
}
