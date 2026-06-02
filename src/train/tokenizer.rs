//! Tokenizer training
//!
//! Trains a custom BPE tokenizer that is right-sized for the model on the actual
//! pretraining dataset.
//!
//! Refs:
//!
//! - arXiv:1508.07909
//! - GPT-2 byte-level BPE + regex pre-tokenization (Radford et al. 2019)
//! - OpenAI tiktoken and Karpathy's nanochat/rustbpe
//! - SuperBPE arXiv:2503.13423
//! - SmolLM2 arXiv:2502.02737
//!
//! Emits a HuggingFace `tokenizer.json` with `pre_tokenizer = Sequence([Split, ByteLevel])`,
//! the exact shape `fastokens` fuses into its fast path at inference.

use crate::chat::SpecialToken;
use crate::config;
use crate::constants::FINEWEB_REPO;
use crate::data::{collect_docs, parquet_sources, swh_sources};
use eyre::{Result, WrapErr};
use std::path::{Path, PathBuf};
use strum::IntoEnumIterator;
use tokenizers::AddedToken;
use tokenizers::Tokenizer;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::TrainerWrapper;
use tokenizers::models::bpe::{BPE, BpeTrainer};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::tokenizer::SplitDelimiterBehavior;

/// GPT-4 like split pattern, with `\p{N}{1,2}` with digit grouping tuned for a small vocab,
/// per nanochat — `{1,3}` would be wasteful in token-space at smaller scales.
const SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

/// Train a byte-level BPE on the pretraining mix and save it as `tokenizer.json`.
///
/// `out` defaults to `$MODEL_DIR/tokenizer.json`.
pub fn run(vocab_size: usize, docs_per_domain: usize, out: Option<PathBuf>) -> Result<()> {
    let rc = config::run();

    // Gather a text sample from the same sources pretraining uses
    let mut texts: Vec<String> = Vec::new();

    println!("sampling web (FineWeb) up to {docs_per_domain} docs...");
    texts.extend(collect_docs(
        &parquet_sources(FINEWEB_REPO, "main", &rc.shards, "text")?,
        docs_per_domain,
    )?);

    if !rc.math_shards.is_empty() {
        println!("sampling math up to {docs_per_domain} docs...");
        texts.extend(collect_docs(
            &parquet_sources(
                &rc.math_repo,
                &rc.math_revision,
                &rc.math_shards,
                &rc.math_text_column,
            )?,
            docs_per_domain,
        )?);
    }

    if !rc.code_shards.is_empty() {
        println!(
            "sampling code (Stack-Edu via Software Heritage) up to {} files...",
            rc.code_max_files.min(docs_per_domain)
        );
        texts.extend(collect_docs(
            &swh_sources(
                &rc.code_repo,
                &rc.code_revision,
                &rc.code_shards,
                &rc.code_blob_column,
                rc.code_max_files.min(docs_per_domain),
            )?,
            docs_per_domain,
        )?);
    }

    println!(
        "training {vocab_size}-token vocab on {} documents...",
        texts.len()
    );

    // Configure a byte-level BPE
    let bpe = BPE::builder()
        .byte_fallback(true)
        .build()
        .map_err(|e| eyre::eyre!("building bpe model: {e}"))?;
    let mut tokenizer = Tokenizer::new(bpe);

    let split = Split::new(
        SplitPattern::Regex(SPLIT_PATTERN.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| eyre::eyre!("building split pre-tokenizer: {e}"))?;
    // ByteLevel::new(add_prefix_space, trim_offsets, use_regex).
    // use_regex=false → "bulk" mode, which fastokens detects to fuse
    // byte-level into BPE.
    let byte_level = ByteLevel::new(false, true, false);
    let pre = Sequence::new(vec![split.into(), byte_level.into()]);
    tokenizer.with_pre_tokenizer(Some(pre));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));

    let bpe_trainer = BpeTrainer::builder()
        .vocab_size(vocab_size)
        .min_frequency(0)
        // `initial_alphabet` wants a std `HashSet`;
        // `ByteLevel::alphabet()` is an ahash set.
        .initial_alphabet(
            ByteLevel::alphabet()
                .into_iter()
                .collect::<std::collections::HashSet<_>>(),
        )
        .special_tokens(
            SpecialToken::iter()
                .map(|t| AddedToken::from(t.as_str(), true))
                .collect(),
        )
        .build();
    // `Tokenizer`'s model is `ModelWrapper`, so `train` wants a `TrainerWrapper`.
    let mut trainer = TrainerWrapper::BpeTrainer(bpe_trainer);

    tokenizer
        .train(&mut trainer, texts.iter().map(String::as_str))
        .map_err(|e| eyre::eyre!("training: {e}"))?;

    let out = out.unwrap_or_else(|| config::artifact_dir().join("tokenizer.json"));
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    tokenizer
        .save(&out, true)
        .map_err(|e| eyre::eyre!("saving tokenizer: {e}"))?;
    println!(
        "saved {}-token tokenizer to {}",
        tokenizer.get_vocab_size(true),
        out.display()
    );

    verify(&out)?;
    Ok(())
}

/// Loads the saved tokenizer through the inference runtime and confirms
///
/// it (a) loads, (b) round-trips, and (c) encodes a control byte
fn verify(path: &Path) -> Result<()> {
    let tok = fastokens::Tokenizer::from_file(path).wrap_err("fastokens loading the result")?;

    let sample = "fn main() { let x = 2 + 2; }  ∑ café 日本語";
    let ids = tok.encode(sample).wrap_err("encoding sample")?;
    let round = tok.decode(&ids, false).wrap_err("decoding sample")?;
    eyre::ensure!(
        round == sample,
        "round-trip mismatch:\n  in:  {sample:?}\n  out: {round:?}"
    );

    let with_control = "control \u{0006} byte";
    tok.encode(with_control)
        .wrap_err("encoding a 0x06 control byte (the SmolLM2 OOV regression)")?;

    // The turn terminator must encode to its known id (the chat stop token).
    let turn_end = tok
        .encode(SpecialToken::TurnClose.as_str())
        .wrap_err("encoding <turn|>")?;
    eyre::ensure!(
        turn_end == [crate::chat::TURN_END_ID as u32],
        "<turn|> must encode to the single id {} (chat::TURN_END_ID), got {turn_end:?}",
        crate::chat::TURN_END_ID,
    );

    println!(
        "verified via fastokens: round-trips, encodes 0x06, sample -> {} tokens",
        ids.len()
    );
    Ok(())
}
