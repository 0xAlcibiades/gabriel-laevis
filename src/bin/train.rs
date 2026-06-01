//! Training entrypoint. One stage per invocation. Run `train --help` for usage.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use eyre::Result;

use gabriel_laevis_0::train::{TrainingContext, dpo, grpo, pretrain, sft};

/// Model training.
#[derive(Parser)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    stage: Stage,
}

#[derive(Subcommand)]
enum Stage {
    /// Stage 0: train the byte-level BPE tokenizer on the pretraining mix. Run before
    /// `pretrain`; it writes the `tokenizer.json` every later stage loads.
    Tokenizer {
        /// Target vocabulary size (256 base bytes + special tokens included).
        #[arg(long, default_value_t = 16_384)]
        vocab_size: usize,
        /// Max documents sampled per domain (web / math / code).
        #[arg(long, default_value_t = 50_000)]
        docs_per_domain: usize,
        /// Output path. Default: `$MODEL_DIR/tokenizer.json`.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Pretrain on with next-token cross-entropy.
    Pretrain {
        /// Corpus budget in tokens, not context length.
        #[arg(default_value_t = 1_000_000)]
        max_tokens: usize,
        /// Passes over the corpus.
        #[arg(default_value_t = 10)]
        epochs: usize,
        /// Initial weights to start from, e.g. `checkpoint/model-8`. Starts a fresh
        /// schedule instead of auto-resuming. Default: fresh init + auto-resume.
        #[arg(long)]
        from: Option<String>,
    },
    /// Instruction-tune the pretrained model with a response-masked loss.
    Sft {
        /// Instruction pairs to load.
        #[arg(default_value_t = 10_000)]
        max_examples: usize,
        /// Passes over the data.
        #[arg(default_value_t = 2)]
        epochs: usize,
        /// Base checkpoint to start from, e.g. `checkpoint/model-8`. Default: `model`.
        #[arg(long)]
        from: Option<String>,
    },
    /// Preference-optimize the SFT model with DPO.
    Dpo {
        /// Preference triples to load.
        #[arg(default_value_t = 2000)]
        max_pairs: usize,
        /// Passes over the data.
        #[arg(default_value_t = 1)]
        epochs: usize,
        /// Base checkpoint to start from. Default: `model_sft`.
        #[arg(long)]
        from: Option<String>,
    },
    /// Run GRPO on GSM8K with a verifiable reward.
    Grpo {
        /// GSM8K prompts to load.
        #[arg(default_value_t = 1000)]
        max_prompts: usize,
        /// Passes over the data.
        #[arg(default_value_t = 1)]
        epochs: usize,
        /// Base checkpoint to start from. Default: `model_dpo` if present, else `model_sft`.
        #[arg(long)]
        from: Option<String>,
    },
    /// Run all stages in order: tokenizer, pretrain, sft, dpo, grpo. Each stage's size
    /// is independently overridable; defaults match running the stages individually.
    All {
        /// Pretrain corpus budget in tokens.
        #[arg(default_value_t = 1_000_000)]
        max_tokens: usize,
        /// Pretrain passes over the corpus.
        #[arg(default_value_t = 10)]
        epochs: usize,
        /// SFT instruction pairs.
        #[arg(long, default_value_t = 10_000)]
        sft_examples: usize,
        /// SFT passes over the data.
        #[arg(long, default_value_t = 2)]
        sft_epochs: usize,
        /// DPO preference pairs.
        #[arg(long, default_value_t = 2000)]
        dpo_pairs: usize,
        /// DPO passes over the data.
        #[arg(long, default_value_t = 1)]
        dpo_epochs: usize,
        /// GRPO prompts.
        #[arg(long, default_value_t = 1000)]
        grpo_prompts: usize,
        /// GRPO passes over the data.
        #[arg(long, default_value_t = 1)]
        grpo_epochs: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // The tokenizer stage *creates* the tokenizer, so it runs before `TrainingContext`
    // (which loads one).
    if let Stage::Tokenizer {
        vocab_size,
        docs_per_domain,
        out,
    } = &cli.stage
    {
        return gabriel_laevis_0::train::tokenizer::run(*vocab_size, *docs_per_domain, out.clone());
    }

    let ctx = TrainingContext::new()?;

    match cli.stage {
        Stage::Tokenizer { .. } => unreachable!("handled before context creation"),
        Stage::Pretrain {
            max_tokens,
            epochs,
            from,
        } => pretrain::run(&ctx, max_tokens, epochs, from.as_deref())?,
        Stage::Sft {
            max_examples,
            epochs,
            from,
        } => sft::run(&ctx, max_examples, epochs, from.as_deref())?,
        Stage::Dpo {
            max_pairs,
            epochs,
            from,
        } => dpo::run(&ctx, max_pairs, epochs, from.as_deref())?,
        Stage::Grpo {
            max_prompts,
            epochs,
            from,
        } => grpo::run(&ctx, max_prompts, epochs, from.as_deref())?,
        Stage::All {
            max_tokens,
            epochs,
            sft_examples,
            sft_epochs,
            dpo_pairs,
            dpo_epochs,
            grpo_prompts,
            grpo_epochs,
        } => {
            pretrain::run(&ctx, max_tokens, epochs, None)?;
            sft::run(&ctx, sft_examples, sft_epochs, None)?;
            dpo::run(&ctx, dpo_pairs, dpo_epochs, None)?;
            grpo::run(&ctx, grpo_prompts, grpo_epochs, None)?;
        }
    }

    Ok(())
}
