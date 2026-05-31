//! External data sources and run-wide defaults.

/// Byte-level BPE tokenizer.
pub const TOKENIZER_REPO: &str = "HuggingFaceTB/SmolLM2-135M";

/// Pretraining corpus.
pub const FINEWEB_REPO: &str = "HuggingFaceFW/fineweb-edu";
pub const FINEWEB_SHARD: &str = "sample/10BT/000_00000.parquet";

/// SFT corpus.
pub const SFT_REPO: &str = "OpenAssistant/oasst_top1_2023-08-25";
pub const SFT_FILE: &str = "default/train/0000.parquet";

/// Dataloader defaults shared across every training stage.
pub const SHUFFLE_SEED: u64 = 42;
pub const NUM_WORKERS: usize = 2;

/// DPO preference corpus.
pub const DPO_REPO: &str = "mlabonne/orpo-dpo-mix-40k-flat";
pub const DPO_FILE: &str = "default/train/0000.parquet";

/// DPO temperature β (implicit-reward / KL strength); 0.1 is the common default.
pub const DPO_BETA: f64 = 0.1;

/// GRPO task: GSM8K grade-school math.
pub const GSM8K_REPO: &str = "openai/gsm8k";
pub const GSM8K_FILE: &str = "main/train/0000.parquet";

/// GRPO group size and KL coefficient.
pub const GRPO_GROUP_SIZE: usize = 4;
pub const GRPO_KL_BETA: f64 = 0.04;

/// Post-training sequence lengths. Pretrain uses `RunConfig::max_seq_len`.
pub const SFT_SEQ_LEN: usize = 512;
pub const DPO_SEQ_LEN: usize = 512;

/// GRPO prompt and sampled-completion length caps.
pub const GRPO_PROMPT_LEN: usize = 256;
pub const GRPO_COMPLETION_LEN: usize = 200;
