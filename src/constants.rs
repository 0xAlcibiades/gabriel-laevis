//! External data sources and run-wide defaults.

/// Web pretraining corpus.
pub const FINEWEB_REPO: &str = "HuggingFaceFW/fineweb-edu";
pub const FINEWEB_SHARD: &str = "sample/10BT/000_00000.parquet";

/// Math pretraining corpus.
pub const FINEMATH_REPO: &str = "HuggingFaceTB/finemath";
pub const FINEMATH_SHARD: &str = "finemath-3plus/train/0000.parquet";

/// Code pretraining corpus.
pub const STACK_EDU_REPO: &str = "HuggingFaceTB/stack-edu";
pub const STACK_EDU_SHARD: &str = "Python/train/0000.parquet";

/// SFT corpus.
pub const SFT_REPO: &str = "OpenAssistant/oasst_top1_2023-08-25";
pub const SFT_FILE: &str = "default/train/0000.parquet";

/// Reasoning SFT corpus
pub const REASON_REPO: &str = "nvidia/Llama-Nemotron-Post-Training-Dataset";
pub const REASON_MATH_FILE: &str = "SFT/partial-math/0000.parquet";
pub const REASON_CHAT_FILE: &str = "SFT/partial-chat/0000.parquet";

/// Dataloader defaults shared across every training stage.
pub const SHUFFLE_SEED: u64 = 42;
pub const NUM_WORKERS: usize = 2;

/// DPO preference corpus.
pub const DPO_REPO: &str = "mlabonne/orpo-dpo-mix-40k-flat";
pub const DPO_FILE: &str = "default/train/0000.parquet";

/// DPO temperature β.
pub const DPO_BETA: f64 = 0.1;

/// GRPO task.
pub const GSM8K_REPO: &str = "openai/gsm8k";
pub const GSM8K_FILE: &str = "main/train/0000.parquet";

/// GRPO group size and KL coefficient.
pub const GRPO_GROUP_SIZE: usize = 4;
pub const GRPO_KL_BETA: f64 = 0.04;

/// Post-training sequence lengths.
pub const SFT_SEQ_LEN: usize = 512;
pub const DPO_SEQ_LEN: usize = 512;

/// Reasoning SFT sequence length. CoT traces are longer than plain SFT responses; rows that
/// exceed it are dropped, not truncated.
pub const REASON_SEQ_LEN: usize = 2048;

/// GRPO prompt and sampled-completion length caps. The completion cap leaves room for a
/// chain-of-thought before the answer.
pub const GRPO_PROMPT_LEN: usize = 256;
pub const GRPO_COMPLETION_LEN: usize = 1024;
