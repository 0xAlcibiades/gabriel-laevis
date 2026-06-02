//! Model shape and runtime knobs.

use burn::config::Config;
use serde::Deserialize;
use std::path::PathBuf;

/// Architecture dimensions for one model.
#[derive(Config, Debug)]
pub struct ModelConfig {
    /// Vocabulary size, which is set at runtime from the tokenizer's `vocab_size`.
    #[config(default = 16384)]
    pub vocab_size: usize,
    /// Residual-stream width.
    #[config(default = 384)]
    pub d_model: usize,
    /// Number of stacked layers.
    #[config(default = 12)]
    pub n_layers: usize,
    /// SSM state size `N`. Must be a multiple of 4 (rotation pairs × half-RoPE).
    #[config(default = 128)]
    pub d_state: usize,
    /// Per-head channel width; `nheads = d_inner / headdim`. Must divide `d_inner`.
    #[config(default = 64)]
    pub headdim: usize,
    /// Number of B/C groups; `nheads` must be a multiple. Heads in a group share
    /// their B/C (GQA-style). 1 = all heads share one B/C.
    #[config(default = 1)]
    pub ngroups: usize,
    /// SwiGLU hidden width.
    #[config(default = 1024)]
    pub d_ff: usize,
    /// Inner expansion factor; `d_inner = expand * d_model`.
    #[config(default = 2)]
    pub expand: usize,
    /// Nominal training sequence length. An SSM has no positional encoding and a
    /// linear-in-length recurrence, so this is metadata rather than a hard
    /// context limit.
    #[config(default = 2048)]
    pub max_seq_len: usize,
    /// Std of the N(0, std) embedding initializer.
    #[config(default = 0.02)]
    pub init_std: f64,
}

impl ModelConfig {
    /// Inner width used inside the Mamba-3 block.
    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    /// Number of attention-style heads.
    pub fn nheads(&self) -> usize {
        self.d_inner() / self.headdim
    }

    /// Check the architectural shape invariants the Mamba-3 block relies on.
    pub fn validate(&self) -> eyre::Result<()> {
        let d_inner = self.d_inner();
        if self.headdim == 0 || !d_inner.is_multiple_of(self.headdim) {
            eyre::bail!(
                "d_inner ({d_inner} = expand {} * d_model {}) must be divisible by headdim {}",
                self.expand,
                self.d_model,
                self.headdim
            );
        }
        let nheads = self.nheads();
        if self.ngroups == 0 || !nheads.is_multiple_of(self.ngroups) {
            eyre::bail!(
                "nheads ({nheads}) must be divisible by ngroups {}",
                self.ngroups
            );
        }
        if !self.d_state.is_multiple_of(4) {
            eyre::bail!(
                "d_state ({}) must be a multiple of 4 (rotation pairs x half-RoPE)",
                self.d_state
            );
        }
        Ok(())
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Output, checkpoint, and token-cache directory. Set `MODEL_DIR` to override the
/// default.
///
/// In contra, dataset and downloads use the standard Hugging Face cache
/// under `HF_HOME`.
pub fn artifact_dir() -> PathBuf {
    std::env::var_os("MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/gabriel-laevis"))
}

/// One training run's configuration: model size, data shards, and the memory and
/// performance knobs. Values come from built-in defaults, then a config file named by
/// `GL_CONFIG`, then `GL_` environment variables, so a run can be resized without
/// recompiling.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RunConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub d_state: usize,
    pub headdim: usize,
    pub ngroups: usize,
    pub d_ff: usize,
    pub expand: usize,
    pub max_seq_len: usize,
    pub init_std: f64,
    pub batch: usize,
    pub chunk: usize,
    pub cache_groups: u64,
    pub shards: Vec<String>,
    pub math_repo: String,
    pub math_revision: String,
    pub math_shards: Vec<String>,
    pub math_text_column: String,
    pub code_repo: String,
    pub code_revision: String,
    pub code_shards: Vec<String>,
    pub code_blob_column: String,
    pub code_max_files: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        // Arch dims share one source of truth with ModelConfig, so the two can't drift.
        let m = ModelConfig::new();
        Self {
            d_model: m.d_model,
            n_layers: m.n_layers,
            d_state: m.d_state,
            headdim: m.headdim,
            ngroups: m.ngroups,
            d_ff: m.d_ff,
            expand: m.expand,
            max_seq_len: m.max_seq_len,
            init_std: m.init_std,
            batch: 16,
            chunk: 32,
            cache_groups: 8,
            shards: vec![crate::constants::FINEWEB_SHARD.to_string()],
            math_repo: crate::constants::FINEMATH_REPO.to_string(),
            math_revision: "refs/convert/parquet".to_string(),
            math_shards: vec![crate::constants::FINEMATH_SHARD.to_string()],
            math_text_column: "text".to_string(),
            code_repo: crate::constants::STACK_EDU_REPO.to_string(),
            code_revision: "refs/convert/parquet".to_string(),
            code_shards: vec![crate::constants::STACK_EDU_SHARD.to_string()],
            code_blob_column: "blob_id".to_string(),
            code_max_files: 4096,
        }
    }
}

impl RunConfig {
    /// The model architecture from these dims, with `vocab_size` from the tokenizer.
    pub fn model(&self, vocab_size: usize) -> ModelConfig {
        ModelConfig::new()
            .with_vocab_size(vocab_size)
            .with_d_model(self.d_model)
            .with_n_layers(self.n_layers)
            .with_d_state(self.d_state)
            .with_headdim(self.headdim)
            .with_ngroups(self.ngroups)
            .with_d_ff(self.d_ff)
            .with_expand(self.expand)
            .with_max_seq_len(self.max_seq_len)
            .with_init_std(self.init_std)
    }

    fn load() -> Self {
        let mut builder = config::Config::builder();
        if let Ok(path) = std::env::var("GL_CONFIG") {
            builder = builder.add_source(config::File::with_name(&path));
        }
        builder = builder.add_source(
            config::Environment::with_prefix("GL")
                .try_parsing(true)
                .list_separator(",")
                .with_list_parse_key("shards")
                .with_list_parse_key("math_shards")
                .with_list_parse_key("code_shards"),
        );
        // Defaults fill anything unset; a malformed source falls back to defaults.
        builder
            .build()
            .and_then(|c| c.try_deserialize::<RunConfig>())
            .unwrap_or_default()
    }
}

/// The process-wide run config, loaded once from defaults, file, then env.
pub fn run() -> &'static RunConfig {
    use std::sync::OnceLock;
    static RUN: OnceLock<RunConfig> = OnceLock::new();
    RUN.get_or_init(|| {
        let c = RunConfig::load();
        if c.chunk > 48 {
            eprintln!(
                "warning: chunk={} > 48 degrades scan/step equivalence; use 48 or less.",
                c.chunk
            );
        }
        c
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let c = ModelConfig::new();
        assert_eq!(c.d_inner(), c.expand * c.d_model);
        c.validate()
            .expect("default config must satisfy the block invariants");
        assert_eq!(ModelConfig::new().with_vocab_size(64).vocab_size, 64);
    }

    #[test]
    fn validate_rejects_bad_shapes() {
        // d_state not a multiple of 4.
        assert!(ModelConfig::new().with_d_state(30).validate().is_err());
        // d_inner not divisible by headdim (768 % 5 != 0).
        assert!(ModelConfig::new().with_headdim(5).validate().is_err());
        // nheads not divisible by ngroups (12 heads, 5 groups).
        assert!(ModelConfig::new().with_ngroups(5).validate().is_err());
    }
}
