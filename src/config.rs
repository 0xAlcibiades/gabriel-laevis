//! Model shape and runtime knobs.

use burn::config::Config;
use serde::Deserialize;

/// Architecture dimensions for one model.
#[derive(Config, Debug)]
pub struct ModelConfig {
    /// Vocabulary size, which is set at runtime from the tokenizer's `vocab_size`.
    #[config(default = 49152)]
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
}

impl ModelConfig {
    /// Inner width used inside the Mamba-3 block.
    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    /// Number of attention-style heads (`d_inner / headdim`).
    pub fn nheads(&self) -> usize {
        self.d_inner() / self.headdim
    }

    /// Check the architectural shape invariants the Mamba-3 block relies on,
    /// returning a clean error instead of panicking deep inside a tensor reshape.
    ///
    /// These were `debug_assert!`s in `Mamba3Block::new`, but real runs build with
    /// `--profile=maxperf`, which sets `debug-assertions = false`, and the dims come
    /// from `GL_` env vars — so a fat-fingered config would otherwise blow up far
    /// from its cause (or, with overflow-checks off, worse). Call this right after
    /// constructing or loading a `ModelConfig`.
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
/// `/tmp/gabriel-laevis` default. Every command reads this. Dataset and tokenizer
/// downloads use the standard Hugging Face cache under `HF_HOME`.
pub fn artifact_dir() -> String {
    std::env::var("MODEL_DIR").unwrap_or_else(|_| "/tmp/gabriel-laevis".to_string())
}

/// One training run's configuration: model size, data shards, and the memory and
/// performance knobs. Values come from built-in defaults, then a config file named by
/// `GL_CONFIG`, then `GL_` environment variables, so a run can be resized without
/// recompiling. `vocab_size` is excluded and comes from the tokenizer at runtime.
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
    /// Pretrain batch size. Dominates LM-head memory; lower it on tight machines.
    pub batch: usize,
    /// SSD scan chunk length. Keep at most 48 for f32 rescale precision.
    pub chunk: usize,
    /// Row groups held tokenized at once by the streaming corpus cache. Memory ≈ this ×
    /// one row group's token size; raise it freely on a big box (re-tokenizing a missed
    /// group is cheap), it just bounds peak RAM so the corpus never lands whole.
    pub cache_groups: u64,
    /// FineWeb-Edu shard files, concatenated in order for pretraining.
    pub shards: Vec<String>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            d_model: 384,
            n_layers: 12,
            d_state: 128,
            headdim: 64,
            ngroups: 1,
            d_ff: 1024,
            expand: 2,
            max_seq_len: 2048,
            batch: 16,
            chunk: 32,
            cache_groups: 8,
            shards: vec![crate::constants::FINEWEB_SHARD.to_string()],
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
                .with_list_parse_key("shards"),
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
