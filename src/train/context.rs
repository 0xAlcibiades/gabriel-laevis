//! Shared context every training stage needs.

use crate::config::ModelConfig;
use crate::model::GabrielLaevis;
use burn::module::Module;
use burn::record::CompactRecorder;
use eyre::{Result, WrapErr};
use std::path::{Path, PathBuf};

/// The compute device type.
pub type Device = burn::tensor::Device<crate::Compute>;

pub struct TrainingContext {
    pub device: Device,
    pub tokenizer: fastokens::Tokenizer,
    pub artifact: PathBuf,
}

impl TrainingContext {
    /// Ensure the artifact directory exists, then load the tokenizer.
    pub fn new() -> Result<Self> {
        let artifact = crate::config::artifact_dir();

        std::fs::create_dir_all(&artifact)
            .wrap_err_with(|| format!("creating artifact dir {}", artifact.display()))?;

        let tokenizer = crate::data::load_tokenizer().wrap_err("could not load tokenizer")?;

        Ok(Self {
            device: Default::default(),
            tokenizer,
            artifact,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    /// Absolute path to a named checkpoint inside the artifact dir.
    pub fn checkpoint_path(&self, name: &str) -> PathBuf {
        self.artifact.join(name)
    }

    /// Whether a named checkpoint exists.
    pub fn has_checkpoint(&self, name: &str) -> bool {
        self.checkpoint_path(name).with_extension("mpk").exists()
    }

    /// Latest epoch with a saved checkpoint or None if there's nothing to resume from.
    pub fn latest_checkpoint_epoch(&self, dir: impl AsRef<Path>) -> Option<usize> {
        let cp = dir.as_ref().join("checkpoint");
        std::fs::read_dir(cp)
            .ok()?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name();
                let stem = name.to_str()?.strip_prefix("model-")?;
                let (epoch_str, _) = stem.split_once('.')?;
                epoch_str.parse::<usize>().ok()
            })
            .max()
    }

    /// A fresh model on the autodiff backend.
    pub fn fresh_model(&self, cfg: &ModelConfig) -> GabrielLaevis<crate::Train> {
        GabrielLaevis::new(cfg, &self.device)
    }

    /// Load a checkpoint onto the autodiff backend.
    pub fn load_model(&self, cfg: &ModelConfig, name: &str) -> Result<GabrielLaevis<crate::Train>> {
        GabrielLaevis::new(cfg, &self.device)
            .load_file(
                self.checkpoint_path(name),
                &CompactRecorder::new(),
                &self.device,
            )
            .wrap_err_with(|| format!("loading checkpoint '{name}'"))
    }

    /// Persist a trained model under `name`.
    pub fn save_model(&self, model: GabrielLaevis<crate::Compute>, name: &str) -> Result<()> {
        model
            .save_file(self.checkpoint_path(name), &CompactRecorder::new())
            .wrap_err_with(|| format!("saving checkpoint '{name}'"))?;
        Ok(())
    }
}
