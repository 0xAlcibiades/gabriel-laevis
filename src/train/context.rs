//! Shared context every training stage needs.

use burn::module::Module;
use burn::record::CompactRecorder;
use eyre::{Result, WrapErr};

use crate::config::ModelConfig;
use crate::model::GabrielLaevis;

/// The compute device type.
pub type Device = burn::tensor::Device<crate::Compute>;

pub struct TrainingContext {
    pub device: Device,
    pub tokenizer: fastokens::Tokenizer,
    pub artifact: String,
}

impl TrainingContext {
    /// Ensure the artifact directory exists, then load the tokenizer. Order matters:
    /// the dir must exist first so a freshly fetched tokenizer can be packaged into it.
    pub fn new() -> Result<Self> {
        let artifact = crate::config::artifact_dir();
        std::fs::create_dir_all(&artifact)
            .wrap_err_with(|| format!("creating artifact dir {artifact}"))?;
        let tokenizer = crate::data::load_tokenizer(crate::constants::TOKENIZER_REPO)
            .wrap_err("loading tokenizer")?;
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
    pub fn checkpoint_path(&self, name: &str) -> String {
        format!("{}/{name}", self.artifact)
    }

    /// Whether a named checkpoint exists. `CompactRecorder` writes `<name>.mpk`.
    pub fn has_checkpoint(&self, name: &str) -> bool {
        std::path::Path::new(&format!("{}.mpk", self.checkpoint_path(name))).exists()
    }

    /// Latest epoch with a saved checkpoint under `${MODEL_DIR}/checkpoint/`,
    /// or None if there's nothing to resume from.
    pub fn latest_checkpoint_epoch(&self, dir: &str) -> Option<usize> {
        let cp = std::path::Path::new(dir).join("checkpoint");
        std::fs::read_dir(cp)
            .ok()?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name();
                let stem = name.to_str()?.strip_prefix("model-")?;
                stem.split('.').next()?.parse::<usize>().ok()
            })
            .max()
    }

    /// A fresh model on the autodiff (training) backend.
    pub fn fresh_model(&self, cfg: &ModelConfig) -> GabrielLaevis<crate::Train> {
        GabrielLaevis::new(cfg, &self.device)
    }

    /// Load a checkpoint onto the autodiff backend (e.g. to continue from a
    /// prior stage's output).
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
