//! Small shared helpers that don't belong to a single stage — model-agnostic, data-agnostic.

use eyre::WrapErr;

use crate::config::artifact_dir;

/// Load the byte-level BPE tokenizer from `$MODEL_DIR/tokenizer.json` (run `train tokenizer`
/// first — this errors rather than fetching anything). Used by both training and `serve`.
pub fn load_tokenizer() -> eyre::Result<fastokens::Tokenizer> {
    let path = artifact_dir().join("tokenizer.json");
    fastokens::Tokenizer::from_file(&path).wrap_err_with(|| {
        format!(
            "loading tokenizer from {} (run `train tokenizer` first)",
            path.display()
        )
    })
}
