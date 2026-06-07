//! Reasoning SFT stage.
//!
//! Instruction-tune the model to emit a chain-of-thought into the `<|channel>thought` channel
//! before its answer, with a controllable on/off toggle cued by `<|think|>` in the system turn.
//! Pure SFT on R1-distilled CoT traces; the response-masked loss supervises the thought channel
//! because it falls inside the model turn, so there are no model code changes. Chains off the
//! instruction-tuned base, `model_sft`.

use crate::config::ModelConfig;
use crate::constants::{
    NUM_WORKERS, REASON_CHAT_FILE, REASON_MATH_FILE, REASON_REPO, REASON_SEQ_LEN, SHUFFLE_SEED,
};
use crate::data::{SftBatcher, SftDataset, load_reason_rows, split_valid};
use crate::train::TrainingContext;
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::transform::SamplerDataset;
use burn::optim::AdamConfig;
use burn::record::CompactRecorder;
use burn::train::metric::LossMetric;
use burn::train::{Learner, SupervisedTraining};
use eyre::{Result, WrapErr};

const BATCH_SIZE: usize = 4; // smaller than SFT's 8; reasoning rows are long
const VALID_ROWS: usize = 64;
const LR: f64 = 1.0e-5;

pub fn run(
    ctx: &TrainingContext,
    max_examples: usize,
    epochs: usize,
    from: Option<&str>,
) -> Result<()> {
    let cfg = ModelConfig::load(ctx.checkpoint_path("config.json"))
        .wrap_err("loading config (run `train pretrain` first)")?;
    cfg.validate()?;

    // Reasoning chains off the instruction-tuned base by default.
    let model = ctx.load_model(&cfg, from.unwrap_or("model_sft"))?;

    // Math and chat SFT subsets give domain-agnostic reasoning; split the budget evenly across
    // them. Reasoning-on rows render their CoT into the thought channel, reasoning-off rows are
    // direct answers; over-length or malformed rows are dropped by the loader.
    let per_subset = (max_examples / 2).max(1);
    println!("loading {REASON_REPO} (up to {per_subset} math + {per_subset} chat rows)...");
    let mut rows = load_reason_rows(
        &ctx.tokenizer,
        REASON_REPO,
        REASON_MATH_FILE,
        per_subset,
        REASON_SEQ_LEN,
    )?;
    rows.extend(load_reason_rows(
        &ctx.tokenizer,
        REASON_REPO,
        REASON_CHAT_FILE,
        per_subset,
        REASON_SEQ_LEN,
    )?);
    println!(
        "built {} reasoning rows (malformed/over-length dropped)",
        rows.len()
    );

    let (rows, valid_rows) = split_valid(rows, VALID_ROWS);

    // Slice into shared checkpoint-epochs with auto-resume (see sft/pretrain).
    let sched = crate::train::schedule(rows.len(), BATCH_SIZE, epochs);
    println!(
        "training {} steps as {} x {}-step checkpoints",
        sched.total_steps,
        sched.ckpt_epochs,
        crate::train::STEPS_PER_CHECKPOINT,
    );

    let train_loader = DataLoaderBuilder::new(SftBatcher)
        .batch_size(BATCH_SIZE)
        .shuffle(SHUFFLE_SEED)
        .num_workers(NUM_WORKERS)
        .build(SamplerDataset::new(
            SftDataset::new(rows),
            sched.epoch_samples,
        ));

    let valid_loader = DataLoaderBuilder::new(SftBatcher)
        .batch_size(BATCH_SIZE)
        .num_workers(NUM_WORKERS)
        .build(SftDataset::new(valid_rows));

    let reason_dir = ctx.checkpoint_path("reason");

    let training = SupervisedTraining::new(reason_dir.clone(), train_loader, valid_loader)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(sched.ckpt_epochs)
        .summary();

    // Auto-resume from the latest checkpoint in the stage dir unless `--from` was given.
    let training = match ctx.resume_epoch(from, &reason_dir) {
        Some(e) => {
            println!("resuming reason from checkpoint epoch {e}");
            training.checkpoint(e)
        }
        None => training,
    };

    let result = training.launch(Learner::new(model, AdamConfig::new().init(), LR));

    ctx.save_model(result.model, "model_reason")?;

    println!(
        "saved reasoning model to {:?}",
        ctx.checkpoint_path("model_reason")
    );

    Ok(())
}
