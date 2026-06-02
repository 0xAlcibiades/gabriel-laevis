//! SFT (Supervised Fine Tuning)
//!
//! Instruction-tune the pretrained model with the template and response-masked
//! loss with a low constant learning rate.

use crate::config::ModelConfig;
use crate::constants::{NUM_WORKERS, SFT_FILE, SFT_REPO, SFT_SEQ_LEN, SHUFFLE_SEED};
use crate::data::{SftBatcher, SftDataset, build_sft_rows, load_sft_threads, split_valid};
use crate::train::TrainingContext;
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::transform::SamplerDataset;
use burn::optim::AdamConfig;
use burn::record::CompactRecorder;
use burn::train::metric::LossMetric;
use burn::train::{Learner, SupervisedTraining};
use eyre::{Result, WrapErr};

const BATCH_SIZE: usize = 8;
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

    let model = ctx.load_model(&cfg, from.unwrap_or("model"))?;

    println!("loading {SFT_REPO} (up to {max_examples} examples)...");

    let threads = load_sft_threads(SFT_REPO, SFT_FILE, max_examples)?;

    // Render each thread into our template; one response-masked row per model turn.
    let mut rows = Vec::new();
    for conv in &threads {
        rows.extend(build_sft_rows(&ctx.tokenizer, conv, SFT_SEQ_LEN)?);
    }

    let (rows, valid_rows) = split_valid(rows, VALID_ROWS);

    // Slice into shared  checkpoint-epochs with auto-resume.
    let sched = crate::train::schedule(rows.len(), BATCH_SIZE, epochs);

    println!(
        "training {} steps as {} x {}-step checkpoints",
        sched.total_steps,
        sched.ckpt_epochs,
        crate::train::STEPS_PER_CHECKPOINT
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

    let sft_dir = ctx.checkpoint_path("sft");

    let training = SupervisedTraining::new(sft_dir.clone(), train_loader, valid_loader)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(sched.ckpt_epochs)
        .summary();

    // Auto-resume from the latest checkpoint in the stage dir unless `--from` was given.
    let training = match ctx.resume_epoch(from, &sft_dir) {
        Some(e) => {
            println!("resuming sft from checkpoint epoch {e}");
            training.checkpoint(e)
        }
        None => training,
    };

    let result = training.launch(Learner::new(model, AdamConfig::new().init(), LR));

    ctx.save_model(result.model, "model_sft")?;

    println!("saved SFT model to {:?}", ctx.checkpoint_path("model_sft"));

    Ok(())
}
