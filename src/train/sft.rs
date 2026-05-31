//! SFT stage: instruction-tune the pretrained model with a ChatML template and
//! response-masked loss with a low constant learing rate.

use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::transform::SamplerDataset;
use burn::optim::AdamConfig;
use burn::record::CompactRecorder;
use burn::train::metric::LossMetric;
use burn::train::{Learner, SupervisedTraining};
use eyre::{Result, WrapErr};

use crate::config::ModelConfig;
use crate::constants::{NUM_WORKERS, SFT_FILE, SFT_REPO, SFT_SEQ_LEN, SHUFFLE_SEED};
use crate::data::{SftBatcher, SftDataset, build_sft_row, load_sft_pairs, split_valid};
use crate::train::TrainingContext;

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
    let pairs = load_sft_pairs(SFT_REPO, SFT_FILE, max_examples).wrap_err("loading sft data")?;
    println!("tokenizing {} instruction pairs...", pairs.len());
    // Drop rows the frozen-vocab tokenizer can't encode (a handful of OASST rows carry
    // bytes with no base token, e.g. control char 0x06) rather than aborting the run.
    let total = pairs.len();
    let rows: Vec<_> = pairs
        .iter()
        .filter_map(|(p, r)| build_sft_row(&ctx.tokenizer, p, r, SFT_SEQ_LEN).ok())
        .collect();
    if rows.len() < total {
        println!("skipped {} unencodable rows", total - rows.len());
    }
    let (rows, valid_rows) = split_valid(rows, VALID_ROWS);

    // Slice into shared 1000-step checkpoint-epochs with auto-resume (see grpo/pretrain).
    let sched = crate::train::schedule(rows.len(), BATCH_SIZE, epochs);
    println!(
        "training {} steps (~{epochs} passes over {} rows) as {} x {}-step checkpoints",
        sched.total_steps,
        rows.len(),
        sched.ckpt_epochs,
        crate::train::STEPS_PER_CHECKPOINT.min(sched.total_steps),
    );

    let train = DataLoaderBuilder::new(SftBatcher)
        .batch_size(BATCH_SIZE)
        .shuffle(SHUFFLE_SEED)
        .num_workers(NUM_WORKERS)
        .build(SamplerDataset::new(
            SftDataset::new(rows),
            sched.epoch_samples,
        ));
    let valid = DataLoaderBuilder::new(SftBatcher)
        .batch_size(BATCH_SIZE)
        .num_workers(NUM_WORKERS)
        .build(SftDataset::new(valid_rows));

    let sft_dir = ctx.checkpoint_path("sft");
    let training = SupervisedTraining::new(sft_dir.clone(), train, valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(sched.ckpt_epochs)
        .summary();

    let training = match (from, ctx.latest_checkpoint_epoch(&sft_dir)) {
        (Some(_), _) | (None, None) => training,
        (None, Some(e)) => {
            println!("resuming sft from checkpoint epoch {e}");
            training.checkpoint(e)
        }
    };

    let result = training.launch(Learner::new(model, AdamConfig::new().init(), LR));
    ctx.save_model(result.model, "model_sft")
        .wrap_err("saving sft model")?;
    println!("saved SFT model to {}", ctx.checkpoint_path("model_sft"));
    Ok(())
}
