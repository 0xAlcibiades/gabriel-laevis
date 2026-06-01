//! Pretraining stage using next-token cross-entropy.

use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::transform::SamplerDataset;
use burn::grad_clipping::GradientClippingConfig;
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::optim::AdamConfig;
use burn::record::CompactRecorder;
use burn::train::metric::{LossMetric, PerplexityMetric};
use burn::train::{Learner, SupervisedTraining};
use eyre::Result;

use crate::constants::{FINEWEB_REPO, NUM_WORKERS, SHUFFLE_SEED, TOKENIZER_REPO};
use crate::data::{StreamingTokenDataset, TokenBatcher, load_tokenizer};
use crate::train::TrainingContext;

const VALID_WINDOWS: usize = 64;
const LR_MAX: f64 = 3.0e-4;
const LR_MIN: f64 = 3.0e-5;
const GRAD_CLIP_NORM: f32 = 1.0;
const ADAM_BETA2: f32 = 0.95;

pub fn run(
    ctx: &TrainingContext,
    max_tokens: usize,
    epochs: usize,
    from: Option<&str>,
) -> Result<()> {
    let rc = crate::config::run();
    let batch_size = rc.batch;
    let seq_len = rc.max_seq_len;

    // Streaming corpus: dense seq_len+1 windows packed per row group, tokenized on
    // access, never held whole in RAM (viable for 5B+ tokens). An owned Arc tokenizer
    // is needed because the dataset must be Send+Sync across dataloader workers.
    println!("indexing FineWeb-Edu shards...");
    let tokenizer = std::sync::Arc::new(load_tokenizer(TOKENIZER_REPO)?);

    let corpus = StreamingTokenDataset::from_hub(
        tokenizer,
        FINEWEB_REPO,
        &rc.shards,
        seq_len,
        rc.cache_groups,
    )?;

    let windows_total = corpus.total();

    // Window counts are exact (the count pass tokenized every group), so the token
    // budget maps directly: keep the first `max_tokens`-worth of dense windows. 0 = all.
    let budget_windows = if max_tokens == 0 {
        windows_total
    } else {
        (max_tokens / (seq_len + 1)).clamp(1, windows_total)
    };

    println!(
        "using {budget_windows} windows (~{} tokens)",
        budget_windows * (seq_len + 1)
    );

    // The model carries its own size: save its config beside the checkpoint so every
    // later stage and `serve` reconstruct the same architecture.
    let cfg = rc.model(ctx.vocab_size());
    cfg.validate()?;
    cfg.save(ctx.checkpoint_path("config.json"))?;

    let model = match from {
        Some(name) => ctx.load_model(&cfg, name)?,
        None => ctx.fresh_model(&cfg),
    };

    // Hold out a contiguous tail of windows for validation, disjoint from train. The
    // split is on a window boundary, so no sequence straddles it and no token leaks.
    let valid_windows = VALID_WINDOWS.min(budget_windows / 5);
    let n_train = budget_windows - valid_windows;

    let sched = crate::train::schedule(n_train, batch_size, epochs);
    let batcher = TokenBatcher { seq_len };

    let train_loader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(batch_size)
        .shuffle(SHUFFLE_SEED)
        .num_workers(NUM_WORKERS)
        .build(SamplerDataset::new(
            corpus.view(0..n_train),
            sched.epoch_samples,
        ));

    let valid_loader = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .num_workers(NUM_WORKERS)
        .build(SamplerDataset::new(
            corpus.view(n_train..budget_windows),
            valid_windows.max(1),
        ));

    let lr = CosineAnnealingLrSchedulerConfig::new(LR_MAX, sched.total_steps.max(1))
        .with_min_lr(LR_MIN)
        .init()
        .map_err(|e| eyre::eyre!("lr scheduler init: {e}"))?;

    let optim = AdamConfig::new()
        .with_beta_2(ADAM_BETA2)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(GRAD_CLIP_NORM)))
        .init();

    let training = SupervisedTraining::new(&ctx.artifact, train_loader, valid_loader)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(PerplexityMetric::new())
        .metric_valid_numeric(PerplexityMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(sched.ckpt_epochs)
        .summary();

    // Auto-resume from the latest checkpoint in the artifact dir unless `--from` was
    // given (which starts a fresh schedule from the named weights).
    let epoch_to_resume = from
        .is_none()
        .then(|| ctx.latest_checkpoint_epoch(&ctx.artifact))
        .flatten();

    let training = if let Some(e) = epoch_to_resume {
        println!("resuming from checkpoint epoch {e}");
        training.checkpoint(e)
    } else {
        training
    };

    let result = training.launch(Learner::new(model, optim, lr));

    ctx.save_model(result.model, "model")?;

    println!("saved model + config to {}", ctx.artifact.display());

    Ok(())
}
