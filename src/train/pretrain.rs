//! Pretraining stage using next-token cross-entropy.

use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::SamplerDataset;
use burn::grad_clipping::GradientClippingConfig;
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::optim::AdamConfig;
use burn::record::CompactRecorder;
use burn::train::metric::{LossMetric, PerplexityMetric};
use burn::train::{Learner, SupervisedTraining};
use eyre::{Result, WrapErr};

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
    // access, never held whole in RAM (viable for 5B+ tokens). Construction runs a
    // streaming count pass to size the exact window index. An owned Arc tokenizer is
    // needed because the dataset must be Send+Sync across dataloader workers.
    println!("indexing FineWeb-Edu shards (streaming dense packing, counting tokens)...");
    let tokenizer =
        std::sync::Arc::new(load_tokenizer(TOKENIZER_REPO).wrap_err("loading tokenizer")?);
    let corpus = StreamingTokenDataset::from_hub(
        tokenizer,
        FINEWEB_REPO,
        &rc.shards,
        seq_len,
        rc.cache_groups,
    )
    .wrap_err("indexing fineweb shards")?;
    let windows_total = corpus.total();
    // Window counts are exact (the count pass tokenized every group), so the token
    // budget maps directly: keep the first `max_tokens`-worth of dense windows. 0 = all.
    let budget_windows = if max_tokens == 0 {
        windows_total
    } else {
        (max_tokens / (seq_len + 1)).clamp(1, windows_total)
    };
    println!(
        "indexed {windows_total} dense windows across {} shard(s); using {budget_windows} \
         (~{} tokens, vocab {})",
        rc.shards.len(),
        budget_windows * (seq_len + 1),
        ctx.vocab_size()
    );

    // The model carries its own size: save its config beside the checkpoint so every
    // later stage and `serve` reconstruct the same architecture.
    let cfg = rc.model(ctx.vocab_size());
    cfg.validate()?;
    cfg.save(ctx.checkpoint_path("config.json"))
        .wrap_err("saving config")?;
    // `--from` seeds weights from a given checkpoint and starts a fresh schedule;
    // otherwise init randomly and rely on auto-resume below.
    let model = match from {
        Some(name) => {
            println!("starting from {name} (fresh optimizer + schedule)");
            ctx.load_model(&cfg, name)?
        }
        None => ctx.fresh_model(&cfg),
    };

    // Hold out a contiguous tail of windows for validation, disjoint from train. The
    // split is on a window boundary, so no sequence straddles it and no token leaks.
    let valid_windows = VALID_WINDOWS.min(budget_windows / 5);
    let n_train = budget_windows - valid_windows;
    let dataset = corpus.view(0..n_train);
    let valid_dataset = corpus.view(n_train..budget_windows);
    let windows = dataset.len().max(1);
    let batcher = TokenBatcher { seq_len };

    // Slice the run into fixed-size checkpoint-epochs (shared with every other stage).
    // The LR schedule must span the actual `total_steps` (rounded up to whole epochs),
    // not the un-rounded request, or the cosine bottoms out early and the tail trains
    // at min_lr.
    let crate::train::Schedule {
        ckpt_epochs,
        epoch_samples,
        total_steps,
    } = crate::train::schedule(windows, batch_size, epochs);
    println!(
        "training {total_steps} steps (~{epochs} passes over {windows} windows) as \
         {ckpt_epochs} x {STEPS_PER_CHECKPOINT}-step checkpoints (batch {batch_size})",
        STEPS_PER_CHECKPOINT = crate::train::STEPS_PER_CHECKPOINT
    );

    let train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(batch_size)
        .shuffle(SHUFFLE_SEED)
        .num_workers(NUM_WORKERS)
        .build(SamplerDataset::new(dataset.clone(), epoch_samples));
    let valid_size = valid_dataset.len().max(1);
    let valid = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .num_workers(NUM_WORKERS)
        .build(SamplerDataset::new(valid_dataset, valid_size));

    let lr = CosineAnnealingLrSchedulerConfig::new(LR_MAX, total_steps)
        .with_min_lr(LR_MIN)
        .init()
        .map_err(|e| eyre::eyre!("lr scheduler init: {e}"))?;
    let optim = AdamConfig::new()
        .with_beta_2(ADAM_BETA2)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(GRAD_CLIP_NORM)))
        .init();

    let training = SupervisedTraining::new(&ctx.artifact, train, valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(PerplexityMetric::new())
        .metric_valid_numeric(PerplexityMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(ckpt_epochs)
        .summary();

    // Auto-resume from the latest checkpoint in the artifact dir, if present.
    // An explicit `--from` overrides this: it starts a fresh schedule from the
    // given weights and ignores any prior checkpoints in the dir.
    let training = match (from, ctx.latest_checkpoint_epoch(&ctx.artifact)) {
        (Some(_), _) => training,
        (None, Some(e)) => {
            println!("resuming from checkpoint epoch {e}");
            training.checkpoint(e)
        }
        (None, None) => training,
    };

    let result = training.launch(Learner::new(model, optim, lr));
    ctx.save_model(result.model, "model")
        .wrap_err("saving pretrained model")?;
    println!("saved model + config to {}", ctx.artifact);
    Ok(())
}
