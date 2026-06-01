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
use crate::data::{
    MixSchedule, MixtureDataset, SoftwareHeritageSource, StreamingTokenDataset, TokenBatcher,
    load_tokenizer,
};
use crate::train::TrainingContext;

const VALID_WINDOWS: usize = 64;
const LR_MAX: f64 = 3.0e-4;
const LR_MIN: f64 = 3.0e-5;
const GRAD_CLIP_NORM: f32 = 1.0;
const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.95;

/// Data-mix decay anneal (web, math, code), after SmolLM3's multi-stage recipe: a
/// web-heavy stable bulk, then upweight math+code over the cosine-LR decay tail. Weights
/// are renormalized over whichever domains are actually configured (math/code are added
/// only when their shard lists are set), so web-only runs are unaffected.
const MIX_STABLE: [f64; 3] = [0.85, 0.03, 0.12]; // ≈ SmolLM3 stage 1
const MIX_DECAY: [f64; 3] = [0.63, 0.13, 0.24]; // ≈ SmolLM3 stage 3
const DECAY_START: f64 = 0.85; // ramp the mix over the final 15% of steps

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
        tokenizer.clone(),
        FINEWEB_REPO,
        "main",
        &rc.shards,
        "text",
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

    // Assemble the training mixture. Web is always present (the budgeted train view);
    // math/code join only when their shard lists are configured. The per-domain stage
    // weights are sliced from MIX_* in the same web/math/code order, so the schedule
    // renormalizes over exactly the domains in play.
    let mut domains = vec![corpus.view(0..n_train)];
    let mut stable = vec![MIX_STABLE[0]];
    let mut decay = vec![MIX_DECAY[0]];

    // Math: a plain text-column parquet corpus (FineMath).
    if !rc.math_shards.is_empty() {
        println!("indexing math corpus ({} shards)...", rc.math_shards.len());
        domains.push(StreamingTokenDataset::from_hub(
            tokenizer.clone(),
            &rc.math_repo,
            &rc.math_revision,
            &rc.math_shards,
            &rc.math_text_column,
            seq_len,
            rc.cache_groups,
        )?);
        stable.push(MIX_STABLE[1]);
        decay.push(MIX_DECAY[1]);
    }

    // Code: Stack-Edu ships SWHIDs, so each shard is a SoftwareHeritageSource that fetches
    // up to `code_max_files` blobs' content from Software Heritage's S3.
    if !rc.code_shards.is_empty() {
        println!(
            "indexing code corpus ({} shards, ≤{} files each via Software Heritage)...",
            rc.code_shards.len(),
            rc.code_max_files
        );
        let mut sources: Vec<Box<dyn crate::data::RowGroupSource>> =
            Vec::with_capacity(rc.code_shards.len());
        for file in &rc.code_shards {
            sources.push(Box::new(SoftwareHeritageSource::from_hub(
                &rc.code_repo,
                &rc.code_revision,
                file,
                &rc.code_blob_column,
                rc.code_max_files,
            )?));
        }
        domains.push(StreamingTokenDataset::new(
            sources,
            tokenizer.clone(),
            seq_len,
            rc.cache_groups,
        )?);
        stable.push(MIX_STABLE[2]);
        decay.push(MIX_DECAY[2]);
    }
    println!(
        "training mix: {domains_n} domain(s)",
        domains_n = domains.len()
    );

    // Progress denominator: total items drawn across the whole run.
    let total_draws = (sched.epoch_samples as u64) * (sched.ckpt_epochs as u64);
    let mixture = MixtureDataset::new(
        domains,
        MixSchedule::new(stable, decay, DECAY_START),
        total_draws,
    );

    let train_loader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(batch_size)
        .shuffle(SHUFFLE_SEED)
        .num_workers(NUM_WORKERS)
        .build(SamplerDataset::new(mixture, sched.epoch_samples));

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
        .with_beta_1(ADAM_BETA1)
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
