//! GRPO (Group Relative Policy Optimization) stage.
//!
//! For each prompt we sample a group of `G` completions with the current
//! policy, score them with a **verifiable** reward (does the completion end in
//! the correct integer?), normalize rewards within the group to advantages, and
//! take a policy-gradient step with a KL penalty toward the frozen SFT
//! reference:
//!
//!   loss = -mean(A · logπ) + β · KL_k3(π ‖ π_ref)

use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::transform::SamplerDataset;
use burn::module::AutodiffModule;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{
    ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
};
use eyre::{Result, WrapErr};

use crate::config::ModelConfig;
use crate::constants::{
    GRPO_COMPLETION_LEN, GRPO_GROUP_SIZE, GRPO_KL_BETA, GRPO_PROMPT_LEN, GSM8K_FILE, GSM8K_REPO,
    NUM_WORKERS, SHUFFLE_SEED, TOKENIZER_REPO,
};
use crate::data::{
    GrpoBatch, GrpoBatcher, GrpoDataset, GrpoExample, load_gsm8k, load_tokenizer, split_valid,
};
use crate::eval::{SolveCounts, SolveRateMetric, extract_last_int, gsm8k_correct};
use crate::model::GabrielLaevis;
use crate::model::lm::{Sampling, sequence_logprob};
use crate::train::TrainingContext;

const BATCH_SIZE: usize = 4; // prompts per step
const VALID_PROMPTS: usize = 32;
const LR: f64 = 1.0e-6;
const TEMPERATURE: f64 = 1.0; // exploration during rollouts
const ADV_EPS: f32 = 1e-4;

/// Verifiable reward: 1 for the correct integer, a small shaping reward for
/// producing *any* parseable integer, 0 otherwise.
fn reward(completion: &str, answer: &str) -> f32 {
    match extract_last_int(completion) {
        Some(g) if g == answer => 1.0,
        Some(_) => 0.1,
        None => 0.0,
    }
}

/// Sample `G` completions per prompt with `gen` and score them. Generic over the
/// generator backend so the train path can pass `policy.valid()` (inner) and the
/// valid path can pass the (already-inner) policy.
fn sample_all<G: Backend>(
    sampler: &GabrielLaevis<G>,
    batch: &GrpoBatch,
    device: &G::Device,
) -> (Vec<Vec<Vec<i64>>>, Vec<Vec<f32>>) {
    let sampling = Sampling {
        temperature: TEMPERATURE,
        stop_token: crate::data::im_end_id(&batch.tokenizer),
        ..Default::default()
    };
    let mut comps = Vec::with_capacity(batch.prompts.len());
    let mut rewards = Vec::with_capacity(batch.prompts.len());
    for (prompt, answer) in batch.prompts.iter().zip(&batch.answers) {
        // One batched pass produces the whole group of GRPO_GROUP_SIZE completions
        // (prompt primed once, rollouts stepped in parallel). Prompts are non-empty by
        // construction; on the off chance generation errors, score the group as empty
        // (zero-reward) completions rather than panic.
        let group = sampler
            .generate_group(
                prompt,
                GRPO_GROUP_SIZE,
                GRPO_COMPLETION_LEN,
                &sampling,
                device,
            )
            .unwrap_or_else(|e| {
                eprintln!("grpo rollout generation failed: {e}");
                vec![Vec::new(); GRPO_GROUP_SIZE]
            });
        let group_r: Vec<f32> = group
            .iter()
            .map(|comp| {
                let ids: Vec<u32> = comp.iter().map(|&x| x as u32).collect();
                let text = batch.tokenizer.decode(&ids, true).unwrap_or_default();
                reward(&text, answer)
            })
            .collect();
        comps.push(group);
        rewards.push(group_r);
    }
    (comps, rewards)
}

#[derive(Module, Debug)]
pub struct GrpoModel<B: Backend> {
    policy: GabrielLaevis<B>,
    reference: GabrielLaevis<B>,
}

impl<B: AutodiffBackend> GrpoModel<B> {
    /// Score sampled groups: build PG + KL loss and the reward-accuracy output. Only
    /// called from the train step (autodiff backend), so the frozen reference runs on
    /// the inner backend — no autograd graph, no per-step leak.
    fn score(
        &self,
        prompts: &[Vec<i64>],
        comps: &[Vec<Vec<i64>>],
        rewards: &[Vec<f32>],
        device: &B::Device,
    ) -> ClassificationOutput<B> {
        let ignore = crate::chat::IGNORE_ID as i64;
        let reference = self.reference.valid(); // inner-backend module, graph-free
        let mut group_losses: Vec<Tensor<B, 1>> = Vec::new();
        let mut corrects: Vec<Tensor<B, 1>> = Vec::new();

        for ((prompt, group), group_r) in prompts.iter().zip(comps).zip(rewards) {
            let g = group.len();
            let p = prompt.len();
            // Completions are ragged now: EOS can end one early, and a failed/immediate-
            // EOS rollout can be empty. Pad the group to its longest sequence; padding
            // (and the prompt prefix) is IGNORE_ID in both input and target, so it is
            // masked out of sequence_logprob and adds no signal — only trailing compute.
            let max_c = group.iter().map(|c| c.len()).max().unwrap_or(0);
            if max_c == 0 {
                continue; // whole group produced no tokens: no gradient signal
            }
            let l = p + max_c - 1; // next-token shifted length of the longest sequence

            // Build [G, l] inputs/targets; targets masked to completion tokens. Token
            // ids are CPU-born (they come from the tokenizer), so this stays host-side.
            let mut in_flat = Vec::with_capacity(g * l);
            let mut tg_flat = Vec::with_capacity(g * l);
            for comp in group {
                let mut full = prompt.clone();
                full.extend_from_slice(comp); // len p + comp.len()
                for j in 0..l {
                    in_flat.push(if j < full.len() { full[j] } else { ignore });
                    tg_flat.push(if j + 1 >= p && j + 1 < full.len() {
                        full[j + 1]
                    } else {
                        ignore
                    });
                }
            }
            let inputs = Tensor::<B, 2, Int>::from_data(TensorData::new(in_flat, [g, l]), device);
            let targets = Tensor::<B, 2, Int>::from_data(TensorData::new(tg_flat, [g, l]), device);

            let logp = sequence_logprob(self.policy.forward(inputs.clone()), targets.clone()); // [G] grad
            // Reference on the inner backend: building its graph on the autodiff backend
            // and `.detach()`-ing only the output still retains every activation across
            // steps (backward consumes only the policy graph) — an unbounded leak. The
            // inner forward holds no graph; `from_inner` lifts the result back to B.
            let logp_ref = Tensor::from_inner(sequence_logprob(
                reference.forward(inputs.inner()),
                targets.inner(),
            )); // [G]

            // Rewards are host-born (string-parsed from completions); upload once,
            // then normalize to group advantages on-device.
            let r = Tensor::<B, 1>::from_data(TensorData::new(group_r.clone(), [g]), device);
            let mean = r.clone().mean();
            let std = r
                .clone()
                .sub(mean.clone())
                .powi_scalar(2)
                .mean()
                .sqrt()
                .add_scalar(ADV_EPS); // [1]
            let adv = r.clone().sub(mean).div(std); // [G]

            let pg = adv.mul(logp.clone()).mean().neg(); // -mean(A·logπ)
            // KL penalty toward the frozen reference via the k3 estimator from
            // DeepSeekMath: with d = logπ_ref - logπ, KL ≈ exp(d) - d - 1. Stays
            // ≥ 0 with far lower variance than the raw log-ratio.
            let d = logp_ref.sub(logp);
            let kl = d.clone().exp().sub(d).sub_scalar(1.0).mean();
            group_losses.push(pg.add(kl.mul_scalar(GRPO_KL_BETA)));

            corrects.push(r.greater_equal_elem(1.0).float()); // [G], 1 where reward == 1
        }

        // Every group was empty (all rollouts immediately hit EOS / failed): no usable
        // signal this step. Return a detached zero loss so the step is a no-op instead
        // of panicking in Tensor::cat on an empty vec.
        if group_losses.is_empty() {
            let loss = Tensor::<B, 1>::zeros([1], device).mean();
            let logits = Tensor::<B, 2>::zeros([1, 2], device);
            let targets = Tensor::<B, 1, Int>::ones([1], device);
            return ClassificationOutput::new(loss, logits, targets);
        }

        let loss = Tensor::cat(group_losses, 0).mean();

        // Reward pass-rate for AccuracyMetric: class 1 = "correct". Build [n, 2] logits
        // [1-c, c] so accuracy reads as the fraction of completions that earned reward 1.
        let c = Tensor::cat(corrects, 0); // [n]
        let n = c.dims()[0];
        let c = c.reshape([n, 1]);
        let logits = Tensor::cat(vec![c.clone().neg().add_scalar(1.0), c], 1); // [n, 2]
        let targets = Tensor::<B, 1, Int>::ones([n], device);
        ClassificationOutput::new(loss, logits, targets)
    }
}

impl<B: AutodiffBackend> TrainStep for GrpoModel<B> {
    type Input = GrpoBatch;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: GrpoBatch) -> TrainOutput<ClassificationOutput<B>> {
        // Sample on the inner backend.
        let sampler = self.policy.valid();
        let idev = sampler.device();
        let (comps, rewards) = sample_all(&sampler, &batch, &idev);

        let device = self.policy.device();
        let item = self.score(&batch.prompts, &comps, &rewards, &device);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for GrpoModel<B> {
    type Input = GrpoBatch;
    type Output = SolveCounts;

    /// Validation = the held-out solve rate (plotted live in the TUI). pass@k reuses
    /// the rollout group (any of `GRPO_GROUP_SIZE` correct); pass@1 adds one greedy
    /// sample. The pass@1↓-while-pass@k-holds gap is the GRPO collapse signal.
    fn step(&self, batch: GrpoBatch) -> SolveCounts {
        let device = self.policy.device();
        let (_comps, rewards) = sample_all(&self.policy, &batch, &device);
        let greedy = Sampling {
            stop_token: crate::data::im_end_id(&batch.tokenizer),
            ..Sampling::greedy()
        };
        let (mut pass1, mut pass_k) = (0usize, 0usize);
        for ((prompt, answer), group_r) in batch.prompts.iter().zip(&batch.answers).zip(&rewards) {
            if group_r.iter().any(|&r| r >= 1.0) {
                pass_k += 1;
            }
            let g = self
                .policy
                .generate(prompt, GRPO_COMPLETION_LEN, &greedy, &device)
                .unwrap_or_else(|e| {
                    eprintln!("grpo eval generation failed: {e}");
                    Vec::new()
                });
            let ids: Vec<u32> = g.iter().map(|&x| x as u32).collect();
            let text = batch.tokenizer.decode(&ids, true).unwrap_or_default();
            if gsm8k_correct(&text, answer) {
                pass1 += 1;
            }
        }
        SolveCounts {
            pass1,
            pass_k,
            total: batch.prompts.len(),
        }
    }
}

pub fn run(
    ctx: &TrainingContext,
    max_examples: usize,
    epochs: usize,
    from: Option<&str>,
) -> Result<()> {
    let cfg = ModelConfig::load(ctx.checkpoint_path("config.json"))
        .wrap_err("loading config (run `train sft` first)")?;
    cfg.validate()?;

    // `--from` wins; otherwise chain off the freshest upstream checkpoint: DPO's
    // output when present, else the SFT model. So SFT→DPO→GRPO composes, and
    // SFT→GRPO still works.
    let base = match from {
        Some(f) => f,
        None if ctx.has_checkpoint("model_dpo") => "model_dpo",
        None => "model_sft",
    };
    let load = |what: &str| {
        GabrielLaevis::<crate::Train>::new(&cfg, &ctx.device)
            .load_file(
                ctx.checkpoint_path(base),
                &CompactRecorder::new(),
                &ctx.device,
            )
            .wrap_err_with(|| format!("loading {base} as {what} (run `train sft`/`dpo` first)"))
    };
    let model = GrpoModel {
        policy: load("policy")?,
        reference: load("reference")?,
    };

    println!("loading {GSM8K_REPO} (up to {max_examples} prompts)...");
    let pairs = load_gsm8k(GSM8K_REPO, GSM8K_FILE, max_examples).wrap_err("loading gsm8k")?;
    // Skip prompts with bytes the frozen-vocab tokenizer can't encode (rare in GSM8K,
    // but cheap to be safe) instead of aborting the stage.
    let total = pairs.len();
    let examples: Vec<GrpoExample> = pairs
        .into_iter()
        .filter_map(|(q, answer)| {
            let mut prompt: Vec<i64> = ctx
                .tokenizer
                .encode(&crate::chat::render_prompt(&q))
                .ok()?
                .into_iter()
                .map(|x| x as i64)
                .collect();
            prompt.truncate(GRPO_PROMPT_LEN);
            Some(GrpoExample { prompt, answer })
        })
        .collect();
    if examples.len() < total {
        println!("skipped {} unencodable prompts", total - examples.len());
    }
    let (examples, valid_examples) = split_valid(examples, VALID_PROMPTS);

    // Owned tokenizer for the batcher (re-loaded from cache; avoids needing Clone).
    let tokenizer = std::sync::Arc::new(load_tokenizer(TOKENIZER_REPO).wrap_err("load tokenizer")?);
    let batcher = GrpoBatcher { tokenizer };

    // Slice into shared 1000-step checkpoint-epochs so an overnight run checkpoints
    // periodically and can auto-resume after a crash, instead of only saving at the end.
    let sched = crate::train::schedule(examples.len(), BATCH_SIZE, epochs);
    println!(
        "training {} steps (~{epochs} passes over {} prompts) as {} x {}-step checkpoints",
        sched.total_steps,
        examples.len(),
        sched.ckpt_epochs,
        crate::train::STEPS_PER_CHECKPOINT.min(sched.total_steps),
    );

    let train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(BATCH_SIZE)
        .shuffle(SHUFFLE_SEED)
        .num_workers(NUM_WORKERS)
        .build(SamplerDataset::new(
            GrpoDataset::new(examples),
            sched.epoch_samples,
        ));
    // Validation runs full rollouts, so keep it an exact single pass over the held-out set.
    let valid = DataLoaderBuilder::new(batcher)
        .batch_size(BATCH_SIZE)
        .num_workers(NUM_WORKERS)
        .build(GrpoDataset::new(valid_examples));

    // Train metrics: the PG+KL loss and the train-rollout reward pass-rate. Valid
    // metrics: held-out greedy pass@1 and sampled pass@k — the honest "did RL help"
    // signal, with the pass@1-vs-pass@k gap exposing mode-collapse live.
    let grpo_dir = ctx.checkpoint_path("grpo");
    let training = SupervisedTraining::new(grpo_dir.clone(), train, valid)
        .metric_train_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(SolveRateMetric::pass1())
        .metric_valid_numeric(SolveRateMetric::pass_k(GRPO_GROUP_SIZE))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(sched.ckpt_epochs)
        .summary();

    // Auto-resume from the latest checkpoint in the stage dir unless `--from` was given
    // (which starts fresh from the named weights). Mirrors pretrain.
    let training = match (from, ctx.latest_checkpoint_epoch(&grpo_dir)) {
        (Some(_), _) | (None, None) => training,
        (None, Some(e)) => {
            println!("resuming grpo from checkpoint epoch {e}");
            training.checkpoint(e)
        }
    };

    let result = training.launch(Learner::new(model, AdamConfig::new().init(), LR));
    ctx.save_model(result.model.policy, "model_grpo")
        .wrap_err("saving grpo model")?;
    println!("saved GRPO model to {}", ctx.checkpoint_path("model_grpo"));
    Ok(())
}
