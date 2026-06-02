//! GRPO (Group Relative Policy Optimization) stage.
//!
//! For each prompt we sample a group of `G` completions with the current
//! policy, score them with a verifiable reward, normalize rewards within
//! the group to advantages, and take a policy-gradient step with a KL
//! penalty toward the frozen SFT reference:
//!
//!    loss = -mean(A · logπ) + β · KL_k3(π ‖ π_ref)

use std::sync::Arc;

use crate::config::ModelConfig;
use crate::constants::{
    GRPO_COMPLETION_LEN, GRPO_GROUP_SIZE, GRPO_KL_BETA, GRPO_PROMPT_LEN, GSM8K_FILE, GSM8K_REPO,
    NUM_WORKERS, SHUFFLE_SEED,
};
use crate::data::{GrpoBatch, GrpoBatcher, GrpoDataset, GrpoExample, load_gsm8k, split_valid};
use crate::model::GabrielLaevis;
use crate::model::lm::{Sampling, sequence_logprob};
use crate::train::TrainingContext;
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::transform::SamplerDataset;
use burn::module::AutodiffModule;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::{
    AccuracyMetric, Adaptor, ItemLazy, LossMetric, Metric, MetricAttributes, MetricMetadata,
    MetricName, Numeric, NumericAttributes, NumericEntry, SerializedEntry,
};
use burn::train::{
    ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
};
use eyre::{Result, WrapErr};

const BATCH_SIZE: usize = 4; // prompts per step
const VALID_PROMPTS: usize = 32;
const LR: f64 = 1.0e-6;
const TEMPERATURE: f64 = 1.0; // exploration during rollouts
const ADV_EPS: f32 = 1e-4;

/// Last signed integer appearing in `s`, or `None` if it has no digits.
fn extract_last_int(s: &str) -> Option<String> {
    let mut last = None;
    let mut cur = String::new();
    let has_digit = |t: &str| t.chars().any(|c| c.is_ascii_digit());
    for ch in s.chars() {
        if ch.is_ascii_digit() || (ch == '-' && cur.is_empty()) {
            cur.push(ch);
        } else {
            if has_digit(&cur) {
                last = Some(cur.clone());
            }
            cur.clear();
        }
    }
    if has_digit(&cur) {
        last = Some(cur);
    }
    last
}

/// GSM8K correctness: the completion's last integer equals the answer.
fn gsm8k_correct(completion: &str, answer: &str) -> bool {
    extract_last_int(completion).as_deref() == Some(answer)
}

/// Verifiable reward: 1 for the correct integer, a small shaping reward for
/// producing *any* parseable integer, 0 otherwise.
fn reward(completion: &str, answer: &str) -> f32 {
    match extract_last_int(completion) {
        Some(g) if g == answer => 1.0,
        Some(_) => 0.1,
        None => 0.0,
    }
}

/// Sample `G` completions per prompt with `gen` and score them.
fn sample_all<G: Backend>(
    sampler: &GabrielLaevis<G>,
    batch: &GrpoBatch,
    device: &G::Device,
) -> (Vec<Vec<Vec<i64>>>, Vec<Vec<f32>>) {
    let sampling = Sampling {
        temperature: TEMPERATURE,
        stop_token: Some(crate::chat::TURN_END_ID),
        ..Default::default()
    };

    let mut comps = Vec::with_capacity(batch.prompts.len());
    let mut rewards = Vec::with_capacity(batch.prompts.len());

    for (prompt, answer) in batch.prompts.iter().zip(&batch.answers) {
        // One batched pass produces the whole group of GRPO_GROUP_SIZE completions.
        // Prompts are non-empty by construction; on the off chance generation
        // errors, score the group as empty (zero-reward) completions rather
        // than panic.
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
    /// Score sampled groups.
    fn score(
        &self,
        prompts: &[Vec<i64>],
        comps: &[Vec<Vec<i64>>],
        rewards: &[Vec<f32>],
        device: &B::Device,
    ) -> ClassificationOutput<B> {
        let ignore = crate::chat::IGNORE_ID as i64;
        let reference = self.reference.valid();
        let mut group_losses: Vec<Tensor<B, 1>> = Vec::new();
        let mut corrects: Vec<Tensor<B, 1>> = Vec::new();

        for ((prompt, group), group_r) in prompts.iter().zip(comps).zip(rewards) {
            let g = group.len();
            let p = prompt.len();
            // Completions are ragged: EOS can end one early, and a failed/immediate-EOS
            // rollout can be empty. Pad the group to its longest sequence; padding and
            // the prompt prefix is IGNORE_ID in both input and target, so it's masked
            // out of sequence_logprob and adds no signal.
            let max_c = group.iter().map(|c| c.len()).max().unwrap_or(0);

            if max_c == 0 {
                continue; // whole group produced no tokens: no gradient signal
            }

            let l = p + max_c - 1; // next-token shifted length of the longest sequence

            // Fill flat [G, l] buffers pre-initialised to IGNORE_ID.
            let mut in_flat = vec![ignore; g * l];
            let mut tg_flat = vec![ignore; g * l];

            for (i, comp) in group.iter().enumerate() {
                let row_start = i * l;
                let c = comp.len();

                if c == 0 {
                    // Empty rollout: input is prompt-only (targets stay all-ignore, so the
                    // row contributes nothing); the final input slot is left padded.
                    let in_row = &mut in_flat[row_start..row_start + p - 1];
                    in_row.copy_from_slice(&prompt[..p - 1]);
                    continue;
                }

                // Inputs: prompt and comp without its final token (the shifted-input frame).
                let in_row = &mut in_flat[row_start..row_start + p + c - 1];
                in_row[..p].copy_from_slice(prompt);
                in_row[p..].copy_from_slice(&comp[..c - 1]);

                // Targets: completion tokens placed at the shifted prompt tail; prompt
                // positions stay ignore, so loss is over completion tokens only.
                let tg_row = &mut tg_flat[row_start + p - 1..row_start + p + c - 1];
                tg_row.copy_from_slice(comp);
            }

            let inputs = Tensor::<B, 2, Int>::from_data(TensorData::new(in_flat, [g, l]), device);
            let targets = Tensor::<B, 2, Int>::from_data(TensorData::new(tg_flat, [g, l]), device);

            let logp = sequence_logprob(self.policy.forward(inputs.clone()), targets.clone()); // [G] grad

            let logp_ref = Tensor::from_inner(sequence_logprob(
                reference.forward(inputs.inner()),
                targets.inner(),
            )); // [G]

            // Rewards are host-born; upload once, then normalize to group advantages on-device.
            let r = Tensor::<B, 1>::from_data(TensorData::new(group_r.clone(), [g]), device);
            let mean = r.clone().mean();

            let std = ((r.clone() - mean.clone()).powi_scalar(2).mean().sqrt()) + ADV_EPS; // [1]
            let adv = (r.clone() - mean) / std; // [G]

            let pg = -(adv * logp.clone()).mean(); // -mean(A·logπ)

            // KL penalty toward the frozen reference via the k3 estimator from DeepSeekMath:
            // with d = logπ_ref - logπ, KL ≈ exp(d) - d - 1. Stays ≥ 0 with far lower
            // variance than the raw log-ratio.
            let d = logp_ref - logp;
            let kl = (d.clone().exp() - d - 1.0).mean();

            group_losses.push(pg + (kl * GRPO_KL_BETA));
            corrects.push(r.greater_equal_elem(1.0).float()); // [G], 1 where reward == 1
        }

        // Every group was empty, no usable signal this step. Return a detached zero loss
        // so the step is a no-op.
        if group_losses.is_empty() {
            let loss = Tensor::<B, 1>::zeros([1], device).mean();
            let logits = Tensor::<B, 2>::zeros([1, 2], device);
            let targets = Tensor::<B, 1, Int>::ones([1], device);
            return ClassificationOutput::new(loss, logits, targets);
        }

        let loss = Tensor::cat(group_losses, 0).mean();

        // Reward pass-rate for AccuracyMetric: class 1 = "correct". Build [n, 2] logits
        // [1-c, c] so accuracy reads as the fraction of completions that earned reward 1.
        let c = Tensor::cat(corrects, 0).unsqueeze_dim(1); // [n, 1]
        let logits = Tensor::cat(vec![-c.clone() + 1.0, c], 1); // [n, 2]
        let [n, _] = logits.dims();
        let targets = Tensor::<B, 1, Int>::ones([n], device);

        ClassificationOutput::new(loss, logits, targets)
    }
}

impl<B: AutodiffBackend> TrainStep for GrpoModel<B> {
    type Input = GrpoBatch;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: GrpoBatch) -> TrainOutput<ClassificationOutput<B>> {
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

    /// Validate with the held-out solve rate. pass@k reuses the rollout group;
    /// pass@1 adds one greedy sample. The pass@1↓-while-pass@k-holds gap is
    /// the GRPO collapse signal.
    fn step(&self, batch: GrpoBatch) -> SolveCounts {
        let device = self.policy.device();
        let (_comps, rewards) = sample_all(&self.policy, &batch, &device);
        let greedy = Sampling {
            stop_token: Some(crate::chat::TURN_END_ID),
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
    let cfg = ModelConfig::load(ctx.checkpoint_path("config.json")).wrap_err("loading config")?;
    cfg.validate()?;

    // `--from` wins; otherwise chain off the freshest upstream checkpoint: DPO's output
    // when present, else the SFT model. So SFT→DPO→GRPO composes, and SFT→GRPO still works.
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
            .wrap_err_with(|| format!("loading {base} as {what}"))
    };

    let model = GrpoModel {
        policy: load("policy")?,
        reference: load("reference")?,
    };

    println!("loading {GSM8K_REPO} with up to {max_examples} prompts...");
    let pairs = load_gsm8k(GSM8K_REPO, GSM8K_FILE, max_examples).wrap_err("loading gsm8k")?;

    let examples: Vec<GrpoExample> = pairs
        .into_iter()
        .map(|(q, answer)| {
            let prompt: Vec<i64> = ctx
                .tokenizer
                .encode(&crate::chat::render_prompt(&q))
                .wrap_err("encoding gsm8k prompt")?
                .into_iter()
                .take(GRPO_PROMPT_LEN) // truncate to the prompt budget as we collect
                .map(|x| x as i64)
                .collect();
            Ok(GrpoExample { prompt, answer })
        })
        .collect::<Result<_>>()?;

    let (examples, valid_examples) = split_valid(examples, VALID_PROMPTS);

    // Owned tokenizer for the batcher (re-loaded from cache; avoids needing Clone).
    let tokenizer = std::sync::Arc::new(crate::utils::load_tokenizer().wrap_err("load tokenizer")?);
    let batcher = GrpoBatcher { tokenizer };

    // Slice into shared 1000-step checkpoint-epochs with auto-resume (see dpo/pretrain).
    let sched = crate::train::schedule(examples.len(), BATCH_SIZE, epochs);
    println!(
        "training {} steps in ~{epochs} passes over {} prompts as {} x {}-step checkpoints",
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

    let grpo_dir = ctx.checkpoint_path("grpo");
    let training = SupervisedTraining::new(grpo_dir.clone(), train, valid)
        .metric_train_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(SolveRateMetric::pass1())
        .metric_valid_numeric(SolveRateMetric::pass_k(GRPO_GROUP_SIZE))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(sched.ckpt_epochs)
        .summary();

    // Auto-resume from the latest checkpoint in the stage dir unless `--from` was given.
    let training = match ctx.resume_epoch(from, &grpo_dir) {
        Some(e) => {
            println!("resuming grpo from checkpoint epoch {e}");
            training.checkpoint(e)
        }
        None => training,
    };

    let result = training.launch(Learner::new(model, AdamConfig::new().init(), LR));
    ctx.save_model(result.model.policy, "model_grpo")
        .wrap_err("saving grpo model")?;

    println!(
        "saved GRPO model to {:?}",
        ctx.checkpoint_path("model_grpo")
    );
    Ok(())
}

// ===========================================================================
// Held-out solve-rate metric
// ===========================================================================
//
// Training loss/accuracy says nothing about whether sampled completions actually
// *solve* held-out problems. So the validation `InferenceStep` reports `SolveCounts`
// — greedy pass@1 and sampled pass@k over the disjoint valid set — and
// `SolveRateMetric` renders them in the Learner TUI alongside the loss. The
// pass@1-vs-pass@k gap over training is the RL mode-collapse signal (Cobbe et al.,
// GSM8K, arXiv:2110.14168, Fig. 3): pass@1 climbs while pass@k craters as the policy's
// coverage narrows.

/// Per-validation-batch solve counts: the `InferenceStep` returns this so the held-out
/// pass@1 / pass@k render live as numeric metrics.
#[derive(Debug, Clone, Copy)]
pub struct SolveCounts {
    /// Prompts solved by the single greedy sample.
    pub pass1: usize,
    /// Prompts solved by at least one of the k sampled completions.
    pub pass_k: usize,
    /// Prompts in this batch.
    pub total: usize,
}

impl ItemLazy for SolveCounts {
    type ItemSync = SolveCounts;
    fn sync(self) -> Self::ItemSync {
        self
    }
}

/// [`SolveRateMetric`] input; the metric picks pass@1 vs pass@k
/// by its [`SolveKind`], so both metrics share one [`Adaptor`] impl.
pub struct SolveRateInput {
    pass1: usize,
    pass_k: usize,
    total: usize,
}

impl Adaptor<SolveRateInput> for SolveCounts {
    fn adapt(&self) -> SolveRateInput {
        SolveRateInput {
            pass1: self.pass1,
            pass_k: self.pass_k,
            total: self.total,
        }
    }
}

#[derive(Clone, Copy)]
enum SolveKind {
    Pass1,
    PassK,
}

/// A held-out solve-rate metric.
#[derive(Clone)]
pub struct SolveRateMetric {
    name: MetricName,
    state: NumericMetricState,
    kind: SolveKind,
}

impl SolveRateMetric {
    /// Greedy pass@1.
    pub fn pass1() -> Self {
        Self {
            name: Arc::new("Pass@1".to_string()),
            state: NumericMetricState::default(),
            kind: SolveKind::Pass1,
        }
    }
    /// Sampled pass@k (`any of k correct`).
    pub fn pass_k(k: usize) -> Self {
        Self {
            name: Arc::new(format!("Pass@{k}")),
            state: NumericMetricState::default(),
            kind: SolveKind::PassK,
        }
    }
}

impl Metric for SolveRateMetric {
    type Input = SolveRateInput;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let solved = match self.kind {
            SolveKind::Pass1 => input.pass1,
            SolveKind::PassK => input.pass_k,
        };
        let pct = solved as f64 / input.total.max(1) as f64 * 100.0;
        self.state.update(
            pct,
            input.total,
            FormatOptions::new(self.name()).unit("%").precision(1),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: Some("%".to_string()),
            higher_is_better: true,
        }
        .into()
    }
}

impl Numeric for SolveRateMetric {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }
    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_trailing_answer() {
        assert_eq!(extract_last_int("so 2+2 = 4").as_deref(), Some("4"));
        assert_eq!(
            extract_last_int("steps: 10, 20, then -5").as_deref(),
            Some("-5")
        );
        assert_eq!(extract_last_int("no digits here"), None);
    }

    #[test]
    fn gsm8k_correct_matches_last_int() {
        assert!(gsm8k_correct("the answer is #### 42", "42"));
        assert!(!gsm8k_correct("the answer is 41", "42"));
        // Trailing prose after the number still resolves to the last integer.
        assert!(gsm8k_correct("42 dollars total, so 7 dozen", "7"));
    }
}
