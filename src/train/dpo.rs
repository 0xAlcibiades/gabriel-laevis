//! DPO (Direct Preference Optimization) stage.
//!
//! The policy is initialised from the SFT checkpoint and optimised to raise the
//! implicit reward margin against a frozen reference (also the SFT model). The
//! reference forward is run **inline** in the train step and detached — no separate
//! precompute pass. (A cache only pays off past a couple of epochs; at 1-2 epochs it
//! costs a full upfront reference pass over the whole dataset for nothing. GRPO scores
//! its reference the same way.)
//!
//!   loss = -log σ(β·((logπ_c - logπ_r) - (logref_c - logref_r)))
//!        = softplus(-β·((logπ_c - logπ_r) - (logref_c - logref_r)))
//!
//! Run `train sft` first to produce `model_sft` + `config.json`.

use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::transform::SamplerDataset;
use burn::module::{AutodiffModule, Module};
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::activation;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{
    ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
};
use eyre::{Result, WrapErr};

use crate::config::ModelConfig;
use crate::constants::{DPO_BETA, DPO_FILE, DPO_REPO, DPO_SEQ_LEN, NUM_WORKERS, SHUFFLE_SEED};
use crate::data::{
    DpoBatch, DpoBatcher, DpoDataset, DpoExample, build_sft_row, load_dpo_triples, split_valid,
};
use crate::model::GabrielLaevis;
use crate::model::lm::sequence_logprob;
use crate::train::TrainingContext;

// DPO retains two policy autograd graphs per step (chosen + rejected) vs SFT's one,
// so batch 4 here ≈ SFT's batch 8 in retained-activation terms. The reference runs on
// the inner backend (graph-free, freed each step), so it only peaks transiently.
const BATCH_SIZE: usize = 4;
const VALID_EXAMPLES: usize = 64;
const LR: f64 = 5.0e-7;

/// Policy + frozen reference. In the train step the policy runs on the autodiff
/// backend and the reference on the **inner** backend (graph-free), so the reference
/// contributes no gradient and — crucially — retains no autograd graph. Mirrors
/// `grpo::GrpoModel`.
#[derive(Module, Debug)]
pub struct DpoModel<B: Backend> {
    policy: GabrielLaevis<B>,
    reference: GabrielLaevis<B>,
}

impl<B: Backend> DpoModel<B> {
    /// Build the DPO loss + reward-accuracy output from policy and reference
    /// per-sequence logprobs (all on backend `B`).
    fn assemble(
        pi_c: Tensor<B, 1>,
        pi_r: Tensor<B, 1>,
        ref_c: Tensor<B, 1>,
        ref_r: Tensor<B, 1>,
    ) -> ClassificationOutput<B> {
        // β·(Δπ - Δref).
        let margin = pi_c.sub(pi_r).sub(ref_c.sub(ref_r)).mul_scalar(DPO_BETA); // [B]

        // loss = -log σ(margin) = softplus(-margin).
        let loss = activation::softplus(margin.clone().neg(), 1.0).mean();

        // Frame as binary "is chosen preferred": logits [0, margin], target = 1,
        // so accuracy = fraction with margin > 0 (the DPO reward accuracy).
        let [b] = margin.dims();
        let device = margin.device();
        let logits = Tensor::cat(
            vec![Tensor::zeros([b, 1], &device), margin.reshape([b, 1])],
            1,
        ); // [B,2]
        let targets = Tensor::<B, 1, Int>::ones([b], &device);
        ClassificationOutput::new(loss, logits, targets)
    }

    /// Validation forward: the whole model is already on the inner (non-autodiff)
    /// backend, so every forward is graph-free. Policy and reference both run plainly.
    fn output_eval(&self, batch: DpoBatch<B>) -> ClassificationOutput<B> {
        let pi_c = sequence_logprob(
            self.policy.forward(batch.chosen_in.clone()),
            batch.chosen_tgt.clone(),
        );
        let pi_r = sequence_logprob(
            self.policy.forward(batch.rejected_in.clone()),
            batch.rejected_tgt.clone(),
        );
        let ref_c = sequence_logprob(self.reference.forward(batch.chosen_in), batch.chosen_tgt);
        let ref_r = sequence_logprob(
            self.reference.forward(batch.rejected_in),
            batch.rejected_tgt,
        );
        Self::assemble(pi_c, pi_r, ref_c, ref_r)
    }
}

impl<B: AutodiffBackend> DpoModel<B> {
    /// Training forward. The policy runs on the autodiff backend (gradient flows); the
    /// frozen reference runs on the INNER backend via `.valid()`, building **no** graph.
    ///
    /// Why not autodiff + `.detach()`: detach only severs the *output* — the reference's
    /// full graph (every activation, the `[B,T,vocab]` logits) is still built, and since
    /// `backward()` only consumes the policy graph, the reference's graph nodes are never
    /// freed. They accumulate every step: an unbounded leak (~9 GB/iter → OOM by iter 10).
    /// The inner forward holds no graph, so its activations free at end of step.
    fn output_train(&self, batch: DpoBatch<B>) -> ClassificationOutput<B> {
        let pi_c = sequence_logprob(
            self.policy.forward(batch.chosen_in.clone()),
            batch.chosen_tgt.clone(),
        );
        let pi_r = sequence_logprob(
            self.policy.forward(batch.rejected_in.clone()),
            batch.rejected_tgt.clone(),
        );
        let reference = self.reference.valid(); // inner-backend module, no autograd
        let ref_c = Tensor::from_inner(sequence_logprob(
            reference.forward(batch.chosen_in.inner()),
            batch.chosen_tgt.inner(),
        ));
        let ref_r = Tensor::from_inner(sequence_logprob(
            reference.forward(batch.rejected_in.inner()),
            batch.rejected_tgt.inner(),
        ));
        Self::assemble(pi_c, pi_r, ref_c, ref_r)
    }
}

impl<B: AutodiffBackend> TrainStep for DpoModel<B> {
    type Input = DpoBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: DpoBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.output_train(batch);
        let grads = item.loss.backward();
        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> InferenceStep for DpoModel<B> {
    type Input = DpoBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: DpoBatch<B>) -> ClassificationOutput<B> {
        self.output_eval(batch)
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

    // Reference and policy both initialise from this base (the SFT model by default).
    let base = from.unwrap_or("model_sft");

    println!("loading {DPO_REPO} (up to {max_examples} pairs)...");
    let triples =
        load_dpo_triples(DPO_REPO, DPO_FILE, max_examples).wrap_err("loading dpo data")?;
    println!("tokenizing + scoring {} preference pairs...", triples.len());

    // Build chosen+rejected together per triple, dropping the whole triple if either
    // side has a byte the frozen-vocab tokenizer can't encode — this keeps the two row
    // lists aligned (they're zipped below) and skips unencodable data instead of aborting.
    let total = triples.len();
    let (chosen_rows, rejected_rows): (Vec<_>, Vec<_>) = triples
        .iter()
        .filter_map(|(p, c, r)| {
            let chosen = build_sft_row(&ctx.tokenizer, p, c, DPO_SEQ_LEN).ok()?;
            let rejected = build_sft_row(&ctx.tokenizer, p, r, DPO_SEQ_LEN).ok()?;
            Some((chosen, rejected))
        })
        .unzip();
    if chosen_rows.len() < total {
        println!("skipped {} unencodable pairs", total - chosen_rows.len());
    }

    let examples: Vec<DpoExample> = chosen_rows
        .into_iter()
        .zip(rejected_rows)
        .map(|(chosen, rejected)| DpoExample { chosen, rejected })
        .collect();
    let (examples, valid_examples) = split_valid(examples, VALID_EXAMPLES);

    // Policy + frozen reference, both from the same base checkpoint. The reference's
    // forward is detached in the step, so it never updates.
    let load = |what: &str| {
        GabrielLaevis::<crate::Train>::new(&cfg, &ctx.device)
            .load_file(
                ctx.checkpoint_path(base),
                &CompactRecorder::new(),
                &ctx.device,
            )
            .wrap_err_with(|| format!("loading {base} as {what} (run `train sft` first)"))
    };
    let model = DpoModel {
        policy: load("policy")?,
        reference: load("reference")?,
    };

    // Slice into shared 1000-step checkpoint-epochs with auto-resume (see grpo/pretrain).
    let sched = crate::train::schedule(examples.len(), BATCH_SIZE, epochs);
    println!(
        "training {} steps (~{epochs} passes over {} pairs) as {} x {}-step checkpoints",
        sched.total_steps,
        examples.len(),
        sched.ckpt_epochs,
        crate::train::STEPS_PER_CHECKPOINT.min(sched.total_steps),
    );

    let train = DataLoaderBuilder::new(DpoBatcher)
        .batch_size(BATCH_SIZE)
        .shuffle(SHUFFLE_SEED)
        .num_workers(NUM_WORKERS)
        .build(SamplerDataset::new(
            DpoDataset::new(examples),
            sched.epoch_samples,
        ));
    let valid = DataLoaderBuilder::new(DpoBatcher)
        .batch_size(BATCH_SIZE)
        .num_workers(NUM_WORKERS)
        .build(DpoDataset::new(valid_examples));

    let dpo_dir = ctx.checkpoint_path("dpo");
    let training = SupervisedTraining::new(dpo_dir.clone(), train, valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(sched.ckpt_epochs)
        .summary();

    let training = match (from, ctx.latest_checkpoint_epoch(&dpo_dir)) {
        (Some(_), _) | (None, None) => training,
        (None, Some(e)) => {
            println!("resuming dpo from checkpoint epoch {e}");
            training.checkpoint(e)
        }
    };

    let result = training.launch(Learner::new(model, AdamConfig::new().init(), LR));
    ctx.save_model(result.model.policy, "model_dpo")
        .wrap_err("saving dpo model")?;
    println!("saved DPO model to {}", ctx.checkpoint_path("model_dpo"));
    Ok(())
}
