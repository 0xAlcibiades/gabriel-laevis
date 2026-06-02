//! Training-only model code, gated behind the `train` feature: the loss / Learner plumbing
//! (`forward_training`, `sequence_logprob`, and the `TrainStep`/`InferenceStep` impls) plus
//! the GRPO batched rollout (`generate_group` and its `step_tokens`/`pick_batch` helpers).
//! A child module of `lm`, so it reaches the model's private state without widening visibility.

use super::{GabrielLaevis, Sampling};
use crate::data::Batch;
use crate::model::mamba3::Mamba3State;
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep};
use eyre::{Result, eyre};
use rayon::prelude::*;

impl<B: Backend> GabrielLaevis<B> {
    /// Next-token cross-entropy over a batch, packaged for the Learner's metrics. Shared
    /// by pretrain and SFT.
    ///
    /// Reduction (and why SFT needs no separate "sum loss"): logits/targets are flattened
    /// to `[B*T]` and Burn's padded CE zeroes the `IGNORE_ID` positions, then means over
    /// the *full* `B*T` — i.e. `Σ(response-token NLL) / (B*T)`, a constant denominator. So
    /// every response token is weighted equally and longer responses contribute
    /// proportionally more: exactly the length-equitable Tulu-3 "sum" behavior, differing
    /// from a literal `sum()` only by the constant `B*T` (an LR rescale). The short-response
    /// -dominance pathology comes from a per-*sequence* mean (each example normalized by its
    /// own length); this flattened global mean never does that.
    pub fn forward_training(&self, batch: Batch<B>) -> ClassificationOutput<B> {
        let [b, t] = batch.inputs.dims();
        let v = self.vocab_size;
        let logits = self.forward(batch.inputs).reshape([b * t, v]);
        let targets = batch.targets.reshape([b * t]);
        let loss = self.criterion.forward(logits.clone(), targets.clone());
        ClassificationOutput::new(loss, logits, targets)
    }

    /// Batched cached step: `g` tokens `[g]` (Int, on device) + per-layer states for
    /// `g` rows → next-token logits `[g, vocab]` and updated states. The per-row
    /// recurrence is independent, so this is `step_token` over a batch.
    fn step_tokens(
        &self,
        tokens: Tensor<B, 1, Int>,
        states: Vec<Mamba3State<B>>,
    ) -> (Tensor<B, 2>, Vec<Mamba3State<B>>) {
        let g = tokens.dims()[0];
        let mut x = self.embed.forward(tokens.reshape([g, 1])); // [g,1,d_model]
        let mut next_states = Vec::with_capacity(states.len());
        for (layer, st) in self.layers.iter().zip(states) {
            let (xx, ns) = layer.step(x, st);
            x = xx;
            next_states.push(ns);
        }
        let x = self.norm.forward(x);
        let dm = x.dims()[2];
        let w_t = self.head_weight();
        let logits = x.reshape([g, dm]).matmul(w_t); // [g, vocab]
        (logits, next_states)
    }

    /// Generate `g` completions for a single `prompt` in one batched recurrent pass:
    /// the prompt is primed **once** at batch 1, its state fanned out to `g` rows, then
    /// `g` sequences are sampled in parallel. Each row stops at `sampling.stop_token`;
    /// rows can end at different lengths, so the returned completions are ragged. Only
    /// the temperature/seed/stop of `sampling` apply — the top-k/p/min-p filters and
    /// penalties are ignored. Errors on an empty prompt.
    pub fn generate_group(
        &self,
        prompt: &[i64],
        g: usize,
        max_new: usize,
        sampling: &Sampling,
        device: &B::Device,
    ) -> Result<Vec<Vec<i64>>> {
        if prompt.is_empty() {
            return Err(eyre!("prompt must be non-empty"));
        }
        // Prime the prompt once at batch 1, in a single parallel pass.
        let (logits1, states) = self.prime_prompt(prompt, device); // [vocab], per-layer states
        let vocab = logits1.dims()[0];

        // Fan the primed state + logits out to g rows. `expand` broadcasts the single
        // primed row to `[g, vocab]` as a stride-only view.
        let mut states: Vec<Mamba3State<B>> =
            states.into_iter().map(|st| st.broadcast_batch(g)).collect();
        let mut logits = logits1.reshape([1, vocab]).expand([g, vocab]); // [g, vocab]

        // One independent PRNG per row, deterministically forked from the seed. Lets the
        // per-token noise fill run row-parallel and keeps rows reproducible and independent.
        // Only built when sampling.
        let mut row_rngs: Option<Vec<fastrand::Rng>> =
            match (sampling.temperature > 0.0, sampling.seed) {
                (true, Some(s)) => {
                    let mut master = fastrand::Rng::with_seed(s);
                    Some((0..g).map(|_| master.fork()).collect())
                }
                _ => None,
            };
        // Output rows pre-sized to their cap.
        let mut out: Vec<Vec<i64>> = (0..g).map(|_| Vec::with_capacity(max_new)).collect();
        let mut finished = vec![false; g];

        for _ in 0..max_new {
            // Per-row noise filled in parallel.
            let noise = row_rngs.as_mut().map(|rngs| {
                let mut u = vec![0.0f32; g * vocab];
                u.par_chunks_mut(vocab)
                    .zip(rngs.par_iter_mut())
                    .for_each(|(chunk, r)| chunk.iter_mut().for_each(|slot| *slot = r.f32()));
                Tensor::<B, 2>::from_data(TensorData::new(u, [g, vocab]), device)
            });
            let ids = sampling.pick_batch(logits, noise); // [g] Int
            let ids_host: Vec<i64> = ids.clone().into_data().iter::<i64>().collect();
            let mut all_done = true;
            for (r, &id) in ids_host.iter().enumerate() {
                if finished[r] {
                    continue;
                }
                if sampling.stop_token == Some(id) {
                    finished[r] = true; // EOS: don't emit the stop token
                    continue;
                }
                out[r].push(id);
                all_done = false;
            }
            if all_done {
                break;
            }
            let (l, s) = self.step_tokens(ids, states);
            states = s;
            logits = l;
        }
        Ok(out)
    }
}

impl Sampling {
    /// Batched pick over `[g, vocab]` → `[g]` ids: greedy (temperature ≤ 0) or
    /// temperature Gumbel-max per row. Unlike [`Sampling::pick`] this applies **only**
    /// the temperature — the top-k/p/min-p filters and penalties are skipped.
    /// `noise`, if given, are seeded host uniforms `[g, vocab]`; otherwise the device
    /// RNG is used.
    fn pick_batch<B: Backend>(
        &self,
        logits: Tensor<B, 2>,
        noise: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 1, Int> {
        let [g, vocab] = logits.dims();
        if self.temperature <= 0.0 {
            return logits.argmax(1).reshape([g]);
        }
        let device = logits.device();
        let l = logits.div_scalar(self.temperature);
        // Upper clamp 0.99 for bf16/f16 safety — see `pick` for the dtype reasoning.
        let u = noise
            .unwrap_or_else(|| {
                Tensor::<B, 2>::random([g, vocab], Distribution::Uniform(0.0, 1.0), &device)
            })
            .clamp(1e-5, 0.99);
        let gumbel = u.log().neg().log().neg();
        l.add(gumbel).argmax(1).reshape([g])
    }
}

/// Per-sequence sum of target log-probabilities under `logits`, over positions
/// whose target is not `IGNORE_ID` (i.e. response tokens only — same masking as
/// SFT). `logits: [B, T, V]`, `targets: [B, T]` → `[B]`. Used by DPO for the
/// policy/reference sequence log-likelihoods.
pub fn sequence_logprob<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    let [b, t, v] = logits.dims();
    // logπ(target) = z_target − logΣexp(z). Compute it directly instead of forming the
    // dense [B,T,V] `log_softmax` only to `gather` one element per position: gather the
    // target logit ([B,T]) and subtract a max-shifted (overflow-safe) log-sum-exp over
    // the vocab ([B,T]). Identical to the log_softmax formulation (see
    // `sequence_logprob_matches_log_softmax`) but without the extra dense head-sized
    // activation.
    let idx = targets.clone().clamp(0, v as i64 - 1).reshape([b, t, 1]);
    let chosen = logits.clone().gather(2, idx).reshape([b, t]); // z_target  [B,T]
    let m = logits.clone().max_dim(2); // [B,T,1] shift for numerical stability
    let lse = logits
        .sub(m.clone())
        .exp()
        .sum_dim(2)
        .log()
        .add(m)
        .reshape([b, t]); // logΣexp(z)
    let logp = chosen.sub(lse); // log p(target_t)  [B,T]
    let mask = targets
        .not_equal_elem(crate::chat::IGNORE_ID as i64)
        .float(); // response positions [B,T]
    logp.mul(mask).sum_dim(1).reshape([b]) // [B]
}

impl<B: AutodiffBackend> TrainStep for GabrielLaevis<B> {
    type Input = Batch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Batch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_training(batch);
        let grads = item.loss.backward();
        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> InferenceStep for GabrielLaevis<B> {
    type Input = Batch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Batch<B>) -> ClassificationOutput<B> {
        self.forward_training(batch)
    }
}
