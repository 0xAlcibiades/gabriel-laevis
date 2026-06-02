//! The language model with embeddings, stacked layers, norm, and tied head.

use crate::config::ModelConfig;
use crate::model::block::Layer;
use crate::model::mamba3::Mamba3State;
use burn::module::Initializer;
use burn::module::Module;
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::nn::{Embedding, EmbeddingConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::activation;
use eyre::{Result, eyre};

// Training-only model code (loss/Learner plumbing + GRPO rollout) lives in the `train`-gated
// child module, keeping this file inference-only. `sequence_logprob` is re-exported so its
// `crate::model::lm::sequence_logprob` path is unchanged for DPO/GRPO.
#[cfg(feature = "train")]
mod training;
#[cfg(feature = "train")]
pub use training::sequence_logprob;

#[derive(Module, Debug)]
pub struct GabrielLaevis<B: Backend> {
    embed: Embedding<B>,
    layers: Vec<Layer<B>>,
    norm: RmsNorm<B>,
    vocab_size: usize,
    criterion: CrossEntropyLoss<B>,
}

impl<B: Backend> GabrielLaevis<B> {
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> Self {
        let layers = (0..cfg.n_layers).map(|_| Layer::new(cfg, device)).collect();
        Self {
            embed: EmbeddingConfig::new(cfg.vocab_size, cfg.d_model)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: cfg.init_std,
                })
                .init(device),
            layers,
            norm: RmsNormConfig::new(cfg.d_model).init(device),
            vocab_size: cfg.vocab_size,
            criterion: CrossEntropyLossConfig::new()
                .with_pad_tokens(Some(vec![crate::chat::IGNORE_ID]))
                .init(device),
        }
    }

    /// The device this model's parameters live on.
    ///
    /// NOTE:
    /// Reads one param, unlike `Module::devices()` which allocates a `Vec`
    /// and walks the whole module graph.
    pub fn device(&self) -> B::Device {
        self.embed.weight.val().device()
    }

    /// The tied output head: the embedding table transposed to `[d_model, vocab]`.
    fn head_weight(&self) -> Tensor<B, 2> {
        self.embed.weight.val().swap_dims(0, 1)
    }

    /// `tokens: [B, T]` (Int) → `logits: [B, T, vocab]`.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, t] = tokens.dims();
        let mut x = self.embed.forward(tokens); // [B, T, d_model]
        for layer in &self.layers {
            x = layer.forward(x);
        }
        let x = self.norm.forward(x);
        let dm = x.dims()[2];
        // Tied head: logits = x · Wᵀ, where W is the embedding table [vocab, d_model].
        let w_t = self.head_weight(); // [d_model, vocab]
        let logits = x.reshape([b * t, dm]).matmul(w_t); // [B*T, vocab]
        logits.reshape([b, t, self.vocab_size])
    }

    /// One cached forward step: a single token (a `[1]` Int tensor already on
    /// device) + per-layer recurrent states → next-token logits `[vocab]` and
    /// updated states.
    fn step_token(
        &self,
        token: Tensor<B, 1, Int>,
        states: Vec<Mamba3State<B>>,
    ) -> (Tensor<B, 1>, Vec<Mamba3State<B>>) {
        let mut x = self.embed.forward(token.reshape([1, 1])); // [1,1,d_model]
        let mut next_states = Vec::with_capacity(states.len());
        for (layer, st) in self.layers.iter().zip(states) {
            let (xx, ns) = layer.step(x, st);
            x = xx;
            next_states.push(ns);
        }
        let x = self.norm.forward(x);
        let dm = x.dims()[2];
        let w_t = self.head_weight();
        let logits = x.reshape([1, dm]).matmul(w_t).reshape([self.vocab_size]);
        (logits, next_states)
    }

    /// Prime a prompt in a single parallel pass: run all layers' parallel prefill over
    /// the `[1, T]` prompt, collecting each layer's carried recurrent state, and return
    /// the next-token logits `[vocab]` alongside the per-layer states ready for decoding.
    /// The caller guarantees a non-empty prompt.
    fn prime_prompt(
        &self,
        prompt: &[i64],
        device: &B::Device,
    ) -> (Tensor<B, 1>, Vec<Mamba3State<B>>) {
        let t = prompt.len();
        let prompt_ids =
            Tensor::<B, 1, Int>::from_data(TensorData::from(prompt), device).reshape([1, t]);
        let mut x = self.embed.forward(prompt_ids); // [1, T, d_model]
        let mut states = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (xx, st) = layer.forward_with_state(x);
            x = xx;
            states.push(st);
        }
        let x = self.norm.forward(x); // [1, T, d_model]
        let dm = x.dims()[2];
        // Only the last timestep's hidden feeds the next-token logits.
        let last = x.slice([0..1, t - 1..t, 0..dm]).reshape([1, dm]);
        let w_t = self.head_weight();
        let logits = last.matmul(w_t).reshape([self.vocab_size]);
        (logits, states)
    }

    /// Autoregressive generation with the O(1)-per-token recurrent state cache,
    /// under the given [`Sampling`] controls. Returns `max_new` ids. Errors on an
    /// empty prompt.
    pub fn generate(
        &self,
        prompt: &[i64],
        max_new: usize,
        sampling: &Sampling,
        device: &B::Device,
    ) -> Result<Vec<i64>> {
        let mut out = Vec::with_capacity(max_new);
        self.generate_with(prompt, max_new, sampling, device, |t| {
            out.push(t);
            true
        })?;
        Ok(out)
    }

    /// Streaming generation: invokes `on_token` with each new token id as it is
    /// produced. Returning `false` from `on_token` stops generation early — e.g.
    /// when a streaming client disconnects. Errors on an empty prompt.
    pub fn generate_with<F: FnMut(i64) -> bool>(
        &self,
        prompt: &[i64],
        max_new: usize,
        sampling: &Sampling,
        device: &B::Device,
        mut on_token: F,
    ) -> Result<()> {
        if prompt.is_empty() {
            return Err(eyre!("prompt must be non-empty"));
        }
        // Prime the prompt
        let (mut logits, mut states) = self.prime_prompt(prompt, device);

        // Per-token generated counts for the penalties, kept on-device as a [vocab]
        // tally only when a penalty is actually active.
        let vocab = logits.dims()[0];
        let penalize = sampling.frequency_penalty != 0.0 || sampling.presence_penalty != 0.0;
        let mut counts = penalize.then(|| Tensor::<B, 1>::zeros([vocab], device));
        // Seeded host-side sampling noise. Only built when sampling.
        let mut rng = match (sampling.temperature > 0.0, sampling.seed) {
            (true, Some(s)) => Some(fastrand::Rng::with_seed(s)),
            _ => None,
        };

        for _ in 0..max_new {
            // The chosen id stays a [1] tensor on-device — straight into the counts
            // tally and the next step. The single host sync is for `on_token`.
            let noise = rng.as_mut().map(|r| {
                let u: Vec<f32> = (0..vocab).map(|_| r.f32()).collect();
                Tensor::<B, 1>::from_data(TensorData::new(u, [vocab]), device)
            });
            let next = sampling.pick(logits, counts.as_ref(), noise);
            let id = next.clone().into_scalar().elem::<i64>();

            // EOS: Stop before emitting the stop token, so it never lands in the
            // output.
            if sampling.stop_token == Some(id) {
                break;
            }
            if let Some(c) = counts.as_mut() {
                *c = c
                    .clone()
                    .add(next.clone().one_hot::<2>(vocab).float().reshape([vocab]));
            }
            if !on_token(id) {
                break;
            }
            let (l, s) = self.step_token(next, states);
            states = s;
            logits = l;
        }
        Ok(())
    }
}

/// Sampling controls. `temperature <= 0` is greedy (`argmax`); otherwise the logits
/// are scaled by `temperature`, optionally filtered by top-k, then min-p, then
/// nucleus (top-p), and sampled. See [`Sampling::pick`].
#[derive(Clone, Debug)]
pub struct Sampling {
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub stop_token: Option<i64>,
    pub seed: Option<u64>,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: None,
            top_p: None,
            min_p: None,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_token: None,
            seed: None,
        }
    }
}

impl Sampling {
    /// Greedy decoding equivalent to argmax.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Default::default()
        }
    }

    /// Picks the next token from `logits` `[vocab]`, entirely on-device in the model's
    /// native dtype, returning the chosen id as a `[1]` Int tensor (so it flows into
    /// the next step and the penalty tally without a host round-trip). `counts` (when
    /// penalties are active) is the `[vocab]` tally of prior generations.
    pub fn pick<B: Backend>(
        &self,
        logits: Tensor<B, 1>,
        counts: Option<&Tensor<B, 1>>,
        noise: Option<Tensor<B, 1>>,
    ) -> Tensor<B, 1, Int> {
        let device = logits.device();
        let [vocab] = logits.dims();

        // Frequency/presence penalties as tensor subtractions.
        let mut l = logits;
        if let Some(counts) = counts {
            if self.frequency_penalty != 0.0 {
                l = l.sub(counts.clone().mul_scalar(self.frequency_penalty));
            }
            if self.presence_penalty != 0.0 {
                let seen = counts.clone().greater_elem(0.0).float();
                l = l.sub(seen.mul_scalar(self.presence_penalty));
            }
        }

        if self.temperature <= 0.0 {
            return l.argmax(0); // greedy
        }

        let l = self.filter(l.div_scalar(self.temperature));

        // Gumbel-max: argmax(logits + g) with g = -ln(-ln u), u ~ U(0,1), draws a
        // categorical sample ∝ softmax(logits) with no explicit softmax or multinomial.
        // `noise` (seeded host uniforms) is used when supplied, else the device RNG.
        //
        // Upper clamp is 0.99, not 1−1e-7: this runs in the model's native dtype, and in
        // bf16 the spacing near 1.0 is 2⁻⁸ ≈ 3.9e-3, so anything above ~0.996 rounds to
        // exactly 1.0 → log(1)=0 → log(0)=−inf → +inf Gumbel → a poisoned argmax. 0.99 is
        // the largest bound that stays strictly below 1.0 in bf16/f16/f32 alike.
        let u = noise
            .unwrap_or_else(|| {
                Tensor::<B, 1>::random([vocab], Distribution::Uniform(0.0, 1.0), &device)
            })
            .clamp(1e-5, 0.99);
        let gumbel = u.log().neg().log().neg();
        l.add(gumbel).argmax(0)
    }

    /// Filter the logits to the sampling support, setting rejected tokens to `-inf`
    /// (so they survive neither softmax nor argmax): top-k, then min-p, then nucleus
    /// (top-p), each composing on the previous mask.
    fn filter<B: Backend>(&self, mut l: Tensor<B, 1>) -> Tensor<B, 1> {
        let neg_inf = f32::NEG_INFINITY;
        let [vocab] = l.dims();

        // top-k: keep the k largest; the k-th largest is the threshold.
        if let Some(k) = self.top_k
            && k < vocab
        {
            let kth = l.clone().topk(k, 0).min(); // [1], smallest of the top-k
            let reject = l.clone().lower(kth);
            l = l.mask_fill(reject, neg_inf);
        }

        // min-p: drop tokens whose prob is below `min_p × max_prob`.
        if let Some(min_p) = self.min_p {
            let p = activation::softmax(l.clone(), 0);
            let cutoff = p.clone().max().mul_scalar(min_p); // [1]
            let reject = p.lower(cutoff);
            l = l.mask_fill(reject, neg_inf);
        }

        // nucleus: keep the smallest prefix of the prob mass reaching `top_p`. The
        // exclusive prefix sum `< top_p` keeps the crossing token and always the top
        // one; the cutoff prob is the smallest kept (sorted descending), and any
        // token below it is dropped.
        //
        // The descending sort is O(V·log V) and irreducible here: Burn's `topk` is itself
        // a full `sort_descending` + `select`, so a "truncate-with-topk-first" pass would
        // be a second sort, not a saving. We do use `sort_descending` with values only
        // rather than `sort_descending_with_indices`
        if let Some(top_p) = self.top_p {
            let p = activation::softmax(l.clone(), 0);
            let sorted = p.clone().sort_descending(0);
            let prefix = sorted.clone().cumsum(0).sub(sorted.clone()); // exclusive
            let keep = prefix.lower_elem(top_p);
            let cutoff = sorted.mask_fill(keep.bool_not(), f32::INFINITY).min(); // [1]
            let reject = p.lower(cutoff);
            l = l.mask_fill(reject, neg_inf);
        }
        l
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn::backend::NdArray;

    fn small(vocab: usize) -> ModelConfig {
        ModelConfig::new()
            .with_vocab_size(vocab)
            .with_d_model(64)
            .with_n_layers(2)
            .with_d_state(32)
            .with_d_ff(128)
    }

    /// The hand-rolled (gather + max-shifted log-sum-exp) `sequence_logprob` must equal
    /// the dense `log_softmax(logits).gather(target)` formulation it replaced, including
    /// the IGNORE_ID masking. This is the correctness guard for the DPO log-likelihoods.
    #[test]
    fn sequence_logprob_matches_log_softmax() {
        let device = Default::default();
        let (bsz, t, v) = (2usize, 6usize, 17usize);
        let logits =
            Tensor::<B, 3>::random([bsz, t, v], burn::tensor::Distribution::Default, &device);

        // Targets in-range, with two IGNORE_ID positions to exercise the mask.
        let mut tg: Vec<i64> = (0..(bsz * t)).map(|i| (i % v) as i64).collect();
        tg[0] = crate::chat::IGNORE_ID as i64;
        tg[t + 1] = crate::chat::IGNORE_ID as i64;
        let targets = Tensor::<B, 2, Int>::from_data(TensorData::new(tg, [bsz, t]), &device);

        // Reference: the original dense formulation.
        let logp_dense = activation::log_softmax(logits.clone(), 2);
        let idx = targets.clone().clamp(0, v as i64 - 1).reshape([bsz, t, 1]);
        let chosen = logp_dense.gather(2, idx).reshape([bsz, t]);
        let mask = targets
            .clone()
            .not_equal_elem(crate::chat::IGNORE_ID as i64)
            .float();
        let reference = chosen.mul(mask).sum_dim(1).reshape([bsz]);

        let got = sequence_logprob(logits, targets);
        let diff = (got - reference)
            .abs()
            .max()
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(
            diff < 1e-5,
            "sequence_logprob diverged from log_softmax: {diff}"
        );
    }

    #[test]
    fn lm_forward_shape() {
        let device = Default::default();
        let cfg = small(100);
        let model = GabrielLaevis::<B>::new(&cfg, &device);
        let tokens = Tensor::<B, 2, Int>::zeros([2, 7], &device);
        let logits = model.forward(tokens);
        assert_eq!(logits.dims(), [2, 7, 100]);
    }

    #[test]
    fn generate_greedy_length() {
        let device = Default::default();
        let cfg = small(100);
        let model = GabrielLaevis::<B>::new(&cfg, &device);
        let out = model
            .generate(&[1, 2, 3], 5, &Sampling::greedy(), &device)
            .unwrap();
        assert_eq!(out.len(), 5);
        assert!(out.iter().all(|&t| (0..100).contains(&t)));
    }

    #[test]
    fn pick_is_on_device_and_selects_correctly() {
        let device = Default::default();
        // Peaked logits: token 3 is the clear max.
        let logits = Tensor::<B, 1>::from_floats([0.0, 1.0, 0.5, 9.0, 2.0, 0.1, 0.0, 3.0], &device);

        let pick = |s: &Sampling| {
            s.pick(logits.clone(), None, None)
                .into_scalar()
                .elem::<i64>()
        };

        // Greedy (temp 0) is the argmax.
        assert_eq!(pick(&Sampling::greedy()), 3);

        // top_k = 1 leaves a single survivor, so Gumbel-max collapses to argmax —
        // deterministic regardless of the sampling noise. Exercises filter + sample.
        let top1 = Sampling {
            temperature: 1.0,
            top_k: Some(1),
            ..Default::default()
        };
        for _ in 0..20 {
            assert_eq!(pick(&top1), 3);
        }
    }

    #[test]
    fn seeded_generation_is_reproducible() {
        let device = Default::default();
        let cfg = small(100);
        let model = GabrielLaevis::<B>::new(&cfg, &device);
        let sampling = Sampling {
            temperature: 1.0,
            seed: Some(1234),
            ..Default::default()
        };
        let a = model.generate(&[1, 2, 3], 12, &sampling, &device).unwrap();
        let b = model.generate(&[1, 2, 3], 12, &sampling, &device).unwrap();
        assert_eq!(a, b, "same seed must give identical samples");
    }

    /// The batched group rollout must reproduce the serial greedy path exactly, for every row.
    #[test]
    fn generate_group_matches_serial_greedy() {
        let device = Default::default();
        let cfg = small(100);
        let model = GabrielLaevis::<B>::new(&cfg, &device);
        let prompt = [1i64, 5, 9, 2, 7];
        let g = 3;
        let greedy = Sampling::greedy();
        let serial = model.generate(&prompt, 8, &greedy, &device).unwrap();
        let group = model
            .generate_group(&prompt, g, 8, &greedy, &device)
            .unwrap();
        assert_eq!(group.len(), g);
        for row in &group {
            assert_eq!(row, &serial, "batched greedy row must equal serial greedy");
        }
    }
}
