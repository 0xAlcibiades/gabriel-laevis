# TODO

Backlog. Items are actionable; in-code debt uses the markers
(`BUG`/`FIXME`/`HACK`/`TODO`/`NOTE`).

## Performance

- [ ] **Fused cross-entropy kernel.** Compute the loss and its gradients straight from
      `hidden` and the tied embedding `W`, never materializing the `[B,T,vocab]` logits
      (cut-cross-entropy). Caps the dominant head-memory term and removes the logits HBM
      round-trips, so it is both lower-memory and somewhat faster on a vocab-heavy head. A
      naive chunked-loss version does NOT help: autodiff retains every chunk's logits for
      backward, so this requires a real fused kernel, not a Rust loop.
- [ ] **Fused scan kernel.** Port Mamba's selective scan to a single cubecl kernel —
      the SSD recurrence, rotations, and trapezoidal terms forward, plus the matching
      backward, since the backward of a scan is itself a scan. This is the real fix for
      the launch-bound dispatch storm (~40% GPU util on M1) and lifts the f32 chunk-size
      cliff at ~48. Hardest item in the repo; start with a cubecl capability check before
      committing.

## Recipe (SOTA-small)

Levers distilled from the small-LM literature, ordered by leverage ÷ effort. ZClip,
anneal, and vocab-shrink are independent and none touch `mamba3.rs`; vocab-shrink and
anneal fold into one re-pretrain. Scale (params) is the separate, dominant lever for the
SOTA goal — these are what you lock cheaply at small scale first.

- [ ] **ZClip adaptive gradient clipping.** Replace the fixed `GRAD_CLIP_NORM = 1.0`
      (pretrain) with z-score spike detection on the EMA mean/std of the gradient norm — clip
      only when `z > ~2.5`, reciprocal adjustment. Enables ~10× higher LR → same loss in far
      fewer steps, eliminates loss spikes, no per-model threshold tuning. Needs a custom
      learner step (Burn's `GradientClippingConfig::Norm` is fixed-threshold only). Cheap and
      compounds — it speeds every experiment after it. Ref: _ZClip: Adaptive Spike Mitigation
      for LLM Pre-Training_, arXiv:2504.02507.
- [ ] **Multi-stage pretrain + decay anneal.** Vary the dataloader mix over training
      instead of one static blend: web-heavy bulk early, then upweight the highest-quality +
      math/reasoning data during the cosine-LR decay tail. Nearly free given we already run a
      cosine schedule — swap the sampling mix in the last ~10–20% of steps. Also plants
      reasoning behavior in pretraining rather than deferring it to GRPO. Refs: SmolLM3
      pretraining-datasets collection (3-stage: 85/12/3 web/code/math → math+code ramp →
      reasoning-heavy decay), HuggingFaceTB; _SmolLM2_, arXiv:2502.02737.
- [ ] **Eval harness (lm-eval + logprobs).** Run EleutherAI's lm-evaluation-harness
      against the OpenAI-compatible `serve` endpoint. Generative tasks (GSM8K) work as-is;
      multiple-choice tasks (MMLU/ARC/HellaSwag/WinoGrande) are loglikelihood-scored, so add
      token logprobs to the completions response first. This is the scoreboard — required
      because loss/perplexity is an unreliable proxy for downstream, especially across
      tokenizer changes. Refs: _SuperBPE_ arXiv:2503.13423 and _Fixing It in Post (TuluTalk)_
      arXiv:2506.06522 both show BPB/loss diverging from task accuracy.
- [ ] **Sum-reduction SFT loss.** Reduce the SFT cross-entropy by token-sum, not mean, so
      short responses don't dominate the gradient (length-equitable weighting). Trivial; SFT
      stage only (moot for fixed-window pretrain). Ref: Tulu 3 / Open-Instruct, via _TuluTalk_
      arXiv:2506.06522.

## Future

- [ ] **Byte-level / tokenizer-free fork.** Drop the tokenizer and the
      embedding sink entirely: model raw bytes with dynamic/learned patching. Mamba is the
      _right_ backbone for this — linear-in-length makes byte sequences affordable where they
      cripple a transformer, and it removes the ~36% vocab table outright. The most original
      SOTA-small direction available given the SSM choice. It is mutually exclusive with
      vocab-shrink and SuperBPE. Refs: _MambaByte_ arXiv:2401.13660 (bytes straight into Mamba);
      _Byte Latent Transformer_ arXiv:2412.09871 (entropy-based dynamic patching); _H-Net: Dynamic
      Chunking for End-to-End Hierarchical Sequence Modeling_ arXiv:2507.07955 (Gu — learned
      boundaries + SSM); _Super Tiny Language Models_ arXiv:2405.14159 (BPE-box byte pooling).
- [ ] **TinyGSM-style math.** The proven small-model GSM8K path (not GRPO): synthetic
      code-solution SFT data + code-execution reward + a verifier model (gen + token
      head, best-of-N). Scaling the verifier beats scaling the generator.
- [ ] **RLHF / PPO.** Reward model (scalar head on preference data) + actor-critic +
      rollouts + clipped objective + KL.
- [ ] **Constitutional / RLAIF.** AI-generated critiques/revisions against written
      principles produce the preference pairs that then feed DPO (or a reward model).
