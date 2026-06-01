# TODO

Backlog. Items are actionable; in-code debt uses the markers
(`BUG`/`FIXME`/`HACK`/`TODO`/`NOTE`).

## Future

- [ ] **Fused scan kernel.** Port Mamba's selective scan to a single cubecl kernel —
      the SSD recurrence, rotations, and trapezoidal terms forward, plus the matching
      backward, since the backward of a scan is itself a scan. This is the real fix for
      the launch-bound dispatch storm (~40% GPU util on M1) and lifts the f32 chunk-size
      cliff at ~48. Hardest item in the repo; start with a cubecl capability check before
      committing.
- [ ] **Fused cross-entropy kernel.** Compute the loss and its gradients straight from
      `hidden` and the tied embedding `W`, never materializing the `[B,T,vocab]` logits
      or their gradient (cut-cross-entropy). Caps the dominant head-memory term and removes
      the logits HBM round-trips. Auto-fusion alone won't do it: under autodiff the full
      logits + their grad are materialized for the matmul/softmax backward regardless of
      forward fusion. The memory win needs a custom chunked VJP (forward + backward loop
      over vocab blocks on the inner backend, recomputing each block's logits in backward —
      no full logits ever); a true cubecl fused kernel is the maximal version that also
      kills the per-block round-trips. Descend-to-the-metal pass — pairs with the fused
      scan kernel; deferred until the model/recipe is firm. Ref: cut-cross-entropy
      (Wijmans et al. 2024), arXiv:2411.09009.
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
