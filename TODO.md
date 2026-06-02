# TODO

Backlog. Items are actionable; in-code debt uses the markers
(`BUG`/`FIXME`/`HACK`/`TODO`/`NOTE`).

- [ ] **Fused scan kernel.** The chunked SSD scan is dozens of small tensor ops, so it's
      launch-bound (~40% GPU util on M1) and the f32 path cliffs at chunk ~48. Cheap first
      step: wrap the compute backend in `Fusion<B>` (the `fusion` feature) so cubecl
      auto-fuses the elementwise decay/mask/cumsum glue — measure that before writing a line
      of kernel. The real fix is one cubecl `#[cube]` kernel: in-chunk and state matmuls via
      `cubecl::linalg`, the cumulative log-decay segment-sum via `cubecl::reduce`, the
      inter-chunk recurrence as a sequential loop, launched through `burn-cubecl` and wired
      into `Mamba3Block` as a custom op. Its backward is itself a scan, so register a matching
      autodiff `Backward` (Burn's `OpsPrep`/`OpsKind`, as `linear`/`embedding` do) that runs
      the same kernel shape in reverse. `scan_matches_step`,
      `forward_with_state_matches_step_carry`, and `chunk_precision_sweep` already
      pin scan output to the serial `step` recurrence — they gate the kernel as-is.
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
- [ ] **Byte-level / tokenizer-free fork.** Drop the tokenizer and the embedding table
      entirely: model raw bytes with dynamic/learned patching. Mamba is the _right_
      backbone for this — linear-in-length makes byte sequences affordable where they
      cripple a transformer, and it removes the vocab table outright (now ~14% of params at
      the 16k vocab, was ~36% at SmolLM2's 49k). The most original SOTA-small direction
      available given the SSM choice; a fork from the just-shipped trained-BPE path, not a
      tweak to it. Refs: _MambaByte_ arXiv:2401.13660 (bytes straight into Mamba); _Byte
      Latent Transformer_ arXiv:2412.09871 (entropy-based dynamic patching); _H-Net_
      arXiv:2507.07955 (Gu — learned boundaries + SSM); _Super Tiny Language Models_
      arXiv:2405.14159 (BPE-box byte pooling).
- [ ] **Hybrid attention–Mamba layers.** Interleave a few full-attention layers into the
      SSM stack. A pure SSM carries a fixed-size state, so exact long-range recall / in-
      context retrieval (copying, KV lookup) degrades; a handful of attention layers
      restores it at near-linear overall cost. Needs an attention-mixer variant alongside
      `Mamba3Block` (the block already abstracts the mixer) reusing the existing half-RoPE,
      and — the real work — a hybrid decode path: recurrent state for SSM layers _plus_ a
      KV cache for the attention layers (inference is pure recurrent state today). The
      attention layers want FA-style / sliding-window kernels in cubecl. Refs: _Jamba_
      arXiv:2403.19887; _Samba_ arXiv:2406.07522 (Mamba + sliding-window attention); NVIDIA
      _An Empirical Study of Mamba-based LMs_ arXiv:2406.07887 (a few attn layers close the
      recall gap).
- [ ] **Multi-token prediction + self-speculative draft.** Train auxiliary heads that
      predict t+2, t+3, … alongside the next-token head; at inference they act as a built-in
      draft for self-speculative decoding (propose a run, verify with the main head, accept
      the matching prefix). Strong fit for an SSM: decode is O(1)/token and bandwidth-bound
      on the tied head matmul, which speculation amortizes. Wrinkle: verify needs a parallel
      forward from a _saved mid-sequence state_, and accept/reject must advance-or-roll-back
      the recurrent state — `forward_with_state` only starts from a zero state today, so it
      must learn to resume from an arbitrary carried state. Refs: multi-token prediction
      (Gloeckle et al.) arXiv:2404.19737; speculative decoding (Leviathan et al.)
      arXiv:2211.17192; _Medusa_ arXiv:2401.10774.
- [ ] **Tool-use loop.** A generation loop that runs tool calls inline: the model emits a
      tool-call block, generation pauses, a sandboxed executor runs it, and the result is
      injected back as forced tokens before decoding continues. The tokenizer already
      reserves the control tokens (`<|tool>`/`<|tool_call>`/`<|tool_response>`, see
      `chat.rs`), so the work is (a) a forced-token injection path in `generate_with` (a
      pure sample loop today), (b) a sandboxed code executor, (c) tool-use training data +
      GRPO reward shaping. This is the infra the TinyGSM item needs (code-execution reward).
      Ref: _ToRA_ arXiv:2309.17452.
- [ ] **Distributed training.** The prerequisite for the 0.5B-class scale that "toy →
      coherent" needs (and only reachable on CUDA, where bf16 works; Metal is effectively
      single-device). Mostly adoption, not a build — Burn 0.21 already ships data-parallel
      training: `ExecutionStrategy::MultiDevice(devices, MultiDeviceOptim)` on the supervised
      `Learner` (one worker per device, gradients synced via `burn-collective`'s ring/tree
      all-reduce), `MultiDeviceOptim::OptimSharded` for ZeRO-style optimizer-state sharding,
      and a `ddp` strategy + `burn-communication`/`burn-remote` for multi-node. Work is wiring
      `SupervisedTraining` to a MultiDevice strategy + collective config and feeding per-device
      batches — not hand-rolling DDP/ZeRO the way nanochat does.
- [ ] **Evaluate via the OpenAI endpoint, don't build a harness.** Flesh out `serve` so
      external harnesses (lm-eval-harness, pi) drive all evaluation over the OpenAI-
      compatible API — no in-repo CORE/bpb/chat-eval code. Generative tasks (GSM8K,
      HumanEval) already work via chat/completions; the missing piece is a **logprobs /
      prompt-scoring path** for likelihood-style multiple-choice tasks (ARC, MMLU,
      HellaSwag), which need per-token logprobs over prompt+continuation rather than
      sampling — the model can produce them (`forward` logits / `sequence_logprob`), the
      server just doesn't expose them. Add `logprobs`/`echo`, then point the harness at the
      endpoint.
- [ ] **TinyGSM-style math.** The proven small-model GSM8K path (not GRPO): synthetic
      code-solution SFT data + code-execution reward + a verifier model (gen + token
      head, best-of-N). Scaling the verifier beats scaling the generator.
- [ ] **RLHF / PPO.** Reward model (scalar head on preference data) + actor-critic +
      rollouts + clipped objective + KL.
- [ ] **Constitutional / RLAIF.** AI-generated critiques/revisions against written
      principles produce the preference pairs that then feed DPO (or a reward model).
