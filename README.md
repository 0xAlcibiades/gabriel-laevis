# Gabriel Laevis

[![CI](https://github.com/0xAlcibiades/gabriel-laevis/actions/workflows/ci.yml/badge.svg)](https://github.com/0xAlcibiades/gabriel-laevis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

![Gabriel Laevis](assets/card.png)

A **Mamba-3** small language model implemented from scratch in [Burn](https://burn.dev).

## Stack

- **Framework:** Burn
- **Backend:** cubecl makes it portable
- **Tokenizer:** fastokens with byte-level BPE a la [nanochat](https://github.com/karpathy/nanochat)
- **Architecture:** [Mamba-3](https://arxiv.org/abs/2603.15569) SISO with a Llama-style backbone; each block pairs a Mamba-3 mixer ([SSD](https://arxiv.org/abs/2405.21060) scan, half-[RoPE](https://arxiv.org/abs/2104.09864)) with a [SwiGLU](https://arxiv.org/abs/2002.05202) MLP, pre-norm [RMSNorm](https://arxiv.org/abs/1910.07467), tied embeddings.
- **Chat format:** Gemma-4-style control tokens; a single Jinja chat template is the one render source — shared by `serve` and the training data path, and shipped beside the tokenizer for HuggingFace/vLLM — with a controllable `<|think|>` reasoning toggle.

## Status

A tiny model trained end to end on a single M2 Ultra Mac Pro in about a day. It produces coherent, on-format chat: it adopts the assistant register and stops at the turn boundary.

Pretraining, SFT, DPO all saw gains. GRPO runs cleanly but does not move the needle at that scale. Verifiable grade-school math is reachable for a small model along the [TinyGSM](https://arxiv.org/abs/2312.09241) path instead, which scales the verifier rather than the generator.

## Scaling

Dimensions are runtime config knobs, so doing a larger run is a config change away. Checkpointing and auto-resume are uniform across stages, long runs survive interruption.

## Usage

`train` trains the model, one stage per run; `serve` serves inference for a trained model. Run either with `--help` for the full set of args.

| Command              | What it does                                                                         |
| -------------------- | ------------------------------------------------------------------------------------ |
| `train pretrain`     | Pretrain on a decay-annealed web/math/code mix                                       |
| `train sft`          | Instruction-tune (ChatML, response-masked)                                           |
| `train reason`       | Teach chain-of-thought with an on/off toggle (R1-distilled CoT-trace SFT)            |
| `train dpo`          | Preference-optimize against a frozen reference (DPO)                                 |
| `train grpo`         | RL on [GSM8K](https://huggingface.co/datasets/openai/gsm8k), verifiable reward, reasoning-on rollouts |
| `train all`          | Run all five stages in order                                                         |
| `serve <checkpoint>` | OpenAI-compatible inference server                                                   |

Each `train` stage continues from the prior stage's checkpoint and auto-resumes from the latest, e.g.:

```sh
cargo run --profile=maxperf --features metal --bin train -- pretrain
cargo run --profile=maxperf --features metal --bin serve -- model
```

`MODEL_DIR` overrides where checkpoints, the config, and the tokenizer live (default `/tmp/gabriel-laevis`).

### Reasoning

The model has a controllable thinking channel, trained by `train reason` on R1-distilled CoT traces. At inference, set OpenAI's `reasoning_effort` on the chat request (`none`|`minimal`|`low`|`medium`|`high`|`xhigh`): `none` answers directly, anything else emits a `<|channel>thought>` chain-of-thought returned in a separate `reasoning_content` field (the DeepSeek/vLLM convention) while the answer goes to `content`.

### Crate Features

When building either binary you should pick one backend feature i.e. `ndarray`/`metal`/`wgpu`/`cuda`. If on a backend where its supported you can use `bf16` to halve memory and compute for training. Adding the `checkpoint` feature enables activation checkpointing, which trades off a bit of memory for compute. Training requires the `train` feature.

## Resources

**Framework & tooling**

- [Burn](https://burn.dev) — Rust deep-learning framework on the cubecl GPU kernel layer.
- [fastokens](https://github.com/crusoecloud/fastokens) — byte-level BPE tokenizer.
- [minijinja](https://github.com/mitsuhiko/minijinja) — Jinja renderer for the chat template.

**Models & data**

- [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) — small-LM training reference.
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — web pretraining corpus.
- [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) — math/reasoning pretraining corpus.
- [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) — code pretraining corpus.
- [OpenAssistant (oasst_top1)](https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25) — SFT instruction data.
- [Llama-Nemotron-Post-Training-Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) — R1-distilled CoT-trace reasoning SFT data.
- [orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k-flat) — DPO preference data.
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) — GRPO task.

## License

MIT, see [LICENSE](LICENSE).
