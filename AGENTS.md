# AGENTS.md

From-scratch Mamba-3 small language model in Burn 0.21 / cubecl (Rust), ~45M params at
the defaults. This file is the working notes; see References for the overview and API.

## Build & test

- Default build is CPU (ndarray): `cargo build`, `cargo test`. CI (`.github/workflows/ci.yml`)
  runs fmt, clippy `-D warnings`, test, and coverage on the default features.
- Before committing: `cargo fmt --all`, `cargo clippy --all-targets -- -D warnings`, `cargo test`.
- Pick exactly one GPU backend feature; never combine them: `metal`, `cuda`, or `wgpu`.
  `bf16` selects a bf16 `Elem` (no-op on CPU); pair it with a backend, e.g. `cuda,bf16`.
  `checkpoint` trades compute for activation memory.
- **Never `--all-features`** — it enables every GPU backend at once and won't build.
- Real runs add `--profile=maxperf` (fat LTO, slow compile, big speedup):
  `cargo run --profile=maxperf --features metal --bin train -- pretrain`.

## Running

- `train <stage>`: `pretrain` (FineWeb-Edu) → `sft` (oasst_top1, ChatML, response-masked)
  → `dpo` (orpo-mix) → `grpo` (GSM8K, verifiable reward); `train all` runs the chain.
  Each stage starts from the prior checkpoint, checkpoints on a fixed step interval, and
  auto-resumes — a crash loses at most one interval. `--help` for per-stage args.
- `serve <checkpoint>`: OpenAI-compatible axum server, SSE streaming, per-request `seed`.
- Resize without recompiling via `GL_*` env / a `GL_CONFIG` file (`src/config.rs`):
  `GL_BATCH` (lower on OOM), `GL_CHUNK` (SSD chunk; exact at any size, larger = slightly
  more float error), `GL_CACHE_GROUPS`, `GL_D_MODEL`, `GL_N_LAYERS`, `GL_SHARDS`, etc.
- `MODEL_DIR` holds checkpoints + token caches (default `/tmp/gabriel-laevis`); `HF_HOME`
  holds dataset/tokenizer downloads.

## Conventions

- Match surrounding style; `eyre` for fallible app/data paths.
- Commits: terse lowercase subject, **no** `Co-Authored-By` / generated-by footer; batch
  related work into one commit.
- Less code = fewer bugs. Don't add speculative abstraction or scaffolding for things
  already covered by training behavior + tests.

## References

- [The README](README.md) — status, scaling, dataset sources, paper citations.
- [The Burn skill](.agents/skills/using-burn/SKILL.md) — verified 0.21 / cubecl cheat-sheet; read before writing model, tensor, or training code.
- [The source code](src/lib.rs) — the source of truth; when in doubt, read it.
