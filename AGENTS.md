# AGENTS.md

From-scratch Mamba-3 small language model in Burn / cubecl (Rust).
This file is the working notes; see References for the overview and API.

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

- `train <stage>`: `tokenizer` → `pretrain` → `sft` → `dpo` → `grpo`; `train all` 
  runs the chain. Each stage starts from the prior checkpoint, checkpoints on a 
  fixed step interval, and auto-resumes — a crash loses at most one interval.
  `--help` for per-stage args.
- `serve <checkpoint>`: OpenAI-compatible axum server, SSE streaming, per-request `seed`.
- Resize without recompiling via `GL_*` env / a `GL_CONFIG` file (`src/config.rs`):
  `GL_BATCH` (lower on OOM), `GL_CHUNK` (SSD chunk; exact at any size, larger = slightly
  more float error), `GL_CACHE_GROUPS`, `GL_D_MODEL`, `GL_N_LAYERS`, `GL_SHARDS`, etc.
- **Multi-stage pretrain mix:** pretraining draws from a web/math/code
  mixture that upweights math+code over the cosine-LR decay tail (`MIX_*` in `pretrain.rs`,
  after SmolLM3). **All three domains are on by default** (web=FineWeb, math=FineMath,
  code=Stack-Edu). Shards live on the `refs/convert/parquet` ref for the auto-converted
  datasets (reliable `{config}/{split}/NNNN.parquet` naming); native repos like FineWeb use
  `main` — hence the per-domain `GL_*_REVISION`. Disable a domain with an empty shard list,
  e.g. `GL_MATH_SHARDS=""` / `GL_CODE_SHARDS=""` (web-only). Each domain runs its own
  streaming count pass, so keep per-domain shard counts small.
  - **Math** is a plain `text`-column parquet: `GL_MATH_SHARDS` / `GL_MATH_REPO` /
    `GL_MATH_REVISION` / `GL_MATH_TEXT_COLUMN`.
  - **Code (Stack-Edu)** ships SWHIDs (`blob_id`), not text — `SoftwareHeritageSource`
    reads `GL_CODE_BLOB_COLUMN` from the shard and fetches each blob's content from
    Software Heritage's public S3 (`content/{blob_id}`, gzipped). Each file is one HTTP
    GET (fetched in parallel), so `GL_CODE_MAX_FILES` caps how many blobs a shard
    contributes (default 4096) to bound startup fetch cost; raise it on a fast link.
- `MODEL_DIR` holds checkpoints + token caches (default `/tmp/gabriel-laevis`); `HF_HOME`
  holds dataset downloads.

## Conventions

- Match surrounding style; `eyre` for fallible app/data paths.
- Commits: terse lowercase subject, **no** `Co-Authored-By` / generated-by footer; batch
  related work into one commit.
- Less code = fewer bugs. Don't add speculative abstraction or scaffolding for things
  already covered by training behavior + tests.

## References

- [The README](README.md) — status, scaling, dataset sources, paper citations.
- [The Burn skill](.agents/skills/using-burn/SKILL.md) — verified burn / cubecl cheat-sheet; read before writing model, tensor, or training code.
- [The source code](src/lib.rs) — the source of truth; when in doubt, read it.
