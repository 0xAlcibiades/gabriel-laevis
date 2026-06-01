//! Data layer: tokenizer, parquet ingestion, and the per-stage datasets/batchers.
//!
//! Layout, top to bottom: tokenizer loading → shared parquet helpers → generic
//! batch/split helpers → pretraining (streaming corpus) → SFT → DPO → GRPO. Each
//! training stage owns one section: its loader, example, `Dataset`, batch, and
//! `Batcher` live together.

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::*;
use eyre::{Result, WrapErr};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Row, RowAccessor};

use crate::config::artifact_dir;

// ===========================================================================
// Tokenizer
// ===========================================================================

/// Load the byte-level BPE tokenizer, preferring the copy packaged beside the model
/// in the artifact dir and falling back to fetching `repo` from HF. fastokens 0.2 can
/// deserialize but not *run* the `Digits` pre-tokenizer, so it is rewritten in memory
/// to the equivalent `Split` regex, preserving SmolLM2's digit splitting. The
/// rewritten tokenizer is then written into the artifact dir, so a trained model is
/// self-contained and later loads work offline.
pub fn load_tokenizer(repo: &str) -> Result<fastokens::Tokenizer> {
    use hf_hub::api::sync::Api;

    let packaged = artifact_dir().join("tokenizer.json");
    let (raw, fetched) = match std::fs::read_to_string(&packaged) {
        Ok(s) => (s, false),
        Err(_) => {
            let path = Api::new()?.model(repo.to_string()).get("tokenizer.json")?;
            (std::fs::read_to_string(path)?, true)
        }
    };

    let mut json: serde_json::Value = serde_json::from_str(&raw)?;
    rewrite_digits_pretokenizer(&mut json);

    // First fetch: package the rewritten tokenizer beside the model so later loads —
    // including `serve` — read it locally without hitting HF. Best-effort: a write
    // failure just means the next load refetches.
    if fetched && let Ok(s) = serde_json::to_string(&json) {
        let _ = std::fs::write(&packaged, s);
    }
    Ok(fastokens::Tokenizer::from_json(json)?)
}

/// Rewrite every `Digits` pre-tokenizer to the equivalent `Split` regex, in place.
/// Idempotent: a packaged copy is already rewritten, so re-running is a no-op.
fn rewrite_digits_pretokenizer(json: &mut serde_json::Value) {
    let Some(pretoks) = json
        .pointer_mut("/pre_tokenizer/pretokenizers")
        .and_then(|v| v.as_array_mut())
    else {
        return;
    };
    for p in pretoks.iter_mut() {
        if p.get("type").and_then(|t| t.as_str()) != Some("Digits") {
            continue;
        }
        // individual_digits → each digit its own pre-token `\d`, else split runs of
        // digits `\d+`; behavior Isolated keeps them as tokens.
        let individual = p
            .get("individual_digits")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        *p = serde_json::json!({
            "type": "Split",
            "pattern": { "Regex": if individual { r"\d" } else { r"\d+" } },
            "behavior": "Isolated",
            "invert": false,
        });
    }
}

// ===========================================================================
// Shared parquet ingestion
// ===========================================================================
//
// The SFT/DPO/GRPO loaders all read a HF-hosted dataset parquet (the auto-converted
// `refs/convert/parquet` revision), resolve a few columns by name, and iterate rows
// until a cap. These three helpers factor out that shape so each loader is just its
// column names plus a per-row extractor.

/// Fetch + open a Hub dataset parquet file from the auto-converted parquet revision.
fn open_hub_dataset_parquet(repo: &str, file: &str) -> Result<SerializedFileReader<File>> {
    use hf_hub::api::sync::Api;
    use hf_hub::{Repo, RepoType};

    let path = Api::new()?
        .repo(Repo::with_revision(
            repo.to_string(),
            RepoType::Dataset,
            "refs/convert/parquet".to_string(),
        ))
        .get(file)?;
    let file = File::open(&path).wrap_err("opening dataset parquet")?;
    SerializedFileReader::new(file).wrap_err("opening parquet reader")
}

/// Index of the column named `name`, or a clean error naming the dataset and column.
fn column_index(reader: &SerializedFileReader<File>, name: &str) -> Result<usize> {
    reader
        .metadata()
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .position(|c| c.name() == name)
        .ok_or_else(|| eyre::eyre!("parquet missing `{name}` column"))
}

/// Iterate rows across every row group, applying `extract` to each and collecting the
/// `Some` results, stopping once `max` are gathered. `extract` returning `Ok(None)`
/// skips a row without counting it; returning `Err` aborts the whole read.
fn read_rows<T>(
    reader: &SerializedFileReader<File>,
    max: usize,
    mut extract: impl FnMut(&Row) -> Result<Option<T>>,
) -> Result<Vec<T>> {
    let mut out = Vec::new();
    for g in 0..reader.num_row_groups() {
        let group = reader.get_row_group(g).wrap_err("reading row group")?;
        for record in group.get_row_iter(None).wrap_err("row iterator")? {
            let row = record.wrap_err("decoding parquet row")?;
            if let Some(item) = extract(&row)? {
                out.push(item);
                if out.len() >= max {
                    return Ok(out);
                }
            }
        }
    }
    Ok(out)
}

// ===========================================================================
// Generic helpers shared across stages
// ===========================================================================

/// Split `rows` into `(train, valid)` with a **disjoint** held-out tail of up to
/// `max_valid` rows, capped at a fifth so a tiny run keeps a trainable majority.
pub fn split_valid<T>(mut rows: Vec<T>, max_valid: usize) -> (Vec<T>, Vec<T>) {
    let n_valid = max_valid.min(rows.len() / 5);
    let valid = rows.split_off(rows.len() - n_valid);
    (rows, valid)
}

/// A training batch: `inputs` and next-token `targets`, both `[batch, seq_len]`.
/// Shared by the pretraining and SFT batchers.
#[derive(Clone, Debug)]
pub struct Batch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

// ===========================================================================
// Pretraining: streaming, seekable corpus
// ===========================================================================

/// One row group's contiguous block of dense windows in the flattened global window
/// index: `n_windows` windows starting at global window `win_start`.
#[derive(Clone, Copy, Debug)]
struct GroupWindows {
    source: usize,
    group: usize,
    win_start: usize,
    n_windows: usize,
}

/// Dense windows a group yields from `n_tokens`: `floor(n_tokens / (seq_len+1))`. The
/// remainder (< seq_len+1 tokens) is dropped. Pure/testable.
fn group_window_count(n_tokens: usize, seq_len: usize) -> usize {
    n_tokens / (seq_len + 1)
}

/// Build the flattened window index from each source's per-group token counts, assigning
/// each group a contiguous block of global window indices. Returns the index and the
/// total window count. Pure (no IO) so the index math is unit-testable.
fn build_window_index(
    per_source_group_tokens: &[Vec<usize>],
    seq_len: usize,
) -> (Vec<GroupWindows>, usize) {
    let mut index = Vec::new();
    let mut win_start = 0;
    for (source, groups) in per_source_group_tokens.iter().enumerate() {
        for (group, &n_tokens) in groups.iter().enumerate() {
            let n_windows = group_window_count(n_tokens, seq_len);
            index.push(GroupWindows {
                source,
                group,
                win_start,
                n_windows,
            });
            win_start += n_windows;
        }
    }
    (index, win_start)
}

/// Resolve a global window index to `(group-descriptor index, local window in group)`,
/// or `None` if out of range. Binary search over the contiguous window blocks;
/// zero-window groups are empty ranges and are correctly skipped. Pure/testable.
fn resolve_window(index: &[GroupWindows], gw: usize) -> Option<(usize, usize)> {
    let i = index
        .binary_search_by(|g| {
            if gw < g.win_start {
                std::cmp::Ordering::Greater
            } else if gw >= g.win_start + g.n_windows {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .ok()?;
    Some((i, gw - index[i].win_start))
}

/// A source of documents addressable by row group — the parquet reader implements it;
/// tests use an in-memory fake. `Send + Sync` so the dataset works under multi-worker
/// dataloaders.
pub trait RowGroupSource: Send + Sync {
    /// Number of row groups (defines the index; read from metadata only).
    fn num_groups(&self) -> usize;
    /// The `text` of every document in row group `g` (the only point that reads data).
    fn group_texts(&self, g: usize) -> Result<Vec<String>>;
}

/// A FineWeb-style parquet shard. Construction reads only the footer (column index +
/// row-group count); `group_texts` reopens the file to read a single row group, so the
/// struct stays `Send + Sync` and holds no live reader. Misses pay one footer read plus
/// the (cheap) tokenize — never the whole shard in RAM.
pub struct ParquetShard {
    path: PathBuf,
    text_idx: usize,
    num_groups: usize,
}

impl ParquetShard {
    /// Open a local parquet file and read its metadata.
    pub fn open(path: PathBuf) -> Result<Self> {
        let file = File::open(&path).wrap_err("opening parquet shard")?;
        let reader = SerializedFileReader::new(file).wrap_err("opening parquet reader")?;
        let text_idx = column_index(&reader, "text")?;
        let num_groups = reader.metadata().num_row_groups();
        Ok(Self {
            path,
            text_idx,
            num_groups,
        })
    }

    /// Fetch shard `file` from `repo` via the HF cache, then open it.
    pub fn from_hub(repo: &str, file: &str) -> Result<Self> {
        use hf_hub::api::sync::Api;
        let path = Api::new()?.dataset(repo.to_string()).get(file)?;
        Self::open(path)
    }
}

impl RowGroupSource for ParquetShard {
    fn num_groups(&self) -> usize {
        self.num_groups
    }

    fn group_texts(&self, g: usize) -> Result<Vec<String>> {
        let file = File::open(&self.path).wrap_err("opening parquet shard")?;
        let reader = SerializedFileReader::new(file).wrap_err("opening parquet reader")?;
        let group = reader.get_row_group(g).wrap_err("reading row group")?;
        group
            .get_row_iter(None)
            .wrap_err("row iterator")?
            .map(|record| {
                Ok(record
                    .wrap_err("decoding parquet row")?
                    .get_string(self.text_idx)
                    .wrap_err("reading text field")?
                    .clone())
            })
            .collect()
    }
}

/// Seekable streaming corpus: dense `seq_len+1` windows packed per row group,
/// tokenized on demand with a bounded row-group cache. Cheap to clone (everything
/// shared); a sub-range `view` carves disjoint train/valid splits that share the cache.
#[derive(Clone)]
pub struct StreamingTokenDataset {
    sources: Arc<Vec<Box<dyn RowGroupSource>>>,
    index: Arc<Vec<GroupWindows>>,
    cache: moka::sync::Cache<u64, Arc<Vec<u32>>>,
    tokenizer: Arc<fastokens::Tokenizer>,
    seq_len: usize,
    offset: usize,
    len: usize,
}

impl StreamingTokenDataset {
    /// Build from already-opened sources. Runs a streaming count pass (tokenize each
    /// group, keep only its token count, discard tokens) to size the dense window index.
    /// `cache_groups` bounds how many groups' token streams are held at once during
    /// training (memory ≈ cache_groups × one group's token bytes).
    pub fn new(
        sources: Vec<Box<dyn RowGroupSource>>,
        tokenizer: Arc<fastokens::Tokenizer>,
        seq_len: usize,
        cache_groups: u64,
    ) -> Result<Self> {
        // Count pass: tokens per group, one group resident at a time.
        let mut per_source_group_tokens: Vec<Vec<usize>> = Vec::with_capacity(sources.len());
        for src in &sources {
            let n_groups = src.num_groups();
            let mut counts = Vec::with_capacity(n_groups);
            for g in 0..n_groups {
                let texts = src.group_texts(g)?;
                let toks = tokenizer
                    .encode_batch(&texts, false)
                    .wrap_err("tokenizing row group (count pass)")?;
                counts.push(toks.iter().map(|t| t.len()).sum());
            }
            per_source_group_tokens.push(counts);
        }
        let (index, total) = build_window_index(&per_source_group_tokens, seq_len);
        Ok(Self {
            sources: Arc::new(sources),
            index: Arc::new(index),
            cache: moka::sync::Cache::new(cache_groups),
            tokenizer,
            seq_len,
            offset: 0,
            len: total,
        })
    }

    /// Fetch + open the listed FineWeb shards and build the dataset.
    pub fn from_hub(
        tokenizer: Arc<fastokens::Tokenizer>,
        repo: &str,
        files: &[String],
        seq_len: usize,
        cache_groups: u64,
    ) -> Result<Self> {
        let mut sources: Vec<Box<dyn RowGroupSource>> = Vec::with_capacity(files.len());
        for file in files {
            sources.push(Box::new(ParquetShard::from_hub(repo, file)?));
        }
        Self::new(sources, tokenizer, seq_len, cache_groups)
    }

    /// Total dense windows across all sources (before any `view`).
    pub fn total(&self) -> usize {
        self.index
            .last()
            .map(|g| g.win_start + g.n_windows)
            .unwrap_or(0)
    }

    /// A sub-range view exposing global windows `range` as local indices, sharing the
    /// same sources and cache. Used to split disjoint train/valid sets.
    pub fn view(&self, range: std::ops::Range<usize>) -> Self {
        let mut v = self.clone();
        v.offset = range.start;
        v.len = range.end.saturating_sub(range.start);
        v
    }

    /// Tokenize + concatenate one row group into its dense token stream (miss path).
    fn load_group(&self, group_idx: usize) -> Result<Arc<Vec<u32>>> {
        let gw = self.index[group_idx];
        let texts = self.sources[gw.source].group_texts(gw.group)?;
        let toks = self
            .tokenizer
            .encode_batch(&texts, false)
            .wrap_err("tokenizing row group")?;
        let mut stream = Vec::with_capacity(toks.iter().map(|t| t.len()).sum());
        for t in &toks {
            stream.extend_from_slice(t);
        }
        Ok(Arc::new(stream))
    }
}

impl Dataset<Vec<i64>> for StreamingTokenDataset {
    fn get(&self, index: usize) -> Option<Vec<i64>> {
        if index >= self.len {
            return None;
        }
        let (group_idx, local) = resolve_window(&self.index, self.offset + index)?;
        // Cache hit → clone the Arc'd token stream; miss → tokenize once and insert. A
        // read error (corrupt shard) logs and drops the item rather than killing the run.
        let stream = self
            .cache
            .try_get_with(group_idx as u64, || self.load_group(group_idx))
            .map_err(|e| eprintln!("streaming dataset: row group {group_idx} failed: {e}"))
            .ok()?;
        // Dense window `local`: tokens [local*w .. local*w + w). The window index was
        // built from this same stream's length, so the slice is always in bounds.
        let w = self.seq_len + 1;
        let start = local * w;
        Some(stream[start..start + w].iter().map(|&t| t as i64).collect())
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[derive(Clone, Debug)]
pub struct TokenBatcher {
    pub seq_len: usize,
}

impl<B: Backend> Batcher<B, Vec<i64>, Batch<B>> for TokenBatcher {
    fn batch(&self, items: Vec<Vec<i64>>, device: &B::Device) -> Batch<B> {
        let bsz = items.len();
        let w = self.seq_len + 1;
        let flat: Vec<i64> = items.into_iter().flatten().collect();
        let full = Tensor::<B, 2, Int>::from_data(TensorData::new(flat, [bsz, w]), device);
        let inputs = full.clone().slice([0..bsz, 0..self.seq_len]);
        let targets = full.slice([0..bsz, 1..w]);
        Batch { inputs, targets }
    }
}

// ===========================================================================
// SFT: instruction tuning
// ===========================================================================

/// Load SFT `(prompt, response)` pairs from the OpenAssistant oasst_top1 parquet
/// (human-written, Apache-2.0). Each row's `text` is an already-ChatML thread
/// (`<|im_start|>user…<|im_end|>…<|im_start|>assistant…`); we take the **first**
/// user→assistant exchange (single-turn is plenty at this scale).
pub fn load_sft_pairs(repo: &str, file: &str, max: usize) -> Result<Vec<(String, String)>> {
    let reader = open_hub_dataset_parquet(repo, file)?;
    let text = column_index(&reader, "text")?;
    read_rows(&reader, max, |row| {
        // A row with no readable text column is skipped, not fatal.
        Ok(row.get_string(text).ok().and_then(|t| oasst_first_pair(t)))
    })
}

/// Extract the first `(user, assistant)` exchange from an oasst_top1 `text` thread —
/// already ChatML (`<|im_start|>user\n…<|im_end|>…<|im_start|>assistant\n…`).
fn oasst_first_pair(text: &str) -> Option<(String, String)> {
    let user = chatml_turn(text, "<|im_start|>user\n")?;
    let assistant = chatml_turn(text, "<|im_start|>assistant\n")?;
    (!user.is_empty() && !assistant.is_empty()).then_some((user, assistant))
}

/// Content of the first turn opened by `open`, up to its `<|im_end|>`.
fn chatml_turn(text: &str, open: &str) -> Option<String> {
    let rest = &text[text.find(open)? + open.len()..];
    let end = rest.find("<|im_end|>").unwrap_or(rest.len());
    Some(rest[..end].trim().to_string())
}

/// Build one padded SFT row `(inputs, targets)` of length `seq_len`: the ChatML
/// sequence, with prompt and pad target positions set to `IGNORE_ID` so the loss only
/// sees the assistant response.
pub fn build_sft_row(
    tok: &fastokens::Tokenizer,
    prompt: &str,
    response: &str,
    seq_len: usize,
) -> Result<(Vec<i64>, Vec<i64>)> {
    let ignore = crate::chat::IGNORE_ID as i64;
    let full: Vec<i64> = tok
        .encode(&crate::chat::render_full(prompt, response))
        .wrap_err("encoding sft example")?
        .into_iter()
        .map(|x| x as i64)
        .collect();
    let prompt_len = tok
        .encode(&crate::chat::render_prompt(prompt))
        .wrap_err("encoding sft prompt")?
        .len();

    let mut inputs = vec![ignore; seq_len];
    let mut targets = vec![ignore; seq_len];
    let n = full.len().min(seq_len + 1);
    for j in 0..n.saturating_sub(1) {
        inputs[j] = full[j];
        // target predicts full[j+1]; train only where that is a response token.
        targets[j] = if j + 1 >= prompt_len {
            full[j + 1]
        } else {
            ignore
        };
    }
    Ok((inputs, targets))
}

/// Pre-tokenized SFT rows; the trivial batcher just stacks them (tokenization happens
/// once up front, so this stays `Send + Sync` for the dataloader).
#[derive(Clone)]
pub struct SftDataset {
    rows: Vec<(Vec<i64>, Vec<i64>)>,
}

impl SftDataset {
    pub fn new(rows: Vec<(Vec<i64>, Vec<i64>)>) -> Self {
        Self { rows }
    }
}

impl Dataset<(Vec<i64>, Vec<i64>)> for SftDataset {
    fn get(&self, index: usize) -> Option<(Vec<i64>, Vec<i64>)> {
        self.rows.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.rows.len()
    }
}

#[derive(Clone, Debug)]
pub struct SftBatcher;

impl<B: Backend> Batcher<B, (Vec<i64>, Vec<i64>), Batch<B>> for SftBatcher {
    fn batch(&self, items: Vec<(Vec<i64>, Vec<i64>)>, device: &B::Device) -> Batch<B> {
        let bsz = items.len();
        let l = items.first().map(|(i, _)| i.len()).unwrap_or(0);
        let inputs: Vec<i64> = items.iter().flat_map(|(i, _)| i.iter().copied()).collect();
        let targets: Vec<i64> = items.iter().flat_map(|(_, t)| t.iter().copied()).collect();
        Batch {
            inputs: Tensor::<B, 2, Int>::from_data(TensorData::new(inputs, [bsz, l]), device),
            targets: Tensor::<B, 2, Int>::from_data(TensorData::new(targets, [bsz, l]), device),
        }
    }
}

// ===========================================================================
// DPO: preference optimization
// ===========================================================================

/// Load preference triples `(prompt, chosen, rejected)` from a parquet dataset on the
/// Hub (flat string columns), up to `max`.
pub fn load_dpo_triples(
    repo: &str,
    file: &str,
    max: usize,
) -> Result<Vec<(String, String, String)>> {
    let reader = open_hub_dataset_parquet(repo, file)?;
    let (prompt, chosen, rejected) = (
        column_index(&reader, "prompt")?,
        column_index(&reader, "chosen")?,
        column_index(&reader, "rejected")?,
    );
    read_rows(&reader, max, |row| {
        Ok(Some((
            row.get_string(prompt).wrap_err("prompt field")?.clone(),
            row.get_string(chosen).wrap_err("chosen field")?.clone(),
            row.get_string(rejected).wrap_err("rejected field")?.clone(),
        )))
    })
}

/// One DPO example: response-masked `(input, target)` rows for the chosen and rejected
/// continuations. The frozen-reference log-probs are computed inline in the train step,
/// not precomputed — at 1-2 epochs a cache costs a full upfront reference pass for no
/// saving.
#[derive(Clone, Debug)]
pub struct DpoExample {
    pub chosen: (Vec<i64>, Vec<i64>),
    pub rejected: (Vec<i64>, Vec<i64>),
}

#[derive(Clone)]
pub struct DpoDataset {
    rows: Vec<DpoExample>,
}

impl DpoDataset {
    pub fn new(rows: Vec<DpoExample>) -> Self {
        Self { rows }
    }
}

impl Dataset<DpoExample> for DpoDataset {
    fn get(&self, index: usize) -> Option<DpoExample> {
        self.rows.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.rows.len()
    }
}

/// A DPO batch: chosen/rejected response-masked sequences `[B, L]`. The reference
/// log-probs are computed inside the train step (see `dpo::DpoModel`), not carried here.
#[derive(Clone, Debug)]
pub struct DpoBatch<B: Backend> {
    pub chosen_in: Tensor<B, 2, Int>,
    pub chosen_tgt: Tensor<B, 2, Int>,
    pub rejected_in: Tensor<B, 2, Int>,
    pub rejected_tgt: Tensor<B, 2, Int>,
}

#[derive(Clone, Debug)]
pub struct DpoBatcher;

impl<B: Backend> Batcher<B, DpoExample, DpoBatch<B>> for DpoBatcher {
    fn batch(&self, items: Vec<DpoExample>, device: &B::Device) -> DpoBatch<B> {
        let bsz = items.len();
        let l = items.first().map(|e| e.chosen.0.len()).unwrap_or(0);
        let col = |select: &dyn Fn(&DpoExample) -> &Vec<i64>| -> Tensor<B, 2, Int> {
            let flat: Vec<i64> = items
                .iter()
                .flat_map(|e| select(e).iter().copied())
                .collect();
            Tensor::from_data(TensorData::new(flat, [bsz, l]), device)
        };
        DpoBatch {
            chosen_in: col(&|e| &e.chosen.0),
            chosen_tgt: col(&|e| &e.chosen.1),
            rejected_in: col(&|e| &e.rejected.0),
            rejected_tgt: col(&|e| &e.rejected.1),
        }
    }
}

// ===========================================================================
// GRPO: GSM8K with verifiable reward
// ===========================================================================

/// Load GSM8K `(question, answer_int)` pairs from the `main` parquet split, up to
/// `max`. The dataset's `answer` ends with `#### <int>`; we keep that int (commas
/// stripped). `rsplit("####")` always yields at least the whole string, so a row with
/// no marker keeps the raw trailing text rather than being dropped — matching the
/// original loader.
pub fn load_gsm8k(repo: &str, file: &str, max: usize) -> Result<Vec<(String, String)>> {
    let reader = open_hub_dataset_parquet(repo, file)?;
    let (question, answer) = (
        column_index(&reader, "question")?,
        column_index(&reader, "answer")?,
    );
    read_rows(&reader, max, |row| {
        let q = row.get_string(question).wrap_err("question field")?.clone();
        let a = row.get_string(answer).wrap_err("answer field")?;
        Ok(a.rsplit("####")
            .next()
            .map(|int| (q, int.trim().replace(',', ""))))
    })
}

/// One GRPO example: a tokenized prompt and its verifiable (integer) answer.
#[derive(Clone, Debug)]
pub struct GrpoExample {
    pub prompt: Vec<i64>,
    pub answer: String,
}

#[derive(Clone)]
pub struct GrpoDataset {
    rows: Vec<GrpoExample>,
}

impl GrpoDataset {
    pub fn new(rows: Vec<GrpoExample>) -> Self {
        Self { rows }
    }
}

impl Dataset<GrpoExample> for GrpoDataset {
    fn get(&self, index: usize) -> Option<GrpoExample> {
        self.rows.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.rows.len()
    }
}

/// A GRPO batch: raw prompts + answers passed through to the training step (which
/// samples completions and scores them). The tokenizer rides along (`Arc`, cheap) so
/// the step can decode sampled completions for the reward.
#[derive(Clone)]
pub struct GrpoBatch {
    pub prompts: Vec<Vec<i64>>,
    pub answers: Vec<String>,
    pub tokenizer: Arc<fastokens::Tokenizer>,
}

impl std::fmt::Debug for GrpoBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GrpoBatch({} prompts)", self.prompts.len())
    }
}

#[derive(Clone)]
pub struct GrpoBatcher {
    pub tokenizer: Arc<fastokens::Tokenizer>,
}

impl std::fmt::Debug for GrpoBatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GrpoBatcher")
    }
}

impl<B: Backend> Batcher<B, GrpoExample, GrpoBatch> for GrpoBatcher {
    fn batch(&self, items: Vec<GrpoExample>, _device: &B::Device) -> GrpoBatch {
        GrpoBatch {
            prompts: items.iter().map(|e| e.prompt.clone()).collect(),
            answers: items.iter().map(|e| e.answer.clone()).collect(),
            tokenizer: self.tokenizer.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oasst_pair_parses_first_chatml_exchange() {
        let text = "<|im_start|>user\nName 3 dogs<|im_end|>\n<|im_start|>assistant\nRex, Fido, Spot<|im_end|>\n<|im_start|>user\nmore<|im_end|>\n";
        assert_eq!(
            oasst_first_pair(text),
            Some(("Name 3 dogs".to_string(), "Rex, Fido, Spot".to_string()))
        );
        assert_eq!(oasst_first_pair("no markers here"), None);
    }

    #[test]
    fn split_valid_holds_out_a_disjoint_tail() {
        let (train, valid) = split_valid((0..100).collect::<Vec<i32>>(), 10);
        assert_eq!(valid, (90..100).collect::<Vec<_>>());
        assert_eq!(
            train,
            (0..90).collect::<Vec<_>>(),
            "train + valid disjoint, cover all"
        );
        // Capped at a fifth so a tiny run still has a trainable majority.
        let (train, valid) = split_valid((0..8).collect::<Vec<i32>>(), 10);
        assert_eq!(valid.len(), 1); // 8 / 5
        assert_eq!(train.len(), 7);
    }

    #[test]
    fn window_count_is_floor_over_full_width() {
        // w = seq_len+1 = 4. 10 tokens → 2 dense windows, remainder 2 dropped.
        assert_eq!(group_window_count(10, 3), 2);
        assert_eq!(group_window_count(8, 3), 2); // exact, no remainder
        assert_eq!(group_window_count(3, 3), 0); // fewer than one window
        assert_eq!(group_window_count(0, 3), 0);
    }

    #[test]
    fn window_index_assigns_contiguous_blocks_and_skips_empty_groups() {
        // w = 4. token counts: shard0 groups [10, 3], shard1 group [8]
        // → windows [2, 0, 2], total 4. The 0-window group must be skipped on resolve.
        let (index, total) = build_window_index(&[vec![10, 3], vec![8]], 3);
        assert_eq!(total, 4);
        assert_eq!(
            (index[0].source, index[0].win_start, index[0].n_windows),
            (0, 0, 2)
        );
        assert_eq!(
            (index[1].source, index[1].win_start, index[1].n_windows),
            (0, 2, 0)
        );
        assert_eq!(
            (index[2].source, index[2].win_start, index[2].n_windows),
            (1, 2, 2)
        );

        // Every global window resolves to (group descriptor, local window), skipping the
        // empty group 1 entirely.
        assert_eq!(resolve_window(&index, 0), Some((0, 0)));
        assert_eq!(resolve_window(&index, 1), Some((0, 1)));
        assert_eq!(resolve_window(&index, 2), Some((2, 0)));
        assert_eq!(resolve_window(&index, 3), Some((2, 1)));
        assert_eq!(resolve_window(&index, 4), None);
    }

    /// In-memory source: each row group is a list of documents. `get` needs a real
    /// tokenizer, so the streaming test loads the production one.
    struct FakeSource {
        groups: Vec<Vec<String>>,
    }
    impl RowGroupSource for FakeSource {
        fn num_groups(&self) -> usize {
            self.groups.len()
        }
        fn group_texts(&self, g: usize) -> Result<Vec<String>> {
            Ok(self.groups[g].clone())
        }
    }

    #[test]
    fn streaming_dataset_packs_densely_and_is_seekable() {
        // Real tokenizer so encode_batch matches production; ASCII letters tokenize to
        // stable ids. Two groups of docs; we only assert structural properties (dense,
        // seekable, disjoint views) that hold regardless of the exact ids.
        let tok = Arc::new(load_tokenizer(crate::constants::TOKENIZER_REPO).unwrap());
        let seq_len = 7;
        let w = seq_len + 1;
        // Documents long enough that each group concatenates to several full windows.
        let para = "the quick brown fox jumps over the lazy dog again and again ";
        let source = FakeSource {
            groups: vec![vec![para.repeat(3), para.repeat(2)], vec![para.repeat(4)]],
        };
        let ds = StreamingTokenDataset::new(vec![Box::new(source)], tok, seq_len, 8).unwrap();

        // len() equals the sum of per-group floor(tokens/w), and every window is exactly
        // w wide with no IGNORE_ID padding (dense packing — the whole point).
        let n = ds.len();
        assert!(n >= 1, "expected at least one dense window");
        for i in 0..n {
            let win = ds.get(i).expect("in-range window");
            assert_eq!(win.len(), w, "window {i} not full width");
            assert!(
                win.iter().all(|&t| t != crate::chat::IGNORE_ID as i64),
                "window {i} contains padding — packing is not dense"
            );
        }
        assert_eq!(ds.get(n), None, "out of range returns None");

        // Disjoint views over the window space partition the corpus exactly.
        let (a, b) = (n / 2, n);
        let train = ds.view(0..a);
        let valid = ds.view(a..b);
        assert_eq!(train.len() + valid.len(), n);
        assert_eq!(train.get(0), ds.get(0), "view 0 maps to global 0");
        if valid.len() > 0 {
            assert_eq!(
                valid.get(0),
                ds.get(a),
                "valid view offset maps to global a"
            );
        }
    }
}
