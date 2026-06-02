//! Data layer

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::*;
use eyre::{Result, WrapErr};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Row, RowAccessor};
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// ===========================================================================
// Dataset ingestion
// ===========================================================================

/// Fetch and open a Hub dataset parquet file from the auto-converted parquet revision.
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

/// Split `rows` into `(train, valid)` with a disjoint held-out tail of up to
/// `max_valid` rows, capped at a fifth so a tiny run keeps a trainable majority.
pub fn split_valid<T>(mut rows: Vec<T>, max_valid: usize) -> (Vec<T>, Vec<T>) {
    let n_valid = max_valid.min(rows.len() / 5);
    let valid = rows.split_off(rows.len() - n_valid);
    (rows, valid)
}

/// A training batch: `inputs` and next-token `targets`, both `[batch, seq_len]`.
#[derive(Clone, Debug)]
pub struct Batch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

// ===========================================================================
// Pretraining using a streaming, seekable corpus
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
/// remainder (< seq_len+1 tokens) is dropped.
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
/// zero-window groups are empty ranges and are correctly skipped.
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

/// A source of documents addressable by row group.
pub trait RowGroupSource: Send + Sync {
    /// Number of row groups.
    fn num_groups(&self) -> usize;
    /// The `text` of every document in row group `g`.
    fn group_texts(&self, g: usize) -> Result<Vec<String>>;
}

/// A FineWeb-style parquet shard. Construction reads only the footer (column index +
/// row-group count); `group_texts` reopens the file to read a single row group, so the
/// struct stays `Send + Sync` and holds no live reader. Misses pay one footer read plus
/// the tokenize.
pub struct ParquetShard {
    path: PathBuf,
    text_idx: usize,
    num_groups: usize,
}

impl ParquetShard {
    /// Open a local parquet file and read its metadata, reading documents from the column.
    pub fn open(path: PathBuf, text_column: &str) -> Result<Self> {
        let file = File::open(&path).wrap_err("opening parquet shard")?;
        let reader = SerializedFileReader::new(file).wrap_err("opening parquet reader")?;
        let text_idx = column_index(&reader, text_column)?;
        let num_groups = reader.metadata().num_row_groups();
        Ok(Self {
            path,
            text_idx,
            num_groups,
        })
    }

    /// Fetch shard `file` from `repo` at git `revision` via the HF cache, then open it on
    /// `text_column`. `revision` selects the branch/ref the shard lives on — `"main"` for
    /// natively-laid-out repos (FineWeb), `"refs/convert/parquet"` for the auto-converted
    /// parquet shards (FineMath).
    pub fn from_hub(repo: &str, file: &str, text_column: &str, revision: &str) -> Result<Self> {
        use hf_hub::api::sync::Api;
        use hf_hub::{Repo, RepoType};
        let path = Api::new()?
            .repo(Repo::with_revision(
                repo.to_string(),
                RepoType::Dataset,
                revision.to_string(),
            ))
            .get(file)?;
        Self::open(path, text_column)
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

/// Blobs fetched per `group_texts` call. Also the granularity of the row-group cache,
/// so a miss re-fetches at most this many files.
const SWH_GROUP_SIZE: usize = 256;

/// A code corpus whose parquet ships **Software-Heritage IDs**, not text (e.g. Stack-Edu):
/// each row is a `blob_id` whose content lives at `content/{blob_id}` (gzipped) in
/// Software Heritage's public S3 bucket. Construction reads up to `max_files` blob ids
/// from the shard (cheap, local); `group_texts` fetches a batch of them over HTTPS and
/// gunzips. A per-blob fetch failure drops that file (empty string) rather than aborting.
pub struct SoftwareHeritageSource {
    blob_ids: Vec<String>,
}

impl SoftwareHeritageSource {
    /// Fetch the Stack-Edu-style shard `file` from `repo`@`revision`, then read up to
    /// `max_files` ids from its `blob_column`.
    pub fn from_hub(
        repo: &str,
        revision: &str,
        file: &str,
        blob_column: &str,
        max_files: usize,
    ) -> Result<Self> {
        use hf_hub::api::sync::Api;
        use hf_hub::{Repo, RepoType};
        let path = Api::new()?
            .repo(Repo::with_revision(
                repo.to_string(),
                RepoType::Dataset,
                revision.to_string(),
            ))
            .get(file)?;
        let reader = SerializedFileReader::new(File::open(&path).wrap_err("opening code shard")?)
            .wrap_err("opening parquet reader")?;
        let idx = column_index(&reader, blob_column)?;

        let mut blob_ids = Vec::with_capacity(max_files.min(1 << 16));
        'outer: for g in 0..reader.num_row_groups() {
            let group = reader.get_row_group(g).wrap_err("reading row group")?;
            for record in group.get_row_iter(None).wrap_err("row iterator")? {
                let row = record.wrap_err("decoding parquet row")?;
                if let Ok(id) = row.get_string(idx) {
                    blob_ids.push(id.clone());
                    if blob_ids.len() >= max_files {
                        break 'outer;
                    }
                }
            }
        }
        Ok(Self { blob_ids })
    }
}

/// Max blob fetches in flight at once against Software Heritage's S3.
const SWH_CONCURRENCY: usize = 32;

/// Process-wide async HTTP client for the SWH fetches, built once.
fn swh_client() -> Result<&'static reqwest::Client> {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    if let Some(client) = CLIENT.get() {
        return Ok(client);
    }
    let client = reqwest::Client::builder()
        .build()
        .wrap_err("building swh http client")?;
    Ok(CLIENT.get_or_init(|| client))
}

/// Process-wide runtime that drives the async fetches. The streaming dataset's
/// `group_texts` is a synchronous call, so it `block_on`s this runtime; the
/// threads are never themselves inside a runtime, so blocking on it is safe.
fn swh_runtime() -> Result<&'static tokio::runtime::Runtime> {
    static RT: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    if let Some(rt) = RT.get() {
        return Ok(rt);
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .wrap_err("building swh runtime")?;
    Ok(RT.get_or_init(|| rt))
}

/// Fetch one blob's content from Software Heritage's public S3 bucket and gunzip it.
/// Objects are stored gzipped and served as `application/octet-stream` with no
/// `Content-Encoding`, so the body is the raw gzip stream.
///
/// TODO: Bytes are decoded lossily because Stack-Edu spans many source encodings
///       likely should convert to utf-8.
async fn fetch_one(client: &reqwest::Client, blob_id: &str) -> Result<String> {
    let url = format!("https://softwareheritage.s3.amazonaws.com/content/{blob_id}");
    let gz = client
        .get(&url)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|e| eyre::eyre!("swh fetch {blob_id}: {e}"))?
        .bytes()
        .await
        .wrap_err("reading swh response")?;
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(&gz[..])
        .read_to_end(&mut bytes)
        .wrap_err("gunzip swh blob")?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Fetch `blob_ids` concurrently (≤ `SWH_CONCURRENCY` in flight), preserving input order.
fn fetch_swh_blobs(blob_ids: &[String]) -> Result<Vec<String>> {
    let client = swh_client()?;
    let out = swh_runtime()?.block_on(async move {
        let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(SWH_CONCURRENCY));
        let mut set = tokio::task::JoinSet::new();
        for (i, id) in blob_ids.iter().cloned().enumerate() {
            let client = client.clone();
            let sem = sem.clone();
            set.spawn(async move {
                // The semaphore is never closed, so acquire can't really fail; if it
                // somehow did, drop this blob rather than the whole group.
                let Ok(_permit) = sem.acquire_owned().await else {
                    return (i, String::new());
                };
                (i, fetch_one(&client, &id).await.unwrap_or_default())
            });
        }
        let mut out = vec![String::new(); blob_ids.len()];
        while let Some(joined) = set.join_next().await {
            if let Ok((i, text)) = joined {
                out[i] = text;
            }
        }
        out
    });
    Ok(out)
}

impl RowGroupSource for SoftwareHeritageSource {
    fn num_groups(&self) -> usize {
        self.blob_ids.len().div_ceil(SWH_GROUP_SIZE).max(1)
    }

    fn group_texts(&self, g: usize) -> Result<Vec<String>> {
        let start = g * SWH_GROUP_SIZE;
        let end = (start + SWH_GROUP_SIZE).min(self.blob_ids.len());
        if start >= end {
            return Ok(Vec::new());
        }
        // One HTTP GET per blob, fetched concurrently; a failed fetch becomes an empty
        // document instead of failing the run.
        fetch_swh_blobs(&self.blob_ids[start..end])
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

/// Build parquet-shard sources for `files` from `repo`@`revision`, reading `text_column`.
pub fn parquet_sources(
    repo: &str,
    revision: &str,
    files: &[String],
    text_column: &str,
) -> Result<Vec<Box<dyn RowGroupSource>>> {
    files
        .iter()
        .map(|f| {
            Ok(
                Box::new(ParquetShard::from_hub(repo, f, text_column, revision)?)
                    as Box<dyn RowGroupSource>,
            )
        })
        .collect()
}

/// Build Software-Heritage code sources for `files` from `repo`@`revision`.
pub fn swh_sources(
    repo: &str,
    revision: &str,
    files: &[String],
    blob_column: &str,
    max_files: usize,
) -> Result<Vec<Box<dyn RowGroupSource>>> {
    files
        .iter()
        .map(|f| {
            Ok(Box::new(SoftwareHeritageSource::from_hub(
                repo,
                revision,
                f,
                blob_column,
                max_files,
            )?) as Box<dyn RowGroupSource>)
        })
        .collect()
}

/// Pull up to `max_docs` documents across `sources`, in order — the tokenizer-training
/// sampler. (Pretraining instead wraps the same sources in a `StreamingTokenDataset`.)
pub fn collect_docs(sources: &[Box<dyn RowGroupSource>], max_docs: usize) -> Result<Vec<String>> {
    let mut out = Vec::new();
    'outer: for src in sources {
        for g in 0..src.num_groups() {
            if out.len() >= max_docs {
                break 'outer;
            }
            for doc in src.group_texts(g)? {
                out.push(doc);
                if out.len() >= max_docs {
                    break 'outer;
                }
            }
        }
    }
    Ok(out)
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

    /// Fetch + open the listed parquet shards from `repo`@`revision` (reading `text_column`)
    /// and build the dataset.
    pub fn from_hub(
        tokenizer: Arc<fastokens::Tokenizer>,
        repo: &str,
        revision: &str,
        files: &[String],
        text_column: &str,
        seq_len: usize,
        cache_groups: u64,
    ) -> Result<Self> {
        Self::new(
            parquet_sources(repo, revision, files, text_column)?,
            tokenizer,
            seq_len,
            cache_groups,
        )
    }

    /// Total dense windows across all sources (before any `view`).
    pub fn total(&self) -> usize {
        self.index
            .last()
            .map(|g| g.win_start + g.n_windows)
            .unwrap_or(0)
    }

    /// A sub-range view exposing global windows `range` as local indices, sharing the
    /// same sources and cache.
    pub fn view(&self, range: std::ops::Range<usize>) -> Self {
        let mut v = self.clone();
        v.offset = range.start;
        v.len = range.end.saturating_sub(range.start);
        v
    }

    /// Tokenize and concatenate one row group into its dense token stream.
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
        // read error logs and drops the item rather than killing the run.
        let stream = self
            .cache
            .try_get_with(group_idx as u64, || self.load_group(group_idx))
            .map_err(|e| eprintln!("streaming dataset: row group {group_idx} failed: {e}"))
            .ok()?;
        // Dense window `local`: tokens [local*w .. local*w + w). The window index is sized
        // from the count pass; a non-deterministic source can reload a *shorter* stream than
        // it counted, so a promised window may not fit. Drop it rather than panic.
        let w = self.seq_len + 1;
        let start = local * w;
        stream
            .get(start..start + w)
            .map(|win| win.iter().map(|&t| t as i64).collect())
    }

    fn len(&self) -> usize {
        self.len
    }
}

// --- Multi-stage data mix (decay anneal) ---
//
// Vary the domain mix over training instead of one static blend: web-heavy early, then
// upweight math/code over the cosine-LR decay tail, following SmolLM3's multi-stage recipe.
// The mechanism is a `MixtureDataset` wrapping one `StreamingTokenDataset` per domain plus a
// shared atomic counter of items drawn; each `get` reads the current progress fraction,
// interpolates the domain weights for it, picks a domain, and draws a window from it.
// This composes with `SamplerDataset` + Burn's `SupervisedTraining` (which owns the epoch
// loop) without a custom training loop. Progress is approximate — worker prefetch runs a
// little ahead, and a resume restarts the counter — but the ramp is coarse, so neither
// matters. The held-out valid set stays a single fixed domain (see `pretrain`) so the
// metric is comparable across the run.

/// Pick an index in `0..weights.len()` by the unnormalized `weights`, using a uniform
/// `u ∈ [0,1)`. Empty/zero weights resolve to domain 0.
fn pick_domain(weights: &[f64], u: f64) -> usize {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return 0;
    }
    let target = u * total;
    let mut acc = 0.0;
    for (i, w) in weights.iter().enumerate() {
        acc += w;
        if target < acc {
            return i;
        }
    }
    weights.len() - 1
}

/// Domain mix weights as a function of training progress: a constant `stable` blend until
/// `decay_start`, then a linear ramp to the `decay` blend by the end. Both weight vectors
/// are indexed by domain in the same order as `MixtureDataset::new`'s `domains`.
#[derive(Clone, Debug)]
pub struct MixSchedule {
    stable: Vec<f64>,
    decay: Vec<f64>,
    decay_start: f64,
}

impl MixSchedule {
    /// `decay_start` is the progress fraction (`0.0..1.0`) where the ramp begins; before
    /// it the mix is `stable`, at progress 1.0 the mix is `decay`.
    pub fn new(stable: Vec<f64>, decay: Vec<f64>, decay_start: f64) -> Self {
        debug_assert_eq!(
            stable.len(),
            decay.len(),
            "stable/decay domain counts differ"
        );
        Self {
            stable,
            decay,
            decay_start: decay_start.clamp(0.0, 1.0),
        }
    }

    /// Per-domain weights at `progress ∈ [0,1]`, linearly interpolated over the decay tail.
    pub fn weights_at(&self, progress: f64) -> Vec<f64> {
        let t = if progress <= self.decay_start {
            0.0
        } else {
            ((progress - self.decay_start) / (1.0 - self.decay_start).max(f64::EPSILON))
                .clamp(0.0, 1.0)
        };
        self.stable
            .iter()
            .zip(&self.decay)
            .map(|(s, d)| s + (d - s) * t)
            .collect()
    }
}

/// A progress-weighted mixture over per-domain streaming corpora.
#[derive(Clone)]
pub struct MixtureDataset {
    domains: Vec<StreamingTokenDataset>,
    schedule: MixSchedule,
    consumed: Arc<AtomicU64>,
    total: u64,
    len: usize,
}

impl MixtureDataset {
    pub fn new(
        domains: Vec<StreamingTokenDataset>,
        schedule: MixSchedule,
        total_draws: u64,
    ) -> Self {
        let len = domains.iter().map(|d| d.len()).sum();
        Self {
            domains,
            schedule,
            consumed: Arc::new(AtomicU64::new(0)),
            total: total_draws.max(1),
            len,
        }
    }
}

impl Dataset<Vec<i64>> for MixtureDataset {
    fn get(&self, index: usize) -> Option<Vec<i64>> {
        let step = self.consumed.fetch_add(1, Ordering::Relaxed);
        let progress = (step as f64 / self.total as f64).min(1.0);
        let weights = self.schedule.weights_at(progress);

        // Seed a PRNG from the (already-shuffled) sampler index so a draw stays a
        // deterministic function of `index`: one uniform picks the domain, the next a
        // window within it.
        let mut rng = fastrand::Rng::with_seed(index as u64);
        let d = pick_domain(&weights, rng.f64());
        let dom = &self.domains[d];
        let dlen = dom.len();
        if dlen == 0 {
            return None;
        }
        dom.get(rng.usize(0..dlen))
    }

    fn len(&self) -> usize {
        self.len.max(1)
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

/// Load OpenAssistant oasst_top1 conversations, up to `max` threads. Each row's `text` is
/// an already-ChatML thread (`<|im_start|>{role}\n…<|im_end|>`, the dataset's own format,
/// parsed as-is); rendering into our template happens in [`build_sft_rows`].
pub fn load_sft_threads(
    repo: &str,
    file: &str,
    max: usize,
) -> Result<Vec<Vec<crate::chat::Message>>> {
    let reader = open_hub_dataset_parquet(repo, file)?;
    let text = column_index(&reader, "text")?;
    read_rows(&reader, max, |row| {
        // A row with no readable text column is skipped, not fatal.
        Ok(row
            .get_string(text)
            .ok()
            .map(|t| oasst_thread(t))
            .filter(|turns| !turns.is_empty()))
    })
}

/// Parse an oasst_top1 `text` thread into ordered turns. The dataset delimits turns with
/// `<|im_start|>{role}\n…<|im_end|>`; we map `user`/`assistant`/`system` to roles and drop
/// empty or unknown-role turns.
fn oasst_thread(text: &str) -> Vec<crate::chat::Message> {
    use crate::chat::Message;
    let mut turns = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find("<|im_start|>") {
        let after = &rest[open + "<|im_start|>".len()..];
        let Some(nl) = after.find('\n') else { break };
        let role = after[..nl].trim();
        let body = &after[nl + 1..];
        let end = body.find("<|im_end|>").unwrap_or(body.len());
        let content = body[..end].trim().to_string();
        if !content.is_empty() {
            match role {
                "user" => turns.push(Message::user(content)),
                "assistant" => turns.push(Message::assistant(content)),
                "system" => turns.push(Message::system(content)),
                _ => {}
            }
        }
        rest = &body[end..];
    }
    turns
}

/// Build a padded `(inputs, targets)` row of length `seq_len` that trains only on
/// `response`: the conversation `history` plus the open model turn is the `IGNORE_ID`-masked
/// prefix, and the response with its closing `<turn|>` are the supervised targets.
/// `render(history, true)` is a token prefix of the full render, which the masking relies on.
fn build_masked_row(
    tok: &fastokens::Tokenizer,
    history: &[crate::chat::Message],
    response: &str,
    seq_len: usize,
) -> Result<(Vec<i64>, Vec<i64>)> {
    let ignore = crate::chat::IGNORE_ID as i64;
    let mut convo = history.to_vec();
    convo.push(crate::chat::Message::assistant(response));
    let full: Vec<i64> = tok
        .encode(&crate::chat::render(&convo, false))
        .wrap_err("encoding sft example")?
        .into_iter()
        .map(|x| x as i64)
        .collect();
    let prompt_len = tok
        .encode(&crate::chat::render(history, true))
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

/// Build one padded row from a single-turn `(prompt, response)`. Used by DPO; a thin wrapper
/// over [`build_masked_row`] with a one-message user history.
pub fn build_sft_row(
    tok: &fastokens::Tokenizer,
    prompt: &str,
    response: &str,
    seq_len: usize,
) -> Result<(Vec<i64>, Vec<i64>)> {
    build_masked_row(
        tok,
        &[crate::chat::Message::user(prompt)],
        response,
        seq_len,
    )
}

/// Build SFT rows from a full conversation: one cumulative-context row per model turn, so
/// every assistant response in the thread is trained on with all of its prior context.
pub fn build_sft_rows(
    tok: &fastokens::Tokenizer,
    conv: &[crate::chat::Message],
    seq_len: usize,
) -> Result<Vec<(Vec<i64>, Vec<i64>)>> {
    let mut rows = Vec::new();
    for (i, m) in conv.iter().enumerate() {
        if m.role == crate::chat::Role::Model && !m.content.is_empty() {
            rows.push(build_masked_row(tok, &conv[..i], &m.content, seq_len)?);
        }
    }
    Ok(rows)
}

/// Pre-tokenized SFT rows.
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
/// continuations.
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
/// log-probs are computed inside the train step, not carried here.
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

/// One GRPO example: a tokenized prompt and its verifiable answer.
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

    fn fixture_tokenizer() -> fastokens::Tokenizer {
        let json = serde_json::from_str(include_str!("../testdata/tokenizer.json"))
            .expect("fixture tokenizer json");
        fastokens::Tokenizer::from_json(json).expect("loading fixture tokenizer")
    }

    #[test]
    fn oasst_thread_parses_all_turns() {
        use crate::chat::Role;
        let text = "<|im_start|>user\nName 3 dogs<|im_end|>\n<|im_start|>assistant\nRex, Fido, Spot<|im_end|>\n<|im_start|>user\nmore<|im_end|>\n";
        let turns = oasst_thread(text);
        assert_eq!(
            turns.len(),
            3,
            "all three turns parsed, not just the first pair"
        );
        assert_eq!(
            (turns[0].role, turns[0].content.as_str()),
            (Role::User, "Name 3 dogs")
        );
        assert_eq!(
            (turns[1].role, turns[1].content.as_str()),
            (Role::Model, "Rex, Fido, Spot")
        );
        assert_eq!(
            (turns[2].role, turns[2].content.as_str()),
            (Role::User, "more")
        );
        assert!(oasst_thread("no markers here").is_empty());
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
        let tok = Arc::new(fixture_tokenizer());
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

    #[test]
    fn streaming_tolerates_a_group_that_shrinks_on_reload() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // A source whose group yields fewer tokens on reload than during the count pass —
        // exactly the SoftwareHeritage flaky-fetch case (a fetch that succeeds while counting
        // fails on reload → empty doc → fewer tokens). The window index is sized from the
        // count pass, so `get` must not slice past the shortened reloaded stream. Regression:
        // panic `range start index .. out of range for slice of length ..` at `get`.
        struct ShrinkingSource {
            calls: AtomicUsize,
            long: String,
            short: String,
        }
        impl RowGroupSource for ShrinkingSource {
            fn num_groups(&self) -> usize {
                1
            }
            fn group_texts(&self, _g: usize) -> Result<Vec<String>> {
                // First call (the count pass) sees the long doc; every reload sees the short.
                let n = self.calls.fetch_add(1, Ordering::Relaxed);
                Ok(vec![if n == 0 {
                    self.long.clone()
                } else {
                    self.short.clone()
                }])
            }
        }

        let tok = Arc::new(fixture_tokenizer());
        let seq_len = 7;
        let para = "the quick brown fox jumps over the lazy dog again and again ";
        let ds = StreamingTokenDataset::new(
            vec![Box::new(ShrinkingSource {
                calls: AtomicUsize::new(0),
                long: para.repeat(8),    // many windows during the count pass
                short: para.to_string(), // far fewer tokens on reload
            })],
            tok,
            seq_len,
            8,
        )
        .unwrap();

        // The index promised more windows than the reload can fill. Every `get` must return
        // a full window or `None` — never panic.
        for i in 0..ds.len() {
            if let Some(win) = ds.get(i) {
                assert_eq!(win.len(), seq_len + 1, "returned window must be full width");
            }
        }
    }

    #[test]
    fn mix_schedule_ramps_stable_to_decay() {
        let s = MixSchedule::new(vec![0.85, 0.15], vec![0.5, 0.5], 0.8);
        assert_eq!(s.weights_at(0.0), vec![0.85, 0.15]); // stable before the tail
        assert_eq!(s.weights_at(0.8), vec![0.85, 0.15]); // ramp begins at decay_start
        let mid = s.weights_at(0.9); // halfway through the 0.8..1.0 tail
        assert!(
            (mid[0] - 0.675).abs() < 1e-9 && (mid[1] - 0.325).abs() < 1e-9,
            "{mid:?}"
        );
        let end = s.weights_at(1.0);
        assert!(
            (end[0] - 0.5).abs() < 1e-9 && (end[1] - 0.5).abs() < 1e-9,
            "{end:?}"
        );
    }

    #[test]
    fn pick_domain_respects_weights() {
        assert_eq!(pick_domain(&[1.0, 0.0, 0.0], 0.99), 0);
        assert_eq!(pick_domain(&[0.0, 0.0, 1.0], 0.0), 2);
        assert_eq!(pick_domain(&[0.0, 0.0, 1.0], 0.99), 2);
        // 50/50: low u falls in the first half, high u in the second.
        assert_eq!(pick_domain(&[0.5, 0.5], 0.1), 0);
        assert_eq!(pick_domain(&[0.5, 0.5], 0.9), 1);
        // Degenerate all-zero weights resolve to domain 0.
        assert_eq!(pick_domain(&[0.0, 0.0], 0.5), 0);
    }

    #[test]
    fn mixture_routes_draws_by_domain_weight() {
        let tok = Arc::new(fixture_tokenizer());
        let seq_len = 7;
        let para = "the quick brown fox jumps over the lazy dog again and again ";
        let web = StreamingTokenDataset::new(
            vec![Box::new(FakeSource {
                groups: vec![vec![para.repeat(4)]],
            })],
            tok.clone(),
            seq_len,
            8,
        )
        .unwrap();
        // A domain whose only doc is too short to fill one window → 0 windows.
        let empty = StreamingTokenDataset::new(
            vec![Box::new(FakeSource {
                groups: vec![vec!["a".to_string()]],
            })],
            tok.clone(),
            seq_len,
            8,
        )
        .unwrap();
        assert!(
            web.len() > 0 && empty.len() == 0,
            "fixture domains malformed"
        );

        // All weight on web → every draw is a full web window.
        let all_web = MixtureDataset::new(
            vec![web.clone(), empty.clone()],
            MixSchedule::new(vec![1.0, 0.0], vec![1.0, 0.0], 0.8),
            100,
        );
        for i in 0..20 {
            assert_eq!(all_web.get(i).expect("web draw").len(), seq_len + 1);
        }

        // All weight on the empty domain → every draw routes there and yields None.
        let all_empty = MixtureDataset::new(
            vec![web, empty],
            MixSchedule::new(vec![0.0, 1.0], vec![0.0, 1.0], 0.8),
            100,
        );
        assert!(
            (0..20).all(|i| all_empty.get(i).is_none()),
            "empty-domain draws must be None"
        );
    }
}
