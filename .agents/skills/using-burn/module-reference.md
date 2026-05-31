# Burn 0.21 — Module, Loss, Metric, Recorder & Dataset Reference

Catalog of built-ins. Every layer follows `XxxConfig::new(..).init(device) -> Xxx`. `cargo check` /
[docs.rs/burn](https://docs.rs/burn/latest/burn/) are authoritative.

## Built-in nn modules (`burn::nn`)

### Activations / general

`BatchNorm`, `Celu`, `Dropout`, `Elu`, `Embedding`, `GaussianNoise`, `Gelu`, `Glu`, `GroupNorm`,
`HardShrink`, `HardSigmoid`, `HardSwish`, `InstanceNorm`, `LayerNorm`, `LocalResponseNorm`,
`LeakyRelu`, `Linear`, `Prelu`, `Relu`, `Selu`, `Sigmoid`, `Softplus`, `SoftShrink`, `Softsign`,
`Shrink`, `RmsNorm`, `SwiGlu`, `Tanh`, `ThresholdedRelu`.

Common configs:

```rust
LinearConfig::new(d_in, d_out).with_bias(false).init(device);
EmbeddingConfig::new(n_embeddings, d_model).init(device);      // forward([B,T] Int) -> [B,T,d_model]
LayerNormConfig::new(d_model).init(device);
RmsNormConfig::new(d_model).init(device);
SwiGluConfig::new(d_in, d_ff).init(device);                    // gated projection only — add a Linear(d_ff, d_model) for a full MLP
DropoutConfig::new(0.1).init();
BatchNormConfig::new(num_features).init(device);               // BatchNorm<B, const D>
```

### Convolutions

`Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d/2d/3d`, `DeformConv2d`.

```rust
Conv2dConfig::new([ch_in, ch_out], [kh, kw])
    .with_stride([1, 1]).with_padding(PaddingConfig2d::Same).with_bias(true).init(device);
```

### Pooling

`AdaptiveAvgPool1d/2d`, `AvgPool1d/2d`, `MaxPool1d/2d`.
`AdaptiveAvgPool2dConfig::new([h, w]).init();`

### Interpolation

`Interpolate1d`, `Interpolate2d`. Config: `output_size` (takes precedence) or `scale_factor`, `mode`
(`InterpolateMode::{Nearest, Linear, Cubic, Lanczos}`), `align_corners`.

### RNNs

`Gru` / `BiGru`, `Lstm` / `BiLstm`, `GateController`.

### Transformer

`MultiHeadAttention`, `TransformerEncoder`, `TransformerDecoder`, `PositionalEncoding`,
`RotaryEncoding`.

## Loss functions (`burn::nn::loss`)

`BinaryCrossEntropyLoss`, `CosineEmbeddingLoss`, `CrossEntropyLoss`, `CTCLoss`, `GramMatrixLoss`,
`HuberLoss`, `KLDivLoss`, `LpLoss`, `MseLoss`, `PoissonNllLoss`, `RNNTLoss`, `SmoothL1Loss`.

```rust
// Config style (recommended): pad-token masking drops positions whose TARGET == id (SFT/padding masks)
let loss = CrossEntropyLossConfig::new()
    .with_pad_tokens(Some(vec![pad_id]))
    .init(&device)
    .forward(logits /* [N, C] */, targets /* [N] Int */);

// Direct style (used in manual loops):
let loss = CrossEntropyLoss::new(None /* pad tokens */, &device).forward(logits, targets);
```

## Training output structs (`burn::train`)

Each implements `Adaptor<MetricInput>` for the metrics it supports, so it plugs straight into the
learner. Construct with `::new(..)`.

| Struct                              | Use case     | Fields                                                                                     | Adapted metrics                                                                        |
| ----------------------------------- | ------------ | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| `ClassificationOutput<B>`           | single-label | `loss: T<B,1>`, `output: T<B,2>`, `targets: T<B,1,Int>`                                    | Accuracy, TopKAccuracy, Perplexity, Precision\*, Recall\*, FBetaScore\*, AUROC\*, Loss |
| `MultiLabelClassificationOutput<B>` | multi-label  | `loss: T<B,1>`, `output: T<B,2>`, `targets: T<B,2,Int>`                                    | HammingScore, Precision\*, Recall\*, FBetaScore\*, AUROC\*, Loss                       |
| `RegressionOutput<B>`               | regression   | `loss: T<B,1>`, `output: T<B,2>`, `targets: T<B,2>`                                        | Loss                                                                                   |
| `SequenceOutput<B>`                 | sequences    | `loss: T<B,1>`, `logits: T<B,3>`, `predictions: Option<T<B,2,Int>>`, `targets: T<B,2,Int>` | Accuracy, TopKAccuracy, Perplexity, CER, WER, Loss                                     |

\* Precision/Recall/FBetaScore/AUROC share `ConfusionStatsInput`, adapted implicitly.

**Custom output / metric:** implement `Adaptor<MetricInput>` for your output struct:

```rust
impl<B: Backend> Adaptor<AccuracyInput<B>> for MyOutput<B> {
    fn adapt(&self) -> AccuracyInput<B> { AccuracyInput::new(self.output.clone(), self.targets.clone()) }
}
```

Implement the `Metric` trait (`type Input`, `name`, `update`, `clear`) for a custom metric; add
`Numeric` (`value`, `running_value`) to make it plottable.

## Built-in metrics (`burn::train::metric`)

Accuracy, TopKAccuracy, Precision, Recall, FBetaScore, AUROC, Loss, CharErrorRate (CER),
WordErrorRate (WER), HammingScore, Perplexity, IterationSpeed, LearningRate, CpuTemperature,
CpuUse, CpuMemory, CudaMetric. Vision: A-FINE, Dice, DISTS, FID, LPIPS, MS-SSIM, PSNR, SSIM.

Register: `.metrics((AccuracyMetric::new(), LossMetric::new()))`, or per split with
`.metric_train(..)` / `.metric_valid(..)`; use `_numeric` variants for plotted metrics.

## Optimizers (`burn::optim`)

`Adam`, `AdamW`, `Sgd`, `RmsProp` (+ `*Config`).

```rust
AdamConfig::new()
    .with_beta_2(0.95)
    .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
    .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
    .init();
```

## LR schedulers (`burn::lr_scheduler`)

`ConstantLr` (or pass a bare `f64`), `CosineAnnealingLrScheduler`, `LinearLrScheduler`,
`ExponentialLrScheduler`, `StepLrScheduler`, `NoamLrScheduler`.

```rust
CosineAnnealingLrSchedulerConfig::new(initial_lr, total_iters).with_min_lr(min_lr).init().unwrap();
```

## Recorders (`burn::record`)

| Recorder                                       | Format                                | Compression        |
| ---------------------------------------------- | ------------------------------------- | ------------------ |
| `DefaultFileRecorder` / `NamedMpkFileRecorder` | named MessagePack                     | none               |
| `NamedMpkGzFileRecorder`                       | named MessagePack                     | gzip               |
| `BinFileRecorder`                              | binary                                | none               |
| `BinGzFileRecorder`                            | binary                                | gzip               |
| `JsonGzFileRecorder`                           | JSON                                  | gzip               |
| `PrettyJsonFileRecorder`                       | pretty JSON                           | gzip               |
| `BinBytesRecorder`                             | in-memory binary                      | none               |
| `CompactRecorder`                              | named MessagePack, **half precision** | (training default) |

Precision settings (decoupled from training precision; auto-converted on load):
`FullPrecisionSettings` (f32/i32), `DoublePrecisionSettings` (f64/i64), `HalfPrecisionSettings` (f16/i16).

> Use the **same recorder kind** to save and load — formats are not interchangeable. Precision is.
> For storage, prefer compressed non-binary (binary may not be backward compatible); for debugging,
> pretty JSON; for `no_std`, in-memory binary via `include_bytes!`.

## burn-store (`burn_store`)

Newer weight store; intended to eventually replace recorders. Zero-copy mmap, cross-framework,
partial/filtered loading.

| Format      | Ext            | Notes                                               |
| ----------- | -------------- | --------------------------------------------------- |
| Burnpack    | `.bpk`         | native; fast, zero-copy, training-state persistence |
| SafeTensors | `.safetensors` | HF standard                                         |
| PyTorch     | `.pt` / `.pth` | read-only                                           |

```rust
let mut store = BurnpackStore::from_file("model.bpk");
model.save_into(&mut store)?;
let result = model.load_from(&mut store)?;       // result.applied / .missing / .errors; result.is_success()
```

Builder methods: `with_regex(pat)` / `with_full_path(p)` / `with_predicate(fn)` (filter),
`with_key_remapping(from, to)` / `remap(KeyRemapper)` (rename), `with_from_adapter` /
`with_to_adapter` (`PyTorchToBurnAdapter`, `BurnToPyTorchAdapter`, `HalfPrecisionAdapter`),
`allow_partial(true)`, `with_top_level_key(k)` (nested PyTorch dict), `map_indices_contiguous(bool)`
(nn.Sequential gaps), `metadata(k, v)`, `zero_copy(true)` / `from_static(&[u8])`.

Direct access: `store.keys()`, `store.get_snapshot(name)`. Model surgery: `model.collect(filter, ..)` /
`model2.apply(snapshots, filter, ..)`.

## Dataset transforms (`burn::data::dataset::transform`) — all lazy

| Transform                                | Purpose                                                                                                      |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `SamplerDataset::new(ds, size)`          | sample (default with replacement); fixes epoch length                                                        |
| `SelectionDataset`                       | select a subset by index; `from_indices_checked`, `new_shuffled(ds, seed\|&mut rng)`, mutable `.shuffle(..)` |
| `ShuffledDataset`                        | thin wrapper over `SelectionDataset` (seed or rng)                                                           |
| `PartialDataset::new(ds, start, end)`    | view of a range — build train/val/test splits                                                                |
| `MapperDataset` + `impl Mapper<In, Out>` | lazy per-item transform                                                                                      |
| `ComposedDataset`                        | concatenate multiple datasets                                                                                |
| `WindowsDataset`                         | overlapping windows (time series / LSTM)                                                                     |

## Dataset storage & sources

Storage: `InMemDataset` (`::from_csv(path, ReaderBuilder)`), `SqliteDataset`, `DataframeDataset` (Polars).

Sources:

- `HuggingfaceDatasetLoader::new("name").dataset("train")` → `SqliteDataset<Item>` (Item must derive
  `serde::{Serialize, Deserialize}`, `Clone`, `Debug`; **requires a Python install**).
- `ImageFolderDataset::new_classification(root)` / `new_multilabel_classification_with_items(items, classes)` /
  `new_segmentation_with_items(items, classes)` / `new_coco_detection(json, img_dir)`.
- `MnistDataset::train()` / `::test()` (`vision` feature).

## DataLoader

```rust
DataLoaderBuilder::new(batcher)
    .batch_size(64)
    .shuffle(seed)
    .num_workers(4)        // batcher must be Send + Sync + Clone
    .build(dataset);       // wrap in SamplerDataset to control epoch size
```
