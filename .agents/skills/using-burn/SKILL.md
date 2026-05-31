---
name: using-burn
description: Use when editing or adding Burn 0.21 (Rust ML framework on cubecl/wgpu/cuda/ndarray/flex backends) code — models, training loops, tensor ops, datasets, save/load, or backend setup. Reach for it on errors like trait-method-not-in-scope (Module/Config/Dataset), recursion_limit on wgpu, bf16-on-CPU, AutodiffBackend / inner-backend mismatches, and when unsure about Param vs constant, Learner vs manual loop, or which recorder/store to use.
---

# Using Burn 0.21

Burn is a Rust deep-learning framework: tensors and modules are **generic over a `Backend`**, autodiff is a backend decorator (`Autodiff<B>`), and most backends run on [CubeCL](https://github.com/tracel-ai/cubecl) (wgpu/cuda/metal). It has strong defaults and a builder/`Config` style throughout.

**`cargo check` is the source of truth.** The API is large and moves; when this doc and the compiler disagree, the compiler wins. For exhaustive op signatures see [docs.rs/burn](https://docs.rs/burn/latest/burn/). Heavy reference lives in companion files:

- **[tensor-reference.md](tensor-reference.md)** — every tensor op (basic/numeric/float/int/bool), activations, grid, linalg, signal, quant values.
- **[module-reference.md](module-reference.md)** — built-in nn layers, losses, metrics, recorders, dataset transforms/storage/sources.

## When to use

- Defining models (`#[derive(Module)]`), configs, custom layers, or raw `Param` tensors.
- Writing tensor math and hitting ownership/clone, shape, or sync issues.
- Setting up training (high-level `Learner` or a hand-written loop), optimizers, LR schedules, metrics, checkpoints.
- Loading/saving weights (recorders or `burn-store`; PyTorch/SafeTensors interop; ONNX import).
- Choosing/configuring a backend, or chasing `recursion_limit` / `AutodiffBackend` / inner-backend errors.

## Setup

```toml
[dependencies]
burn = { version = "~0.21", features = ["train", "wgpu"] }     # add "vision" for MnistDataset/ImageFolderDataset
# Fine-grained (disable defaults → re-enable std):
# burn = { version = "~0.21", default-features = false, features = ["std", "train", "tui", "wgpu", "fusion", "vision"] }
```

Key features: `train` (Learner + metrics + autodiff utils), `tui` (training dashboard), `vision` (vision datasets), `fusion` (kernel fusion for cubecl), `wgpu`/`cuda`/`metal`/`ndarray`/`flex` (backends).

**wgpu/cubecl backends require this at the crate root** (`main.rs`/`lib.rs`), or you get recursive-type-evaluation errors (default limit 128 < required ~130–150):

```rust
#![recursion_limit = "256"]
```

Import most things at once:

```rust
use burn::prelude::*;
// = config::Config, module::Module, nn,
//   tensor::{backend::Backend, Bool, Device, ElementConversion, Float, Int, Shape, Tensor, TensorData}
```

> Trait methods need the trait **in scope** to call them: `Module` (for `save_file`/`load_file`/`load_record`/`valid`), `Config` (for `save`/`load`), `Dataset` (for `get`/`len`), `ElementConversion` (for `.elem()`). "method not found" usually means a missing `use`. The prelude covers the common ones.

## Backends

`Backend` abstracts tensor ops, device, and element types. `AutodiffBackend` adds the backward pass and a `B::Gradients` type. Add autodiff to any backend with the decorator `Autodiff<B>` — it keeps a dynamic graph and delegates ops to the inner backend.

| Backend                   | Type                                  | Elements                       | Notes                                                        |
| ------------------------- | ------------------------------------- | ------------------------------ | ------------------------------------------------------------ |
| wgpu (cross-platform GPU) | `Wgpu<F, I, G>` e.g. `Wgpu<f32, i32>` | f32/f16/bf16, int              | `Metal`/`WebGpu` are wgpu graphics variants; `G` defaults    |
| CUDA                      | `Cuda<F, I>`                          | f32/f16/bf16, int              | NVIDIA only                                                  |
| ndarray (CPU)             | `NdArray<E>`                          | **f32/f64 only — no f16/bf16** | pure-CPU fallback                                            |
| Flex (CPU, newer default) | `Flex`                                | —                              | `no_std`/WASM-capable; becoming the default CPU/test backend |

```rust
type MyBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;     // train with this; eval/infer with MyBackend

let device = WgpuDevice::default();               // or Default::default(); Device::<B>::Cuda(0)
B::seed(&device, 42);                             // deterministic
```

**Inner backend:** with `B: AutodiffBackend`, `tensor.inner()` / `module.valid()` drop autodiff → `B::InnerBackend`. Validation/inference run there. You **cannot** call `backward()` on a non-autodiff backend (compile error — the type system prevents forgetting `no_grad`).

## Tensors

`Tensor<B, D, K>` — `D` is the **rank** (compile-time const), not the shape; shape is inferred at runtime. Kinds `K`: `Float` (default), `Int`, `Bool`.

```rust
Tensor<B, 3>          // float, rank 3
Tensor<B, 2, Int>
Tensor<B, 1, Bool>
```

**Creation** (full catalog in tensor-reference.md):

```rust
let a = Tensor::<B, 2>::from_data([[2., 3.], [4., 5.]], &device);
let b = Tensor::<B, 1>::from_floats([1., 2., 3.], &device);        // recommended for f32
let z = Tensor::<B, 2>::zeros([4, 8], &device);                    // ones / full([..], v, &dev)
let r = Tensor::<B, 2>::random([4, 8], Distribution::Default, &device);
let d = Tensor::<B, 2>::from_data(TensorData::new(vec, [2, 3]), &device);
let i = Tensor::<B, 1, Int>::arange(0..10, &device);
```

`TensorData` is the backend-agnostic storage struct (`shape`, `dtype`, bytes). Convert element precision to the active backend with `.convert::<B::FloatElem>()`; convert a scalar with `.elem::<B::IntElem>()` (needs `ElementConversion`).

**Ownership & cloning.** Almost every op **takes `self` by value**. Reuse a tensor → `.clone()` it. Cloning is cheap: it bumps a refcount on the buffer, doesn't copy data, and lets Burn fuse ops / do in-place when a tensor is used once. There are no explicit in-place ops.

```rust
let min = input.clone().min();
let max = input.clone().max();
let norm = (input - min.clone()).div(max - min);   // last use of input/min: no clone needed
```

**Extraction triggers a device sync** (`into_scalar`, `into_data`, `to_data`, `sync`):

```rust
let v: f32 = t.clone().into_scalar().elem::<f32>();
let vec: Vec<f32> = t.into_data().to_vec::<f32>().unwrap();   // into_data: one-shot; to_data: keep tensor
let has_nan = t.contains_nan().into_scalar();
```

Broadcasting applies to all ops unless noted. Print with `println!("{t}")` or `{:.2}` for precision; `set_print_options(PrintOptions { precision: Some(2), .. })` globally; `check_closeness(&a, &b)` for debugging numerical drift.

## Modules

`#[derive(Module, Debug)]` makes a struct a trainable, (de)serializable parameter container. **Every field must itself be a `Module`**: built-in layers, nested modules, `Vec<M>`, `Param<Tensor<B,D>>`, or plain constants (`usize`/`bool` — fine as-is).

```rust
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    linear1: Linear<B>,
    dropout: Dropout,
    activation: Relu,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(x);
        let x = self.dropout.forward(x);
        self.linear1.forward(x.flatten(1, 3))
    }
}
```

`forward` is not special — name it anything, define several. Built-in layers list: **module-reference.md**.

**Tensors inside a module — pick the right wrapper:**

| Declaration                                        | Meaning                                                                                                                     |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `Param<Tensor<B, D>>`                              | Trainable parameter. Gets a `ParamId`, **saved in the record**, updated by the optimizer.                                   |
| `Param<Tensor<B, D>>` + `.set_require_grad(false)` | Saved with weights but **not** optimized.                                                                                   |
| `Tensor<B, D>` (bare)                              | Constant; recreated at `init`, **not** in the record (records hold only params). Use for sinusoidal embeddings, masks, etc. |

Raw param init: `Initializer::Normal { mean: 0.0, std: 0.02 }.init([shape], device)` (also `Ones`/`Zeros`/`Constant { value }`); read with `.val()`.

**Module methods** (need `Module`/`AutodiffModule` in scope): `to_device` / `fork` / `no_grad` / `num_params` / `visit` / `map` / `into_record` / `load_record` / `save_file` / `load_file`; and `valid()` (≈ `.eval()`, autodiff → inner backend). Implement `ModuleMapper`/`ModuleVisitor` to fold over params (optimizers are mappers); `#[module(custom_display)]` + `impl ModuleDisplay` to customize printing.

## Config

`#[derive(Config, Debug)]` generates `new(...)` (non-default fields, in declaration order), `with_*` builders, and `save("x.json")`/`load("x.json")`. Requires `Debug`. Use `#[config(default = ...)]` for defaults.

```rust
#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = 0.5)]
    dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: Relu::new(),
        }
    }
}
```

The `config → init(device) → module` convention is universal: built-in layers are `XxxConfig::new(..).init(device)`. **Parameter init is lazy** — no allocation/kernels until first use — so `config.init(&dev).load_record(record)` carries no real cost.

## Autodiff

Gradients are **returned by `backward()`**, not stored on tensors (unlike PyTorch). Map them to module params with `GradientsParams` before stepping an optimizer.

```rust
let grads = loss.backward();                              // B::Gradients
let grad_of_t = t.grad(&grads);                           // peek; grad_remove pops (enables in-place)
let gp = GradientsParams::from_grads(grads, &model);      // link grads → param ids
model = optim.step(lr, model, gp);
```

`detach` / `require_grad` / `is_require_grad` / `set_require_grad` exist on tensors. No `no_grad` block needed — use the non-autodiff backend (`model.valid()` / `tensor.inner()`).

## Training — high-level (`Learner` + `SupervisedTraining`)

Your step returns an **output struct** the learner understands. Built-ins (fields + metrics they support) are in module-reference.md; the common one:

```rust
ClassificationOutput::new(loss /* T<B,1> */, logits /* T<B,2> */, targets /* T<B,1,Int> */)
```

`TrainStep`/`InferenceStep` use **associated types** (`type Input; type Output;`), not generics:

```rust
impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = MnistBatch<B>;
    type Output = ClassificationOutput<B>;
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for Model<B> {           // note: Backend, not AutodiffBackend
    type Input = MnistBatch<B>;
    type Output = ClassificationOutput<B>;
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}
```

Build and launch (TUI dashboard via the `tui` feature):

```rust
let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
    .metrics((AccuracyMetric::new(), LossMetric::new()))   // metric must impl Adaptor for the output
    .with_file_checkpointer(CompactRecorder::new())
    .num_epochs(config.num_epochs)
    .summary();

let model = config.model.init::<B>(&device);               // B: AutodiffBackend
let result = training.launch(Learner::new(model, config.optimizer.init(), config.learning_rate));
result.model                                               // ⚠ on the INNER (non-autodiff) backend — do NOT call .valid()
    .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
    .unwrap();
```

- The 3rd `Learner::new` arg is an **LR scheduler**; a bare `f64` becomes a constant schedule. LR is applied at step time, not stored in the optimizer config.
- Metrics need their output to implement `Adaptor<MetricInput>`; the built-in output structs already do for their listed metrics (see module-reference.md). Use `.metric_train_numeric(...)`/`.metric_valid_numeric(...)` for plotted (numeric) metrics.
- Artifacts land under `artifact_dir/`: `experiment.log`, `checkpoint/{model,optim,scheduler}-N.mpk(.gz)`, `train|valid/epoch-N/<Metric>.log`. The file checkpointer can auto-prune old checkpoints. Resume via the Checkpoint option.

## Training — manual loop

When `Learner` doesn't fit, write the loop. Setup (config, batcher, dataloaders) is identical; only the loop differs:

```rust
pub fn run<B: AutodiffBackend>(device: B::Device) {
    let mut model = config.model.init::<B>(&device);
    let mut optim = config.optimizer.init();

    for epoch in 1..=config.num_epochs {
        for batch in dataloader_train.iter() {
            let output = model.forward(batch.images);
            let loss = CrossEntropyLoss::new(None, &output.device())
                .forward(output.clone(), batch.targets.clone());

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.lr, model, grads);   // no zero_grad; grads consumed by step
        }

        let model_valid = model.valid();                   // inner backend, no autodiff
        for batch in dataloader_valid.iter() {
            let _ = model_valid.forward(batch.images);      // metrics on B::InnerBackend
        }
    }
}
```

- **Gradient accumulation:** `GradientsAccumulator::new()` → `accumulate(&model, gp)` repeatedly → `grads()` to pop.
- **Per-part / multiple optimizers:** `GradientsParams::from_module(&mut grads, &model.conv1)` (and `from_params(.., &[id])`) extract subsets; step each with its own LR/optimizer. They _remove_ gradients, so a final `from_grads(grads, &model)` catches the rest.
- Wrapping model+optimizer in a struct needs care with generics; if the backend `B` is otherwise unconstrained, carry a `PhantomData<B>`.

## Optimizers & schedulers

```rust
let optim = AdamConfig::new()
    .with_beta_2(0.95)
    .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
    .init();

let lr = CosineAnnealingLrSchedulerConfig::new(initial_lr, total_iters)
    .with_min_lr(min_lr).init().unwrap();          // or just pass an f64 for constant LR
```

Optimizers: `Adam`/`AdamW`/`Sgd`/`RmsProp` (+ configs). Schedulers live in `burn::lr_scheduler`.

## Data

Two traits. `Dataset<I>: Send + Sync` is fixed-length random access (no streaming API by design — `len()` is just iterations-per-checkpoint; you may return different items for the same index):

```rust
pub trait Dataset<I> { fn get(&self, index: usize) -> Option<I>; fn len(&self) -> usize; }
```

`Batcher<B, Item, Batch>` collates items into a batched tensor. It's **generic over the backend** and must be `Clone` (+ `Send + Sync` for workers) — keep it trivial (tokenize/decode up front, not in `batch`):

```rust
#[derive(Clone, Default)]
pub struct MnistBatcher {}

impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images = items.iter()
            .map(|it| TensorData::from(it.image).convert::<B::FloatElem>())
            .map(|d| Tensor::<B, 2>::from_data(d, device).reshape([1, 28, 28]))
            .collect();
        let targets = items.iter()
            .map(|it| Tensor::<B, 1, Int>::from_data([(it.label as i64).elem::<B::IntElem>()], device))
            .collect();
        MnistBatch { images: Tensor::cat(images, 0), targets: Tensor::cat(targets, 0) }
    }
}

let loader = DataLoaderBuilder::new(batcher)
    .batch_size(64).shuffle(seed).num_workers(4)
    .build(MnistDataset::train());                  // or SamplerDataset::new(ds, epoch_size)
```

Transforms (all lazy), storage, and sources catalog: **module-reference.md**. Common: `SamplerDataset` (fixed-size sampling = controls epoch length), `ShuffledDataset`/`SelectionDataset` (+ seed), `PartialDataset` (splits), `MapperDataset`, `ComposedDataset`, `WindowsDataset` (time series). Storage: `InMemDataset` (+ `from_csv`), `SqliteDataset`, `DataframeDataset`. Sources: `HuggingfaceDatasetLoader` (needs Python), `ImageFolderDataset`, `MnistDataset` (`vision` feature).

## Save / load

**Recorder API** (records hold params + optimizer/scheduler state, backend-independent, precision auto-converts on load — but **formats are not interchangeable**: load with the same recorder kind you saved with).

```rust
let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();   // CompactRecorder = mpk + half precision
model.save_file(path, &recorder).unwrap();                              // extension handled automatically

// Two load styles:
let model = Model::new(&cfg, &dev).load_file(path, &recorder, &dev);    // ⚠ 3 args incl. device
// or, since init is lazy:
let record = recorder.load(path.into(), &dev).unwrap();
let model = config.model.init::<B>(&dev).load_record(record);
```

Recorders & precision settings: **module-reference.md**. `CompactRecorder` (MessagePack, half precision) is the training default.

**`burn-store`** (newer, will eventually replace recorders) — zero-copy mmap, cross-framework, partial/filtered loading:

```rust
use burn_store::{ModuleSnapshot, BurnpackStore, SafetensorsStore, PytorchStore};

let mut store = BurnpackStore::from_file("model.bpk");      // .bpk native: fast, zero-copy, training state
model.save_into(&mut store)?;
let result = model.load_from(&mut store)?;                  // result.applied / .missing / .errors

// PyTorch / SafeTensors interop (use the adapter for PyTorch-exported tensors):
let mut store = SafetensorsStore::from_file("m.safetensors").with_from_adapter(PyTorchToBurnAdapter);
let mut store = PytorchStore::from_file("m.pt").with_top_level_key("state_dict").allow_partial(true);
```

Extras: `with_key_remapping(regex, repl)` / `KeyRemapper`, `with_regex(..)` filtering, `zero_copy(true)` / `from_static(bytes)`, `HalfPrecisionAdapter`. Export from PyTorch with `torch.save(model.state_dict(), ...)` (the **state_dict**, not the whole model).

## ONNX import

Generate native Rust + a `.bpk` weights file at build time (no ONNX runtime dependency; output is trainable). Recommend **opset 16+**.

```toml
[build-dependencies]
burn-onnx = "~0.21"
```

```rust
// build.rs
burn_onnx::ModelGen::new().input("src/model/m.onnx").out_dir("model/").run_from_script();
// src/model/mod.rs
pub mod m { include!(concat!(env!("OUT_DIR"), "/model/m.rs")); }
// usage
let model: Model<Backend> = Model::default();          // or ::new(&device) / ::from_file / ::from_bytes
```

`LoadStrategy::{File (default), Embedded, Bytes, None}` controls which constructors are generated (use `Bytes` for WASM/embedded).

## Performance

- **Sync points** force the device to finish: `into_data`, `into_scalar`, `sync` (and `to_device` on some backends). Minimize them. Batch reads into one sync with a `Transaction`:
  ```rust
  let [output, loss, targets] = Transaction::default()
      .register(output).register(loss).register(targets)
      .execute().try_into().unwrap();                    // single sync → TensorData on CPU
  ```
- **Kernel fusion** (`fusion` feature): keep tensors short-lived (a tensor used later forces a global-memory write); **group view ops together** (`slice`/`reshape`/`swap_dims`/`unsqueeze`/`transpose`/`select`/`gather` only touch metadata) before compute blocks; don't `clone()` right before a compute-bound op _unless_ the tensor is materialized (e.g. a model param).
- **Autotune** picks kernels via runtime micro-benchmarks (cold start, cached on disk). Bundle the cache for spot instances/deploys.
- **Shapes**: prefer multiples of 32 / powers of two (`[1024, 1024]`); avoid multiples of 10 (`[1000, 1000]`) — they trigger bounds checks and hurt vectorization. If stuck with an odd shape, reshape it to a good one in a single kernel.
- **Data on a different device/backend** than training to avoid stalling the training device; do batch `cat` on the data device, then move with `Tensor::from_data(items.into_data(), &train_dev)`.
- `cumsum`/`cumprod` are real kernels — prefer them over `tril(0).matmul(x)` (O(T²)).

## Quantization (PTQ only; in active development)

Post-training quantization of weights; activations are dequantized to float for compute (no integer ops yet). QAT not supported.

```rust
let scheme = QuantScheme::default()
    .with_level(QuantLevel::Block(32)).with_value(QuantValue::Q4F).with_param(QuantParam::F16);
let mut quantizer = Quantizer { calibration: Calibration::MinMax, scheme };
let model = model.quantize_weights(&mut quantizer);
```

Values: `Q8F/Q4F/Q2F`, `Q8S/Q4S/Q2S`, `E5M2/E4M3/E2M1`. Levels: `Tensor`, `Block(size)`. See tensor-reference.md.

## no_std / WASM

`Flex` backend supports `no_std` (provide a `#[global_allocator]`) and WASM (`Flex`/`WebGpu`; `getrandom` needs explicit config on `WebGpu`). ONNX models embed weights with `LoadStrategy::Bytes`/`Embedded`. Otherwise the API is identical.

## Common trip-ups

| Symptom                                               | Cause / fix                                                                                                |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| "method not found" on `get`/`save_file`/`save`/`elem` | Trait not in scope — `use burn::prelude::*;` or the specific trait.                                        |
| Recursive type evaluation error (wgpu)                | Add `#![recursion_limit = "256"]` at crate root.                                                           |
| bf16/f16 won't compile on CPU                         | ndarray is f32/f64 only; bf16/f16 need a GPU/cubecl backend (or guard the element type so CPU stays f32).  |
| `backward()` not available                            | Backend isn't `Autodiff<_>` — train on `AutodiffBackend`, eval on inner.                                   |
| Calling `.valid()` on `result.model`                  | `LearningResult.model` is already on the inner backend.                                                    |
| Validation won't compile                              | Validation model/batcher must be on `B::InnerBackend` (use `model.valid()`).                               |
| Slow / extra GPU writes                               | Don't keep tensors alive or clone before compute-bound ops; group view ops; check shapes are 32-multiples. |
| Frequent stalls                                       | Hidden sync from `into_data`/`into_scalar` in a hot loop — batch with `Transaction`.                       |
| `burn-store` "missing source values"                  | Saved the whole PyTorch model — re-export `state_dict()`.                                                  |
| Load returns wrong/missing keys                       | Use `with_key_remapping(..)`; inspect `store.keys()`.                                                      |

Validate novel kernels/layers by **training to overfit a synthetic task** (loss → ~0), not by keeping a reference impl and an equivalence test.
