//! Compile-time backend selection. Burn provides the backend types but can't choose one, so
//! the crate aliases them here from the backend features: `Elem` (float element), `Compute`
//! (the chosen inference backend), and `Train` (its autodiff wrapper).

// Float element. `bf16` is CUDA-only: a no-op on CPU (f32-only) and broken on Metal (cubecl
// MSL codegen emits an invalid float→bfloat cast for exp/log). On CUDA the matmuls accumulate
// in f32 on the tensor cores regardless of bf16 storage, and the chunked scan stays
// overflow-safe (bf16 has f32's 8-bit exponent; decay is segment-summed in log space and
// masked before exp). Use `--features cuda,bf16`; other backends fall back to f32.
#[cfg(all(feature = "bf16", feature = "cuda"))]
pub type Elem = burn::tensor::bf16;
#[cfg(not(all(feature = "bf16", feature = "cuda")))]
pub type Elem = f32;

// Exactly one backend, by priority (cuda > metal > wgpu > cpu). The `not(...)` chain resolves
// the common case where the default `cpu` is enabled alongside an explicitly added GPU backend.
#[cfg(feature = "cuda")]
pub type Compute = burn::backend::Cuda<Elem>;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub type Compute = burn::backend::Metal<Elem>;
#[cfg(all(feature = "wgpu", not(feature = "metal"), not(feature = "cuda")))]
pub type Compute = burn::backend::Wgpu<Elem>;
#[cfg(all(
    feature = "cpu",
    not(feature = "wgpu"),
    not(feature = "metal"),
    not(feature = "cuda")
))]
pub type Compute = burn::backend::NdArray<Elem>;

// Autodiff wrapper used for training. `checkpoint` opts into Burn's BalancedCheckpointing —
// recompute cheap (elementwise) ops in backward instead of storing them, trading a little
// compute for activation memory; off by default. See README.
#[cfg(all(feature = "train", not(feature = "checkpoint")))]
pub type Train = burn::backend::Autodiff<Compute>;
#[cfg(all(feature = "train", feature = "checkpoint"))]
pub type Train = burn::backend::Autodiff<
    Compute,
    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
>;
