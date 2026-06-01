#![doc = include_str!("../README.md")]
// Required by Burn's wgpu/cubecl backend: its deeply nested associated types
// exceed the default recursion limit (128) during trait resolution.
#![recursion_limit = "256"]

pub mod chat;
pub mod config;
pub mod constants;
pub mod data;
pub mod model;
pub mod train;

// Float element: bf16 (feature) or f32. The whole model — scan included — runs in
// `Elem`; there is no internal f32 cast. bf16 is fine for the scan: it has the same
// 8-bit exponent as f32 (identical dynamic range), the chunked decay is segment-summed
// in log space and masked before exp (overflow-free, no catastrophic cancellation), and
// on CUDA the matmuls accumulate in f32 on the tensor cores regardless of bf16 storage.
// bf16 is a no-op on CPU (f32-only) and broken on Metal (cubecl MSL codegen emits an
// invalid float→bfloat cast for exp/log), so it's CUDA-only: `--features cuda,bf16`.
#[cfg(all(
    feature = "bf16",
    any(feature = "cuda", feature = "metal", feature = "wgpu")
))]
pub type Elem = burn::tensor::bf16;
#[cfg(not(all(
    feature = "bf16",
    any(feature = "cuda", feature = "metal", feature = "wgpu")
)))]
pub type Elem = f32;

// Selected compute backend (cuda > metal > wgpu > cpu) and its autodiff wrapper.
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

// Autodiff wrapper. `checkpoint` opts into Burn's BalancedCheckpointing strategy —
// recompute cheap (elementwise) ops in backward instead of storing them, trading a
// little compute for activation memory. Off by default: it's near a wash at the
// laevis tier (the head logits, not activations, dominate memory), but a real lever
// once memory-bound at larger sizes / long context. See README.
#[cfg(not(feature = "checkpoint"))]
pub type Train = burn::backend::Autodiff<Compute>;
#[cfg(feature = "checkpoint")]
pub type Train = burn::backend::Autodiff<
    Compute,
    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
>;
