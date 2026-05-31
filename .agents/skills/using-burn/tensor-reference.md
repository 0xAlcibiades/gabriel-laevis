# Burn 0.21 — Tensor Operation Reference

Exhaustive op catalog for `Tensor<B, D, K>`. Type signatures omitted for brevity; see
[docs.rs/burn Tensor](https://docs.rs/burn/latest/burn/tensor/struct.Tensor.html). `cargo check` is
the source of truth. Most ops take `self` by value — `.clone()` to reuse.

## Tensor kinds

```rust
Tensor<B, D>           // Float (default)
Tensor<B, D, Float>    // explicit float
Tensor<B, D, Int>      // int
Tensor<B, D, Bool>     // bool
```

`D` = rank (compile-time const), not shape. Concrete element types (`f32`, `f16`, `bf16`, `i32`, …)
are fixed by the backend, not the tensor type.

## Initialization

```rust
Tensor::<B, 2>::from_data([[2., 3.], [4., 5.]], &device);
Tensor::<B, 1>::from_data(TensorData::from([1.0, 2.0, 3.0]), &device);
Tensor::<B, 1>::from_floats([1., 2., 3.], &device);            // recommended for f32
Tensor::<B, 2>::from_data(TensorData::new(vec, [rows, cols]), &device);
Tensor::<B, 1, Int>::from_data(TensorData::from(&arr[0..3]), &device);
```

`TensorData` fields: `shape`, `dtype`; bytes are private — access via `as_slice` / `as_mut_slice` /
`to_vec` / `iter`. `.convert::<B::FloatElem>()` adjusts precision to the backend. `.elem::<T>()`
(needs `ElementConversion`) converts a single scalar.

Retrieve: `.to_data()` (keep tensor) vs `.into_data()` (one-shot). Both **sync the device**.

## Basic operations (Int, Float, Bool)

| Op                                                                              | PyTorch                          |
| ------------------------------------------------------------------------------- | -------------------------------- |
| `Tensor::cat(tensors, dim)`                                                     | `torch.cat`                      |
| `Tensor::stack(tensors, dim)`                                                   | `torch.stack`                    |
| `Tensor::empty(shape, &dev)`                                                    | `torch.empty`                    |
| `Tensor::zeros/ones/full(shape, [v,] &dev)`                                     | `torch.zeros/ones/full`          |
| `Tensor::from_primitive` / `tensor.into_primitive`                              | N/A                              |
| `tensor.all()` / `all_dim(dim)`                                                 | `tensor.all([dim])`              |
| `tensor.any()` / `any_dim(dim)`                                                 | `tensor.any([dim])`              |
| `tensor.chunk(n, dim)`                                                          | `tensor.chunk`                   |
| `tensor.split(size, dim)` / `split_with_sizes(sizes, dim)`                      | `tensor.split`                   |
| `tensor.device()` / `dtype()` / `dims()` / `shape()`                            | `tensor.device/dtype/size/shape` |
| `tensor.equal(other)` / `equal_elem(s)`                                         | `x == y` / `tensor.eq`           |
| `tensor.not_equal(other)` / `not_equal_elem(s)`                                 | `x != y` / `tensor.ne`           |
| `tensor.expand(shape)`                                                          | `tensor.expand`                  |
| `tensor.flatten(start, end)`                                                    | `tensor.flatten`                 |
| `tensor.flip(axes)`                                                             | `tensor.flip`                    |
| `tensor.full_like(v)` / `ones_like()` / `zeros_like()`                          | `torch.*_like`                   |
| `tensor.gather(dim, idx)` / `scatter(dim, idx, vals)`                           | `torch.gather` / `scatter_add`   |
| `tensor.gather_nd(idx)` / `scatter_nd(idx, vals)`                               | N/A                              |
| `tensor.select(dim, idx)` / `select_assign(dim, idx, vals)`                     | `index_select` / `index_add`     |
| `tensor.into_scalar()`                                                          | `tensor.item()`                  |
| `tensor.mask_fill(mask, v)` / `mask_where(mask, vals)`                          | `masked_fill` / `where`          |
| `tensor.movedim(src, dst)`                                                      | `tensor.movedim`                 |
| `tensor.narrow(dim, start, len)`                                                | `tensor.narrow`                  |
| `tensor.permute(axes)`                                                          | `tensor.permute`                 |
| `tensor.repeat(sizes)` / `repeat_dim(dim, times)`                               | `tensor.repeat`                  |
| `tensor.reshape(shape)`                                                         | `tensor.view`                    |
| `tensor.roll(shifts, dims)` / `roll_dim(shift, dim)`                            | `tensor.roll`                    |
| `tensor.slice(ranges)` / `slice_assign(ranges, vals)` / `slice_fill(ranges, v)` | `tensor[ranges]`                 |
| `tensor.slice_dim(dim, slice)`                                                  | N/A                              |
| `tensor.squeeze()` / `squeeze_dim(dim)` / `squeeze_dims(dims)`                  | `tensor.squeeze`                 |
| `tensor.unsqueeze()` / `unsqueeze_dim(dim)` / `unsqueeze_dims(dims)`            | `tensor.unsqueeze`               |
| `tensor.swap_dims(d1, d2)`                                                      | `tensor.transpose(d1, d2)`       |
| `tensor.transpose()` / `t()`                                                    | `tensor.T`                       |
| `tensor.take(dim, idx)`                                                         | `numpy.take`                     |
| `tensor.to_device(dev)`                                                         | `tensor.to(dev)`                 |

Note: `unsqueeze_dim::<D2>(dim)` lets you grow the rank with a turbofish target rank.

## Numeric operations (Int, Float)

Arithmetic: `add`/`sub`/`mul`/`div`/`rem` (or `+ - * / %`); scalar variants `add_scalar`/`sub_scalar`/
`mul_scalar`/`div_scalar` (or `+ - * /` with a scalar); `neg()` / `-tensor`; `scalar - tensor` works.

| Op                                                                     | PyTorch                             |
| ---------------------------------------------------------------------- | ----------------------------------- |
| `abs()`                                                                | `torch.abs`                         |
| `all_close(other, atol, rtol)`                                         | `torch.allclose`                    |
| `argmax(dim)` / `argmin(dim)`                                          | `tensor.argmax/argmin`              |
| `argsort(dim)` / `argsort_descending(dim)`                             | `tensor.argsort`                    |
| `sort(dim)` / `sort_descending(dim)` / `*_with_indices(dim)`           | `tensor.sort`                       |
| `topk(k, dim)` / `topk_with_indices(k, dim)`                           | `tensor.topk`                       |
| `bool()`                                                               | `tensor.bool()`                     |
| `clamp(min, max)` / `clamp_min(min)` / `clamp_max(max)`                | `torch.clamp`                       |
| `cumsum(dim)` / `cumprod(dim)` / `cummin(dim)` / `cummax(dim)`         | `tensor.cum*`                       |
| `dot(other)`                                                           | `torch.dot`                         |
| `greater[_equal][_elem]` / `lower[_equal][_elem]`                      | `gt/ge/lt/le`                       |
| `max()` / `min()`                                                      | `tensor.max/min`                    |
| `max_dim(dim)` / `min_dim(dim)` / `*_dims(dims)` (keepdim)             | `tensor.max/min(dim, keepdim=True)` |
| `max_dim_with_indices(dim)` / `min_dim_with_indices(dim)`              | N/A                                 |
| `max_pair(other)` / `min_pair(other)`                                  | `torch.max/min(a, b)`               |
| `max_abs()` / `max_abs_dim(dim)` / `max_abs_dims(dims)`                | `tensor.abs().max(..)`              |
| `mean()` / `mean_dim(dim)` / `mean_dims(dims)` (keepdim)               | `tensor.mean([dim])`                |
| `sum()` / `sum_dim(dim)` / `sum_dims(dims)` / `sum_dims_squeeze(dims)` | `tensor.sum([dim])`                 |
| `prod()` / `prod_dim(dim)` / `prod_dims(dims)`                         | `tensor.prod([dim])`                |
| `one_hot(num_classes)` / `one_hot_fill(n, on, off, axis)`              | `F.one_hot`                         |
| `pad(pads, mode)`                                                      | `F.pad`                             |
| `powf(t)` / `powi(t)` / `powf_scalar(s)` / `powi_scalar(s)`            | `tensor.pow`                        |
| `sign()`                                                               | `tensor.sign`                       |
| `tril(diagonal)` / `triu(diagonal)`                                    | `torch.tril/triu`                   |
| `unfold(dim, size, step)`                                              | `tensor.unfold`                     |
| `Tensor::eye(size, &dev)`                                              | `torch.eye`                         |

## Float-only operations

Trig/inverse-trig/hyperbolic: `sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh atan2(t)`.
Rounding: `ceil floor round trunc`. Misc:

| Op                                                                      | PyTorch                                  |
| ----------------------------------------------------------------------- | ---------------------------------------- |
| `cast(dtype)`                                                           | `tensor.to(dtype)`                       |
| `contains_nan()`                                                        | N/A                                      |
| `cross(other)`                                                          | `torch.cross`                            |
| `deg2rad()` / `rad2deg()`                                               | `torch.deg2rad/rad2deg`                  |
| `erf()`                                                                 | `tensor.erf`                             |
| `exp()` / `log()` / `log1p()`                                           | `tensor.exp/log/log1p`                   |
| `fmod(t)` / `fmod_scalar(s)`                                            | `tensor.fmod`                            |
| `int()`                                                                 | `tensor.to(torch.long)`                  |
| `is_close(other, atol, rtol)` / `is_finite()` / `is_inf()` / `is_nan()` | `torch.isclose/isfinite/isinf/isnan`     |
| `matmul(other)`                                                         | `tensor.matmul` (batched on last 2 dims) |
| `random(shape, dist, &dev)` / `random_like(dist)`                       | N/A / `torch.rand_like`                  |
| `recip()` / `1.0 / tensor`                                              | `tensor.reciprocal`                      |
| `square()` / `sqrt()`                                                   | `tensor.square/sqrt`                     |
| `var(dim)` / `var_bias(dim)` / `var_mean(dim)` / `var_mean_bias(dim)`   | `tensor.var`                             |
| `median(dim)` / `median_with_indices(dim)`                              | `tensor.median`                          |

`Distribution`: `Default` (uniform [0,1)), `Uniform(lo, hi)`, `Normal(mean, std)`, `Bernoulli(p)`.

## Int-only operations

`Tensor::arange(5..10, &dev)`, `Tensor::arange_step(5..10, 2, &dev)`, `from_ints(ints)`,
`cartesian_grid(shape, &dev)`, `float()`. Bitwise: `bitwise_and/or/xor[_scalar]`, `bitwise_not`,
`bitwise_left_shift[_scalar]`, `bitwise_right_shift[_scalar]`.

## Bool-only operations

`Tensor::diag_mask(shape, diag)`, `Tensor::tril_mask(shape, diag)`, `Tensor::triu_mask(shape, diag)`,
`argwhere()`, `nonzero()`, `bool_and/bool_or/bool_xor/bool_not`, `float()`, `int()`.

## Activation functions (`burn::tensor::activation`)

`celu(t, α)`, `elu(t, α)`, `gelu(t)`, `glu(t, dim)`, `hard_shrink(t, λ)`, `hard_sigmoid(t, α, β)`,
`hard_swish(t)`, `leaky_relu(t, slope)`, `log_sigmoid(t)`, `log_softmax(t, dim)`, `mish(t)`,
`prelu(t, α)`, `quiet_softmax(t, dim)`, `relu(t)`, `selu(t)`, `shrink(t, λ, bias)`,
`soft_shrink(t, λ)`, `sigmoid(t)`, `silu(t)`, `softmax(t, dim)`, `softmin(t, dim)`,
`softplus(t, β)`, `softsign(t)`, `tanh(t)`, `thresholded_relu(t, α)`.

## Grid (`burn::tensor::grid`)

`affine_grid_2d(theta, dims)`, `meshgrid(tensors, GridIndexing::Matrix | Cartesian)`,
`meshgrid_stack(tensors, index_pos)`.

## Linalg (`burn::tensor::linalg`)

`cosine_similarity(x1, x2, dim, eps)`, `det(t)`, `diag(t)`, `lu(t)`, `trace(t)`, `outer(a, b)`,
`outer_dim(a, b, dim)`, `matvec(m, v)`, `vector_norm(t, p, dim)`, `vector_normalize(t, norm, dim, eps)`,
`l0_norm`/`l1_norm`/`l2_norm(t, dim)`, `lp_norm(t, p, dim)`, `max_abs_norm`/`min_abs_norm(t, dim)`.

## Signal (`burn::tensor::signal`)

Real-valued float tensors. FFT length `n` (and `n_fft`) **must be a power of two** (else panics at the
API boundary); output has `n/2 + 1` bins.

`rfft(t, dim, n)`, `irfft(re, im, dim, n)`, `stft(signal, window, opts)`, `istft(matrix, window, length, opts)`,
`blackman_window/hamming_window/hann_window(size, periodic, opts)`.

`StftOptions::new(n_fft)` → PyTorch defaults (`hop_length = n_fft/4`, `win_length = None`,
`center = true`, `onesided = true`); validated on entry (`hop_length <= win_length`, COLA for invertibility).

## Quantization values (PTQ)

`QuantScheme::default().with_mode(QuantMode::Symmetric).with_level(..).with_value(..).with_store(..).with_param(..)`

- **Level**: `Tensor` (one param set) | `Block(size)` / `block([d0, d1])`.
- **Value**: `Q8F Q4F Q2F` (full-range), `Q8S Q4S Q2S` (symmetric), `E5M2 E4M3` (8-bit float), `E2M1` (4-bit float).
- **Store**: `Native` (not for sub-byte) | `PackedNative(dim)` | `PackedU32(dim)`.
- **Param** (scale precision): `F32 | F16 | BF16`.
- **Calibration**: `MinMax`.

## Display & debugging

```rust
println!("{tensor}");          // full
println!("{:.2}", tensor);     // 2 decimals
set_print_options(PrintOptions { precision: Some(2), threshold: Some(1000), edge_items: Some(3), ..Default::default() });
check_closeness(&a, &b);       // element-wise tolerance report (PASS/WARN/FAIL) — great for porting models
```
