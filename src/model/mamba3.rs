//! Mamba-3 (SISO) block — implemented from the paper (arXiv:2603.15569)

use burn::module::Initializer;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::activation;

use crate::config::ModelConfig;

/// Minimum decay rate: A is clamped to `≤ -A_FLOOR` so the state always forgets a
/// little (matches the reference `A_floor`).
const A_FLOOR: f64 = 1e-4;
/// Δ initialization range (reference `dt_min`/`dt_max`), inverse-softplus'd into
/// `dt_bias` so the initial Δ is spread log-uniformly over this range.
const DT_MIN: f64 = 0.001;
const DT_MAX: f64 = 0.1;
const DT_INIT_FLOOR: f64 = 1e-4;

/// Apply real 2×2 rotations to paired channels a.k.a the "RoPE trick" - paper Prop. 2.
///
/// `v`: `[batch, pairs, 2]` (each row a complex number `(re, im)`),
/// `angle`: `[batch, pairs]`. Returns the rotated pairs `[batch, pairs, 2]`:
/// `(x cosθ − y sinθ, x sinθ + y cosθ)`.
pub fn rotate_pairs<B: Backend>(v: Tensor<B, 3>, angle: Tensor<B, 2>) -> Tensor<B, 3> {
    let [batch, pairs, _] = v.dims();
    let cos = angle.clone().cos().unsqueeze_dim::<3>(2); // [batch, pairs, 1]
    let sin = angle.sin().unsqueeze_dim::<3>(2);
    let x = v.clone().slice([0..batch, 0..pairs, 0..1]);
    let y = v.slice([0..batch, 0..pairs, 1..2]);
    let xr = x.clone().mul(cos.clone()).sub(y.clone().mul(sin.clone()));
    let yr = x.mul(sin).add(y.mul(cos));
    Tensor::cat(vec![xr, yr], 2)
}

/// Rotate the leading `2*pairs` channels of a `[R, T, K]` sequence pairwise by
/// per-pair angles `[R, T, pairs]`; channels past `2*pairs` are left unrotated
/// (half-RoPE). `R` is any flattened leading (batch·heads) dimension.
fn rotate_seq<B: Backend>(v: Tensor<B, 3>, angle: Tensor<B, 3>, pairs: usize) -> Tensor<B, 3> {
    let [r, t, k] = v.dims();
    let rd = 2 * pairs;
    let head = v.clone().slice([0..r, 0..t, 0..rd]);
    let rot = rotate_pairs(
        head.reshape([r * t, pairs, 2]),
        angle.reshape([r * t, pairs]),
    );
    let rotated = rot.reshape([r, t, rd]);
    if rd == k {
        rotated
    } else {
        Tensor::cat(vec![rotated, v.slice([0..r, 0..t, rd..k])], 2)
    }
}

/// Per-timestep SSM coefficients produced by [`Mamba3Block::project`].
/// Shapes are for input `[batch, seq, d_model]`.
pub struct Coeffs<B: Backend> {
    /// SSM input (x-branch), `[B, T, d_inner]`.
    pub x: Tensor<B, 3>,
    /// Gate branch (pre-activation), `[B, T, d_inner]`.
    pub z: Tensor<B, 3>,
    /// Per-step log-decay `La_t = Δ_t·A_t` (≤ 0), **per head** `[B, T, nheads]`.
    pub la: Tensor<B, 3>,
    /// Trapezoidal coefficient for the previous input, `[B, T, d_inner]`.
    pub beta: Tensor<B, 3>,
    /// Trapezoidal coefficient for the current input, `[B, T, d_inner]`.
    pub gamma: Tensor<B, 3>,
    /// Input projection `B_t` (post BCNorm + bias), **per group** `[B, T, ngroups, N]`.
    pub b: Tensor<B, 4>,
    /// Output projection `C_t` (post BCNorm + bias), **per group** `[B, T, ngroups, N]`.
    pub c: Tensor<B, 4>,
    /// Per-pair rotation rates `θ_t` for the half-RoPE, `[B, T, num_rope_angles]`.
    pub theta: Tensor<B, 3>,
}

/// Recurrent SSM state for O(1)-per-token incremental (cached) generation.
pub struct Mamba3State<B: Backend> {
    h: Tensor<B, 3>,      // [B, d_inner, N]
    phi: Tensor<B, 2>,    // [B, num_rope_angles] cumulative rotation phase
    x_prev: Tensor<B, 2>, // [B, d_inner]
    b_prev: Tensor<B, 3>, // [B, nheads, N] (rotated B, per head)
}

impl<B: Backend> Mamba3State<B> {
    /// Tile a batch-1 state to `g` identical rows (used to prime a prompt once and
    /// fan it out to a group of parallel rollouts). `cat` of `g` clones materializes a
    /// contiguous `[g, ...]` state; the per-row recurrence is independent thereafter.
    pub fn broadcast_batch(self, g: usize) -> Self {
        debug_assert_eq!(
            self.h.dims()[0],
            1,
            "broadcast_batch expects a batch-1 state"
        );
        Mamba3State {
            h: Tensor::cat(vec![self.h; g], 0),
            phi: Tensor::cat(vec![self.phi; g], 0),
            x_prev: Tensor::cat(vec![self.x_prev; g], 0),
            b_prev: Tensor::cat(vec![self.b_prev; g], 0),
        }
    }
}

#[derive(Module, Debug)]
pub struct Mamba3Block<B: Backend> {
    // All seven per-step input projections share the same input `u` and the same fan-in
    // (d_model), so they are fused into one `d_model -> proj_out` matmul and sliced in
    // `project` (one GEMM/launch per block instead of seven; KaimingUniform keys off
    // fan-in, so the fused init is distributionally identical to seven separate ones).
    // Output layout: [ xz(2·d_inner) | B(ngroups·N) | C(ngroups·N) | Δ(nheads) |
    //                  A(nheads) | λ(nheads) | θ(num_rope_angles) ].
    in_proj: Linear<B>,
    out_proj: Linear<B>,          // d_inner -> d_model
    bc_norm: RmsNorm<B>,          // QK-norm on B and C (over d_state)
    dt_bias: Param<Tensor<B, 1>>, // [nheads], inverse-softplus init
    d: Param<Tensor<B, 1>>,       // [nheads], skip, init ones
    b_bias: Param<Tensor<B, 1>>,  // [d_state], init ones
    c_bias: Param<Tensor<B, 1>>,  // [d_state], init ones
    d_inner: usize,
    d_state: usize,
    nheads: usize,
    headdim: usize,
    ngroups: usize,
    num_rope_angles: usize,
}

impl<B: Backend> Mamba3Block<B> {
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> Self {
        let d_model = cfg.d_model;
        let d_inner = cfg.d_inner();
        let n = cfg.d_state;
        let headdim = cfg.headdim;
        let ngroups = cfg.ngroups;
        debug_assert!(
            d_inner.is_multiple_of(headdim),
            "d_inner must be divisible by headdim"
        );
        debug_assert!(
            n.is_multiple_of(4),
            "d_state must be a multiple of 4 for half-RoPE"
        );
        let nheads = d_inner / headdim;
        debug_assert!(
            nheads.is_multiple_of(ngroups),
            "nheads must be divisible by ngroups"
        );
        let num_rope_angles = n / 4; // rope_fraction = 0.5: rotate N/2 dims = N/4 pairs
        let lin = |i: usize, o: usize| LinearConfig::new(i, o).with_bias(false).init(device);

        // dt_bias: inverse-softplus of Δ ~ exp(U[ln dt_min, ln dt_max]) so softplus(
        // dt_bias) reproduces a log-uniform initial Δ (reference dt init).
        let dt = Tensor::<B, 1>::random(
            [nheads],
            Distribution::Uniform(DT_MIN.ln(), DT_MAX.ln()),
            device,
        )
        .exp()
        .clamp_min(DT_INIT_FLOOR);
        // dt_bias = dt + log(1 - exp(-dt)) = dt + log(-expm1(-dt)).
        let dt_bias = dt.clone().add(dt.neg().exp().neg().add_scalar(1.0).log());

        // Fused input projection width: xz | B | C | Δ | A | λ | θ.
        let proj_out = 2 * d_inner + 2 * ngroups * n + 3 * nheads + num_rope_angles;

        Self {
            in_proj: lin(d_model, proj_out),
            out_proj: lin(d_inner, d_model),
            bc_norm: RmsNormConfig::new(n).init(device),
            dt_bias: Param::from_tensor(dt_bias),
            d: Initializer::Ones.init([nheads], device),
            b_bias: Initializer::Ones.init([n], device),
            c_bias: Initializer::Ones.init([n], device),
            d_inner,
            d_state: n,
            nheads,
            headdim,
            ngroups,
            num_rope_angles,
        }
    }

    /// Repeat a per-head `[..., nheads]` tensor over `headdim` to a per-channel
    /// `[..., d_inner]` one (channel `c` takes head `c / headdim`).
    fn heads_to_channels(&self, h: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, _] = h.dims();
        h.unsqueeze_dim::<4>(3)
            .expand([b, t, self.nheads, self.headdim])
            .reshape([b, t, self.d_inner])
    }

    /// `D` broadcast to `[1, 1, d_inner]` (per-head skip repeated over `headdim`).
    fn d_channels(&self) -> Tensor<B, 3> {
        self.d
            .val()
            .reshape([self.nheads, 1])
            .expand([self.nheads, self.headdim])
            .reshape([1, 1, self.d_inner])
    }

    /// Normalize + bias a B/C projection `[B, T, ngroups*N]` → `[B, T, ngroups, N]`.
    fn norm_bc(&self, proj: Tensor<B, 3>, bias: Tensor<B, 1>) -> Tensor<B, 4> {
        let [b, t, _] = proj.dims();
        let n = self.d_state;
        let g = self.ngroups;
        let normed = self.bc_norm.forward(proj.reshape([b, t, g, n]));
        normed.add(bias.reshape([1, 1, 1, n]))
    }

    /// Input `[B, T, d_model]` → per-timestep SSM coefficients.
    pub fn project(&self, u: Tensor<B, 3>) -> Coeffs<B> {
        let [b, t, _] = u.dims();
        let di = self.d_inner;
        let nh = self.nheads;
        let n = self.d_state;
        let gn = self.ngroups * n;
        let na = self.num_rope_angles;

        // Single fused matmul, then slice each segment (all views — grouped before the
        // compute below). Layout: xz | B | C | Δ | A | λ | θ.
        let proj = self.in_proj.forward(u); // [B,T,proj_out]
        let seg = |start: usize, end: usize| proj.clone().slice([0..b, 0..t, start..end]);
        let two_di = 2 * di;
        let off_b = two_di;
        let off_c = off_b + gn;
        let off_dt = off_c + gn;
        let off_a = off_dt + nh;
        let off_trap = off_a + nh;
        let off_theta = off_trap + nh;

        let xz = seg(0, two_di);
        let x = xz.clone().slice([0..b, 0..t, 0..di]);
        let z = xz.slice([0..b, 0..t, di..two_di]);
        let b_raw = seg(off_b, off_c);
        let c_raw = seg(off_c, off_dt);
        let dt_raw = seg(off_dt, off_a);
        let a_raw = seg(off_a, off_trap);
        let trap_raw = seg(off_trap, off_theta);
        let theta = seg(off_theta, off_theta + na);

        // Per-head Δ, A (selective), λ.
        let dt_bias = self.dt_bias.val().reshape([1, 1, nh]);
        let delta = activation::softplus(dt_raw.add(dt_bias), 1.0); // [B,T,nh]
        let a = activation::softplus(a_raw, 1.0).neg().clamp_max(-A_FLOOR); // A ≤ -A_FLOOR [B,T,nh]

        // Per-step log-decay La = Δ·A ≤ 0 (per head; the chunked segment-sum decay
        // bounds it, so no rate clamp is needed). α = exp(La).
        let la = delta.clone().mul(a); // [B,T,nh]
        let alpha = la.clone().exp();

        // λ = σ(trap); β = (1-λ)·Δ·α; γ = λ·Δ — broadcast per head to per channel.
        let lambda = activation::sigmoid(trap_raw);
        let one_minus_lambda = lambda.clone().neg().add_scalar(1.0);
        let beta = self.heads_to_channels(one_minus_lambda.mul(delta.clone()).mul(alpha));
        let gamma = self.heads_to_channels(lambda.mul(delta));

        // B_t, C_t = BCNorm(slice) + bias, per group [B,T,ngroups,N].
        let b_t = self.norm_bc(b_raw, self.b_bias.val());
        let c_t = self.norm_bc(c_raw, self.c_bias.val());

        Coeffs {
            x,
            z,
            la,
            beta,
            gamma,
            b: b_t,
            c: c_t,
            theta,
        }
    }

    /// Expand per-group B/C `[B, T, ngroups, N]` to per-head `[B, nheads, T, N]`
    /// (head `h` takes group `h / (nheads/ngroups)`) and apply the half-RoPE by the
    /// shared cumulative phase `phi` `[B, T, na]`.
    fn rope_heads(&self, bc: Tensor<B, 4>, phi: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, t, g, n] = bc.dims();
        let nh = self.nheads;
        let na = self.num_rope_angles;
        let hpg = nh / g;
        // group → head, then [B, nheads, T, N]
        let heads = bc
            .unsqueeze_dim::<5>(3)
            .expand([b, t, g, hpg, n])
            .reshape([b, t, nh, n])
            .swap_dims(1, 2); // [B, nh, T, N]
        // rotate each head by the shared phi
        let phih = phi
            .unsqueeze_dim::<4>(1)
            .expand([b, nh, t, na])
            .reshape([b * nh, t, na]);
        rotate_seq(heads.reshape([b * nh, t, n]), phih, na).reshape([b, nh, t, n])
    }

    /// Block forward: `[B, T, d_model] -> [B, T, d_model]`.
    pub fn forward(&self, u: Tensor<B, 3>) -> Tensor<B, 3> {
        let co = self.project(u);
        let y = self.scan(&co); // [B, T, d_inner]
        let y = y.add(self.d_channels().mul(co.x.clone())); // D skip
        let gated = y.mul(activation::silu(co.z)); // gate by SiLU(z)
        self.out_proj.forward(gated)
    }

    /// Parallel prefill that also returns the recurrent [`Mamba3State`] left after the
    /// last timestep, so a prompt can be consumed in one chunked scan (instead of
    /// `T` serial `step`s) and decoding resumes from the carried state. The `[B,T,d_model]`
    /// output is identical to [`Mamba3Block::forward`]; the returned state is identical
    /// (to float precision) to having called `step` `T` times — see
    /// `forward_with_state_matches_step_carry`.
    pub fn forward_with_state(&self, u: Tensor<B, 3>) -> (Tensor<B, 3>, Mamba3State<B>) {
        let co = self.project(u);
        let (y, h) = self.scan_chunked_carry(&co, crate::config::run().chunk);
        let y = y.add(self.d_channels().mul(co.x.clone())); // D skip
        let gated = y.mul(activation::silu(co.z.clone())); // gate by SiLU(z)
        let out = self.out_proj.forward(gated);

        // The remaining three state fields are last-timestep quantities. They need only
        // the final token, so derive them from that single column rather than over all T:
        //   phi    = Σ_t theta  (the cumulative phase at T-1; decode does phi += theta)
        //   x_prev = x[T-1]
        //   b_prev = rope(B[T-1], phi)  (rope a single step, not the whole sequence)
        let [bsz, t_len, di] = co.x.dims();
        let n = self.d_state;
        let nh = self.nheads;
        let na = self.num_rope_angles;
        let g = self.ngroups;
        let phi = co.theta.clone().sum_dim(1).reshape([bsz, na]); // Σ theta = cumsum[T-1]
        let x_prev =
            co.x.clone()
                .slice([0..bsz, t_len - 1..t_len, 0..di])
                .reshape([bsz, di]);
        // Rope only the last token's B (one [.,.,1,.] rope) instead of all T.
        let b_last = co.b.clone().slice([0..bsz, t_len - 1..t_len, 0..g, 0..n]); // [B,1,g,N]
        let b_prev = self
            .rope_heads(b_last, phi.clone().reshape([bsz, 1, na]))
            .reshape([bsz, nh, n]); // [B,nh,N]

        (
            out,
            Mamba3State {
                h,
                phi,
                x_prev,
                b_prev,
            },
        )
    }

    /// Chunked SSD scan with the run-configured chunk length → `[B, T, d_inner]`.
    fn scan(&self, co: &Coeffs<B>) -> Tensor<B, 3> {
        self.scan_chunked(co, crate::config::run().chunk)
    }

    /// Chunked SSD scan → `[B, T, d_inner]` (pre-gate, pre-skip), per head, following
    /// `ssd_minimal_discrete`. Per chunk it builds the per-head segment-sum decay
    /// `D[h,τ,σ] = exp(La_τ − La_σ)` (masked to `σ ≤ τ` before the exp → overflow-free),
    /// forms the intra-chunk output `(C·Bᵀ ∘ D)·V` for the current (`γx`,`B`) and
    /// previous (`βx_{-1}`,`B_{-1}`) trapezoidal terms, reads the carried state out
    /// decayed by `exp(La_τ)`, and carries the state to the chunk end. `c` is the chunk
    /// length: chunk size is mathematically irrelevant (the factorization is exact), so
    /// any drift vs `c` is pure float accumulation in the `[L,L]` score/decay matmuls —
    /// see the `chunk_precision_sweep` test.
    fn scan_chunked(&self, co: &Coeffs<B>, c: usize) -> Tensor<B, 3> {
        self.scan_chunked_carry(co, c).0
    }

    /// Same chunked SSD scan as [`scan_chunked`], additionally returning the final
    /// carried state `h` `[B, d_inner, N]` so a prefill can seed incremental decoding.
    fn scan_chunked_carry(&self, co: &Coeffs<B>, c: usize) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let c = c.max(1);
        let [bsz, t_len, di] = co.x.dims();
        let n = self.d_state;
        let nh = self.nheads;
        let hp = self.headdim;
        let device = co.x.device();

        // Per-head, half-RoPE'd B, C and the shifted previous B.
        let phi = co.theta.clone().cumsum(1); // [B,T,na]
        let b_rot = self.rope_heads(co.b.clone(), phi.clone()); // [B,nh,T,N]
        let c_rot = self.rope_heads(co.c.clone(), phi);
        let b_prev = {
            let zeros = Tensor::<B, 4>::zeros([bsz, nh, 1, n], &device);
            Tensor::cat(
                vec![
                    zeros,
                    b_rot.clone().slice([0..bsz, 0..nh, 0..t_len - 1, 0..n]),
                ],
                2,
            )
        };

        let gx = co.gamma.clone().mul(co.x.clone()); // γ·x                 [B,T,di]
        let bx = co.beta.clone().mul({
            let zeros = Tensor::<B, 3>::zeros([bsz, 1, di], &device);
            Tensor::cat(
                vec![zeros, co.x.clone().slice([0..bsz, 0..t_len - 1, 0..di])],
                1,
            )
        }); // β·x_{t-1}                                                    [B,T,di]

        let mut h = Tensor::<B, 3>::zeros([bsz, di, n], &device); // carried state [B,di,N]
        let mut ys: Vec<Tensor<B, 3>> = Vec::with_capacity(t_len.div_ceil(c));

        // Strict-upper (τ<σ) mask, built once at the full chunk width and sliced per chunk.
        // Every chunk but the last has l == width; the last slices the top-left [l,l]
        // block, which is exactly that chunk's strict-upper mask.
        let mask_w = c.min(t_len);
        let upper_full = Tensor::<B, 2>::ones([mask_w, mask_w], &device)
            .triu(1)
            .bool();

        let mut t0 = 0;
        while t0 < t_len {
            let l = (t_len - t0).min(c);
            let t1 = t0 + l;
            // [B,nh,T,*] → chunk [B,nh,L,*]
            let slh = |x: &Tensor<B, 4>, w: usize| x.clone().slice([0..bsz, 0..nh, t0..t1, 0..w]);
            // [B,T,di] → per-head chunk [B,nh,L,hp]
            let vheads = |v: Tensor<B, 3>| {
                v.slice([0..bsz, t0..t1, 0..di])
                    .reshape([bsz, l, nh, hp])
                    .swap_dims(1, 2)
            };

            // Per-head cumulative log-decay and segment-sum decay D[h,τ,σ].
            let la_c = co.la.clone().slice([0..bsz, t0..t1, 0..nh]).swap_dims(1, 2); // [B,nh,L]
            let acs = la_c.cumsum(2); // [B,nh,L]
            let diff = acs
                .clone()
                .unsqueeze_dim::<4>(3)
                .sub(acs.clone().unsqueeze_dim::<4>(2)); // [B,nh,L,L]: acs[τ]-acs[σ]
            let upper = upper_full
                .clone()
                .slice([0..l, 0..l])
                .reshape([1, 1, l, l])
                .expand([bsz, nh, l, l]);
            let d_mat = diff.mask_fill(upper, f32::NEG_INFINITY).exp(); // [B,nh,L,L] ∈ [0,1]

            // Per-head C·Bᵀ scores, decay-weighted.
            let crot = slh(&c_rot, n); // [B,nh,L,N]
            let brot = slh(&b_rot, n);
            let bprev = slh(&b_prev, n);
            let s_cur = crot.clone().matmul(brot.clone().swap_dims(2, 3)); // [B,nh,L,L]
            let s_prev = crot.clone().matmul(bprev.clone().swap_dims(2, 3));
            let score_cur = s_cur.mul(d_mat.clone());
            let score_prev = s_prev.mul(d_mat);

            // Intra-chunk: (C·Bᵀ ∘ D)·V for the current and previous terms.
            let gx_c = vheads(gx.clone()); // [B,nh,L,hp]
            let bx_c = vheads(bx.clone());
            let y_intra = score_cur
                .matmul(gx_c.clone())
                .add(score_prev.matmul(bx_c.clone())); // [B,nh,L,hp]

            // Inter-chunk: carried state read out, decayed by exp(La_τ).
            let h_heads = h.clone().reshape([bsz, nh, hp, n]); // [B,nh,hp,N]
            let ch = crot.matmul(h_heads.swap_dims(2, 3)); // [B,nh,L,hp]
            let y_inter = acs.clone().exp().unsqueeze_dim::<4>(3).mul(ch); // [B,nh,L,hp]

            let y_chunk = y_intra.add(y_inter).swap_dims(1, 2).reshape([bsz, l, di]);
            ys.push(y_chunk);

            // State carry: h_end = exp(La_last)·h + Σ_σ exp(La_last−La_σ)·V_σ⊗B_σ.
            let la_last = acs.clone().slice([0..bsz, 0..nh, l - 1..l]); // [B,nh,1]
            let decay = la_last.clone().sub(acs).exp().unsqueeze_dim::<4>(3); // [B,nh,L,1] ∈(0,1]
            let g_end = decay.clone().mul(gx_c).swap_dims(2, 3); // [B,nh,hp,L]
            let b_end = decay.mul(bx_c).swap_dims(2, 3);
            let contrib = g_end
                .matmul(brot) // [B,nh,hp,L]·[B,nh,L,N] → [B,nh,hp,N]
                .add(b_end.matmul(bprev))
                .reshape([bsz, di, n]);
            let e_last = la_last
                .exp()
                .reshape([bsz, nh, 1, 1])
                .expand([bsz, nh, hp, 1])
                .reshape([bsz, di, 1]); // exp(La_last) per channel [B,di,1]
            h = e_last.mul(h).add(contrib);

            t0 = t1;
        }

        (Tensor::cat(ys, 1), h) // [B, T, d_inner], final carry [B, d_inner, N]
    }

    /// Zeroed recurrent state for incremental generation.
    pub fn init_state(&self, batch: usize, device: &B::Device) -> Mamba3State<B> {
        Mamba3State {
            h: Tensor::zeros([batch, self.d_inner, self.d_state], device),
            phi: Tensor::zeros([batch, self.num_rope_angles], device),
            x_prev: Tensor::zeros([batch, self.d_inner], device),
            b_prev: Tensor::zeros([batch, self.nheads, self.d_state], device),
        }
    }

    /// One recurrent step for cached generation: `[B,1,d_model]` + state →
    /// (`[B,1,d_model]`, next state). The same per-head recurrence as `scan`.
    pub fn step(&self, u: Tensor<B, 3>, state: Mamba3State<B>) -> (Tensor<B, 3>, Mamba3State<B>) {
        let [b, _, _] = u.dims();
        let di = self.d_inner;
        let n = self.d_state;
        let nh = self.nheads;
        let hp = self.headdim;
        let g = self.ngroups;
        let na = self.num_rope_angles;
        let rd = 2 * na;

        let co = self.project(u);
        // Per-head, per-channel scalars (T = 1).
        let alpha = self
            .heads_to_channels(co.la.clone().exp())
            .reshape([b, nh, hp]);
        let beta = co.beta.reshape([b, nh, hp]);
        let gamma = co.gamma.reshape([b, nh, hp]);
        let x = co.x.reshape([b, nh, hp]);
        let z = co.z.reshape([b, di]);
        let theta = co.theta.reshape([b, na]);

        let phi = state.phi.add(theta); // [B,na]
        // Per-head B/C (group → head), then half-RoPE by the shared phi.
        let to_heads = |bc: Tensor<B, 4>| {
            bc.reshape([b, g, n])
                .unsqueeze_dim::<4>(2)
                .expand([b, g, nh / g, n])
                .reshape([b, nh, n]) // [B,nh,N]
        };
        let rotate = |full: Tensor<B, 3>| {
            let head = full.clone().slice([0..b, 0..nh, 0..rd]);
            let phih = phi
                .clone()
                .unsqueeze_dim::<3>(1)
                .expand([b, nh, na])
                .reshape([b * nh, na]);
            let rot = rotate_pairs(head.reshape([b * nh, na, 2]), phih).reshape([b, nh, rd]);
            if rd == n {
                rot
            } else {
                Tensor::cat(vec![rot, full.slice([0..b, 0..nh, rd..n])], 2)
            }
        };
        let b_rot = rotate(to_heads(co.b)); // [B,nh,N]
        let c_rot = rotate(to_heads(co.c));

        // h = α⊙h + (γ·x)⊗B + (β·x_prev)⊗B_prev, per head [B,nh,hp,N].
        let h_prev = state.h.reshape([b, nh, hp, n]);
        let xprev = state.x_prev.reshape([b, nh, hp]);
        let cur = gamma
            .mul(x.clone())
            .unsqueeze_dim::<4>(3)
            .mul(b_rot.clone().unsqueeze_dim::<4>(2));
        let prev = beta
            .mul(xprev)
            .unsqueeze_dim::<4>(3)
            .mul(state.b_prev.unsqueeze_dim::<4>(2));
        let h = alpha.unsqueeze_dim::<4>(3).mul(h_prev).add(cur).add(prev); // [B,nh,hp,N]

        let y = h
            .clone()
            .mul(c_rot.unsqueeze_dim::<4>(2))
            .sum_dim(3)
            .reshape([b, di]);
        let h = h.reshape([b, di, n]);
        let x = x.reshape([b, di]);
        // D skip, then gate by SiLU(z).
        let d = self.d_channels().reshape([1, di]);
        let y = y.add(d.mul(x.clone()));
        let out = self.out_proj.forward(y.mul(activation::silu(z))); // [B, d_model]
        let dm = out.dims()[1];
        let out = out.reshape([b, 1, dm]);

        (
            out,
            Mamba3State {
                h,
                phi,
                x_prev: x,
                b_prev: b_rot,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    type B = burn::backend::NdArray;

    /// Small config so tests stay fast. headdim 32 divides d_inner = expand·d_model =
    /// 128 → 4 heads; d_state 32 → 8 rope angles.
    fn small(vocab: usize) -> ModelConfig {
        ModelConfig::new()
            .with_vocab_size(vocab)
            .with_d_model(64)
            .with_n_layers(2)
            .with_d_state(32)
            .with_headdim(32)
            .with_d_ff(128)
    }

    #[test]
    fn rotation_helper_is_correct() {
        let device = Default::default();
        let v = Tensor::<B, 3>::from_floats([[[1.0f32, 0.0]]], &device);

        let r0 = rotate_pairs(v.clone(), Tensor::<B, 2>::zeros([1, 1], &device));
        let r0 = r0.into_data().to_vec::<f32>().unwrap();
        assert!(
            (r0[0] - 1.0).abs() < 1e-5 && r0[1].abs() < 1e-5,
            "rot0 = {r0:?}"
        );

        let ang = Tensor::<B, 2>::from_floats([[std::f32::consts::FRAC_PI_2]], &device);
        let r90 = rotate_pairs(v, ang).into_data().to_vec::<f32>().unwrap();
        assert!(
            r90[0].abs() < 1e-5 && (r90[1] - 1.0).abs() < 1e-5,
            "rot90 = {r90:?}"
        );
    }

    #[test]
    fn forward_shape_and_finite() {
        let device = Default::default();
        let cfg = small(64);
        let block = Mamba3Block::<B>::new(&cfg, &device);
        let u = Tensor::<B, 3>::random(
            [2, 130, cfg.d_model],
            burn::tensor::Distribution::Default,
            &device,
        );
        let out = block.forward(u);
        assert_eq!(out.dims(), [2, 130, cfg.d_model]);
        assert!(!out.contains_nan().into_scalar(), "forward produced NaN");
    }

    /// Parallel chunked scan == recurrent step, crossing a chunk boundary. Run for
    /// both `ngroups = 1` (shared B/C) and `ngroups = 2` (GQA-style groups).
    fn check_scan_matches_step(cfg: &ModelConfig) {
        let device = Default::default();
        let block = Mamba3Block::<B>::new(cfg, &device);
        let dm = cfg.d_model;
        let t = 80;

        let u = Tensor::<B, 3>::random([1, t, dm], burn::tensor::Distribution::Default, &device);
        let y_par = block.forward(u.clone());

        let mut state = block.init_state(1, &device);
        let mut ys = Vec::with_capacity(t);
        for i in 0..t {
            let (yi, s) = block.step(u.clone().slice([0..1, i..i + 1, 0..dm]), state);
            state = s;
            ys.push(yi);
        }
        let y_rec = Tensor::cat(ys, 1);

        let max_diff = (y_par - y_rec)
            .abs()
            .max()
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(
            max_diff < 1e-3,
            "scan vs step diverged (ngroups={}): max abs diff {max_diff}",
            cfg.ngroups
        );
    }

    #[test]
    fn scan_matches_step() {
        check_scan_matches_step(&small(64));
    }

    /// `forward_with_state` must return (a) the same output as `forward` and (b) a carry
    /// state identical to having called `step` `T` times — this is the correctness
    /// guarantee for parallel prompt prefill. Checks all four state fields, crossing a
    /// chunk boundary.
    #[test]
    fn forward_with_state_matches_step_carry() {
        let device = Default::default();
        let cfg = small(64);
        let block = Mamba3Block::<B>::new(&cfg, &device);
        let dm = cfg.d_model;
        let t = 80;
        let u = Tensor::<B, 3>::random([1, t, dm], burn::tensor::Distribution::Default, &device);

        let (y_par, st_par) = block.forward_with_state(u.clone());
        let y_ref = block.forward(u.clone());

        // Output parity with the plain parallel forward.
        let out_diff = (y_par - y_ref)
            .abs()
            .max()
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(
            out_diff < 1e-4,
            "forward_with_state output drifted: {out_diff}"
        );

        // Carry-state parity with the serial recurrence.
        let mut state = block.init_state(1, &device);
        for i in 0..t {
            let (_, s) = block.step(u.clone().slice([0..1, i..i + 1, 0..dm]), state);
            state = s;
        }
        let field_diff = |a: Tensor<B, 1>, b: Tensor<B, 1>| {
            (a - b).abs().max().into_data().to_vec::<f32>().unwrap()[0]
        };
        let n = cfg.d_state;
        let di = cfg.d_inner();
        let nh = di / cfg.headdim;
        let na = n / 4;
        let h = field_diff(
            st_par.h.clone().reshape([di * n]),
            state.h.clone().reshape([di * n]),
        );
        let phi = field_diff(
            st_par.phi.clone().reshape([na]),
            state.phi.clone().reshape([na]),
        );
        let xp = field_diff(
            st_par.x_prev.clone().reshape([di]),
            state.x_prev.clone().reshape([di]),
        );
        let bp = field_diff(
            st_par.b_prev.clone().reshape([nh * n]),
            state.b_prev.clone().reshape([nh * n]),
        );
        assert!(h < 1e-3, "carry h drifted: {h}");
        assert!(phi < 1e-3, "carry phi drifted: {phi}");
        assert!(xp < 1e-3, "carry x_prev drifted: {xp}");
        assert!(bp < 1e-3, "carry b_prev drifted: {bp}");
    }

    #[test]
    fn scan_matches_step_grouped() {
        check_scan_matches_step(&small(64).with_ngroups(2));
    }

    /// Characterize the chunk-size precision cliff. The SSD factorization is exact at
    /// any chunk length, so the only thing that changes across chunk sizes is float
    /// accumulation in the `[L,L]` score/decay matmuls. `c=1` is the per-token-exact
    /// reference; we sweep larger chunks and report the max abs drift. ndarray runs in
    /// f32, so this is the f32 accumulation floor — small at every size, which is itself
    /// the finding: the documented "≤48" Metal cliff is a device/codegen issue, not an
    /// inherent property of the math or of f32. Run with `--nocapture` to see the table.
    #[test]
    fn chunk_precision_sweep() {
        let device = Default::default();
        let cfg = small(64);
        let block = Mamba3Block::<B>::new(&cfg, &device);
        let t = 160;
        let u = Tensor::<B, 3>::random(
            [1, t, cfg.d_model],
            burn::tensor::Distribution::Default,
            &device,
        );
        let co = block.project(u);

        let reference = block.scan_chunked(&co, 1);
        let maxdiff = |y: Tensor<B, 3>| {
            (y - reference.clone())
                .abs()
                .max()
                .into_data()
                .to_vec::<f32>()
                .unwrap()[0]
        };
        for c in [2usize, 4, 8, 16, 32, 48, 64, 96, 128] {
            let d = maxdiff(block.scan_chunked(&co, c));
            println!("chunk {c:>3}: max abs diff vs c=1  {d:.3e}");
            if c <= 48 {
                assert!(
                    d < 2e-3,
                    "chunk {c} drifts {d} from the c=1 reference (>2e-3)"
                );
            }
        }
    }

    #[test]
    fn coefficient_shapes() {
        let device = Default::default();
        let cfg = small(64).with_ngroups(2);
        let block = Mamba3Block::<B>::new(&cfg, &device);

        let (bsz, t) = (2, 5);
        let u = Tensor::<B, 3>::zeros([bsz, t, cfg.d_model], &device);
        let co = block.project(u);

        let di = cfg.d_inner();
        let n = cfg.d_state;
        let nh = di / cfg.headdim;
        assert_eq!(co.x.dims(), [bsz, t, di]);
        assert_eq!(co.la.dims(), [bsz, t, nh]);
        assert_eq!(co.beta.dims(), [bsz, t, di]);
        assert_eq!(co.b.dims(), [bsz, t, cfg.ngroups, n]);
        assert_eq!(co.c.dims(), [bsz, t, cfg.ngroups, n]);
        assert_eq!(co.theta.dims(), [bsz, t, n / 4]);
    }
}
