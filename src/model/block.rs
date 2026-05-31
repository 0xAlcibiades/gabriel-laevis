//! A single residual layer using a Mamba-3 mixer and SwiGLU MLP, both pre-normed.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig, SwiGlu, SwiGluConfig};
use burn::prelude::*;

use crate::config::ModelConfig;
use crate::model::mamba3::{Mamba3Block, Mamba3State};

#[derive(Module, Debug)]
pub struct Layer<B: Backend> {
    mixer_norm: RmsNorm<B>,
    mixer: Mamba3Block<B>,
    ff_norm: RmsNorm<B>,
    ff_gate: SwiGlu<B>, // d_model -> d_ff gated by SwiGLU
    ff_down: Linear<B>, // d_ff -> d_model
}

impl<B: Backend> Layer<B> {
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> Self {
        Self {
            mixer_norm: RmsNormConfig::new(cfg.d_model).init(device),
            mixer: Mamba3Block::new(cfg, device),
            ff_norm: RmsNormConfig::new(cfg.d_model).init(device),
            ff_gate: SwiGluConfig::new(cfg.d_model, cfg.d_ff).init(device),
            ff_down: LinearConfig::new(cfg.d_ff, cfg.d_model)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Pre-norm residual around the Mamba-3 mixer.
        let x = x
            .clone()
            .add(self.mixer.forward(self.mixer_norm.forward(x)));
        // Pre-norm residual around the SwiGLU MLP.
        let ff = self
            .ff_down
            .forward(self.ff_gate.forward(self.ff_norm.forward(x.clone())));
        x.add(ff)
    }

    pub fn init_state(&self, batch: usize, device: &B::Device) -> Mamba3State<B> {
        self.mixer.init_state(batch, device)
    }

    /// Pre-norm residual Mamba-3 step and stateless SwiGLU MLP.
    pub fn step(&self, x: Tensor<B, 3>, state: Mamba3State<B>) -> (Tensor<B, 3>, Mamba3State<B>) {
        let (mixed, state) = self.mixer.step(self.mixer_norm.forward(x.clone()), state);
        let x = x.add(mixed);
        let ff = self
            .ff_down
            .forward(self.ff_gate.forward(self.ff_norm.forward(x.clone())));
        (x.add(ff), state)
    }
}
