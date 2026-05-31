//! The model: Mamba-3 mixer, the residual layer, and the full LM.

pub mod block;
pub mod lm;
pub mod mamba3;

pub use block::Layer;
pub use lm::GabrielLaevis;
pub use mamba3::Mamba3Block;
