#![doc = include_str!("../README.md")]
// Required by Burn's wgpu/cubecl backend: its deeply nested associated types
// exceed the default recursion limit during trait resolution.
#![recursion_limit = "256"]

pub mod backend;
pub mod chat;
pub mod config;
pub mod constants;
pub mod model;
pub mod utils;
pub use backend::{Compute, Elem};

#[cfg(feature = "train")]
pub use backend::Train;
#[cfg(feature = "train")]
pub mod data;
#[cfg(feature = "train")]
pub mod train;
