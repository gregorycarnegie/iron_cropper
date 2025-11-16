//! GPU inference building blocks for the YuNet model.
//!
//! Phase 13.3 Option C starts by reimplementing the fundamental layers (conv,
//! batch-norm, activations) as WGSL compute shaders. These building blocks
//! will power the end-to-end YuNet port in subsequent increments.

mod memory;
pub mod ops;

pub use ops::{ActivationKind, BatchNormConfig, Conv2dConfig, GpuInferenceOps};
pub mod tensor;
pub use tensor::{GpuTensor, TensorShape};
pub mod onnx;
pub use onnx::{OnnxInitializerMap, OnnxTensor};
const MAX_POOL_WGSL: &str = include_str!("pool.wgsl");
const ADD_WGSL: &str = include_str!("add.wgsl");
const UPSAMPLE2X_WGSL: &str = include_str!("resize2x.wgsl");
pub mod graph;
pub mod runtime;
pub use runtime::GpuYuNet;
