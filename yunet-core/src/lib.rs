//! Core YuNet inference primitives.
//!
//! This crate will eventually load the YuNet ONNX model, run inference with tract,
//! and expose ergonomic preprocessing and postprocessing helpers.

pub mod detector;
pub mod model;
pub mod postprocess;
pub mod preprocess;

pub use detector::{DetectionOutput, YuNetDetector};
pub use model::YuNetModel;
pub use postprocess::{BoundingBox, Detection, Landmark, PostprocessConfig, apply_postprocess};
pub use preprocess::{
    InputSize, PreprocessConfig, PreprocessOutput, preprocess_dynamic_image, preprocess_image,
};

/// Returns the crate version for diagnostics.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
