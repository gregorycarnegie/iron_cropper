//! Core YuNet inference primitives.
//!
//! This crate loads the YuNet ONNX model, runs inference with `tract-onnx`,
//! and provides preprocessing and postprocessing helpers.

/// High-level face detection runner.
pub mod detector;
/// ONNX model loading and execution.
pub mod model;
/// Detection post-processing (NMS, score filtering).
pub mod postprocess;
/// Image pre-processing (resizing, tensor conversion).
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
