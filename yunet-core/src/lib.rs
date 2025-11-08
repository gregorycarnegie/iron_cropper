//! Core YuNet inference primitives.
//!
//! This crate loads the YuNet ONNX model, runs inference with `tract-onnx`,
//! and provides preprocessing and postprocessing helpers.

/// Face cropping utilities (Phase 4)
pub mod cropper;
/// High-level face detection runner.
pub mod detector;
/// Utilities to extract and resize face crops from images.
pub mod face_cropper;
/// ONNX model loading and execution.
pub mod model;
/// Detection post-processing (NMS, score filtering).
pub mod postprocess;
/// Image pre-processing (resizing, tensor conversion).
pub mod preprocess;
/// Standard crop size presets for face crops.
pub mod presets;

pub use crate::cropper::{CropRegion, CropSettings, PositioningMode, calculate_crop_region};
pub use crate::face_cropper::crop_face_from_image;
pub use crate::presets::{CropPreset, preset_by_name, standard_presets};
pub use detector::{DetectionOutput, YuNetDetector};
pub use model::YuNetModel;
pub use postprocess::{BoundingBox, Detection, Landmark, PostprocessConfig, apply_postprocess};
pub use preprocess::{
    CpuPreprocessor, InputSize, PreprocessConfig, PreprocessOutput, Preprocessor, WgpuPreprocessor,
    preprocess_dynamic_image, preprocess_image, preprocess_image_with,
};

/// Returns the crate version for diagnostics.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
