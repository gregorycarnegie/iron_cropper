//! High-level detector abstraction combining preprocessing, model inference, and postprocessing.
//!
//! This module exposes [`YuNetDetector`], the primary interface used by both the CLI and GUI
//! front-ends to run YuNet against images.

use std::{path::Path, sync::Arc};

use anyhow::{Context, Result};
use image::DynamicImage;

use crate::model::YuNetModel;
use crate::postprocess::{Detection, PostprocessConfig, apply_postprocess};
use crate::preprocess::{CpuPreprocessor, PreprocessConfig, PreprocessOutput, Preprocessor};
use yunet_utils::load_image;
use yunet_utils::timing_guard;

/// Result of running YuNet on an image.
///
/// Contains the final list of detections along with metadata to map them
/// back to the original image's coordinate space.
#[derive(Debug)]
pub struct DetectionOutput {
    /// A list of detected faces.
    pub detections: Vec<Detection>,
    /// The horizontal scale factor to convert detection coordinates to the original image space.
    pub scale_x: f32,
    /// The vertical scale factor to convert detection coordinates to the original image space.
    pub scale_y: f32,
    /// The original dimensions of the input image.
    pub original_size: (u32, u32),
}

/// Convenience wrapper that couples the YuNet model with preprocessing and postprocessing settings.
///
/// This is the main entry point for running face detection.
#[derive(Debug)]
pub struct YuNetDetector {
    model: YuNetModel,
    preprocess: PreprocessConfig,
    postprocess: PostprocessConfig,
    preprocessor: Arc<dyn Preprocessor>,
}

impl YuNetDetector {
    /// Construct a detector from a model path and configuration.
    ///
    /// # Arguments
    ///
    /// * `model_path` - The path to the ONNX model file.
    /// * `preprocess` - The configuration for image preprocessing.
    /// * `postprocess` - The configuration for detection postprocessing.
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        preprocess: PreprocessConfig,
        postprocess: PostprocessConfig,
    ) -> Result<Self> {
        let cpu = Arc::new(CpuPreprocessor);
        Self::with_preprocessor(model_path, preprocess, postprocess, cpu)
    }

    /// Construct a detector with a custom preprocessor implementation.
    pub fn with_preprocessor<P: AsRef<Path>>(
        model_path: P,
        preprocess: PreprocessConfig,
        postprocess: PostprocessConfig,
        preprocessor: Arc<dyn Preprocessor>,
    ) -> Result<Self> {
        let model = YuNetModel::load(model_path, preprocess.input_size)?;
        Ok(Self {
            model,
            preprocess,
            postprocess,
            preprocessor,
        })
    }

    /// Run detection on an image file path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the image file.
    pub fn detect_path<P: AsRef<Path>>(&self, path: P) -> Result<DetectionOutput> {
        let _guard = timing_guard("yunet_core::detect_path", log::Level::Debug);
        let path_ref = path.as_ref();
        anyhow::ensure!(
            path_ref.exists(),
            "input image does not exist: {}",
            path_ref.display()
        );
        let image = load_image(path_ref)
            .with_context(|| format!("failed to load image from {}", path_ref.display()))?;
        let prep = self.preprocessor.preprocess(&image, &self.preprocess)?;
        self.run_preprocessed(prep)
    }

    /// Run detection on an in-memory dynamic image.
    ///
    /// # Arguments
    ///
    /// * `image` - The dynamic image to process.
    pub fn detect_image(&self, image: &DynamicImage) -> Result<DetectionOutput> {
        let _guard = timing_guard("yunet_core::detect_image", log::Level::Debug);
        let prep = self.preprocessor.preprocess(image, &self.preprocess)?;
        self.run_preprocessed(prep)
    }

    /// Access the underlying postprocess configuration.
    pub fn postprocess_config(&self) -> &PostprocessConfig {
        &self.postprocess
    }

    /// Access the preprocessing configuration.
    pub fn preprocess_config(&self) -> &PreprocessConfig {
        &self.preprocess
    }

    /// Run the model on a preprocessed tensor and return the final detections.
    fn run_preprocessed(&self, prep: PreprocessOutput) -> Result<DetectionOutput> {
        let _guard = timing_guard("yunet_core::run_preprocessed", log::Level::Trace);

        let PreprocessOutput {
            tensor,
            scale_x,
            scale_y,
            original_size,
        } = prep;

        let raw = {
            let _guard = timing_guard("yunet_core::onnx_inference", log::Level::Debug);
            self.model.run(tensor)?
        };

        let detections = {
            let _guard = timing_guard("yunet_core::postprocess", log::Level::Debug);
            apply_postprocess(&raw, scale_x, scale_y, &self.postprocess)?
        };

        Ok(DetectionOutput {
            detections,
            scale_x,
            scale_y,
            original_size,
        })
    }
}
