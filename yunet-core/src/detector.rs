use std::path::Path;

use anyhow::Result;
use image::DynamicImage;

use crate::model::YuNetModel;
use crate::postprocess::{Detection, PostprocessConfig, apply_postprocess};
use crate::preprocess::{
    PreprocessConfig, PreprocessOutput, preprocess_dynamic_image, preprocess_image,
};

/// Result of running YuNet on an image.
#[derive(Debug)]
pub struct DetectionOutput {
    pub detections: Vec<Detection>,
    pub scale_x: f32,
    pub scale_y: f32,
    pub original_size: (u32, u32),
}

/// Convenience wrapper that couples the YuNet model with preprocessing and postprocessing settings.
#[derive(Debug)]
pub struct YuNetDetector {
    model: YuNetModel,
    preprocess: PreprocessConfig,
    postprocess: PostprocessConfig,
}

impl YuNetDetector {
    /// Construct a detector from a model path and configuration.
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        preprocess: PreprocessConfig,
        postprocess: PostprocessConfig,
    ) -> Result<Self> {
        let model = YuNetModel::load(model_path, preprocess.input_size)?;
        Ok(Self {
            model,
            preprocess,
            postprocess,
        })
    }

    /// Run detection on an image file path.
    pub fn detect_path<P: AsRef<Path>>(&self, path: P) -> Result<DetectionOutput> {
        let prep = preprocess_image(path, &self.preprocess)?;
        self.run_preprocessed(prep)
    }

    /// Run detection on an in-memory dynamic image.
    pub fn detect_image(&self, image: &DynamicImage) -> Result<DetectionOutput> {
        let prep = preprocess_dynamic_image(image, &self.preprocess)?;
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

    fn run_preprocessed(&self, prep: PreprocessOutput) -> Result<DetectionOutput> {
        let raw = self.model.run(&prep.tensor)?;
        let detections = apply_postprocess(&raw, prep.scale_x, prep.scale_y, &self.postprocess)?;

        Ok(DetectionOutput {
            detections,
            scale_x: prep.scale_x,
            scale_y: prep.scale_y,
            original_size: prep.original_size,
        })
    }
}
