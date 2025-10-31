use std::path::Path;

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use tract_onnx::prelude::Tensor;
use yunet_utils::{
    compute_resize_scales, config::InputDimensions, load_image, resize_image, rgb_to_bgr_chw,
};

/// Desired input resolution for YuNet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputSize {
    /// The width of the input tensor.
    pub width: u32,
    /// The height of the input tensor.
    pub height: u32,
}

impl InputSize {
    /// Creates a new `InputSize`.
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

impl Default for InputSize {
    fn default() -> Self {
        Self {
            width: 640,
            height: 640,
        }
    }
}

/// Configuration for preprocessing an image before inference.
#[derive(Debug, Clone, Default)]
pub struct PreprocessConfig {
    /// The target input size for the model.
    pub input_size: InputSize,
}

/// Output of preprocessing: tensor plus metadata for rescaling detections.
#[derive(Debug)]
pub struct PreprocessOutput {
    /// The preprocessed image tensor, ready for inference.
    pub tensor: Tensor,
    /// The horizontal scale factor to convert detection coordinates to the original image space.
    pub scale_x: f32,
    /// The vertical scale factor to convert detection coordinates to the original image space.
    pub scale_y: f32,
    /// The original dimensions of the input image.
    pub original_size: (u32, u32),
}

/// Preprocess an image file into a YuNet-ready tensor in `[1, 3, H, W]` (CHW) BGR format matching OpenCV's `blobFromImage`.
///
/// # Arguments
///
/// * `path` - The path to the image file.
/// * `config` - The configuration for preprocessing.
pub fn preprocess_image<P: AsRef<Path>>(
    path: P,
    config: &PreprocessConfig,
) -> Result<PreprocessOutput> {
    let path_ref = path.as_ref();
    anyhow::ensure!(
        path_ref.exists(),
        "input image does not exist: {}",
        path_ref.display()
    );

    let image = load_image(path_ref)
        .with_context(|| format!("failed to load image from {}", path_ref.display()))?;
    preprocess_dynamic_image(&image, config)
}

/// Preprocess an in-memory image (useful for tests).
///
/// # Arguments
///
/// * `image` - The dynamic image to process.
/// * `config` - The configuration for preprocessing.
pub fn preprocess_dynamic_image(
    image: &DynamicImage,
    config: &PreprocessConfig,
) -> Result<PreprocessOutput> {
    let input_w = config.input_size.width;
    let input_h = config.input_size.height;
    anyhow::ensure!(
        input_w > 0 && input_h > 0,
        "input dimensions must be greater than zero"
    );

    let (orig_w, orig_h) = image.dimensions();
    let resized = resize_image(image, input_w, input_h, FilterType::Triangle);
    let chw = rgb_to_bgr_chw(&resized);

    let shape = [1usize, 3, input_h as usize, input_w as usize];
    #[allow(deprecated)]
    let data = chw.into_raw_vec();
    let tensor = Tensor::from_shape(&shape, &data)
        .map_err(|e| anyhow::anyhow!("failed to build tensor: {e}"))?;

    let (scale_x, scale_y) = compute_resize_scales((orig_w, orig_h), (input_w, input_h))?;

    Ok(PreprocessOutput {
        tensor,
        scale_x,
        scale_y,
        original_size: (orig_w, orig_h),
    })
}

impl From<InputDimensions> for InputSize {
    fn from(dimensions: InputDimensions) -> Self {
        InputSize::new(dimensions.width, dimensions.height)
    }
}

impl From<&InputDimensions> for InputSize {
    fn from(dimensions: &InputDimensions) -> Self {
        (*dimensions).into()
    }
}

impl From<InputDimensions> for PreprocessConfig {
    fn from(dimensions: InputDimensions) -> Self {
        PreprocessConfig {
            input_size: dimensions.into(),
        }
    }
}

impl From<&InputDimensions> for PreprocessConfig {
    fn from(dimensions: &InputDimensions) -> Self {
        (*dimensions).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};
    use yunet_utils::config::InputDimensions;

    #[test]
    fn preprocess_generates_bgr_tensor() {
        let mut img = ImageBuffer::<Rgb<u8>, _>::new(4, 4);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            let value = ((x + y) * 32) as u8;
            *pixel = Rgb([value, value / 2, 255]);
        }

        let dynamic = DynamicImage::ImageRgb8(img);
        let config = PreprocessConfig {
            input_size: InputSize::new(2, 2),
        };

        let output =
            preprocess_dynamic_image(&dynamic, &config).expect("preprocess should succeed");

        assert_eq!(output.original_size, (4, 4));
        assert_eq!(output.scale_x, 2.0);
        assert_eq!(output.scale_y, 2.0);
        assert_eq!(output.tensor.shape(), &[1, 3, 2, 2]);

        let data = output.tensor.as_slice::<f32>().unwrap();
        assert!(data.iter().all(|v| *v >= 0.0 && *v <= 255.0));
    }

    #[test]
    fn converts_dimensions_into_configs() {
        let dims = InputDimensions {
            width: 320,
            height: 240,
        };

        let size: InputSize = dims.into();
        assert_eq!(size.width, 320);
        assert_eq!(size.height, 240);

        let config: PreprocessConfig = dims.into();
        assert_eq!(config.input_size.width, 320);
        assert_eq!(config.input_size.height, 240);
    }
}
