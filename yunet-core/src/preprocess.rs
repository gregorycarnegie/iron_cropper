use std::path::Path;

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use tract_onnx::prelude::Tensor;
use yunet_utils::{compute_resize_scales, load_image, resize_image, rgb_to_normalized_chw};

/// Desired input resolution for YuNet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputSize {
    pub width: u32,
    pub height: u32,
}

impl InputSize {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

impl Default for InputSize {
    fn default() -> Self {
        Self {
            width: 320,
            height: 320,
        }
    }
}

/// Configuration for preprocessing an image before inference.
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    pub input_size: InputSize,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            input_size: InputSize::default(),
        }
    }
}

/// Output of preprocessing: normalized tensor plus metadata for rescaling detections.
#[derive(Debug)]
pub struct PreprocessOutput {
    pub tensor: Tensor,
    pub scale_x: f32,
    pub scale_y: f32,
    pub original_size: (u32, u32),
}

/// Preprocess an image file into a YuNet-ready tensor in `[1, 3, H, W]` (CHW) format with values in `[0, 1]`.
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
    let chw = rgb_to_normalized_chw(&resized);

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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn preprocess_generates_normalized_tensor() {
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
        assert!(data.iter().all(|v| *v >= 0.0 && *v <= 1.0));
    }
}
