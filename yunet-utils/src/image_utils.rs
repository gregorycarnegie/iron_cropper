use std::path::Path;

use anyhow::{Context, Result};
use image::{DynamicImage, RgbImage, imageops::FilterType};
use ndarray::Array3;

/// Load an image from disk into memory.
///
/// # Arguments
///
/// * `path` - The path to the image file.
pub fn load_image<P: AsRef<Path>>(path: P) -> Result<DynamicImage> {
    let path_ref = path.as_ref();
    image::open(path_ref).with_context(|| format!("failed to open image {}", path_ref.display()))
}

/// Resize an image to the requested resolution using the provided filter.
///
/// # Arguments
///
/// * `image` - The image to resize.
/// * `width` - The target width.
/// * `height` - The target height.
/// * `filter` - The sampling filter to use for resizing.
pub fn resize_image(image: &DynamicImage, width: u32, height: u32, filter: FilterType) -> RgbImage {
    image.resize_exact(width, height, filter).to_rgb8()
}

/// Convert an RGB image into a BGR CHW array with values matching OpenCV's `blobFromImage`.
///
/// This function rearranges the memory layout from HWC (height, width, channels) to
/// CHW (channels, height, width) and swaps the red and blue channels.
///
/// # Arguments
///
/// * `image` - The RGB image to convert.
pub fn rgb_to_bgr_chw(image: &RgbImage) -> Array3<f32> {
    let (width, height) = image.dimensions();
    let mut array = Array3::<f32>::zeros((3, height as usize, width as usize));
    for (x, y, pixel) in image.enumerate_pixels() {
        let (xi, yi) = (x as usize, y as usize);
        array[(0, yi, xi)] = pixel[2] as f32; // Blue
        array[(1, yi, xi)] = pixel[1] as f32; // Green
        array[(2, yi, xi)] = pixel[0] as f32; // Red
    }
    array
}

/// Convert any dynamic image into a BGR CHW array by first converting to RGB.
///
/// # Arguments
///
/// * `image` - The dynamic image to convert.
pub fn dynamic_to_bgr_chw(image: &DynamicImage) -> Array3<f32> {
    rgb_to_bgr_chw(&image.to_rgb8())
}

/// Compute scale factors used to reproject detections from model space to original space.
///
/// This is necessary when the model runs on a resized version of the original image.
///
/// # Arguments
///
/// * `original` - A tuple of the original image's (width, height).
/// * `target` - A tuple of the resized image's (width, height).
pub fn compute_resize_scales(original: (u32, u32), target: (u32, u32)) -> Result<(f32, f32)> {
    let (orig_w, orig_h) = original;
    let (target_w, target_h) = target;
    anyhow::ensure!(
        target_w > 0 && target_h > 0,
        "target dimensions must be non-zero"
    );
    anyhow::ensure!(
        orig_w > 0 && orig_h > 0,
        "original dimensions must be non-zero"
    );
    Ok((
        orig_w as f32 / target_w as f32,
        orig_h as f32 / target_h as f32,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_to_bgr_chw_converts_correctly() {
        let mut image = RgbImage::new(2, 2);
        image.put_pixel(0, 0, image::Rgb([0, 128, 255]));
        image.put_pixel(1, 0, image::Rgb([255, 128, 0]));
        image.put_pixel(0, 1, image::Rgb([64, 64, 64]));
        image.put_pixel(1, 1, image::Rgb([255, 255, 255]));

        let array = rgb_to_bgr_chw(&image);
        assert_eq!(array.shape(), &[3, 2, 2]);

        assert_eq!(array[(0, 0, 0)], 255.0);
        assert_eq!(array[(2, 0, 0)], 0.0);
        assert_eq!(array[(1, 0, 1)], 128.0);
    }

    #[test]
    fn compute_resize_scales_returns_expected_values() {
        let (sx, sy) = compute_resize_scales((640, 480), (320, 240)).unwrap();
        assert_eq!(sx, 2.0);
        assert_eq!(sy, 2.0);
    }

    #[test]
    fn compute_resize_scales_rejects_zero() {
        assert!(compute_resize_scales((0, 480), (320, 240)).is_err());
        assert!(compute_resize_scales((640, 480), (0, 240)).is_err());
    }
}
