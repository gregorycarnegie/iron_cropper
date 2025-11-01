//! Face extraction and resizing utilities.
//!
//! Provides a simple `crop_face_from_image` helper that ties a `Detection` to
//! a `CropSettings` and returns an owned `DynamicImage` sized to the requested output.

use crate::cropper::{CropRegion, CropSettings, calculate_crop_region};
use crate::postprocess::Detection;
use image::{DynamicImage, GenericImageView, imageops::FilterType};

/// Crop a face from `img` according to `detection` and `settings`.
///
/// The returned image is resized to `settings.output_width` x `settings.output_height`.
pub fn crop_face_from_image(
    img: &DynamicImage,
    detection: &Detection,
    settings: &CropSettings,
) -> DynamicImage {
    let (img_w, img_h) = img.dimensions();

    let region: CropRegion = calculate_crop_region(img_w, img_h, detection.bbox, settings);

    // Use an immutable crop view then convert to owned image buffer.
    let sub =
        image::imageops::crop_imm(img, region.x, region.y, region.width, region.height).to_image();

    // If output dimensions are zero, return the raw crop as DynamicImage.
    if settings.output_width == 0 || settings.output_height == 0 {
        return DynamicImage::ImageRgba8(sub);
    }

    let resized = image::imageops::resize(
        &DynamicImage::ImageRgba8(sub),
        settings.output_width,
        settings.output_height,
        FilterType::Lanczos3,
    );

    image::DynamicImage::ImageRgba8(resized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cropper::CropSettings;
    use crate::postprocess::BoundingBox;
    use image::{DynamicImage, Rgba, RgbaImage};

    #[test]
    fn crop_face_resizes_to_output() {
        // Create a simple synthetic image with a neutral color
        let mut img = RgbaImage::from_pixel(800, 600, Rgba([128u8, 128u8, 128u8, 255u8]));
        // draw a bright square where face would be (not necessary for this test but helpful)
        for y in 250..350 {
            for x in 350..450 {
                img.put_pixel(x, y, Rgba([200u8, 100u8, 100u8, 255u8]));
            }
        }

        let img_dyn = DynamicImage::ImageRgba8(img);

        let detection = Detection {
            bbox: BoundingBox {
                x: 350.0,
                y: 250.0,
                width: 100.0,
                height: 100.0,
            },
            landmarks: [
                crate::postprocess::Landmark { x: 360.0, y: 260.0 },
                crate::postprocess::Landmark { x: 390.0, y: 260.0 },
                crate::postprocess::Landmark { x: 375.0, y: 285.0 },
                crate::postprocess::Landmark { x: 365.0, y: 310.0 },
                crate::postprocess::Landmark { x: 385.0, y: 310.0 },
            ],
            score: 0.95,
        };

        let settings = CropSettings {
            output_width: 200,
            output_height: 300,
            face_height_pct: 60.0,
            positioning_mode: crate::cropper::PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
        };

        let out = crop_face_from_image(&img_dyn, &detection, &settings);
        assert_eq!(out.width(), 200);
        assert_eq!(out.height(), 300);
    }
}
