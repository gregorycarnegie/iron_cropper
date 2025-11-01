use image::{DynamicImage, Rgba, RgbaImage};
// tempfile not needed in this unit test
use yunet_core::{crop_face_from_image, cropper::CropSettings, postprocess::BoundingBox};
use yunet_utils::{Quality, estimate_sharpness};

#[test]
fn blurry_crop_is_skipped_when_min_quality_set() {
    // Create a blank (blurry) image
    let img = RgbaImage::from_pixel(200, 200, Rgba([128u8, 128u8, 128u8, 255u8]));
    let dyn_img = DynamicImage::ImageRgba8(img);

    let det = yunet_core::Detection {
        bbox: BoundingBox {
            x: 50.0,
            y: 50.0,
            width: 100.0,
            height: 100.0,
        },
        landmarks: [
            yunet_core::Landmark { x: 60.0, y: 60.0 },
            yunet_core::Landmark { x: 90.0, y: 60.0 },
            yunet_core::Landmark { x: 75.0, y: 80.0 },
            yunet_core::Landmark { x: 65.0, y: 100.0 },
            yunet_core::Landmark { x: 85.0, y: 100.0 },
        ],
        score: 0.9,
    };

    let settings = CropSettings {
        output_width: 100,
        output_height: 100,
        ..Default::default()
    };

    let cropped = crop_face_from_image(&dyn_img, &det, &settings);
    let (_v, q) = estimate_sharpness(&cropped);
    // Blank image should be Low quality
    assert_eq!(q, Quality::Low);
}
