use image::DynamicImage;
use log::warn;
use std::sync::Arc;
use yunet_core::FillColor;
use yunet_utils::{CropShape, WgpuEnhancer, apply_shape_mask, color::RgbaColor};

pub fn apply_mask_with_gpu(
    image: DynamicImage,
    shape: &CropShape,
    vignette_softness: f32,
    vignette_intensity: f32,
    vignette_color: RgbaColor,
    enhancer: Option<&Arc<WgpuEnhancer>>,
) -> DynamicImage {
    if let Some(enhancer) = enhancer {
        match enhancer.apply_shape_mask_gpu(
            &image,
            shape,
            vignette_softness,
            vignette_intensity,
            vignette_color,
        ) {
            Ok(Some(masked)) => {
                return masked;
            }
            Ok(None) => {}
            Err(err) => warn!("GPU shape mask failed: {err}; falling back to CPU path."),
        }
    }
    let mut rgba = image.to_rgba8();
    apply_shape_mask(
        &mut rgba,
        shape,
        vignette_softness,
        vignette_intensity,
        vignette_color,
    );
    DynamicImage::ImageRgba8(rgba)
}

/// Composites an RGBA image onto a solid fill color background.
/// This makes transparent areas show the fill color, matching how the final export will look.
pub fn composite_with_fill_color(image: &mut image::RgbaImage, fill: FillColor) {
    use image::Rgba;
    let bg_r = fill.red as f32;
    let bg_g = fill.green as f32;
    let bg_b = fill.blue as f32;
    let bg_a = fill.alpha as f32 / 255.0;

    for pixel in image.pixels_mut() {
        let src_a = pixel[3] as f32 / 255.0;
        if src_a >= 1.0 {
            // Fully opaque, no compositing needed
            continue;
        }
        // Standard alpha compositing: result = src * src_a + bg * (1 - src_a)
        let inv_a = 1.0 - src_a;

        let func = |src: u8, bg: f32| (src as f32 * src_a + bg * bg_a * inv_a) as u8;
        let r = func(pixel[0], bg_r);
        let g = func(pixel[1], bg_g);
        let b = func(pixel[2], bg_b);
        // Output alpha: src_a + bg_a * (1 - src_a), clamped to 255
        let a = ((src_a + bg_a * inv_a) * 255.0).min(255.0) as u8;
        *pixel = Rgba([r, g, b, a]);
    }
}
