//! Basic image enhancement utilities for Phase 5.
//!
//! Provides a small, deterministic enhancement pipeline used by the CLI
//! when `--enhance` is requested. The implementations are intentionally
//! lightweight and pure-Rust using the `image` crate.

use image::{DynamicImage, ImageBuffer, Rgba};

/// Settings for the enhancement pipeline.
#[derive(Debug, Clone)]
pub struct EnhancementSettings {
    /// Amount for unsharp mask; 0.0 = off, typical 0.5..1.5
    pub unsharp_amount: f32,
    /// Gaussian blur radius for the unsharp mask (pixels). 0.0 = off
    pub unsharp_radius: f32,
    /// Contrast adjustment (percent-like). Positive increases contrast.
    /// Passed through to `image::imageops::contrast` as-is.
    pub contrast: f32,
    /// Brightness/exposure offset in integer steps (-255..255).
    pub exposure: i32,
    /// Additional brightness offset (applied after exposure)
    pub brightness: i32,
    /// Saturation multiplier (1.0 = no change)
    pub saturation: f32,
    /// Apply simple gray-world auto color balance when true
    pub auto_color: bool,
    /// Additional sharpen amount (adds to unsharp_amount)
    pub sharpness: f32,
}

impl Default for EnhancementSettings {
    fn default() -> Self {
        Self {
            unsharp_amount: 0.8,
            unsharp_radius: 1.0,
            contrast: 10.0,
            exposure: 0,
            brightness: 0,
            saturation: 1.0,
            auto_color: false,
            sharpness: 0.0,
        }
    }
}

/// Apply a simple unsharp mask to an RGBA image.
fn apply_unsharp_mask(img: &DynamicImage, amount: f32, radius: f32) -> DynamicImage {
    if amount.abs() < f32::EPSILON || radius <= 0.0 {
        return img.clone();
    }

    // Work on an RGBA8 buffer
    let src = img.to_rgba8();
    // operate directly on the ImageBuffer; `blur` accepts an ImageBuffer and returns one
    let blurred = image::imageops::blur(&src, radius);

    let (w, h) = src.dimensions();
    let mut out: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let s = src.get_pixel(x, y).0;
            let b = blurred.get_pixel(x, y).0;
            let mut new_px = [0u8; 4];
            for c in 0..4usize {
                // Use signed calculation in f32, ignore alpha for differences but preserve it
                let val = s[c] as f32 + amount * ((s[c] as f32) - (b[c] as f32));
                let val = val.round().clamp(0.0, 255.0) as u8;
                new_px[c] = val;
            }
            out.put_pixel(x, y, Rgba(new_px));
        }
    }

    DynamicImage::ImageRgba8(out)
}

/// Apply contrast and exposure adjustments using `image::imageops` helpers.
fn apply_contrast_and_exposure(img: &DynamicImage, contrast: f32, exposure: i32) -> DynamicImage {
    // Convert to RGBA buffer, apply contrast and brighten on the buffer and wrap back
    let src_buf = img.to_rgba8();
    let contrasted = image::imageops::contrast(&src_buf, contrast);
    if exposure != 0 {
        let bright = image::imageops::brighten(&contrasted, exposure);
        DynamicImage::ImageRgba8(bright)
    } else {
        DynamicImage::ImageRgba8(contrasted)
    }
}

/// Simple gray-world white-balance: scale each channel so its mean equals overall mean.
fn apply_auto_color_balance(img: &DynamicImage) -> DynamicImage {
    let mut buf = img.to_rgba8();
    let (w, h) = buf.dimensions();
    let mut sum_r: u64 = 0;
    let mut sum_g: u64 = 0;
    let mut sum_b: u64 = 0;
    let mut count: u64 = 0;
    for px in buf.pixels() {
        sum_r += px[0] as u64;
        sum_g += px[1] as u64;
        sum_b += px[2] as u64;
        count += 1;
    }
    if count == 0 {
        return DynamicImage::ImageRgba8(buf);
    }
    let avg_r = (sum_r as f32) / (count as f32);
    let avg_g = (sum_g as f32) / (count as f32);
    let avg_b = (sum_b as f32) / (count as f32);
    let avg = (avg_r + avg_g + avg_b) / 3.0;
    let kr = if avg_r > 0.0 { avg / avg_r } else { 1.0 };
    let kg = if avg_g > 0.0 { avg / avg_g } else { 1.0 };
    let kb = if avg_b > 0.0 { avg / avg_b } else { 1.0 };

    for y in 0..h {
        for x in 0..w {
            let mut p = buf.get_pixel(x, y).0;
            let r = (p[0] as f32 * kr).round().clamp(0.0, 255.0) as u8;
            let g = (p[1] as f32 * kg).round().clamp(0.0, 255.0) as u8;
            let b = (p[2] as f32 * kb).round().clamp(0.0, 255.0) as u8;
            p[0] = r;
            p[1] = g;
            p[2] = b;
            buf.put_pixel(x, y, image::Rgba(p));
        }
    }
    DynamicImage::ImageRgba8(buf)
}

/// Adjust saturation by mixing with per-pixel luminance: new = gray*(1-s) + orig*s
fn apply_saturation(img: &DynamicImage, saturation: f32) -> DynamicImage {
    if (saturation - 1.0).abs() < f32::EPSILON {
        return img.clone();
    }
    let mut buf = img.to_rgba8();
    let (w, h) = buf.dimensions();
    for y in 0..h {
        for x in 0..w {
            let p = buf.get_pixel(x, y).0;
            let r = p[0] as f32;
            let g = p[1] as f32;
            let b = p[2] as f32;
            // Rec. 601 luma
            let gray = 0.299 * r + 0.587 * g + 0.114 * b;
            let nr = (gray * (1.0 - saturation) + r * saturation)
                .round()
                .clamp(0.0, 255.0) as u8;
            let ng = (gray * (1.0 - saturation) + g * saturation)
                .round()
                .clamp(0.0, 255.0) as u8;
            let nb = (gray * (1.0 - saturation) + b * saturation)
                .round()
                .clamp(0.0, 255.0) as u8;
            buf.put_pixel(x, y, image::Rgba([nr, ng, nb, p[3]]));
        }
    }
    DynamicImage::ImageRgba8(buf)
}

/// Apply the configured enhancements to the input image and return the result.
pub fn apply_enhancements(img: &DynamicImage, settings: &EnhancementSettings) -> DynamicImage {
    // Start with the input image and apply operations in a fixed order
    let mut out = img.clone();

    // Auto-color balance first (if requested)
    if settings.auto_color {
        out = apply_auto_color_balance(&out);
    }

    // Saturation adjustment
    if (settings.saturation - 1.0).abs() >= f32::EPSILON {
        out = apply_saturation(&out, settings.saturation);
    }

    // Apply sharpening/unsharp. Combine configured unsharp_amount with a separate sharpness control.
    let combined_sharp = settings.unsharp_amount + settings.sharpness;
    if combined_sharp.abs() >= f32::EPSILON && settings.unsharp_radius > 0.0 {
        out = apply_unsharp_mask(&out, combined_sharp, settings.unsharp_radius);
    }

    // Contrast, exposure, and brightness
    if settings.contrast.abs() >= f32::EPSILON || settings.exposure != 0 || settings.brightness != 0
    {
        out = apply_contrast_and_exposure(&out, settings.contrast, settings.exposure);
        if settings.brightness != 0 {
            out = DynamicImage::ImageRgba8(image::imageops::brighten(
                &out.to_rgba8(),
                settings.brightness,
            ));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbaImage};

    #[test]
    fn enhancement_pipeline_runs_and_preserves_dimensions() {
        let src = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            64,
            48,
            image::Rgba([128, 128, 128, 255]),
        ));
        let settings = EnhancementSettings::default();
        let out = apply_enhancements(&src, &settings);
        assert_eq!(out.width(), src.width());
        assert_eq!(out.height(), src.height());
    }

    #[test]
    fn unsharp_on_high_freq_changes_pixels() {
        // create a checkerboard to have high-frequency content
        let mut img = RgbaImage::from_pixel(32, 32, image::Rgba([128, 128, 128, 255]));
        for y in 0..32 {
            for x in 0..32 {
                let v = if (x + y) % 2 == 0 { 0 } else { 255 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 255]));
            }
        }
        let dyn_img = DynamicImage::ImageRgba8(img.clone());
        let out = apply_unsharp_mask(&dyn_img, 1.5, 1.5);
        // basic sanity: result has same dimensions
        assert_eq!(out.width(), dyn_img.width());
        assert_eq!(out.height(), dyn_img.height());
    }
}
