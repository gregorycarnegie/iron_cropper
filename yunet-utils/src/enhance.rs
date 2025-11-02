//! Basic image enhancement utilities for Phase 5.
//!
//! Provides a small, deterministic enhancement pipeline used by the CLI
//! when `--enhance` is requested. The implementations are intentionally
//! lightweight and pure-Rust using the `image` crate.

use image::{DynamicImage, ImageBuffer, Rgba};

const EPSILON: f32 = 1e-6;

/// Settings for the enhancement pipeline.
#[derive(Debug, Clone)]
pub struct EnhancementSettings {
    /// Apply histogram-equalization based auto color correction.
    pub auto_color: bool,
    /// Exposure adjustment expressed in stops (-2.0..=2.0).
    pub exposure_stops: f32,
    /// Additional brightness offset applied after exposure.
    pub brightness: i32,
    /// Contrast multiplier (0.5..=2.0, 1.0 = unchanged).
    pub contrast: f32,
    /// Saturation multiplier (0.0..=2.5, 1.0 = unchanged).
    pub saturation: f32,
    /// Strength of unsharp mask (0.0..=2.0).
    pub unsharp_amount: f32,
    /// Blur radius used for the unsharp mask in pixels.
    pub unsharp_radius: f32,
    /// Additional sharpening control layered on top of the base amount.
    pub sharpness: f32,
}

impl Default for EnhancementSettings {
    fn default() -> Self {
        Self {
            auto_color: false,
            exposure_stops: 0.0,
            brightness: 0,
            contrast: 1.0,
            saturation: 1.0,
            unsharp_amount: 0.6,
            unsharp_radius: 1.0,
            sharpness: 0.0,
        }
    }
}

fn identity_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (i, item) in lut.iter_mut().enumerate() {
        *item = i as u8;
    }
    lut
}

fn build_equalization_lut(hist: &[u32; 256], total: u32) -> [u8; 256] {
    if total == 0 {
        return identity_lut();
    }

    let mut cdf = [0u32; 256];
    let mut cumulative = 0u32;
    let mut cdf_min = None;
    for (idx, count) in hist.iter().enumerate() {
        cumulative += *count;
        cdf[idx] = cumulative;
        if cdf_min.is_none() && *count > 0 {
            cdf_min = Some(cumulative);
        }
    }

    let cdf_min = match cdf_min {
        Some(v) => v,
        None => return identity_lut(),
    };

    if cdf_min == total {
        return identity_lut();
    }

    let denom = (total - cdf_min) as f32;
    let mut lut = [0u8; 256];
    for i in 0..=255 {
        let cdf_val = cdf[i];
        let numerator = if cdf_val > cdf_min {
            (cdf_val - cdf_min) as f32
        } else {
            0.0
        };
        let mapped = (numerator / denom * 255.0).round().clamp(0.0, 255.0) as u8;
        lut[i] = mapped;
    }

    lut
}

fn apply_histogram_equalization(img: &DynamicImage) -> DynamicImage {
    let mut buf = img.to_rgba8();
    let (w, h) = buf.dimensions();
    if w == 0 || h == 0 {
        return DynamicImage::ImageRgba8(buf);
    }

    let mut hist_r = [0u32; 256];
    let mut hist_g = [0u32; 256];
    let mut hist_b = [0u32; 256];

    for px in buf.pixels() {
        hist_r[px[0] as usize] += 1;
        hist_g[px[1] as usize] += 1;
        hist_b[px[2] as usize] += 1;
    }
    let total = w * h;
    let lut_r = build_equalization_lut(&hist_r, total);
    let lut_g = build_equalization_lut(&hist_g, total);
    let lut_b = build_equalization_lut(&hist_b, total);

    for y in 0..h {
        for x in 0..w {
            let mut px = buf.get_pixel(x, y).0;
            px[0] = lut_r[px[0] as usize];
            px[1] = lut_g[px[1] as usize];
            px[2] = lut_b[px[2] as usize];
            buf.put_pixel(x, y, image::Rgba(px));
        }
    }

    DynamicImage::ImageRgba8(buf)
}

fn apply_exposure(img: &DynamicImage, stops: f32) -> DynamicImage {
    if stops.abs() < EPSILON {
        return img.clone();
    }
    let factor = 2f32.powf(stops.clamp(-2.0, 2.0));
    let mut buf = img.to_rgba8();
    let (w, h) = buf.dimensions();
    for y in 0..h {
        for x in 0..w {
            let mut px = buf.get_pixel(x, y).0;
            for channel in px.iter_mut().take(3) {
                let val = (*channel as f32 * factor).round().clamp(0.0, 255.0) as u8;
                *channel = val;
            }
            buf.put_pixel(x, y, image::Rgba(px));
        }
    }
    DynamicImage::ImageRgba8(buf)
}

fn apply_brightness(img: &DynamicImage, offset: i32) -> DynamicImage {
    if offset == 0 {
        return img.clone();
    }
    DynamicImage::ImageRgba8(image::imageops::brighten(&img.to_rgba8(), offset))
}

fn apply_contrast(img: &DynamicImage, multiplier: f32) -> DynamicImage {
    if (multiplier - 1.0).abs() < EPSILON {
        return img.clone();
    }
    let multiplier = multiplier.clamp(0.5, 2.0);
    let mut buf = img.to_rgba8();
    let (w, h) = buf.dimensions();
    for y in 0..h {
        for x in 0..w {
            let mut px = buf.get_pixel(x, y).0;
            for channel in px.iter_mut().take(3) {
                let normalized = *channel as f32 / 255.0;
                let contrasted = ((normalized - 0.5) * multiplier + 0.5)
                    .clamp(0.0, 1.0)
                    .mul_add(255.0, 0.0)
                    .round() as u8;
                *channel = contrasted;
            }
            buf.put_pixel(x, y, image::Rgba(px));
        }
    }
    DynamicImage::ImageRgba8(buf)
}

/// Adjust saturation by mixing with per-pixel luminance: new = gray*(1-s) + orig*s.
fn apply_saturation(img: &DynamicImage, saturation: f32) -> DynamicImage {
    if (saturation - 1.0).abs() < EPSILON {
        return img.clone();
    }
    let multiplier = saturation.clamp(0.0, 2.5);
    let mut buf = img.to_rgba8();
    let (w, h) = buf.dimensions();
    for y in 0..h {
        for x in 0..w {
            let mut p = buf.get_pixel(x, y).0;
            let r = p[0] as f32;
            let g = p[1] as f32;
            let b = p[2] as f32;
            // Rec. 601 luma
            let gray = 0.299 * r + 0.587 * g + 0.114 * b;
            let nr = (gray * (1.0 - multiplier) + r * multiplier)
                .round()
                .clamp(0.0, 255.0) as u8;
            let ng = (gray * (1.0 - multiplier) + g * multiplier)
                .round()
                .clamp(0.0, 255.0) as u8;
            let nb = (gray * (1.0 - multiplier) + b * multiplier)
                .round()
                .clamp(0.0, 255.0) as u8;
            p[0] = nr;
            p[1] = ng;
            p[2] = nb;
            buf.put_pixel(x, y, image::Rgba(p));
        }
    }
    DynamicImage::ImageRgba8(buf)
}

/// Apply a simple unsharp mask to an RGBA image.
fn apply_unsharp_mask(img: &DynamicImage, amount: f32, radius: f32) -> DynamicImage {
    if amount <= 0.0 || radius <= 0.0 {
        return img.clone();
    }

    let amount = amount.clamp(0.0, 2.0);
    let src = img.to_rgba8();
    let blurred = image::imageops::blur(&src, radius);
    let (w, h) = src.dimensions();
    let mut out: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let s = src.get_pixel(x, y).0;
            let b = blurred.get_pixel(x, y).0;
            let mut new_px = [0u8; 4];
            for c in 0..4usize {
                if c == 3 {
                    new_px[c] = s[c];
                    continue;
                }
                let val = s[c] as f32 + amount * (s[c] as f32 - b[c] as f32);
                new_px[c] = val.round().clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(x, y, Rgba(new_px));
        }
    }

    DynamicImage::ImageRgba8(out)
}

/// Apply the configured enhancements to the input image and return the result.
pub fn apply_enhancements(img: &DynamicImage, settings: &EnhancementSettings) -> DynamicImage {
    let mut out = img.clone();

    if settings.auto_color {
        out = apply_histogram_equalization(&out);
    }

    if settings.exposure_stops.abs() >= EPSILON {
        out = apply_exposure(&out, settings.exposure_stops);
    }

    if settings.brightness != 0 {
        out = apply_brightness(&out, settings.brightness);
    }

    if (settings.contrast - 1.0).abs() >= EPSILON {
        out = apply_contrast(&out, settings.contrast);
    }

    if (settings.saturation - 1.0).abs() >= EPSILON {
        out = apply_saturation(&out, settings.saturation);
    }

    let combined_sharp = (settings.unsharp_amount + settings.sharpness).clamp(0.0, 2.0);
    if combined_sharp > 0.0 && settings.unsharp_radius > 0.0 {
        out = apply_unsharp_mask(&out, combined_sharp, settings.unsharp_radius);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbaImage;

    fn solid(color: [u8; 4]) -> DynamicImage {
        DynamicImage::ImageRgba8(RgbaImage::from_pixel(4, 4, image::Rgba(color)))
    }

    #[test]
    fn histogram_equalization_stretches_levels() {
        let mut img = RgbaImage::new(4, 1);
        for x in 0..2 {
            img.put_pixel(x, 0, image::Rgba([40, 80, 120, 255]));
        }
        for x in 2..4 {
            img.put_pixel(x, 0, image::Rgba([200, 160, 100, 255]));
        }
        let out = apply_histogram_equalization(&DynamicImage::ImageRgba8(img));
        let buf = out.to_rgba8();
        let mut mins = [u8::MAX; 3];
        let mut maxs = [0u8; 3];
        for pixel in buf.pixels() {
            for c in 0..3 {
                mins[c] = mins[c].min(pixel[c]);
                maxs[c] = maxs[c].max(pixel[c]);
            }
        }
        for c in 0..3 {
            assert!(
                maxs[c] > mins[c],
                "expected channel {} max {} > min {}",
                c,
                maxs[c],
                mins[c]
            );
            assert!(
                (maxs[c] as i16 - mins[c] as i16) >= 100,
                "expected channel {} spread >=100, got {}",
                c,
                maxs[c] as i16 - mins[c] as i16
            );
        }
    }

    #[test]
    fn exposure_positive_increases_values() {
        let img = solid([64, 64, 64, 255]);
        let out = apply_exposure(&img, 1.0);
        let buf = out.to_rgba8();
        let px = buf.get_pixel(0, 0);
        assert_eq!(px[0], 128);
    }

    #[test]
    fn exposure_negative_darkens_values() {
        let img = solid([200, 200, 200, 255]);
        let out = apply_exposure(&img, -1.0);
        let buf = out.to_rgba8();
        let px = buf.get_pixel(0, 0);
        assert_eq!(px[0], 100);
    }

    #[test]
    fn brightness_offsets_channels() {
        let img = solid([100, 100, 100, 255]);
        let out = apply_brightness(&img, 20);
        let buf = out.to_rgba8();
        let px = buf.get_pixel(0, 0);
        assert_eq!(px[0], 120);
    }

    #[test]
    fn contrast_multiplier_expands_range() {
        let mut img = RgbaImage::from_pixel(4, 1, image::Rgba([128, 128, 128, 255]));
        img.put_pixel(0, 0, image::Rgba([80, 80, 80, 255]));
        img.put_pixel(3, 0, image::Rgba([180, 180, 180, 255]));
        let out = apply_contrast(&DynamicImage::ImageRgba8(img), 1.5);
        let buf = out.to_rgba8();
        assert!(buf.get_pixel(0, 0)[0] < 80);
        assert!(buf.get_pixel(3, 0)[0] > 180);
    }

    #[test]
    fn saturation_zero_grays_image() {
        let img = solid([200, 100, 50, 255]);
        let out = apply_saturation(&img, 0.0);
        let buf = out.to_rgba8();
        let px = buf.get_pixel(0, 0);
        assert_eq!(px[0], px[1]);
        assert_eq!(px[1], px[2]);
    }

    #[test]
    fn unsharp_preserves_alpha() {
        let mut img = RgbaImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                let val = if (x + y) % 2 == 0 { 0 } else { 255 };
                img.put_pixel(x, y, image::Rgba([val, val, val, 128]));
            }
        }
        let dyn_img = DynamicImage::ImageRgba8(img);
        let out = apply_unsharp_mask(&dyn_img, 1.5, 1.5).to_rgba8();
        assert_eq!(out.get_pixel(1, 1)[3], 128);
    }

    #[test]
    fn pipeline_preserves_dimensions() {
        let img = solid([128, 128, 128, 255]);
        let out = apply_enhancements(
            &img,
            &EnhancementSettings {
                auto_color: true,
                exposure_stops: 0.5,
                brightness: 10,
                contrast: 1.2,
                saturation: 1.1,
                unsharp_amount: 0.8,
                unsharp_radius: 1.2,
                sharpness: 0.1,
            },
        );
        assert_eq!(out.width(), img.width());
        assert_eq!(out.height(), img.height());
    }
}
