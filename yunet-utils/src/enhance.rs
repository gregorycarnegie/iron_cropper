//! Basic image enhancement utilities for Phase 5.
//!
//! Provides a small, deterministic enhancement pipeline used by the CLI
//! when `--enhance` is requested. The implementations are intentionally
//! lightweight and pure-Rust using the `image` crate.

use image::{DynamicImage, ImageBuffer, Rgba};
use wide::f32x4;

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
    /// Skin smoothing strength (0.0 = off, 1.0 = maximum).
    pub skin_smooth_amount: f32,
    /// Spatial sigma for bilateral filter (controls spatial extent).
    pub skin_smooth_sigma_space: f32,
    /// Color sigma for bilateral filter (controls color similarity threshold).
    pub skin_smooth_sigma_color: f32,
    /// Enable automated red-eye removal.
    pub red_eye_removal: bool,
    /// Red-eye detection threshold (higher = more selective).
    pub red_eye_threshold: f32,
    /// Enable background blur (portrait mode effect).
    pub background_blur: bool,
    /// Background blur strength (radius in pixels).
    pub background_blur_radius: f32,
    /// Background blur mask size (0.0-1.0, larger = more area kept sharp).
    pub background_blur_mask_size: f32,
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
            skin_smooth_amount: 0.0,
            skin_smooth_sigma_space: 3.0,
            skin_smooth_sigma_color: 25.0,
            red_eye_removal: false,
            red_eye_threshold: 1.5,
            background_blur: false,
            background_blur_radius: 15.0,
            background_blur_mask_size: 0.6,
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

#[inline]
fn build_lut(mut mapper: impl FnMut(u8) -> u8) -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (value, slot) in lut.iter_mut().enumerate() {
        *slot = mapper(value as u8);
    }
    lut
}

#[inline]
fn apply_lut_rgb(img: &DynamicImage, lut: &[u8; 256]) -> DynamicImage {
    let mut buf = img.to_rgba8();
    for pixel in buf.as_mut().chunks_exact_mut(4) {
        pixel[0] = lut[pixel[0] as usize];
        pixel[1] = lut[pixel[1] as usize];
        pixel[2] = lut[pixel[2] as usize];
    }
    DynamicImage::ImageRgba8(buf)
}

#[inline]
fn clamp_vec_to_u8(vec: f32x4) -> [u8; 4] {
    let rounded: [f32; 4] = vec.round().into();
    let mut out = [0u8; 4];
    for (idx, value) in rounded.iter().enumerate() {
        out[idx] = value.clamp(0.0, 255.0) as u8;
    }
    out
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
    let lut = build_lut(|value| {
        let boosted = (value as f32 * factor).round().clamp(0.0, 255.0);
        boosted as u8
    });
    apply_lut_rgb(img, &lut)
}

fn apply_brightness(img: &DynamicImage, offset: i32) -> DynamicImage {
    if offset == 0 {
        return img.clone();
    }
    let lut = build_lut(|value| {
        let value = value as i32 + offset;
        value.clamp(0, 255) as u8
    });
    apply_lut_rgb(img, &lut)
}

fn apply_contrast(img: &DynamicImage, multiplier: f32) -> DynamicImage {
    if (multiplier - 1.0).abs() < EPSILON {
        return img.clone();
    }
    let multiplier = multiplier.clamp(0.5, 2.0);
    let lut = build_lut(|value| {
        let normalized = value as f32 / 255.0;
        let contrasted = ((normalized - 0.5) * multiplier + 0.5).clamp(0.0, 1.0) * 255.0;
        contrasted.round() as u8
    });
    apply_lut_rgb(img, &lut)
}

/// Adjust saturation by mixing with per-pixel luminance: new = gray*(1-s) + orig*s.
fn apply_saturation(img: &DynamicImage, saturation: f32) -> DynamicImage {
    if (saturation - 1.0).abs() < EPSILON {
        return img.clone();
    }
    let multiplier = saturation.clamp(0.0, 2.5);
    let mut buf = img.to_rgba8();
    let data = buf.as_mut();
    let vec_inv = f32x4::splat(1.0 - multiplier);
    let vec_mul = f32x4::splat(multiplier);
    let coeff_r = f32x4::splat(0.299);
    let coeff_g = f32x4::splat(0.587);
    let coeff_b = f32x4::splat(0.114);
    let mut idx = 0;

    while idx + 16 <= data.len() {
        let mut r = [0.0f32; 4];
        let mut g = [0.0f32; 4];
        let mut b = [0.0f32; 4];
        let mut a = [0u8; 4];
        for lane in 0..4 {
            let base = idx + lane * 4;
            r[lane] = data[base] as f32;
            g[lane] = data[base + 1] as f32;
            b[lane] = data[base + 2] as f32;
            a[lane] = data[base + 3];
        }

        let rv = f32x4::from(r);
        let gv = f32x4::from(g);
        let bv = f32x4::from(b);
        let gray = rv * coeff_r + gv * coeff_g + bv * coeff_b;

        let new_r = clamp_vec_to_u8(gray * vec_inv + rv * vec_mul);
        let new_g = clamp_vec_to_u8(gray * vec_inv + gv * vec_mul);
        let new_b = clamp_vec_to_u8(gray * vec_inv + bv * vec_mul);

        for lane in 0..4 {
            let base = idx + lane * 4;
            data[base] = new_r[lane];
            data[base + 1] = new_g[lane];
            data[base + 2] = new_b[lane];
            data[base + 3] = a[lane];
        }

        idx += 16;
    }

    for pixel in data[idx..].chunks_exact_mut(4) {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;
        let gray = 0.299 * r + 0.587 * g + 0.114 * b;
        pixel[0] = (gray * (1.0 - multiplier) + r * multiplier)
            .round()
            .clamp(0.0, 255.0) as u8;
        pixel[1] = (gray * (1.0 - multiplier) + g * multiplier)
            .round()
            .clamp(0.0, 255.0) as u8;
        pixel[2] = (gray * (1.0 - multiplier) + b * multiplier)
            .round()
            .clamp(0.0, 255.0) as u8;
    }

    DynamicImage::ImageRgba8(buf)
}

/// Apply bilateral filter for skin smoothing (edge-preserving blur).
///
/// The bilateral filter smooths flat regions while preserving edges by
/// weighting pixels based on both spatial distance and color similarity.
fn apply_skin_smoothing(
    img: &DynamicImage,
    amount: f32,
    sigma_space: f32,
    sigma_color: f32,
) -> DynamicImage {
    if amount <= 0.0 {
        return img.clone();
    }

    let amount = amount.clamp(0.0, 1.0);
    let src = img.to_rgba8();
    let (w, h) = src.dimensions();
    let mut out = src.clone();

    // Kernel radius based on spatial sigma (typically 2-3 sigmas)
    let radius = (sigma_space * 2.0).ceil() as i32;

    // Precompute spatial weights (Gaussian based on distance)
    let mut spatial_weights =
        vec![vec![0.0f32; (2 * radius + 1) as usize]; (2 * radius + 1) as usize];
    let spatial_coeff = -0.5 / (sigma_space * sigma_space);
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let dist_sq = (dx * dx + dy * dy) as f32;
            spatial_weights[(dy + radius) as usize][(dx + radius) as usize] =
                (spatial_coeff * dist_sq).exp();
        }
    }

    // Color similarity coefficient
    let color_coeff = -0.5 / (sigma_color * sigma_color);

    for y in 0..h {
        for x in 0..w {
            let center = src.get_pixel(x, y).0;
            let mut sum_r = 0.0f32;
            let mut sum_g = 0.0f32;
            let mut sum_b = 0.0f32;
            let mut sum_weight = 0.0f32;

            // Iterate over neighborhood
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as u32;
                    let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as u32;
                    let neighbor = src.get_pixel(nx, ny).0;

                    // Color distance (Euclidean in RGB space)
                    let dr = center[0] as f32 - neighbor[0] as f32;
                    let dg = center[1] as f32 - neighbor[1] as f32;
                    let db = center[2] as f32 - neighbor[2] as f32;
                    let color_dist_sq = dr * dr + dg * dg + db * db;

                    // Combine spatial and color weights
                    let spatial_w = spatial_weights[(dy + radius) as usize][(dx + radius) as usize];
                    let color_w = (color_coeff * color_dist_sq).exp();
                    let weight = spatial_w * color_w;

                    sum_r += neighbor[0] as f32 * weight;
                    sum_g += neighbor[1] as f32 * weight;
                    sum_b += neighbor[2] as f32 * weight;
                    sum_weight += weight;
                }
            }

            if sum_weight > 0.0 {
                let filtered_r = (sum_r / sum_weight).round().clamp(0.0, 255.0) as u8;
                let filtered_g = (sum_g / sum_weight).round().clamp(0.0, 255.0) as u8;
                let filtered_b = (sum_b / sum_weight).round().clamp(0.0, 255.0) as u8;

                // Blend with original based on amount
                let final_r = ((1.0 - amount) * center[0] as f32 + amount * filtered_r as f32)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                let final_g = ((1.0 - amount) * center[1] as f32 + amount * filtered_g as f32)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                let final_b = ((1.0 - amount) * center[2] as f32 + amount * filtered_b as f32)
                    .round()
                    .clamp(0.0, 255.0) as u8;

                out.put_pixel(x, y, image::Rgba([final_r, final_g, final_b, center[3]]));
            }
        }
    }

    DynamicImage::ImageRgba8(out)
}

/// Apply automated red-eye reduction.
///
/// Detects and desaturates pixels where the red channel is significantly
/// higher than the green and blue channels, which is characteristic of red-eye.
fn apply_red_eye_removal(img: &DynamicImage, threshold: f32) -> DynamicImage {
    let src = img.to_rgba8();
    let (w, h) = src.dimensions();
    let mut out = src.clone();

    for y in 0..h {
        for x in 0..w {
            let px = src.get_pixel(x, y).0;
            let r = px[0] as f32;
            let g = px[1] as f32;
            let b = px[2] as f32;

            // Calculate red dominance: red / average(green, blue)
            let avg_gb = (g + b) / 2.0 + EPSILON;
            let red_ratio = r / avg_gb;

            // If red is significantly dominant (typical red-eye has ratio > 1.5)
            if red_ratio > threshold && r > 80.0 {
                // Desaturate by replacing red with the average of green and blue
                let corrected_r = avg_gb.round().clamp(0.0, 255.0) as u8;
                out.put_pixel(x, y, image::Rgba([corrected_r, px[1], px[2], px[3]]));
            }
        }
    }

    DynamicImage::ImageRgba8(out)
}

/// Apply background blur with centered elliptical mask (portrait mode effect).
///
/// Blurs the background while keeping a centered elliptical region (the face) sharp.
/// Creates a professional portrait look similar to smartphone portrait mode.
fn apply_background_blur(img: &DynamicImage, radius: f32, mask_size: f32) -> DynamicImage {
    let src = img.to_rgba8();
    let (w, h) = src.dimensions();

    // Blur the entire image
    let blurred = image::imageops::blur(&src, radius);

    // Create an elliptical mask centered on the image
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let mask_size = mask_size.clamp(0.3, 1.0);
    let rx = (w as f32 / 2.0) * mask_size;
    let ry = (h as f32 / 2.0) * mask_size;

    let mut out: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let dx = (x as f32 - cx) / rx;
            let dy = (y as f32 - cy) / ry;
            let dist = (dx * dx + dy * dy).sqrt();

            // Calculate blend factor (0 = sharp, 1 = blurred)
            // Use a smooth transition at the edge of the ellipse
            let blend = if dist < 0.9 {
                0.0 // Inside ellipse - keep sharp
            } else if dist > 1.1 {
                1.0 // Outside ellipse - fully blurred
            } else {
                // Smooth transition zone
                (dist - 0.9) / 0.2
            };

            let sharp_px = src.get_pixel(x, y).0;
            let blur_px = blurred.get_pixel(x, y).0;

            // Blend sharp and blurred pixels
            let mut result = [0u8; 4];
            for c in 0..4 {
                result[c] = ((1.0 - blend) * sharp_px[c] as f32 + blend * blur_px[c] as f32)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }

            out.put_pixel(x, y, Rgba(result));
        }
    }

    DynamicImage::ImageRgba8(out)
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

    if settings.red_eye_removal {
        out = apply_red_eye_removal(&out, settings.red_eye_threshold);
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

    if settings.skin_smooth_amount > 0.0 {
        out = apply_skin_smoothing(
            &out,
            settings.skin_smooth_amount,
            settings.skin_smooth_sigma_space,
            settings.skin_smooth_sigma_color,
        );
    }

    let combined_sharp = (settings.unsharp_amount + settings.sharpness).clamp(0.0, 2.0);
    if combined_sharp > 0.0 && settings.unsharp_radius > 0.0 {
        out = apply_unsharp_mask(&out, combined_sharp, settings.unsharp_radius);
    }

    if settings.background_blur {
        out = apply_background_blur(
            &out,
            settings.background_blur_radius,
            settings.background_blur_mask_size,
        );
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GenericImageView, RgbaImage};

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
    fn skin_smoothing_reduces_noise() {
        // Create a noisy image with alternating pixel values
        let mut img = RgbaImage::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                let val = if (x + y) % 2 == 0 { 100 } else { 110 };
                img.put_pixel(x, y, image::Rgba([val, val, val, 255]));
            }
        }
        let dyn_img = DynamicImage::ImageRgba8(img);

        // Apply skin smoothing
        let smoothed = apply_skin_smoothing(&dyn_img, 0.8, 3.0, 25.0).to_rgba8();

        // Check that neighboring pixels have become more similar (variance reduced)
        let p1 = smoothed.get_pixel(1, 1)[0];
        let p2 = smoothed.get_pixel(1, 2)[0];
        let diff = (p1 as i32 - p2 as i32).abs();

        // The difference should be less than the original 10
        assert!(diff < 10, "Smoothing should reduce pixel differences");

        // Alpha should be preserved
        assert_eq!(smoothed.get_pixel(1, 1)[3], 255);
    }

    #[test]
    fn skin_smoothing_zero_amount_unchanged() {
        let img = solid([150, 120, 100, 255]);
        let out = apply_skin_smoothing(&img, 0.0, 3.0, 25.0);
        assert_eq!(
            out.to_rgba8().get_pixel(0, 0),
            img.to_rgba8().get_pixel(0, 0)
        );
    }

    #[test]
    fn red_eye_removal_reduces_red_dominance() {
        // Create an image with a red-eye pixel (high red, low green/blue)
        let mut img = RgbaImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                // Normal pixel
                img.put_pixel(x, y, image::Rgba([100, 100, 100, 255]));
            }
        }
        // Add a red-eye pixel in the center (very high red)
        img.put_pixel(1, 1, image::Rgba([200, 50, 50, 255]));
        img.put_pixel(2, 1, image::Rgba([220, 60, 55, 255]));

        let dyn_img = DynamicImage::ImageRgba8(img);
        let corrected = apply_red_eye_removal(&dyn_img, 1.5).to_rgba8();

        // Check that the red-eye pixels have been corrected
        let px1 = corrected.get_pixel(1, 1);
        let px2 = corrected.get_pixel(2, 1);

        // Red channel should be reduced (closer to green/blue average)
        assert!(px1[0] < 200, "Red channel should be reduced from 200");
        assert!(px2[0] < 220, "Red channel should be reduced from 220");

        // Normal pixels should be unchanged
        let normal_px = corrected.get_pixel(0, 0);
        assert_eq!(normal_px[0], 100);
        assert_eq!(normal_px[1], 100);
    }

    #[test]
    fn red_eye_removal_preserves_alpha() {
        let img = solid([200, 50, 50, 128]);
        let out = apply_red_eye_removal(&img, 1.5).to_rgba8();
        assert_eq!(out.get_pixel(0, 0)[3], 128);
    }

    #[test]
    fn background_blur_keeps_center_sharp() {
        // Create a simple gradient image
        let mut img = RgbaImage::new(100, 100);
        for y in 0..100 {
            for x in 0..100 {
                let val = ((x + y) / 2) as u8;
                img.put_pixel(x, y, image::Rgba([val, val, val, 255]));
            }
        }
        let dyn_img = DynamicImage::ImageRgba8(img.clone());

        // Apply background blur
        let blurred = apply_background_blur(&dyn_img, 10.0, 0.5).to_rgba8();

        // Center pixel should be nearly unchanged (inside mask)
        let center_orig = img.get_pixel(50, 50)[0];
        let center_blur = blurred.get_pixel(50, 50)[0];
        let diff = (center_orig as i32 - center_blur as i32).abs();
        assert!(diff < 3, "Center pixel should be nearly unchanged");

        // Corner pixel should be blurred (outside mask)
        // We can't directly test blur, but dimensions should be preserved
        assert_eq!(blurred.dimensions(), img.dimensions());
    }

    #[test]
    fn background_blur_preserves_dimensions() {
        let img = solid([128, 128, 128, 255]);
        let out = apply_background_blur(&img, 15.0, 0.6);
        assert_eq!(out.dimensions(), img.dimensions());
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
                skin_smooth_amount: 0.5,
                skin_smooth_sigma_space: 3.0,
                skin_smooth_sigma_color: 25.0,
                red_eye_removal: true,
                red_eye_threshold: 1.5,
                background_blur: true,
                background_blur_radius: 15.0,
                background_blur_mask_size: 0.6,
            },
        );
        assert_eq!(out.width(), img.width());
        assert_eq!(out.height(), img.height());
    }

    #[test]
    fn pipeline_auto_color_matches_direct_equalization() {
        let mut img = RgbaImage::new(8, 1);
        for x in 0..4 {
            img.put_pixel(x, 0, image::Rgba([32, 64, 128, 255]));
        }
        for x in 4..8 {
            img.put_pixel(x, 0, image::Rgba([220, 180, 140, 255]));
        }
        let source = DynamicImage::ImageRgba8(img);

        let settings = EnhancementSettings {
            auto_color: true,
            unsharp_amount: 0.0,
            unsharp_radius: 0.0,
            sharpness: 0.0,
            ..EnhancementSettings::default()
        };

        let pipeline = apply_enhancements(&source, &settings).to_rgba8();
        let expected = apply_histogram_equalization(&source).to_rgba8();

        assert_eq!(pipeline, expected, "auto_color should reuse equalization");
        assert_ne!(
            pipeline,
            source.to_rgba8(),
            "auto_color should adjust levels"
        );
    }

    #[test]
    fn pipeline_sharpness_combines_with_unsharp_amount() {
        let mut img = RgbaImage::new(5, 1);
        for x in 0..5 {
            let val = (x * 40 + 40) as u8;
            img.put_pixel(x, 0, image::Rgba([val, val, val, 255]));
        }
        let source = DynamicImage::ImageRgba8(img);

        let settings = EnhancementSettings {
            unsharp_amount: 0.0,
            sharpness: 0.6,
            unsharp_radius: 1.0,
            ..EnhancementSettings::default()
        };

        let pipeline = apply_enhancements(&source, &settings).to_rgba8();
        let expected = apply_unsharp_mask(&source, 0.6, 1.0).to_rgba8();

        assert_eq!(
            pipeline, expected,
            "sharpness setting should fold into unsharp mask amount"
        );
        assert_ne!(
            pipeline,
            source.to_rgba8(),
            "sharpening should modify pixels"
        );
    }

    #[test]
    fn pipeline_red_eye_removal_matches_direct_call() {
        let mut img = RgbaImage::new(4, 2);
        for y in 0..2 {
            for x in 0..4 {
                img.put_pixel(x, y, image::Rgba([90, 90, 90, 255]));
            }
        }
        img.put_pixel(1, 0, image::Rgba([220, 40, 40, 255]));
        img.put_pixel(2, 0, image::Rgba([210, 45, 60, 255]));
        let source = DynamicImage::ImageRgba8(img);

        let settings = EnhancementSettings {
            red_eye_removal: true,
            red_eye_threshold: 1.2,
            unsharp_amount: 0.0,
            unsharp_radius: 0.0,
            sharpness: 0.0,
            ..EnhancementSettings::default()
        };

        let pipeline = apply_enhancements(&source, &settings).to_rgba8();
        let expected = apply_red_eye_removal(&source, 1.2).to_rgba8();

        assert_eq!(pipeline, expected);
        assert!(pipeline.get_pixel(1, 0)[0] < 200);
        assert!(pipeline.get_pixel(2, 0)[0] < 210);
    }

    #[test]
    fn pipeline_background_blur_matches_direct_call() {
        let mut img = RgbaImage::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                let val = ((x + y) * 4).clamp(0, 255) as u8;
                img.put_pixel(x, y, image::Rgba([val, 255 - val, val / 2, 255]));
            }
        }
        let source = DynamicImage::ImageRgba8(img);

        let settings = EnhancementSettings {
            background_blur: true,
            background_blur_radius: 6.0,
            background_blur_mask_size: 0.5,
            unsharp_amount: 0.0,
            unsharp_radius: 0.0,
            sharpness: 0.0,
            ..EnhancementSettings::default()
        };

        let pipeline = apply_enhancements(&source, &settings).to_rgba8();
        let expected = apply_background_blur(&source, 6.0, 0.5).to_rgba8();

        assert_eq!(pipeline, expected);
        // ensure the blur keeps center mostly intact while affecting a corner
        let center_diff = (pipeline.get_pixel(16, 16)[0] as i16
            - source.to_rgba8().get_pixel(16, 16)[0] as i16)
            .abs();
        assert!(center_diff < 5, "central region should remain sharp");
        let corner_diff = (pipeline.get_pixel(0, 0)[0] as i16
            - source.to_rgba8().get_pixel(0, 0)[0] as i16)
            .abs();
        assert!(corner_diff > 0, "corner should show blur impact");
    }
}
