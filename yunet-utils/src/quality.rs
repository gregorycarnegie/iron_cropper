//! Image quality analysis utilities.
//!
//! Implements a Laplacian-variance based blur detector and exposes a simple
//! `Quality` enum and `estimate_sharpness` helper.

use image::{DynamicImage, GrayImage};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Quality levels derived from Laplacian variance thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quality {
    Low,
    Medium,
    High,
}

impl Quality {
    /// Map a Laplacian variance score into a `Quality` bucket.
    pub fn from_variance(v: f64) -> Self {
        if v > 1000.0 {
            Quality::High
        } else if v > 300.0 {
            Quality::Medium
        } else {
            Quality::Low
        }
    }
}

impl std::str::FromStr for Quality {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "low" => Ok(Quality::Low),
            "medium" | "med" => Ok(Quality::Medium),
            "high" => Ok(Quality::High),
            other => Err(format!(
                "unknown quality '{}', expected low/medium/high",
                other
            )),
        }
    }
}

/// A small configuration object used to centralize quality filtering behavior.
#[derive(Debug, Clone)]
pub struct QualityFilter {
    /// Minimum quality required to accept an image.
    pub min_quality: Option<Quality>,
    /// If true, choose the highest available crop automatically (future use).
    pub auto_select: bool,
    /// Optional fallback quality if no crop meets `min_quality`.
    pub fallback: Option<Quality>,
    /// Skip exporting when no detection meets `Quality::High`.
    pub auto_skip_no_high: bool,
    /// Append quality suffixes to exported filenames.
    pub suffix_enabled: bool,
}

impl QualityFilter {
    /// Simple constructor.
    pub fn new(min_quality: Option<Quality>) -> Self {
        Self {
            min_quality,
            auto_select: false,
            fallback: None,
            auto_skip_no_high: false,
            suffix_enabled: false,
        }
    }

    /// Return true if the provided detection quality should be skipped.
    pub fn should_skip(&self, q: Quality) -> bool {
        if let Some(min) = self.min_quality {
            return q < min;
        }
        false
    }

    /// When `auto_skip_no_high` is enabled, return true if the best available quality
    /// does not reach `Quality::High`.
    pub fn should_skip_image(&self, best_quality: Option<Quality>) -> bool {
        if self.auto_skip_no_high {
            !matches!(best_quality, Some(Quality::High))
        } else {
            false
        }
    }

    /// Pick the index of the best quality entry from `(Quality, score)` tuples.
    /// `Quality::High` beats `Medium`, which beats `Low`. Ties fall back to the raw score.
    pub fn select_best_index(&self, qualities: &[(Quality, f64)]) -> Option<usize> {
        qualities
            .iter()
            .enumerate()
            .max_by(|(_, (qa, sa)), (_, (qb, sb))| match qa.cmp(qb) {
                std::cmp::Ordering::Equal => {
                    sa.partial_cmp(sb).unwrap_or(std::cmp::Ordering::Equal)
                }
                other => other,
            })
            .map(|(idx, _)| idx)
    }

    /// Return a filename suffix (e.g., `_highq`) when suffixing is enabled.
    pub fn suffix_for(&self, q: Quality) -> Option<&'static str> {
        if !self.suffix_enabled {
            return None;
        }
        match q {
            Quality::High => Some("_highq"),
            Quality::Medium => Some("_medq"),
            Quality::Low => Some("_lowq"),
        }
    }
}

/// Compute the Laplacian variance for an image region. Higher values mean
/// the image is sharper (less blurry).
pub fn laplacian_variance(img: &DynamicImage) -> f64 {
    // Convert to grayscale u8 image
    let gray: GrayImage = img.to_luma8();
    let (w, h) = gray.dimensions();

    if w == 0 || h == 0 {
        return 0.0;
    }

    // Convert to ndarray for convolution
    let mut arr = Array2::<f64>::zeros((h as usize, w as usize));
    for (y, mut row) in arr.rows_mut().into_iter().enumerate() {
        for (x, val) in row.iter_mut().enumerate() {
            *val = gray.get_pixel(x as u32, y as u32)[0] as f64;
        }
    }

    // 3x3 Laplacian kernel
    let k: [[f64; 3]; 3] = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]];

    let mut lap = Array2::<f64>::zeros((h as usize, w as usize));

    for y in 1..(h as usize - 1).max(1) {
        for x in 1..(w as usize - 1).max(1) {
            let mut s = 0.0;
            for (ky, krow) in k.iter().enumerate() {
                for (kx, kval) in krow.iter().enumerate() {
                    let px = x + kx - 1;
                    let py = y + ky - 1;
                    s += kval * arr[[py, px]];
                }
            }
            lap[[y, x]] = s;
        }
    }

    // compute variance of lap values
    let flat: Vec<f64> = lap.iter().copied().collect();
    let mean = flat.iter().sum::<f64>() / (flat.len() as f64);
    flat.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (flat.len() as f64)
}

/// Estimate the quality bucket for the image using Laplacian variance.
pub fn estimate_sharpness(img: &DynamicImage) -> (f64, Quality) {
    let v = laplacian_variance(img);
    (v, Quality::from_variance(v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbaImage;

    #[test]
    fn variance_of_blank_image_is_low() {
        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            64,
            64,
            image::Rgba([128, 128, 128, 255]),
        ));
        let (v, q) = estimate_sharpness(&img);
        assert!(v >= 0.0);
        assert_eq!(q, Quality::Low);
    }

    #[test]
    fn variance_of_sharp_image_is_higher() {
        let mut img = RgbaImage::from_pixel(64, 64, image::Rgba([128, 128, 128, 255]));
        // draw a high-frequency pattern
        for y in 0..64 {
            for x in 0..64 {
                let v = if (x + y) % 2 == 0 { 255u8 } else { 0u8 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 255]));
            }
        }
        let dynimg = DynamicImage::ImageRgba8(img);
        let (v, q) = estimate_sharpness(&dynimg);
        assert!(v > 0.0);
        assert!(q == Quality::Medium || q == Quality::High);
    }
}
