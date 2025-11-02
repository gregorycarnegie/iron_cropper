//! Crop calculation utilities for face-centric image crops.
//!
//! Implements a scale-based approach where the source crop region is computed
//! from a desired face height percentage of the output image. The crop region
//! preserves the output aspect ratio and is clamped to the image boundaries.

use crate::postprocess::BoundingBox;

/// How to position the face within the crop region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositioningMode {
    /// Center the face in the crop region.
    Center,
    /// Place the face roughly at the upper third (rule of thirds).
    RuleOfThirds,
    /// Use custom offsets (fractions) to nudge the face relative to crop center.
    Custom,
}

/// Settings controlling how crops are computed and the output size.
#[derive(Debug, Clone)]
pub struct CropSettings {
    /// Desired output width in pixels.
    pub output_width: u32,
    /// Desired output height in pixels.
    pub output_height: u32,
    /// Percentage of the output's height that the face should occupy (e.g. 70.0).
    /// Must be > 0.0 and <= 100.0.
    pub face_height_pct: f32,
    /// Positioning mode to use when placing the face inside the crop.
    pub positioning_mode: PositioningMode,
    /// Horizontal offset used only for `Custom` mode. Range expected -1.0..=1.0
    /// where negative moves the face left and positive moves it right.
    pub horizontal_offset: f32,
    /// Vertical offset used only for `Custom` mode. Range expected -1.0..=1.0
    /// where negative moves the face upward and positive moves it downward.
    pub vertical_offset: f32,
}

/// Integer crop region in source image coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropRegion {
    /// Top-left x coordinate.
    pub x: u32,
    /// Top-left y coordinate.
    pub y: u32,
    /// Width of the crop region.
    pub width: u32,
    /// Height of the crop region.
    pub height: u32,
}

impl Default for CropSettings {
    fn default() -> Self {
        Self {
            output_width: 400,
            output_height: 400,
            face_height_pct: 70.0,
            positioning_mode: PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
        }
    }
}

/// Calculate a crop region in source image coordinates.
///
/// The algorithm proceeds as follows:
/// 1. Clamp `face_height_pct` into the `[1, 100]` range to keep the math stable.
/// 2. Derive the source crop height so that, after resizing, the detected face
///    occupies the requested percentage of the output height.
/// 3. Derive the source crop width that preserves the requested output aspect ratio.
/// 4. Shift the crop center according to [`PositioningMode`], applying clamped custom offsets.
/// 5. Clamp the crop rectangle to the image bounds and return integer coordinates.
///
/// # Arguments
/// - `img_w`, `img_h`: source image dimensions in pixels.
/// - `face_bbox`: detected face bounding box in source coordinates.
/// - `settings`: configuration that controls output size and target face coverage.
///
/// # Examples
///
/// ```rust
/// # use yunet_core::cropper::{calculate_crop_region, CropSettings, PositioningMode};
/// # use yunet_core::postprocess::BoundingBox;
/// let settings = CropSettings {
///     output_width: 400,
///     output_height: 400,
///     face_height_pct: 50.0,
///     positioning_mode: PositioningMode::Center,
///     ..Default::default()
/// };
/// let crop = calculate_crop_region(
///     1000,
///     1000,
///     BoundingBox {
///         x: 400.0,
///         y: 400.0,
///         width: 200.0,
///         height: 200.0,
///     },
///     &settings,
/// );
/// assert_eq!(crop.x, 300);
/// assert_eq!(crop.y, 300);
/// assert_eq!(crop.width, 400);
/// assert_eq!(crop.height, 400);
/// ```
///
/// Positioning modes allow the caller to shift the face within the crop:
///
/// ```rust
/// # use yunet_core::cropper::{calculate_crop_region, CropSettings, PositioningMode};
/// # use yunet_core::postprocess::BoundingBox;
/// let bbox = BoundingBox {
///     x: 400.0,
///     y: 300.0,
///     width: 200.0,
///     height: 200.0,
/// };
/// let center = calculate_crop_region(
///     1200,
///     800,
///     bbox,
///     &CropSettings {
///         output_width: 600,
///         output_height: 400,
///         face_height_pct: 50.0,
///         positioning_mode: PositioningMode::Center,
///         ..Default::default()
///     },
/// );
/// let thirds = calculate_crop_region(
///     1200,
///     800,
///     bbox,
///     &CropSettings {
///         output_width: 600,
///         output_height: 400,
///         face_height_pct: 50.0,
///         positioning_mode: PositioningMode::RuleOfThirds,
///         ..Default::default()
///     },
/// );
/// let custom = calculate_crop_region(
///     1200,
///     800,
///     bbox,
///     &CropSettings {
///         output_width: 600,
///         output_height: 400,
///         face_height_pct: 50.0,
///         positioning_mode: PositioningMode::Custom,
///         horizontal_offset: 0.5,
///         vertical_offset: -0.5,
///         ..Default::default()
///     },
/// );
/// assert!(thirds.y < center.y, "rule of thirds nudges the face upward");
/// assert!(custom.x > center.x, "positive horizontal offset moves right");
/// ```
pub fn calculate_crop_region(
    img_w: u32,
    img_h: u32,
    face_bbox: BoundingBox,
    settings: &CropSettings,
) -> CropRegion {
    // Safety clamps and early guards
    let img_w_f = img_w as f32;
    let img_h_f = img_h as f32;

    let face_h = face_bbox.height.max(1.0);
    let face_cx = face_bbox.x + face_bbox.width / 2.0;
    let face_cy = face_bbox.y + face_bbox.height / 2.0;

    let face_height_pct = settings.face_height_pct.clamp(1.0, 100.0);

    // Source region height (in source pixels) so that after resizing the face
    // will occupy `face_height_pct` percent of the output height.
    // Derived from: (face_h / src_region_h) = face_height_pct/100
    let mut src_h = face_h * (100.0 / face_height_pct);

    // Maintain output aspect ratio for width.
    let out_w = settings.output_width as f32;
    let out_h = settings.output_height as f32;
    let aspect = if out_h > 0.0 { out_w / out_h } else { 1.0 };
    let mut src_w = src_h * aspect;

    // If requested source width/height exceed image bounds, clamp and adjust
    if src_w > img_w_f {
        src_w = img_w_f;
        // adjust src_h to preserve aspect
        src_h = src_w / aspect;
    }
    if src_h > img_h_f {
        src_h = img_h_f;
        src_w = src_h * aspect;
    }

    // Determine crop center based on positioning mode
    let mut cx = face_cx;
    let mut cy = face_cy;

    match settings.positioning_mode {
        PositioningMode::Center => {
            // face center remains as-is
        }
        PositioningMode::RuleOfThirds => {
            // Place face center at 1/3 down from the top of the crop region.
            // So crop_top = face_cy - src_h * (1.0/3.0)
            cy = face_cy + (src_h * (1.0 / 3.0) - src_h / 2.0);
        }
        PositioningMode::Custom => {
            // Offsets are fractions in [-1,1] relative to half the crop dimension.
            let ho = settings.horizontal_offset.clamp(-1.0, 1.0);
            let vo = settings.vertical_offset.clamp(-1.0, 1.0);
            cx = face_cx + ho * (src_w * 0.5);
            cy = face_cy + vo * (src_h * 0.5);
        }
    }

    // Compute top-left corner from center
    let mut left = cx - src_w / 2.0;
    let mut top = cy - src_h / 2.0;

    // Clamp to image bounds
    if left < 0.0 {
        left = 0.0;
    }
    if top < 0.0 {
        top = 0.0;
    }
    if left + src_w > img_w_f {
        left = (img_w_f - src_w).max(0.0);
    }
    if top + src_h > img_h_f {
        top = (img_h_f - src_h).max(0.0);
    }

    // Round to integers and ensure inside image
    let x = left.round().max(0.0).min(img_w_f).floor() as u32;
    let y = top.round().max(0.0).min(img_h_f).floor() as u32;
    let width = src_w.round().max(1.0).min((img_w - x) as f32).floor() as u32;
    let height = src_h.round().max(1.0).min((img_h - y) as f32).floor() as u32;

    CropRegion {
        x,
        y,
        width,
        height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_crop_basic() {
        let img_w = 1000;
        let img_h = 1000;
        let face = BoundingBox {
            x: 400.0,
            y: 400.0,
            width: 200.0,
            height: 200.0,
        };

        let settings = CropSettings {
            output_width: 400,
            output_height: 400,
            face_height_pct: 50.0,
            positioning_mode: PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
        };

        let crop = calculate_crop_region(img_w, img_h, face, &settings);
        // face_h = 200, src_h = 200 * (100/50) = 400, aspect=1 -> src_w=400
        assert_eq!(crop.width, 400);
        assert_eq!(crop.height, 400);
        // center at (500,500) => top-left = (300,300)
        assert_eq!(crop.x, 300);
        assert_eq!(crop.y, 300);
    }

    #[test]
    fn clamp_near_top_edge() {
        let img_w = 800;
        let img_h = 600;
        let face = BoundingBox {
            x: 350.0,
            y: 10.0,
            width: 100.0,
            height: 100.0,
        };

        let settings = CropSettings {
            output_width: 400,
            output_height: 600,
            face_height_pct: 30.0,
            positioning_mode: PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
        };

        let crop = calculate_crop_region(img_w, img_h, face, &settings);
        // src_h = 100 * (100/30) ~= 333 -> will clamp if exceeds img_h
        assert!(crop.y == 0 || crop.y + crop.height <= img_h);
        assert!(crop.x + crop.width <= img_w);
    }

    #[test]
    fn rule_of_thirds_moves_face_upwards() {
        let img_w = 1000;
        let img_h = 1000;
        let face = BoundingBox {
            x: 400.0,
            y: 400.0,
            width: 200.0,
            height: 200.0,
        };

        let settings = CropSettings {
            output_width: 400,
            output_height: 600,
            face_height_pct: 50.0,
            positioning_mode: PositioningMode::RuleOfThirds,
            ..Default::default()
        };

        let crop = calculate_crop_region(img_w, img_h, face, &settings);
        // For rule of thirds the crop should be shifted so the face is above center.
        // Compare to center positioning to ensure y is smaller (moved up).
        let center_settings = CropSettings {
            positioning_mode: PositioningMode::Center,
            ..settings.clone()
        };
        let center_crop = calculate_crop_region(img_w, img_h, face, &center_settings);
        assert!(
            crop.y < center_crop.y,
            "rule of thirds should move crop upwards"
        );
    }

    #[test]
    fn custom_offset_moves_crop() {
        let img_w = 1000;
        let img_h = 1000;
        let face = BoundingBox {
            x: 400.0,
            y: 400.0,
            width: 200.0,
            height: 200.0,
        };

        let settings = CropSettings {
            output_width: 400,
            output_height: 400,
            face_height_pct: 50.0,
            positioning_mode: PositioningMode::Custom,
            horizontal_offset: 0.5,
            vertical_offset: -0.5,
        };

        let crop = calculate_crop_region(img_w, img_h, face, &settings);
        // Compared to center, x should be larger and y should be smaller
        let center_settings = CropSettings {
            positioning_mode: PositioningMode::Center,
            ..settings.clone()
        };
        let center_crop = calculate_crop_region(img_w, img_h, face, &center_settings);
        assert!(
            crop.x > center_crop.x,
            "custom horizontal offset should move crop right"
        );
        assert!(
            crop.y < center_crop.y,
            "custom vertical offset should move crop up"
        );
    }

    #[test]
    fn clamps_when_face_near_left_and_top_edges() {
        let img_w = 600;
        let img_h = 400;
        let face = BoundingBox {
            x: 5.0,
            y: 8.0,
            width: 90.0,
            height: 110.0,
        };

        let settings = CropSettings {
            output_width: 400,
            output_height: 400,
            face_height_pct: 80.0,
            positioning_mode: PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
        };

        let crop = calculate_crop_region(img_w, img_h, face, &settings);
        assert_eq!(crop.x, 0, "crop should clamp to left edge");
        assert_eq!(crop.y, 0, "crop should clamp to top edge");
        assert!(crop.width <= img_w);
        assert!(crop.height <= img_h);
    }

    #[test]
    fn clamps_when_face_near_bottom_right_edges() {
        let img_w = 640;
        let img_h = 480;
        let face = BoundingBox {
            x: 520.0,
            y: 360.0,
            width: 100.0,
            height: 100.0,
        };

        let settings = CropSettings {
            output_width: 320,
            output_height: 480,
            face_height_pct: 60.0,
            positioning_mode: PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
        };

        let crop = calculate_crop_region(img_w, img_h, face, &settings);
        assert!(
            crop.x + crop.width <= img_w,
            "crop should not extend past right edge"
        );
        assert!(
            crop.y + crop.height <= img_h,
            "crop should not extend past bottom edge"
        );
    }

    #[test]
    fn respects_non_square_output_aspect_ratio() {
        let img_w = 1200;
        let img_h = 800;
        let face = BoundingBox {
            x: 500.0,
            y: 300.0,
            width: 120.0,
            height: 120.0,
        };

        let settings = CropSettings {
            output_width: 600,
            output_height: 300,
            face_height_pct: 50.0,
            positioning_mode: PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
        };

        let crop = calculate_crop_region(img_w, img_h, face, &settings);
        assert_eq!(crop.height, 240);
        assert_eq!(crop.width, 480);
        assert!(
            (crop.width as f32 / crop.height as f32 - 2.0).abs() < f32::EPSILON,
            "crop width/height should match requested aspect ratio"
        );
    }

    #[test]
    fn face_height_percent_is_clamped() {
        let img_w = 800;
        let img_h = 800;
        let face = BoundingBox {
            x: 300.0,
            y: 300.0,
            width: 200.0,
            height: 200.0,
        };

        let settings = CropSettings {
            output_width: 400,
            output_height: 400,
            face_height_pct: 150.0,
            positioning_mode: PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
        };

        let crop = calculate_crop_region(img_w, img_h, face, &settings);
        // With clamp to 100%, the source region equals bounding box size.
        assert_eq!(crop.width, 200);
        assert_eq!(crop.height, 200);
    }
}
