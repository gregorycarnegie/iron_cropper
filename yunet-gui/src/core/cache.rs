//! Caching logic for crop previews and textures.

use std::path::PathBuf;
use std::sync::Arc;

use egui::{Context as EguiContext, TextureHandle, TextureOptions};
use image::DynamicImage;
use log::warn;
use yunet_core::{CropSettings as CoreCropSettings, PositioningMode, calculate_crop_region};
use yunet_utils::{
    apply_shape_mask,
    config::CropSettings as ConfigCropSettings,
    enhance::{EnhancementSettings, apply_enhancements},
    load_image,
};

use crate::{
    CropPreviewCacheEntry, CropPreviewKey, DetectionWithQuality, EnhancementSignature,
    ShapeSignature, YuNetApp,
};

/// Shared parameters required to build or fetch a crop preview entry.
pub struct CropPreviewRequest<'a> {
    pub path: &'a PathBuf,
    pub face_idx: usize,
    pub detection: &'a DetectionWithQuality,
    pub source_image: &'a mut Option<Arc<DynamicImage>>,
    pub crop_settings: &'a CoreCropSettings,
    pub crop_config: &'a ConfigCropSettings,
    pub enhancement_settings: &'a EnhancementSettings,
    pub enhance_enabled: bool,
}

/// Ensures a crop preview entry exists in the cache, creating it if necessary.
pub fn ensure_crop_preview_entry(
    cache: &mut std::collections::HashMap<CropPreviewKey, CropPreviewCacheEntry>,
    request: CropPreviewRequest<'_>,
) -> Option<(CropPreviewKey, Arc<DynamicImage>)> {
    let CropPreviewRequest {
        path,
        face_idx,
        detection,
        source_image,
        crop_settings,
        crop_config,
        enhancement_settings,
        enhance_enabled,
    } = request;

    let signature = enhancement_signature(enhancement_settings);

    let key = CropPreviewKey {
        path: path.clone(),
        face_index: face_idx,
        output_width: crop_settings.output_width,
        output_height: crop_settings.output_height,
        positioning_mode: positioning_mode_id(crop_settings.positioning_mode),
        face_height_bits: crop_settings.face_height_pct.to_bits(),
        horizontal_bits: crop_settings.horizontal_offset.to_bits(),
        vertical_bits: crop_settings.vertical_offset.to_bits(),
        shape: shape_signature(crop_config),
        enhancement: signature,
        enhance_enabled,
    };

    if let Some(entry) = cache.get(&key) {
        return Some((key, entry.image.clone()));
    }

    let img = if let Some(img) = source_image.as_ref() {
        img.clone()
    } else {
        match load_image(path) {
            Ok(img) => {
                let arc = Arc::new(img);
                *source_image = Some(arc.clone());
                arc
            }
            Err(err) => {
                warn!(
                    "Failed to load {} for crop preview: {}",
                    path.display(),
                    err
                );
                return None;
            }
        }
    };

    let bbox = detection.active_bbox();
    let crop_region = calculate_crop_region(img.width(), img.height(), bbox, crop_settings);
    let cropped = img.crop_imm(
        crop_region.x,
        crop_region.y,
        crop_region.width,
        crop_region.height,
    );
    let resized = cropped.resize_exact(
        crop_settings.output_width,
        crop_settings.output_height,
        image::imageops::FilterType::Lanczos3,
    );
    let processed = if enhance_enabled {
        apply_enhancements(&resized, enhancement_settings)
    } else {
        resized
    };
    let mut rgba = processed.to_rgba8();
    apply_shape_mask(&mut rgba, &crop_config.shape);
    let final_image = DynamicImage::ImageRgba8(rgba);
    let arc_image = Arc::new(final_image);
    cache.insert(
        key.clone(),
        CropPreviewCacheEntry {
            image: arc_image.clone(),
            texture: None,
        },
    );
    Some((key, arc_image))
}

/// Loads a texture from a DynamicImage.
pub fn load_texture_from_image(
    ctx: &EguiContext,
    image: &DynamicImage,
    texture_seq: &mut u64,
) -> TextureHandle {
    let rgba = image.to_rgba8();
    let size = [rgba.width() as usize, rgba.height() as usize];
    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
    let texture_name = format!("yunet-crop-preview-{}", texture_seq);
    *texture_seq = texture_seq.wrapping_add(1);
    ctx.load_texture(texture_name, color_image, TextureOptions::LINEAR)
}

/// Gets or creates a texture for a crop preview.
pub fn crop_preview_texture_for(
    ctx: &EguiContext,
    cache: &mut std::collections::HashMap<CropPreviewKey, CropPreviewCacheEntry>,
    request: CropPreviewRequest<'_>,
    texture_seq: &mut u64,
) -> Option<TextureHandle> {
    let (key, image) = ensure_crop_preview_entry(cache, request)?;

    if let Some(texture) = cache.get(&key).and_then(|entry| entry.texture.clone()) {
        return Some(texture);
    }

    let texture = load_texture_from_image(ctx, &image, texture_seq);
    if let Some(entry) = cache.get_mut(&key) {
        entry.texture = Some(texture.clone());
    }
    Some(texture)
}

/// Creates an enhancement signature from settings for cache keying.
fn enhancement_signature(settings: &EnhancementSettings) -> EnhancementSignature {
    EnhancementSignature {
        auto_color: settings.auto_color,
        exposure_bits: settings.exposure_stops.to_bits(),
        brightness: settings.brightness,
        contrast_bits: settings.contrast.to_bits(),
        saturation_bits: settings.saturation.to_bits(),
        unsharp_amount_bits: settings.unsharp_amount.to_bits(),
        unsharp_radius_bits: settings.unsharp_radius.to_bits(),
        sharpness_bits: settings.sharpness.to_bits(),
        skin_smooth_bits: settings.skin_smooth_amount.to_bits(),
        skin_sigma_space_bits: settings.skin_smooth_sigma_space.to_bits(),
        skin_sigma_color_bits: settings.skin_smooth_sigma_color.to_bits(),
        red_eye_removal: settings.red_eye_removal,
        red_eye_threshold_bits: settings.red_eye_threshold.to_bits(),
        background_blur: settings.background_blur,
        background_blur_radius_bits: settings.background_blur_radius.to_bits(),
        background_blur_mask_bits: settings.background_blur_mask_size.to_bits(),
    }
}

/// Converts a positioning mode to an ID for cache keying.
fn positioning_mode_id(mode: PositioningMode) -> u8 {
    match mode {
        PositioningMode::Center => 0,
        PositioningMode::RuleOfThirds => 1,
        PositioningMode::Custom => 2,
    }
}

/// Creates a shape signature from crop settings for cache keying.
fn shape_signature(settings: &ConfigCropSettings) -> ShapeSignature {
    use yunet_utils::{CropShape, PolygonCornerStyle};

    let shape = settings.shape.clone().sanitized();
    match shape {
        CropShape::Rectangle => ShapeSignature {
            kind: 0,
            primary_bits: 0,
            secondary_bits: 0,
            sides: 0,
            rotation_bits: 0,
        },
        CropShape::Ellipse => ShapeSignature {
            kind: 1,
            primary_bits: 0,
            secondary_bits: 0,
            sides: 0,
            rotation_bits: 0,
        },
        CropShape::RoundedRectangle { radius_pct } => ShapeSignature {
            kind: 2,
            primary_bits: radius_pct.to_bits(),
            secondary_bits: 0,
            sides: 0,
            rotation_bits: 0,
        },
        CropShape::ChamferedRectangle { size_pct } => ShapeSignature {
            kind: 3,
            primary_bits: size_pct.to_bits(),
            secondary_bits: 0,
            sides: 0,
            rotation_bits: 0,
        },
        CropShape::Polygon {
            sides,
            rotation_deg,
            corner_style,
        } => {
            let (kind, primary, secondary) = match corner_style {
                PolygonCornerStyle::Sharp => (4_u8, 0_u32, 0_u32),
                PolygonCornerStyle::Rounded { radius_pct } => (5_u8, radius_pct.to_bits(), 0_u32),
                PolygonCornerStyle::Chamfered { size_pct } => (6_u8, size_pct.to_bits(), 0_u32),
            };
            ShapeSignature {
                kind,
                primary_bits: primary,
                secondary_bits: secondary,
                sides,
                rotation_bits: rotation_deg.to_bits(),
            }
        }
    }
}

impl YuNetApp {
    /// Clears the crop preview cache.
    pub(crate) fn clear_crop_preview_cache(&mut self) {
        self.crop_preview_cache.clear();
    }

    /// Clears the detection cache and updates the status message.
    pub(crate) fn clear_cache(&mut self, message: impl Into<String>) {
        self.cache.clear();
        self.crop_preview_cache.clear();
        self.show_success(message);
    }

    /// Invalidates the detector, forcing it to be reloaded.
    pub(crate) fn invalidate_detector(&mut self) {
        self.detector = None;
        self.cache.clear();
        self.crop_preview_cache.clear();
        self.show_success("Model settings changed. Detector will reload on next detection.");
    }

    /// Persists the current settings to disk and provides feedback.
    pub(crate) fn persist_settings_with_feedback(&mut self) {
        use crate::core::settings;
        if let Err(err) =
            settings::persist_settings_with_feedback(&self.settings, &self.settings_path)
        {
            self.show_error("Failed to save settings", err);
        }
    }
}
