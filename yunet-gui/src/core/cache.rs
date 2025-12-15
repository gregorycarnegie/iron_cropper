//! Caching logic for crop previews and textures.

use crate::{
    CropPreviewCacheEntry, CropPreviewKey, DetectionWithQuality, EnhancementSignature,
    ShapeSignature, YuNetApp,
};
use yunet_utils::gpu::{GpuStatusIndicator, red_eye::RedEye};

use egui::{Context as EguiContext, TextureHandle, TextureOptions};
use image::DynamicImage;
use log::warn;
use lru::LruCache;
use std::{path::PathBuf, sync::Arc};
use yunet_core::{
    CropSettings as CoreCropSettings, FillColor, PositioningMode, calculate_crop_region,
    crop_face_from_image,
};
use yunet_utils::{
    BatchCropRequest, GpuBatchCropper, WgpuEnhancer, apply_shape_mask,
    config::CropSettings as ConfigCropSettings,
    enhance::{EnhancementSettings, apply_enhancements},
    load_image,
};

/// Helper to get or load an image using the app-level cache.
pub fn get_or_load_cached_image(
    image_cache: &mut LruCache<PathBuf, Arc<DynamicImage>>,
    path: &PathBuf,
) -> Result<Arc<DynamicImage>, anyhow::Error> {
    if let Some(cached) = image_cache.get(path) {
        return Ok(cached.clone());
    }

    let loaded = load_image(path)?;
    let arc = Arc::new(loaded);
    image_cache.put(path.clone(), arc.clone());
    Ok(arc)
}

/// Shared parameters required to build or fetch a crop preview entry.
pub struct CropPreviewRequest<'a> {
    pub path: &'a PathBuf,
    pub face_idx: usize,
    pub detection: &'a DetectionWithQuality,
    pub source_image: &'a mut Option<Arc<DynamicImage>>,
    pub image_cache: &'a mut LruCache<PathBuf, Arc<DynamicImage>>,
    pub crop_settings: &'a CoreCropSettings,
    pub crop_config: &'a ConfigCropSettings,
    pub enhancement_settings: &'a EnhancementSettings,
    pub enhance_enabled: bool,
    pub gpu_enhancer: Option<Arc<WgpuEnhancer>>,
    pub gpu_cropper: Option<Arc<GpuBatchCropper>>,
}

pub fn enhance_with_gpu(
    image: &DynamicImage,
    settings: &EnhancementSettings,
    enhancer: Option<&Arc<WgpuEnhancer>>,
    eyes: Option<&[RedEye]>,
) -> DynamicImage {
    if let Some(enhancer) = enhancer {
        match enhancer.apply(image, settings, eyes) {
            Ok(result) => return result,
            Err(err) => {
                warn!("GPU enhancement failed: {err}; falling back to CPU pipeline.");
            }
        }
    }
    apply_enhancements(image, settings, eyes)
}

pub fn calculate_eyes_relative_to_crop(
    landmarks: &[yunet_core::Landmark; 5],
    crop_region: &yunet_core::CropRegion,
    img_width: u32,
    img_height: u32,
    target_width: u32,
    target_height: u32,
) -> Vec<RedEye> {
    let (cx, cy, cw, ch) = crop_region
        .in_bounds_rect(img_width, img_height)
        .unwrap_or((0, 0, img_width, img_height));
    let (ow, oh) = if target_width > 0 && target_height > 0 {
        (target_width, target_height)
    } else {
        (cw, ch)
    };

    let scale_x = ow as f32 / cw as f32;
    let scale_y = oh as f32 / ch as f32;

    // Landmarks 0 and 1 are eyes
    let mut eye_list = Vec::with_capacity(2);
    for i in 0..2 {
        let lx = landmarks[i].x;
        let ly = landmarks[i].y;
        let new_x = (lx - cx as f32) * scale_x;
        let new_y = (ly - cy as f32) * scale_y;

        // Approximate radius based on inter-pupillary distance
        let dx = landmarks[1].x - landmarks[0].x;
        let dy = landmarks[1].y - landmarks[0].y;
        let dist = (dx * dx + dy * dy).sqrt() * scale_x;
        let radius = dist * 0.15;

        eye_list.push(RedEye {
            x: new_x,
            y: new_y,
            radius,
            _pad: 0.0,
        });
    }
    eye_list
}

/// Ensures a crop preview entry exists in the cache, creating it if necessary.
pub fn ensure_crop_preview_entry(
    cache: &mut LruCache<CropPreviewKey, CropPreviewCacheEntry>,
    request: CropPreviewRequest<'_>,
) -> Option<(CropPreviewKey, Arc<DynamicImage>)> {
    let CropPreviewRequest {
        path,
        face_idx,
        detection,
        source_image,
        image_cache,
        crop_settings,
        crop_config,
        enhancement_settings,
        enhance_enabled,
        gpu_enhancer,
        gpu_cropper,
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
        fill_color_bits: pack_fill_color(crop_settings.fill_color),
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
        // Try app-level image cache first
        match get_or_load_cached_image(image_cache, path) {
            Ok(cached) => {
                *source_image = Some(cached.clone());
                cached
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
    let detection_for_crop = {
        let mut det = detection.detection.clone();
        det.bbox = bbox;
        det
    };
    let cpu_crop = || crop_face_from_image(img.as_ref(), &detection_for_crop, crop_settings);

    let resized = if crop_settings.output_width > 0
        && crop_settings.output_height > 0
        && gpu_cropper.is_some()
        && !crop_region.requires_padding()
    {
        let bounds = crop_region
            .in_bounds_rect(img.width(), img.height())
            .map(|(x, y, w, h)| BatchCropRequest {
                source_x: x,
                source_y: y,
                source_width: w.max(1),
                source_height: h.max(1),
                output_width: crop_settings.output_width,
                output_height: crop_settings.output_height,
            });
        match gpu_cropper.as_ref().and_then(|cropper| {
            bounds.and_then(|request| match cropper.crop(img.as_ref(), &[request]) {
                Ok(mut images) => {
                    if images.len() == 1 {
                        images.pop()
                    } else {
                        warn!(
                            "GPU cropper returned {} preview images (expected 1)",
                            images.len()
                        );
                        None
                    }
                }
                Err(err) => {
                    warn!("GPU preview crop failed: {err}; falling back to CPU path.");
                    None
                }
            })
        }) {
            Some(image) => image,
            None => cpu_crop(),
        }
    } else {
        cpu_crop()
    };
    let processed = if enhance_enabled {
        let eyes = if enhancement_settings.red_eye_removal {
            let landmarks = &detection.detection.landmarks;
            Some(calculate_eyes_relative_to_crop(
                landmarks,
                &crop_region,
                img.width(),
                img.height(),
                crop_settings.output_width,
                crop_settings.output_height,
            ))
        } else {
            None
        };

        enhance_with_gpu(
            &resized,
            enhancement_settings,
            gpu_enhancer.as_ref(),
            eyes.as_deref(),
        )
    } else {
        resized
    };
    let masked = apply_mask_with_gpu(processed, &crop_config.shape, gpu_enhancer.as_ref());
    let rgba = masked.to_rgba8();
    let final_image = DynamicImage::ImageRgba8(rgba);
    let arc_image = Arc::new(final_image);
    cache.put(
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
    cache: &mut LruCache<CropPreviewKey, CropPreviewCacheEntry>,
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

pub fn apply_mask_with_gpu(
    image: DynamicImage,
    shape: &yunet_utils::CropShape,
    enhancer: Option<&Arc<WgpuEnhancer>>,
) -> DynamicImage {
    if let Some(enhancer) = enhancer {
        match enhancer.apply_shape_mask_gpu(&image, shape) {
            Ok(Some(masked)) => return masked,
            Ok(None) => {}
            Err(err) => warn!("GPU shape mask failed: {err}; falling back to CPU path."),
        }
    }
    let mut rgba = image.to_rgba8();
    apply_shape_mask(&mut rgba, shape);
    DynamicImage::ImageRgba8(rgba)
}

/// Converts a positioning mode to an ID for cache keying.
fn positioning_mode_id(mode: PositioningMode) -> u8 {
    match mode {
        PositioningMode::Center => 0,
        PositioningMode::RuleOfThirds => 1,
        PositioningMode::Custom => 2,
    }
}

fn pack_fill_color(color: FillColor) -> u32 {
    ((color.alpha as u32) << 24)
        | ((color.red as u32) << 16)
        | ((color.green as u32) << 8)
        | (color.blue as u32)
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
                PolygonCornerStyle::Bezier { tension } => (7_u8, tension.to_bits(), 0_u32),
            };
            ShapeSignature {
                kind,
                primary_bits: primary,
                secondary_bits: secondary,
                sides,
                rotation_bits: rotation_deg.to_bits(),
            }
        }
        CropShape::Star {
            points,
            inner_radius_pct,
            rotation_deg,
        } => ShapeSignature {
            kind: 8,
            primary_bits: inner_radius_pct.to_bits(),
            secondary_bits: 0,
            sides: points,
            rotation_bits: rotation_deg.to_bits(),
        },
        CropShape::KochPolygon {
            sides,
            rotation_deg,
            iterations,
        } => ShapeSignature {
            kind: 9,
            primary_bits: iterations as u32,
            secondary_bits: 0,
            sides,
            rotation_bits: rotation_deg.to_bits(),
        },
        CropShape::KochRectangle { iterations } => ShapeSignature {
            kind: 10,
            primary_bits: iterations as u32,
            secondary_bits: 0,
            sides: 0,
            rotation_bits: 0,
        },
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
        self.gpu_context = None;
        self.gpu_enhancer = None;
        self.gpu_status = GpuStatusIndicator::pending();
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
