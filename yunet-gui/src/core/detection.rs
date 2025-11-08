//! Face detection workflow and detector management.

use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};

use anyhow::{Context as AnyhowContext, Result};
use egui::{Context as EguiContext, TextureOptions};
use image::DynamicImage;
use log::{error, info};
use yunet_core::{PostprocessConfig, PreprocessConfig, YuNetDetector};
use yunet_utils::{config::AppSettings, load_image, quality::estimate_sharpness};

use crate::{CacheKey, DetectionJobSuccess, DetectionOrigin, DetectionWithQuality, JobMessage};

/// Builds a `YuNetDetector` from the given application settings.
pub fn build_detector(settings: &AppSettings) -> Result<YuNetDetector> {
    let model_path = settings
        .model_path
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("no model path configured"))?;

    let preprocess: PreprocessConfig = settings.input.into();

    let postprocess: PostprocessConfig = (&settings.detection).into();

    YuNetDetector::new(model_path, preprocess, postprocess).with_context(|| {
        format!(
            "failed to load YuNet model from configured path {}",
            model_path
        )
    })
}

/// Performs face detection on an image and returns the results.
pub fn perform_detection(
    detector: Arc<YuNetDetector>,
    path: PathBuf,
) -> Result<DetectionJobSuccess> {
    let image = Arc::new(
        load_image(&path)
            .with_context(|| format!("failed to load image from {}", path.display()))?,
    );
    let detection_output = detector
        .detect_image(&image)
        .with_context(|| format!("YuNet detection failed for {}", path.display()))?;

    // Calculate quality scores for each detected face
    let detections_with_quality: Vec<DetectionWithQuality> = detection_output
        .detections
        .into_iter()
        .map(|detection| {
            // Crop face region for quality analysis
            let bbox = detection.bbox;
            let x = bbox.x.max(0.0) as u32;
            let y = bbox.y.max(0.0) as u32;
            let w = bbox.width.max(1.0) as u32;
            let h = bbox.height.max(1.0) as u32;

            // Clamp to image bounds
            let img_w = image.width();
            let img_h = image.height();
            let x = x.min(img_w.saturating_sub(1));
            let y = y.min(img_h.saturating_sub(1));
            let w = w.min(img_w.saturating_sub(x));
            let h = h.min(img_h.saturating_sub(y));

            let face_region = image.crop_imm(x, y, w, h);
            let (quality_score, quality) = estimate_sharpness(&face_region);

            DetectionWithQuality {
                detection,
                quality_score,
                quality,
                thumbnail: None, // Will be created on the GUI thread
                current_bbox: bbox,
                original_bbox: bbox,
                origin: DetectionOrigin::Detector,
            }
        })
        .collect();

    let rgba = image.to_rgba8();
    let size = [rgba.width() as usize, rgba.height() as usize];
    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());

    Ok(DetectionJobSuccess {
        path,
        color_image,
        detections: detections_with_quality,
        original_size: detection_output.original_size,
        original_image: image,
    })
}

/// Creates thumbnail textures for detected faces.
pub fn create_thumbnails(
    ctx: &EguiContext,
    detections: &mut [DetectionWithQuality],
    image: &DynamicImage,
    texture_seq: &mut u64,
) {
    for (index, det) in detections.iter_mut().enumerate() {
        let bbox = det.active_bbox();
        let x = bbox.x.max(0.0) as u32;
        let y = bbox.y.max(0.0) as u32;
        let w = bbox.width.max(1.0) as u32;
        let h = bbox.height.max(1.0) as u32;

        // Clamp to image bounds
        let img_w = image.width();
        let img_h = image.height();
        let x = x.min(img_w.saturating_sub(1));
        let y = y.min(img_h.saturating_sub(1));
        let w = w.min(img_w.saturating_sub(x));
        let h = h.min(img_h.saturating_sub(y));

        let face_region = image.crop_imm(x, y, w, h);
        // Resize to thumbnail size (96x96)
        let thumb = face_region.resize(96, 96, image::imageops::FilterType::Lanczos3);
        let thumb_rgba = thumb.to_rgba8();
        let thumb_size = [thumb_rgba.width() as usize, thumb_rgba.height() as usize];
        let thumb_color = egui::ColorImage::from_rgba_unmultiplied(thumb_size, thumb_rgba.as_raw());

        let texture_name = format!("yunet-face-thumb-{}-{}", texture_seq, index);
        *texture_seq = texture_seq.wrapping_add(1);
        let texture = ctx.load_texture(texture_name, thumb_color, TextureOptions::LINEAR);
        det.thumbnail = Some(texture);
    }
}

/// Starts a new detection job for the given image path.
///
/// This function checks the cache first, and if not found, launches a background job.
pub fn start_detection(
    path: PathBuf,
    cache_key: CacheKey,
    detector: Arc<YuNetDetector>,
    job_id: u64,
    job_tx: mpsc::Sender<JobMessage>,
) {
    info!("Launching detection job {} for {}", job_id, path.display());

    rayon::spawn(move || {
        let payload = match perform_detection(detector, path.clone()) {
            Ok(data) => JobMessage::DetectionFinished {
                job_id,
                cache_key,
                data,
            },
            Err(err) => JobMessage::DetectionFailed {
                job_id,
                error: format!("{err}"),
            },
        };

        if job_tx.send(payload).is_err() {
            error!("GUI dropped detection result for {}", path.display());
        }
    });
}

/// Ensures that the detector is loaded, building it if necessary.
pub fn ensure_detector(
    detector: &mut Option<Arc<YuNetDetector>>,
    settings: &AppSettings,
) -> Result<Arc<YuNetDetector>> {
    if let Some(detector) = detector {
        return Ok(detector.clone());
    }

    let new_detector = Arc::new(build_detector(settings)?);
    *detector = Some(new_detector.clone());
    Ok(new_detector)
}

/// Creates a cache key for the given image path and current settings.
pub fn cache_key_for_path(path: &Path, settings: &AppSettings) -> CacheKey {
    CacheKey {
        path: path.to_path_buf(),
        model_path: settings.model_path.clone(),
        input_width: settings.input.width,
        input_height: settings.input.height,
        score_bits: settings.detection.score_threshold.to_bits(),
        nms_bits: settings.detection.nms_threshold.to_bits(),
        top_k: settings.detection.top_k,
    }
}
