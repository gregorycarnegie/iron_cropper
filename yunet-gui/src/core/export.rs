//! Export and batch processing operations.

use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use image::DynamicImage;
use log::{info, warn};
use rayon::prelude::*;
use yunet_core::{CropSettings as CoreCropSettings, YuNetDetector, calculate_crop_region};
use yunet_utils::{
    MetadataContext, OutputOptions, WgpuEnhancer, append_suffix_to_filename,
    config::CropSettings as ConfigCropSettings,
    enhance::EnhancementSettings,
    load_image,
    quality::{Quality, estimate_sharpness},
    save_dynamic_image,
};

use crate::{
    BatchFileStatus, YuNetApp,
    core::cache::{apply_mask_with_gpu, enhance_with_gpu},
};

/// Configuration for batch export jobs.
#[derive(Clone)]
pub struct BatchJobConfig {
    pub output_dir: PathBuf,
    pub crop_settings: CoreCropSettings,
    pub crop_config: ConfigCropSettings,
    pub enhancement_settings: EnhancementSettings,
    pub enhance_enabled: bool,
    pub output_options: OutputOptions,
    pub gpu_enhancer: Option<Arc<WgpuEnhancer>>,
}

/// Runs a single batch job for one image file.
pub fn run_batch_job(
    detector: Arc<YuNetDetector>,
    path: PathBuf,
    config: Arc<BatchJobConfig>,
    output_override: Option<PathBuf>,
) -> BatchFileStatus {
    let source_image = match load_image(&path) {
        Ok(img) => img,
        Err(e) => {
            warn!("Failed to load {}: {}", path.display(), e);
            return BatchFileStatus::Failed {
                error: format!("Failed to load: {e}"),
            };
        }
    };

    let detection_output = match detector.detect_image(&source_image) {
        Ok(output) => output,
        Err(e) => {
            warn!("Detection failed for {}: {}", path.display(), e);
            return BatchFileStatus::Failed {
                error: format!("Detection failed: {e}"),
            };
        }
    };

    let faces_detected = detection_output.detections.len();

    struct FaceCandidate {
        index: usize,
        quality: Quality,
        quality_score: f64,
        score: f32,
    }

    let mut candidates = Vec::with_capacity(faces_detected);
    for (face_idx, detection) in detection_output.detections.iter().enumerate() {
        let crop_region = calculate_crop_region(
            source_image.width(),
            source_image.height(),
            detection.bbox,
            &config.crop_settings,
        );

        let cropped = source_image.crop_imm(
            crop_region.x,
            crop_region.y,
            crop_region.width,
            crop_region.height,
        );

        let resized = cropped.resize_exact(
            config.crop_settings.output_width,
            config.crop_settings.output_height,
            image::imageops::FilterType::Lanczos3,
        );

        let processed = if config.enhance_enabled {
            enhance_with_gpu(
                &resized,
                &config.enhancement_settings,
                config.gpu_enhancer.as_ref(),
            )
        } else {
            resized
        };
        let masked = apply_mask_with_gpu(
            processed,
            &config.crop_config.shape,
            config.gpu_enhancer.as_ref(),
        );
        let rgba = masked.to_rgba8();
        let final_image = DynamicImage::ImageRgba8(rgba);

        let (quality_score, quality) = estimate_sharpness(&final_image);
        candidates.push(FaceCandidate {
            index: face_idx,
            quality,
            quality_score,
            score: detection.score,
        });
    }

    if candidates.is_empty() {
        return BatchFileStatus::Completed {
            faces_detected,
            faces_exported: 0,
        };
    }

    let quality_rules = &config.crop_config.quality_rules;
    let best_quality = candidates.iter().map(|c| c.quality).max();

    if quality_rules.auto_skip_no_high_quality && best_quality != Some(Quality::High) {
        warn!(
            "Skipping {} - no high-quality faces detected",
            path.display()
        );
        return BatchFileStatus::Completed {
            faces_detected,
            faces_exported: 0,
        };
    }

    let mut exports = candidates;
    if quality_rules.auto_select_best_face
        && exports.len() > 1
        && let Some((best_idx, _)) = exports.iter().enumerate().max_by(|a, b| {
            a.1.quality.cmp(&b.1.quality).then_with(|| {
                a.1.quality_score
                    .partial_cmp(&b.1.quality_score)
                    .unwrap_or(Ordering::Equal)
            })
        })
    {
        let best_index = exports[best_idx].index;
        exports.retain(|c| c.index == best_index);
    }

    let multi_face = exports.len() > 1;

    let mut faces_exported = 0;
    for candidate in exports.into_iter() {
        let should_skip = if let Some(min) = quality_rules.min_quality {
            candidate.quality < min
        } else {
            false
        };
        if should_skip {
            info!(
                "Skipping face {} from {} due to {:?} quality",
                candidate.index + 1,
                path.display(),
                candidate.quality
            );
            continue;
        }

        let detection = match detection_output.detections.get(candidate.index) {
            Some(det) => det,
            None => continue,
        };

        let crop_region = calculate_crop_region(
            source_image.width(),
            source_image.height(),
            detection.bbox,
            &config.crop_settings,
        );

        let cropped = source_image.crop_imm(
            crop_region.x,
            crop_region.y,
            crop_region.width,
            crop_region.height,
        );

        let resized = cropped.resize_exact(
            config.crop_settings.output_width,
            config.crop_settings.output_height,
            image::imageops::FilterType::Lanczos3,
        );

        let final_image = if config.enhance_enabled {
            enhance_with_gpu(
                &resized,
                &config.enhancement_settings,
                config.gpu_enhancer.as_ref(),
            )
        } else {
            resized
        };

        let source_stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("face");
        let mut ext = config.crop_config.output_format.clone();
        if ext.is_empty() {
            ext = "png".to_string();
        }
        let ext = ext.to_ascii_lowercase();

        let output_path = if let Some(custom) = output_override.as_ref() {
            resolve_mapping_override_path(
                &config.output_dir,
                custom,
                &ext,
                candidate.index,
                multi_face,
            )
        } else {
            let mut output_filename =
                format!("{}_face_{:02}.{}", source_stem, candidate.index + 1, ext);
            if quality_rules.quality_suffix {
                let suffix = match candidate.quality {
                    Quality::High => Some("_highq"),
                    Quality::Medium => Some("_medq"),
                    Quality::Low => Some("_lowq"),
                };
                if let Some(suffix) = suffix {
                    output_filename = append_suffix_to_filename(&output_filename, suffix);
                }
            }
            config.output_dir.join(output_filename)
        };

        if let Some(parent) = output_path.parent()
            && let Err(err) = std::fs::create_dir_all(parent)
        {
            warn!(
                "Failed to create output directory {}: {err}",
                parent.display()
            );
        }

        let metadata_ctx = MetadataContext {
            source_path: Some(path.as_path()),
            crop_settings: Some(&config.crop_config),
            detection_score: Some(candidate.score),
            quality: Some(candidate.quality),
            quality_score: Some(candidate.quality_score),
        };

        match save_dynamic_image(
            &final_image,
            &output_path,
            &config.output_options,
            &metadata_ctx,
        ) {
            Ok(_) => {
                faces_exported += 1;
            }
            Err(err) => {
                warn!(
                    "Failed to save face {} from {}: {}",
                    candidate.index + 1,
                    path.display(),
                    err
                );
            }
        }
    }

    BatchFileStatus::Completed {
        faces_detected,
        faces_exported,
    }
}

/// Resolves the output path for a mapped file, handling relative and absolute paths.
fn resolve_mapping_override_path(
    output_dir: &Path,
    override_target: &Path,
    ext: &str,
    face_index: usize,
    multi_face: bool,
) -> PathBuf {
    let cleaned_ext = ext.trim_start_matches('.').to_string();
    let parent = if override_target.is_absolute() {
        override_target
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default()
    } else {
        let rel_parent = override_target.parent().unwrap_or_else(|| Path::new(""));
        output_dir.join(rel_parent)
    };
    let base_name = override_target
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "output".to_string());
    let final_base = if multi_face {
        format!("{base_name}_face{}", face_index + 1)
    } else {
        base_name
    };
    let mut final_path = parent;
    final_path.push(final_base);
    final_path.set_extension(cleaned_ext);
    final_path
}

/// Runs batch export processing in parallel.
pub fn run_batch_export(
    tasks: Vec<(usize, PathBuf, Option<PathBuf>)>,
    detector: Arc<YuNetDetector>,
    config: Arc<BatchJobConfig>,
) -> Vec<(usize, BatchFileStatus)> {
    tasks
        .into_par_iter()
        .map(|(idx, path, override_path)| {
            let status = run_batch_job(detector.clone(), path, config.clone(), override_path);
            (idx, status)
        })
        .collect()
}

/// Exports selected faces from the preview to disk.
pub fn export_selected_faces(app: &mut YuNetApp) {
    use rfd::FileDialog;
    use yunet_core::calculate_crop_region;

    if app.selected_faces.is_empty() {
        app.show_error("Export failed", "No faces selected for export");
        return;
    }

    let Some(source_image) = app.preview.source_image.as_ref() else {
        app.show_error("Export failed", "No image loaded");
        return;
    };

    // Ask user to pick output directory
    let Some(output_dir) = FileDialog::new().pick_folder() else {
        return; // User cancelled
    };

    if let Err(err) = std::fs::create_dir_all(&output_dir) {
        app.show_error(
            "Export failed",
            format!("Failed to create output directory: {err}"),
        );
        return;
    }

    let source_path = app.preview.image_path.as_ref();
    let source_stem = source_path
        .and_then(|p| p.file_stem())
        .and_then(|s| s.to_str())
        .unwrap_or("face");

    let crop_settings = app.build_crop_settings();
    let enhancement_settings = app.build_enhancement_settings();
    let ext = app.settings.crop.output_format.to_ascii_lowercase();
    let ext = if ext.is_empty() { "png" } else { &ext };

    let mut exported = 0;
    let mut failed = 0;

    for &face_idx in app.selected_faces.iter() {
        let Some(det) = app.preview.detections.get(face_idx) else {
            continue;
        };

        let crop_region = calculate_crop_region(
            source_image.width(),
            source_image.height(),
            det.active_bbox(),
            &crop_settings,
        );

        let cropped = source_image.crop_imm(
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

        let final_image = if app.settings.enhance.enabled {
            enhance_with_gpu(&resized, &enhancement_settings, app.gpu_enhancer.as_ref())
        } else {
            resized
        };
        let final_image = apply_mask_with_gpu(
            final_image,
            &app.settings.crop.shape,
            app.gpu_enhancer.as_ref(),
        );

        let mut output_filename = format!("{}_face_{:02}.{}", source_stem, face_idx + 1, ext);
        if let Some(suffix) = app.quality_suffix(det.quality) {
            output_filename = append_suffix_to_filename(&output_filename, suffix);
        }

        let output_path = output_dir.join(output_filename);

        let metadata_ctx = MetadataContext {
            source_path: source_path.map(|p| p.as_path()),
            crop_settings: Some(&app.settings.crop),
            detection_score: Some(det.detection.score),
            quality: Some(det.quality),
            quality_score: Some(det.quality_score),
        };

        let output_options = OutputOptions::from_crop_settings(&app.settings.crop);

        match save_dynamic_image(&final_image, &output_path, &output_options, &metadata_ctx) {
            Ok(_) => {
                exported += 1;
                info!(
                    "Exported face {} to {}",
                    face_idx + 1,
                    output_path.display()
                );
            }
            Err(err) => {
                failed += 1;
                warn!(
                    "Failed to save face {} to {}: {}",
                    face_idx + 1,
                    output_path.display(),
                    err
                );
            }
        }
    }

    if failed == 0 {
        app.show_success(format!("Exported {} face(s)", exported));
    } else {
        app.show_error(
            format!("Exported {} face(s)", exported),
            format!("{} failed", failed),
        );
    }
}

/// Starts batch export processing for all batch files.
pub fn start_batch_export(app: &mut YuNetApp) {
    use crate::BatchFileStatus;
    use rfd::FileDialog;

    if app.batch_files.is_empty() {
        app.show_error("Batch export failed", "No batch files loaded");
        return;
    }

    let detector = match app.detector.as_ref() {
        Some(d) => d.clone(),
        None => {
            app.show_error("Batch export failed", "No detector loaded");
            return;
        }
    };

    // Ask user to pick output directory
    let Some(output_dir) = FileDialog::new().pick_folder() else {
        return; // User cancelled
    };

    if let Err(err) = std::fs::create_dir_all(&output_dir) {
        app.show_error(
            "Batch export failed",
            format!("Failed to create output directory: {err}"),
        );
        return;
    }

    let config = Arc::new(BatchJobConfig {
        output_dir: output_dir.clone(),
        crop_settings: app.build_crop_settings(),
        crop_config: app.settings.crop.clone(),
        enhancement_settings: app.build_enhancement_settings(),
        enhance_enabled: app.settings.enhance.enabled,
        output_options: OutputOptions::from_crop_settings(&app.settings.crop),
        gpu_enhancer: app.gpu_enhancer.clone(),
    });

    let tasks: Vec<_> = app
        .batch_files
        .iter()
        .enumerate()
        .map(|(idx, file)| (idx, file.path.clone(), file.output_override.clone()))
        .collect();

    for file in &mut app.batch_files {
        file.status = BatchFileStatus::Pending;
    }

    let total = tasks.len();
    app.show_success(format!("Starting batch export of {} file(s)...", total));
    let results = run_batch_export(tasks, detector, config);

    for (idx, status) in results {
        if let Some(file) = app.batch_files.get_mut(idx) {
            file.status = status;
        }
    }

    let completed = app
        .batch_files
        .iter()
        .filter(|f| matches!(f.status, BatchFileStatus::Completed { .. }))
        .count();
    let failed = app
        .batch_files
        .iter()
        .filter(|f| matches!(f.status, BatchFileStatus::Failed { .. }))
        .count();

    app.show_success(format!(
        "Batch export complete: {} succeeded, {} failed",
        completed, failed
    ));
}
