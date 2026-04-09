//! Export and batch processing operations.

use crate::{
    BatchFileStatus, BatchJobConfig, JobMessage, YuNetApp,
    core::cache::{calculate_eyes_relative_to_crop, enhance_with_gpu},
    core::compositing::{apply_mask_with_gpu, composite_with_fill_color},
};

use fcs_core::{
    CropRegion, CropSettings as CoreCropSettings, Detection, YuNetDetector, calculate_crop_region,
    crop_face_from_image,
};
use fcs_utils::{
    MetadataContext, OutputOptions, append_suffix_to_filename,
    config::{BatchLogFormat, CropSettings as AppCropSettings, QualityAutomationSettings},
    gpu::BatchCropRequest,
    load_image,
    quality::{Quality, estimate_sharpness},
    save_dynamic_image,
};
use image::DynamicImage;
use log::{info, warn};
use rayon::prelude::*;
use rfd::FileDialog;
use serde::Serialize;
use std::{
    cmp::Ordering,
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, mpsc::Sender},
};

struct FaceCandidate {
    index: usize,
    quality: Quality,
    quality_score: f64,
    score: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct BatchLogRow {
    index: usize,
    path: String,
    error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    faces_detected: Option<usize>,
}

fn composite_export_background(image: DynamicImage, fill: fcs_core::FillColor) -> DynamicImage {
    let mut rgba = image.to_rgba8();
    composite_with_fill_color(&mut rgba, fill);
    DynamicImage::ImageRgba8(rgba)
}

fn prepare_crop_regions(
    source_image: &DynamicImage,
    detections: &[Detection],
    crop_settings: &CoreCropSettings,
) -> Vec<CropRegion> {
    detections
        .iter()
        .map(|detection| {
            calculate_crop_region(
                source_image.width(),
                source_image.height(),
                detection.bbox,
                crop_settings,
            )
        })
        .collect()
}

fn build_gpu_crop_requests(
    source_image: &DynamicImage,
    crop_regions: &[CropRegion],
    crop_settings: &CoreCropSettings,
) -> Vec<BatchCropRequest> {
    crop_regions
        .iter()
        .filter_map(|region| {
            region
                .in_bounds_rect(source_image.width(), source_image.height())
                .map(|(x, y, w, h)| BatchCropRequest {
                    source_x: x,
                    source_y: y,
                    source_width: w.max(1),
                    source_height: h.max(1),
                    output_width: crop_settings.output_width,
                    output_height: crop_settings.output_height,
                })
        })
        .collect()
}

fn maybe_gpu_crop_faces(
    source_image: &DynamicImage,
    crop_regions: &[CropRegion],
    config: &BatchJobConfig,
) -> Option<Vec<DynamicImage>> {
    if config.crop_settings.output_width == 0
        || config.crop_settings.output_height == 0
        || crop_regions.is_empty()
        || crop_regions.iter().any(CropRegion::requires_padding)
    {
        return None;
    }

    let cropper = config.gpu_cropper.as_ref()?;
    let requests = build_gpu_crop_requests(source_image, crop_regions, &config.crop_settings);

    match cropper.crop(source_image, &requests) {
        Ok(images) if images.len() == requests.len() => Some(images),
        Ok(images) => {
            warn!(
                "GPU cropper returned {} images for {} requests",
                images.len(),
                requests.len()
            );
            None
        }
        Err(err) => {
            warn!("GPU batch crop failed: {err}");
            None
        }
    }
}

fn build_processed_face(
    face_idx: usize,
    detection: &Detection,
    source_image: &DynamicImage,
    crop_region: &CropRegion,
    config: &BatchJobConfig,
    gpu_crop: Option<DynamicImage>,
) -> (Arc<DynamicImage>, FaceCandidate) {
    let resized = gpu_crop
        .unwrap_or_else(|| crop_face_from_image(source_image, detection, &config.crop_settings));

    let processed = if config.enhance_enabled {
        let eyes = if config.enhancement_settings.red_eye_removal {
            Some(calculate_eyes_relative_to_crop(
                &detection.landmarks,
                crop_region,
                source_image.width(),
                source_image.height(),
                config.crop_settings.output_width,
                config.crop_settings.output_height,
            ))
        } else {
            None
        };
        enhance_with_gpu(
            &resized,
            &config.enhancement_settings,
            config.gpu_enhancer.as_ref(),
            eyes.as_deref(),
        )
    } else {
        resized
    };

    let masked = apply_mask_with_gpu(
        processed,
        &config.crop_config.shape,
        config.crop_config.vignette_softness,
        config.crop_config.vignette_intensity,
        config.crop_config.vignette_color,
        config.gpu_enhancer.as_ref(),
    );
    let composited = composite_export_background(masked, config.crop_settings.fill_color);
    let final_image = Arc::new(composited);

    let (quality_score, quality) = estimate_sharpness(final_image.as_ref());
    (
        final_image,
        FaceCandidate {
            index: face_idx,
            quality,
            quality_score,
            score: detection.score,
        },
    )
}

fn select_export_candidate_indices(
    candidates: &[FaceCandidate],
    quality_rules: &QualityAutomationSettings,
) -> Option<Vec<usize>> {
    let best_quality = candidates.iter().map(|c| c.quality).max();
    if quality_rules.auto_skip_no_high_quality && best_quality != Some(Quality::High) {
        return None;
    }

    let mut selected: Vec<usize> = (0..candidates.len()).collect();
    if quality_rules.auto_select_best_face && selected.len() > 1 {
        let best_index = selected
            .iter()
            .copied()
            .max_by(|a, b| {
                candidates[*a]
                    .quality
                    .cmp(&candidates[*b].quality)
                    .then_with(|| {
                        candidates[*a]
                            .quality_score
                            .partial_cmp(&candidates[*b].quality_score)
                            .unwrap_or(Ordering::Equal)
                    })
            })
            .unwrap();
        selected.retain(|idx| *idx == best_index);
    }

    Some(selected)
}

fn export_output_extension(output_format: &str) -> String {
    let mut ext = output_format.to_string();
    if ext.is_empty() {
        ext = "png".to_string();
    }
    ext.to_ascii_lowercase()
}

fn build_export_output_path(
    output_dir: &Path,
    source_path: &Path,
    candidate: &FaceCandidate,
    quality_rules: &QualityAutomationSettings,
    output_format: &str,
    output_override: Option<&PathBuf>,
    multi_face: bool,
) -> PathBuf {
    let ext = export_output_extension(output_format);

    if let Some(custom) = output_override {
        return resolve_mapping_override_path(
            output_dir,
            custom,
            &ext,
            candidate.index,
            multi_face,
        );
    }

    let source_stem = source_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("face");
    let mut output_filename = format!("{}_face_{:02}.{}", source_stem, candidate.index + 1, ext);
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

    output_dir.join(output_filename)
}

fn save_processed_crop(
    image: &DynamicImage,
    output_path: &Path,
    output_options: &OutputOptions,
    source_path: &Path,
    crop_config: &AppCropSettings,
    candidate: &FaceCandidate,
) -> anyhow::Result<()> {
    let metadata_ctx = MetadataContext {
        source_path: Some(source_path),
        crop_settings: Some(crop_config),
        detection_score: Some(candidate.score),
        quality: Some(candidate.quality),
        quality_score: Some(candidate.quality_score),
    };

    save_dynamic_image(image, output_path, output_options, &metadata_ctx)
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

    let crop_regions = prepare_crop_regions(
        &source_image,
        &detection_output.detections,
        &config.crop_settings,
    );
    let gpu_crops = maybe_gpu_crop_faces(&source_image, &crop_regions, &config);

    let mut processed_faces: Vec<Arc<DynamicImage>> = Vec::with_capacity(faces_detected);
    let mut candidates = Vec::with_capacity(faces_detected);
    for (face_idx, detection) in detection_output.detections.iter().enumerate() {
        let gpu_crop = gpu_crops
            .as_ref()
            .and_then(|images| images.get(face_idx).cloned());
        let (final_image, candidate) = build_processed_face(
            face_idx,
            detection,
            &source_image,
            &crop_regions[face_idx],
            &config,
            gpu_crop,
        );
        processed_faces.push(final_image);
        candidates.push(candidate);
    }

    if candidates.is_empty() {
        return BatchFileStatus::Completed {
            faces_detected,
            faces_exported: 0,
        };
    }

    let quality_rules = &config.crop_config.quality_rules;
    let Some(export_indices) = select_export_candidate_indices(&candidates, quality_rules) else {
        warn!(
            "Skipping {} - no high-quality faces detected",
            path.display()
        );
        return BatchFileStatus::Completed {
            faces_detected,
            faces_exported: 0,
        };
    };

    let multi_face = export_indices.len() > 1;

    let mut faces_exported = 0;
    for candidate_index in export_indices {
        let candidate = &candidates[candidate_index];
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

        let Some(final_image) = processed_faces.get(candidate.index).cloned() else {
            continue;
        };

        let output_path = build_export_output_path(
            &config.output_dir,
            path.as_path(),
            candidate,
            quality_rules,
            &config.crop_config.output_format,
            output_override.as_ref(),
            multi_face,
        );

        if let Some(parent) = output_path.parent()
            && let Err(err) = std::fs::create_dir_all(parent)
        {
            warn!(
                "Failed to create output directory {}: {err}",
                parent.display()
            );
        }

        match save_processed_crop(
            final_image.as_ref(),
            &output_path,
            &config.output_options,
            path.as_path(),
            &config.crop_config,
            candidate,
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

/// Resolves the output path for a mapped file.
///
/// Security: absolute paths and `..` traversal components from the mapping
/// data are stripped so the output always stays inside `output_dir`.
fn resolve_mapping_override_path(
    output_dir: &Path,
    override_target: &Path,
    ext: &str,
    face_index: usize,
    multi_face: bool,
) -> PathBuf {
    let cleaned_ext = ext.trim_start_matches('.').to_string();

    // Strip traversal components so the path is always relative.
    let safe_target = sanitize_override_path(override_target);

    let rel_parent = safe_target.parent().unwrap_or_else(|| Path::new(""));
    let parent = output_dir.join(rel_parent);

    let base_name = safe_target
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
    final_path.push(format!("{final_base}.{cleaned_ext}"));
    final_path
}

/// Strip path traversal components and root prefixes so the result is always
/// a relative path that stays inside the output directory.
fn sanitize_override_path(p: &Path) -> PathBuf {
    use std::path::Component;
    p.components()
        .filter(|c| matches!(c, Component::Normal(_)))
        .collect()
}

fn build_batch_log_rows(
    tasks: &[(usize, PathBuf, Option<PathBuf>)],
    results: &[(usize, BatchFileStatus)],
) -> Vec<BatchLogRow> {
    results
        .iter()
        .filter_map(|(idx, status)| {
            let file_path = tasks
                .get(*idx)
                .map(|(_, p, _)| p.display().to_string())
                .unwrap_or_default();

            match status {
                BatchFileStatus::Failed { error } => Some(BatchLogRow {
                    index: *idx,
                    path: file_path,
                    error: error.clone(),
                    faces_detected: None,
                }),
                BatchFileStatus::Completed {
                    faces_detected,
                    faces_exported: 0,
                } => Some(BatchLogRow {
                    index: *idx,
                    path: file_path,
                    error: if *faces_detected == 0 {
                        "No faces detected".to_string()
                    } else {
                        "Faces detected but skipped (quality checks)".to_string()
                    },
                    faces_detected: Some(*faces_detected),
                }),
                _ => None,
            }
        })
        .collect()
}

fn format_batch_log_json(rows: &[BatchLogRow]) -> std::io::Result<Vec<u8>> {
    serde_json::to_string_pretty(rows)
        .map(|s| s.into_bytes())
        .map_err(std::io::Error::other)
}

fn escape_csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn format_batch_log_csv(rows: &[BatchLogRow]) -> std::io::Result<Vec<u8>> {
    let mut wtr = Vec::new();
    writeln!(&mut wtr, "index,path,error,faces_detected")?;
    for row in rows {
        let faces_count = row
            .faces_detected
            .map(|count| count.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        writeln!(
            &mut wtr,
            "{},{},{},{}",
            row.index,
            escape_csv_field(&row.path),
            escape_csv_field(&row.error),
            faces_count
        )?;
    }
    Ok(wtr)
}

/// Maximum number of images processed concurrently in a batch to prevent OOM.
/// Each image can require ~100 MB of staging + intermediate buffers, so
/// capping this avoids exhausting system memory on large batches.
const MAX_BATCH_PARALLELISM: usize = 4;

/// Runs batch export processing in parallel.
pub fn run_batch_export(
    tasks: Vec<(usize, PathBuf, Option<PathBuf>)>,
    detector: Arc<YuNetDetector>,
    config: Arc<BatchJobConfig>,
    tx: Option<Sender<JobMessage>>,
) -> Vec<(usize, BatchFileStatus)> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(MAX_BATCH_PARALLELISM)
        .build()
        .expect("failed to build batch thread pool");

    pool.install(|| {
        tasks
            .into_par_iter()
            .map(|(idx, path, override_path)| {
                let status = run_batch_job(detector.clone(), path, config.clone(), override_path);
                if let Some(tx) = &tx {
                    let _ = tx.send(JobMessage::BatchProgress {
                        index: idx,
                        status: status.clone(),
                    });
                }
                (idx, status)
            })
            .collect()
    })
}

/// Exports selected faces from the preview to disk.
pub fn export_selected_faces(app: &mut YuNetApp) {
    if app.selected_faces.is_empty() {
        app.show_error("Export failed", "No faces selected for export");
        return;
    }

    let Some(source_image) = app.preview.source_image.as_ref() else {
        app.show_error("Export failed", "No image loaded");
        return;
    };
    let source_image = source_image.as_ref();

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

        let mut detection_for_crop = det.detection.clone();
        detection_for_crop.bbox = det.active_bbox();
        let resized = crop_face_from_image(source_image, &detection_for_crop, &crop_settings);

        let final_image = if app.settings.enhance.enabled {
            let eyes = if enhancement_settings.red_eye_removal {
                let landmarks = &det.detection.landmarks;
                let bbox = det.active_bbox();
                // We need the crop region. But here we only have settings.
                // We need to re-calculate crop region.
                // crop_face_from_image calls calculate_crop_region internally if we just pass settings?
                // No, crop_face_from_image calls calculate_crop_region.
                // We need it here for eyes.
                let region = calculate_crop_region(
                    source_image.width(),
                    source_image.height(),
                    bbox,
                    &crop_settings,
                );
                Some(calculate_eyes_relative_to_crop(
                    landmarks,
                    &region,
                    source_image.width(),
                    source_image.height(),
                    crop_settings.output_width,
                    crop_settings.output_height,
                ))
            } else {
                None
            };
            enhance_with_gpu(
                &resized,
                &enhancement_settings,
                app.gpu_enhancer.as_ref(),
                eyes.as_deref(),
            )
        } else {
            resized
        };
        let final_image = apply_mask_with_gpu(
            final_image,
            &app.settings.crop.shape,
            app.settings.crop.vignette_softness,
            app.settings.crop.vignette_intensity,
            app.settings.crop.vignette_color,
            app.gpu_enhancer.as_ref(),
        );
        let final_image = composite_export_background(final_image, app.settings.crop.fill_color);

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
        batch_logging: app.settings.batch_logging.clone(),
        gpu_enhancer: app.gpu_enhancer.clone(),
        gpu_cropper: app.gpu_batch_cropper.clone(),
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

    let tx = app.job_tx.clone();
    std::thread::spawn(move || {
        // Clone tasks so we can use the original list for path lookups during logging
        let tasks_for_processing = tasks.clone();
        let results = run_batch_export(
            tasks_for_processing,
            detector,
            config.clone(),
            Some(tx.clone()),
        );

        let completed = results
            .iter()
            .filter(|(_, s)| matches!(s, BatchFileStatus::Completed { .. }))
            .count();
        let failed_items: Vec<_> = results
            .iter()
            .filter(|(_, s)| matches!(s, BatchFileStatus::Failed { .. }))
            .collect();
        let failed = failed_items.len();

        let log_rows = build_batch_log_rows(&tasks, &results);

        if config.batch_logging.enabled && !log_rows.is_empty() {
            let log_filename = match config.batch_logging.format {
                BatchLogFormat::Json => "batch_failures.json",
                BatchLogFormat::Csv => "batch_failures.csv",
            };
            let path = config.output_dir.join(log_filename);

            let write_result = match config.batch_logging.format {
                BatchLogFormat::Json => format_batch_log_json(&log_rows),
                BatchLogFormat::Csv => format_batch_log_csv(&log_rows),
            };

            match write_result {
                Ok(bytes) => {
                    if let Err(e) = std::fs::write(&path, bytes) {
                        warn!("Failed to write batch log to {}: {}", path.display(), e);
                    } else {
                        info!("Batch failure log written to {}", path.display());
                    }
                }
                Err(e) => {
                    warn!("Failed to format batch log: {}", e);
                }
            }
        }

        let _ = tx.send(JobMessage::BatchComplete { completed, failed });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use fcs_core::FillColor;
    use fcs_utils::config::QualityAutomationSettings;
    use fcs_utils::quality::Quality;
    use image::{DynamicImage, Rgba, RgbaImage};
    use serde_json::Value;
    use std::path::PathBuf;

    #[test]
    fn export_background_compositing_applies_fill_color_to_transparent_pixels() {
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(1, 1, Rgba([0, 0, 0, 0])));
        let fill = FillColor::opaque(12, 34, 56);

        let composited = composite_export_background(image, fill).to_rgba8();
        let pixel = composited.get_pixel(0, 0);

        assert_eq!(*pixel, Rgba([12, 34, 56, 255]));
    }

    #[test]
    fn export_background_compositing_handles_semitransparent_fill() {
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(1, 1, Rgba([0, 0, 0, 0])));
        let fill = FillColor {
            red: 100,
            green: 150,
            blue: 200,
            alpha: 128,
        };

        let composited = composite_export_background(image, fill).to_rgba8();
        let pixel = composited.get_pixel(0, 0);

        assert_eq!(*pixel, Rgba([50, 75, 100, 128]));
    }

    #[test]
    fn resolve_mapping_override_path_sanitizes_absolute_and_parent_components() {
        let output = resolve_mapping_override_path(
            Path::new("/exports"),
            Path::new("/tmp/../nested/custom.name.jpg"),
            ".png",
            0,
            false,
        );

        assert_eq!(
            output,
            PathBuf::from("/exports")
                .join("tmp")
                .join("nested")
                .join("custom.name.png")
        );
    }

    #[test]
    fn resolve_mapping_override_path_appends_face_suffix_for_multi_face_exports() {
        let output = resolve_mapping_override_path(
            Path::new("/exports"),
            Path::new("gallery/portrait.jpg"),
            "webp",
            2,
            true,
        );

        assert_eq!(
            output,
            PathBuf::from("/exports")
                .join("gallery")
                .join("portrait_face3.webp")
        );
    }

    #[test]
    fn build_batch_log_rows_includes_failed_and_zero_export_statuses() {
        let tasks = vec![
            (0, PathBuf::from("a.jpg"), None),
            (1, PathBuf::from("b.jpg"), None),
            (2, PathBuf::from("c.jpg"), None),
            (3, PathBuf::from("d.jpg"), None),
        ];
        let results = vec![
            (
                0,
                BatchFileStatus::Completed {
                    faces_detected: 2,
                    faces_exported: 2,
                },
            ),
            (
                1,
                BatchFileStatus::Failed {
                    error: "disk full".to_string(),
                },
            ),
            (
                2,
                BatchFileStatus::Completed {
                    faces_detected: 0,
                    faces_exported: 0,
                },
            ),
            (
                3,
                BatchFileStatus::Completed {
                    faces_detected: 3,
                    faces_exported: 0,
                },
            ),
        ];

        let rows = build_batch_log_rows(&tasks, &results);

        assert_eq!(
            rows,
            vec![
                BatchLogRow {
                    index: 1,
                    path: "b.jpg".to_string(),
                    error: "disk full".to_string(),
                    faces_detected: None,
                },
                BatchLogRow {
                    index: 2,
                    path: "c.jpg".to_string(),
                    error: "No faces detected".to_string(),
                    faces_detected: Some(0),
                },
                BatchLogRow {
                    index: 3,
                    path: "d.jpg".to_string(),
                    error: "Faces detected but skipped (quality checks)".to_string(),
                    faces_detected: Some(3),
                },
            ]
        );
    }

    #[test]
    fn format_batch_log_json_omits_faces_detected_for_failures() {
        let rows = vec![
            BatchLogRow {
                index: 1,
                path: "b.jpg".to_string(),
                error: "disk full".to_string(),
                faces_detected: None,
            },
            BatchLogRow {
                index: 2,
                path: "c.jpg".to_string(),
                error: "No faces detected".to_string(),
                faces_detected: Some(0),
            },
        ];

        let bytes = format_batch_log_json(&rows).unwrap();
        let parsed: Value = serde_json::from_slice(&bytes).unwrap();
        let items = parsed.as_array().unwrap();

        assert!(items[0].get("faces_detected").is_none());
        assert_eq!(items[1]["faces_detected"].as_u64(), Some(0));
    }

    #[test]
    fn format_batch_log_csv_escapes_commas_and_quotes() {
        let rows = vec![BatchLogRow {
            index: 7,
            path: r#"folder,"quoted",name.jpg"#.to_string(),
            error: r#"failed, reason "oops""#.to_string(),
            faces_detected: Some(2),
        }];

        let csv = String::from_utf8(format_batch_log_csv(&rows).unwrap()).unwrap();

        assert!(csv.starts_with("index,path,error,faces_detected\n"));
        assert!(csv.contains(r#"7,"folder,""quoted"",name.jpg","failed, reason ""oops""",2"#));
    }

    fn candidate(index: usize, quality: Quality, quality_score: f64) -> FaceCandidate {
        FaceCandidate {
            index,
            quality,
            quality_score,
            score: 0.95 - index as f32 * 0.1,
        }
    }

    #[test]
    fn select_export_candidate_indices_prefers_best_quality_then_score() {
        let candidates = vec![
            candidate(0, Quality::Medium, 0.9),
            candidate(1, Quality::High, 0.2),
            candidate(2, Quality::High, 0.8),
        ];
        let rules = QualityAutomationSettings {
            auto_select_best_face: true,
            min_quality: None,
            auto_skip_no_high_quality: false,
            quality_suffix: false,
        };

        let selected = select_export_candidate_indices(&candidates, &rules).unwrap();
        assert_eq!(selected, vec![2]);
    }

    #[test]
    fn select_export_candidate_indices_skips_when_no_high_quality_face_exists() {
        let candidates = vec![
            candidate(0, Quality::Low, 0.2),
            candidate(1, Quality::Medium, 0.7),
        ];
        let rules = QualityAutomationSettings {
            auto_select_best_face: false,
            min_quality: None,
            auto_skip_no_high_quality: true,
            quality_suffix: false,
        };

        assert_eq!(select_export_candidate_indices(&candidates, &rules), None);
    }

    #[test]
    fn build_export_output_path_adds_quality_suffix_for_standard_exports() {
        let output = build_export_output_path(
            Path::new("/exports"),
            Path::new("/images/portrait.jpg"),
            &candidate(1, Quality::High, 0.9),
            &QualityAutomationSettings {
                auto_select_best_face: false,
                min_quality: None,
                auto_skip_no_high_quality: false,
                quality_suffix: true,
            },
            "PNG",
            None,
            false,
        );

        assert_eq!(
            output,
            PathBuf::from("/exports").join("portrait_face_02_highq.png")
        );
    }
}
