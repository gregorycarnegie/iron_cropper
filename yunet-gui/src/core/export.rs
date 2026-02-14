//! Export and batch processing operations.

use crate::{
    BatchFileStatus, BatchJobConfig, JobMessage, YuNetApp,
    core::cache::{calculate_eyes_relative_to_crop, enhance_with_gpu},
    core::compositing::{apply_mask_with_gpu, composite_with_fill_color},
};

use image::DynamicImage;
use log::{info, warn};
use rayon::prelude::*;
use rfd::FileDialog;
use std::{
    cmp::Ordering,
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, mpsc::Sender},
};
use yunet_core::{YuNetDetector, calculate_crop_region, crop_face_from_image};
use yunet_utils::{
    MetadataContext, OutputOptions, append_suffix_to_filename,
    config::BatchLogFormat,
    gpu::BatchCropRequest,
    load_image,
    quality::{Quality, estimate_sharpness},
    save_dynamic_image,
};

struct FaceCandidate {
    index: usize,
    quality: Quality,
    quality_score: f64,
    score: f32,
}

fn composite_export_background(image: DynamicImage, fill: yunet_core::FillColor) -> DynamicImage {
    let mut rgba = image.to_rgba8();
    composite_with_fill_color(&mut rgba, fill);
    DynamicImage::ImageRgba8(rgba)
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

    let crop_regions: Vec<_> = detection_output
        .detections
        .iter()
        .map(|detection| {
            calculate_crop_region(
                source_image.width(),
                source_image.height(),
                detection.bbox,
                &config.crop_settings,
            )
        })
        .collect();

    let gpu_allowed = crop_regions.iter().all(|region| !region.requires_padding());
    let gpu_crops = if config.crop_settings.output_width > 0
        && config.crop_settings.output_height > 0
        && !crop_regions.is_empty()
        && gpu_allowed
    {
        config.gpu_cropper.as_ref().and_then(|cropper| {
            let requests: Vec<BatchCropRequest> = crop_regions
                .iter()
                .filter_map(|region| {
                    region
                        .in_bounds_rect(source_image.width(), source_image.height())
                        .map(|(x, y, w, h)| BatchCropRequest {
                            source_x: x,
                            source_y: y,
                            source_width: w.max(1),
                            source_height: h.max(1),
                            output_width: config.crop_settings.output_width,
                            output_height: config.crop_settings.output_height,
                        })
                })
                .collect();
            match cropper.crop(&source_image, &requests) {
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
        })
    } else {
        None
    };

    let mut processed_faces: Vec<Arc<DynamicImage>> = Vec::with_capacity(faces_detected);
    let mut candidates = Vec::with_capacity(faces_detected);
    for (face_idx, detection) in detection_output.detections.iter().enumerate() {
        let resized = if let Some(images) = gpu_crops.as_ref() {
            images.get(face_idx).cloned().unwrap_or_else(|| {
                crop_face_from_image(&source_image, detection, &config.crop_settings)
            })
        } else {
            crop_face_from_image(&source_image, detection, &config.crop_settings)
        };

        let processed = if config.enhance_enabled {
            let eyes = if config.enhancement_settings.red_eye_removal {
                let landmarks = &detection.landmarks;
                let region = &crop_regions[face_idx];
                Some(calculate_eyes_relative_to_crop(
                    landmarks,
                    region,
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
        candidates.push(FaceCandidate {
            index: face_idx,
            quality,
            quality_score,
            score: detection.score,
        });
        processed_faces.push(final_image);
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

        let Some(final_image) = processed_faces.get(candidate.index).cloned() else {
            continue;
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
            final_image.as_ref(),
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
    final_path.push(final_base);
    final_path.set_extension(cleaned_ext);
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

        // Include "failed" items OR items where 0 faces were exported (no detection or filtered out)
        let logged_items: Vec<_> = results
            .iter()
            .filter(|(_, s)| match s {
                BatchFileStatus::Failed { .. } => true,
                BatchFileStatus::Completed { faces_exported, .. } => *faces_exported == 0,
                _ => false,
            })
            .collect();

        if config.batch_logging.enabled && !logged_items.is_empty() {
            let log_filename = match config.batch_logging.format {
                BatchLogFormat::Json => "batch_failures.json",
                BatchLogFormat::Csv => "batch_failures.csv",
            };
            let path = config.output_dir.join(log_filename);

            let write_result = match config.batch_logging.format {
                BatchLogFormat::Json => serde_json::to_string_pretty(
                    &logged_items
                        .iter()
                        .map(|(idx, status)| {
                            let file_path = tasks
                                .get(*idx)
                                .map(|(_, p, _)| p.display().to_string())
                                .unwrap_or_default();
                            let error_msg = match status {
                                BatchFileStatus::Failed { error } => error.clone(),
                                BatchFileStatus::Completed {
                                    faces_detected,
                                    faces_exported: 0,
                                } => {
                                    if *faces_detected == 0 {
                                        "No faces detected".to_string()
                                    } else {
                                        "Faces detected but skipped (quality checks)".to_string()
                                    }
                                }
                                _ => "Unknown error".to_string(),
                            };

                            let mut json_obj = serde_json::json!({
                                "index": idx,
                                "path": file_path,
                                "error": error_msg
                            });

                            if let BatchFileStatus::Completed { faces_detected, .. } = status {
                                json_obj["faces_detected"] = serde_json::json!(faces_detected);
                            }

                            json_obj
                        })
                        .collect::<Vec<_>>(),
                )
                .map(|s| s.into_bytes())
                .map_err(std::io::Error::other),
                BatchLogFormat::Csv => {
                    let mut wtr = Vec::new();
                    writeln!(&mut wtr, "index,path,error,faces_detected").unwrap();
                    for (idx, status) in logged_items {
                        let file_path = tasks
                            .get(*idx)
                            .map(|(_, p, _)| p.display().to_string())
                            .unwrap_or_default();
                        let (error_msg, faces_count) = match status {
                            BatchFileStatus::Failed { error } => (error.clone(), "N/A".to_string()),
                            BatchFileStatus::Completed {
                                faces_detected,
                                faces_exported: 0,
                            } => {
                                let msg = if *faces_detected == 0 {
                                    "No faces detected"
                                } else {
                                    "Faces detected but skipped (quality checks)"
                                };
                                (msg.to_string(), faces_detected.to_string())
                            }
                            _ => ("Unknown error".to_string(), "N/A".to_string()),
                        };

                        // CSV escaping helper
                        let escape_csv = |s: &str| -> String {
                            if s.contains(',') || s.contains('"') {
                                format!("\"{}\"", s.replace('"', "\"\""))
                            } else {
                                s.to_string()
                            }
                        };

                        let clean_path = escape_csv(&file_path);
                        let clean_err = escape_csv(&error_msg);

                        writeln!(
                            &mut wtr,
                            "{},{},{},{}",
                            idx, clean_path, clean_err, faces_count
                        )
                        .unwrap();
                    }
                    Ok(wtr)
                }
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
    use super::composite_export_background;
    use image::{DynamicImage, Rgba, RgbaImage};
    use yunet_core::FillColor;

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
}
