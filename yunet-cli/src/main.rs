//! Command-line interface for running YuNet face detection.

use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::SystemTime,
};

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use image::{DynamicImage, GenericImageView};
use log::{debug, info, warn};
use rayon::prelude::*;
use yunet_core::{
    BoundingBox, Detection, DetectionOutput, PostprocessConfig, PreprocessConfig,
    crop_face_from_image,
};
use yunet_utils::{
    MetadataContext, OutputOptions, Quality, append_suffix_to_filename, config::AppSettings,
    configure_telemetry, estimate_sharpness, init_logging, normalize_path, save_dynamic_image,
};

mod annotate;
mod args;
mod benchmark;
mod color;
mod config;
mod detector;
mod enhancement;
mod gpu;
mod input;
mod quality;
mod types;
mod webcam;

use args::DetectArgs;
use config::{apply_cli_overrides, build_core_crop_settings, load_settings};
use detector::build_cli_detector;
use enhancement::build_enhancement_settings;
use gpu::init_cli_gpu_runtime;
use input::{collect_mapping_targets, collect_standard_targets, resolve_override_output_path};
use quality::build_quality_filter;
use types::{DetectionRecord, ImageDetections};

#[derive(Clone, Debug)]
struct ProcessedCrop {
    index: usize,
    image: DynamicImage,
    quality: Quality,
    quality_score: f64,
    score: f32,
}

fn main() -> Result<()> {
    let args = DetectArgs::parse();

    let mut settings = load_settings(args.config.as_ref())?;
    apply_cli_overrides(&mut settings, &args);

    configure_telemetry(
        settings.telemetry.enabled,
        settings.telemetry.level_filter(),
    );
    init_logging(log::LevelFilter::Info)?;

    if settings.telemetry.enabled {
        info!(
            "Telemetry logging enabled (level={:?})",
            settings.telemetry.level_filter()
        );
    }

    let model_path = normalize_path(&args.model)?;
    let annotate_dir = if let Some(dir) = args.annotate.as_ref() {
        fs::create_dir_all(dir)
            .with_context(|| format!("failed to create annotation directory {}", dir.display()))?;
        Some(normalize_path(dir)?)
    } else {
        None
    };

    // Build a centralized quality filter using resolved automation settings so the same
    // policy is used for cropping, batch export, and future GUI wiring.
    let quality_filter = build_quality_filter(&settings.crop.quality_rules);
    let gpu_runtime = Arc::new(init_cli_gpu_runtime(&settings)?);

    let preprocess_config: PreprocessConfig = settings.input.into();
    let postprocess_config: PostprocessConfig = (&settings.detection).into();
    let input_size = preprocess_config.input_size;

    // Check if webcam mode is enabled
    if args.webcam {
        info!(
            "Loading YuNet model from {} at resolution {}x{}",
            model_path.display(),
            input_size.width,
            input_size.height
        );
        let prefer_gpu_inference = settings.gpu.enabled && settings.gpu.inference;
        let detector = build_cli_detector(
            &model_path,
            &preprocess_config,
            &postprocess_config,
            gpu_runtime.as_ref(),
            prefer_gpu_inference,
        )?;
        let detector = Arc::new(detector);
        let settings = Arc::new(settings);
        let quality_filter = Arc::new(quality_filter);
        let enhancement_settings = build_enhancement_settings(&args).map(Arc::new);

        return webcam::run_webcam_mode(
            &args,
            detector,
            settings,
            gpu_runtime,
            quality_filter,
            enhancement_settings,
        );
    }

    let processing_items = if let Some(mapping_file) = args.mapping_file.as_ref() {
        collect_mapping_targets(mapping_file, &args)?
    } else {
        let input_arg = args
            .input
            .as_ref()
            .ok_or_else(|| anyhow!("--input is required when --mapping-file is not provided"))?;
        let input_path = normalize_path(input_arg)?;
        collect_standard_targets(&input_path)?
    };
    if processing_items.is_empty() {
        anyhow::bail!("no images were queued for processing");
    }

    if args.benchmark_preprocess {
        benchmark::run_preprocess_benchmark(
            &processing_items,
            &preprocess_config,
            gpu_runtime.context(),
        )?;
        return Ok(());
    }

    info!(
        "Loading YuNet model from {} at resolution {}x{}",
        model_path.display(),
        input_size.width,
        input_size.height
    );
    let prefer_gpu_inference = settings.gpu.enabled && settings.gpu.inference;
    let detector = build_cli_detector(
        &model_path,
        &preprocess_config,
        &postprocess_config,
        gpu_runtime.as_ref(),
        prefer_gpu_inference,
    )?;

    if args.mapping_file.is_some() && !args.crop {
        info!(
            "Mapping loaded without --crop; output overrides will be applied when cropping is executed."
        );
    }

    info!("Processing {} target(s)...", processing_items.len());

    // Wrap detector in Arc for thread-safe shared access
    let detector = Arc::new(detector);
    let annotate_dir = Arc::new(annotate_dir);

    // Prepare crop output directory if requested
    let crop_enabled = args.crop;
    let crop_output_dir = if crop_enabled {
        if let Some(dir) = args.output_dir.as_ref() {
            fs::create_dir_all(dir)
                .with_context(|| format!("failed to create output dir {}", dir.display()))?;
            Some(normalize_path(dir)?)
        } else {
            anyhow::bail!("--crop requires --output-dir to be specified");
        }
    } else {
        None
    };
    let crop_output_dir = Arc::new(crop_output_dir);
    let shared_settings = Arc::new(settings.clone());
    let quality_filter = Arc::new(quality_filter);
    let enhancement_settings = build_enhancement_settings(&args).map(Arc::new);

    // Process images in parallel
    // Progress counters (shared across parallel tasks)
    let images_processed = Arc::new(AtomicUsize::new(0));
    let faces_detected = Arc::new(AtomicUsize::new(0));
    let crops_saved = Arc::new(AtomicUsize::new(0));
    let crops_skipped_quality = Arc::new(AtomicUsize::new(0));

    let results: Vec<ImageDetections> = processing_items
        .par_iter()
        .filter_map(|target| {
            process_single_image(
                target,
                &detector,
                &annotate_dir,
                &shared_settings,
                &quality_filter,
                &enhancement_settings,
                &gpu_runtime,
                &args,
                crop_enabled,
                &crop_output_dir,
                &images_processed,
                &faces_detected,
                &crops_saved,
                &crops_skipped_quality,
            )
        })
        .collect();

    if results.is_empty() {
        anyhow::bail!("all detections failed; cannot produce output");
    }

    if let Some(json_path) = args.json.as_ref() {
        let parent = json_path.parent();
        if let Some(dir) = parent {
            fs::create_dir_all(dir)
                .with_context(|| format!("failed to create directory {}", dir.display()))?;
        }
        let file = File::create(json_path)
            .with_context(|| format!("failed to create {}", json_path.display()))?;
        serde_json::to_writer_pretty(file, &results).with_context(|| {
            format!("failed to write detection JSON to {}", json_path.display())
        })?;
        info!("Wrote detections to {}", json_path.display());
    } else {
        let json =
            serde_json::to_string_pretty(&results).context("failed to serialize detections")?;
        println!("{json}");
    }

    // Summary progress report
    info!(
        "Summary: images_processed={} faces_detected={} crops_saved={} crops_skipped_quality={}",
        images_processed.load(Ordering::Relaxed),
        faces_detected.load(Ordering::Relaxed),
        crops_saved.load(Ordering::Relaxed),
        crops_skipped_quality.load(Ordering::Relaxed)
    );
    // Also print a concise single-line summary for interactive users
    println!(
        "images_processed={} faces_detected={} crops_saved={} crops_skipped_quality={}",
        images_processed.load(Ordering::Relaxed),
        faces_detected.load(Ordering::Relaxed),
        crops_saved.load(Ordering::Relaxed),
        crops_skipped_quality.load(Ordering::Relaxed)
    );

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_single_image(
    target: &input::ProcessingItem,
    detector: &Arc<yunet_core::YuNetDetector>,
    annotate_dir: &Arc<Option<std::path::PathBuf>>,
    settings: &Arc<AppSettings>,
    quality_filter: &Arc<yunet_utils::QualityFilter>,
    enhancement_settings: &Option<Arc<yunet_utils::EnhancementSettings>>,
    runtime: &Arc<gpu::CliGpuRuntime>,
    args: &DetectArgs,
    crop_enabled: bool,
    crop_output_dir: &Arc<Option<std::path::PathBuf>>,
    images_processed: &Arc<AtomicUsize>,
    faces_detected: &Arc<AtomicUsize>,
    crops_saved: &Arc<AtomicUsize>,
    crops_skipped_quality: &Arc<AtomicUsize>,
) -> Option<ImageDetections> {
    images_processed.fetch_add(1, Ordering::Relaxed);
    let image_path = &target.source;
    let override_target = target.output_override.clone();
    if let Some(row) = target.mapping_row {
        debug!("Mapping row {} -> {}", row, image_path.display());
    }

    let output = detect_image(detector, image_path)?;
    faces_detected.fetch_add(output.detections.len(), Ordering::Relaxed);

    info!(
        "{} -> {} detection(s)",
        image_path.display(),
        output.detections.len()
    );

    let img_opt = load_source_image(image_path);
    let annotated_path = annotate_image_if_requested(image_path, &output.detections, annotate_dir);

    match (crop_enabled, crop_output_dir.as_ref(), img_opt.as_ref()) {
        (true, Some(out_dir), Some(img)) => {
            process_crops(
                img,
                image_path,
                &output.detections,
                settings,
                quality_filter,
                enhancement_settings,
                runtime,
                args,
                out_dir,
                override_target.as_ref(),
                crops_saved,
                crops_skipped_quality,
            );
        }
        (true, Some(_), None) => {
            warn!(
                "Cannot crop {} because the source image failed to load",
                image_path.display()
            );
        }
        _ => {}
    }

    let detection_records = build_detection_records(&output.detections, img_opt.as_ref());

    Some(ImageDetections {
        image: image_path.display().to_string(),
        detections: detection_records,
        annotated: annotated_path,
    })
}

#[allow(clippy::too_many_arguments)]
fn process_crops(
    img: &image::DynamicImage,
    image_path: &Path,
    detections: &[yunet_core::Detection],
    settings: &Arc<AppSettings>,
    quality_filter: &Arc<yunet_utils::QualityFilter>,
    enhancement_settings: &Option<Arc<yunet_utils::EnhancementSettings>>,
    runtime: &Arc<gpu::CliGpuRuntime>,
    args: &DetectArgs,
    out_dir: &Path,
    override_target: Option<&std::path::PathBuf>,
    crops_saved: &Arc<AtomicUsize>,
    crops_skipped_quality: &Arc<AtomicUsize>,
) {
    let output_options = OutputOptions::from_crop_settings(&settings.crop);
    let core_settings = build_core_crop_settings(&settings.crop);
    let processed = generate_processed_crops(
        img,
        detections,
        settings,
        &core_settings,
        enhancement_settings.as_ref(),
        runtime.as_ref(),
    );

    if processed.is_empty() {
        debug!(
            "No crops generated for {} (no detections)",
            image_path.display()
        );
        return;
    }

    let best_quality = processed.iter().map(|c| c.quality).max();

    if quality_filter.should_skip_image(best_quality) {
        info!(
            "Skipping exports for {} because no face reached high quality",
            image_path.display()
        );
        crops_skipped_quality.fetch_add(processed.len().max(1), Ordering::Relaxed);
        return;
    }

    let mut exports = processed;

    if let Some(fidx) = args.face_index {
        if fidx == 0 {
            warn!(
                "--face-index is 1-based; ignoring 0 for {}",
                image_path.display()
            );
            exports.clear();
        } else {
            let target = fidx - 1;
            let available = exports.len();
            exports = retain_requested_face(exports, target);
            if exports.is_empty() {
                warn!(
                    "Requested face index {} not found for {} ({} detections)",
                    fidx,
                    image_path.display(),
                    available
                );
            }
        }
    } else if quality_filter.auto_select && exports.len() > 1 {
        if let Some(best_rel) = select_best_quality_index(&exports, quality_filter) {
            let best_idx = exports[best_rel].index;
            exports = retain_requested_face(exports, best_idx);
            debug!(
                "Auto-selected face {} for {} based on quality {:?}",
                best_idx + 1,
                image_path.display(),
                exports.first().map(|c| c.quality)
            );
        }
    }

    let multi_face = exports.len() > 1;
    for crop in exports.into_iter() {
        if quality_filter.should_skip(crop.quality) {
            info!(
                "Skipping crop for {} face {} due to {:?} quality (score {:.1})",
                image_path.display(),
                crop.index + 1,
                crop.quality,
                crop.quality_score
            );
            crops_skipped_quality.fetch_add(1, Ordering::Relaxed);
            continue;
        }

        let ext = normalized_output_extension(&settings.crop.output_format);
        let timestamp = if args.naming_template.is_some() {
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        } else {
            0
        };

        let out_path = if let Some(custom) = override_target {
            resolve_override_output_path(out_dir, custom, &ext, crop.index, multi_face)
        } else {
            build_crop_output_path(
                out_dir,
                image_path,
                crop.index,
                settings.crop.output_width,
                settings.crop.output_height,
                &ext,
                args.naming_template.as_deref(),
                quality_filter.suffix_for(crop.quality),
                timestamp,
            )
        };

        if let Some(parent) = out_path.parent()
            && let Err(err) = fs::create_dir_all(parent)
        {
            warn!("Failed to create directory {}: {err}", parent.display());
        }

        match save_processed_crop(&crop, &out_path, image_path, settings, &output_options) {
            Ok(_) => {
                info!("Saved crop to {}", out_path.display());
                crops_saved.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                warn!("Failed to export crop {}: {e:?}", out_path.display());
            }
        }
    }
}

fn detect_image(
    detector: &Arc<yunet_core::YuNetDetector>,
    image_path: &Path,
) -> Option<DetectionOutput> {
    match detector.detect_path(image_path) {
        Ok(out) => Some(out),
        Err(err) => {
            warn!("Failed to process {}: {err}", image_path.display());
            None
        }
    }
}

fn load_source_image(image_path: &Path) -> Option<DynamicImage> {
    match image::open(image_path) {
        Ok(img) => Some(img),
        Err(e) => {
            warn!("Failed to open image {}: {}", image_path.display(), e);
            None
        }
    }
}

fn annotate_image_if_requested(
    image_path: &Path,
    detections: &[Detection],
    annotate_dir: &Arc<Option<PathBuf>>,
) -> Option<String> {
    let Some(dir) = annotate_dir.as_ref() else {
        return None;
    };

    match annotate::annotate_image(image_path, detections, dir) {
        Ok(path) => {
            info!("Annotated image saved to {}", path.display());
            Some(path.display().to_string())
        }
        Err(err) => {
            warn!("Failed to annotate {}: {err}", image_path.display());
            None
        }
    }
}

fn detection_crop_rect(
    bbox: &BoundingBox,
    image_width: u32,
    image_height: u32,
) -> Option<(u32, u32, u32, u32)> {
    let x1 = bbox.x.clamp(0.0, image_width as f32);
    let y1 = bbox.y.clamp(0.0, image_height as f32);
    let x2 = (bbox.x + bbox.width).clamp(0.0, image_width as f32);
    let y2 = (bbox.y + bbox.height).clamp(0.0, image_height as f32);
    let w = (x2 - x1).max(1.0).round() as u32;
    let h = (y2 - y1).max(1.0).round() as u32;
    let left = x1.round() as u32;
    let top = y1.round() as u32;

    if left + w <= image_width && top + h <= image_height {
        Some((left, top, w, h))
    } else {
        None
    }
}

fn detection_quality(image: &DynamicImage, bbox: &BoundingBox) -> Option<(f64, String)> {
    let (image_width, image_height) = image.dimensions();
    let (left, top, width, height) = detection_crop_rect(bbox, image_width, image_height)?;
    let tmp = image.clone();
    let sub = image::imageops::crop_imm(&tmp, left, top, width, height).to_image();
    let dynsub = image::DynamicImage::ImageRgba8(sub);
    let (score, quality) = estimate_sharpness(&dynsub);
    Some((score, format!("{:?}", quality)))
}

fn build_detection_record(det: &Detection, image: Option<&DynamicImage>) -> DetectionRecord {
    let mut record = DetectionRecord::from(det);
    if let Some(img) = image
        && let Some((score, quality)) = detection_quality(img, &det.bbox)
    {
        record.quality_score = Some(score);
        record.quality = Some(quality);
    }
    record
}

fn build_detection_records(
    detections: &[Detection],
    image: Option<&DynamicImage>,
) -> Vec<DetectionRecord> {
    detections
        .iter()
        .map(|det| build_detection_record(det, image))
        .collect()
}

fn build_processed_crop(
    index: usize,
    detection: &Detection,
    mut crop_img: DynamicImage,
    settings: &AppSettings,
    enhancement_settings: Option<&Arc<yunet_utils::EnhancementSettings>>,
    runtime: &gpu::CliGpuRuntime,
) -> ProcessedCrop {
    if let Some(enh) = enhancement_settings {
        crop_img = runtime.enhance(&crop_img, enh);
    }

    let (quality_score, quality) = estimate_sharpness(&crop_img);
    crop_img = runtime.apply_shape_mask(
        &crop_img,
        &settings.crop.shape,
        settings.crop.vignette_softness,
        settings.crop.vignette_intensity,
        settings.crop.vignette_color,
    );

    ProcessedCrop {
        index,
        image: crop_img,
        quality,
        quality_score,
        score: detection.score,
    }
}

fn generate_processed_crops(
    img: &DynamicImage,
    detections: &[Detection],
    settings: &AppSettings,
    core_settings: &yunet_core::CropSettings,
    enhancement_settings: Option<&Arc<yunet_utils::EnhancementSettings>>,
    runtime: &gpu::CliGpuRuntime,
) -> Vec<ProcessedCrop> {
    let mut processed = Vec::with_capacity(detections.len());

    if let Some(gpu_images) = runtime.crop_faces_gpu(img, detections, core_settings) {
        for ((idx, det), crop_img) in detections.iter().enumerate().zip(gpu_images.into_iter()) {
            processed.push(build_processed_crop(
                idx,
                det,
                crop_img,
                settings,
                enhancement_settings,
                runtime,
            ));
        }
    }

    if processed.is_empty() {
        for (idx, det) in detections.iter().enumerate() {
            let crop_img = crop_face_from_image(img, det, core_settings);
            processed.push(build_processed_crop(
                idx,
                det,
                crop_img,
                settings,
                enhancement_settings,
                runtime,
            ));
        }
    }

    processed
}

fn retain_requested_face(processed: Vec<ProcessedCrop>, face_index: usize) -> Vec<ProcessedCrop> {
    processed
        .into_iter()
        .filter(|crop| crop.index == face_index)
        .collect()
}

fn select_best_quality_index(
    processed: &[ProcessedCrop],
    quality_filter: &yunet_utils::QualityFilter,
) -> Option<usize> {
    let qualities: Vec<(Quality, f64)> = processed
        .iter()
        .map(|crop| (crop.quality, crop.quality_score))
        .collect();
    quality_filter.select_best_index(&qualities)
}

fn normalized_output_extension(output_format: &str) -> String {
    let ext = if output_format.is_empty() {
        "png".to_string()
    } else {
        output_format.to_string()
    };
    ext.to_ascii_lowercase()
}

fn build_crop_filename(
    source_stem: &str,
    crop_index: usize,
    output_width: u32,
    output_height: u32,
    ext: &str,
    naming_template: Option<&str>,
    quality_suffix: Option<&str>,
    timestamp: u64,
) -> String {
    let mut out_name = if let Some(tmpl) = naming_template {
        let mut name = tmpl.to_string();
        name = name.replace("{original}", source_stem);
        name = name.replace("{index}", &(crop_index + 1).to_string());
        name = name.replace("{width}", &output_width.to_string());
        name = name.replace("{height}", &output_height.to_string());
        name = name.replace("{ext}", ext);
        name = name.replace("{timestamp}", &timestamp.to_string());
        if !tmpl.contains("{ext}") {
            format!("{}.{}", name, ext)
        } else {
            name
        }
    } else {
        format!("{}_face{}.{}", source_stem, crop_index + 1, ext)
    };

    if let Some(suffix) = quality_suffix {
        out_name = append_suffix_to_filename(&out_name, suffix);
    }

    out_name
}

fn build_crop_output_path(
    out_dir: &Path,
    source_path: &Path,
    crop_index: usize,
    output_width: u32,
    output_height: u32,
    ext: &str,
    naming_template: Option<&str>,
    quality_suffix: Option<&str>,
    timestamp: u64,
) -> PathBuf {
    let source_stem = source_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");
    out_dir.join(build_crop_filename(
        source_stem,
        crop_index,
        output_width,
        output_height,
        ext,
        naming_template,
        quality_suffix,
        timestamp,
    ))
}

fn save_processed_crop(
    crop: &ProcessedCrop,
    out_path: &Path,
    image_path: &Path,
    settings: &AppSettings,
    output_options: &OutputOptions,
) -> Result<()> {
    let metadata_ctx = MetadataContext {
        source_path: Some(image_path),
        crop_settings: Some(&settings.crop),
        detection_score: Some(crop.score),
        quality: Some(crop.quality),
        quality_score: Some(crop.quality_score),
    };

    save_dynamic_image(&crop.image, out_path, output_options, &metadata_ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};
    use tempfile::tempdir;
    use yunet_core::Landmark;
    use yunet_utils::QualityFilter;

    fn sample_image(width: u32, height: u32) -> DynamicImage {
        let mut image = RgbaImage::new(width, height);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            let value = ((x + y) % 255) as u8;
            *pixel = Rgba([value, value.wrapping_add(1), value.wrapping_add(2), 255]);
        }
        DynamicImage::ImageRgba8(image)
    }

    fn sample_detection(x: f32, y: f32, width: f32, height: f32, score: f32) -> Detection {
        Detection {
            bbox: BoundingBox {
                x,
                y,
                width,
                height,
            },
            landmarks: [
                Landmark { x: 1.0, y: 1.0 },
                Landmark { x: 2.0, y: 1.0 },
                Landmark { x: 1.5, y: 1.5 },
                Landmark { x: 1.25, y: 2.0 },
                Landmark { x: 1.75, y: 2.0 },
            ],
            score,
        }
    }

    fn sample_processed_crop(index: usize, quality: Quality, quality_score: f64) -> ProcessedCrop {
        ProcessedCrop {
            index,
            image: sample_image(2, 2),
            quality,
            quality_score,
            score: 0.5,
        }
    }

    #[test]
    fn detection_crop_rect_clamps_to_image_bounds() {
        let rect = detection_crop_rect(
            &BoundingBox {
                x: -3.4,
                y: 2.4,
                width: 8.2,
                height: 12.0,
            },
            10,
            10,
        );
        assert_eq!(rect, Some((0, 2, 5, 8)));
    }

    #[test]
    fn build_detection_record_populates_quality_for_valid_crop() {
        let image = sample_image(10, 10);
        let detection = sample_detection(1.0, 2.0, 4.0, 4.0, 0.95);

        let record = build_detection_record(&detection, Some(&image));
        assert_eq!(record.score, 0.95);
        assert!(record.quality_score.is_some());
        assert!(record.quality.is_some());
    }

    #[test]
    fn build_detection_record_leaves_quality_empty_without_image() {
        let detection = sample_detection(1.0, 2.0, 4.0, 4.0, 0.95);

        let record = build_detection_record(&detection, None);
        assert_eq!(record.score, 0.95);
        assert_eq!(record.quality_score, None);
        assert_eq!(record.quality, None);
    }

    #[test]
    fn retain_requested_face_keeps_only_matching_index() {
        let crops = vec![
            sample_processed_crop(0, Quality::Low, 0.2),
            sample_processed_crop(1, Quality::Medium, 0.5),
            sample_processed_crop(2, Quality::High, 0.8),
        ];

        let filtered = retain_requested_face(crops, 1);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].index, 1);
        assert_eq!(filtered[0].quality, Quality::Medium);
    }

    #[test]
    fn select_best_quality_index_prefers_quality_then_score() {
        let crops = vec![
            sample_processed_crop(0, Quality::High, 0.1),
            sample_processed_crop(1, Quality::Medium, 1.0),
            sample_processed_crop(2, Quality::High, 0.9),
        ];
        let filter = QualityFilter {
            min_quality: None,
            auto_select: true,
            fallback: None,
            auto_skip_no_high: false,
            suffix_enabled: false,
        };

        let selected = select_best_quality_index(&crops, &filter);
        assert_eq!(selected, Some(2));
    }

    #[test]
    fn build_crop_filename_applies_template_and_quality_suffix() {
        let name = build_crop_filename(
            "portrait",
            1,
            512,
            640,
            "jpg",
            Some("{original}_{index}_{width}x{height}_{timestamp}"),
            Some("_highq"),
            1234,
        );

        assert_eq!(name, "portrait_2_512x640_1234_highq.jpg");
    }

    #[test]
    fn build_crop_filename_appends_extension_when_template_omits_it() {
        let name = build_crop_filename(
            "portrait",
            0,
            512,
            640,
            "png",
            Some("{original}_face{index}"),
            None,
            0,
        );

        assert_eq!(name, "portrait_face1.png");
    }

    #[test]
    fn normalized_output_extension_defaults_to_png() {
        assert_eq!(normalized_output_extension(""), "png");
        assert_eq!(normalized_output_extension("JPG"), "jpg");
    }

    #[test]
    fn build_crop_output_path_joins_directory_and_filename() {
        let dir = tempdir().unwrap();
        let out = build_crop_output_path(
            dir.path(),
            Path::new("/tmp/portrait.jpg"),
            0,
            512,
            640,
            "png",
            None,
            Some("_lowq"),
            0,
        );

        assert_eq!(
            out.file_name().and_then(|s| s.to_str()),
            Some("portrait_face1_lowq.png")
        );
    }
}
