//! Command-line interface for running YuNet face detection.

use std::{
    fs::{self, File},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::SystemTime,
};

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use image::GenericImageView;
use log::{debug, info, warn};
use rayon::prelude::*;
use yunet_core::{PostprocessConfig, PreprocessConfig, crop_face_from_image};
use yunet_utils::{
    MetadataContext, OutputOptions, append_suffix_to_filename, config::AppSettings,
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
    // mark image as started
    images_processed.fetch_add(1, Ordering::Relaxed);
    let image_path = &target.source;
    let override_target = target.output_override.clone();
    if let Some(row) = target.mapping_row {
        debug!("Mapping row {} -> {}", row, image_path.display());
    }

    let output = match detector.detect_path(image_path) {
        Ok(out) => out,
        Err(err) => {
            warn!("Failed to process {}: {err}", image_path.display());
            return None;
        }
    };
    // accumulate face count
    faces_detected.fetch_add(output.detections.len(), Ordering::Relaxed);

    info!(
        "{} -> {} detection(s)",
        image_path.display(),
        output.detections.len()
    );

    // Try to open the source image once for optional use (annotation, quality estimation, etc.)
    let img_opt = match image::open(image_path) {
        Ok(img) => Some(img),
        Err(e) => {
            warn!("Failed to open image {}: {}", image_path.display(), e);
            None
        }
    };

    let annotated_path = if let Some(dir) = annotate_dir.as_ref() {
        match annotate::annotate_image(image_path, &output.detections, dir) {
            Ok(path) => {
                info!("Annotated image saved to {}", path.display());
                Some(path.display().to_string())
            }
            Err(err) => {
                warn!("Failed to annotate {}: {err}", image_path.display());
                None
            }
        }
    } else {
        None
    };

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

    // Build detection records, including optional quality estimates per-detection
    let mut detection_records: Vec<DetectionRecord> = Vec::with_capacity(output.detections.len());
    for det in &output.detections {
        let mut rec = DetectionRecord::from(det);
        if let Some(img) = img_opt.as_ref() {
            // Attempt to crop the bbox region and estimate sharpness
            let bbox = &det.bbox;
            let (img_w, img_h) = img.dimensions();
            let x1 = bbox.x.clamp(0.0, img_w as f32);
            let y1 = bbox.y.clamp(0.0, img_h as f32);
            let x2 = (bbox.x + bbox.width).clamp(0.0, img_w as f32);
            let y2 = (bbox.y + bbox.height).clamp(0.0, img_h as f32);
            let w = (x2 - x1).max(1.0).round() as u32;
            let h = (y2 - y1).max(1.0).round() as u32;
            // clone a small buffer for safety
            let tmp = img.clone();
            if x1.round() as u32 + w <= tmp.width() && y1.round() as u32 + h <= tmp.height() {
                let sub =
                    image::imageops::crop_imm(&tmp, x1.round() as u32, y1.round() as u32, w, h)
                        .to_image();
                let dynsub = image::DynamicImage::ImageRgba8(sub);
                let (score, q) = estimate_sharpness(&dynsub);
                rec.quality_score = Some(score);
                rec.quality = Some(format!("{:?}", q));
            }
        }
        detection_records.push(rec);
    }

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
    use yunet_utils::Quality;

    let output_options = OutputOptions::from_crop_settings(&settings.crop);
    let core_settings = build_core_crop_settings(&settings.crop);

    struct ProcessedCrop {
        index: usize,
        image: image::DynamicImage,
        quality: Quality,
        quality_score: f64,
        score: f32,
    }

    let mut processed: Vec<ProcessedCrop> = Vec::with_capacity(detections.len());

    let gpu_crops = runtime.crop_faces_gpu(img, detections, &core_settings);
    if let Some(gpu_images) = gpu_crops {
        for ((idx, det), mut crop_img) in detections.iter().enumerate().zip(gpu_images.into_iter())
        {
            if let Some(enh) = enhancement_settings.as_ref() {
                crop_img = runtime.enhance(&crop_img, enh);
            }
            let (quality_score, quality) = estimate_sharpness(&crop_img);
            crop_img = runtime.apply_shape_mask(&crop_img, &settings.crop.shape);
            processed.push(ProcessedCrop {
                index: idx,
                image: crop_img,
                quality,
                quality_score,
                score: det.score,
            });
        }
    }

    if processed.is_empty() {
        for (idx, det) in detections.iter().enumerate() {
            let mut crop_img = crop_face_from_image(img, det, &core_settings);
            if let Some(enh) = enhancement_settings.as_ref() {
                crop_img = runtime.enhance(&crop_img, enh);
            }
            let (quality_score, quality) = estimate_sharpness(&crop_img);
            crop_img = runtime.apply_shape_mask(&crop_img, &settings.crop.shape);
            processed.push(ProcessedCrop {
                index: idx,
                image: crop_img,
                quality,
                quality_score,
                score: det.score,
            });
        }
    }

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
            exports.retain(|c| c.index == target);
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
        let qualities: Vec<(Quality, f64)> = exports
            .iter()
            .map(|c| (c.quality, c.quality_score))
            .collect();
        if let Some(best_rel) = quality_filter.select_best_index(&qualities) {
            let best_idx = exports[best_rel].index;
            exports.retain(|c| c.index == best_idx);
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

        let mut ext = settings.crop.output_format.clone();
        if ext.is_empty() {
            ext = "png".to_string();
        }
        let ext = ext.to_ascii_lowercase();

        let out_path = if let Some(custom) = override_target {
            resolve_override_output_path(out_dir, custom, &ext, crop.index, multi_face)
        } else {
            let stem = image_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("image");
            let mut out_name = if let Some(tmpl) = args.naming_template.as_ref() {
                let ts = SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let mut name = tmpl.clone();
                name = name.replace("{original}", stem);
                name = name.replace("{index}", &(crop.index + 1).to_string());
                name = name.replace("{width}", &settings.crop.output_width.to_string());
                name = name.replace("{height}", &settings.crop.output_height.to_string());
                name = name.replace("{ext}", &ext);
                name = name.replace("{timestamp}", &ts.to_string());
                if !tmpl.contains("{ext}") {
                    format!("{}.{}", name, ext)
                } else {
                    name
                }
            } else {
                format!("{}_face{}.{}", stem, crop.index + 1, ext)
            };

            if let Some(suffix) = quality_filter.suffix_for(crop.quality) {
                out_name = append_suffix_to_filename(&out_name, suffix);
            }

            out_dir.join(&out_name)
        };

        if let Some(parent) = out_path.parent()
            && let Err(err) = fs::create_dir_all(parent)
        {
            warn!("Failed to create directory {}: {err}", parent.display());
        }

        let metadata_ctx = MetadataContext {
            source_path: Some(image_path),
            crop_settings: Some(&settings.crop),
            detection_score: Some(crop.score),
            quality: Some(crop.quality),
            quality_score: Some(crop.quality_score),
        };

        match save_dynamic_image(&crop.image, &out_path, &output_options, &metadata_ctx) {
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
