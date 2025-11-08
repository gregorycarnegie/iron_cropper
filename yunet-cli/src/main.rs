//! Command-line interface for running YuNet face detection.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result, anyhow};
use clap::{ArgAction, Parser};
use image::GenericImageView;
use log::{debug, info, warn};
use rayon::prelude::*;
use serde::Serialize;
use std::sync::atomic::{AtomicUsize, Ordering};
use walkdir::WalkDir;
use yunet_core::{BoundingBox, Detection, PostprocessConfig, PreprocessConfig, YuNetDetector};
use yunet_core::{CropSettings, PositioningMode, crop_face_from_image, preset_by_name};
use yunet_utils::{
    EnhancementSettings, MetadataContext, OutputOptions, Quality, QualityFilter,
    append_suffix_to_filename, apply_enhancements, apply_shape_mask_dynamic,
    config::{
        AppSettings, CropSettings as ConfigCropSettings, MetadataMode, QualityAutomationSettings,
        ResizeQuality, default_settings_path,
    },
    configure_telemetry, estimate_sharpness, init_logging,
    mapping::ColumnSelector,
    mapping::MappingFormat,
    mapping::MappingReadOptions,
    mapping::detect_format as detect_mapping_format,
    mapping::load_mapping_entries,
    normalize_path, save_dynamic_image,
};

/// Run YuNet face detection over images or directories.
#[derive(Debug, Parser)]
#[command(author, version, about)]
struct DetectArgs {
    /// Path to an image file or a directory containing images.
    #[arg(short, long, required_unless_present = "mapping_file")]
    input: Option<PathBuf>,

    /// Path to the YuNet ONNX model.
    #[arg(
        short,
        long,
        default_value = "models/face_detection_yunet_2023mar_640.onnx"
    )]
    model: PathBuf,

    /// Optional settings JSON. Defaults to `config/gui_settings.json` when present, otherwise built-in parameters.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Enable telemetry timing logs (defaults to settings file).
    #[arg(long, action = ArgAction::SetTrue)]
    telemetry: bool,

    /// Override telemetry logging level (error, warn, info, debug, trace).
    #[arg(long, value_name = "LEVEL")]
    telemetry_level: Option<String>,

    /// Override input width (pixels).
    #[arg(long)]
    width: Option<u32>,

    /// Override input height (pixels).
    #[arg(long)]
    height: Option<u32>,

    /// Resize quality mode: `quality` (Triangle) or `speed` (fast Nearest).
    #[arg(long, value_name = "MODE")]
    resize_quality: Option<ResizeQuality>,

    /// Override score threshold.
    #[arg(long)]
    score_threshold: Option<f32>,

    /// Override NMS threshold.
    #[arg(long)]
    nms_threshold: Option<f32>,

    /// Override top_k limit.
    #[arg(long)]
    top_k: Option<usize>,

    /// Write detections to a JSON file instead of stdout.
    #[arg(long)]
    json: Option<PathBuf>,

    /// Directory to write annotated images with bounding boxes overlaid.
    #[arg(long)]
    annotate: Option<PathBuf>,

    /// Enable cropping mode: save cropped face images for each detection.
    #[arg(long)]
    crop: bool,

    /// Output directory for cropped face images (required when --crop is used).
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Preset name for output size (e.g., LinkedIn, Passport, Instagram). If set, overrides --output-width/--output-height.
    #[arg(long)]
    preset: Option<String>,

    /// Output width for crops (pixels).
    #[arg(long)]
    output_width: Option<u32>,

    /// Output height for crops (pixels).
    #[arg(long)]
    output_height: Option<u32>,

    /// Face height percentage in the output image (default: 70).
    #[arg(long, default_value_t = 70.0)]
    face_height_pct: f32,

    /// Positioning mode for crop: center, rule_of_thirds, custom
    #[arg(long, default_value = "center")]
    positioning_mode: String,

    /// Horizontal offset for custom positioning (fraction -1.0..1.0).
    #[arg(long, default_value_t = 0.0)]
    horizontal_offset: f32,

    /// Vertical offset for custom positioning (fraction -1.0..1.0).
    #[arg(long, default_value_t = 0.0)]
    vertical_offset: f32,

    /// Output image format for saved crops: png, jpeg, webp
    #[arg(long, default_value = "png")]
    output_format: String,

    /// JPEG quality when saving as JPEG (1-100).
    #[arg(long, default_value_t = 90u8)]
    jpeg_quality: u8,

    /// PNG compression strategy: fast, default, best, or numeric level 0-9.
    #[arg(long)]
    png_compression: Option<String>,

    /// WebP quality when saving as WebP (0-100).
    #[arg(long)]
    webp_quality: Option<u8>,

    /// Automatically detect output format from the file extension.
    #[arg(long)]
    auto_detect_format: Option<bool>,

    /// Select face index (1-based) to save only a specific face. Default: all faces.
    #[arg(long)]
    face_index: Option<usize>,

    /// Minimum quality to save crops (low, medium, high). If set, crops below this level are skipped.
    #[arg(long)]
    min_quality: Option<String>,

    /// Shortcut to skip low-quality crops (equivalent to `--min-quality medium`).
    #[arg(long)]
    skip_low_quality: Option<bool>,

    /// Automatically select the highest-quality face per image. Use `--auto-select-best=false` to disable.
    #[arg(long)]
    auto_select_best: Option<bool>,

    /// Skip exporting when no high-quality faces are detected. Use `--skip-no-high-quality=false` to require manual review.
    #[arg(long)]
    skip_no_high_quality: Option<bool>,

    /// Append a quality suffix (e.g., `_highq`) to exported filenames. Use `--quality-suffix=false` to disable.
    #[arg(long)]
    quality_suffix: Option<bool>,

    /// Metadata handling mode: preserve, strip, or custom.
    #[arg(long)]
    metadata_mode: Option<String>,

    /// Include crop settings metadata in output files.
    #[arg(long)]
    metadata_include_crop: Option<bool>,

    /// Include quality scores in output metadata.
    #[arg(long)]
    metadata_include_quality: Option<bool>,

    /// Custom metadata tags in KEY=VALUE form (may be repeated).
    #[arg(long = "metadata-tag")]
    metadata_tags: Vec<String>,

    /// Apply image enhancement pipeline (unsharp mask, contrast, exposure)
    /// to each crop before quality estimation and saving.
    #[arg(long)]
    enhance: Option<bool>,

    /// Unsharp mask amount. If provided, overrides preset/default.
    #[arg(long)]
    unsharp_amount: Option<f32>,

    /// Unsharp mask blur radius in pixels.
    #[arg(long)]
    unsharp_radius: Option<f32>,

    /// Contrast multiplier (0.5-2.0, 1.0 = unchanged).
    #[arg(long)]
    enhance_contrast: Option<f32>,

    /// Exposure adjustment in stops (-2.0..=2.0).
    #[arg(long)]
    enhance_exposure: Option<f32>,
    /// Additional brightness offset (integer steps applied after exposure)
    #[arg(long)]
    enhance_brightness: Option<i32>,

    /// Saturation multiplier (1.0 = unchanged, <1 desaturate, >1 increase)
    #[arg(long)]
    enhance_saturation: Option<f32>,

    /// Apply gray-world auto color correction to crops when --enhance is set
    #[arg(long)]
    enhance_auto_color: Option<bool>,

    /// Additional sharpening strength (added to unsharp-amount)
    #[arg(long)]
    enhance_sharpness: Option<f32>,

    /// Skin smoothing strength (0.0-1.0, uses bilateral filter)
    #[arg(long)]
    enhance_skin_smooth: Option<f32>,

    /// Enable automated red-eye removal
    #[arg(long)]
    enhance_red_eye_removal: Option<bool>,

    /// Enable background blur (portrait mode effect)
    #[arg(long)]
    enhance_background_blur: Option<bool>,

    /// Naming template for output crop files. Variables: {original}, {index}, {width}, {height}, {ext}, {timestamp}
    #[arg(long)]
    naming_template: Option<String>,
    /// Enhancement preset to apply when --enhance is set. Options: natural, vivid, professional
    #[arg(long)]
    enhancement_preset: Option<String>,

    /// Optional mapping file that lists source images and desired output names.
    #[arg(long = "mapping-file")]
    mapping_file: Option<PathBuf>,
    /// Column containing source paths inside the mapping file (name or zero-based index).
    #[arg(long = "mapping-source-col")]
    mapping_source_col: Option<String>,
    /// Column containing output names inside the mapping file (name or zero-based index).
    #[arg(long = "mapping-output-col")]
    mapping_output_col: Option<String>,
    /// Whether the mapping file contains a header row (defaults to true for CSV/Excel/Parquet).
    #[arg(long = "mapping-has-headers")]
    mapping_has_headers: Option<bool>,
    /// Optional delimiter for CSV/TSV mappings (defaults to comma).
    #[arg(long = "mapping-delimiter")]
    mapping_delimiter: Option<char>,
    /// Optional sheet name when loading from Excel.
    #[arg(long = "mapping-sheet")]
    mapping_sheet: Option<String>,
    /// Optional explicit mapping format (csv, excel, parquet, sqlite).
    #[arg(long = "mapping-format")]
    mapping_format: Option<String>,
    /// SQLite table to read when using .db/.sqlite files (defaults to the first table).
    #[arg(long = "mapping-sql-table")]
    mapping_sql_table: Option<String>,
    /// Custom SQL query to run when using SQLite mapping files.
    #[arg(long = "mapping-sql-query")]
    mapping_sql_query: Option<String>,
}

/// A serializable representation of a single detection.
#[derive(Debug, Serialize)]
struct DetectionRecord {
    score: f32,
    bbox: [f32; 4],
    landmarks: [[f32; 2]; 5],
    #[serde(skip_serializing_if = "Option::is_none")]
    quality_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<String>,
}

/// A serializable representation of all detections for a single image.
#[derive(Debug, Serialize)]
struct ImageDetections {
    image: String,
    detections: Vec<DetectionRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    annotated: Option<String>,
}

#[derive(Clone)]
struct ProcessingItem {
    source: PathBuf,
    output_override: Option<PathBuf>,
    mapping_row: Option<usize>,
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

    let preprocess_config: PreprocessConfig = settings.input.into();
    let postprocess_config: PostprocessConfig = (&settings.detection).into();
    let input_size = preprocess_config.input_size;

    info!(
        "Loading YuNet model from {} at resolution {}x{}",
        model_path.display(),
        input_size.width,
        input_size.height
    );
    let detector = YuNetDetector::new(&model_path, preprocess_config, postprocess_config)?;

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

    if args.mapping_file.is_some() && !args.crop {
        info!(
            "Mapping loaded without --crop; output overrides will be applied when cropping is executed."
        );
    }

    info!("Processing {} target(s)...", processing_items.len());

    // Wrap detector in Arc<Mutex<>> for thread-safe shared access
    let detector = Arc::new(Mutex::new(detector));
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
    let enhancement_settings = Arc::new(build_enhancement_settings(&args));

    // Process images in parallel
    // Progress counters (shared across parallel tasks)
    let images_processed = Arc::new(AtomicUsize::new(0));
    let faces_detected = Arc::new(AtomicUsize::new(0));
    let crops_saved = Arc::new(AtomicUsize::new(0));
    let crops_skipped_quality = Arc::new(AtomicUsize::new(0));

    let results: Vec<ImageDetections> = processing_items
        .par_iter()
        .filter_map(|target| {
            // mark image as started
            images_processed.fetch_add(1, Ordering::Relaxed);
            let detector = Arc::clone(&detector);
            let annotate_dir = Arc::clone(&annotate_dir);
            let settings = Arc::clone(&shared_settings);
            let quality_filter = Arc::clone(&quality_filter);
            let enhancement_settings = Arc::clone(&enhancement_settings);
            let image_path = &target.source;
            let override_target = target.output_override.clone();
            if let Some(row) = target.mapping_row {
                debug!("Mapping row {} -> {}", row, image_path.display());
            }

            // Lock the detector for this thread
            let output = match detector.lock().unwrap().detect_path(image_path) {
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
                match annotate_image(image_path, &output.detections, dir) {
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
                    let output_options = OutputOptions::from_crop_settings(&settings.crop);
                    let core_settings = build_core_crop_settings(&settings.crop);

                    struct ProcessedCrop {
                        index: usize,
                        image: image::DynamicImage,
                        quality: Quality,
                        quality_score: f64,
                        score: f32,
                    }

                    let mut processed: Vec<ProcessedCrop> =
                        Vec::with_capacity(output.detections.len());
                    for (idx, det) in output.detections.iter().enumerate() {
                        let mut crop_img = crop_face_from_image(img, det, &core_settings);
                        if let Some(enh) = enhancement_settings.as_ref() {
                            crop_img = apply_enhancements(&crop_img, enh);
                        }
                        let (quality_score, quality) = estimate_sharpness(&crop_img);
                        apply_shape_mask_dynamic(&mut crop_img, &settings.crop.shape);
                        processed.push(ProcessedCrop {
                            index: idx,
                            image: crop_img,
                            quality,
                            quality_score,
                            score: det.score,
                        });
                    }

                    if processed.is_empty() {
                        debug!(
                            "No crops generated for {} (no detections)",
                            image_path.display()
                        );
                    } else {
                        let best_quality = processed.iter().map(|c| c.quality).max();

                        if quality_filter.should_skip_image(best_quality) {
                            info!(
                                "Skipping exports for {} because no face reached high quality",
                                image_path.display()
                            );
                            crops_skipped_quality
                                .fetch_add(processed.len().max(1), Ordering::Relaxed);
                        } else {
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
                                if let Some(best_rel) =
                                    quality_filter.select_best_index(&qualities)
                                {
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

                                let out_path = if let Some(custom) = override_target.as_ref() {
                                    resolve_override_output_path(
                                        out_dir,
                                        custom,
                                        &ext,
                                        crop.index,
                                        multi_face,
                                    )
                                } else {
                                    let stem = image_path
                                        .file_stem()
                                        .and_then(|s| s.to_str())
                                        .unwrap_or("image");
                                    let mut out_name =
                                        if let Some(tmpl) = args.naming_template.as_ref() {
                                            use std::time::{SystemTime, UNIX_EPOCH};
                                            let ts = SystemTime::now()
                                                .duration_since(UNIX_EPOCH)
                                                .map(|d| d.as_secs())
                                                .unwrap_or(0);
                                            let mut name = tmpl.clone();
                                            name = name.replace("{original}", stem);
                                            name = name.replace(
                                                "{index}",
                                                &(crop.index + 1).to_string(),
                                            );
                                            name = name.replace(
                                                "{width}",
                                                &settings.crop.output_width.to_string(),
                                            );
                                            name = name.replace(
                                                "{height}",
                                                &settings.crop.output_height.to_string(),
                                            );
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
                                    warn!(
                                        "Failed to create directory {}: {err}",
                                        parent.display()
                                    );
                                }

                                let metadata_ctx = MetadataContext {
                                    source_path: Some(image_path.as_path()),
                                    crop_settings: Some(&settings.crop),
                                    detection_score: Some(crop.score),
                                    quality: Some(crop.quality),
                                    quality_score: Some(crop.quality_score),
                                };

                                match save_dynamic_image(
                                    &crop.image,
                                    &out_path,
                                    &output_options,
                                    &metadata_ctx,
                                ) {
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
                    }
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
                        let sub = image::imageops::crop_imm(&tmp, x1.round() as u32, y1.round() as u32, w, h).to_image();
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

/// Construct a `QualityFilter` from persistent automation settings.
fn build_quality_filter(settings: &QualityAutomationSettings) -> QualityFilter {
    let mut filter = QualityFilter::new(settings.min_quality);
    filter.auto_select = settings.auto_select_best_face;
    filter.auto_skip_no_high = settings.auto_skip_no_high_quality;
    filter.suffix_enabled = settings.quality_suffix;
    filter
}

/// Build EnhancementSettings from CLI args. Returns None when enhancements aren't enabled.
fn build_enhancement_settings(args: &DetectArgs) -> Option<EnhancementSettings> {
    // Only construct enhancement settings when `--enhance` is explicitly set to true
    if !args.enhance.unwrap_or(false) {
        return None;
    }

    // Base: defaults or preset
    let mut base = EnhancementSettings::default();
    if let Some(pname) = args.enhancement_preset.as_ref() {
        match pname.as_str() {
            "natural" => {
                base = EnhancementSettings {
                    auto_color: true,
                    exposure_stops: 0.1,
                    brightness: 0,
                    contrast: 1.1,
                    saturation: 1.05,
                    unsharp_amount: 0.6,
                    unsharp_radius: 1.0,
                    sharpness: 0.2,
                    skin_smooth_amount: 0.0,
                    skin_smooth_sigma_space: 3.0,
                    skin_smooth_sigma_color: 25.0,
                    red_eye_removal: false,
                    red_eye_threshold: 1.5,
                    background_blur: false,
                    background_blur_radius: 15.0,
                    background_blur_mask_size: 0.6,
                }
            }
            "vivid" => {
                base = EnhancementSettings {
                    auto_color: false,
                    exposure_stops: 0.3,
                    brightness: 10,
                    contrast: 1.25,
                    saturation: 1.3,
                    unsharp_amount: 0.9,
                    unsharp_radius: 1.2,
                    sharpness: 0.5,
                    skin_smooth_amount: 0.0,
                    skin_smooth_sigma_space: 3.0,
                    skin_smooth_sigma_color: 25.0,
                    red_eye_removal: false,
                    red_eye_threshold: 1.5,
                    background_blur: false,
                    background_blur_radius: 15.0,
                    background_blur_mask_size: 0.6,
                }
            }
            "professional" => {
                base = EnhancementSettings {
                    auto_color: true,
                    exposure_stops: 0.2,
                    brightness: 0,
                    contrast: 1.15,
                    saturation: 1.05,
                    unsharp_amount: 1.2,
                    unsharp_radius: 1.0,
                    sharpness: 0.8,
                    skin_smooth_amount: 0.0,
                    skin_smooth_sigma_space: 3.0,
                    skin_smooth_sigma_color: 25.0,
                    red_eye_removal: false,
                    red_eye_threshold: 1.5,
                    background_blur: false,
                    background_blur_radius: 15.0,
                    background_blur_mask_size: 0.6,
                }
            }
            other => warn!("unknown enhancement preset '{}', using defaults", other),
        }
    }

    // Apply explicit overrides if provided
    if let Some(v) = args.unsharp_amount {
        base.unsharp_amount = v;
    }
    if let Some(v) = args.unsharp_radius {
        base.unsharp_radius = v;
    }
    if let Some(v) = args.enhance_contrast {
        base.contrast = v;
    }
    if let Some(v) = args.enhance_exposure {
        base.exposure_stops = v;
    }
    if let Some(v) = args.enhance_brightness {
        base.brightness = v;
    }
    if let Some(v) = args.enhance_saturation {
        base.saturation = v;
    }
    if let Some(v) = args.enhance_auto_color {
        base.auto_color = v;
    }
    if let Some(v) = args.enhance_sharpness {
        base.sharpness = v;
    }
    if let Some(v) = args.enhance_skin_smooth {
        base.skin_smooth_amount = v;
    }
    if let Some(v) = args.enhance_red_eye_removal {
        base.red_eye_removal = v;
    }
    if let Some(v) = args.enhance_background_blur {
        base.background_blur = v;
    }

    Some(base)
}

#[allow(clippy::items_after_test_module)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_quality_filter_reflects_settings() {
        let settings = QualityAutomationSettings {
            auto_select_best_face: true,
            min_quality: Some(Quality::High),
            auto_skip_no_high_quality: true,
            quality_suffix: true,
        };

        let filter = build_quality_filter(&settings);
        assert_eq!(filter.min_quality, Some(Quality::High));
        assert!(filter.auto_select);
        assert!(filter.auto_skip_no_high);
        assert!(filter.suffix_enabled);
    }

    #[test]
    fn enhancement_preset_vivid_applies_defaults_and_overrides() {
        let args = DetectArgs {
            input: Some(PathBuf::from("image.png")),
            model: PathBuf::from("model.onnx"),
            config: None,
            telemetry: false,
            telemetry_level: None,
            width: None,
            height: None,
            resize_quality: None,
            score_threshold: None,
            nms_threshold: None,
            top_k: None,
            json: None,
            annotate: None,
            crop: false,
            output_dir: None,
            preset: None,
            output_width: None,
            output_height: None,
            face_height_pct: 70.0,
            positioning_mode: "center".to_string(),
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            output_format: "png".to_string(),
            jpeg_quality: 90u8,
            png_compression: None,
            webp_quality: None,
            auto_detect_format: None,
            face_index: None,
            min_quality: None,
            skip_low_quality: None,
            auto_select_best: None,
            skip_no_high_quality: None,
            quality_suffix: None,
            metadata_mode: None,
            metadata_include_crop: None,
            metadata_include_quality: None,
            metadata_tags: Vec::new(),
            enhance: Some(true),
            unsharp_amount: None,
            unsharp_radius: None,
            enhance_contrast: None,
            enhance_exposure: None,
            enhance_brightness: None,
            enhance_saturation: None,
            enhance_auto_color: None,
            enhance_sharpness: None,
            enhance_skin_smooth: None,
            enhance_red_eye_removal: None,
            enhance_background_blur: None,
            naming_template: None,
            enhancement_preset: Some("vivid".to_string()),
            mapping_file: None,
            mapping_source_col: None,
            mapping_output_col: None,
            mapping_has_headers: None,
            mapping_delimiter: None,
            mapping_sheet: None,
            mapping_format: None,
            mapping_sql_table: None,
            mapping_sql_query: None,
        };
        let enh = build_enhancement_settings(&args).expect("should build");
        assert_eq!(enh.unsharp_amount, 0.9);
        assert!((enh.saturation - 1.3).abs() < f32::EPSILON);
        assert!((enh.contrast - 1.25).abs() < f32::EPSILON);
        assert!((enh.exposure_stops - 0.3).abs() < f32::EPSILON);
        assert_eq!(enh.brightness, 10);
    }

    #[test]
    fn enhancement_preset_allows_explicit_override() {
        let args = DetectArgs {
            input: Some(PathBuf::from("image.png")),
            model: PathBuf::from("model.onnx"),
            config: None,
            telemetry: false,
            telemetry_level: None,
            width: None,
            height: None,
            resize_quality: None,
            score_threshold: None,
            nms_threshold: None,
            top_k: None,
            json: None,
            annotate: None,
            crop: false,
            output_dir: None,
            preset: None,
            output_width: None,
            output_height: None,
            face_height_pct: 70.0,
            positioning_mode: "center".to_string(),
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            output_format: "png".to_string(),
            jpeg_quality: 90u8,
            png_compression: None,
            webp_quality: None,
            auto_detect_format: None,
            face_index: None,
            min_quality: None,
            skip_low_quality: None,
            auto_select_best: None,
            skip_no_high_quality: None,
            quality_suffix: None,
            metadata_mode: None,
            metadata_include_crop: None,
            metadata_include_quality: None,
            metadata_tags: Vec::new(),
            enhance: Some(true),
            unsharp_amount: Some(0.25),
            unsharp_radius: None,
            enhance_contrast: None,
            enhance_exposure: None,
            enhance_brightness: None,
            enhance_saturation: None,
            enhance_auto_color: None,
            enhance_sharpness: None,
            enhance_skin_smooth: None,
            enhance_red_eye_removal: None,
            enhance_background_blur: None,
            naming_template: None,
            enhancement_preset: Some("vivid".to_string()),
            mapping_file: None,
            mapping_source_col: None,
            mapping_output_col: None,
            mapping_has_headers: None,
            mapping_delimiter: None,
            mapping_sheet: None,
            mapping_format: None,
            mapping_sql_table: None,
            mapping_sql_query: None,
        };
        let enh = build_enhancement_settings(&args).expect("should build");
        // explicit override should win
        assert_eq!(enh.unsharp_amount, 0.25);
        // other vivid params still present
        assert!((enh.contrast - 1.25).abs() < f32::EPSILON);
        assert_eq!(enh.brightness, 10);
    }

    #[test]
    fn synthetic_crop_enhance_saves_file() {
        use std::fs;
        use std::time::{SystemTime, UNIX_EPOCH};

        // Create temp dir
        let base = std::env::temp_dir().join(format!(
            "iron_cropper_test_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&base).expect("create tmp dir");

        // Create a synthetic image
        let img_path = base.join("input.png");
        let img = image::RgbaImage::from_pixel(200, 200, image::Rgba([128, 128, 128, 255]));
        let dyn_img = image::DynamicImage::ImageRgba8(img.clone());
        dyn_img.save(&img_path).expect("save input");

        // Build a fake detection centered in the image
        let det = Detection {
            bbox: BoundingBox {
                x: 50.0,
                y: 50.0,
                width: 100.0,
                height: 100.0,
            },
            landmarks: [
                yunet_core::Landmark { x: 80.0, y: 80.0 },
                yunet_core::Landmark { x: 120.0, y: 80.0 },
                yunet_core::Landmark { x: 100.0, y: 100.0 },
                yunet_core::Landmark { x: 85.0, y: 130.0 },
                yunet_core::Landmark { x: 115.0, y: 130.0 },
            ],
            score: 0.99,
        };

        // Crop settings and crop
        let settings = CropSettings::default();
        let cropped = crop_face_from_image(&dyn_img, &det, &settings);

        // Build args to request enhancement with a preset
        let args = DetectArgs {
            input: Some(img_path.clone()),
            model: PathBuf::from("model.onnx"),
            config: None,
            telemetry: false,
            telemetry_level: None,
            width: None,
            height: None,
            resize_quality: None,
            score_threshold: None,
            nms_threshold: None,
            top_k: None,
            json: None,
            annotate: None,
            crop: false,
            output_dir: None,
            preset: None,
            output_width: None,
            output_height: None,
            face_height_pct: 70.0,
            positioning_mode: "center".to_string(),
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            output_format: "png".to_string(),
            jpeg_quality: 90u8,
            png_compression: None,
            webp_quality: None,
            auto_detect_format: None,
            face_index: None,
            min_quality: None,
            skip_low_quality: None,
            auto_select_best: None,
            skip_no_high_quality: None,
            quality_suffix: None,
            metadata_mode: None,
            metadata_include_crop: None,
            metadata_include_quality: None,
            metadata_tags: Vec::new(),
            enhance: Some(true),
            unsharp_amount: None,
            unsharp_radius: None,
            enhance_contrast: None,
            enhance_exposure: None,
            enhance_brightness: None,
            enhance_saturation: None,
            enhance_auto_color: Some(true),
            enhance_sharpness: None,
            enhance_skin_smooth: None,
            enhance_red_eye_removal: None,
            enhance_background_blur: None,
            naming_template: None,
            enhancement_preset: Some("natural".to_string()),
            mapping_file: None,
            mapping_source_col: None,
            mapping_output_col: None,
            mapping_has_headers: None,
            mapping_delimiter: None,
            mapping_sheet: None,
            mapping_format: None,
            mapping_sql_table: None,
            mapping_sql_query: None,
        };

        let enh = build_enhancement_settings(&args).expect("enh settings");
        let final_crop = apply_enhancements(&cropped, &enh);

        // Save final crop to disk and assert file exists and is non-empty
        let out = base.join("out.png");
        final_crop.save(&out).expect("save crop");
        let md = fs::metadata(&out).expect("metadata");
        assert!(md.len() > 0, "saved file should be non-empty");

        // Cleanup
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn build_quality_filter_defaults_to_none() {
        let settings = QualityAutomationSettings::default();
        let filter = build_quality_filter(&settings);
        assert_eq!(filter.min_quality, None);
        assert!(!filter.auto_select);
        assert!(!filter.auto_skip_no_high);
        assert!(!filter.suffix_enabled);
    }
}

/// Load application settings from a file or use defaults.
fn load_settings(config_path: Option<&PathBuf>) -> Result<AppSettings> {
    if let Some(path) = config_path {
        let resolved = normalize_path(path)?;
        let settings = AppSettings::load_from_path(&resolved)?;
        info!("Loaded settings from {}", resolved.display());
        Ok(settings)
    } else {
        let default_path = default_settings_path();
        if default_path.exists() {
            let settings = AppSettings::load_from_path(&default_path).with_context(|| {
                format!(
                    "failed to load default settings from {}",
                    default_path.display()
                )
            })?;
            info!("Loaded settings from {}", default_path.display());
            Ok(settings)
        } else {
            Ok(AppSettings::default())
        }
    }
}

/// Apply command-line arguments to override loaded or default settings.
fn apply_cli_overrides(settings: &mut AppSettings, args: &DetectArgs) {
    if args.telemetry {
        settings.telemetry.enabled = true;
    }
    if let Some(level) = args.telemetry_level.as_ref() {
        let normalized = level.trim();
        if !normalized.is_empty() {
            let lower = normalized.to_ascii_lowercase();
            settings.telemetry.level = lower.clone();
            if lower == "off" {
                settings.telemetry.enabled = false;
            }
        }
    }

    if let Some(width) = args.width {
        settings.input.width = width;
    }
    if let Some(height) = args.height {
        settings.input.height = height;
    }
    if let Some(mode) = args.resize_quality {
        settings.input.resize_quality = mode;
    }
    if let Some(score) = args.score_threshold {
        settings.detection.score_threshold = score;
    }
    if let Some(nms) = args.nms_threshold {
        settings.detection.nms_threshold = nms;
    }
    if let Some(top_k) = args.top_k {
        settings.detection.top_k = top_k;
    }

    if let Some(preset_name) = args.preset.as_ref() {
        settings.crop.preset = preset_name.to_ascii_lowercase();
        if settings.crop.preset != "custom"
            && let Some(preset) = preset_by_name(preset_name)
            && preset.width > 0
            && preset.height > 0
        {
            settings.crop.output_width = preset.width;
            settings.crop.output_height = preset.height;
        }
    }
    if let Some(width) = args.output_width {
        settings.crop.output_width = width;
        settings.crop.preset = "custom".to_string();
    }
    if let Some(height) = args.output_height {
        settings.crop.output_height = height;
        settings.crop.preset = "custom".to_string();
    }

    settings.crop.face_height_pct = args.face_height_pct;
    settings.crop.horizontal_offset = args.horizontal_offset;
    settings.crop.vertical_offset = args.vertical_offset;
    settings.crop.positioning_mode = args.positioning_mode.replace('_', "-");
    settings.crop.output_format = args.output_format.to_ascii_lowercase();
    settings.crop.jpeg_quality = args.jpeg_quality;
    if let Some(ref compression) = args.png_compression {
        settings.crop.png_compression = compression.clone();
    }
    if let Some(webp) = args.webp_quality {
        settings.crop.webp_quality = webp;
    }
    if let Some(auto) = args.auto_detect_format {
        settings.crop.auto_detect_format = auto;
    }

    if let Some(auto) = args.auto_select_best {
        settings.crop.quality_rules.auto_select_best_face = auto;
    }
    if let Some(skip) = args.skip_no_high_quality {
        settings.crop.quality_rules.auto_skip_no_high_quality = skip;
    }
    if let Some(flag) = args.quality_suffix {
        settings.crop.quality_rules.quality_suffix = flag;
    }
    if let Some(skip) = args.skip_low_quality {
        if skip {
            settings.crop.quality_rules.min_quality = Some(Quality::Medium);
        } else {
            settings.crop.quality_rules.min_quality = None;
        }
    } else if let Some(ref s) = args.min_quality {
        match s.parse::<Quality>() {
            Ok(q) => {
                settings.crop.quality_rules.min_quality = Some(q);
            }
            Err(_) => warn!("unknown --min-quality value '{}', ignoring", s),
        }
    }

    if let Some(ref mode) = args.metadata_mode {
        match mode.parse::<MetadataMode>() {
            Ok(mode) => settings.crop.metadata.mode = mode,
            Err(err) => warn!("{err}"),
        }
    }
    if let Some(include) = args.metadata_include_crop {
        settings.crop.metadata.include_crop_settings = include;
    }
    if let Some(include) = args.metadata_include_quality {
        settings.crop.metadata.include_quality_metrics = include;
    }
    if !args.metadata_tags.is_empty() {
        settings.crop.metadata.custom_tags = parse_metadata_tags_args(&args.metadata_tags);
    }

    settings.crop.sanitize();
}

fn build_core_crop_settings(cfg: &ConfigCropSettings) -> CropSettings {
    CropSettings {
        output_width: cfg.output_width,
        output_height: cfg.output_height,
        face_height_pct: cfg.face_height_pct,
        positioning_mode: parse_positioning_mode(&cfg.positioning_mode),
        horizontal_offset: cfg.horizontal_offset,
        vertical_offset: cfg.vertical_offset,
    }
}

fn parse_positioning_mode(value: &str) -> PositioningMode {
    match value.to_ascii_lowercase().as_str() {
        "rule_of_thirds" | "rule-of-thirds" | "ruleofthirds" => PositioningMode::RuleOfThirds,
        "custom" => PositioningMode::Custom,
        _ => PositioningMode::Center,
    }
}

fn parse_metadata_tags_args(entries: &[String]) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for entry in entries {
        if let Some((key, value)) = entry.split_once('=') {
            let key = key.trim();
            if key.is_empty() {
                warn!("Ignoring metadata tag with empty key: '{entry}'");
                continue;
            }
            map.insert(key.to_string(), value.trim().to_string());
        } else {
            warn!("Invalid metadata tag '{entry}', expected key=value");
        }
    }
    map
}

/// Collect all image paths from a file or directory.
fn collect_images(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    if !path.is_dir() {
        anyhow::bail!(
            "input path is neither file nor directory: {}",
            path.display()
        );
    }

    let exts = ["jpg", "jpeg", "png", "bmp", "webp"];
    let mut images = Vec::new();
    for entry in WalkDir::new(path)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
            let ext_lower = ext.to_ascii_lowercase();
            if exts.contains(&ext_lower.as_str()) {
                images.push(entry.path().to_path_buf());
            } else {
                debug!("Skipping non-image file {}", entry.path().display());
            }
        }
    }
    images.sort();
    Ok(images)
}

fn collect_standard_targets(input_path: &Path) -> Result<Vec<ProcessingItem>> {
    let images = collect_images(input_path)?;
    if images.is_empty() {
        anyhow::bail!(
            "no images found at {} (supported extensions: jpg, jpeg, png, bmp)",
            input_path.display()
        );
    }
    Ok(images
        .into_iter()
        .map(|path| ProcessingItem {
            source: path,
            output_override: None,
            mapping_row: None,
        })
        .collect())
}

fn collect_mapping_targets(mapping_file: &Path, args: &DetectArgs) -> Result<Vec<ProcessingItem>> {
    let mapping_path = normalize_path(mapping_file)?;
    let source_selector = ColumnSelector::parse_token(
        args.mapping_source_col
            .as_deref()
            .ok_or_else(|| anyhow!("--mapping-source-col is required with --mapping-file"))?,
    )?;
    let output_selector = ColumnSelector::parse_token(
        args.mapping_output_col
            .as_deref()
            .ok_or_else(|| anyhow!("--mapping-output-col is required with --mapping-file"))?,
    )?;

    let user_format = match args.mapping_format.as_deref() {
        Some(token) => Some(parse_mapping_format_token(token)?),
        None => None,
    };
    let mut read_options = MappingReadOptions {
        format: user_format,
        has_headers: args.mapping_has_headers,
        delimiter: args.mapping_delimiter.map(|c| c as u8),
        sheet_name: args.mapping_sheet.clone(),
        sql_table: args.mapping_sql_table.clone(),
        sql_query: args.mapping_sql_query.clone(),
        ..Default::default()
    };
    let resolved_format = read_options
        .format
        .unwrap_or_else(|| detect_mapping_format(&mapping_path));
    read_options.format = Some(resolved_format);

    info!(
        "Loading mapping ({}) from {}",
        resolved_format.display_name(),
        mapping_path.display()
    );

    let entries = load_mapping_entries(
        &mapping_path,
        &read_options,
        &source_selector,
        &output_selector,
    )
    .with_context(|| format!("failed to load mapping from {}", mapping_path.display()))?;
    if entries.is_empty() {
        anyhow::bail!(
            "no usable rows found in mapping file {}",
            mapping_path.display()
        );
    }

    let mapping_dir = mapping_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    let mut items = Vec::new();
    for (idx, entry) in entries.into_iter().enumerate() {
        let row_no = idx + 1;
        let raw_source = PathBuf::from(entry.source_path);
        let resolved_source = if raw_source.is_absolute() {
            raw_source
        } else {
            mapping_dir.join(raw_source)
        };
        if !resolved_source.exists() {
            warn!(
                "Skipping mapping row {}: source {} was not found",
                row_no,
                resolved_source.display()
            );
            continue;
        }
        items.push(ProcessingItem {
            source: resolved_source,
            output_override: Some(PathBuf::from(entry.output_name)),
            mapping_row: Some(row_no),
        });
    }

    if items.is_empty() {
        anyhow::bail!(
            "mapping file {} did not produce any usable rows",
            mapping_path.display()
        );
    }

    info!(
        "Loaded {} mapping row(s) from {}",
        items.len(),
        mapping_path.display()
    );

    Ok(items)
}

fn parse_mapping_format_token(token: &str) -> Result<MappingFormat> {
    match token.to_ascii_lowercase().as_str() {
        "csv" | "delimited" | "text" => Ok(MappingFormat::Csv),
        "excel" | "xlsx" | "xls" => Ok(MappingFormat::Excel),
        "parquet" | "pq" => Ok(MappingFormat::Parquet),
        "sqlite" | "sql" | "db" => Ok(MappingFormat::Sqlite),
        other => anyhow::bail!(
            "unknown mapping format '{other}' (supported: csv, excel, parquet, sqlite)"
        ),
    }
}

fn resolve_override_output_path(
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

/// Draw detections on an image and save it to a directory.
fn annotate_image(
    image_path: &Path,
    detections: &[Detection],
    output_dir: &Path,
) -> Result<PathBuf> {
    use image::Rgba;
    use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_rect_mut};

    let mut image = image::open(image_path)
        .with_context(|| format!("failed to open image {}", image_path.display()))?
        .to_rgba8();
    let (img_w, img_h) = image.dimensions();

    if img_w == 0 || img_h == 0 {
        anyhow::bail!(
            "cannot annotate image with zero dimensions: {}",
            image_path.display()
        );
    }

    let rect_color = Rgba([255, 0, 0, 255]);
    let landmark_color = Rgba([0, 255, 0, 255]);

    for detection in detections {
        let rect = rect_from_bbox(&detection.bbox, img_w, img_h);
        draw_hollow_rect_mut(&mut image, rect, rect_color);
        for lm in &detection.landmarks {
            let cx = clamp_to_i32(lm.x, img_w);
            let cy = clamp_to_i32(lm.y, img_h);
            draw_filled_circle_mut(&mut image, (cx, cy), 2, landmark_color);
        }
    }

    let file_name = image_path
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("frame.png"));
    let output_path = output_dir.join(file_name);

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    image
        .save(&output_path)
        .with_context(|| format!("failed to save annotated image {}", output_path.display()))?;

    Ok(output_path)
}

/// Convert a floating-point `BoundingBox` to an integer `imageproc::rect::Rect`.
fn rect_from_bbox(bbox: &BoundingBox, img_w: u32, img_h: u32) -> imageproc::rect::Rect {
    use imageproc::rect::Rect;

    let max_x = if img_w == 0 { 0.0 } else { (img_w - 1) as f32 };
    let max_y = if img_h == 0 { 0.0 } else { (img_h - 1) as f32 };

    let x1 = bbox.x.clamp(0.0, max_x);
    let y1 = bbox.y.clamp(0.0, max_y);
    let x2 = (bbox.x + bbox.width).clamp(0.0, max_x);
    let y2 = (bbox.y + bbox.height).clamp(0.0, max_y);

    let width = (x2 - x1).max(1.0).round() as u32;
    let height = (y2 - y1).max(1.0).round() as u32;

    Rect::at(x1.round() as i32, y1.round() as i32).of_size(width, height)
}

/// Clamp a floating-point coordinate to a valid integer pixel index.
fn clamp_to_i32(value: f32, max_extent: u32) -> i32 {
    if max_extent == 0 {
        return 0;
    }
    let max = (max_extent - 1) as f32;
    value.clamp(0.0, max).round() as i32
}

impl From<&Detection> for DetectionRecord {
    fn from(detection: &Detection) -> Self {
        Self {
            score: detection.score,
            bbox: [
                detection.bbox.x,
                detection.bbox.y,
                detection.bbox.width,
                detection.bbox.height,
            ],
            landmarks: detection.landmarks.map(|lm| [lm.x, lm.y]),
            quality_score: None,
            quality: None,
        }
    }
}
