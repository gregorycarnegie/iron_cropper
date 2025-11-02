//! Command-line interface for running YuNet face detection.

use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result};
use clap::Parser;
use image::GenericImageView;
use log::{debug, info, warn};
use rayon::prelude::*;
use serde::Serialize;
use std::sync::atomic::{AtomicUsize, Ordering};
use walkdir::WalkDir;
use yunet_core::{BoundingBox, Detection, PostprocessConfig, PreprocessConfig, YuNetDetector};
use yunet_core::{CropSettings, PositioningMode, crop_face_from_image, preset_by_name};
use yunet_utils::{
    EnhancementSettings, Quality, QualityFilter, apply_enhancements, config::AppSettings,
    estimate_sharpness, init_logging, normalize_path,
};

/// Run YuNet face detection over images or directories.
#[derive(Debug, Parser)]
#[command(author, version, about)]
struct DetectArgs {
    /// Path to an image file or a directory containing images.
    #[arg(short, long)]
    input: PathBuf,

    /// Path to the YuNet ONNX model.
    #[arg(
        short,
        long,
        default_value = "models/face_detection_yunet_2023mar_640.onnx"
    )]
    model: PathBuf,

    /// Optional settings JSON (defaults to built-in YuNet parameters).
    #[arg(long)]
    config: Option<PathBuf>,

    /// Override input width (pixels).
    #[arg(long)]
    width: Option<u32>,

    /// Override input height (pixels).
    #[arg(long)]
    height: Option<u32>,

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

    /// Select face index (1-based) to save only a specific face. Default: all faces.
    #[arg(long)]
    face_index: Option<usize>,

    /// Minimum quality to save crops (low, medium, high). If set, crops below this level are skipped.
    #[arg(long)]
    min_quality: Option<String>,

    /// Shortcut to skip low-quality crops (equivalent to `--min-quality medium`).
    #[arg(long)]
    skip_low_quality: Option<bool>,

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

    /// Naming template for output crop files. Variables: {original}, {index}, {width}, {height}, {ext}, {timestamp}
    #[arg(long)]
    naming_template: Option<String>,
    /// Enhancement preset to apply when --enhance is set. Options: natural, vivid, professional
    #[arg(long)]
    enhancement_preset: Option<String>,
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

fn main() -> Result<()> {
    init_logging(log::LevelFilter::Info)?;
    let args = DetectArgs::parse();

    let input_path = normalize_path(&args.input)?;
    let model_path = normalize_path(&args.model)?;
    let annotate_dir = if let Some(dir) = args.annotate.as_ref() {
        fs::create_dir_all(dir)
            .with_context(|| format!("failed to create annotation directory {}", dir.display()))?;
        Some(normalize_path(dir)?)
    } else {
        None
    };

    let mut settings = load_settings(args.config.as_ref())?;
    apply_cli_overrides(&mut settings, &args);

    // Build a centralized quality filter from CLI args so the same policy
    // is used for skipping crops and for future GUI/CLI wiring.
    let quality_filter = build_quality_filter(&args);

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

    let images = collect_images(&input_path)?;
    if images.is_empty() {
        anyhow::bail!(
            "no images found at {} (supported extensions: jpg, jpeg, png, bmp)",
            input_path.display()
        );
    }

    info!("Processing {} image(s)...", images.len());

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

    // Process images in parallel
    // Progress counters (shared across parallel tasks)
    let images_processed = Arc::new(AtomicUsize::new(0));
    let faces_detected = Arc::new(AtomicUsize::new(0));
    let crops_saved = Arc::new(AtomicUsize::new(0));
    let crops_skipped_quality = Arc::new(AtomicUsize::new(0));

    let results: Vec<ImageDetections> = images
        .par_iter()
        .filter_map(|image_path| {
            // mark image as started
            images_processed.fetch_add(1, Ordering::Relaxed);
            let detector = Arc::clone(&detector);
            let annotate_dir = Arc::clone(&annotate_dir);

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

            // Optionally produce cropped face images for each detection
            #[allow(clippy::collapsible_if)]
            if crop_enabled {
                if let Some(out_dir) = crop_output_dir.as_ref() {
                    // Open source image once per-task
                    match image::open(image_path) {
                        Ok(img) => {
                                for (idx, det) in output.detections.iter().enumerate() {
                                    // If user requested a specific face index, skip others.
                                    if let Some(fidx) = args.face_index {
                                        // CLI exposes 1-based indexes; convert to 0-based
                                        if fidx == 0 || fidx - 1 != idx {
                                            continue;
                                        }
                                    }
                                // Build CropSettings from CLI args / preset
                                let mut settings = CropSettings::default();
                                // preset overrides explicit dims
                                #[allow(clippy::collapsible_if)]
                                if let Some(pname) = args.preset.as_ref() {
                                    if let Some(p) = preset_by_name(pname) {
                                        if p.width > 0 && p.height > 0 {
                                            settings.output_width = p.width;
                                            settings.output_height = p.height;
                                        }
                                    }
                                }
                                if let Some(w) = args.output_width {
                                    settings.output_width = w;
                                }
                                if let Some(h) = args.output_height {
                                    settings.output_height = h;
                                }
                                settings.face_height_pct = args.face_height_pct;
                                settings.horizontal_offset = args.horizontal_offset;
                                settings.vertical_offset = args.vertical_offset;
                                settings.positioning_mode = match args.positioning_mode.as_str() {
                                    "rule_of_thirds" | "rule-of-thirds" | "ruleofthirds" => PositioningMode::RuleOfThirds,
                                    "custom" => PositioningMode::Custom,
                                    _ => PositioningMode::Center,
                                };

                                let cropped = crop_face_from_image(&img, det, &settings);
                                // Optionally apply enhancement pipeline before quality checks and saving
                                let mut final_crop = cropped;
                                // Build enhancement settings from args (presets + explicit overrides)
                                if let Some(enh) = build_enhancement_settings(&args) {
                                    final_crop = apply_enhancements(&final_crop, &enh);
                                }
                                let stem = image_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
                                let ext = args.output_format.to_lowercase();

                                // Build output filename using optional naming template
                                let out_name = if let Some(tmpl) = args.naming_template.as_ref() {
                                    use std::time::{SystemTime, UNIX_EPOCH};
                                    let ts = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
                                    let mut name = tmpl.clone();
                                    name = name.replace("{original}", stem);
                                    name = name.replace("{index}", &(idx + 1).to_string());
                                    // settings here refers to the local CropSettings built above
                                    name = name.replace("{width}", &settings.output_width.to_string());
                                    name = name.replace("{height}", &settings.output_height.to_string());
                                    name = name.replace("{ext}", &ext);
                                    name = name.replace("{timestamp}", &ts.to_string());
                                    // If user did not include {ext}, append it
                                    if !tmpl.contains("{ext}") {
                                        format!("{}.{}", name, ext)
                                    } else {
                                        name
                                    }
                                } else {
                                    format!("{}_face{}.{}", stem, idx + 1, ext)
                                };
                                let out_path = out_dir.join(&out_name);
                                // Optionally skip based on centralized QualityFilter
                                let mut should_skip = false;
                                if quality_filter.min_quality.is_some() {
                                    let (score, q) = estimate_sharpness(&final_crop);
                                    debug!("crop sharpness {} -> {:?}", score, q);
                                    if quality_filter.should_skip(q) {
                                        should_skip = true;
                                    }
                                }

                                if should_skip {
                                    info!("Skipping crop for {} face {} due to low quality", image_path.display(), idx + 1);
                                    crops_skipped_quality.fetch_add(1, Ordering::Relaxed);
                                    continue;
                                }

                                // Save with requested format/quality
                                match ext.as_str() {
                                    "jpeg" | "jpg" => {
                                        // Use a JPEG encoder to honor the requested quality.
                                        match File::create(&out_path) {
                                            Ok(mut f) => {
                                                let buf = final_crop.to_rgba8();
                                                let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut f, args.jpeg_quality);
                                                if let Err(e) = encoder.encode(buf.as_raw(), buf.width(), buf.height(), image::ColorType::Rgba8.into()) {
                                                    warn!("Failed to encode JPEG {}: {}", out_path.display(), e);
                                                } else {
                                                    info!("Saved crop to {}", out_path.display());
                                                    crops_saved.fetch_add(1, Ordering::Relaxed);
                                                }
                                            }
                                            Err(e) => {
                                                warn!("Failed to create file {}: {}", out_path.display(), e);
                                            }
                                        }
                                    }
                                    "webp" => {
                                        if let Err(e) = final_crop.save_with_format(&out_path, image::ImageFormat::WebP) {
                                            warn!("Failed to save WebP {}: {}", out_path.display(), e);
                                        } else {
                                            info!("Saved crop to {}", out_path.display());
                                            crops_saved.fetch_add(1, Ordering::Relaxed);
                                        }
                                    }
                                    _ => {
                                        // default to PNG
                                        if let Err(e) = final_crop.save(&out_path) {
                                            warn!("Failed to save crop {}: {}", out_path.display(), e);
                                        } else {
                                            info!("Saved crop to {}", out_path.display());
                                            crops_saved.fetch_add(1, Ordering::Relaxed);
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Failed to open image for cropping {}: {}", image_path.display(), e);
                        }
                    }
                }
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

/// Construct a `QualityFilter` from CLI arguments.
fn build_quality_filter(args: &DetectArgs) -> QualityFilter {
    if args.skip_low_quality.unwrap_or(false) {
        QualityFilter::new(Some(Quality::Medium))
    } else if let Some(ref s) = args.min_quality {
        match s.parse::<Quality>() {
            Ok(q) => QualityFilter::new(Some(q)),
            Err(_) => {
                warn!("unknown --min-quality value '{}', ignoring", s);
                QualityFilter::new(None)
            }
        }
    } else {
        QualityFilter::new(None)
    }
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

    Some(base)
}

#[allow(clippy::items_after_test_module)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_quality_filter_skip_low_maps_to_medium() {
        let args = DetectArgs {
            input: PathBuf::from("image.png"),
            model: PathBuf::from("model.onnx"),
            config: None,
            width: None,
            height: None,
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
            face_index: None,
            min_quality: None,
            skip_low_quality: Some(true),
            enhance: None,
            unsharp_amount: None,
            unsharp_radius: None,
            enhance_contrast: None,
            enhance_exposure: None,
            enhance_brightness: None,
            enhance_saturation: None,
            enhance_auto_color: None,
            enhance_sharpness: None,
            naming_template: None,
            enhancement_preset: None,
        };
        let qf = build_quality_filter(&args);
        assert_eq!(qf.min_quality, Some(Quality::Medium));
    }

    #[test]
    fn enhancement_preset_vivid_applies_defaults_and_overrides() {
        let args = DetectArgs {
            input: PathBuf::from("image.png"),
            model: PathBuf::from("model.onnx"),
            config: None,
            width: None,
            height: None,
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
            face_index: None,
            min_quality: None,
            skip_low_quality: None,
            enhance: Some(true),
            unsharp_amount: None,
            unsharp_radius: None,
            enhance_contrast: None,
            enhance_exposure: None,
            enhance_brightness: None,
            enhance_saturation: None,
            enhance_auto_color: None,
            enhance_sharpness: None,
            naming_template: None,
            enhancement_preset: Some("vivid".to_string()),
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
            input: PathBuf::from("image.png"),
            model: PathBuf::from("model.onnx"),
            config: None,
            width: None,
            height: None,
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
            face_index: None,
            min_quality: None,
            skip_low_quality: None,
            enhance: Some(true),
            unsharp_amount: Some(0.25),
            unsharp_radius: None,
            enhance_contrast: None,
            enhance_exposure: None,
            enhance_brightness: None,
            enhance_saturation: None,
            enhance_auto_color: None,
            enhance_sharpness: None,
            naming_template: None,
            enhancement_preset: Some("vivid".to_string()),
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
            input: img_path.clone(),
            model: PathBuf::from("model.onnx"),
            config: None,
            width: None,
            height: None,
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
            face_index: None,
            min_quality: None,
            skip_low_quality: None,
            enhance: Some(true),
            unsharp_amount: None,
            unsharp_radius: None,
            enhance_contrast: None,
            enhance_exposure: None,
            enhance_brightness: None,
            enhance_saturation: None,
            enhance_auto_color: Some(true),
            enhance_sharpness: None,
            naming_template: None,
            enhancement_preset: Some("natural".to_string()),
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
    fn build_quality_filter_parses_min_quality() {
        let args = DetectArgs {
            input: PathBuf::from("image.png"),
            model: PathBuf::from("model.onnx"),
            config: None,
            width: None,
            height: None,
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
            face_index: None,
            min_quality: Some("high".to_string()),
            skip_low_quality: None,
            enhance: None,
            unsharp_amount: None,
            unsharp_radius: None,
            enhance_contrast: None,
            enhance_exposure: None,
            enhance_brightness: None,
            enhance_saturation: None,
            enhance_auto_color: None,
            enhance_sharpness: None,
            naming_template: None,
            enhancement_preset: None,
        };
        let qf = build_quality_filter(&args);
        assert_eq!(qf.min_quality, Some(Quality::High));
    }

    #[test]
    fn build_quality_filter_none_when_not_set() {
        let args = DetectArgs {
            input: PathBuf::from("image.png"),
            model: PathBuf::from("model.onnx"),
            config: None,
            width: None,
            height: None,
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
            face_index: None,
            min_quality: None,
            skip_low_quality: None,
            enhance: None,
            unsharp_amount: None,
            unsharp_radius: None,
            enhance_contrast: None,
            enhance_exposure: None,
            enhance_brightness: None,
            enhance_saturation: None,
            enhance_auto_color: None,
            enhance_sharpness: None,
            naming_template: None,
            enhancement_preset: None,
        };
        let qf = build_quality_filter(&args);
        assert_eq!(qf.min_quality, None);
    }
}

/// Load application settings from a file or use defaults.
fn load_settings(config_path: Option<&PathBuf>) -> Result<AppSettings> {
    if let Some(path) = config_path {
        let resolved = normalize_path(path)?;
        AppSettings::load_from_path(&resolved)
    } else {
        Ok(AppSettings::default())
    }
}

/// Apply command-line arguments to override loaded or default settings.
fn apply_cli_overrides(settings: &mut AppSettings, args: &DetectArgs) {
    if let Some(width) = args.width {
        settings.input.width = width;
    }
    if let Some(height) = args.height {
        settings.input.height = height;
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
