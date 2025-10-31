use std::{
    fs::{self, File},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use clap::Parser;
use log::{debug, info, warn};
use serde::Serialize;
use walkdir::WalkDir;
use yunet_core::{BoundingBox, Detection, PostprocessConfig, PreprocessConfig, YuNetDetector};
use yunet_utils::{config::AppSettings, init_logging, normalize_path};

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
}

#[derive(Debug, Serialize)]
struct DetectionRecord {
    score: f32,
    bbox: [f32; 4],
    landmarks: [[f32; 2]; 5],
}

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
    let mut results = Vec::with_capacity(images.len());
    for image_path in images {
        match detector.detect_path(&image_path) {
            Ok(output) => {
                info!(
                    "{} -> {} detection(s)",
                    image_path.display(),
                    output.detections.len()
                );
                let annotated_path = if let Some(dir) = annotate_dir.as_ref() {
                    match annotate_image(&image_path, &output.detections, dir) {
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

                let detection_records: Vec<DetectionRecord> = output
                    .detections
                    .iter()
                    .map(DetectionRecord::from)
                    .collect();
                results.push(ImageDetections {
                    image: image_path.display().to_string(),
                    detections: detection_records,
                    annotated: annotated_path,
                });
            }
            Err(err) => {
                warn!("Failed to process {}: {err}", image_path.display());
            }
        }
    }

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

    Ok(())
}

fn load_settings(config_path: Option<&PathBuf>) -> Result<AppSettings> {
    if let Some(path) = config_path {
        let resolved = normalize_path(path)?;
        AppSettings::load_from_path(&resolved)
    } else {
        Ok(AppSettings::default())
    }
}

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
        }
    }
}
