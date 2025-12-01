//! Shared configuration types consumed across the YuNet workspace.
//!
//! These structures provide a common representation for inference, detection, cropping, and
//! enhancement settings that can be serialized to disk and reused by CLI and GUI front-ends.

use crate::{color::RgbaColor, gpu::GpuContextOptions, quality::Quality, shape::CropShape};

use anyhow::{Context, Result};
use log::LevelFilter;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    env, fmt, fs,
    path::{Path, PathBuf},
    str::FromStr,
};

/// Shared detection parameters that should mirror YuNet defaults.
///
/// These settings directly control the behavior of the post-processing steps,
/// such as non-maximum suppression (NMS) and score filtering.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DetectionSettings {
    /// Minimum confidence score for a detection to be considered valid.
    pub score_threshold: f32,
    /// Threshold for non-maximum suppression to merge overlapping bounding boxes.
    pub nms_threshold: f32,
    /// The maximum number of detections to return.
    pub top_k: usize,
}

impl Default for DetectionSettings {
    fn default() -> Self {
        Self {
            score_threshold: 0.9,
            nms_threshold: 0.3,
            top_k: 5_000,
        }
    }
}

/// Inference input resolution in pixels (width x height).
///
/// The input image will be resized to these dimensions before being passed to the model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResizeQuality {
    /// Preserve visual quality when resizing (default, Triangle filter).
    #[default]
    Quality,
    /// Prioritize throughput for batch inference (Nearest filter).
    Speed,
}

impl ResizeQuality {
    pub fn as_label(self) -> &'static str {
        match self {
            ResizeQuality::Quality => "Quality",
            ResizeQuality::Speed => "Speed",
        }
    }
}

impl fmt::Display for ResizeQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ResizeQuality::Quality => "quality",
                ResizeQuality::Speed => "speed",
            }
        )
    }
}

impl FromStr for ResizeQuality {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "quality" => Ok(ResizeQuality::Quality),
            "speed" => Ok(ResizeQuality::Speed),
            other => Err(format!(
                "invalid resize quality '{other}'; expected 'quality' or 'speed'"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct InputDimensions {
    pub width: u32,
    pub height: u32,
    /// Choose between quality-focused or speed-focused resizing.
    pub resize_quality: ResizeQuality,
}

impl Default for InputDimensions {
    fn default() -> Self {
        Self {
            width: 640,
            height: 640,
            resize_quality: ResizeQuality::Speed,
        }
    }
}

/// Settings for face cropping operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CropSettings {
    /// Crop preset name (e.g., "linkedin", "passport", "custom")
    pub preset: String,
    /// Output width in pixels (used when preset is "custom")
    pub output_width: u32,
    /// Output height in pixels (used when preset is "custom")
    pub output_height: u32,
    /// Face height as percentage of output height (0-100)
    pub face_height_pct: f32,
    /// Positioning mode: "center", "rule-of-thirds", or "custom"
    pub positioning_mode: String,
    /// Vertical offset for custom positioning (-1.0 to 1.0)
    pub vertical_offset: f32,
    /// Horizontal offset for custom positioning (-1.0 to 1.0)
    pub horizontal_offset: f32,
    /// Background color applied when the crop extends outside the source image bounds.
    pub fill_color: RgbaColor,
    /// Output format: "png", "jpeg", or "webp"
    pub output_format: String,
    /// JPEG quality (1-100, only used when format is jpeg)
    pub jpeg_quality: u8,
    /// PNG compression strategy ("fast", "default", "best") or numeric level (0-9)
    pub png_compression: String,
    /// WebP quality (0-100, lossy encoding)
    pub webp_quality: u8,
    /// Automatically detect output format from the file extension.
    pub auto_detect_format: bool,
    /// Metadata behavior for exported crops.
    pub metadata: MetadataSettings,
    /// Quality-based automation options.
    pub quality_rules: QualityAutomationSettings,
    /// Geometric shape applied to the exported crop.
    pub shape: CropShape,
}

impl CropSettings {
    /// Clamp values to sensible ranges.
    pub fn sanitize(&mut self) {
        self.shape = self.shape.sanitized();
    }
}

/// Settings for image enhancement operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct EnhanceSettings {
    /// Enable enhancements
    pub enabled: bool,
    /// Enhancement preset: "none", "natural", "vivid", or "professional"
    pub preset: String,
    /// Apply histogram-equalization based auto color correction
    pub auto_color: bool,
    /// Exposure adjustment in stops (-2.0 to 2.0)
    pub exposure_stops: f32,
    /// Additional brightness offset (-100 to 100)
    pub brightness: i32,
    /// Contrast multiplier (0.5 to 2.0)
    pub contrast: f32,
    /// Saturation multiplier (0.0 to 2.5)
    pub saturation: f32,
    /// Sharpness (0.0 to 2.0)
    pub sharpness: f32,
    /// Skin smoothing strength (0.0 to 1.0)
    pub skin_smooth: f32,
    /// Enable automated red-eye removal
    pub red_eye_removal: bool,
    /// Enable background blur (portrait mode effect)
    pub background_blur: bool,
}

impl Default for CropSettings {
    fn default() -> Self {
        Self {
            preset: "linkedin".to_string(),
            output_width: 400,
            output_height: 400,
            face_height_pct: 70.0,
            positioning_mode: "center".to_string(),
            vertical_offset: 0.0,
            horizontal_offset: 0.0,
            fill_color: RgbaColor::default(),
            output_format: "png".to_string(),
            jpeg_quality: 90,
            png_compression: "default".to_string(),
            webp_quality: 90,
            auto_detect_format: true,
            metadata: MetadataSettings::default(),
            quality_rules: QualityAutomationSettings::default(),
            shape: CropShape::Rectangle,
        }
    }
}

/// How metadata should be handled for exported crops.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MetadataMode {
    #[default]
    Preserve,
    Strip,
    Custom,
}

impl std::str::FromStr for MetadataMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "preserve" => Ok(MetadataMode::Preserve),
            "strip" => Ok(MetadataMode::Strip),
            "custom" => Ok(MetadataMode::Custom),
            other => Err(format!("unknown metadata mode '{}'", other)),
        }
    }
}

/// Metadata configuration for exported crops.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct MetadataSettings {
    /// Desired metadata strategy.
    pub mode: MetadataMode,
    /// Include crop settings (size, offsets, preset) as custom metadata.
    pub include_crop_settings: bool,
    /// Include detection quality metrics as custom metadata.
    pub include_quality_metrics: bool,
    /// Arbitrary user-defined metadata key/value pairs.
    pub custom_tags: BTreeMap<String, String>,
}

impl Default for MetadataSettings {
    fn default() -> Self {
        Self {
            mode: MetadataMode::Preserve,
            include_crop_settings: true,
            include_quality_metrics: true,
            custom_tags: BTreeMap::new(),
        }
    }
}

/// Automation options driven by quality analysis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct QualityAutomationSettings {
    /// Automatically select the highest quality face when multiple are detected.
    pub auto_select_best_face: bool,
    /// Minimum quality required to keep a crop.
    pub min_quality: Option<Quality>,
    /// Skip exporting entirely when no face meets `Quality::High`.
    pub auto_skip_no_high_quality: bool,
    /// Append a quality suffix (e.g., `_highq`) to exported filenames.
    pub quality_suffix: bool,
}

impl Default for EnhanceSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            preset: "none".to_string(),
            auto_color: false,
            exposure_stops: 0.0,
            brightness: 0,
            contrast: 1.0,
            saturation: 1.0,
            sharpness: 0.0,
            skin_smooth: 0.0,
            red_eye_removal: false,
            background_blur: false,
        }
    }
}

/// Settings controlling optional runtime telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TelemetrySettings {
    /// Whether telemetry timing logs are enabled.
    pub enabled: bool,
    /// Logging level for telemetry output (error, warn, info, debug, trace).
    pub level: String,
}

impl Default for TelemetrySettings {
    fn default() -> Self {
        Self {
            enabled: false,
            level: "debug".to_string(),
        }
    }
}

impl TelemetrySettings {
    /// Resolve the configured level string into a `LevelFilter`.
    pub fn level_filter(&self) -> LevelFilter {
        match self.level.trim().to_ascii_lowercase().as_str() {
            "off" => LevelFilter::Off,
            "error" => LevelFilter::Error,
            "warn" | "warning" => LevelFilter::Warn,
            "info" => LevelFilter::Info,
            "trace" => LevelFilter::Trace,
            _ => LevelFilter::Debug,
        }
    }

    /// Update the level string from a `LevelFilter` value.
    pub fn set_level(&mut self, level: LevelFilter) {
        let label = match level {
            LevelFilter::Off => "off",
            LevelFilter::Error => "error",
            LevelFilter::Warn => "warn",
            LevelFilter::Info => "info",
            LevelFilter::Debug => "debug",
            LevelFilter::Trace => "trace",
        };
        self.level = label.to_string();
    }
}

/// Persistent application settings consumed by CLI and GUI front ends.
///
/// This struct aggregates all user-configurable parameters, allowing them to be
/// loaded from and saved to a JSON file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    /// Optional override for the YuNet ONNX model path.
    /// If `None`, a default path is used.
    pub model_path: Option<String>,
    /// The input dimensions for model inference.
    pub input: InputDimensions,
    /// The parameters for detection post-processing.
    pub detection: DetectionSettings,
    /// The parameters for face cropping.
    pub crop: CropSettings,
    /// The parameters for image enhancement.
    pub enhance: EnhanceSettings,
    /// Telemetry and diagnostics preferences.
    pub telemetry: TelemetrySettings,
    /// GPU runtime preferences shared across CLI and GUI.
    pub gpu: GpuSettings,
}

impl Default for AppSettings {
    fn default() -> Self {
        let mut settings = Self {
            model_path: Some("models/face_detection_yunet_2023mar_640.onnx".into()),
            input: InputDimensions::default(),
            detection: DetectionSettings::default(),
            crop: CropSettings::default(),
            enhance: EnhanceSettings::default(),
            telemetry: TelemetrySettings::default(),
            gpu: GpuSettings::default(),
        };
        settings.crop.sanitize();
        settings
    }
}

impl AppSettings {
    /// Load settings from a JSON file.
    ///
    /// If the file does not exist or cannot be parsed, an error is returned.
    /// If the `model_path` is missing from the JSON, it falls back to the default.
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path)
            .with_context(|| format!("failed to read settings file {}", path.display()))?;
        let mut settings: AppSettings = serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse settings JSON at {}", path.display()))?;

        if settings.model_path.is_none() {
            settings.model_path = Some(AppSettings::default().model_path.unwrap());
        }

        settings.crop.sanitize();

        Ok(settings)
    }

    /// Serialize settings to disk in pretty-printed JSON.
    ///
    /// This will overwrite the file if it already exists.
    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let payload =
            serde_json::to_string_pretty(self).context("failed to serialize settings JSON")?;
        fs::write(path, payload)
            .with_context(|| format!("failed to write settings file {}", path.display()))?;
        Ok(())
    }
}

/// Returns the default path for persisted application settings (`config/gui_settings.json`).
pub fn default_settings_path() -> PathBuf {
    env::current_dir()
        .map(|dir| dir.join("config/gui_settings.json"))
        .unwrap_or_else(|_| PathBuf::from("config/gui_settings.json"))
}

/// GPU-specific runtime preferences.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GpuSettings {
    /// Whether GPU acceleration should be attempted (auto-detect by default).
    pub enabled: bool,
    /// Respect `WGPU_*` environment overrides when initializing the backend.
    pub respect_env: bool,
    /// Execute YuNet inference on the GPU when supported.
    pub inference: bool,
    /// Use GPU for image preprocessing (resize, color conversion).
    pub preprocessing: bool,
}

impl Default for GpuSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            respect_env: true,
            inference: true,
            preprocessing: true, // Enable GPU preprocessing by default when GPU is available
        }
    }
}

impl From<GpuSettings> for GpuContextOptions {
    fn from(settings: GpuSettings) -> Self {
        GpuContextOptions {
            enabled: settings.enabled,
            respect_env: settings.respect_env,
            ..Default::default()
        }
    }
}

impl From<&GpuSettings> for GpuContextOptions {
    fn from(settings: &GpuSettings) -> Self {
        settings.clone().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn default_settings_round_trip() {
        let file = NamedTempFile::new().expect("tempfile");
        let settings = AppSettings::default();
        settings.save_to_path(file.path()).expect("save");

        let loaded = AppSettings::load_from_path(file.path()).expect("load");
        assert_eq!(loaded.input, settings.input);
        assert_eq!(loaded.detection.top_k, settings.detection.top_k);
        assert_eq!(loaded.model_path, settings.model_path);
        assert_eq!(loaded.telemetry.enabled, settings.telemetry.enabled);
        assert_eq!(loaded.telemetry.level, settings.telemetry.level);
        assert_eq!(loaded.gpu.enabled, settings.gpu.enabled);
        assert_eq!(loaded.gpu.respect_env, settings.gpu.respect_env);
    }

    #[test]
    fn missing_model_path_uses_default() {
        let file = NamedTempFile::new().expect("tempfile");
        let json = r#"{
            "input": { "width": 640, "height": 640 },
            "detection": { "score_threshold": 0.8, "nms_threshold": 0.25, "top_k": 123 }
        }"#;
        fs::write(file.path(), json).expect("write custom settings");

        let loaded = AppSettings::load_from_path(file.path()).expect("load");
        assert_eq!(
            loaded.input,
            InputDimensions {
                width: 640,
                height: 640,
                resize_quality: ResizeQuality::Speed,
            }
        );
        assert_eq!(loaded.detection.top_k, 123);
        assert!(loaded.model_path.is_some());
        assert!(!loaded.telemetry.enabled);
        assert_eq!(loaded.telemetry.level_filter(), LevelFilter::Debug);
        assert!(loaded.gpu.enabled);
        assert!(loaded.gpu.respect_env);
    }

    #[test]
    fn telemetry_level_parses_variants() {
        let telemetry = TelemetrySettings {
            level: "TRACE".into(),
            ..TelemetrySettings::default()
        };
        assert_eq!(telemetry.level_filter(), LevelFilter::Trace);

        let telemetry = TelemetrySettings {
            level: "Warn".into(),
            ..TelemetrySettings::default()
        };
        assert_eq!(telemetry.level_filter(), LevelFilter::Warn);

        let mut telemetry = TelemetrySettings::default();
        telemetry.set_level(LevelFilter::Info);
        assert_eq!(telemetry.level, "info");
    }
}
