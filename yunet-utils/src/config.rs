use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct InputDimensions {
    pub width: u32,
    pub height: u32,
}

impl Default for InputDimensions {
    fn default() -> Self {
        Self {
            width: 640,
            height: 640,
        }
    }
}

/// Settings for face cropping operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Output format: "png", "jpeg", or "webp"
    pub output_format: String,
    /// JPEG quality (1-100, only used when format is jpeg)
    pub jpeg_quality: u8,
}

/// Settings for image enhancement operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
            output_format: "png".to_string(),
            jpeg_quality: 90,
        }
    }
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
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            model_path: Some("models/face_detection_yunet_2023mar_640.onnx".into()),
            input: InputDimensions::default(),
            detection: DetectionSettings::default(),
            crop: CropSettings::default(),
            enhance: EnhanceSettings::default(),
        }
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
                height: 640
            }
        );
        assert_eq!(loaded.detection.top_k, 123);
        assert!(loaded.model_path.is_some());
    }
}
