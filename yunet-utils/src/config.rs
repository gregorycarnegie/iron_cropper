use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Shared detection parameters that should mirror YuNet defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DetectionSettings {
    pub score_threshold: f32,
    pub nms_threshold: f32,
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
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct InputDimensions {
    pub width: u32,
    pub height: u32,
}

impl Default for InputDimensions {
    fn default() -> Self {
        Self {
            width: 320,
            height: 320,
        }
    }
}

/// Persistent application settings consumed by CLI and GUI front ends.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    /// Optional override for the YuNet ONNX model path.
    pub model_path: Option<String>,
    pub input: InputDimensions,
    pub detection: DetectionSettings,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            model_path: Some("models/face_detection_yunet_2023mar.onnx".into()),
            input: InputDimensions::default(),
            detection: DetectionSettings::default(),
        }
    }
}

impl AppSettings {
    /// Load settings from a JSON file.
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
