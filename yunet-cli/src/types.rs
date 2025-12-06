//! Shared types and conversions for yunet-cli.

use serde::Serialize;
use yunet_core::Detection;

/// A serializable representation of a single detection.
#[derive(Debug, Serialize)]
pub struct DetectionRecord {
    pub score: f32,
    pub bbox: [f32; 4],
    pub landmarks: [[f32; 2]; 5],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
}

/// A serializable representation of all detections for a single image.
#[derive(Debug, Serialize)]
pub struct ImageDetections {
    pub image: String,
    pub detections: Vec<DetectionRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotated: Option<String>,
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
