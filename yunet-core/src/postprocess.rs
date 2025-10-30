use anyhow::Result;
use std::cmp::Ordering;
use tract_onnx::prelude::{Tensor, tract_ndarray::ArrayView2};

/// Canonical YuNet detection configuration.
#[derive(Debug, Clone)]
pub struct PostprocessConfig {
    pub score_threshold: f32,
    pub nms_threshold: f32,
    pub top_k: usize,
}

impl Default for PostprocessConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.9,
            nms_threshold: 0.3,
            top_k: 5_000,
        }
    }
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl BoundingBox {
    pub fn area(&self) -> f32 {
        (self.width.max(0.0)) * (self.height.max(0.0))
    }

    pub fn iou(&self, other: &Self) -> f32 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);

        let intersection_w = (x2 - x1).max(0.0);
        let intersection_h = (y2 - y1).max(0.0);
        let intersection = intersection_w * intersection_h;

        if intersection <= 0.0 {
            return 0.0;
        }

        let union = self.area() + other.area() - intersection;
        if union <= 0.0 {
            0.0
        } else {
            intersection / union
        }
    }
}

/// Facial landmark coordinate (x, y) in image space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Landmark {
    pub x: f32,
    pub y: f32,
}

/// A single YuNet detection result.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub bbox: BoundingBox,
    pub landmarks: [Landmark; 5],
    pub score: f32,
}

/// Decode YuNet outputs into filtered detections.
pub fn apply_postprocess(
    output: &Tensor,
    scale_x: f32,
    scale_y: f32,
    config: &PostprocessConfig,
) -> Result<Vec<Detection>> {
    let rows = detection_rows(output)?;
    anyhow::ensure!(
        rows.shape()[1] == 15,
        "YuNet output must have 15 columns per detection"
    );

    let mut detections = Vec::with_capacity(rows.nrows());
    for row in rows.rows() {
        let score = row[14];
        if !score.is_finite() || score < config.score_threshold {
            continue;
        }

        let bbox = BoundingBox {
            x: row[0] * scale_x,
            y: row[1] * scale_y,
            width: row[2] * scale_x,
            height: row[3] * scale_y,
        };
        if bbox.width <= 0.0 || bbox.height <= 0.0 {
            continue;
        }

        let landmarks = [
            Landmark {
                x: row[4] * scale_x,
                y: row[5] * scale_y,
            },
            Landmark {
                x: row[6] * scale_x,
                y: row[7] * scale_y,
            },
            Landmark {
                x: row[8] * scale_x,
                y: row[9] * scale_y,
            },
            Landmark {
                x: row[10] * scale_x,
                y: row[11] * scale_y,
            },
            Landmark {
                x: row[12] * scale_x,
                y: row[13] * scale_y,
            },
        ];

        detections.push(Detection {
            bbox,
            landmarks,
            score,
        });
    }

    detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

    if config.top_k > 0 && detections.len() > config.top_k {
        detections.truncate(config.top_k);
    }

    if config.nms_threshold > 0.0 && detections.len() > 1 {
        detections = non_max_suppression(detections, config.nms_threshold);
    }

    Ok(detections)
}

fn detection_rows<'a>(output: &'a Tensor) -> Result<ArrayView2<'a, f32>> {
    let shape = output.shape();
    let rows = match shape {
        [rows, 15] => *rows,
        [1, rows, 15] => *rows,
        other => anyhow::bail!(
            "YuNet output must have shape [N, 15] or [1, N, 15] (got {:?})",
            other
        ),
    };

    let slice = output
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("YuNet output is not f32: {e}"))?;

    ArrayView2::from_shape((rows, 15), slice)
        .map_err(|_| anyhow::anyhow!("YuNet output data is not contiguous"))
}

fn non_max_suppression(mut detections: Vec<Detection>, threshold: f32) -> Vec<Detection> {
    let mut result: Vec<Detection> = Vec::with_capacity(detections.len());
    for detection in detections.drain(..) {
        let mut suppressed = false;
        for kept in &result {
            if detection.bbox.iou(&kept.bbox) > threshold {
                suppressed = true;
                break;
            }
        }
        if !suppressed {
            result.push(detection);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor_from_rows(rows: &[[f32; 15]]) -> Tensor {
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        Tensor::from_shape(&[rows.len(), 15], &flat).unwrap()
    }

    #[test]
    fn filters_by_score_and_scales_coordinates() {
        let tensor = tensor_from_rows(&[
            [
                10.0, 20.0, 30.0, 40.0, // bbox
                12.0, 22.0, // landmarks...
                18.0, 25.0, //
                25.0, 30.0, //
                15.0, 35.0, //
                20.0, 28.0, //
                0.95, // score
            ],
            [
                5.0, 5.0, 10.0, 10.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 10.0, 10.0, 0.2,
            ],
        ]);

        let detections = apply_postprocess(
            &tensor,
            2.0,
            0.5,
            &PostprocessConfig {
                score_threshold: 0.3,
                ..Default::default()
            },
        )
        .expect("postprocess should succeed");

        assert_eq!(detections.len(), 1);
        let det = &detections[0];
        assert_eq!(det.score, 0.95);
        assert!((det.bbox.x - 20.0).abs() < f32::EPSILON);
        assert!((det.bbox.y - 10.0).abs() < f32::EPSILON);
        assert!((det.bbox.width - 60.0).abs() < f32::EPSILON);
        assert!((det.bbox.height - 20.0).abs() < f32::EPSILON);
        assert!((det.landmarks[0].x - 24.0).abs() < f32::EPSILON);
        assert!((det.landmarks[0].y - 11.0).abs() < f32::EPSILON);
    }

    #[test]
    fn applies_non_max_suppression() {
        let tensor = tensor_from_rows(&[
            [
                0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99,
            ],
            [
                1.0, 1.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95,
            ],
        ]);

        let detections = apply_postprocess(
            &tensor,
            1.0,
            1.0,
            &PostprocessConfig {
                score_threshold: 0.3,
                nms_threshold: 0.3,
                top_k: 10,
            },
        )
        .expect("postprocess should succeed");

        assert_eq!(detections.len(), 1);
        assert!((detections[0].score - 0.99).abs() < f32::EPSILON);
    }

    #[test]
    fn handles_batched_output_shape() {
        let tensor = Tensor::from_shape(
            &[1, 1, 15],
            &[
                0.0f32, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95,
            ],
        )
        .unwrap();

        let detections = apply_postprocess(&tensor, 1.0, 1.0, &PostprocessConfig::default())
            .expect("postprocess should succeed");
        assert_eq!(detections.len(), 1);
    }
}
