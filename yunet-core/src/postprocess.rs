use anyhow::Result;
use std::cmp::Ordering;
use tract_onnx::prelude::{Tensor, tract_ndarray::ArrayView2};
use yunet_utils::{config::DetectionSettings, point::Point};

/// Canonical YuNet detection configuration.
///
/// These parameters control how raw model outputs are filtered and refined.
#[derive(Debug, Clone)]
pub struct PostprocessConfig {
    /// Minimum confidence score for a detection to be considered valid.
    pub score_threshold: f32,
    /// Threshold for non-maximum suppression to merge overlapping bounding boxes.
    pub nms_threshold: f32,
    /// The maximum number of detections to return after sorting by score.
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

/// Axis-aligned bounding box in image coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    /// The x-coordinate of the top-left corner.
    pub x: f32,
    /// The y-coordinate of the top-left corner.
    pub y: f32,
    /// The width of the box.
    pub width: f32,
    /// The height of the box.
    pub height: f32,
}

impl BoundingBox {
    /// Calculates the area of the bounding box.
    #[inline]
    pub fn area(&self) -> f32 {
        (self.width.max(0.0)) * (self.height.max(0.0))
    }

    /// Calculates the Intersection over Union (IoU) with another bounding box.
    #[inline]
    pub fn iou(&self, other: &Self) -> f32 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);

        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);

        let union = self.area() + other.area() - intersection;
        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }
}

/// Facial landmark coordinate (x, y) in image space.
pub type Landmark = Point;

/// A single YuNet detection result, including a bounding box, landmarks, and confidence score.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// The bounding box of the detected face.
    pub bbox: BoundingBox,
    /// An array of 5 facial landmarks (right eye, left eye, nose tip, right mouth corner, left mouth corner).
    pub landmarks: [Landmark; 5],
    /// The confidence score of the detection.
    pub score: f32,
}

/// Decode YuNet outputs into filtered detections.
///
/// This function takes the raw tensor output from the model and applies:
/// 1. Score filtering.
/// 2. Coordinate scaling to match the original image dimensions.
/// 3. Non-maximum suppression (NMS).
///
/// # Arguments
///
/// * `output` - The raw output tensor from the YuNet model.
/// * `scale_x` - The horizontal scale factor to map coordinates to the original image.
/// * `scale_y` - The vertical scale factor to map coordinates to the original image.
/// * `config` - The post-processing parameters.
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

    sort_by_score_desc(&mut detections, config.top_k);

    if config.nms_threshold > 0.0 && detections.len() > 1 {
        apply_nms_in_place(&mut detections, config.nms_threshold);
    }

    Ok(detections)
}

/// Extract the detection rows from the model's output tensor.
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

/// Apply non-maximum suppression to a list of detections.
fn sort_by_score_desc(detections: &mut Vec<Detection>, top_k: usize) {
    fn cmp(a: &Detection, b: &Detection) -> Ordering {
        b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal)
    }

    if top_k > 0 && detections.len() > top_k {
        let nth = top_k - 1;
        detections.select_nth_unstable_by(nth, cmp);
        let (head, _) = detections.split_at_mut(top_k);
        head.sort_unstable_by(cmp);
        detections.truncate(top_k);
    } else {
        detections.sort_unstable_by(cmp);
    }
}

fn apply_nms_in_place(detections: &mut Vec<Detection>, threshold: f32) {
    let len = detections.len();
    if len <= 1 {
        return;
    }

    // For small datasets, the overhead of building the grid outweighs the benefit.
    // The break-even point is typically around 100-200 items.
    if len < 200 {
        apply_nms_naive(detections, threshold);
        return;
    }

    // 1. Compute scene bounds
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    for d in detections.iter() {
        let b = d.bbox;
        if b.x < min_x {
            min_x = b.x;
        }
        if b.y < min_y {
            min_y = b.y;
        }
        // Use max to ensure width/height are accounted for
        let right = b.x + b.width;
        let bottom = b.y + b.height;
        if right > max_x {
            max_x = right;
        }
        if bottom > max_y {
            max_y = bottom;
        }
    }

    let width = max_x - min_x;
    let height = max_y - min_y;

    // Degenerate scene or empty area
    if width <= f32::EPSILON || height <= f32::EPSILON {
        apply_nms_naive(detections, threshold);
        return;
    }

    // 2. Setup Grid
    // 32x32 = 1024 cells. Efficient for up to ~10k items.
    const GRID_SIZE: usize = 32;
    let cell_w = width / GRID_SIZE as f32;
    let cell_h = height / GRID_SIZE as f32;

    // Use a flat vector of vectors to store indices
    // Pre-allocate assuming uniform distribution (len / cells) * safety_factor?
    // Just a small capacity is fine.
    let mut grid: Vec<Vec<usize>> = (0..GRID_SIZE * GRID_SIZE)
        .map(|_| Vec::with_capacity(len / (GRID_SIZE * GRID_SIZE / 4).max(1)))
        .collect();

    // 3. Populate Grid
    let grid_helper = |b: f32, cell_d: f32| -> usize {
        ((b) / cell_d).floor().clamp(0.0, (GRID_SIZE - 1) as f32) as usize
    };

    for (i, d) in detections.iter().enumerate() {
        let b = d.bbox;
        let c_min = grid_helper(b.x - min_x, cell_w);
        let c_max = grid_helper(b.x + b.width - min_x, cell_w);
        let r_min = grid_helper(b.y - min_y, cell_h);
        let r_max = grid_helper(b.y + b.height - min_y, cell_h);

        for r in r_min..=r_max {
            let row_offset = r * GRID_SIZE;
            for c in c_min..=c_max {
                grid[row_offset + c].push(i);
            }
        }
    }

    // 4. Suppression Loop
    let mut suppressed = vec![false; len];
    // check_token[j] == i means we already checked 'j' against 'i'
    let mut check_token = vec![usize::MAX; len];

    for i in 0..len {
        if suppressed[i] {
            continue;
        }

        let b = detections[i].bbox;
        let c_min = grid_helper(b.x - min_x, cell_w);
        let c_max = grid_helper(b.x + b.width - min_x, cell_w);
        let r_min = grid_helper(b.y - min_y, cell_h);
        let r_max = grid_helper(b.y + b.height - min_y, cell_h);

        for r in r_min..=r_max {
            let row_offset = r * GRID_SIZE;
            for c in c_min..=c_max {
                let cell = &grid[row_offset + c];
                for &candidate in cell {
                    // Only check candidates that appear later in the sorted list (lower score)
                    // and haven't been suppressed yet.
                    if candidate > i && !suppressed[candidate] {
                        // Avoid duplicate checks for the same candidate across different cells
                        if check_token[candidate] != i {
                            check_token[candidate] = i;

                            if b.iou(&detections[candidate].bbox) > threshold {
                                suppressed[candidate] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // 5. Compact the vector
    let mut keep = 0;
    for (i, &is_suppressed) in suppressed.iter().enumerate() {
        if !is_suppressed {
            if i != keep {
                detections.swap(i, keep);
            }
            keep += 1;
        }
    }
    detections.truncate(keep);
}

fn apply_nms_naive(detections: &mut Vec<Detection>, threshold: f32) {
    let len = detections.len();
    let mut suppressed = vec![false; len];
    let mut keep = 0;

    for i in 0..len {
        if suppressed[i] {
            continue;
        }

        if keep != i {
            detections.swap(keep, i);
            suppressed.swap(keep, i);
        }

        let reference_bbox = detections[keep].bbox;
        for j in (keep + 1)..len {
            if !suppressed[j] && reference_bbox.iou(&detections[j].bbox) > threshold {
                suppressed[j] = true;
            }
        }

        keep += 1;
    }

    detections.truncate(keep);
}

impl From<DetectionSettings> for PostprocessConfig {
    fn from(settings: DetectionSettings) -> Self {
        PostprocessConfig {
            score_threshold: settings.score_threshold,
            nms_threshold: settings.nms_threshold,
            top_k: settings.top_k,
        }
    }
}

impl From<&DetectionSettings> for PostprocessConfig {
    fn from(settings: &DetectionSettings) -> Self {
        settings.clone().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yunet_utils::config::DetectionSettings;

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

    #[test]
    fn converts_detection_settings_into_config() {
        let settings = DetectionSettings {
            score_threshold: 0.75,
            nms_threshold: 0.25,
            top_k: 123,
        };

        let config: PostprocessConfig = settings.clone().into();
        assert_eq!(config.score_threshold, settings.score_threshold);
        assert_eq!(config.nms_threshold, settings.nms_threshold);
        assert_eq!(config.top_k, settings.top_k);

        let config_from_ref: PostprocessConfig = (&settings).into();
        assert_eq!(config_from_ref.score_threshold, settings.score_threshold);
        assert_eq!(config_from_ref.nms_threshold, settings.nms_threshold);
        assert_eq!(config_from_ref.top_k, settings.top_k);
    }
}

#[cfg(test)]
mod benches {
    use super::*;
    use std::time::Instant;

    fn apply_nms_in_place_baseline(detections: &mut Vec<Detection>, threshold: f32) {
        let len = detections.len();
        if len <= 1 {
            return;
        }

        let mut suppressed = vec![false; len];
        let mut keep = 0;

        for i in 0..len {
            if suppressed[i] {
                continue;
            }

            if keep != i {
                detections.swap(keep, i);
                suppressed.swap(keep, i);
            }

            let reference_bbox = detections[keep].bbox;
            for j in (keep + 1)..len {
                if !suppressed[j] && reference_bbox.iou(&detections[j].bbox) > threshold {
                    suppressed[j] = true;
                }
            }

            keep += 1;
        }

        detections.truncate(keep);
    }

    struct SimpleRng(u64);
    impl SimpleRng {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 32) as u32) as f32 / 4294967296.0
        }
    }

    fn synthetic_detections(count: usize) -> Vec<Detection> {
        let mut out = Vec::with_capacity(count);
        let mut rng = SimpleRng(12345);
        for _ in 0..count {
            out.push(Detection {
                bbox: BoundingBox {
                    x: rng.next_f32() * 2000.0,
                    y: rng.next_f32() * 2000.0,
                    width: 20.0 + rng.next_f32() * 100.0,
                    height: 20.0 + rng.next_f32() * 100.0,
                },
                landmarks: [Landmark { x: 0.0, y: 0.0 }; 5],
                score: rng.next_f32(),
            });
        }
        // Essential: Sort by score descending to simulate real model output
        out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        out
    }

    #[test]
    #[ignore]
    fn bench_nms_variants() {
        let template = synthetic_detections(5_000);
        let iterations = 20;

        let mut optimized_total = std::time::Duration::ZERO;
        let mut baseline_total = std::time::Duration::ZERO;

        for i in 0..iterations {
            if i % 2 == 0 {
                let mut data = template.clone();
                let start = Instant::now();
                apply_nms_in_place(&mut data, 0.3);
                optimized_total += start.elapsed();

                let mut baseline = template.clone();
                let start = Instant::now();
                apply_nms_in_place_baseline(&mut baseline, 0.3);
                baseline_total += start.elapsed();
            } else {
                let mut baseline = template.clone();
                let start = Instant::now();
                apply_nms_in_place_baseline(&mut baseline, 0.3);
                baseline_total += start.elapsed();

                let mut data = template.clone();
                let start = Instant::now();
                apply_nms_in_place(&mut data, 0.3);
                optimized_total += start.elapsed();
            }
        }

        let diff = baseline_total.as_secs_f64() / optimized_total.as_secs_f64();
        println!(
            "NMS Benchmark (5k random items, {} iters):\n  Optimized (Grid): {:?}\n  Baseline (Naive): {:?}\n  Speedup:          {:.2}x",
            iterations,
            optimized_total / iterations as u32,
            baseline_total / iterations as u32,
            diff
        );
    }
}
