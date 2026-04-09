use anyhow::Result;
use std::cmp::Ordering;
use tract_onnx::prelude::{Tensor, tract_ndarray::ArrayView2};
use yunet_utils::{config::DetectionSettings, point::Point};

/// Default minimum confidence score for a detection to be considered valid.
pub const DEFAULT_SCORE_THRESHOLD: f32 = 0.9;
/// Default threshold for non-maximum suppression to merge overlapping bounding boxes.
pub const DEFAULT_NMS_THRESHOLD: f32 = 0.3;
/// Default maximum number of detections to return after sorting by score.
pub const DEFAULT_TOP_K: usize = 5_000;

/// Number of columns in YuNet detection output (bbox + landmarks + score).
const DETECTION_OUTPUT_COLS: usize = 15;
/// Index of the confidence score in a detection row.
const DETECTION_SCORE_INDEX: usize = 14;
const GRID_SIZE: usize = 32;

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
            score_threshold: DEFAULT_SCORE_THRESHOLD,
            nms_threshold: DEFAULT_NMS_THRESHOLD,
            top_k: DEFAULT_TOP_K,
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
        rows.shape()[1] == DETECTION_OUTPUT_COLS,
        "YuNet output must have {} columns per detection",
        DETECTION_OUTPUT_COLS
    );

    let mut detections = Vec::with_capacity(rows.nrows());
    for row in rows.rows() {
        let score = row[DETECTION_SCORE_INDEX];
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
        [rows, DETECTION_OUTPUT_COLS] => *rows,
        [1, rows, DETECTION_OUTPUT_COLS] => *rows,
        other => anyhow::bail!(
            "YuNet output must have shape [N, {}] or [1, N, {}] (got {:?})",
            DETECTION_OUTPUT_COLS,
            DETECTION_OUTPUT_COLS,
            other
        ),
    };

    let slice = output
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("YuNet output is not f32: {e}"))?;

    ArrayView2::from_shape((rows, DETECTION_OUTPUT_COLS), slice)
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

#[derive(Clone, Copy, Debug, PartialEq)]
struct SceneBounds {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
}

impl SceneBounds {
    fn width(&self) -> f32 {
        self.max_x - self.min_x
    }

    fn height(&self) -> f32 {
        self.max_y - self.min_y
    }

    fn cell_range_for_bbox(&self, bbox: &BoundingBox, grid_size: usize) -> CellRange {
        let cell_w = self.width() / grid_size as f32;
        let cell_h = self.height() / grid_size as f32;
        CellRange {
            min_col: grid_cell_index(bbox.x - self.min_x, cell_w, grid_size),
            max_col: grid_cell_index(bbox.x + bbox.width - self.min_x, cell_w, grid_size),
            min_row: grid_cell_index(bbox.y - self.min_y, cell_h, grid_size),
            max_row: grid_cell_index(bbox.y + bbox.height - self.min_y, cell_h, grid_size),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CellRange {
    min_col: usize,
    max_col: usize,
    min_row: usize,
    max_row: usize,
}

struct SpatialGrid {
    grid_size: usize,
    bounds: SceneBounds,
    cells: Vec<Vec<usize>>,
}

fn compute_scene_bounds(detections: &[Detection]) -> Option<SceneBounds> {
    if detections.is_empty() {
        return None;
    }

    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    for detection in detections {
        let bbox = detection.bbox;
        if bbox.x < min_x {
            min_x = bbox.x;
        }
        if bbox.y < min_y {
            min_y = bbox.y;
        }

        let right = bbox.x + bbox.width;
        let bottom = bbox.y + bbox.height;
        if right > max_x {
            max_x = right;
        }
        if bottom > max_y {
            max_y = bottom;
        }
    }

    let bounds = SceneBounds {
        min_x,
        min_y,
        max_x,
        max_y,
    };

    if bounds.width() <= f32::EPSILON || bounds.height() <= f32::EPSILON {
        None
    } else {
        Some(bounds)
    }
}

fn grid_cell_index(offset: f32, cell_d: f32, grid_size: usize) -> usize {
    if cell_d <= f32::EPSILON {
        return 0;
    }

    (offset / cell_d).floor().clamp(0.0, (grid_size - 1) as f32) as usize
}

fn build_spatial_grid(detections: &[Detection], bounds: SceneBounds) -> SpatialGrid {
    let mut cells: Vec<Vec<usize>> = (0..GRID_SIZE * GRID_SIZE)
        .map(|_| Vec::with_capacity(detections.len() / (GRID_SIZE * GRID_SIZE / 4).max(1)))
        .collect();

    for (i, detection) in detections.iter().enumerate() {
        let range = bounds.cell_range_for_bbox(&detection.bbox, GRID_SIZE);
        for row in range.min_row..=range.max_row {
            let row_offset = row * GRID_SIZE;
            for col in range.min_col..=range.max_col {
                cells[row_offset + col].push(i);
            }
        }
    }

    SpatialGrid {
        grid_size: GRID_SIZE,
        bounds,
        cells,
    }
}

fn suppress_overlapping_candidates(
    detections: &[Detection],
    threshold: f32,
    grid: &SpatialGrid,
) -> Vec<bool> {
    let len = detections.len();
    let mut suppressed = vec![false; len];
    let mut check_token = vec![usize::MAX; len];

    for i in 0..len {
        if suppressed[i] {
            continue;
        }

        let bbox = detections[i].bbox;
        let range = grid.bounds.cell_range_for_bbox(&bbox, grid.grid_size);

        for row in range.min_row..=range.max_row {
            let row_offset = row * grid.grid_size;
            for col in range.min_col..=range.max_col {
                let cell = &grid.cells[row_offset + col];
                for &candidate in cell {
                    if candidate > i && !suppressed[candidate] {
                        if check_token[candidate] != i {
                            check_token[candidate] = i;
                            if bbox.iou(&detections[candidate].bbox) > threshold {
                                suppressed[candidate] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    suppressed
}

fn compact_unsuppressed_detections(detections: &mut Vec<Detection>, suppressed: &[bool]) {
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

    let Some(bounds) = compute_scene_bounds(detections) else {
        apply_nms_naive(detections, threshold);
        return;
    };

    let grid = build_spatial_grid(detections, bounds);
    let suppressed = suppress_overlapping_candidates(detections, threshold, &grid);
    compact_unsuppressed_detections(detections, &suppressed);
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

    fn detection_with_score(score: f32, bbox: BoundingBox) -> Detection {
        Detection {
            bbox,
            landmarks: [Landmark::new(0.0, 0.0); 5],
            score,
        }
    }

    #[test]
    fn bounding_box_area_and_iou_handle_overlap_and_degenerate_boxes() {
        let a = BoundingBox {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
        };
        let b = BoundingBox {
            x: 5.0,
            y: 5.0,
            width: 10.0,
            height: 10.0,
        };
        let degenerate = BoundingBox {
            x: 0.0,
            y: 0.0,
            width: -5.0,
            height: 2.0,
        };

        assert_eq!(a.area(), 100.0);
        assert_eq!(degenerate.area(), 0.0);
        assert!((a.iou(&b) - (25.0 / 175.0)).abs() < 1e-6);
        assert_eq!(a.iou(&degenerate), 0.0);
    }

    #[test]
    fn apply_postprocess_rejects_invalid_tensor_shape_and_type() {
        let wrong_shape = Tensor::from_shape(&[15], &[0.0f32; 15]).unwrap();
        let err = apply_postprocess(&wrong_shape, 1.0, 1.0, &PostprocessConfig::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("YuNet output must have shape"));

        let wrong_type = Tensor::from_shape(&[1, 15], &[1i32; 15]).unwrap();
        let err = apply_postprocess(&wrong_type, 1.0, 1.0, &PostprocessConfig::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("YuNet output is not f32"));
    }

    #[test]
    fn sort_by_score_desc_honors_top_k_and_zero_as_unlimited() {
        let mut detections = vec![
            detection_with_score(
                0.1,
                BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                },
            ),
            detection_with_score(
                0.9,
                BoundingBox {
                    x: 2.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                },
            ),
            detection_with_score(
                0.5,
                BoundingBox {
                    x: 4.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                },
            ),
        ];

        sort_by_score_desc(&mut detections, 2);
        assert_eq!(detections.len(), 2);
        assert!((detections[0].score - 0.9).abs() < f32::EPSILON);
        assert!((detections[1].score - 0.5).abs() < f32::EPSILON);

        let mut detections = vec![
            detection_with_score(
                0.1,
                BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                },
            ),
            detection_with_score(
                0.9,
                BoundingBox {
                    x: 2.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                },
            ),
            detection_with_score(
                0.5,
                BoundingBox {
                    x: 4.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                },
            ),
        ];
        sort_by_score_desc(&mut detections, 0);
        assert_eq!(detections.len(), 3);
        assert!((detections[0].score - 0.9).abs() < f32::EPSILON);
        assert!((detections[1].score - 0.5).abs() < f32::EPSILON);
        assert!((detections[2].score - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn compute_scene_bounds_returns_none_for_degenerate_scene() {
        let detections = vec![
            detection_with_score(
                1.0,
                BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 0.0,
                    height: 0.0,
                },
            ),
            detection_with_score(
                0.9,
                BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 0.0,
                    height: 0.0,
                },
            ),
        ];

        assert!(compute_scene_bounds(&detections).is_none());
    }

    #[test]
    fn build_spatial_grid_covers_all_cells_touched_by_bbox() {
        let detections = vec![
            detection_with_score(
                1.0,
                BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 96.0,
                    height: 96.0,
                },
            ),
            detection_with_score(
                0.9,
                BoundingBox {
                    x: 48.0,
                    y: 48.0,
                    width: 1.0,
                    height: 1.0,
                },
            ),
        ];
        let bounds = compute_scene_bounds(&detections).expect("scene should be valid");
        let grid = build_spatial_grid(&detections, bounds);

        let range = grid
            .bounds
            .cell_range_for_bbox(&detections[0].bbox, grid.grid_size);
        for row in range.min_row..=range.max_row {
            for col in range.min_col..=range.max_col {
                assert!(
                    grid.cells[row * grid.grid_size + col].contains(&0),
                    "cell ({row}, {col}) should contain detection 0"
                );
            }
        }
    }

    #[test]
    fn compact_unsuppressed_detections_preserves_kept_order() {
        let mut detections = vec![
            detection_with_score(
                0.9,
                BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                },
            ),
            detection_with_score(
                0.8,
                BoundingBox {
                    x: 20.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                },
            ),
            detection_with_score(
                0.7,
                BoundingBox {
                    x: 40.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                },
            ),
        ];
        let suppressed = vec![false, true, false];

        compact_unsuppressed_detections(&mut detections, &suppressed);

        assert_eq!(detections.len(), 2);
        assert!((detections[0].score - 0.9).abs() < f32::EPSILON);
        assert!((detections[1].score - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn apply_nms_in_place_uses_large_grid_path_for_large_inputs() {
        let mut detections: Vec<_> = (0..200)
            .map(|i| {
                detection_with_score(
                    1.0 - i as f32 * 0.001,
                    BoundingBox {
                        x: i as f32 * 20.0,
                        y: 0.0,
                        width: 10.0,
                        height: 10.0,
                    },
                )
            })
            .collect();
        detections.push(detection_with_score(
            0.0,
            BoundingBox {
                x: 1.0,
                y: 1.0,
                width: 10.0,
                height: 10.0,
            },
        ));

        sort_by_score_desc(&mut detections, 0);
        apply_nms_in_place(&mut detections, 0.3);

        assert_eq!(detections.len(), 200);
        assert!((detections[0].score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn apply_nms_in_place_handles_zero_and_one_items() {
        let mut empty: Vec<Detection> = vec![];
        apply_nms_in_place(&mut empty, 0.3);
        assert_eq!(empty.len(), 0);

        let single = detection_with_score(
            0.9,
            BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 10.0,
                height: 10.0,
            },
        );
        let mut one = vec![single];
        apply_nms_in_place(&mut one, 0.3);
        assert_eq!(one.len(), 1);
    }

    #[test]
    fn apply_nms_naive_handles_zero_and_one_items() {
        let mut empty: Vec<Detection> = vec![];
        apply_nms_naive(&mut empty, 0.3);
        assert_eq!(empty.len(), 0);

        let mut one = vec![detection_with_score(
            0.9,
            BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 5.0,
                height: 5.0,
            },
        )];
        apply_nms_naive(&mut one, 0.3);
        assert_eq!(one.len(), 1);
    }

    #[test]
    fn apply_postprocess_filters_non_finite_scores() {
        // Rows with NaN/Inf scores should be dropped regardless of threshold.
        let tensor = tensor_from_rows(&[
            [
                0.0,
                0.0,
                5.0,
                5.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                f32::NAN,
            ],
            [
                0.0,
                0.0,
                5.0,
                5.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                f32::INFINITY,
            ],
            [
                0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95,
            ],
        ]);
        let detections = apply_postprocess(
            &tensor,
            1.0,
            1.0,
            &PostprocessConfig {
                score_threshold: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(detections.len(), 1);
        assert!((detections[0].score - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn apply_postprocess_drops_zero_area_boxes() {
        let tensor = tensor_from_rows(&[
            // zero width
            [
                0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95,
            ],
            // zero height
            [
                0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95,
            ],
            // valid box
            [
                1.0, 1.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9,
            ],
        ]);
        let detections = apply_postprocess(
            &tensor,
            1.0,
            1.0,
            &PostprocessConfig {
                score_threshold: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(detections.len(), 1);
        assert!((detections[0].bbox.x - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn apply_nms_in_place_degenerate_scene_falls_back_to_naive() {
        // All boxes at the same point → zero scene extent, triggers naive fallback.
        let mut detections: Vec<_> = (0..5)
            .map(|i| {
                detection_with_score(
                    0.9 - i as f32 * 0.1,
                    BoundingBox {
                        x: 0.0,
                        y: 0.0,
                        width: 0.0,
                        height: 0.0,
                    },
                )
            })
            .collect();
        // Should not panic.
        apply_nms_in_place(&mut detections, 0.3);
    }

    #[test]
    fn iou_returns_zero_when_union_underflows_to_zero() {
        // Both boxes have positive but subnormal dimensions: area() underflows to 0.0 in f32,
        // so union = 0 + 0 - 0 = 0 → covers the `union > 0` else branch.
        let tiny = f32::MIN_POSITIVE;
        let a = BoundingBox {
            x: 0.0,
            y: 0.0,
            width: tiny,
            height: tiny,
        };
        let b = BoundingBox {
            x: 0.0,
            y: 0.0,
            width: tiny,
            height: tiny,
        };
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn grid_nms_degenerate_scene_falls_back_to_naive_for_large_input() {
        // 200+ items all with width=0 and height=0 → scene extent is zero → degenerate path.
        let mut detections: Vec<_> = (0..201)
            .map(|i| {
                detection_with_score(
                    1.0 - i as f32 * 0.004,
                    BoundingBox {
                        x: 0.0,
                        y: i as f32 * 10.0,
                        width: 0.0,
                        height: 0.0,
                    },
                )
            })
            .collect();
        // Should not panic and should survive (all non-overlapping zero-area boxes).
        apply_nms_in_place(&mut detections, 0.3);
        assert_eq!(detections.len(), 201);
    }

    #[test]
    fn grid_nms_compaction_swap_fires_when_suppressed_item_in_middle() {
        // Build 201 detections so the grid path is used (len >= 200).
        // Detection at index 1 overlaps detection at index 0 and is suppressed.
        // When compacting, detection at index 2 ends up at keep=1 → swap fires (line 342).
        let mut detections = Vec::with_capacity(201);

        // Index 0: highest score, box at x=0
        detections.push(detection_with_score(
            1.0,
            BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 50.0,
                height: 50.0,
            },
        ));
        // Index 1: overlaps index 0 heavily (will be suppressed)
        detections.push(detection_with_score(
            0.999,
            BoundingBox {
                x: 5.0,
                y: 5.0,
                width: 50.0,
                height: 50.0,
            },
        ));
        // Indices 2..200: non-overlapping boxes spread across the scene
        for i in 2..201 {
            detections.push(detection_with_score(
                1.0 - i as f32 * 0.004,
                BoundingBox {
                    x: i as f32 * 200.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                },
            ));
        }

        // Already sorted by score descending; NMS threshold low enough to suppress index 1.
        apply_nms_in_place(&mut detections, 0.3);

        // Index 1 was suppressed, all others survive.
        assert_eq!(detections.len(), 200);
        // Highest-score detection still first.
        assert!((detections[0].score - 1.0).abs() < f32::EPSILON);
    }
}

#[cfg(test)]
mod benches {
    use super::*;
    use std::time::{Duration, Instant};

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
                    width: rng.next_f32().mul_add(100.0, 20.0),
                    height: rng.next_f32().mul_add(100.0, 20.0),
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

        let mut optimized_total = Duration::ZERO;
        let mut baseline_total = Duration::ZERO;

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
