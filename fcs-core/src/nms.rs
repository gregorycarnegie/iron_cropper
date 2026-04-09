use crate::postprocess::{BoundingBox, Detection};

const GRID_SIZE: usize = 32;

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
                    if candidate > i
                        && !suppressed[candidate]
                        && check_token[candidate] != i
                    {
                        check_token[candidate] = i;
                        if bbox.iou(&detections[candidate].bbox) > threshold {
                            suppressed[candidate] = true;
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

pub(crate) fn apply_nms_in_place(detections: &mut Vec<Detection>, threshold: f32) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postprocess::{BoundingBox, Detection, Landmark};

    fn detection_with_score(score: f32, bbox: BoundingBox) -> Detection {
        Detection {
            bbox,
            landmarks: [Landmark::new(0.0, 0.0); 5],
            score,
        }
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
        // Already in descending score order; add overlapping low-score box last.
        detections.push(detection_with_score(
            0.0,
            BoundingBox {
                x: 1.0,
                y: 1.0,
                width: 10.0,
                height: 10.0,
            },
        ));

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
    use crate::postprocess::{BoundingBox, Detection, Landmark};
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
