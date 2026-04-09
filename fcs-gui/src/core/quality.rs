//! Quality assessment and filtering helpers.

use crate::DetectionWithQuality;

use egui::{Context as EguiContext, TextureOptions};
use fcs_utils::quality::Quality;
use image::DynamicImage;
use std::{cmp::Ordering, collections::HashSet};

/// Applies quality rules to preview detections and updates selection.
pub fn apply_quality_rules_to_preview(
    detections: &[DetectionWithQuality],
    selected_faces: &mut HashSet<usize>,
    auto_skip_no_high_quality: bool,
    auto_select_best_face: bool,
) -> bool {
    selected_faces.clear();
    if detections.is_empty() {
        return false;
    }

    let best_quality = detections.iter().map(|d| d.quality).max();

    if auto_skip_no_high_quality && best_quality != Some(Quality::High) {
        return false;
    }

    if auto_select_best_face
        && detections.len() > 1
        && let Some((best_idx, _)) = detections.iter().enumerate().max_by(|a, b| {
            a.1.quality.cmp(&b.1.quality).then_with(|| {
                a.1.quality_score
                    .partial_cmp(&b.1.quality_score)
                    .unwrap_or(Ordering::Equal)
            })
        })
    {
        selected_faces.insert(best_idx);
        return true;
    }

    for idx in 0..detections.len() {
        selected_faces.insert(idx);
    }
    true
}

/// Refreshes the detection thumbnail at the given index.
pub fn refresh_detection_thumbnail_at(
    ctx: &EguiContext,
    detections: &mut [DetectionWithQuality],
    index: usize,
    image: &DynamicImage,
    texture_seq: &mut u64,
) {
    let Some(det) = detections.get_mut(index) else {
        return;
    };
    let bbox = det.active_bbox();

    let mut x = bbox.x.max(0.0) as u32;
    let mut y = bbox.y.max(0.0) as u32;
    let mut w = bbox.width.max(1.0) as u32;
    let mut h = bbox.height.max(1.0) as u32;

    let img_w = image.width();
    let img_h = image.height();
    x = x.min(img_w.saturating_sub(1));
    y = y.min(img_h.saturating_sub(1));
    w = w.min(img_w.saturating_sub(x));
    h = h.min(img_h.saturating_sub(y));

    let face_region = image.crop_imm(x, y, w, h);
    let thumb = face_region.resize(96, 96, image::imageops::FilterType::Lanczos3);
    let thumb_rgba = thumb.to_rgba8();
    let thumb_size = [thumb_rgba.width() as usize, thumb_rgba.height() as usize];
    let thumb_color = egui::ColorImage::from_rgba_unmultiplied(thumb_size, thumb_rgba.as_raw());
    let texture_name = format!("yunet-face-thumb-{}-{}", texture_seq, index);
    *texture_seq = texture_seq.wrapping_add(1);
    let texture = ctx.load_texture(texture_name, thumb_color, TextureOptions::LINEAR);
    det.thumbnail = Some(texture);
}

/// Resets a detection's bounding box to its original state.
pub fn reset_detection_bbox(detections: &mut [DetectionWithQuality], index: usize) {
    if let Some(det) = detections.get_mut(index) {
        det.reset_bbox();
    }
}

/// Removes a detection and updates the selection set accordingly.
pub fn remove_detection(
    detections: &mut Vec<DetectionWithQuality>,
    selected_faces: &mut HashSet<usize>,
    index: usize,
) {
    if index >= detections.len() {
        return;
    }
    detections.remove(index);
    let mut new_selection = HashSet::new();
    for face_idx in selected_faces.iter().copied() {
        if face_idx == index {
            continue;
        } else if face_idx > index {
            new_selection.insert(face_idx - 1);
        } else {
            new_selection.insert(face_idx);
        }
    }
    *selected_faces = new_selection;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DetectionOrigin, DetectionWithQuality};
    use fcs_core::{BoundingBox, Detection, Landmark};
    use fcs_utils::quality::Quality;
    use image::{DynamicImage, Rgba, RgbaImage};
    use std::collections::HashSet;

    fn sample_detection(
        bbox: BoundingBox,
        quality: Quality,
        quality_score: f64,
    ) -> DetectionWithQuality {
        DetectionWithQuality {
            detection: Detection {
                bbox,
                landmarks: [Landmark::new(10.0, 10.0); 5],
                score: 0.95,
            },
            quality_score,
            quality,
            thumbnail: None,
            current_bbox: bbox,
            original_bbox: bbox,
            origin: DetectionOrigin::Detector,
        }
    }

    #[test]
    fn apply_quality_rules_to_preview_returns_false_for_empty_or_low_quality_sets() {
        let mut selected_faces = HashSet::from([99usize]);
        assert!(!apply_quality_rules_to_preview(
            &[],
            &mut selected_faces,
            false,
            false
        ));
        assert!(selected_faces.is_empty());

        let detections = vec![
            sample_detection(
                BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                },
                Quality::Medium,
                12.0,
            ),
            sample_detection(
                BoundingBox {
                    x: 20.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                },
                Quality::Low,
                5.0,
            ),
        ];

        assert!(!apply_quality_rules_to_preview(
            &detections,
            &mut selected_faces,
            true,
            false
        ));
        assert!(selected_faces.is_empty());
    }

    #[test]
    fn apply_quality_rules_to_preview_can_select_all_or_best_face() {
        let detections = vec![
            sample_detection(
                BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                },
                Quality::Medium,
                50.0,
            ),
            sample_detection(
                BoundingBox {
                    x: 20.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                },
                Quality::High,
                10.0,
            ),
            sample_detection(
                BoundingBox {
                    x: 40.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                },
                Quality::High,
                20.0,
            ),
        ];

        let mut selected_faces = HashSet::new();
        assert!(apply_quality_rules_to_preview(
            &detections,
            &mut selected_faces,
            false,
            false
        ));
        assert_eq!(selected_faces, HashSet::from([0usize, 1usize, 2usize]));

        assert!(apply_quality_rules_to_preview(
            &detections,
            &mut selected_faces,
            false,
            true
        ));
        assert_eq!(selected_faces, HashSet::from([2usize]));
    }

    #[test]
    fn refresh_detection_thumbnail_at_clamps_bbox_and_updates_texture_sequence() {
        let ctx = EguiContext::default();
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(32, 24, Rgba([1, 2, 3, 255])));
        let mut detections = vec![sample_detection(
            BoundingBox {
                x: -10.0,
                y: -5.0,
                width: 100.0,
                height: 100.0,
            },
            Quality::High,
            100.0,
        )];
        let mut texture_seq = 7u64;

        refresh_detection_thumbnail_at(&ctx, &mut detections, 0, &image, &mut texture_seq);

        assert!(detections[0].thumbnail.is_some());
        assert_eq!(texture_seq, 8);

        refresh_detection_thumbnail_at(&ctx, &mut detections, 99, &image, &mut texture_seq);
        assert_eq!(texture_seq, 8);
    }

    #[test]
    fn reset_and_remove_detection_update_state_consistently() {
        let bbox = BoundingBox {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
        };
        let mut detections = vec![
            sample_detection(bbox, Quality::Low, 1.0),
            sample_detection(
                BoundingBox {
                    x: 20.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                },
                Quality::Medium,
                2.0,
            ),
            sample_detection(
                BoundingBox {
                    x: 40.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                },
                Quality::High,
                3.0,
            ),
        ];
        detections[1].current_bbox = BoundingBox {
            x: 25.0,
            y: 5.0,
            width: 8.0,
            height: 8.0,
        };

        reset_detection_bbox(&mut detections, 1);
        assert_eq!(detections[1].current_bbox, detections[1].original_bbox);

        let mut selected_faces = HashSet::from([0usize, 2usize]);
        remove_detection(&mut detections, &mut selected_faces, 1);

        assert_eq!(detections.len(), 2);
        assert_eq!(selected_faces, HashSet::from([0usize, 1usize]));
    }
}
