//! Quality assessment and filtering helpers.

use std::cmp::Ordering;
use std::collections::HashSet;

use egui::{Context as EguiContext, TextureOptions};
use image::DynamicImage;
use yunet_utils::quality::Quality;

use crate::DetectionWithQuality;

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
