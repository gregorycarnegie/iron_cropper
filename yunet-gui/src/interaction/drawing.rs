//! Manual bounding box drawing functionality.

use super::coords::clamp_bbox_to_image;

use crate::types::{
    DetectionOrigin, DetectionWithQuality, ManualBoxDraft, PointerSnapshot, PreviewSpace,
};
use yunet_utils::quality::Quality;

use egui::{Pos2, Rect};
use yunet_core::{BoundingBox, Detection, Landmark};

/// Converts a draft box (with start and current positions) into a bounding box.
/// Returns None if the box is too small.
pub fn draft_to_bbox(draft: ManualBoxDraft, image_size: (u32, u32)) -> Option<BoundingBox> {
    let x1 = draft.start.x.min(draft.current.x);
    let x2 = draft.start.x.max(draft.current.x);
    let y1 = draft.start.y.min(draft.current.y);
    let y2 = draft.start.y.max(draft.current.y);
    let width = (x2 - x1).abs();
    let height = (y2 - y1).abs();
    if width < 8.0 || height < 8.0 {
        return None;
    }
    let bbox = BoundingBox {
        x: x1,
        y: y1,
        width,
        height,
    };
    Some(clamp_bbox_to_image(bbox, image_size))
}

/// Creates placeholder landmarks for a manually drawn bounding box.
pub fn placeholder_landmarks(bbox: BoundingBox) -> [Landmark; 5] {
    [
        Landmark {
            x: bbox.width.mul_add(0.3, bbox.x),
            y: bbox.height.mul_add(0.35, bbox.y),
        },
        Landmark {
            x: bbox.width.mul_add(0.7, bbox.x),
            y: bbox.height.mul_add(0.35, bbox.y),
        },
        Landmark {
            x: bbox.width.mul_add(0.5, bbox.x),
            y: bbox.height.mul_add(0.5, bbox.y),
        },
        Landmark {
            x: bbox.width.mul_add(0.35, bbox.x),
            y: bbox.height.mul_add(0.65, bbox.y),
        },
        Landmark {
            x: bbox.width.mul_add(0.65, bbox.x),
            y: bbox.height.mul_add(0.65, bbox.y),
        },
    ]
}

/// Checks if the pointer is over any existing bounding box.
pub fn pointer_over_any_bbox(
    detections: &[DetectionWithQuality],
    image_rect: Rect,
    image_size: (u32, u32),
    pointer: Pos2,
) -> bool {
    if !image_rect.contains(pointer) {
        return false;
    }
    detections.iter().any(|det| {
        super::coords::bbox_to_screen_rect(det.active_bbox(), image_rect, image_size)
            .expand(8.0)
            .contains(pointer)
    })
}

/// Creates a DetectionWithQuality from a manually drawn bounding box.
pub fn create_manual_detection(bbox: BoundingBox) -> DetectionWithQuality {
    let landmarks = placeholder_landmarks(bbox);
    let detection = Detection {
        bbox,
        landmarks,
        score: 1.0,
    };
    DetectionWithQuality {
        detection,
        quality_score: 0.0,
        quality: Quality::High,
        thumbnail: None,
        current_bbox: bbox,
        original_bbox: bbox,
        origin: DetectionOrigin::Manual,
    }
}

/// Updates the manual box draft state based on pointer interactions.
/// Handles starting, updating, and finalizing manual bounding box drawing.
pub fn update_manual_box_draft(
    draft: &mut Option<ManualBoxDraft>,
    detections: &mut Vec<DetectionWithQuality>,
    pointer: &PointerSnapshot,
    preview_space: &PreviewSpace,
) {
    let Some(pos) = pointer.pos else {
        return;
    };

    if !preview_space.rect.contains(pos) {
        if pointer.released {
            *draft = None;
        }
        return;
    }

    // Convert screen position to image coordinates
    let image_x = ((pos.x - preview_space.rect.left()) / preview_space.rect.width())
        * preview_space.image_size.0 as f32;
    let image_y = ((pos.y - preview_space.rect.top()) / preview_space.rect.height())
        * preview_space.image_size.1 as f32;
    let image_pos = egui::pos2(image_x, image_y);

    if pointer.pressed {
        // Don't start drawing if pointer is over an existing bbox
        if !pointer_over_any_bbox(
            detections,
            preview_space.rect,
            preview_space.image_size,
            pos,
        ) {
            *draft = Some(ManualBoxDraft {
                start: image_pos,
                current: image_pos,
            });
        }
    } else if pointer.down {
        if let Some(d) = draft.as_mut() {
            d.current = image_pos;
        }
    } else if pointer.released
        && let Some(d) = *draft
    {
        if let Some(bbox) = draft_to_bbox(d, preview_space.image_size) {
            let new_detection = create_manual_detection(bbox);
            detections.push(new_detection);
        }
        *draft = None;
    }
}
