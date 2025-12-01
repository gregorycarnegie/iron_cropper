//! Bounding box dragging and resizing functionality.

use super::coords::clamp_bbox_to_image;

use crate::{ActiveBoxDrag, DetectionWithQuality, DragHandle, PointerSnapshot, PreviewSpace};

use egui::{Rect, pos2, vec2};
use yunet_core::BoundingBox;

/// Gets the screen rectangle for a drag handle control.
pub fn handle_rect_for(rect: Rect, handle: DragHandle) -> Rect {
    let size = 12.0;
    let center = match handle {
        DragHandle::NorthWest => pos2(rect.left(), rect.top()),
        DragHandle::NorthEast => pos2(rect.right(), rect.top()),
        DragHandle::SouthWest => pos2(rect.left(), rect.bottom()),
        DragHandle::SouthEast => pos2(rect.right(), rect.bottom()),
        DragHandle::Move => rect.center(),
    };
    Rect::from_center_size(center, vec2(size, size))
}

/// Applies a drag operation to a bounding box.
/// Returns the updated bounding box clamped to image bounds.
pub fn apply_drag_to_bbox(
    active: ActiveBoxDrag,
    drag_delta: egui::Vec2,
    image_rect: Rect,
    image_size: (u32, u32),
) -> BoundingBox {
    if image_size.0 == 0 || image_size.1 == 0 {
        return active.start_bbox;
    }

    let scale_x = image_rect.width() / image_size.0 as f32;
    let scale_y = image_rect.height() / image_size.1 as f32;
    if scale_x <= 0.0 || scale_y <= 0.0 {
        return active.start_bbox;
    }

    let mut bbox = active.start_bbox;
    let delta_x = drag_delta.x / scale_x;
    let delta_y = drag_delta.y / scale_y;

    match active.handle {
        DragHandle::Move => {
            bbox.x += delta_x;
            bbox.y += delta_y;
        }
        DragHandle::NorthWest => {
            bbox.x += delta_x;
            bbox.y += delta_y;
            bbox.width -= delta_x;
            bbox.height -= delta_y;
        }
        DragHandle::NorthEast => {
            bbox.y += delta_y;
            bbox.width += delta_x;
            bbox.height -= delta_y;
        }
        DragHandle::SouthWest => {
            bbox.x += delta_x;
            bbox.width -= delta_x;
            bbox.height += delta_y;
        }
        DragHandle::SouthEast => {
            bbox.width += delta_x;
            bbox.height += delta_y;
        }
    }

    clamp_bbox_to_image(bbox, image_size)
}

/// Handles bounding box drag interactions (starting, updating, ending drags).
pub fn handle_bbox_drag_interactions(
    active_drag: &mut Option<ActiveBoxDrag>,
    detections: &mut [DetectionWithQuality],
    pointer: &PointerSnapshot,
    preview_space: &PreviewSpace,
) {
    let Some(pos) = pointer.pos else {
        return;
    };

    // If drag is active, update it
    if let Some(drag) = *active_drag {
        if pointer.down {
            // Continue dragging
            if let Some(origin) = pointer.press_origin {
                let drag_delta = pos - origin;
                let new_bbox = apply_drag_to_bbox(
                    drag,
                    drag_delta,
                    preview_space.rect,
                    preview_space.image_size,
                );
                if let Some(det) = detections.get_mut(drag.index) {
                    det.set_bbox(new_bbox);
                }
            }
        } else {
            // End drag
            *active_drag = None;
        }
        return;
    }

    // If not dragging and pointer is pressed, check if we should start a drag
    if pointer.pressed && preview_space.rect.contains(pos) {
        // Check each detection to see if we're clicking on a handle or the bbox
        for (idx, det) in detections.iter().enumerate() {
            let bbox = det.active_bbox();
            let bbox_rect = super::coords::bbox_to_screen_rect(
                bbox,
                preview_space.rect,
                preview_space.image_size,
            );

            // Check corner handles first (higher priority)
            let handles = [
                DragHandle::NorthWest,
                DragHandle::NorthEast,
                DragHandle::SouthWest,
                DragHandle::SouthEast,
            ];

            for handle in handles {
                let handle_rect = handle_rect_for(bbox_rect, handle);
                if handle_rect.contains(pos) {
                    *active_drag = Some(ActiveBoxDrag {
                        index: idx,
                        handle,
                        start_bbox: bbox,
                    });
                    return;
                }
            }

            // Check if clicking inside the bbox for move
            if bbox_rect.expand(-8.0).contains(pos) {
                *active_drag = Some(ActiveBoxDrag {
                    index: idx,
                    handle: DragHandle::Move,
                    start_bbox: bbox,
                });
                return;
            }
        }
    }
}
