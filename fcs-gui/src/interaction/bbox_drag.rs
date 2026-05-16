//! Bounding box drag/resize interaction.

use crate::types::{ActiveBoxDrag, DragHandle};
use egui::{Pos2, Rect};
use fcs_core::BoundingBox;

pub const HANDLE_SIZE: f32 = 9.0;

/// Returns which drag handle (if any) is under `pos` for a bounding box `rect`.
pub fn hit_test_handle(rect: Rect, pos: Pos2) -> Option<DragHandle> {
    let _hs = HANDLE_SIZE / 2.0;
    let corners = [
        (DragHandle::NorthWest, Pos2::new(rect.min.x, rect.min.y)),
        (DragHandle::NorthEast, Pos2::new(rect.max.x, rect.min.y)),
        (DragHandle::SouthWest, Pos2::new(rect.min.x, rect.max.y)),
        (DragHandle::SouthEast, Pos2::new(rect.max.x, rect.max.y)),
    ];
    for (handle, corner) in corners {
        let h_rect = Rect::from_center_size(corner, egui::Vec2::splat(HANDLE_SIZE + 4.0));
        if h_rect.contains(pos) {
            return Some(handle);
        }
    }
    if rect.shrink(4.0).contains(pos) {
        return Some(DragHandle::Move);
    }
    None
}

/// Apply a drag delta to a bounding box in image pixel space.
pub fn apply_drag(
    drag: &ActiveBoxDrag,
    delta_px: egui::Vec2,
    img_w: f32,
    img_h: f32,
) -> BoundingBox {
    let mut bbox = drag.start_bbox;
    match drag.handle {
        DragHandle::Move => {
            bbox.x = (bbox.x + delta_px.x).clamp(0.0, img_w - bbox.width);
            bbox.y = (bbox.y + delta_px.y).clamp(0.0, img_h - bbox.height);
        }
        DragHandle::NorthWest => {
            move_left_edge(&mut bbox, delta_px.x);
            move_top_edge(&mut bbox, delta_px.y);
        }
        DragHandle::NorthEast => {
            move_top_edge(&mut bbox, delta_px.y);
            extend_right_edge(&mut bbox, delta_px.x, img_w);
        }
        DragHandle::SouthWest => {
            move_left_edge(&mut bbox, delta_px.x);
            extend_bottom_edge(&mut bbox, delta_px.y, img_h);
        }
        DragHandle::SouthEast => {
            extend_right_edge(&mut bbox, delta_px.x, img_w);
            extend_bottom_edge(&mut bbox, delta_px.y, img_h);
        }
    }
    bbox
}

/// Minimum width/height a corner drag is allowed to produce; corners cannot
/// cross over to invert the box.
const MIN_BBOX_EXTENT: f32 = 10.0;

fn move_left_edge(bbox: &mut BoundingBox, delta_x: f32) {
    let new_x = (bbox.x + delta_x).clamp(0.0, bbox.x + bbox.width - MIN_BBOX_EXTENT);
    bbox.width += bbox.x - new_x;
    bbox.x = new_x;
}

fn move_top_edge(bbox: &mut BoundingBox, delta_y: f32) {
    let new_y = (bbox.y + delta_y).clamp(0.0, bbox.y + bbox.height - MIN_BBOX_EXTENT);
    bbox.height += bbox.y - new_y;
    bbox.y = new_y;
}

fn extend_right_edge(bbox: &mut BoundingBox, delta_x: f32, img_w: f32) {
    bbox.width = (bbox.width + delta_x).clamp(MIN_BBOX_EXTENT, img_w - bbox.x);
}

fn extend_bottom_edge(bbox: &mut BoundingBox, delta_y: f32, img_h: f32) {
    bbox.height = (bbox.height + delta_y).clamp(MIN_BBOX_EXTENT, img_h - bbox.y);
}
