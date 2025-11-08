//! Coordinate transformation utilities between screen space and image space.

use egui::{Pos2, Rect, pos2, vec2};
use yunet_core::BoundingBox;

/// Converts an image point to screen coordinates.
pub fn image_point_to_screen(point: Pos2, image_rect: Rect, image_size: (u32, u32)) -> Pos2 {
    if image_size.0 == 0 || image_size.1 == 0 {
        return pos2(image_rect.left(), image_rect.top());
    }
    let scale_x = image_rect.width() / image_size.0 as f32;
    let scale_y = image_rect.height() / image_size.1 as f32;
    pos2(
        image_rect.left() + point.x * scale_x,
        image_rect.top() + point.y * scale_y,
    )
}

/// Converts a bounding box from image coordinates to screen space.
pub fn bbox_to_screen_rect(bbox: BoundingBox, image_rect: Rect, image_size: (u32, u32)) -> Rect {
    if image_size.0 == 0 || image_size.1 == 0 {
        return Rect::from_min_size(image_rect.left_top(), vec2(0.0, 0.0));
    }
    let scale_x = image_rect.width() / image_size.0 as f32;
    let scale_y = image_rect.height() / image_size.1 as f32;
    let top_left = pos2(
        image_rect.left() + bbox.x * scale_x,
        image_rect.top() + bbox.y * scale_y,
    );
    let size = vec2(bbox.width * scale_x, bbox.height * scale_y);
    Rect::from_min_size(top_left, size)
}

/// Clamps a bounding box to fit within the image dimensions.
pub fn clamp_bbox_to_image(mut bbox: BoundingBox, image_size: (u32, u32)) -> BoundingBox {
    let img_w = image_size.0 as f32;
    let img_h = image_size.1 as f32;
    let min_size = 8.0;
    bbox.width = bbox.width.max(min_size).min(img_w.max(1.0));
    bbox.height = bbox.height.max(min_size).min(img_h.max(1.0));
    bbox.x = bbox.x.clamp(0.0, (img_w - bbox.width).max(0.0));
    bbox.y = bbox.y.clamp(0.0, (img_h - bbox.height).max(0.0));
    bbox
}
