//! Coordinate transforms between image space and screen space.

use egui::{Pos2, Rect};

/// Maps a point in normalized image space (0..1) to screen rect coordinates.
pub fn image_to_screen(norm_pt: Pos2, screen_rect: Rect) -> Pos2 {
    Pos2::new(
        screen_rect.min.x + norm_pt.x * screen_rect.width(),
        screen_rect.min.y + norm_pt.y * screen_rect.height(),
    )
}

/// Maps a screen point back to normalized image space.
pub fn screen_to_image(screen_pt: Pos2, screen_rect: Rect) -> Pos2 {
    Pos2::new(
        (screen_pt.x - screen_rect.min.x) / screen_rect.width(),
        (screen_pt.y - screen_rect.min.y) / screen_rect.height(),
    )
}

/// Convert a bounding box (x, y, w, h in pixels) to a normalized [0..1] Rect.
pub fn bbox_to_norm_rect(x: f32, y: f32, w: f32, h: f32, img_w: f32, img_h: f32) -> Rect {
    Rect::from_min_max(
        Pos2::new(x / img_w, y / img_h),
        Pos2::new((x + w) / img_w, (y + h) / img_h),
    )
}

/// Convert a normalized Rect to screen-space Rect.
pub fn norm_to_screen(norm: Rect, screen: Rect) -> Rect {
    Rect::from_min_max(
        image_to_screen(norm.min, screen),
        image_to_screen(norm.max, screen),
    )
}
