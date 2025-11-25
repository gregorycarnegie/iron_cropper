use egui::{Align2, Color32, FontId, Rect, Response, Sense, Ui, emath, lerp, pos2, remap, vec2};

/// A custom slider with specific styling:
/// - Dark rounded background container
/// - Colored active track, White inactive track
/// - Colored circular thumb
pub fn custom_slider<Num: emath::Numeric>(
    ui: &mut Ui,
    value: &mut Num,
    range: std::ops::RangeInclusive<Num>,
    text: Option<&str>,
    accent_color: Option<Color32>,
) -> Response {
    let min_width = 180.0;
    let desired_size = vec2(ui.available_width().max(min_width), 24.0);
    let (rect, mut response) = ui.allocate_exact_size(desired_size, Sense::click_and_drag());

    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
    }
    if response.dragged() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
    }

    if response.dragged() || response.clicked() {
        let min = range.start().to_f64();
        let max = range.end().to_f64();

        if let Some(pointer_pos) = response.interact_pointer_pos() {
            // Calculate value from mouse position
            // The slider track is inside the container, let's give it some padding
            let padding = 8.0;
            let track_rect = rect.shrink2(vec2(padding, 0.0));

            let new_val = remap(
                pointer_pos.x as f64,
                (track_rect.left() as f64)..=(track_rect.right() as f64),
                min..=max,
            );
            *value = Num::from_f64(new_val.clamp(min, max));
            response.mark_changed();
        }
    }

    if ui.is_rect_visible(rect) {
        let palette = crate::theme::palette();
        let accent = accent_color.unwrap_or(palette.accent);

        // 1. Draw Container (Dark rounded background)
        // Using a very dark color for the "cutout" look
        let container_color = Color32::from_rgb(14, 20, 30);
        ui.painter()
            .rect_filled(rect, egui::CornerRadius::same(12), container_color);

        // 2. Draw Track
        // The track sits in the middle vertically
        let padding = 8.0;
        let track_height = 6.0;
        let track_rect = Rect::from_min_max(
            pos2(rect.left() + padding, rect.center().y - track_height / 2.0),
            pos2(rect.right() - padding, rect.center().y + track_height / 2.0),
        );

        let min = range.start().to_f64();
        let max = range.end().to_f64();
        let val = value.to_f64();

        // Normalize value to 0..1
        let t = remap(val, min..=max, 0.0..=1.0).clamp(0.0, 1.0);

        // Calculate the split point
        let split_x = lerp(track_rect.left()..=track_rect.right(), t as f32);

        // Active part (Left) - Colored
        let active_rect = Rect::from_min_max(track_rect.min, pos2(split_x, track_rect.max.y));

        // Inactive part (Right) - White/Light Gray
        let inactive_rect = Rect::from_min_max(pos2(split_x, track_rect.min.y), track_rect.max);

        // Draw Inactive Track (Right side)
        ui.painter().rect_filled(
            inactive_rect,
            egui::CornerRadius::same(3), // Fully rounded caps (6.0 / 2)
            Color32::from_gray(220),     // Bright/White-ish
        );

        // Draw Active Track (Left side)
        ui.painter()
            .rect_filled(active_rect, egui::CornerRadius::same(3), accent);

        // 3. Draw Thumb (Circle)
        let thumb_radius = 8.0;
        let thumb_center = pos2(split_x, rect.center().y);

        ui.painter()
            .circle_filled(thumb_center, thumb_radius, accent);

        // Optional: Add a small border or shadow to the thumb to make it pop?
        // The reference image has a clean blue circle.

        // 4. Draw Text (Label)
        // If text is provided, we can draw it inside or outside.
        // Standard slider puts it inside if it fits, or to the right.
        // Given the container look, maybe we overlay it or just rely on the caller to put a label above/beside.
        // The reference image doesn't show text *on* the slider.
        // But our existing usage passes `.text("Score threshold")`.
        // Let's render the text on top, centered or to the left, if provided.

        if let Some(text) = text {
            let text_color = if t > 0.5 {
                Color32::WHITE
            } else {
                palette.panel_dark
            };

            // Let's just put it in the center for now, or maybe to the right if we want to mimic standard slider behavior?
            // Actually, the user's design shows NO text on the slider itself.
            // But we need to show the label somewhere.
            // For now, let's draw it centered in the container, with a shadow for contrast.

            ui.painter().text(
                rect.center(),
                Align2::CENTER_CENTER,
                text,
                FontId::proportional(12.0),
                text_color,
            );
        }
    }

    response.on_hover_cursor(egui::CursorIcon::Grab)
}

/// Helper function to render a slider row with label and drag value.
pub fn slider_row<Num: emath::Numeric>(
    ui: &mut Ui,
    value: &mut Num,
    range: std::ops::RangeInclusive<Num>,
    label: &str,
    speed: f64,
    hover_text: Option<&str>,
    accent_color: Option<Color32>,
) -> bool {
    ui.label(label);
    let mut changed = false;
    ui.horizontal(|ui| {
        let drag_width = 50.0;
        let spacing = 8.0;
        let total_width = ui.available_width();
        let slider_width = (total_width - drag_width - spacing).max(10.0);

        ui.allocate_ui_with_layout(
            vec2(slider_width, 24.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                if custom_slider(ui, value, range, None, accent_color).changed() {
                    changed = true;
                }
            },
        );

        let response = ui.add_sized([drag_width, 20.0], egui::DragValue::new(value).speed(speed));
        if response.changed() {
            changed = true;
        }
        if let Some(text) = hover_text {
            response.on_hover_text(text);
        }
    });
    changed
}
