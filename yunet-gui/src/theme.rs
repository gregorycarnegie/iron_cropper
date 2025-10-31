//! Global theme customizations for the YuNet GUI.

use egui::{Color32, Context, Margin, Stroke, Visuals};

/// Apply the global YuNet GUI theme to the provided egui context.
///
/// This function sets up custom spacing, padding, and color schemes.
pub fn apply(ctx: &Context) {
    let mut style = (*ctx.style()).clone();

    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 6.0);
    style.spacing.window_margin = Margin::same(12);
    style.visuals = dark_visuals();

    ctx.set_style(style);
}

/// Creates a custom dark theme for the application.
fn dark_visuals() -> Visuals {
    let mut visuals = Visuals::dark();
    visuals.override_text_color = Some(Color32::from_rgb(220, 220, 230));
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, Color32::from_rgb(180, 180, 190));
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(36, 36, 42);
    visuals.widgets.active.bg_fill = Color32::from_rgb(52, 54, 64);
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(46, 48, 56);
    visuals.extreme_bg_color = Color32::from_rgb(18, 18, 24);
    visuals
}
