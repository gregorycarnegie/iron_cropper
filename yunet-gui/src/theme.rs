//! Global theme customizations for the YuNet GUI.

use egui::{Color32, Context, CornerRadius, Margin, Shadow, Stroke, Visuals};

// Theme spacing constants
const ITEM_SPACING_X: f32 = 10.0;
const ITEM_SPACING_Y: f32 = 8.0;
const BUTTON_PADDING_X: f32 = 12.0;
const BUTTON_PADDING_Y: f32 = 8.0;
const WINDOW_MARGIN: i8 = 14;
const WINDOW_CORNER_RADIUS: u8 = 18;
const MENU_CORNER_RADIUS: u8 = 12;

// Window shadow constants
const WINDOW_SHADOW_OFFSET_Y: i8 = 6;
const WINDOW_SHADOW_BLUR: u8 = 24;
const WINDOW_SHADOW_SPREAD: u8 = 2;
const POPUP_SHADOW_OFFSET_Y: i8 = 4;
const POPUP_SHADOW_BLUR: u8 = 20;
const POPUP_SHADOW_SPREAD: u8 = 1;

/// Shared color palette used by the GUI.
#[derive(Clone, Copy)]
pub struct Palette {
    pub canvas: Color32,
    pub panel: Color32,
    pub panel_dark: Color32,
    pub panel_light: Color32,
    pub accent: Color32,
    pub accent_soft: Color32,
    pub success: Color32,
    pub warning: Color32,
    pub danger: Color32,
    pub subtle_text: Color32,
    pub outline: Color32,
}

/// Returns the default palette.
pub fn palette() -> Palette {
    Palette {
        canvas: Color32::from_rgb(8, 13, 24),
        panel: Color32::from_rgb(18, 26, 40),
        panel_dark: Color32::from_rgb(10, 16, 28),
        panel_light: Color32::from_rgb(36, 48, 74),
        accent: Color32::from_rgb(88, 182, 255),
        accent_soft: Color32::from_rgba_unmultiplied(88, 182, 255, 70),
        success: Color32::from_rgb(92, 214, 172),
        warning: Color32::from_rgb(255, 194, 122),
        danger: Color32::from_rgb(255, 128, 140),
        subtle_text: Color32::from_rgb(196, 207, 223),
        outline: Color32::from_rgba_unmultiplied(90, 106, 140, 150),
    }
}

/// Apply the global YuNet GUI theme to the provided egui context.
pub fn apply(ctx: &Context) {
    let palette = palette();
    let mut style = (*ctx.style()).clone();

    style.spacing.item_spacing = egui::vec2(ITEM_SPACING_X, ITEM_SPACING_Y);
    style.spacing.button_padding = egui::vec2(BUTTON_PADDING_X, BUTTON_PADDING_Y);
    style.spacing.window_margin = Margin::same(WINDOW_MARGIN);
    style.visuals = visuals_from_palette(palette);

    ctx.set_style(style);
}

fn visuals_from_palette(palette: Palette) -> Visuals {
    let mut visuals = Visuals::dark();
    visuals.override_text_color = Some(Color32::from_rgb(232, 236, 245));
    visuals.hyperlink_color = palette.accent;
    visuals.panel_fill = palette.panel;
    visuals.extreme_bg_color = palette.canvas;

    visuals.widgets.noninteractive.bg_fill = palette.panel_dark;
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, palette.subtle_text);

    visuals.widgets.inactive.bg_fill = palette.panel;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, palette.outline);

    visuals.widgets.hovered.bg_fill = palette.panel_light;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, palette.accent_soft);

    visuals.widgets.active.bg_fill = palette.panel_light;
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, palette.accent);

    visuals.widgets.open.bg_fill = palette.panel_light;
    visuals.selection.bg_fill = palette.accent;
    visuals.selection.stroke = Stroke::new(1.5, palette.panel_dark);

    visuals.window_corner_radius = CornerRadius::same(WINDOW_CORNER_RADIUS);
    visuals.menu_corner_radius = CornerRadius::same(MENU_CORNER_RADIUS);
    visuals.window_shadow = Shadow {
        offset: [0, WINDOW_SHADOW_OFFSET_Y],
        blur: WINDOW_SHADOW_BLUR,
        spread: WINDOW_SHADOW_SPREAD,
        color: Color32::from_rgba_unmultiplied(0, 0, 0, 220),
    };
    visuals.popup_shadow = Shadow {
        offset: [0, POPUP_SHADOW_OFFSET_Y],
        blur: POPUP_SHADOW_BLUR,
        spread: POPUP_SHADOW_SPREAD,
        color: Color32::from_rgba_unmultiplied(0, 0, 0, 200),
    };

    visuals
}
