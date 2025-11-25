//! Enhancement settings UI controls.

use crate::YuNetApp;
use crate::ui::widgets;
use egui::{ComboBox, Ui};

/// Shows the enhancement settings section.
pub fn show_enhancement_section(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    enhancement_changed: &mut bool,
) {
    ui.separator();
    ui.heading("Enhancement Settings");

    let enable_response = ui.checkbox(&mut app.settings.enhance.enabled, "Enable enhancements");
    if enable_response.changed() {
        *settings_changed = true;
        *enhancement_changed = true;
    }
    ui.add_space(6.0);

    if app.settings.enhance.enabled {
        // Preset selection
        show_preset_selector(app, ui, settings_changed, enhancement_changed);

        // Auto color correction
        if ui
            .checkbox(
                &mut app.settings.enhance.auto_color,
                "Auto color correction",
            )
            .changed()
        {
            *settings_changed = true;
            *enhancement_changed = true;
        }

        // Exposure control
        let mut exp = app.settings.enhance.exposure_stops;
        if widgets::slider_row(
            ui,
            &mut exp,
            -2.0..=2.0,
            "Exposure (stops)",
            0.01,
            None,
            None,
        ) {
            app.settings.enhance.exposure_stops = exp;
            *settings_changed = true;
            *enhancement_changed = true;
        }

        // Brightness control
        let mut bright = app.settings.enhance.brightness;
        if widgets::slider_row(ui, &mut bright, -100..=100, "Brightness", 1.0, None, None) {
            app.settings.enhance.brightness = bright;
            *settings_changed = true;
            *enhancement_changed = true;
        }

        // Contrast control
        let mut con = app.settings.enhance.contrast;
        if widgets::slider_row(ui, &mut con, 0.5..=2.0, "Contrast", 0.01, None, None) {
            app.settings.enhance.contrast = con;
            *settings_changed = true;
            *enhancement_changed = true;
        }

        // Saturation control
        let mut sat = app.settings.enhance.saturation;
        if widgets::slider_row(ui, &mut sat, 0.0..=2.5, "Saturation", 0.01, None, None) {
            app.settings.enhance.saturation = sat;
            *settings_changed = true;
            *enhancement_changed = true;
        }

        // Sharpness control
        let mut sharp = app.settings.enhance.sharpness;
        if widgets::slider_row(ui, &mut sharp, 0.0..=2.0, "Sharpness", 0.01, None, None) {
            app.settings.enhance.sharpness = sharp;
            *settings_changed = true;
            *enhancement_changed = true;
        }

        // Skin smoothing control
        let mut skin_smooth = app.settings.enhance.skin_smooth;
        if widgets::slider_row(
            ui,
            &mut skin_smooth,
            0.0..=1.0,
            "Skin Smoothing",
            0.01,
            None,
            None,
        ) {
            app.settings.enhance.skin_smooth = skin_smooth;
            *settings_changed = true;
            *enhancement_changed = true;
        }

        // Red-eye removal
        if ui
            .checkbox(&mut app.settings.enhance.red_eye_removal, "Red-Eye Removal")
            .changed()
        {
            *settings_changed = true;
            *enhancement_changed = true;
        }

        // Background blur
        if ui
            .checkbox(&mut app.settings.enhance.background_blur, "Background Blur")
            .changed()
        {
            *settings_changed = true;
            *enhancement_changed = true;
        }

        // Reset button
        show_reset_button(app, ui, settings_changed, enhancement_changed);
    }
}

/// Shows the enhancement preset selector.
fn show_preset_selector(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    enhancement_changed: &mut bool,
) {
    ui.label("Preset");
    ComboBox::from_label("  ")
        .selected_text(&app.settings.enhance.preset)
        .show_ui(ui, |ui| {
            let presets = [
                ("none", "None (Manual)"),
                ("natural", "Natural"),
                ("vivid", "Vivid"),
                ("professional", "Professional"),
            ];
            for (value, label) in presets {
                if ui
                    .selectable_label(app.settings.enhance.preset == value, label)
                    .clicked()
                {
                    app.settings.enhance.preset = value.to_string();
                    app.apply_enhancement_preset();
                    *settings_changed = true;
                    *enhancement_changed = true;
                }
            }
        });
}

/// Shows the reset to defaults button.
fn show_reset_button(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    enhancement_changed: &mut bool,
) {
    ui.add_space(6.0);
    if ui.button("Reset to defaults").clicked() {
        app.settings.enhance = yunet_utils::config::EnhanceSettings::default();
        *settings_changed = true;
        *enhancement_changed = true;
    }
}
