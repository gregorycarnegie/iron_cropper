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

        egui::Grid::new("enhancement_grid")
            .num_columns(3)
            .spacing([8.0, 8.0])
            .show(ui, |ui| {
                // Auto color correction
                ui.label("Auto color correction");
                if ui
                    .checkbox(&mut app.settings.enhance.auto_color, "")
                    .changed()
                {
                    *settings_changed = true;
                    *enhancement_changed = true;
                }
                ui.label(""); // Placeholder for 3rd column
                ui.end_row();

                // Exposure control
                ui.label("Exposure (stops)");
                let mut exp = app.settings.enhance.exposure_stops;
                let r1 = widgets::custom_slider(ui, &mut exp, -2.0..=2.0, None);
                let r2 = ui.add_sized([50.0, 20.0], egui::DragValue::new(&mut exp).speed(0.01));
                if r1.changed() || r2.changed() {
                    app.settings.enhance.exposure_stops = exp;
                    *settings_changed = true;
                    *enhancement_changed = true;
                }
                ui.end_row();

                // Brightness control
                ui.label("Brightness");
                let mut bright = app.settings.enhance.brightness;
                let r1 = widgets::custom_slider(ui, &mut bright, -100..=100, None);
                let r2 = ui.add_sized([50.0, 20.0], egui::DragValue::new(&mut bright).speed(1.0));
                if r1.changed() || r2.changed() {
                    app.settings.enhance.brightness = bright;
                    *settings_changed = true;
                    *enhancement_changed = true;
                }
                ui.end_row();

                // Contrast control
                ui.label("Contrast");
                let mut con = app.settings.enhance.contrast;
                let r1 = widgets::custom_slider(ui, &mut con, 0.5..=2.0, None);
                let r2 = ui.add_sized([50.0, 20.0], egui::DragValue::new(&mut con).speed(0.01));
                if r1.changed() || r2.changed() {
                    app.settings.enhance.contrast = con;
                    *settings_changed = true;
                    *enhancement_changed = true;
                }
                ui.end_row();

                // Saturation control
                ui.label("Saturation");
                let mut sat = app.settings.enhance.saturation;
                let r1 = widgets::custom_slider(ui, &mut sat, 0.0..=2.5, None);
                let r2 = ui.add_sized([50.0, 20.0], egui::DragValue::new(&mut sat).speed(0.01));
                if r1.changed() || r2.changed() {
                    app.settings.enhance.saturation = sat;
                    *settings_changed = true;
                    *enhancement_changed = true;
                }
                ui.end_row();

                // Sharpness control
                ui.label("Sharpness");
                let mut sharp = app.settings.enhance.sharpness;
                let r1 = widgets::custom_slider(ui, &mut sharp, 0.0..=2.0, None);
                let r2 = ui.add_sized([50.0, 20.0], egui::DragValue::new(&mut sharp).speed(0.01));
                if r1.changed() || r2.changed() {
                    app.settings.enhance.sharpness = sharp;
                    *settings_changed = true;
                    *enhancement_changed = true;
                }
                ui.end_row();

                // Skin smoothing control
                ui.label("Skin Smoothing");
                let mut skin_smooth = app.settings.enhance.skin_smooth;
                let r1 = widgets::custom_slider(ui, &mut skin_smooth, 0.0..=1.0, None);
                let r2 = ui.add_sized(
                    [50.0, 20.0],
                    egui::DragValue::new(&mut skin_smooth).speed(0.01),
                );
                if r1.changed() || r2.changed() {
                    app.settings.enhance.skin_smooth = skin_smooth;
                    *settings_changed = true;
                    *enhancement_changed = true;
                }
                ui.end_row();

                // Red-eye removal
                ui.label("Red-Eye Removal");
                if ui
                    .checkbox(&mut app.settings.enhance.red_eye_removal, "")
                    .changed()
                {
                    *settings_changed = true;
                    *enhancement_changed = true;
                }
                ui.label(""); // Placeholder for 3rd column
                ui.end_row();

                // Background blur
                ui.label("Background Blur");
                if ui
                    .checkbox(&mut app.settings.enhance.background_blur, "")
                    .changed()
                {
                    *settings_changed = true;
                    *enhancement_changed = true;
                }
                ui.label(""); // Placeholder for 3rd column
                ui.end_row();
            });

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
