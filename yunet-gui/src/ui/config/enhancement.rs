//! Enhancement settings UI controls.

use crate::YuNetApp;
use egui::{ComboBox, Slider, Ui};

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
        show_auto_color_control(app, ui, settings_changed, enhancement_changed);

        // Exposure control
        show_exposure_control(app, ui, settings_changed, enhancement_changed);

        // Brightness control
        show_brightness_control(app, ui, settings_changed, enhancement_changed);

        // Contrast control
        show_contrast_control(app, ui, settings_changed, enhancement_changed);

        // Saturation control
        show_saturation_control(app, ui, settings_changed, enhancement_changed);

        // Sharpness control
        show_sharpness_control(app, ui, settings_changed, enhancement_changed);

        // Skin smoothing control
        show_skin_smooth_control(app, ui, settings_changed, enhancement_changed);

        // Red-eye removal
        show_red_eye_control(app, ui, settings_changed, enhancement_changed);

        // Background blur
        show_background_blur_control(app, ui, settings_changed, enhancement_changed);

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

/// Shows the auto color correction checkbox.
fn show_auto_color_control(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    enhancement_changed: &mut bool,
) {
    ui.add_space(6.0);
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
}

/// Shows the exposure slider control.
fn show_exposure_control(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    enhancement_changed: &mut bool,
) {
    ui.add_space(6.0);
    let mut exp = app.settings.enhance.exposure_stops;
    if ui
        .add(Slider::new(&mut exp, -2.0..=2.0).text("Exposure (stops)"))
        .changed()
    {
        app.settings.enhance.exposure_stops = exp;
        *settings_changed = true;
        *enhancement_changed = true;
    }
}

/// Shows the brightness slider control.
fn show_brightness_control(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    enhancement_changed: &mut bool,
) {
    let mut bright = app.settings.enhance.brightness;
    if ui
        .add(Slider::new(&mut bright, -100..=100).text("Brightness"))
        .changed()
    {
        app.settings.enhance.brightness = bright;
        *settings_changed = true;
        *enhancement_changed = true;
    }
}

/// Shows the contrast slider control.
fn show_contrast_control(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    enhancement_changed: &mut bool,
) {
    let mut con = app.settings.enhance.contrast;
    if ui
        .add(Slider::new(&mut con, 0.5..=2.0).text("Contrast"))
        .changed()
    {
        app.settings.enhance.contrast = con;
        *settings_changed = true;
        *enhancement_changed = true;
    }
}

/// Shows the saturation slider control.
fn show_saturation_control(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    enhancement_changed: &mut bool,
) {
    let mut sat = app.settings.enhance.saturation;
    if ui
        .add(Slider::new(&mut sat, 0.0..=2.5).text("Saturation"))
        .changed()
    {
        app.settings.enhance.saturation = sat;
        *settings_changed = true;
        *enhancement_changed = true;
    }
}

/// Shows the sharpness slider control.
fn show_sharpness_control(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    enhancement_changed: &mut bool,
) {
    let mut sharp = app.settings.enhance.sharpness;
    if ui
        .add(Slider::new(&mut sharp, 0.0..=2.0).text("Sharpness"))
        .changed()
    {
        app.settings.enhance.sharpness = sharp;
        *settings_changed = true;
        *enhancement_changed = true;
    }
}

/// Shows the skin smoothing slider control.
fn show_skin_smooth_control(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    enhancement_changed: &mut bool,
) {
    let mut skin_smooth = app.settings.enhance.skin_smooth;
    if ui
        .add(Slider::new(&mut skin_smooth, 0.0..=1.0).text("Skin Smoothing"))
        .changed()
    {
        app.settings.enhance.skin_smooth = skin_smooth;
        *settings_changed = true;
        *enhancement_changed = true;
    }
}

/// Shows the red-eye removal checkbox.
fn show_red_eye_control(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    enhancement_changed: &mut bool,
) {
    if ui
        .checkbox(&mut app.settings.enhance.red_eye_removal, "Red-Eye Removal")
        .changed()
    {
        *settings_changed = true;
        *enhancement_changed = true;
    }
}

/// Shows the background blur checkbox.
fn show_background_blur_control(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    enhancement_changed: &mut bool,
) {
    if ui
        .checkbox(&mut app.settings.enhance.background_blur, "Background Blur")
        .changed()
    {
        *settings_changed = true;
        *enhancement_changed = true;
    }
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
