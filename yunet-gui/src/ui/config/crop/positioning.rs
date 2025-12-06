use crate::YuNetApp;
use crate::ui::widgets;
use egui::{ComboBox, Ui};

/// Shows the positioning mode selector and custom offset controls.
pub fn show_positioning_controls(app: &mut YuNetApp, ui: &mut Ui, settings_changed: &mut bool) {
    ui.add_space(6.0);
    let positioning_combo = ComboBox::from_label("Positioning mode")
        .selected_text(&app.settings.crop.positioning_mode)
        .show_ui(ui, |ui| {
            let modes = [
                ("center", "Center"),
                ("rule-of-thirds", "Rule of Thirds"),
                ("custom", "Custom offsets"),
            ];
            for (value, label) in modes {
                if ui
                    .selectable_label(app.settings.crop.positioning_mode == value, label)
                    .clicked()
                {
                    app.settings.crop.positioning_mode = value.to_string();
                    *settings_changed = true;
                }
            }
        });
    positioning_combo
        .response
        .on_hover_text("Adjusts how the crop is aligned around the detected face.");

    if app.settings.crop.positioning_mode == "custom" {
        ui.add_space(4.0);
        let mut vert = app.settings.crop.vertical_offset;
        if widgets::slider_row(
            ui,
            &mut vert,
            -1.0..=1.0,
            "Vertical offset",
            0.01,
            Some("Negative values move the crop up, positive values move it down."),
            None,
        ) {
            app.settings.crop.vertical_offset = vert;
            *settings_changed = true;
        }
        let mut horiz = app.settings.crop.horizontal_offset;
        if widgets::slider_row(
            ui,
            &mut horiz,
            -1.0..=1.0,
            "Horizontal offset",
            0.01,
            Some("Negative values move the crop left, positive values move it right."),
            None,
        ) {
            app.settings.crop.horizontal_offset = horiz;
            *settings_changed = true;
        }
    }
}
