use crate::YuNetApp;
use crate::ui::widgets;
use egui::Ui;

/// Shows the width and height dimension controls.
pub fn show_dimensions_controls(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    preview_invalidated: &mut bool,
) {
    ui.add_space(4.0);
    let (mut width, mut height) = app.resolved_output_dimensions();
    let mut dimensions_changed = false;

    egui::Grid::new("dimensions_grid")
        .num_columns(3)
        .spacing([8.0, 4.0])
        .show(ui, |ui| {
            // Labels row
            ui.label("Width")
                .on_hover_text("Export width in pixels for the crop.");
            ui.label(" "); // Spacer for lock button column
            ui.label("Height")
                .on_hover_text("Export height in pixels for the crop.");
            ui.end_row();

            // Inputs row
            let width_response = widgets::integer_input(ui, &mut width, 64..=4096, 80.0, None)
                .on_hover_text(
                    "Type to set the output width. Editing switches to the Custom preset.",
                );

            // Lock button
            ui.with_layout(
                egui::Layout::centered_and_justified(egui::Direction::LeftToRight),
                |ui| {
                    let icon = if app.aspect_ratio_locked {
                        app.icons.lock(14.0)
                    } else {
                        app.icons.unlock(14.0)
                    };
                    if ui
                        .add(egui::Button::image(icon).frame(false))
                        .on_hover_text("Lock aspect ratio")
                        .clicked()
                    {
                        app.aspect_ratio_locked = !app.aspect_ratio_locked;
                    }
                },
            );

            let height_response = widgets::integer_input(ui, &mut height, 64..=4096, 80.0, None)
                .on_hover_text(
                    "Type to set the output height. Editing switches to the Custom preset.",
                );

            if width_response.changed() {
                if app.aspect_ratio_locked {
                    let ratio = app.settings.crop.output_width as f32
                        / app.settings.crop.output_height as f32;
                    height = (width as f32 / ratio).round() as u32;
                }
                app.settings.crop.output_width = width;
                app.settings.crop.output_height = height;
                dimensions_changed = true;
            } else if height_response.changed() {
                if app.aspect_ratio_locked {
                    let ratio = app.settings.crop.output_width as f32
                        / app.settings.crop.output_height as f32;
                    width = (height as f32 * ratio).round() as u32;
                }
                app.settings.crop.output_height = height;
                app.settings.crop.output_width = width;
                dimensions_changed = true;
            }
            ui.end_row();
        });

    if dimensions_changed {
        if app.settings.crop.preset != "custom" {
            app.settings.crop.preset = "custom".to_string();
        }
        app.clear_crop_preview_cache();
        *preview_invalidated = true;
        *settings_changed = true;
    }
}
