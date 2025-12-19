use crate::YuNetApp;
use crate::ui::widgets;
use egui::{ComboBox, Ui};

/// Shows the output format settings.
pub fn show_output_format(app: &mut YuNetApp, ui: &mut Ui, settings_changed: &mut bool) {
    ui.separator();
    ui.label("Output format");

    ComboBox::from_id_salt("output_format_combo")
        .selected_text(app.settings.crop.output_format.to_ascii_uppercase())
        .show_ui(ui, |ui| {
            for option in ["png", "jpeg", "webp", "tiff", "bmp", "avif"] {
                if ui
                    .selectable_label(
                        app.settings.crop.output_format == option,
                        option.to_ascii_uppercase(),
                    )
                    .clicked()
                {
                    app.settings.crop.output_format = option.to_string();
                    *settings_changed = true;
                }
            }
        });

    let mut auto_detect = app.settings.crop.auto_detect_format;
    if ui
        .checkbox(&mut auto_detect, "Auto-detect format from file extension")
        .changed()
    {
        app.settings.crop.auto_detect_format = auto_detect;
        *settings_changed = true;
    }

    // PNG compression
    if app.settings.crop.output_format == "png" {
        ui.horizontal(|ui| {
            ui.label("PNG compression");
            let current = app.settings.crop.png_compression.to_ascii_lowercase();
            let label = match current.as_str() {
                "fast" => "Fast".to_string(),
                "default" => "Default".to_string(),
                "best" => "Best".to_string(),
                other => format!("Custom ({other})"),
            };
            ComboBox::from_id_salt("png_compression_combo")
                .selected_text(label)
                .show_ui(ui, |ui| {
                    for (value, text) in
                        [("fast", "Fast"), ("default", "Default"), ("best", "Best")]
                    {
                        if ui
                            .selectable_label(
                                app.settings
                                    .crop
                                    .png_compression
                                    .eq_ignore_ascii_case(value),
                                text,
                            )
                            .clicked()
                        {
                            app.settings.crop.png_compression = value.to_string();
                            *settings_changed = true;
                        }
                    }
                });

            let mut level = app
                .settings
                .crop
                .png_compression
                .parse::<i32>()
                .unwrap_or(6);
            let prev = level;
            ui.label("Level");
            if widgets::integer_input(ui, &mut level, 0..=9, 40.0, None).changed() {
                level = level.clamp(0, 9);
                if level != prev {
                    app.settings.crop.png_compression = level.to_string();
                    *settings_changed = true;
                }
            }
        });
    }

    // JPEG quality
    if app.settings.crop.output_format == "jpeg" {
        let mut jpeg_quality = i32::from(app.settings.crop.jpeg_quality);
        crate::constrained_slider_row!(
            ui,
            &mut jpeg_quality,
            1..=100,
            "JPEG quality",
            1.0,
            None,
            None,
            {
                app.settings.crop.jpeg_quality = jpeg_quality as u8;
                *settings_changed = true;
            }
        );
    }

    // WebP quality
    if app.settings.crop.output_format == "webp" {
        let mut webp_quality = i32::from(app.settings.crop.webp_quality);
        crate::constrained_slider_row!(
            ui,
            &mut webp_quality,
            0..=100,
            "WebP quality",
            1.0,
            None,
            None,
            {
                app.settings.crop.webp_quality = webp_quality as u8;
                *settings_changed = true;
            }
        );
    }
}
