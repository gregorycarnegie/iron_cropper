//! Crop settings UI controls.

use crate::YuNetApp;
use crate::types::ColorMode;
use crate::ui::widgets;

use egui::{Color32, ComboBox, DragValue, TextEdit, Ui, color_picker};
use std::collections::BTreeMap;
use yunet_core::preset_by_name;
use yunet_utils::{
    CropShape, PolygonCornerStyle, RgbaColor, cmyk_to_rgb, config::MetadataMode, hsl_to_rgb,
    hsv_to_rgb, parse_hex_color, quality::Quality, rgb_to_cmyk, rgb_to_hsl, rgb_to_hsv,
};

/// Shows the crop settings section.
pub fn show_crop_section(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    preview_invalidated: &mut bool,
    metadata_tags_changed: &mut bool,
) {
    ui.separator();
    ui.heading("Crop Settings");

    ui.checkbox(&mut app.show_crop_overlay, "Show crop preview overlay")
        .on_hover_text("Draws the proposed crop rectangle for every detected face.");
    ui.add_space(6.0);

    // Preset selection
    show_preset_selector(app, ui, settings_changed, preview_invalidated);

    // Output dimensions
    show_dimensions_controls(app, ui, settings_changed, preview_invalidated);

    // Shape controls
    if edit_shape_controls(app, ui) {
        app.settings.crop.sanitize();
        app.clear_crop_preview_cache();
        *preview_invalidated = true;
        *settings_changed = true;
    }

    if show_fill_color_controls(app, ui) {
        app.clear_crop_preview_cache();
        *preview_invalidated = true;
        *settings_changed = true;
    }

    // Positioning mode
    show_positioning_controls(app, ui, settings_changed);

    // Quality automation
    show_quality_automation(app, ui, settings_changed);

    // Output format
    show_output_format(app, ui, settings_changed);

    // Metadata
    show_metadata_section(app, ui, settings_changed, metadata_tags_changed);
}

/// Shows the preset selector combo box.
fn show_preset_selector(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    preview_invalidated: &mut bool,
) {
    let preset_combo = ComboBox::from_label("Preset")
        .selected_text(&app.settings.crop.preset)
        .show_ui(ui, |ui| {
            let presets = [
                ("linkedin", "LinkedIn (400×400)"),
                ("passport", "Passport (413×531)"),
                ("instagram", "Instagram (1080×1080)"),
                ("idcard", "ID Card (332×498)"),
                ("avatar", "Avatar (512×512)"),
                ("headshot", "Headshot (600×800)"),
                ("custom", "Custom size"),
            ];
            let icon_size = app.icons.default_size();
            for (value, label) in presets {
                let selected = app.settings.crop.preset == value;
                let icon = match value {
                    "linkedin" => Some(app.icons.linkedin(icon_size)),
                    "passport" => Some(app.icons.passport(icon_size)),
                    "instagram" => Some(app.icons.instagram(icon_size)),
                    "idcard" => Some(app.icons.id_card(icon_size)),
                    "avatar" => Some(app.icons.account(icon_size)),
                    "headshot" => Some(app.icons.portrait(icon_size)),
                    _ => None,
                };

                let clicked = if let Some(icon) = icon {
                    ui.add(egui::Button::image_and_text(icon, label).selected(selected))
                        .clicked()
                } else {
                    ui.selectable_label(selected, label).clicked()
                };

                if clicked && !selected {
                    app.settings.crop.preset = value.to_string();
                    if value != "custom"
                        && let Some(preset) = preset_by_name(value)
                    {
                        app.settings.crop.output_width = preset.width;
                        app.settings.crop.output_height = preset.height;
                    }
                    app.clear_crop_preview_cache();
                    *preview_invalidated = true;
                    *settings_changed = true;
                }
            }
        });
    preset_combo
        .response
        .on_hover_text("Choose a predefined output size/aspect ratio.");
}

/// Shows the width and height dimension controls.
fn show_dimensions_controls(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    preview_invalidated: &mut bool,
) {
    ui.add_space(4.0);
    let (mut width, mut height) = app.resolved_output_dimensions();
    let mut dimensions_changed = false;

    ui.horizontal(|ui| {
        ui.label("Width")
            .on_hover_text("Export width in pixels for the crop.");
        let response = ui
            .add(DragValue::new(&mut width).range(64..=4096).speed(16.0))
            .on_hover_text(
                "Drag or type to set the output width. Editing switches to the Custom preset.",
            );
        if response.changed() {
            app.settings.crop.output_width = width;
            dimensions_changed = true;
        }
    });
    ui.horizontal(|ui| {
        ui.label("Height")
            .on_hover_text("Export height in pixels for the crop.");
        let response = ui
            .add(DragValue::new(&mut height).range(64..=4096).speed(16.0))
            .on_hover_text(
                "Drag or type to set the output height. Editing switches to the Custom preset.",
            );
        if response.changed() {
            app.settings.crop.output_height = height;
            dimensions_changed = true;
        }
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

fn show_fill_color_controls(app: &mut YuNetApp, ui: &mut Ui) -> bool {
    ui.separator();
    ui.label("Out-of-bounds fill color");

    let mut changed = false;
    let mut picker_color = Color32::from_rgba_unmultiplied(
        app.settings.crop.fill_color.red,
        app.settings.crop.fill_color.green,
        app.settings.crop.fill_color.blue,
        app.settings.crop.fill_color.alpha,
    );
    let picker_response = color_picker::color_edit_button_srgba(
        ui,
        &mut picker_color,
        color_picker::Alpha::Opaque,
    )
    .on_hover_text(
        "Choose the color used to fill pixels outside the source image when the crop extends beyond the bounds.",
    );
    if picker_response.changed() {
        let arr = picker_color.to_array();
        let new_color = RgbaColor {
            red: arr[0],
            green: arr[1],
            blue: arr[2],
            alpha: arr[3],
        };
        if app.set_fill_color(new_color) {
            changed = true;
        }
    }

    let mut hex_error = false;

    // Prepare RGB variables
    let alpha = app.settings.crop.fill_color.alpha;
    let mut rgb_changed = false;
    let mut r = app.settings.crop.fill_color.red as i32;
    let mut g = app.settings.crop.fill_color.green as i32;
    let mut b = app.settings.crop.fill_color.blue as i32;

    // Prepare HSV variables
    let (h0, s0, v0) = rgb_to_hsv(
        app.settings.crop.fill_color.red,
        app.settings.crop.fill_color.green,
        app.settings.crop.fill_color.blue,
    );
    let mut hue = h0;
    let mut sat_pct = s0 * 100.0;
    let mut val_pct = v0 * 100.0;
    let mut hsv_changed = false;

    // Prepare HSL variables
    let (h_l, s_l, l_l) = rgb_to_hsl(
        app.settings.crop.fill_color.red,
        app.settings.crop.fill_color.green,
        app.settings.crop.fill_color.blue,
    );
    let mut hue_l = h_l;
    let mut sat_l_pct = s_l * 100.0;
    let mut light_pct = l_l * 100.0;
    let mut hsl_changed = false;

    // Prepare CMYK variables
    let (c0, m0, y0, k0) = rgb_to_cmyk(
        app.settings.crop.fill_color.red,
        app.settings.crop.fill_color.green,
        app.settings.crop.fill_color.blue,
    );
    let mut c_pct = c0 * 100.0;
    let mut m_pct = m0 * 100.0;
    let mut y_pct = y0 * 100.0;
    let mut k_pct = k0 * 100.0;
    let mut cmyk_changed = false;

    egui::Grid::new("color_picker_grid")
        .num_columns(2)
        .spacing([8.0, 8.0])
        .show(ui, |ui| {
            ui.label("Hex");
            let hex_response = ui.add_sized(
                [95.0, 20.0],
                TextEdit::singleline(&mut app.crop_fill_hex_input)
                    .hint_text("#RRGGBB or #RRGGBBAA"),
            );
            if hex_response.changed() {
                if let Some(color) = parse_hex_color(&app.crop_fill_hex_input) {
                    if app.set_fill_color(color) {
                        changed = true;
                    } else {
                        app.refresh_fill_color_hex_input();
                    }
                } else {
                    hex_error = true;
                }
            }
            ui.end_row();

            ui.label("Model");
            egui::ComboBox::from_id_salt("color_mode_combo")
                .selected_text(match app.fill_color_mode {
                    ColorMode::Rgb => "RGB",
                    ColorMode::Hsv => "HSV",
                    ColorMode::Hsl => "HSL",
                    ColorMode::Cmyk => "CMYK",
                })
                .show_ui(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.add(app.icons.rgb(14.0));
                        ui.selectable_value(&mut app.fill_color_mode, ColorMode::Rgb, "RGB");
                    });
                    ui.horizontal(|ui| {
                        ui.add(app.icons.hsv(14.0));
                        ui.selectable_value(&mut app.fill_color_mode, ColorMode::Hsv, "HSV");
                    });
                    ui.horizontal(|ui| {
                        ui.add(app.icons.hsl(14.0));
                        ui.selectable_value(&mut app.fill_color_mode, ColorMode::Hsl, "HSL");
                    });
                    ui.horizontal(|ui| {
                        ui.add(app.icons.cmyk(14.0));
                        ui.selectable_value(&mut app.fill_color_mode, ColorMode::Cmyk, "CMYK");
                    });
                });
            ui.end_row();

            match app.fill_color_mode {
                ColorMode::Rgb => {
                    ui.label("RGB");
                    egui::Grid::new("rgb_inner_grid")
                        .num_columns(3)
                        .spacing([8.0, 0.0])
                        .show(ui, |ui| {
                            if ui
                                .add_sized(
                                    [80.0, 20.0],
                                    DragValue::new(&mut r)
                                        .range(0..=255)
                                        .speed(1.0)
                                        .suffix(" R"),
                                )
                                .changed()
                            {
                                rgb_changed = true;
                            }
                            if ui
                                .add_sized(
                                    [80.0, 20.0],
                                    DragValue::new(&mut g)
                                        .range(0..=255)
                                        .speed(1.0)
                                        .suffix(" G"),
                                )
                                .changed()
                            {
                                rgb_changed = true;
                            }
                            if ui
                                .add_sized(
                                    [80.0, 20.0],
                                    DragValue::new(&mut b)
                                        .range(0..=255)
                                        .speed(1.0)
                                        .suffix(" B"),
                                )
                                .changed()
                            {
                                rgb_changed = true;
                            }
                        });
                }
                ColorMode::Hsv => {
                    ui.label("HSV");
                    egui::Grid::new("hsv_inner_grid")
                        .num_columns(3)
                        .spacing([8.0, 0.0])
                        .show(ui, |ui| {
                            if ui
                                .add_sized(
                                    [80.0, 20.0],
                                    DragValue::new(&mut hue)
                                        .range(0.0..=360.0)
                                        .speed(1.0)
                                        .suffix("°"),
                                )
                                .changed()
                            {
                                hsv_changed = true;
                            }
                            if ui
                                .add_sized(
                                    [80.0, 20.0],
                                    DragValue::new(&mut sat_pct)
                                        .range(0.0..=100.0)
                                        .speed(1.0)
                                        .suffix("% S"),
                                )
                                .changed()
                            {
                                hsv_changed = true;
                            }
                            if ui
                                .add_sized(
                                    [80.0, 20.0],
                                    DragValue::new(&mut val_pct)
                                        .range(0.0..=100.0)
                                        .speed(1.0)
                                        .suffix("% V"),
                                )
                                .changed()
                            {
                                hsv_changed = true;
                            }
                        });
                }
                ColorMode::Hsl => {
                    ui.label("HSL");
                    egui::Grid::new("hsl_inner_grid")
                        .num_columns(3)
                        .spacing([8.0, 0.0])
                        .show(ui, |ui| {
                            if ui
                                .add_sized(
                                    [80.0, 20.0],
                                    DragValue::new(&mut hue_l)
                                        .range(0.0..=360.0)
                                        .speed(1.0)
                                        .suffix("°"),
                                )
                                .changed()
                            {
                                hsl_changed = true;
                            }
                            if ui
                                .add_sized(
                                    [80.0, 20.0],
                                    DragValue::new(&mut sat_l_pct)
                                        .range(0.0..=100.0)
                                        .speed(1.0)
                                        .suffix("% S"),
                                )
                                .changed()
                            {
                                hsl_changed = true;
                            }
                            if ui
                                .add_sized(
                                    [80.0, 20.0],
                                    DragValue::new(&mut light_pct)
                                        .range(0.0..=100.0)
                                        .speed(1.0)
                                        .suffix("% L"),
                                )
                                .changed()
                            {
                                hsl_changed = true;
                            }
                        });
                }
                ColorMode::Cmyk => {
                    ui.label("CMYK");
                    egui::Grid::new("cmyk_inner_grid")
                        .num_columns(4)
                        .spacing([8.0, 0.0])
                        .show(ui, |ui| {
                            if ui
                                .add_sized(
                                    [60.0, 20.0],
                                    DragValue::new(&mut c_pct)
                                        .range(0.0..=100.0)
                                        .speed(1.0)
                                        .suffix("% C"),
                                )
                                .changed()
                            {
                                cmyk_changed = true;
                            }
                            if ui
                                .add_sized(
                                    [60.0, 20.0],
                                    DragValue::new(&mut m_pct)
                                        .range(0.0..=100.0)
                                        .speed(1.0)
                                        .suffix("% M"),
                                )
                                .changed()
                            {
                                cmyk_changed = true;
                            }
                            if ui
                                .add_sized(
                                    [60.0, 20.0],
                                    DragValue::new(&mut y_pct)
                                        .range(0.0..=100.0)
                                        .speed(1.0)
                                        .suffix("% Y"),
                                )
                                .changed()
                            {
                                cmyk_changed = true;
                            }
                            if ui
                                .add_sized(
                                    [60.0, 20.0],
                                    DragValue::new(&mut k_pct)
                                        .range(0.0..=100.0)
                                        .speed(1.0)
                                        .suffix("% K"),
                                )
                                .changed()
                            {
                                cmyk_changed = true;
                            }
                        });
                }
            }
            ui.end_row();
        });

    if hex_error {
        ui.colored_label(
            Color32::from_rgb(255, 120, 120),
            "Invalid hex (use #RRGGBB or #RRGGBBAA)",
        );
    }

    if rgb_changed {
        let new_color = RgbaColor {
            red: r.clamp(0, 255) as u8,
            green: g.clamp(0, 255) as u8,
            blue: b.clamp(0, 255) as u8,
            alpha,
        };
        if app.set_fill_color(new_color) {
            changed = true;
        }
    }

    if hsv_changed {
        let (nr, ng, nb) = hsv_to_rgb(
            hue,
            (sat_pct / 100.0).clamp(0.0, 1.0),
            (val_pct / 100.0).clamp(0.0, 1.0),
        );
        if app.set_fill_color(RgbaColor {
            red: nr,
            green: ng,
            blue: nb,
            alpha,
        }) {
            changed = true;
        }
    }

    if hsl_changed {
        let (nr, ng, nb) = hsl_to_rgb(
            hue_l,
            (sat_l_pct / 100.0).clamp(0.0, 1.0),
            (light_pct / 100.0).clamp(0.0, 1.0),
        );
        if app.set_fill_color(RgbaColor {
            red: nr,
            green: ng,
            blue: nb,
            alpha,
        }) {
            changed = true;
        }
    }

    if cmyk_changed {
        let (nr, ng, nb) = cmyk_to_rgb(
            (c_pct / 100.0).clamp(0.0, 1.0),
            (m_pct / 100.0).clamp(0.0, 1.0),
            (y_pct / 100.0).clamp(0.0, 1.0),
            (k_pct / 100.0).clamp(0.0, 1.0),
        );
        if app.set_fill_color(RgbaColor {
            red: nr,
            green: ng,
            blue: nb,
            alpha,
        }) {
            changed = true;
        }
    }

    changed
}

/// Shows the positioning mode selector and custom offset controls.
fn show_positioning_controls(app: &mut YuNetApp, ui: &mut Ui, settings_changed: &mut bool) {
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

/// Shows the quality automation settings.
fn show_quality_automation(app: &mut YuNetApp, ui: &mut Ui, settings_changed: &mut bool) {
    ui.separator();
    ui.label("Quality automation");

    let mut auto_select = app.settings.crop.quality_rules.auto_select_best_face;
    if ui
        .checkbox(&mut auto_select, "Auto-select highest quality face")
        .changed()
    {
        app.settings.crop.quality_rules.auto_select_best_face = auto_select;
        *settings_changed = true;
    }

    let mut skip_no_high = app.settings.crop.quality_rules.auto_skip_no_high_quality;
    if ui
        .checkbox(&mut skip_no_high, "Skip export when no high-quality faces")
        .changed()
    {
        app.settings.crop.quality_rules.auto_skip_no_high_quality = skip_no_high;
        *settings_changed = true;
    }

    let mut suffix_enabled = app.settings.crop.quality_rules.quality_suffix;
    if ui
        .checkbox(&mut suffix_enabled, "Append quality suffix to filenames")
        .changed()
    {
        app.settings.crop.quality_rules.quality_suffix = suffix_enabled;
        *settings_changed = true;
    }

    ui.horizontal(|ui| {
        ui.label("Minimum quality to export");
        let current = app.settings.crop.quality_rules.min_quality;
        let label = match current {
            Some(Quality::Low) => "Low",
            Some(Quality::Medium) => "Medium",
            Some(Quality::High) => "High",
            None => "Off",
        };
        ComboBox::from_id_salt("min_quality_combo")
            .selected_text(label)
            .show_ui(ui, |ui| {
                if ui.selectable_label(current.is_none(), "Off").clicked() {
                    app.settings.crop.quality_rules.min_quality = None;
                    *settings_changed = true;
                }
                if ui
                    .selectable_label(current == Some(Quality::Low), "Low")
                    .clicked()
                {
                    app.settings.crop.quality_rules.min_quality = Some(Quality::Low);
                    *settings_changed = true;
                }
                if ui
                    .selectable_label(current == Some(Quality::Medium), "Medium")
                    .clicked()
                {
                    app.settings.crop.quality_rules.min_quality = Some(Quality::Medium);
                    *settings_changed = true;
                }
                if ui
                    .selectable_label(current == Some(Quality::High), "High")
                    .clicked()
                {
                    app.settings.crop.quality_rules.min_quality = Some(Quality::High);
                    *settings_changed = true;
                }
            });
    });
}

/// Shows the output format settings.
fn show_output_format(app: &mut YuNetApp, ui: &mut Ui, settings_changed: &mut bool) {
    ui.separator();
    ui.label("Output format");

    ComboBox::from_id_salt("output_format_combo")
        .selected_text(app.settings.crop.output_format.to_ascii_uppercase())
        .show_ui(ui, |ui| {
            for option in ["png", "jpeg", "webp"] {
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
                for (value, text) in [("fast", "Fast"), ("default", "Default"), ("best", "Best")] {
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
        if ui
            .add(DragValue::new(&mut level).range(0..=9).prefix("Level "))
            .changed()
        {
            level = level.clamp(0, 9);
            if level != prev {
                app.settings.crop.png_compression = level.to_string();
                *settings_changed = true;
            }
        }
    });
    // JPEG quality
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

    // WebP quality
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

/// Shows the metadata settings section.
fn show_metadata_section(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    metadata_tags_changed: &mut bool,
) {
    ui.separator();
    ui.label("Metadata");

    let mode_label = match app.settings.crop.metadata.mode {
        MetadataMode::Preserve => "Preserve",
        MetadataMode::Strip => "Strip",
        MetadataMode::Custom => "Custom",
    };
    ComboBox::from_id_salt("metadata_mode_combo")
        .selected_text(mode_label)
        .show_ui(ui, |ui| {
            for (value, text) in [
                (MetadataMode::Preserve, "Preserve"),
                (MetadataMode::Strip, "Strip"),
                (MetadataMode::Custom, "Custom"),
            ] {
                if ui
                    .selectable_label(app.settings.crop.metadata.mode == value, text)
                    .clicked()
                {
                    app.settings.crop.metadata.mode = value;
                    *settings_changed = true;
                }
            }
        });

    let mut include_crop = app.settings.crop.metadata.include_crop_settings;
    if ui
        .checkbox(&mut include_crop, "Include crop settings metadata")
        .changed()
    {
        app.settings.crop.metadata.include_crop_settings = include_crop;
        *settings_changed = true;
    }

    let mut include_quality = app.settings.crop.metadata.include_quality_metrics;
    if ui
        .checkbox(&mut include_quality, "Include quality metrics metadata")
        .changed()
    {
        app.settings.crop.metadata.include_quality_metrics = include_quality;
        *settings_changed = true;
    }

    if ui
        .text_edit_multiline(&mut app.metadata_tags_input)
        .changed()
    {
        app.settings.crop.metadata.custom_tags = parse_metadata_tags(&app.metadata_tags_input);
        *settings_changed = true;
        *metadata_tags_changed = true;
    }
    ui.label("Enter custom tags as key=value, one per line.")
        .on_hover_text("Tags are embedded into output metadata when mode is preserve or custom.");
}

/// Shape controls extracted from edit_shape_controls method.
fn edit_shape_controls(app: &mut YuNetApp, ui: &mut Ui) -> bool {
    let mut shape = app.settings.crop.shape.clone();
    let mut changed = false;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Variant {
        Rectangle,
        RoundedRect,
        ChamferRect,
        Ellipse,
        PolygonSharp,
        PolygonRounded,
        PolygonChamfered,
        PolygonBezier,
        Star,
    }

    let mut variant = match &shape {
        CropShape::Rectangle => Variant::Rectangle,
        CropShape::RoundedRectangle { .. } => Variant::RoundedRect,
        CropShape::ChamferedRectangle { .. } => Variant::ChamferRect,
        CropShape::Ellipse => Variant::Ellipse,
        CropShape::Polygon { corner_style, .. } => match corner_style {
            PolygonCornerStyle::Sharp => Variant::PolygonSharp,
            PolygonCornerStyle::Rounded { .. } => Variant::PolygonRounded,
            PolygonCornerStyle::Chamfered { .. } => Variant::PolygonChamfered,
            PolygonCornerStyle::Bezier { .. } => Variant::PolygonBezier,
        },
        CropShape::Star { .. } => Variant::Star,
    };

    let current_label = match variant {
        Variant::Rectangle => "Rectangle",
        Variant::RoundedRect => "Rounded rectangle",
        Variant::ChamferRect => "Chamfered rectangle",
        Variant::Ellipse => "Ellipse",
        Variant::PolygonSharp => "Polygon",
        Variant::PolygonRounded => "Polygon (rounded)",
        Variant::PolygonChamfered => "Polygon (chamfered)",
        Variant::PolygonBezier => "Polygon (bezier)",
        Variant::Star => "Star",
    };

    let mut variant_changed = false;
    ComboBox::from_label("Shape")
        .selected_text(current_label)
        .show_ui(ui, |ui| {
            let mut select_variant = |label: &str, target: Variant| {
                let selected = variant == target;
                if ui.selectable_label(selected, label).clicked() && !selected {
                    variant = target;
                    variant_changed = true;
                }
            };
            select_variant("Rectangle", Variant::Rectangle);
            select_variant("Rounded rectangle", Variant::RoundedRect);
            select_variant("Chamfered rectangle", Variant::ChamferRect);
            select_variant("Ellipse", Variant::Ellipse);
            select_variant("Polygon", Variant::PolygonSharp);
            select_variant("Polygon (rounded)", Variant::PolygonRounded);
            select_variant("Polygon (chamfered)", Variant::PolygonChamfered);
            select_variant("Polygon (bezier)", Variant::PolygonBezier);
            select_variant("Star", Variant::Star);
        });

    if variant_changed {
        shape = match variant {
            Variant::Rectangle => CropShape::Rectangle,
            Variant::RoundedRect => CropShape::RoundedRectangle { radius_pct: 0.12 },
            Variant::ChamferRect => CropShape::ChamferedRectangle { size_pct: 0.12 },
            Variant::Ellipse => CropShape::Ellipse,
            Variant::PolygonSharp => CropShape::Polygon {
                sides: 6,
                rotation_deg: 0.0,
                corner_style: PolygonCornerStyle::Sharp,
            },
            Variant::PolygonRounded => CropShape::Polygon {
                sides: 6,
                rotation_deg: 0.0,
                corner_style: PolygonCornerStyle::Rounded { radius_pct: 0.1 },
            },
            Variant::PolygonChamfered => CropShape::Polygon {
                sides: 6,
                rotation_deg: 0.0,
                corner_style: PolygonCornerStyle::Chamfered { size_pct: 0.1 },
            },
            Variant::PolygonBezier => CropShape::Polygon {
                sides: 6,
                rotation_deg: 0.0,
                corner_style: PolygonCornerStyle::Bezier { tension: 0.5 },
            },
            Variant::Star => CropShape::Star {
                points: 5,
                inner_radius_pct: 0.5,
                rotation_deg: 0.0,
            },
        };
        changed = true;
    }

    match &mut shape {
        CropShape::RoundedRectangle { radius_pct } => {
            let mut radius = (*radius_pct * 100.0).clamp(0.0, 50.0);
            crate::constrained_slider_row!(
                ui,
                &mut radius,
                0.0..=50.0,
                "Corner radius (%)",
                1.0,
                None,
                None,
                {
                    *radius_pct = (radius / 100.0).clamp(0.0, 0.5);
                    changed = true;
                }
            );
        }
        CropShape::ChamferedRectangle { size_pct } => {
            let mut size = (*size_pct * 100.0).clamp(0.0, 50.0);
            crate::constrained_slider_row!(
                ui,
                &mut size,
                0.0..=50.0,
                "Chamfer size (%)",
                1.0,
                None,
                None,
                {
                    *size_pct = (size / 100.0).clamp(0.0, 0.5);
                    changed = true;
                }
            );
        }
        CropShape::Polygon {
            sides,
            rotation_deg,
            corner_style,
        } => {
            let mut sides_u32 = *sides as u32;
            if ui
                .add(
                    DragValue::new(&mut sides_u32)
                        .range(3..=24)
                        .speed(1.0)
                        .suffix(" sides"),
                )
                .changed()
            {
                *sides = sides_u32.clamp(3, 24) as u8;
                changed = true;
            }
            crate::constrained_slider_row!(
                ui,
                rotation_deg,
                -180.0..=180.0,
                "Rotation (°)",
                1.0,
                None,
                None,
                {
                    changed = true;
                }
            );

            match corner_style {
                PolygonCornerStyle::Sharp => {}
                PolygonCornerStyle::Rounded { radius_pct } => {
                    let mut radius = (*radius_pct * 100.0).clamp(0.0, 40.0);
                    crate::constrained_slider_row!(
                        ui,
                        &mut radius,
                        0.0..=40.0,
                        "Corner radius (%)",
                        1.0,
                        None,
                        None,
                        {
                            *radius_pct = (radius / 100.0).clamp(0.0, 0.5);
                            changed = true;
                        }
                    );
                }
                PolygonCornerStyle::Chamfered { size_pct } => {
                    let mut size = (*size_pct * 100.0).clamp(0.0, 40.0);
                    crate::constrained_slider_row!(
                        ui,
                        &mut size,
                        0.0..=40.0,
                        "Chamfer size (%)",
                        1.0,
                        None,
                        None,
                        {
                            *size_pct = (size / 100.0).clamp(0.0, 0.5);
                            changed = true;
                        }
                    );
                }
                PolygonCornerStyle::Bezier { tension } => {
                    crate::constrained_slider_row!(
                        ui,
                        tension,
                        0.0..=2.0,
                        "Tension",
                        0.01,
                        Some(
                            "Adjusts the curvature of the corners. 0 is sharp, higher values are smoother."
                        ),
                        None,
                        {
                            changed = true;
                        }
                    );
                }
            }
        }
        CropShape::Star {
            points,
            inner_radius_pct,
            rotation_deg,
        } => {
            let mut points_u32 = *points as u32;
            if ui
                .add(
                    DragValue::new(&mut points_u32)
                        .range(3..=24)
                        .speed(1.0)
                        .suffix(" points"),
                )
                .changed()
            {
                *points = points_u32.clamp(3, 24) as u8;
                changed = true;
            }

            let mut inner = (*inner_radius_pct * 100.0).clamp(10.0, 90.0);
            crate::constrained_slider_row!(
                ui,
                &mut inner,
                10.0..=90.0,
                "Inner radius (%)",
                1.0,
                None,
                None,
                {
                    *inner_radius_pct = (inner / 100.0).clamp(0.1, 0.9);
                    changed = true;
                }
            );

            crate::constrained_slider_row!(
                ui,
                rotation_deg,
                -180.0..=180.0,
                "Rotation (°)",
                1.0,
                None,
                None,
                {
                    changed = true;
                }
            );
        }
        CropShape::Rectangle | CropShape::Ellipse => {}
    }

    let sanitized = shape.sanitized();
    if sanitized != app.settings.crop.shape {
        app.settings.crop.shape = sanitized;
        changed = true;
    }

    changed
}

/// Parses metadata tags from text input.
fn parse_metadata_tags(text: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((key, value)) = trimmed.split_once('=') {
            let key = key.trim();
            if key.is_empty() {
                continue;
            }
            map.insert(key.to_string(), value.trim().to_string());
        }
    }
    map
}
