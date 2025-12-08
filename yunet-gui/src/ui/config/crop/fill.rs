use crate::YuNetApp;
use crate::types::ColorMode;
use crate::ui::widgets;
use egui::{Color32, TextEdit, Ui, color_picker};
use yunet_utils::{
    RgbaColor, cmyk_to_rgb, hsl_to_rgb, hsv_to_rgb, parse_hex_color, rgb_to_cmyk, rgb_to_hsl,
    rgb_to_hsv,
};

macro_rules! color_input_cell {
    ($ui:expr, $val:expr, $range:expr, $width:expr, $suffix:expr, $changed:ident, $func:path) => {
        $ui.horizontal(|ui| {
            if $func(ui, $val, $range, $width, Some($suffix)).changed() {
                $changed = true;
            }
        });
    };
}

pub fn show_fill_color_controls(app: &mut YuNetApp, ui: &mut Ui) -> bool {
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
                    let modes = [
                        (ColorMode::Rgb, "RGB"),
                        (ColorMode::Hsv, "HSV"),
                        (ColorMode::Hsl, "HSL"),
                        (ColorMode::Cmyk, "CMYK"),
                    ];
                    for (mode, label) in modes {
                        let selected = app.fill_color_mode == mode;
                        let icon = match mode {
                            ColorMode::Rgb => app.icons.rgb(14.0),
                            ColorMode::Hsv => app.icons.hsv(14.0),
                            ColorMode::Hsl => app.icons.hsl(14.0),
                            ColorMode::Cmyk => app.icons.cmyk(14.0),
                        };
                        if ui
                            .add(egui::Button::image_and_text(icon, label).selected(selected))
                            .clicked()
                        {
                            app.fill_color_mode = mode;
                        }
                    }
                });
            ui.end_row();

            match app.fill_color_mode {
                ColorMode::Rgb => {
                    ui.label("RGB");
                    egui::Grid::new("rgb_inner_grid")
                        .num_columns(3)
                        .spacing([8.0, 0.0])
                        .show(ui, |ui| {
                            color_input_cell!(
                                ui,
                                &mut r,
                                0..=255,
                                60.0,
                                " R",
                                rgb_changed,
                                widgets::integer_input
                            );
                            color_input_cell!(
                                ui,
                                &mut g,
                                0..=255,
                                60.0,
                                " G",
                                rgb_changed,
                                widgets::integer_input
                            );
                            color_input_cell!(
                                ui,
                                &mut b,
                                0..=255,
                                60.0,
                                " B",
                                rgb_changed,
                                widgets::integer_input
                            );
                        });
                }
                ColorMode::Hsv => {
                    ui.label("HSV");
                    egui::Grid::new("hsv_inner_grid")
                        .num_columns(3)
                        .spacing([8.0, 0.0])
                        .show(ui, |ui| {
                            color_input_cell!(
                                ui,
                                &mut hue,
                                0.0..=360.0,
                                60.0,
                                "°",
                                hsv_changed,
                                widgets::numeric_input
                            );
                            color_input_cell!(
                                ui,
                                &mut sat_pct,
                                0.0..=100.0,
                                60.0,
                                " % S",
                                hsv_changed,
                                widgets::numeric_input
                            );
                            color_input_cell!(
                                ui,
                                &mut val_pct,
                                0.0..=100.0,
                                60.0,
                                " % V",
                                hsv_changed,
                                widgets::numeric_input
                            );
                        });
                }
                ColorMode::Hsl => {
                    ui.label("HSL");
                    egui::Grid::new("hsl_inner_grid")
                        .num_columns(3)
                        .spacing([8.0, 0.0])
                        .show(ui, |ui| {
                            color_input_cell!(
                                ui,
                                &mut hue_l,
                                0.0..=360.0,
                                60.0,
                                "°",
                                hsl_changed,
                                widgets::numeric_input
                            );
                            color_input_cell!(
                                ui,
                                &mut sat_l_pct,
                                0.0..=100.0,
                                60.0,
                                " % S",
                                hsl_changed,
                                widgets::numeric_input
                            );
                            color_input_cell!(
                                ui,
                                &mut light_pct,
                                0.0..=100.0,
                                60.0,
                                " % L",
                                hsl_changed,
                                widgets::numeric_input
                            );
                        });
                }
                ColorMode::Cmyk => {
                    ui.label("CMYK");
                    egui::Grid::new("cmyk_inner_grid")
                        .num_columns(4)
                        .spacing([8.0, 0.0])
                        .show(ui, |ui| {
                            color_input_cell!(
                                ui,
                                &mut c_pct,
                                0.0..=100.0,
                                50.0,
                                " % C",
                                cmyk_changed,
                                widgets::numeric_input
                            );
                            color_input_cell!(
                                ui,
                                &mut m_pct,
                                0.0..=100.0,
                                50.0,
                                " % M",
                                cmyk_changed,
                                widgets::numeric_input
                            );
                            color_input_cell!(
                                ui,
                                &mut y_pct,
                                0.0..=100.0,
                                50.0,
                                " % Y",
                                cmyk_changed,
                                widgets::numeric_input
                            );
                            color_input_cell!(
                                ui,
                                &mut k_pct,
                                0.0..=100.0,
                                50.0,
                                " % K",
                                cmyk_changed,
                                widgets::numeric_input
                            );
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
