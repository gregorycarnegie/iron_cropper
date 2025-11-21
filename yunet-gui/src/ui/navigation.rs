//! Left panel with simple/common settings and quick actions.

use egui::{Align, Frame, Layout, Margin, RichText, ScrollArea, SidePanel, Slider, Stroke, Ui};

use crate::{YuNetApp, theme};

impl YuNetApp {
    /// Renders the left panel with common settings and actions.
    pub fn show_navigation_panel(&mut self, ctx: &egui::Context) {
        let palette = theme::palette();
        SidePanel::left("yunet_navigation_panel")
            .resizable(false)
            .exact_width(280.0)
            .frame(
                Frame::new()
                    .fill(palette.panel)
                    .inner_margin(Margin::symmetric(16, 18))
                    .stroke(Stroke::new(1.0, palette.outline)),
            )
            .show(ctx, |ui| {
                ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        self.draw_nav_logo(ui, palette);
                        ui.add_space(18.0);

                        // Quick actions
                        ui.heading("Quick Actions");
                        ui.add_space(8.0);

                        let btn_width = (ui.available_width() - 8.0) / 2.0;
                        egui::Grid::new("quick_actions_grid")
                            .num_columns(2)
                            .spacing([8.0, 8.0])
                            .show(ui, |ui| {
                                if ui
                                    .add_sized([btn_width, 30.0], egui::Button::new("Open image"))
                                    .clicked()
                                {
                                    self.open_image_dialog();
                                }
                                if ui
                                    .add_sized([btn_width, 30.0], egui::Button::new("Load batch"))
                                    .clicked()
                                {
                                    self.open_batch_dialog();
                                }
                                ui.end_row();

                                if ui
                                    .add_sized([btn_width, 30.0], egui::Button::new("Batch Queue"))
                                    .clicked()
                                {
                                    self.show_batch_window = true;
                                }
                                if ui
                                    .add_sized(
                                        [btn_width, 30.0],
                                        egui::Button::new("Mapping Import"),
                                    )
                                    .clicked()
                                {
                                    self.show_mapping_window = true;
                                }
                                ui.end_row();
                            });

                        egui::Grid::new("quick_actions_grid_2")
                            .num_columns(2)
                            .spacing([8.0, 8.0])
                            .show(ui, |ui| {
                                let export_enabled = self.can_export_selected();
                                if ui
                                    .add_enabled(
                                        export_enabled,
                                        egui::Button::new("Export selected")
                                            .min_size(egui::vec2(btn_width, 30.0)),
                                    )
                                    .clicked()
                                {
                                    self.export_selected_faces();
                                }
                                let batch_enabled = !self.batch_files.is_empty();
                                if ui
                                    .add_enabled(
                                        batch_enabled,
                                        egui::Button::new("Run batch")
                                            .min_size(egui::vec2(btn_width, 30.0)),
                                    )
                                    .clicked()
                                {
                                    self.start_batch_export();
                                }
                                ui.end_row();
                            });

                        ui.add_space(12.0);
                        ui.separator();
                        ui.add_space(8.0);

                        // Simple settings
                        self.show_simple_settings(ui, palette);

                        ui.add_space(12.0);
                        ui.separator();
                        ui.add_space(8.0);

                        // Stats
                        ui.heading("Status");
                        ui.add_space(8.0);
                        self.nav_stat(
                            ui,
                            palette,
                            "Detections",
                            format!("{}", self.preview.detections.len()),
                        );
                        self.nav_stat(
                            ui,
                            palette,
                            "Selected",
                            format!("{}", self.selected_faces.len()),
                        );
                        self.nav_stat(
                            ui,
                            palette,
                            "Batch",
                            if self.batch_files.is_empty() {
                                "Empty".to_string()
                            } else {
                                format!("{} items", self.batch_files.len())
                            },
                        );
                    });
            });
    }

    fn show_simple_settings(&mut self, ui: &mut Ui, _palette: theme::Palette) {
        ui.heading("Adjustments");
        ui.add_space(6.0);

        egui::Grid::new("simple_settings_grid")
            .num_columns(3)
            .spacing([8.0, 8.0])
            .show(ui, |ui| {
                ui.label("Face fill (%)");
                let mut face_fill = self.settings.crop.face_height_pct;
                let r1 = ui.add(Slider::new(&mut face_fill, 20.0..=95.0).show_value(false));
                let r2 = ui.add_sized(
                    [50.0, 20.0],
                    egui::DragValue::new(&mut face_fill).speed(1.0),
                );
                if r1.changed() || r2.changed() {
                    self.settings.crop.face_height_pct = face_fill;
                    self.push_crop_history();
                    self.persist_settings_with_feedback();
                    self.clear_crop_preview_cache();
                }
                ui.end_row();

                ui.label("Eye alignment");
                let mut horizontal = self.settings.crop.horizontal_offset;
                let r1 = ui.add(Slider::new(&mut horizontal, -1.0..=1.0).show_value(false));
                let r2 = ui.add_sized(
                    [50.0, 20.0],
                    egui::DragValue::new(&mut horizontal).speed(0.01),
                );
                if r1.changed() || r2.changed() {
                    self.settings.crop.horizontal_offset = horizontal.clamp(-1.0, 1.0);
                    self.push_crop_history();
                    self.persist_settings_with_feedback();
                    self.clear_crop_preview_cache();
                }
                ui.end_row();

                ui.label("Vertical lift");
                let mut vertical = self.settings.crop.vertical_offset;
                let r1 = ui.add(Slider::new(&mut vertical, -1.0..=1.0).show_value(false));
                let r2 = ui.add_sized(
                    [50.0, 20.0],
                    egui::DragValue::new(&mut vertical).speed(0.01),
                );
                if r1.changed() || r2.changed() {
                    self.settings.crop.vertical_offset = vertical.clamp(-1.0, 1.0);
                    self.push_crop_history();
                    self.persist_settings_with_feedback();
                    self.clear_crop_preview_cache();
                }
                ui.end_row();
            });

        ui.add_space(8.0);
        ui.label(RichText::new("Automation").strong());
        if ui
            .checkbox(
                &mut self.settings.crop.quality_rules.auto_select_best_face,
                "Auto-select best face",
            )
            .changed()
        {
            self.push_crop_history();
            self.persist_settings_with_feedback();
            self.apply_quality_rules_to_preview();
        }
        if ui
            .checkbox(
                &mut self.settings.crop.quality_rules.auto_skip_no_high_quality,
                "Skip export without high-quality face",
            )
            .changed()
        {
            self.push_crop_history();
            self.persist_settings_with_feedback();
        }

        ui.add_space(8.0);
        ui.label(RichText::new("Enhancements").strong());

        let mut enhance_enabled = self.settings.enhance.enabled;
        if ui
            .checkbox(&mut enhance_enabled, "Enable enhancements")
            .changed()
        {
            self.settings.enhance.enabled = enhance_enabled;
            self.persist_settings_with_feedback();
            self.clear_crop_preview_cache();
            if !self.preview.detections.is_empty() {
                for idx in 0..self.preview.detections.len() {
                    let _ = self.crop_preview_texture_for(ui.ctx(), idx);
                }
            }
            ui.ctx().request_repaint();
        }

        ui.add_enabled_ui(self.settings.enhance.enabled, |ui| {
            let mut background_blur = self.settings.enhance.background_blur;
            if ui
                .checkbox(&mut background_blur, "Background blur")
                .changed()
            {
                self.settings.enhance.background_blur = background_blur;
                self.persist_settings_with_feedback();
                self.clear_crop_preview_cache();
            }
            let mut red_eye = self.settings.enhance.red_eye_removal;
            if ui.checkbox(&mut red_eye, "Red-eye removal").changed() {
                self.settings.enhance.red_eye_removal = red_eye;
                self.persist_settings_with_feedback();
                self.clear_crop_preview_cache();
            }
        });
    }

    fn draw_nav_logo(&self, ui: &mut Ui, palette: theme::Palette) {
        ui.vertical_centered(|ui| {
            ui.add_space(8.0);

            // Install image loaders once (safe to call multiple times)
            egui_extras::install_image_loaders(ui.ctx());

            // Load and display the SVG logo using egui_extras
            let logo_size = egui::vec2(80.0, 80.0);
            ui.add(
                egui::Image::from_bytes(
                    "bytes://app_logo.svg",
                    include_bytes!("../../assets/app_logo.svg"),
                )
                .max_size(logo_size),
            );

            ui.add_space(8.0);
            ui.label(
                RichText::new("Face Crop Studio")
                    .size(17.0)
                    .strong()
                    .color(egui::Color32::WHITE),
            );
            ui.label(
                RichText::new("Batch-perfect crops with YuNet precision")
                    .size(10.0)
                    .color(palette.subtle_text),
            );
            ui.add_space(4.0);
        });
    }

    fn nav_stat(&self, ui: &mut Ui, palette: theme::Palette, label: &str, value: String) {
        Frame::new()
            .fill(palette.panel)
            .stroke(Stroke::new(1.0, palette.outline))
            .inner_margin(Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                    ui.label(
                        RichText::new(value)
                            .size(16.0)
                            .strong()
                            .color(palette.accent),
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(RichText::new(label).size(12.0).color(palette.subtle_text));
                    });
                });
            });
    }
}
