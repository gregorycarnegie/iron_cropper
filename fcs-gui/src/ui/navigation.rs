//! Left panel with simple/common settings and quick actions.

use crate::{YuNetApp, theme};

use egui::{
    Align, Button, DragValue, Frame, Grid, Image, Layout, Margin, RichText, ScrollArea, Stroke, Ui,
};

impl YuNetApp {
    /// Renders the left panel with common settings and actions.
    pub fn show_navigation_panel(&mut self, ui: &mut Ui) {
        let palette = theme::palette();
        let icon_size = self.icons.default_size();

        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.draw_nav_logo(ui, palette);
                ui.add_space(18.0);

                // Quick actions
                ui.horizontal(|ui| {
                    ui.add(self.icons.workflow(icon_size).tint(palette.accent));
                    ui.heading("Quick Actions");
                });
                ui.add_space(8.0);

                let btn_width = (ui.available_width() - 8.0) * 0.5;
                let button_height = 32.0;
                Grid::new("quick_actions_grid")
                    .num_columns(2)
                    .spacing([8.0, 8.0])
                    .show(ui, |ui| {
                        if ui
                            .add_sized(
                                [btn_width, button_height],
                                Button::image_and_text(self.icons.photo(icon_size), "Open image"),
                            )
                            .clicked()
                        {
                            self.open_image_dialog();
                        }
                        if ui
                            .add_sized(
                                [btn_width, button_height],
                                Button::image_and_text(
                                    self.icons.folder_open(icon_size),
                                    "Load folder",
                                ),
                            )
                            .clicked()
                        {
                            self.open_folder_dialog();
                        }
                        ui.end_row();

                        if ui
                            .add_sized(
                                [btn_width, button_height],
                                Button::image_and_text(self.icons.gallery(icon_size), "Load batch"),
                            )
                            .clicked()
                        {
                            self.open_batch_dialog();
                        }
                        if ui
                            .add_sized(
                                [btn_width, button_height],
                                Button::image_and_text(self.icons.batch(icon_size), "Batch Queue"),
                            )
                            .clicked()
                        {
                            self.show_batch_window = true;
                        }
                        ui.end_row();

                        if ui
                            .add_sized(
                                [btn_width, button_height],
                                Button::image_and_text(self.icons.network(icon_size), "Import Map"),
                            )
                            .clicked()
                        {
                            self.show_mapping_window = true;
                        }
                        ui.end_row();
                    });

                Grid::new("quick_actions_grid_2")
                    .num_columns(2)
                    .spacing([8.0, 8.0])
                    .show(ui, |ui| {
                        let export_enabled = self.can_export_selected();
                        ui.add_enabled_ui(export_enabled, |ui| {
                            if ui
                                .add_sized(
                                    [btn_width, button_height],
                                    Button::image_and_text(self.icons.export(icon_size), "Export"),
                                )
                                .clicked()
                            {
                                self.export_selected_faces();
                            }
                        });

                        let batch_enabled = !self.batch_files.is_empty();
                        ui.add_enabled_ui(batch_enabled, |ui| {
                            if ui
                                .add_sized(
                                    [btn_width, button_height],
                                    Button::image_and_text(self.icons.run(icon_size), "Run batch"),
                                )
                                .clicked()
                            {
                                self.start_batch_export();
                            }
                        });
                        ui.end_row();
                    });

                ui.add_space(12.0);
                ui.separator();
                ui.add_space(8.0);

                // Webcam section
                self.show_webcam_controls(ui, icon_size);

                ui.add_space(12.0);
                ui.separator();
                ui.add_space(8.0);

                // Simple settings
                self.show_simple_settings(ui, palette, icon_size);

                ui.add_space(12.0);
                ui.separator();
                ui.add_space(8.0);

                // Stats
                ui.horizontal(|ui| {
                    ui.add(self.icons.spreadsheet(icon_size).tint(palette.subtle_text));
                    ui.heading("Status");
                });
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
    }

    fn show_simple_settings(&mut self, ui: &mut Ui, palette: theme::Palette, icon_size: f32) {
        ui.horizontal(|ui| {
            ui.add(self.icons.palette(icon_size).tint(palette.accent));
            ui.heading("Adjustments");
        });
        ui.add_space(6.0);

        let mut face_fill = self.settings.crop.face_height_pct;
        crate::constrained_slider_row!(
            ui,
            &mut face_fill,
            20.0..=95.0,
            "Face fill (%)",
            1.0,
            None,
            None,
            {
                self.settings.crop.face_height_pct = face_fill;
                self.push_crop_history();
                self.persist_settings_with_feedback();
                self.clear_crop_preview_cache();
            }
        );

        let mut horizontal = self.settings.crop.horizontal_offset;
        crate::constrained_slider_row!(
            ui,
            &mut horizontal,
            -1.0..=1.0,
            "Eye alignment",
            0.01,
            None,
            None,
            {
                self.settings.crop.horizontal_offset = horizontal.clamp(-1.0, 1.0);
                self.push_crop_history();
                self.persist_settings_with_feedback();
                self.clear_crop_preview_cache();
            }
        );

        let mut vertical = self.settings.crop.vertical_offset;
        crate::constrained_slider_row!(
            ui,
            &mut vertical,
            -1.0..=1.0,
            "Vertical lift",
            0.01,
            None,
            None,
            {
                self.settings.crop.vertical_offset = vertical.clamp(-1.0, 1.0);
                self.push_crop_history();
                self.persist_settings_with_feedback();
                self.clear_crop_preview_cache();
            }
        );

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.add(self.icons.automation(icon_size).tint(palette.subtle_text));
            ui.label(RichText::new("Automation").strong());
        });
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
        ui.horizontal(|ui| {
            ui.add(self.icons.enhance(icon_size).tint(palette.accent));
            ui.label(RichText::new("Enhancements").strong());
        });

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
            let logo_size = egui::vec2(160.0, 160.0);
            ui.add(
                Image::from_bytes(
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

    fn show_webcam_controls(&mut self, ui: &mut Ui, icon_size: f32) {
        use crate::types::WebcamStatus;

        let palette = theme::palette();
        ui.heading("Webcam");
        ui.add_space(8.0);

        let is_active = matches!(
            self.webcam_state.status,
            WebcamStatus::Active | WebcamStatus::Starting
        );
        let is_inactive = matches!(self.webcam_state.status, WebcamStatus::Inactive);

        // Start/Stop button
        let btn_text = if is_active {
            "Stop Webcam"
        } else {
            "Start Webcam"
        };
        let button_icon = self.icons.webcam(icon_size).tint(if is_active {
            palette.danger
        } else {
            palette.success
        });

        if ui
            .add_sized(
                [ui.available_width(), 32.0],
                Button::image_and_text(button_icon, btn_text),
            )
            .clicked()
        {
            if is_active {
                self.stop_webcam();
            } else {
                self.start_webcam();
            }
        }

        // Status indicator
        let (status_text, status_color) = match self.webcam_state.status {
            WebcamStatus::Inactive => ("Inactive", palette.subtle_text),
            WebcamStatus::Starting => ("Starting...", palette.accent),
            WebcamStatus::Active => ("Active", palette.success),
            WebcamStatus::Stopping => ("Stopping...", palette.warning),
            WebcamStatus::Error => ("Error", palette.danger),
        };
        ui.horizontal(|ui| {
            ui.add(
                self.icons
                    .webcam((icon_size - 2.0).max(14.0))
                    .tint(status_color),
            );
            ui.colored_label(status_color, status_text);
        });

        // Show stats when active
        if is_active {
            ui.add_space(4.0);
            ui.label(format!("Frames: {}", self.webcam_state.frames_captured));
            ui.label(format!("Faces: {}", self.webcam_state.total_faces));
            ui.label(format!(
                "Resolution: {}x{}",
                self.webcam_state.width, self.webcam_state.height
            ));
        }

        // Show error message if any
        if let Some(ref error) = self.webcam_state.error_message {
            ui.add_space(4.0);
            ui.colored_label(palette.danger, error);
        }

        // Configuration (only when inactive)
        ui.add_enabled_ui(is_inactive, |ui| {
            ui.add_space(8.0);
            ui.label("Device:");
            ui.add(
                DragValue::new(&mut self.webcam_state.device_index)
                    .speed(1.0)
                    .range(0..=10),
            );

            ui.horizontal(|ui| {
                ui.label("Resolution:");
                ui.add(
                    DragValue::new(&mut self.webcam_state.width)
                        .speed(1.0)
                        .range(320..=1920),
                );
                ui.label("x");
                ui.add(
                    DragValue::new(&mut self.webcam_state.height)
                        .speed(1.0)
                        .range(240..=1080),
                );
            });

            ui.horizontal(|ui| {
                ui.label("FPS:");
                ui.add(
                    DragValue::new(&mut self.webcam_state.fps)
                        .speed(1.0)
                        .range(1..=60),
                );
            });

            ui.checkbox(&mut self.webcam_state.show_overlay, "Show detections");
            ui.checkbox(&mut self.webcam_state.auto_crop, "Auto-save crops");
        });
    }
}
