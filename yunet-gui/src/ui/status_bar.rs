//! Status bar UI components for the YuNet GUI.

use egui::{
    Align, Button, Color32, CornerRadius, Layout, Margin, Response, RichText, Spinner, Stroke,
    TopBottomPanel, Ui, vec2,
};

use crate::{GpuStatusMode, YuNetApp, theme};

impl YuNetApp {
    /// Renders the top status bar with quick stats and actions.
    pub fn show_status_bar(&mut self, ctx: &egui::Context) {
        let palette = theme::palette();
        TopBottomPanel::top("yunet_status_bar")
            .frame(
                egui::Frame::new()
                    .fill(palette.panel_dark)
                    .stroke(Stroke::new(1.0, palette.outline))
                    .inner_margin(Margin::symmetric(20, 16)),
            )
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 6.0;
                    ui.horizontal(|ui| {
                        ui.heading(RichText::new("YuNet Studio").size(26.0).strong());
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            self.draw_status_badge(ui, palette);
                        });
                    });

                    ui.label(RichText::new(&self.status_line).color(palette.subtle_text));

                    if let Some(err) = &self.last_error {
                        ui.colored_label(palette.danger, err);
                    } else if self.preview.detections.is_empty() && !self.is_busy {
                        ui.label(
                            RichText::new("Choose an image to begin detecting faces.")
                                .color(palette.subtle_text),
                        );
                    }

                    ui.add_space(6.0);
                    self.draw_status_chips(ui, palette);
                    ui.add_space(10.0);
                    self.draw_quick_actions(ui, palette);
                });
            });
    }

    fn draw_status_badge(&self, ui: &mut Ui, palette: theme::Palette) {
        let (label, color) = if self.is_busy {
            ("Detecting...", palette.accent)
        } else if self.detector.is_none() {
            ("Model Required", palette.warning)
        } else {
            ("Ready", palette.success)
        };

        egui::Frame::new()
            .fill(palette.panel_light)
            .stroke(Stroke::new(1.0, color))
            .corner_radius(CornerRadius::same(64))
            .inner_margin(Margin::symmetric(14, 6))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if self.is_busy {
                        ui.add(Spinner::new().size(16.0));
                    }
                    ui.label(RichText::new(label).size(15.0).strong());
                });
            });
    }

    fn draw_status_chips(&self, ui: &mut Ui, palette: theme::Palette) {
        ui.horizontal_wrapped(|ui| {
            self.status_chip(
                ui,
                palette,
                format!("Faces {}", self.preview.detections.len()),
                palette.accent,
            );
            self.status_chip(
                ui,
                palette,
                format!("Selected {}", self.selected_faces.len()),
                if self.selected_faces.is_empty() {
                    palette.subtle_text
                } else {
                    palette.success
                },
            );

            self.status_chip(
                ui,
                palette,
                format!("Batch {}", self.batch_files.len()),
                if self.batch_files.is_empty() {
                    palette.subtle_text
                } else {
                    palette.warning
                },
            );
            let (gpu_text, gpu_color) = self.gpu_status_chip(palette);
            self.status_chip(ui, palette, gpu_text, gpu_color);
        });
    }

    fn draw_quick_actions(&mut self, ui: &mut Ui, palette: theme::Palette) {
        ui.horizontal_wrapped(|ui| {
            if self
                .quick_action_button(ui, palette, "Open Image", "Pick a single file", true)
                .clicked()
            {
                self.open_image_dialog();
            }
            if self
                .quick_action_button(ui, palette, "Load Batch", "Queue multiple images", true)
                .clicked()
            {
                self.open_batch_dialog();
            }

            let export_enabled = !self.selected_faces.is_empty();
            if self
                .quick_action_button(
                    ui,
                    palette,
                    "Export Selected",
                    "Sends crops to disk",
                    export_enabled,
                )
                .clicked()
            {
                self.export_selected_faces();
            }

            let batch_enabled = !self.batch_files.is_empty();
            let subtitle = format!("{} queued", self.batch_files.len());
            if self
                .quick_action_button(ui, palette, "Run Batch", &subtitle, batch_enabled)
                .clicked()
            {
                self.start_batch_export();
            }
        });
    }

    fn quick_action_button(
        &self,
        ui: &mut Ui,
        palette: theme::Palette,
        title: &str,
        subtitle: &str,
        enabled: bool,
    ) -> Response {
        let text = format!("{title}\n{subtitle}");
        ui.add_enabled(
            enabled,
            Button::new(RichText::new(text).size(15.0))
                .wrap()
                .min_size(vec2(150.0, 64.0))
                .fill(if enabled {
                    palette.panel_light
                } else {
                    palette.panel_dark
                })
                .stroke(Stroke::new(1.0, palette.outline))
                .corner_radius(CornerRadius::same(16)),
        )
    }

    pub(crate) fn status_chip(
        &self,
        ui: &mut Ui,
        palette: theme::Palette,
        text: impl Into<String>,
        accent: Color32,
    ) {
        status_chip_helper(self, ui, palette, text, accent);
    }

    fn gpu_status_chip(&self, palette: theme::Palette) -> (String, Color32) {
        use GpuStatusMode::*;

        let mode = self.gpu_status.mode;
        let color = match mode {
            Available => palette.success,
            Disabled => palette.subtle_text,
            Pending => palette.subtle_text,
            Fallback => palette.warning,
            Error => palette.danger,
        };

        let text = match mode {
            Available => {
                let adapter = self
                    .gpu_status
                    .adapter_name
                    .as_deref()
                    .unwrap_or("GPU ready");
                let backend = self.gpu_status.backend.as_deref().unwrap_or("wgpu");
                format!("GPU {adapter} ({backend})")
            }
            Disabled => "GPU disabled".to_string(),
            Pending => "GPU probing…".to_string(),
            Fallback => self
                .gpu_status
                .detail
                .as_deref()
                .map(|reason| format!("GPU fallback ({reason})"))
                .unwrap_or_else(|| "GPU fallback".to_string()),
            Error => self
                .gpu_status
                .detail
                .as_deref()
                .map(|detail| format!("GPU error ({detail})"))
                .unwrap_or_else(|| "GPU error".to_string()),
        };

        (text, color)
    }
}

/// Helper function for drawing status chips (exported for use in other modules).
pub(crate) fn status_chip_helper(
    _app: &crate::YuNetApp,
    ui: &mut Ui,
    palette: crate::theme::Palette,
    text: impl Into<String>,
    accent: Color32,
) {
    egui::Frame::new()
        .fill(palette.panel_dark)
        .stroke(Stroke::new(1.0, accent))
        .corner_radius(CornerRadius::same(24))
        .inner_margin(Margin::symmetric(12, 4))
        .show(ui, |ui| {
            ui.label(
                RichText::new(text.into())
                    .size(14.0)
                    .color(palette.subtle_text),
            );
        });
}
