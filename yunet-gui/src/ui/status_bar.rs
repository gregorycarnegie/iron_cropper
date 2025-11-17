//! Status bar UI components for the YuNet GUI.

use egui::{
    Align, Color32, CornerRadius, Layout, Margin, RichText, Spinner, Stroke, TopBottomPanel, Ui,
};

use crate::{GpuStatusMode, YuNetApp, theme};

impl YuNetApp {
    /// Renders the slim bottom status bar with stats and status badge.
    pub fn show_status_bar(&mut self, ctx: &egui::Context) {
        let palette = theme::palette();
        TopBottomPanel::bottom("yunet_status_bar")
            .frame(
                egui::Frame::new()
                    .fill(palette.canvas)
                    .stroke(Stroke::new(1.0, palette.outline))
                    .inner_margin(Margin::symmetric(16, 8)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    // Left side: status chips
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

                    // Right side: status badge
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        self.draw_status_badge(ui, palette);
                    });
                });
            });
    }

    pub(crate) fn can_export_selected(&self) -> bool {
        !self.selected_faces.is_empty() && !self.preview.detections.is_empty()
    }

    fn draw_status_badge(&self, ui: &mut Ui, palette: theme::Palette) {
        let (label, color) = if self.is_busy {
            ("Detecting…", palette.accent)
        } else if self.detector.is_none() {
            ("Model required", palette.warning)
        } else {
            ("Ready", palette.success)
        };

        egui::Frame::new()
            .fill(palette.panel_light)
            .stroke(Stroke::new(1.0, color))
            .corner_radius(CornerRadius::same(20))
            .inner_margin(Margin::symmetric(18, 8))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if self.is_busy {
                        ui.add(Spinner::new().size(16.0));
                    }
                    ui.label(RichText::new(label).size(15.0).strong());
                });
            });
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
            Pending => "GPU probing...".to_string(),
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

/// Helper function for drawing status chips.
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
        .inner_margin(Margin::symmetric(12, 6))
        .show(ui, |ui| {
            ui.label(
                RichText::new(text.into())
                    .size(14.0)
                    .color(palette.subtle_text),
            );
        });
}
