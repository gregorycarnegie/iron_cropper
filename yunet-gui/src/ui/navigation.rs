//! Navigation rail shown on the left edge of the application.

use egui::{
    Align, CornerRadius, Frame, Layout, Margin, RichText, Sense, SidePanel, Stroke, StrokeKind, Ui,
    Vec2, emath::Align2,
};

use crate::{YuNetApp, theme};

impl YuNetApp {
    /// Renders the compact navigation rail with quick actions.
    pub fn show_navigation_panel(&mut self, ctx: &egui::Context) {
        let palette = theme::palette();
        SidePanel::left("yunet_navigation_panel")
            .resizable(false)
            .exact_width(150.0)
            .frame(
                Frame::new()
                    .fill(palette.panel_dark)
                    .inner_margin(Margin::symmetric(12, 18))
                    .stroke(Stroke::new(1.0, palette.outline)),
            )
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    self.draw_nav_logo(ui, palette);
                    ui.add_space(14.0);
                    self.nav_button(
                        ui,
                        palette,
                        "Import",
                        "Pick a single image",
                        !self.is_busy,
                        |app| app.open_image_dialog(),
                    );
                    self.nav_button(ui, palette, "Batch", "Queue multiple files", true, |app| {
                        app.open_batch_dialog()
                    });
                    self.nav_button(ui, palette, "Settings", "Choose YuNet model", true, |app| {
                        app.open_model_dialog()
                    });
                    self.nav_button(ui, palette, "Mapping", "Load CSV / Excel", true, |app| {
                        app.pick_mapping_file_from_dialog()
                    });
                });

                ui.with_layout(Layout::bottom_up(Align::Center), |ui| {
                    ui.add_space(8.0);
                    self.nav_stat(
                        ui,
                        palette,
                        "Selected",
                        format!("{}", self.selected_faces.len()),
                    );
                    self.nav_stat(
                        ui,
                        palette,
                        "Detections",
                        format!("{}", self.preview.detections.len()),
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

    fn draw_nav_logo(&self, ui: &mut Ui, palette: theme::Palette) {
        let (rect, _) = ui.allocate_exact_size(Vec2::splat(56.0), Sense::hover());
        let painter = ui.painter();
        painter.circle_filled(rect.center(), 26.0, palette.panel_light);
        painter.circle_filled(rect.center(), 20.0, palette.canvas);
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            "FC",
            egui::FontId::new(20.0, egui::FontFamily::Proportional),
            palette.accent,
        );
        ui.label(
            RichText::new("Face Crop Studio")
                .size(15.0)
                .strong()
                .color(palette.subtle_text),
        );
    }

    fn nav_button<F>(
        &mut self,
        ui: &mut Ui,
        palette: theme::Palette,
        title: &str,
        subtitle: &str,
        enabled: bool,
        mut action: F,
    ) where
        F: FnMut(&mut Self),
    {
        let desired = Vec2::new(ui.available_width(), 72.0);
        let (rect, response) = ui.allocate_exact_size(desired, Sense::click());
        let hover = response.hovered();
        let fill = if enabled {
            if hover {
                palette.panel_light
            } else {
                palette.panel
            }
        } else {
            palette.panel_dark
        };
        ui.painter().rect(
            rect,
            CornerRadius::same(18),
            fill,
            Stroke::new(1.0, palette.outline),
            StrokeKind::Outside,
        );

        let label = format!("{title}\n{subtitle}");
        let text_color = if enabled {
            palette.subtle_text
        } else {
            palette.subtle_text.gamma_multiply(0.6)
        };
        ui.painter().text(
            rect.left_top() + Vec2::new(14.0, 12.0),
            Align2::LEFT_TOP,
            label,
            egui::FontId::new(15.0, egui::FontFamily::Proportional),
            text_color,
        );

        if enabled && response.clicked() {
            action(self);
        }
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
