//! Preview panel UI components for the YuNet GUI.

use egui::{
    Align, CentralPanel, Color32, CornerRadius, Layout, Margin, Rect, Rgba, RichText, Sense,
    Spinner, Stroke, Ui, UiBuilder, pos2, vec2,
};

use crate::{YuNetApp, theme};

impl YuNetApp {
    /// Renders the main image preview panel.
    pub fn show_preview(&mut self, ctx: &egui::Context) {
        let palette = theme::palette();
        CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(palette.canvas)
                    .inner_margin(Margin::symmetric(16, 16)),
            )
            .show(ctx, |ui| {
                let total_height = ui.available_height().max(360.0);
                let detection_height = (total_height * 0.25).max(180.0);
                let preview_height = (total_height - detection_height).max(220.0);
                let width = ui.available_width();

                ui.allocate_ui_with_layout(
                    vec2(width, preview_height),
                    Layout::top_down(Align::Center),
                    |ui| {
                        egui::Frame::new()
                            .fill(palette.panel_dark)
                            .stroke(Stroke::new(1.0, palette.outline))
                            .corner_radius(CornerRadius::same(28))
                            .inner_margin(Margin::symmetric(18, 18))
                            .show(ui, |ui| {
                                ui.set_min_height(preview_height);
                                self.render_preview_area(ui, ctx, palette);
                            });
                    },
                );

                ui.add_space(12.0);
                ui.allocate_ui_with_layout(
                    vec2(width, detection_height),
                    Layout::top_down(Align::Min),
                    |ui| {
                        ui.set_min_height(detection_height);
                        crate::ui::config::detections::show_detection_carousel(
                            self, ctx, ui, palette,
                        );
                    },
                );
            });
    }

    fn render_preview_area(&mut self, ui: &mut Ui, ctx: &egui::Context, palette: theme::Palette) {
        if let Some(texture) = self.preview.texture.clone() {
            let image_dimensions = self.preview.image_size;
            let available = ui.available_size();
            if available.x > 0.0 && available.y > 0.0 {
                let tex_size = texture.size_vec2();
                if tex_size.x > 0.0 && tex_size.y > 0.0 {
                    let scale = (available.x / tex_size.x)
                        .min(available.y / tex_size.y)
                        .max(0.0);
                    let scale = if scale.is_finite() && scale > 0.0 {
                        scale
                    } else {
                        1.0
                    };
                    let scaled = tex_size * scale;
                    ui.centered_and_justified(|ui| {
                        let image_widget = egui::Image::new(&texture).fit_to_exact_size(scaled);
                        let response = ui.add(image_widget);
                        if let Some(dimensions) = image_dimensions {
                            let image_rect = Rect::from_center_size(response.rect.center(), scaled);
                            self.handle_preview_interactions(ctx, image_rect, dimensions);
                            self.paint_detections(ui, image_rect, dimensions);
                            self.preview_overlay(ui, image_rect, palette);
                        }
                    });
                }
            }
        } else if self.preview.is_loading {
            ui.vertical_centered(|ui| {
                ui.add_space(64.0);
                ui.add(Spinner::new().size(28.0));
                ui.label(RichText::new("Loading image and running detection...").size(16.0));
            });
        } else {
            ui.vertical_centered(|ui| {
                ui.add_space(64.0);
                ui.heading("Drop an image or pick one from Quick Actions.");
                ui.label("The preview area will light up once detection finishes.");
            });
        }
    }

    fn preview_overlay(&mut self, ui: &mut Ui, image_rect: Rect, palette: theme::Palette) {
        let overlay_size = self.preview_hud_size();
        if image_rect.width() < overlay_size.x || image_rect.height() < overlay_size.y {
            return;
        }

        let mut overlay_rect = self.preview_hud_rect(image_rect, overlay_size);
        let overlay_id = ui.make_persistent_id("preview_hud_overlay");
        let drag_response = ui.interact(overlay_rect, overlay_id, Sense::drag());

        if drag_response.drag_started() {
            self.preview_hud_drag_origin = Some(overlay_rect.left_top());
        }

        if let Some(origin) = self.preview_hud_drag_origin {
            let desired = origin + drag_response.drag_delta();
            overlay_rect = self.update_hud_anchor_from_top_left(image_rect, overlay_size, desired);
        }

        if self.preview_hud_drag_origin.is_some() && !ui.ctx().input(|i| i.pointer.primary_down()) {
            self.preview_hud_drag_origin = None;
        }

        let hovered = drag_response.hovered() || drag_response.dragged();
        let fill = if hovered {
            palette.panel_dark
        } else {
            translucent_color(palette.panel_dark, 0.7)
        };

        ui.scope_builder(UiBuilder::new().max_rect(overlay_rect), |overlay_ui| {
            egui::Frame::new()
                .fill(fill)
                .stroke(Stroke::new(1.0, palette.outline))
                .corner_radius(CornerRadius::same(16))
                .inner_margin(Margin::symmetric(14, 10))
                .show(overlay_ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Preview HUD").strong());
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            let label = if self.preview_hud_minimized {
                                "Expand"
                            } else {
                                "Minimize"
                            };
                            if ui.small_button(label).clicked() {
                                self.preview_hud_minimized = !self.preview_hud_minimized;
                                ui.ctx().request_repaint();
                            }
                        });
                    });

                    if self.preview_hud_minimized {
                        ui.add_space(4.0);
                        ui.label(format!(
                            "Faces: {}  |  Selected: {}",
                            self.preview.detections.len(),
                            self.selected_faces.len()
                        ));
                        ui.label(
                            RichText::new("Drag to reposition / expand for actions")
                                .color(palette.subtle_text),
                        );
                        return;
                    }

                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
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
                    });

                    ui.add_space(6.0);
                    self.quality_legend(ui, palette);
                    ui.add_space(6.0);

                    if self.preview.detections.is_empty() {
                        ui.label(RichText::new("No faces detected yet.").weak());
                    } else {
                        ui.checkbox(&mut self.show_crop_overlay, "Show crop guides");
                    }

                    ui.add_space(6.0);
                    let selected = self.selected_faces.len();
                    if selected > 0 {
                        let button_label = format!(
                            "Export {} face{}",
                            selected,
                            if selected == 1 { "" } else { "s" }
                        );
                        if ui.button(button_label).clicked() {
                            self.export_selected_faces();
                        }
                    } else {
                        ui.label(
                            RichText::new("Select faces from the list to export.")
                                .color(palette.subtle_text),
                        );
                    }
                });
        });
    }

    fn preview_hud_size(&self) -> egui::Vec2 {
        if self.preview_hud_minimized {
            vec2(220.0, 72.0)
        } else {
            vec2(260.0, 190.0)
        }
    }

    fn preview_hud_rect(&self, image_rect: Rect, overlay_size: egui::Vec2) -> Rect {
        let available_width = (image_rect.width() - overlay_size.x).max(0.0);
        let available_height = (image_rect.height() - overlay_size.y).max(0.0);
        let anchor_x = self.preview_hud_anchor.x.clamp(0.0, 1.0);
        let anchor_y = self.preview_hud_anchor.y.clamp(0.0, 1.0);
        let top_left = pos2(
            image_rect.left() + anchor_x * available_width,
            image_rect.top() + anchor_y * available_height,
        );
        Rect::from_min_size(top_left, overlay_size)
    }

    fn update_hud_anchor_from_top_left(
        &mut self,
        image_rect: Rect,
        overlay_size: egui::Vec2,
        desired_top_left: egui::Pos2,
    ) -> Rect {
        let clamped = self.clamp_hud_top_left(image_rect, overlay_size, desired_top_left);
        self.preview_hud_anchor = self.anchor_from_top_left(image_rect, overlay_size, clamped);
        Rect::from_min_size(clamped, overlay_size)
    }

    fn clamp_hud_top_left(
        &self,
        image_rect: Rect,
        overlay_size: egui::Vec2,
        desired: egui::Pos2,
    ) -> egui::Pos2 {
        let min_x = image_rect.left();
        let min_y = image_rect.top();
        let max_x = (image_rect.right() - overlay_size.x).max(min_x);
        let max_y = (image_rect.bottom() - overlay_size.y).max(min_y);
        pos2(desired.x.clamp(min_x, max_x), desired.y.clamp(min_y, max_y))
    }

    fn anchor_from_top_left(
        &self,
        image_rect: Rect,
        overlay_size: egui::Vec2,
        top_left: egui::Pos2,
    ) -> egui::Vec2 {
        let denom_x = (image_rect.width() - overlay_size.x).max(1.0);
        let denom_y = (image_rect.height() - overlay_size.y).max(1.0);
        vec2(
            ((top_left.x - image_rect.left()) / denom_x).clamp(0.0, 1.0),
            ((top_left.y - image_rect.top()) / denom_y).clamp(0.0, 1.0),
        )
    }
}

/// Helper function to create a translucent color.
fn translucent_color(color: Color32, alpha_multiplier: f32) -> Color32 {
    let rgba: Rgba = color.into();
    let new_alpha = (rgba.a() * alpha_multiplier).clamp(0.0, 1.0);
    Rgba::from_rgba_premultiplied(rgba.r(), rgba.g(), rgba.b(), new_alpha).into()
}
