//! Preview panel UI components for the YuNet GUI.

use egui::{
    Align, Color32, CornerRadius, Layout, Margin, Rect, Rgba, RichText, Sense, Spinner, Stroke, Ui,
    UiBuilder, pos2, vec2,
};

use crate::{YuNetApp, theme};

impl YuNetApp {
    /// Renders the main image preview panel.
    pub fn show_preview(&mut self, ui: &mut Ui, ctx: &egui::Context) {
        let palette = theme::palette();

        let total_height = ui.available_height();
        let spacing = 12.0;
        let min_preview = 220.0;
        let mut preview_height = (total_height * 0.85).max(min_preview);
        preview_height = preview_height.min(total_height.max(0.0));
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

        ui.add_space(spacing);
        ui.add_space(spacing);
        ui.horizontal(|ui| {
            ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                let enabled = self.preview.texture.is_some();
                let label = if self.manual_box_tool_enabled {
                    "Exit draw mode"
                } else {
                    "Draw manual box"
                };
                if ui.add_enabled(enabled, egui::Button::new(label)).clicked() {
                    self.manual_box_tool_enabled = !self.manual_box_tool_enabled;
                    if !self.manual_box_tool_enabled {
                        self.manual_box_draft = None;
                    }
                }
                if ui.button("Show Detections").clicked() {
                    self.show_detection_window = true;
                }
            });
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
                    let preview_bounds = ui.max_rect();
                    ui.centered_and_justified(|ui| {
                        let image_widget = egui::Image::new(&texture).fit_to_exact_size(scaled);
                        let response = ui.add(image_widget);
                        if let Some(dimensions) = image_dimensions {
                            let image_rect = Rect::from_center_size(response.rect.center(), scaled);
                            self.handle_preview_interactions(ctx, image_rect, dimensions);
                            self.paint_detections(ui, image_rect, dimensions);
                            self.preview_overlay(ui, preview_bounds, palette);
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

    fn preview_overlay(&mut self, ui: &mut Ui, boundary_rect: Rect, palette: theme::Palette) {
        let overlay_size = self.preview_hud_size();
        if boundary_rect.width() < overlay_size.x || boundary_rect.height() < overlay_size.y {
            return;
        }

        let mut overlay_rect = self.preview_hud_rect(boundary_rect, overlay_size);
        let overlay_id = ui.make_persistent_id("preview_hud_overlay");

        if let Some(origin) = self.preview_hud_drag_origin {
            let delta = ui.ctx().input(|i| i.pointer.delta());
            let desired = origin + delta;
            overlay_rect =
                self.update_hud_anchor_from_top_left(boundary_rect, overlay_size, desired);
            self.preview_hud_drag_origin = Some(overlay_rect.left_top());
        }

        if self.preview_hud_drag_origin.is_some() && !ui.ctx().input(|i| i.pointer.primary_down()) {
            self.preview_hud_drag_origin = None;
        }

        let painter = ui.painter();
        let drag_response = ui.interact(overlay_rect, overlay_id, Sense::drag());

        if drag_response.drag_started() {
            self.preview_hud_drag_origin = Some(overlay_rect.left_top());
        }

        let hovered = drag_response.hovered() || drag_response.dragged();
        let fill = if hovered {
            palette.panel_dark
        } else {
            translucent_color(palette.panel_dark, 0.7)
        };

        painter.rect_filled(overlay_rect, 16.0, fill);
        painter.rect_stroke(
            overlay_rect,
            16.0,
            Stroke::new(1.0, palette.outline),
            egui::StrokeKind::Outside,
        );

        let content_rect = overlay_rect.shrink2(vec2(14.0, 10.0));
        let mut child_ui = ui.new_child(
            UiBuilder::new()
                .max_rect(content_rect)
                .layout(Layout::top_down(Align::Min)),
        );
        child_ui.set_clip_rect(content_rect);
        {
            let overlay_ui = &mut child_ui;
            overlay_ui.horizontal(|ui| {
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
                overlay_ui.add_space(4.0);
                overlay_ui.label(format!(
                    "Faces: {}  |  Selected: {}",
                    self.preview.detections.len(),
                    self.selected_faces.len()
                ));
                overlay_ui.label(
                    RichText::new("Drag to reposition")
                        .color(palette.subtle_text)
                        .size(13.0),
                );
                return;
            }

            overlay_ui.add_space(4.0);
            overlay_ui.horizontal(|ui| {
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

            overlay_ui.add_space(6.0);
            self.quality_legend(overlay_ui, palette);
            overlay_ui.add_space(6.0);

            if self.preview.detections.is_empty() {
                overlay_ui.label(RichText::new("No faces detected yet.").weak());
            } else {
                overlay_ui.checkbox(&mut self.show_crop_overlay, "Show crop guides");
            }

            overlay_ui.add_space(6.0);
            let selected = self.selected_faces.len();
            if selected > 0 {
                let button_label = format!(
                    "Export {} face{}",
                    selected,
                    if selected == 1 { "" } else { "s" }
                );
                if overlay_ui.button(button_label).clicked() {
                    self.export_selected_faces();
                }
            } else {
                overlay_ui.label(
                    RichText::new("Select faces from the list to export.")
                        .color(palette.subtle_text),
                );
            }
        }
    }

    fn preview_hud_size(&self) -> egui::Vec2 {
        if self.preview_hud_minimized {
            vec2(230.0, 80.0)
        } else {
            vec2(260.0, 190.0)
        }
    }

    fn preview_hud_rect(&self, boundary_rect: Rect, overlay_size: egui::Vec2) -> Rect {
        let available_width = (boundary_rect.width() - overlay_size.x).max(0.0);
        let available_height = (boundary_rect.height() - overlay_size.y).max(0.0);
        let anchor_x = self.preview_hud_anchor.x.clamp(0.0, 1.0);
        let anchor_y = self.preview_hud_anchor.y.clamp(0.0, 1.0);
        let top_left = pos2(
            boundary_rect.left() + anchor_x * available_width,
            boundary_rect.top() + anchor_y * available_height,
        );
        Rect::from_min_size(top_left, overlay_size)
    }

    fn update_hud_anchor_from_top_left(
        &mut self,
        boundary_rect: Rect,
        overlay_size: egui::Vec2,
        desired_top_left: egui::Pos2,
    ) -> Rect {
        let clamped = self.clamp_hud_top_left(boundary_rect, overlay_size, desired_top_left);
        self.preview_hud_anchor = self.anchor_from_top_left(boundary_rect, overlay_size, clamped);
        Rect::from_min_size(clamped, overlay_size)
    }

    fn clamp_hud_top_left(
        &self,
        boundary_rect: Rect,
        overlay_size: egui::Vec2,
        desired: egui::Pos2,
    ) -> egui::Pos2 {
        let min_x = boundary_rect.left();
        let min_y = boundary_rect.top();
        let max_x = (boundary_rect.right() - overlay_size.x).max(min_x);
        let max_y = (boundary_rect.bottom() - overlay_size.y).max(min_y);
        pos2(desired.x.clamp(min_x, max_x), desired.y.clamp(min_y, max_y))
    }

    fn anchor_from_top_left(
        &self,
        boundary_rect: Rect,
        overlay_size: egui::Vec2,
        top_left: egui::Pos2,
    ) -> egui::Vec2 {
        let denom_x = (boundary_rect.width() - overlay_size.x).max(1.0);
        let denom_y = (boundary_rect.height() - overlay_size.y).max(1.0);
        vec2(
            ((top_left.x - boundary_rect.left()) / denom_x).clamp(0.0, 1.0),
            ((top_left.y - boundary_rect.top()) / denom_y).clamp(0.0, 1.0),
        )
    }
}

/// Helper function to create a translucent color.
fn translucent_color(color: Color32, alpha_multiplier: f32) -> Color32 {
    let rgba: Rgba = color.into();
    let new_alpha = (rgba.a() * alpha_multiplier).clamp(0.0, 1.0);
    Rgba::from_rgba_premultiplied(rgba.r(), rgba.g(), rgba.b(), new_alpha).into()
}
