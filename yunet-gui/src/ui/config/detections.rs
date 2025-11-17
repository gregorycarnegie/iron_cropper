//! Face detections carousel used by the preview area.

use egui::{
    Align, Color32, Context as EguiContext, CornerRadius, Frame, Layout, Margin, ProgressBar,
    RichText, ScrollArea, Stroke, StrokeKind, Ui, Vec2,
};

use crate::{YuNetApp, theme};
use yunet_utils::quality::Quality;

/// Shows a horizontal filmstrip of detections plus selection controls.
pub fn show_detection_carousel(
    app: &mut YuNetApp,
    ctx: &EguiContext,
    ui: &mut Ui,
    palette: theme::Palette,
) {
    let max_height = ui.available_height();
    ui.scope(|ui| {
        ui.set_max_height(max_height);
        Frame::new()
            .fill(palette.panel_dark)
            .stroke(Stroke::new(1.0, palette.outline))
            .corner_radius(CornerRadius::same(20))
            .inner_margin(Margin::symmetric(14, 12))
            .show(ui, |ui| {
                ScrollArea::vertical()
                    .id_salt("detections_panel_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.vertical(|ui| {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("Detected faces").strong());
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    let enabled = app.preview.texture.is_some();
                                    let label = if app.manual_box_tool_enabled {
                                        "Exit draw mode"
                                    } else {
                                        "Draw manual box"
                                    };
                                    if ui.add_enabled(enabled, egui::Button::new(label)).clicked() {
                                        app.manual_box_tool_enabled = !app.manual_box_tool_enabled;
                                        if !app.manual_box_tool_enabled {
                                            app.manual_box_draft = None;
                                        }
                                    }
                                });
                            });

                            if app.preview.detections.is_empty() {
                                ui.add_space(24.0);
                                if app.is_busy {
                                    ui.label("Running detection…");
                                } else {
                                    ui.label("No faces detected yet.");
                                }
                                return;
                            }

                            ui.add_space(6.0);
                            ScrollArea::horizontal()
                                .id_salt("detections_carousel")
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        render_detections_strip(app, ctx, ui, palette);
                                    });
                                });

                            ui.add_space(8.0);
                            ui.horizontal(|ui| {
                                if ui.button("Select all").clicked() {
                                    app.selected_faces =
                                        (0..app.preview.detections.len()).collect();
                                }
                                if ui.button("Deselect").clicked() {
                                    app.selected_faces.clear();
                                }
                                if ui.button("Refresh thumbnails").clicked() {
                                    for idx in 0..app.preview.detections.len() {
                                        if app.preview.detections[idx].thumbnail.is_some() {
                                            app.refresh_detection_thumbnail_at(ctx, idx);
                                        }
                                    }
                                }
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    ui.checkbox(&mut app.show_crop_overlay, "Show crop guides");
                                });
                            });

                            let total = app.preview.detections.len() as f32;
                            let selected = app.selected_faces.len() as f32;
                            let ratio = if total.abs() < f32::EPSILON {
                                0.0
                            } else {
                                selected / total
                            };
                            ui.add(
                                ProgressBar::new(ratio)
                                    .desired_width(ui.available_width())
                                    .text(format!(
                                        "{} of {} faces selected",
                                        selected as usize, total as usize
                                    )),
                            );
                        });
                    });
            });
    });
}

fn render_detections_strip(
    app: &mut YuNetApp,
    ctx: &EguiContext,
    ui: &mut Ui,
    palette: theme::Palette,
) {
    let mut pending_removal: Option<usize> = None;
    let len = app.preview.detections.len();
    let available_height = ui.available_height().max(120.0);
    let preview_height = available_height.clamp(100.0, 200.0);
    let preview_width = (preview_height * 0.75).clamp(90.0, 220.0);
    let card_width = preview_width + 60.0;

    for index in 0..len {
        if pending_removal.is_some() {
            break;
        }

        let (score, quality, quality_score, thumbnail, is_manual, is_modified) = {
            let det = &app.preview.detections[index];
            (
                det.detection.score,
                det.quality,
                det.quality_score,
                det.thumbnail.clone(),
                det.is_manual(),
                det.is_modified(),
            )
        };
        let is_selected = app.selected_faces.contains(&index);

        let frame_fill = if is_selected {
            palette.panel_light
        } else {
            palette.panel_dark
        };
        let frame_stroke = if is_selected {
            Stroke::new(2.0, palette.accent)
        } else {
            Stroke::new(1.0, palette.outline)
        };

        let response = Frame::new()
            .fill(frame_fill)
            .stroke(frame_stroke)
            .corner_radius(CornerRadius::same(14))
            .inner_margin(Margin::symmetric(12, 10))
            .show(ui, |ui| {
                ui.set_width(card_width);
                ui.vertical(|ui| {
                    let mut clicked = false;
                    let preview_size = Vec2::new(preview_width, preview_height);
                    if let Some(texture) = app.crop_preview_texture_for(ctx, index) {
                        let preview = egui::Image::new((texture.id(), texture.size_vec2()))
                            .corner_radius(CornerRadius::same(12))
                            .fit_to_exact_size(preview_size);
                        if ui.add(preview).clicked() {
                            clicked = true;
                        }
                    } else if let Some(thumbnail) = thumbnail {
                        let preview = egui::Image::new((thumbnail.id(), thumbnail.size_vec2()))
                            .fit_to_exact_size(preview_size)
                            .corner_radius(CornerRadius::same(12));
                        if ui.add(preview).clicked() {
                            clicked = true;
                        }
                    } else {
                        let (rect, resp) =
                            ui.allocate_exact_size(preview_size, egui::Sense::click());
                        if resp.clicked() {
                            clicked = true;
                        }
                        ui.painter().rect(
                            rect,
                            CornerRadius::same(12),
                            palette.panel,
                            Stroke::new(1.0, palette.outline),
                            StrokeKind::Outside,
                        );
                        ui.painter().text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            "Loading…",
                            egui::FontId::new(13.0, egui::FontFamily::Proportional),
                            palette.subtle_text,
                        );
                    }
                    if clicked {
                        toggle_selection(app, index);
                    }

                    let title = if is_manual {
                        format!("Manual {}", index + 1)
                    } else {
                        format!("Face {}", index + 1)
                    };
                    ui.label(RichText::new(title).strong());

                    if is_manual {
                        ui.label(RichText::new("User box").color(palette.accent));
                    } else if is_modified {
                        ui.label(RichText::new("Adjusted box").color(palette.accent));
                    }

                    let quality_color = match quality {
                        Quality::High => Color32::from_rgb(0, 200, 100),
                        Quality::Medium => Color32::from_rgb(255, 180, 0),
                        Quality::Low => Color32::from_rgb(255, 80, 80),
                    };
                    ui.label(format!("Confidence {:.2}", score));
                    ui.colored_label(quality_color, format!("Quality {:?}", quality));
                    ui.label(format!("Score {:.0}", quality_score));

                    ui.horizontal(|ui| {
                        if ui
                            .button(if is_selected { "Deselect" } else { "Select" })
                            .clicked()
                        {
                            toggle_selection(app, index);
                        }
                        if ui.button("Reset").clicked() {
                            app.reset_detection_bbox(ctx, index);
                        }
                        if is_manual && ui.button("Remove").clicked() {
                            pending_removal = Some(index);
                        }
                    });
                });
            })
            .response;

        if response.clicked() {
            toggle_selection(app, index);
        }
    }

    if let Some(index) = pending_removal {
        app.remove_detection(index);
    }
}

fn toggle_selection(app: &mut YuNetApp, index: usize) {
    if app.selected_faces.contains(&index) {
        app.selected_faces.remove(&index);
    } else {
        app.selected_faces.insert(index);
    }
}
