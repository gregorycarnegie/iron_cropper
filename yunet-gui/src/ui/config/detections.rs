//! Face detections listing and controls.

use egui::{
    Button, Color32, Context as EguiContext, CornerRadius, Margin, RichText, ScrollArea, Stroke,
    Ui, vec2,
};

use crate::{YuNetApp, theme};
use yunet_utils::quality::Quality;

/// Shows the detected faces section with thumbnails and controls.
pub fn show_detections_section(
    app: &mut YuNetApp,
    ctx: &EguiContext,
    ui: &mut Ui,
    palette: theme::Palette,
) {
    ui.heading("Detected Faces");

    // Manual bounding box tool
    show_manual_box_controls(app, ui, palette);

    if app.preview.detections.is_empty() {
        if app.is_busy {
            ui.label("Waiting for results…");
        } else {
            ui.label("No faces detected yet.");
        }
    } else {
        ui.label(format!(
            "{} face(s) detected. Click to select for cropping.",
            app.preview.detections.len()
        ));
        ui.add_space(8.0);

        // Face list with thumbnails
        show_face_list(app, ctx, ui, palette);

        // Selection controls
        show_selection_controls(app, ui);

        // Export button
        show_export_button(app, ui);
    }
}

/// Shows the manual bounding box drawing controls.
fn show_manual_box_controls(app: &mut YuNetApp, ui: &mut Ui, palette: theme::Palette) {
    ui.horizontal(|ui| {
        let enabled = app.preview.texture.is_some();
        let label = if app.manual_box_tool_enabled {
            "Exit draw mode"
        } else {
            "Draw bounding box"
        };
        if ui.add_enabled(enabled, Button::new(label)).clicked() {
            app.manual_box_tool_enabled = !app.manual_box_tool_enabled;
            if !app.manual_box_tool_enabled {
                app.manual_box_draft = None;
            }
        }
        if app.manual_box_tool_enabled {
            ui.label(
                RichText::new("Click and drag in the preview area").color(palette.subtle_text),
            );
        } else if !enabled {
            ui.label(RichText::new("Load an image to draw boxes").color(palette.subtle_text));
        }
    });
}

/// Shows the scrollable list of detected faces with thumbnails.
fn show_face_list(app: &mut YuNetApp, ctx: &EguiContext, ui: &mut Ui, palette: theme::Palette) {
    let mut pending_removal: Option<usize> = None;

    ScrollArea::vertical()
        .id_salt("detected_faces_scroll")
        .max_height(220.0)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let len = app.preview.detections.len();
            for index in 0..len {
                if pending_removal.is_some() {
                    break;
                }

                let (detection_score, quality, quality_score, thumbnail, is_manual, is_modified) = {
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

                let quality_color = match quality {
                    Quality::High => Color32::from_rgb(0, 200, 100),
                    Quality::Medium => Color32::from_rgb(255, 180, 0),
                    Quality::Low => Color32::from_rgb(255, 80, 80),
                };

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

                let mut remove_requested = false;
                let response = egui::Frame::new()
                    .fill(frame_fill)
                    .stroke(frame_stroke)
                    .corner_radius(CornerRadius::same(12))
                    .inner_margin(Margin::symmetric(10, 8))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            // Thumbnail or crop preview
                            if let Some(texture) = app.crop_preview_texture_for(ctx, index) {
                                let preview = egui::Image::new((texture.id(), texture.size_vec2()))
                                    .max_size(vec2(100.0, 100.0));
                                let preview_response = ui.add(preview);
                                if preview_response.clicked() {
                                    toggle_selection(app, index);
                                }
                            } else if let Some(thumbnail) = thumbnail {
                                let thumb_response = ui.add(
                                    egui::Image::new((thumbnail.id(), thumbnail.size_vec2()))
                                        .max_width(80.0),
                                );
                                if thumb_response.clicked() {
                                    toggle_selection(app, index);
                                }
                            }

                            ui.vertical(|ui| {
                                // Title
                                let title = if is_manual {
                                    format!("Manual {}", index + 1)
                                } else {
                                    format!("Face {}", index + 1)
                                };
                                ui.label(RichText::new(title).strong());

                                // Status badges
                                if is_manual {
                                    ui.label(RichText::new("User box").color(palette.accent));
                                } else if is_modified {
                                    ui.label(RichText::new("Adjusted box").color(palette.accent));
                                }

                                // Metrics
                                ui.label(format!("Conf: {:.2}", detection_score));
                                ui.horizontal(|ui| {
                                    ui.label("Quality:");
                                    ui.colored_label(quality_color, format!("{:?}", quality));
                                });
                                ui.label(format!("Score: {:.0}", quality_score));

                                // Action buttons
                                ui.horizontal(|ui| {
                                    if ui
                                        .small_button(if is_selected {
                                            "Deselect"
                                        } else {
                                            "Select"
                                        })
                                        .clicked()
                                    {
                                        toggle_selection(app, index);
                                    }

                                    if ui.small_button("Reset box").clicked() {
                                        app.reset_detection_bbox(ctx, index);
                                    }

                                    if is_manual && ui.small_button("Remove").clicked() {
                                        remove_requested = true;
                                    }
                                });
                            });
                        });
                    });

                if response.response.clicked() {
                    toggle_selection(app, index);
                }

                if remove_requested {
                    pending_removal = Some(index);
                }
            }
        });

    if let Some(index) = pending_removal {
        app.remove_detection(index);
    }
}

/// Toggles the selection state of a face.
fn toggle_selection(app: &mut YuNetApp, index: usize) {
    if app.selected_faces.contains(&index) {
        app.selected_faces.remove(&index);
    } else {
        app.selected_faces.insert(index);
    }
}

/// Shows the selection control buttons (Select All / Deselect All).
fn show_selection_controls(app: &mut YuNetApp, ui: &mut Ui) {
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        if ui.button("Select All").clicked() {
            app.selected_faces = (0..app.preview.detections.len()).collect();
        }
        if ui.button("Deselect All").clicked() {
            app.selected_faces.clear();
        }
    });

    ui.add_space(8.0);
    ui.separator();
}

/// Shows the export button for selected faces.
fn show_export_button(app: &mut YuNetApp, ui: &mut Ui) {
    let num_selected = app.selected_faces.len();
    if num_selected > 0 {
        let button_label = format!(
            "Export {} Selected Face{}",
            num_selected,
            if num_selected == 1 { "" } else { "s" }
        );
        if ui.button(button_label).clicked() {
            app.export_selected_faces();
        }
    } else {
        ui.label(RichText::new("Select faces to enable export").weak());
    }
}
