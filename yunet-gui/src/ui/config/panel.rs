//! Main configuration panel orchestration.

use egui::{Context as EguiContext, Frame, Margin, RichText, ScrollArea, SidePanel, Stroke, Ui};

use crate::YuNetApp;
use crate::theme;

/// Renders the right-hand advanced configuration panel.
pub fn show_configuration_panel(app: &mut YuNetApp, ctx: &EguiContext) {
    let palette = theme::palette();
    SidePanel::right("yunet_adjustments_panel")
        .resizable(false)
        .exact_width(360.0)
        .frame(
            Frame::new()
                .fill(palette.panel)
                .stroke(Stroke::new(1.0, palette.outline))
                .inner_margin(Margin::symmetric(16, 18)),
        )
        .show(ctx, |ui| {
            ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let initial_crop_settings = app.settings.crop.clone();
                    let initial_enhance_settings = app.settings.enhance.clone();
                    let mut crop_settings_changed = false;
                    let mut enhancement_settings_changed = false;
                    let mut enhancement_changed = false;
                    let mut preview_invalidated = false;
                    let mut metadata_tags_changed = false;

                    ui.heading("Advanced Settings");
                    ui.add_space(8.0);

                    super::crop::show_crop_section(
                        app,
                        ui,
                        &mut crop_settings_changed,
                        &mut preview_invalidated,
                        &mut metadata_tags_changed,
                    );

                    if crop_settings_changed {
                        app.push_crop_history();
                        app.persist_settings_with_feedback();
                        app.apply_quality_rules_to_preview();
                        if !metadata_tags_changed {
                            app.refresh_metadata_tags_input();
                        }
                    }

                    ui.separator();
                    super::enhancement::show_enhancement_section(
                        app,
                        ui,
                        &mut enhancement_settings_changed,
                        &mut enhancement_changed,
                    );

                    if enhancement_changed {
                        app.clear_crop_preview_cache();
                        if !app.preview.detections.is_empty() {
                            for idx in 0..app.preview.detections.len() {
                                let _ = app.crop_preview_texture_for(ctx, idx);
                            }
                        }
                        ctx.request_repaint();
                    }
                    if enhancement_settings_changed {
                        app.persist_settings_with_feedback();
                    }

                    ui.separator();
                    ui.heading("Batch Queue");
                    show_batch_section(app, ui, palette);

                    ui.separator();
                    if ui.button("⚙ Settings").clicked() {
                        app.show_settings_window = true;
                    }

                    ui.add_space(8.0);
                    ui.small(
                        RichText::new(format!("Settings file: {}", app.settings_path.display()))
                            .weak(),
                    );

                    let crop_changed = app.settings.crop != initial_crop_settings;
                    let enhance_changed_now =
                        app.settings.enhance.clone() != initial_enhance_settings;

                    if crop_changed && !preview_invalidated {
                        app.clear_crop_preview_cache();
                    }
                    if enhance_changed_now && !enhancement_changed {
                        app.clear_crop_preview_cache();
                    }

                    ui.separator();
                    ui.heading("Mapping Import");
                    app.show_mapping_panel(ui, palette);
                });
        });
}

/// Shows the batch processing section.
fn show_batch_section(app: &mut YuNetApp, ui: &mut Ui, palette: theme::Palette) {
    use crate::BatchFileStatus;
    use egui::{Color32, ProgressBar, RichText, ScrollArea};

    if !app.batch_files.is_empty() {
        let total = app.batch_files.len();
        let completed = app
            .batch_files
            .iter()
            .filter(|f| matches!(f.status, BatchFileStatus::Completed { .. }))
            .count();
        let failed = app
            .batch_files
            .iter()
            .filter(|f| matches!(f.status, BatchFileStatus::Failed { .. }))
            .count();

        let processed = completed + failed;
        let progress_ratio = if total == 0 {
            0.0
        } else {
            processed as f32 / total as f32
        };
        ui.add(
            ProgressBar::new(progress_ratio)
                .desired_width(ui.available_width())
                .text(format!("{processed}/{total} files")),
        );
        if failed > 0 {
            ui.label(
                RichText::new(format!("({} failed)", failed)).color(Color32::from_rgb(255, 80, 80)),
            );
        }

        ui.add_space(6.0);
        ScrollArea::vertical()
            .id_salt("batch_files_scroll")
            .max_height(200.0)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (idx, batch_file) in app.batch_files.iter().enumerate() {
                    let filename = batch_file
                        .path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown");

                    let (status_text, status_color, tooltip) = match &batch_file.status {
                        BatchFileStatus::Pending => ("Pending".to_string(), Color32::GRAY, None),
                        BatchFileStatus::Processing => (
                            "Processing...".to_string(),
                            Color32::from_rgb(100, 150, 255),
                            None,
                        ),
                        BatchFileStatus::Completed {
                            faces_detected,
                            faces_exported,
                        } => (
                            format!("{} faces, {} exported", faces_detected, faces_exported),
                            Color32::from_rgb(0, 200, 100),
                            None,
                        ),
                        BatchFileStatus::Failed { error } => (
                            "Failed".to_string(),
                            Color32::from_rgb(255, 80, 80),
                            Some(error.clone()),
                        ),
                    };

                    ui.horizontal(|ui| {
                        ui.label(format!("{}.", idx + 1));
                        ui.label(filename);
                        if batch_file.output_override.is_some() {
                            ui.label(RichText::new("Mapping").color(palette.accent));
                        }
                        let status_resp = ui.colored_label(status_color, status_text);
                        if let Some(tip) = tooltip.as_deref() {
                            status_resp.on_hover_text(tip);
                        }
                    });
                }
            });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button("Export All Batch Files").clicked() {
                app.start_batch_export();
            }
            if ui.button("Clear Batch").clicked() {
                app.batch_files.clear();
                app.batch_current_index = None;
            }
        });
    } else {
        ui.label("No batch files loaded.");
    }
}
