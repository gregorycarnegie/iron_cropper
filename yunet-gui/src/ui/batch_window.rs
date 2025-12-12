//! Batch queue window.

use crate::BatchFileStatus;
use crate::{YuNetApp, theme};
use yunet_utils::config::BatchLogFormat;

use egui::{
    Button, CentralPanel, Color32, Context, ProgressBar, RichText, ScrollArea, TopBottomPanel,
    ViewportBuilder, ViewportClass, ViewportId,
};

/// Shows the batch queue window.
pub fn show_batch_window(app: &mut YuNetApp, ctx: &Context) {
    let mut open = app.show_batch_window;
    ctx.show_viewport_immediate(
        ViewportId::from_hash_of("batch_viewport"),
        ViewportBuilder::default()
            .with_title("Batch Queue")
            .with_inner_size([500.0, 500.0]), // Slightly taller default
        |ctx, class| {
            assert!(
                class == ViewportClass::Immediate,
                "This egui backend doesn't support multiple viewports"
            );

            if ctx.input(|i| i.viewport().close_requested()) {
                open = false;
            }

            let palette = theme::palette();

            if app.batch_files.is_empty() {
                CentralPanel::default().show(ctx, |ui| {
                    ui.centered_and_justified(|ui| {
                        ui.label(RichText::new("Batch queue is empty").color(palette.subtle_text));
                    });
                });
                return;
            }

            // Bottom Panel: Actions & Logging
            TopBottomPanel::bottom("batch_footer")
                .resizable(false)
                .frame(
                    egui::Frame::NONE
                        .fill(palette.panel_light)
                        .inner_margin(8.0),
                )
                .show(ctx, |ui| {
                    show_batch_footer(app, ui);
                });

            // Top Panel: Progress
            TopBottomPanel::top("batch_header")
                .resizable(false)
                .frame(
                    egui::Frame::NONE
                        .fill(palette.panel_light)
                        .inner_margin(8.0),
                )
                .show(ctx, |ui| {
                    show_batch_progress(app, ui);
                });

            // Central Panel: List
            CentralPanel::default().show(ctx, |ui| {
                show_batch_list(app, ui, palette);
            });
        },
    );
    app.show_batch_window = open;
}

fn show_batch_progress(app: &YuNetApp, ui: &mut egui::Ui) {
    let total = app.batch_files.len();
    if total == 0 {
        return;
    }

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
    let progress_ratio = processed as f32 / total as f32;

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
}

fn show_batch_list(app: &YuNetApp, ui: &mut egui::Ui, palette: theme::Palette) {
    ScrollArea::vertical()
        .id_salt("batch_files_scroll")
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
}

fn show_batch_footer(app: &mut YuNetApp, ui: &mut egui::Ui) {
    let icon_size = app.icons.default_size();

    ui.horizontal(|ui| {
        if ui
            .add(Button::image_and_text(
                app.icons.export(icon_size),
                "Export All Batch Files",
            ))
            .clicked()
        {
            app.start_batch_export();
        }
        if ui
            .add(Button::image_and_text(
                app.icons.folder_open(icon_size),
                "Clear Batch",
            ))
            .clicked()
        {
            app.batch_files.clear();
            app.batch_current_index = None;
        }
    });

    ui.separator();
    ui.label("Failure Logging");
    ui.horizontal(|ui| {
        ui.checkbox(
            &mut app.settings.batch_logging.enabled,
            "Log failures to file",
        );
    });

    if app.settings.batch_logging.enabled {
        ui.horizontal(|ui| {
            ui.label("Format:");
            egui::ComboBox::from_id_salt("log_format_combo")
                .selected_text(match app.settings.batch_logging.format {
                    BatchLogFormat::Json => "JSON",
                    BatchLogFormat::Csv => "CSV",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut app.settings.batch_logging.format,
                        BatchLogFormat::Json,
                        "JSON",
                    );
                    ui.selectable_value(
                        &mut app.settings.batch_logging.format,
                        BatchLogFormat::Csv,
                        "CSV",
                    );
                });
        });
    }
}
