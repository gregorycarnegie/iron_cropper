//! Batch queue window.

use crate::{YuNetApp, theme};
use egui::Context;

/// Shows the batch queue window.
pub fn show_batch_window(app: &mut YuNetApp, ctx: &Context) {
    let mut open = app.show_batch_window;
    ctx.show_viewport_immediate(
        egui::ViewportId::from_hash_of("batch_viewport"),
        egui::ViewportBuilder::default()
            .with_title("Batch Queue")
            .with_inner_size([500.0, 400.0]),
        |ctx, class| {
            assert!(
                class == egui::ViewportClass::Immediate,
                "This egui backend doesn't support multiple viewports"
            );

            egui::CentralPanel::default().show(ctx, |ui| {
                if ctx.input(|i| i.viewport().close_requested()) {
                    open = false;
                }

                let palette = theme::palette();
                show_batch_content(app, ui, palette);
            });
        },
    );
    app.show_batch_window = open;
}

/// Shows the content of the batch processing section.
fn show_batch_content(app: &mut YuNetApp, ui: &mut egui::Ui, palette: theme::Palette) {
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
            .max_height(300.0) // Increased height for window
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
