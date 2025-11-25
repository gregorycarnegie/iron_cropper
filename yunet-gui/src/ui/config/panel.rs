//! Main configuration panel orchestration.

use egui::{Context as EguiContext, RichText, ScrollArea, Ui};

use crate::YuNetApp;

/// Renders the right-hand advanced configuration panel.
pub fn show_configuration_panel(app: &mut YuNetApp, ui: &mut Ui, ctx: &EguiContext) {
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

            ui.add_space(8.0);
            ui.small(
                RichText::new(format!("Settings file: {}", app.settings_path.display())).weak(),
            );

            let crop_changed = app.settings.crop != initial_crop_settings;
            let enhance_changed_now = app.settings.enhance.clone() != initial_enhance_settings;

            if crop_changed && !preview_invalidated {
                app.clear_crop_preview_cache();
            }
            if enhance_changed_now && !enhancement_changed {
                app.clear_crop_preview_cache();
            }
        });
}
