//! Main configuration panel orchestration.

use egui::{
    Context as EguiContext, Frame, Margin, RichText, ScrollArea, SidePanel, Slider, Stroke, Ui,
};

use crate::YuNetApp;
use crate::theme;

/// Renders the left-hand configuration panel.
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
                    let mut engine_changed = false;
                    let mut enhancement_changed = false;
                    let mut preview_invalidated = false;
                    let mut metadata_tags_changed = false;
                    let mut requires_detector_reset = false;
                    let mut requires_cache_refresh = false;

                    ui.heading("Adjustments");
                    ui.label("Fine-tune the framing and automation for exports.");
                    show_primary_adjustments(
                        app,
                        ui,
                        &mut crop_settings_changed,
                        &mut preview_invalidated,
                        &mut enhancement_changed,
                    );

                    ui.separator();
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
                    ui.heading("Detection Engine");
                    show_model_settings(
                        app,
                        ui,
                        &mut engine_changed,
                        &mut requires_detector_reset,
                        &mut requires_cache_refresh,
                    );

                    ui.separator();
                    show_gpu_section(
                        app,
                        ui,
                        palette,
                        &mut engine_changed,
                        &mut requires_detector_reset,
                    );

                    ui.separator();
                    show_diagnostics_section(app, ui, &mut engine_changed);

                    if engine_changed {
                        app.apply_settings_changes(requires_detector_reset, requires_cache_refresh);
                        app.persist_settings_with_feedback();
                    }

                    ui.add_space(8.0);
                    ui.small(
                        RichText::new(format!("Settings file: {}", app.settings_path.display()))
                            .weak(),
                    );

                    let crop_changed = app.settings.crop != initial_crop_settings;
                    let enhance_changed_now = app.settings.enhance != initial_enhance_settings;

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

fn show_primary_adjustments(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    preview_invalidated: &mut bool,
    enhancement_changed: &mut bool,
) {
    let mut face_fill = app.settings.crop.face_height_pct;
    if ui
        .add(Slider::new(&mut face_fill, 20.0..=95.0).text("Face fill (%)"))
        .changed()
    {
        app.settings.crop.face_height_pct = face_fill;
        *settings_changed = true;
        *preview_invalidated = true;
    }

    let mut horizontal = app.settings.crop.horizontal_offset;
    if ui
        .add(Slider::new(&mut horizontal, -1.0..=1.0).text("Eye alignment"))
        .changed()
    {
        app.settings.crop.horizontal_offset = horizontal.clamp(-1.0, 1.0);
        *settings_changed = true;
        *preview_invalidated = true;
    }

    let mut vertical = app.settings.crop.vertical_offset;
    if ui
        .add(Slider::new(&mut vertical, -1.0..=1.0).text("Vertical lift"))
        .changed()
    {
        app.settings.crop.vertical_offset = vertical.clamp(-1.0, 1.0);
        *settings_changed = true;
        *preview_invalidated = true;
    }

    ui.add_space(6.0);
    ui.label("Automation");
    if ui
        .checkbox(
            &mut app.settings.crop.quality_rules.auto_select_best_face,
            "Auto-select best face",
        )
        .changed()
    {
        *settings_changed = true;
    }
    if ui
        .checkbox(
            &mut app.settings.crop.quality_rules.auto_skip_no_high_quality,
            "Skip export without a high-quality face",
        )
        .changed()
    {
        *settings_changed = true;
    }

    ui.add_space(6.0);
    ui.label("Enhancements");
    if ui
        .checkbox(&mut app.settings.enhance.enabled, "Enable enhancements")
        .changed()
    {
        *settings_changed = true;
        *enhancement_changed = true;
    }
    ui.add_enabled_ui(app.settings.enhance.enabled, |ui| {
        let mut background_blur = app.settings.enhance.background_blur;
        if ui
            .checkbox(&mut background_blur, "Background blur")
            .changed()
        {
            app.settings.enhance.background_blur = background_blur;
            *settings_changed = true;
            *enhancement_changed = true;
        }
        let mut red_eye = app.settings.enhance.red_eye_removal;
        if ui.checkbox(&mut red_eye, "Red-eye removal").changed() {
            app.settings.enhance.red_eye_removal = red_eye;
            *settings_changed = true;
            *enhancement_changed = true;
        }
    });
}

fn show_gpu_section(
    app: &mut YuNetApp,
    ui: &mut Ui,
    palette: theme::Palette,
    settings_changed: &mut bool,
    requires_detector_reset: &mut bool,
) {
    use crate::GpuStatusMode;

    let mut enabled = app.settings.gpu.enabled;
    if ui
        .checkbox(&mut enabled, "Enable GPU preprocessing")
        .changed()
    {
        app.settings.gpu.enabled = enabled;
        if !enabled {
            app.settings.gpu.inference = false;
        }
        *settings_changed = true;
        *requires_detector_reset = true;
    }

    let mut respect_env = app.settings.gpu.respect_env;
    ui.add_enabled_ui(enabled, |ui| {
        if ui
            .checkbox(
                &mut respect_env,
                "Respect WGPU_* environment overrides (advanced)",
            )
            .changed()
        {
            app.settings.gpu.respect_env = respect_env;
            *settings_changed = true;
            *requires_detector_reset = true;
        }
    });
    ui.small("Disable overrides to force the default backend if diagnostics require it.");

    ui.add_enabled_ui(enabled, |ui| {
        let mut gpu_inference = app.settings.gpu.inference;
        if ui
            .checkbox(&mut gpu_inference, "Enable GPU inference (experimental)")
            .changed()
        {
            app.settings.gpu.inference = gpu_inference;
            *settings_changed = true;
            *requires_detector_reset = true;
        }
        ui.label(
            RichText::new("Runs the YuNet ONNX graph on the GPU; automatically falls back to CPU if initialization fails.")
                .color(palette.subtle_text),
        );
    });

    let status = &app.gpu_status;
    let headline_color = match status.mode {
        GpuStatusMode::Available => palette.success,
        GpuStatusMode::Pending => palette.subtle_text,
        GpuStatusMode::Disabled => palette.subtle_text,
        GpuStatusMode::Fallback => palette.warning,
        GpuStatusMode::Error => palette.danger,
    };

    ui.add_space(6.0);
    ui.colored_label(
        headline_color,
        RichText::new(status.summary.clone()).strong(),
    );
    if let Some(detail) = &status.detail {
        ui.label(RichText::new(detail).color(palette.subtle_text));
    }

    ui.add_space(4.0);
    if let Some(adapter) = &status.adapter_name {
        ui.label(format!("Adapter: {}", adapter));
    }
    if let Some(backend) = &status.backend {
        ui.label(format!("Backend: {backend}"));
    }
    if let Some(driver) = &status.driver {
        ui.label(format!("Driver: {driver}"));
    }
    if let (Some(vendor), Some(device)) = (status.vendor_id, status.device_id) {
        ui.label(format!("PCI IDs: {vendor:#06x}:{device:#06x}"));
    }

    if matches!(status.mode, GpuStatusMode::Fallback | GpuStatusMode::Error) {
        ui.label(
            RichText::new(
                "The application fell back to CPU preprocessing. Detection still works, \
                 but throughput matches the CPU baseline.",
            )
            .italics()
            .color(palette.warning),
        );
    }
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

/// Shows the model settings section.
fn show_model_settings(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    requires_detector_reset: &mut bool,
    requires_cache_refresh: &mut bool,
) {
    use egui::{ComboBox, DragValue, Key, RichText, Slider};
    use yunet_utils::config::ResizeQuality;

    ui.separator();
    ui.heading("Model Settings");
    ui.label("Model path");
    ui.horizontal(|ui| {
        let response = ui.text_edit_singleline(&mut app.model_path_input);
        if response.changed() {
            app.model_path_dirty = true;
        }
        let enter_pressed = ui.input(|i| i.key_pressed(Key::Enter));
        if app.model_path_dirty && (response.lost_focus() || enter_pressed) {
            app.apply_model_path_input();
        }
        if ui.button("Browse…").clicked() {
            app.open_model_dialog();
        }
    });
    if app.model_path_dirty {
        ui.label(RichText::new("Press Enter to apply model path changes.").weak());
    }

    ui.add_space(6.0);
    ui.label("Input size (W×H)");
    ui.horizontal(|ui| {
        ui.label("Width");
        let mut width = app.settings.input.width;
        if ui
            .add(DragValue::new(&mut width).range(64..=4096).speed(16.0))
            .changed()
        {
            app.settings.input.width = width;
            *settings_changed = true;
            *requires_detector_reset = true;
        }
    });
    ui.horizontal(|ui| {
        ui.label("Height");
        let mut height = app.settings.input.height;
        if ui
            .add(DragValue::new(&mut height).range(64..=4096).speed(16.0))
            .changed()
        {
            app.settings.input.height = height;
            *settings_changed = true;
            *requires_detector_reset = true;
        }
    });

    let describe_quality = |mode: ResizeQuality| -> &'static str {
        match mode {
            ResizeQuality::Quality => "Quality (Triangle)",
            ResizeQuality::Speed => "Speed (Nearest)",
        }
    };

    ui.horizontal(|ui| {
        ui.label("Resize mode");
        let mut selected = app.settings.input.resize_quality;
        ComboBox::from_id_salt("resize_quality_combo")
            .selected_text(describe_quality(selected))
            .show_ui(ui, |ui| {
                for option in [ResizeQuality::Quality, ResizeQuality::Speed] {
                    let label = describe_quality(option);
                    if ui.selectable_label(selected == option, label).clicked() {
                        selected = option;
                    }
                }
            });
        if selected != app.settings.input.resize_quality {
            app.settings.input.resize_quality = selected;
            *settings_changed = true;
            *requires_detector_reset = true;
            *requires_cache_refresh = true;
        }
    });
    ui.small("Speed mode prioritizes throughput; quality preserves smoother resampling.");

    ui.add_space(6.0);
    ui.label("Detection thresholds");
    let mut score = app.settings.detection.score_threshold;
    if ui
        .add(Slider::new(&mut score, 0.0..=1.0).text("Score threshold"))
        .changed()
    {
        app.settings.detection.score_threshold = score;
        *settings_changed = true;
        *requires_cache_refresh = true;
    }
    let mut nms = app.settings.detection.nms_threshold;
    if ui
        .add(Slider::new(&mut nms, 0.0..=1.0).text("NMS threshold"))
        .changed()
    {
        app.settings.detection.nms_threshold = nms;
        *settings_changed = true;
        *requires_cache_refresh = true;
    }
    let mut top_k = app.settings.detection.top_k as i64;
    if ui
        .add(
            DragValue::new(&mut top_k)
                .range(1..=20_000)
                .speed(100.0)
                .suffix(" detections"),
        )
        .changed()
    {
        app.settings.detection.top_k = top_k.max(1) as usize;
        *settings_changed = true;
        *requires_cache_refresh = true;
    }
}

/// Shows the diagnostics section.
fn show_diagnostics_section(app: &mut YuNetApp, ui: &mut Ui, settings_changed: &mut bool) {
    use log::LevelFilter;
    use log::info;
    use yunet_utils::configure_telemetry;

    let mut telemetry_changed = false;
    if ui
        .checkbox(
            &mut app.settings.telemetry.enabled,
            "Enable telemetry logging",
        )
        .changed()
    {
        *settings_changed = true;
        telemetry_changed = true;
    }

    let level_options = [
        (LevelFilter::Error, "Error"),
        (LevelFilter::Warn, "Warn"),
        (LevelFilter::Info, "Info"),
        (LevelFilter::Debug, "Debug"),
        (LevelFilter::Trace, "Trace"),
    ];

    ui.add_enabled_ui(app.settings.telemetry.enabled, |ui| {
        let current_level = app.settings.telemetry.level_filter();
        let current_label = level_options
            .iter()
            .find(|(level, _)| *level == current_level)
            .map(|(_, label)| *label)
            .unwrap_or("Debug");
        let mut selected_level = current_level;
        egui::ComboBox::from_id_salt("telemetry_level_combo")
            .selected_text(current_label)
            .show_ui(ui, |ui| {
                for (level, label) in level_options.iter() {
                    if ui
                        .selectable_label(selected_level == *level, *label)
                        .clicked()
                    {
                        selected_level = *level;
                    }
                }
            });
        if selected_level != current_level {
            app.settings.telemetry.set_level(selected_level);
            *settings_changed = true;
            telemetry_changed = true;
        }
        ui.add(egui::Label::new("Timings are written to the application log.").wrap());
    });

    if telemetry_changed {
        configure_telemetry(
            app.settings.telemetry.enabled,
            app.settings.telemetry.level_filter(),
        );
        if app.settings.telemetry.enabled {
            info!(
                "Telemetry logging enabled (level={:?})",
                app.settings.telemetry.level_filter()
            );
        } else {
            info!("Telemetry logging disabled");
        }
    }
}
