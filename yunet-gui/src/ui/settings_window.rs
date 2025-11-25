use crate::YuNetApp;
use crate::theme;
use crate::ui::widgets;
use egui::{ComboBox, Context, DragValue, RichText, ScrollArea, Ui};

pub fn show_settings_window(app: &mut YuNetApp, ctx: &Context) {
    let mut open = app.show_settings_window;
    ctx.show_viewport_immediate(
        egui::ViewportId::from_hash_of("settings_viewport"),
        egui::ViewportBuilder::default()
            .with_title("Settings")
            .with_inner_size([400.0, 600.0]),
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
                ScrollArea::vertical().show(ui, |ui| {
                    let mut engine_changed = false;
                    let mut requires_detector_reset = false;
                    let mut requires_cache_refresh = false;

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
                });
            });
        },
    );
    app.show_settings_window = open;
}

fn show_model_settings(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    requires_detector_reset: &mut bool,
    requires_cache_refresh: &mut bool,
) {
    use egui::Key;
    use yunet_utils::config::ResizeQuality;

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
    if widgets::slider_row(
        ui,
        &mut score,
        0.0..=1.0,
        "Score threshold",
        0.01,
        None,
        Some(egui::Color32::RED),
    ) {
        app.settings.detection.score_threshold = score;
        *settings_changed = true;
        *requires_cache_refresh = true;
    }
    let mut nms = app.settings.detection.nms_threshold;
    if widgets::slider_row(
        ui,
        &mut nms,
        0.0..=1.0,
        "NMS threshold",
        0.01,
        None,
        Some(egui::Color32::RED),
    ) {
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
