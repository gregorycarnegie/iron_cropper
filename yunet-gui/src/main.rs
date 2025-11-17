//! Desktop GUI for YuNet face detection.

mod app;
mod app_impl;
mod core;
mod interaction;
mod rendering;
mod theme;
mod types;
mod ui;

use std::{io::Cursor, sync::Arc};

use eframe::{App, CreationContext, Frame, NativeOptions, egui};
use ico::IconDir;
use log::{info, warn};

// Re-export types for use by submodules
pub use types::{
    ActiveBoxDrag, BatchFile, BatchFileStatus, BatchJobConfig, CacheKey, CropPreviewCacheEntry,
    CropPreviewKey, DetectionCacheEntry, DetectionJobSuccess, DetectionOrigin,
    DetectionWithQuality, DragHandle, EnhancementSignature, JobMessage, ManualBoxDraft,
    MappingUiState, PointerSnapshot, PreviewSpace, PreviewState, ShapeSignature, YuNetApp,
};
pub use yunet_utils::gpu::{GpuStatusIndicator, GpuStatusMode};

use yunet_utils::{
    WgpuEnhancer,
    config::default_settings_path,
    gpu::{GpuBatchCropper, GpuContext},
    init_logging,
    quality::Quality,
};

type GpuPipelineInit = (
    Option<Arc<GpuContext>>,
    Option<Arc<WgpuEnhancer>>,
    Option<Arc<GpuBatchCropper>>,
);

/// Main entry point for the GUI application.
fn main() -> eframe::Result<()> {
    init_logging(log::LevelFilter::Info).expect("failed to initialize logging");
    let mut options = NativeOptions::default();

    // Set initial window size to avoid scrunched UI on first launch
    options.viewport = options.viewport.with_inner_size([1280.0, 800.0]);

    if let Some(icon) = load_app_icon() {
        options.viewport = options.viewport.with_icon(Arc::new(icon));
    }

    eframe::run_native(
        "YuNet Desktop",
        options,
        Box::new(|cc| Ok(Box::new(YuNetApp::new(cc)))),
    )
}

/// Load the embedded ICO file and convert it into an `eframe` icon, if possible.
fn load_app_icon() -> Option<egui::IconData> {
    const ICON_BYTES: &[u8] = include_bytes!("../assets/app_icon.ico");

    let dir = match IconDir::read(Cursor::new(ICON_BYTES)) {
        Ok(dir) => dir,
        Err(err) => {
            warn!("Failed to read embedded app icon: {err}");
            return None;
        }
    };

    let mut best: Option<(ico::IconImage, u32)> = None;
    for entry in dir.entries() {
        match entry.decode() {
            Ok(image) => {
                let score = image.width().saturating_mul(image.height());
                if best
                    .as_ref()
                    .is_none_or(|(_, best_score)| score > *best_score)
                {
                    best = Some((image, score));
                }
            }
            Err(err) => warn!("Failed to decode icon entry: {err}"),
        }
    }

    if let Some((image, _)) = best {
        Some(egui::IconData {
            rgba: image.rgba_data().to_vec(),
            width: image.width(),
            height: image.height(),
        })
    } else {
        warn!("Embedded icon did not yield any usable RGBA data");
        None
    }
}

impl YuNetApp {
    /// Creates a new `YuNetApp` instance.
    fn new(cc: &CreationContext<'_>) -> Self {
        let settings_path = default_settings_path();

        // Try to extract the WGPU render state from eframe
        let shared_gpu_context = cc.wgpu_render_state.as_ref().map(|render_state| {
            info!("Extracting GPU context from eframe WGPU renderer");
            let device = render_state.device.clone();
            let queue = render_state.queue.clone();
            let info = render_state.adapter.get_info();

            Arc::new(GpuContext::from_existing(
                None, // eframe doesn't expose the instance
                None, // eframe doesn't expose the adapter directly
                device, queue, info,
            ))
        });

        if shared_gpu_context.is_some() {
            info!("Successfully created shared GPU context from eframe renderer");
        } else {
            info!(
                "No WGPU render state available from eframe, will create separate GPU context if needed"
            );
        }

        Self::create(&cc.egui_ctx, settings_path, shared_gpu_context)
    }

    /// Creates a new `YuNetApp` instance with a specific settings path.
    pub(crate) fn create(
        ctx: &egui::Context,
        settings_path: std::path::PathBuf,
        shared_gpu_context: Option<Arc<GpuContext>>,
    ) -> Self {
        use core::{detection::build_detector, settings::load_settings};
        use std::sync::mpsc;
        use types::MappingUiState;
        use yunet_utils::configure_telemetry;

        theme::apply(ctx);

        info!("Loading GUI settings from {}", settings_path.display());
        let settings = load_settings(&settings_path);
        configure_telemetry(
            settings.telemetry.enabled,
            settings.telemetry.level_filter(),
        );
        if settings.telemetry.enabled {
            info!(
                "Telemetry logging enabled (level={:?})",
                settings.telemetry.level_filter()
            );
        }
        let (job_tx, job_rx) = mpsc::channel();

        let (initial_gpu_status, initial_gpu_context, detector_result) =
            build_detector(&settings, shared_gpu_context);
        let (gpu_context, gpu_enhancer, gpu_batch_cropper) =
            YuNetApp::init_gpu_pipelines(initial_gpu_context);
        let detector = match detector_result {
            Ok(detector) => {
                info!("Loaded YuNet model from configuration");
                Some(Arc::new(detector))
            }
            Err(err) => {
                warn!("Unable to initialize YuNet model: {err}");
                None
            }
        };

        let status_line = if detector.is_some() {
            "Model ready. Select an image to run detection.".to_owned()
        } else {
            "Model not loaded. Configure the model path before running detection.".to_owned()
        };

        let model_path_input = settings.model_path.clone().unwrap_or_default();

        let crop_history = vec![settings.crop.clone()];
        let crop_history_index = crop_history.len() - 1;
        let metadata_tags_input =
            YuNetApp::format_metadata_tags(&settings.crop.metadata.custom_tags);
        let crop_fill_hex_input = YuNetApp::format_fill_color_hex(settings.crop.fill_color);

        Self {
            settings,
            settings_path,
            status_line,
            last_error: None,
            gpu_status: initial_gpu_status,
            gpu_context,
            gpu_enhancer,
            gpu_batch_cropper,
            detector,
            job_tx,
            job_rx,
            preview: Default::default(),
            cache: Default::default(),
            crop_preview_cache: Default::default(),
            model_path_input,
            model_path_dirty: false,
            is_busy: false,
            texture_seq: 0,
            job_counter: 0,
            current_job: None,
            show_crop_overlay: true,
            selected_faces: Default::default(),
            crop_history,
            crop_history_index,
            crop_fill_hex_input,
            metadata_tags_input,
            batch_files: Vec::new(),
            batch_current_index: None,
            mapping: MappingUiState::new(),
            preview_hud_anchor: egui::vec2(0.02, 0.02),
            preview_hud_minimized: true,
            preview_hud_drag_origin: None,
            manual_box_tool_enabled: false,
            manual_box_draft: None,
            active_bbox_drag: None,
        }
    }

    fn init_gpu_pipelines(context: Option<Arc<GpuContext>>) -> GpuPipelineInit {
        if let Some(ctx) = context {
            let enhancer = match WgpuEnhancer::new(ctx.clone()) {
                Ok(enhancer) => {
                    info!(
                        "GPU enhancer ready on '{}' ({:?})",
                        ctx.adapter_info().name,
                        ctx.adapter_info().backend
                    );
                    Some(Arc::new(enhancer))
                }
                Err(err) => {
                    warn!("Failed to initialize GPU enhancer: {err}");
                    None
                }
            };
            let cropper = match GpuBatchCropper::new(ctx.clone()) {
                Ok(cropper) => Some(Arc::new(cropper)),
                Err(err) => {
                    warn!("Failed to initialize GPU batch cropper: {err}");
                    None
                }
            };
            (Some(ctx), enhancer, cropper)
        } else {
            (None, None, None)
        }
    }

    pub(crate) fn refresh_gpu_pipelines(&mut self, context: Option<Arc<GpuContext>>) {
        let (ctx, enhancer, cropper) = Self::init_gpu_pipelines(context);
        self.gpu_context = ctx;
        self.gpu_enhancer = enhancer;
        self.gpu_batch_cropper = cropper;
    }
}

impl App for YuNetApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        self.poll_worker(ctx);
        self.show_status_bar(ctx);
        self.show_navigation_panel(ctx);
        ui::config::panel::show_configuration_panel(self, ctx);
        self.show_preview(ctx);

        self.handle_shortcuts(ctx);

        if self.is_busy {
            ctx.request_repaint();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn app_with_temp_settings() -> (YuNetApp, tempfile::TempDir, egui::Context) {
        let ctx = egui::Context::default();
        let temp = tempdir().expect("tempdir");
        let settings_path = temp.path().join("gui_settings_test.json");
        let app = YuNetApp::create(&ctx, settings_path, None);
        (app, temp, ctx)
    }

    #[test]
    fn smoke_initializes_and_persists_settings() {
        use crate::core::settings::persist_settings;

        let ctx = egui::Context::default();
        let temp = tempdir().expect("tempdir");
        let settings_path = temp.path().join("config").join("gui_settings_smoke.json");

        let mut app = YuNetApp::create(&ctx, settings_path.clone(), None);
        assert!(app.detector.is_none());
        assert!(
            app.status_line.contains("Model not loaded")
                || app.status_line.contains("Select an image"),
            "status line should mention initial state, got {}",
            app.status_line
        );
        assert_eq!(app.settings_path, settings_path);

        app.settings.detection.score_threshold = 0.42;
        persist_settings(&app.settings, &app.settings_path).expect("persist settings");

        let saved = std::fs::read_to_string(&app.settings_path).expect("read settings");
        let json: serde_json::Value = serde_json::from_str(&saved).expect("parse settings");
        assert_eq!(json["detection"]["score_threshold"], json!(0.42));
    }

    #[test]
    fn crop_adjustments_are_clamped_and_record_history() {
        let (mut app, temp_dir, _ctx) = app_with_temp_settings();

        let base_history = app.crop_history.len();

        app.adjust_horizontal_offset(0.8);
        assert!((app.settings.crop.horizontal_offset - 0.8).abs() < 1e-6);
        assert_eq!(app.crop_history.len(), base_history + 1);

        app.adjust_vertical_offset(-2.0);
        assert_eq!(app.settings.crop.vertical_offset, -1.0);
        assert_eq!(app.crop_history.len(), base_history + 2);

        app.adjust_face_height(50.0);
        assert!((app.settings.crop.face_height_pct - 100.0).abs() < 1e-6);
        assert_eq!(app.crop_history.len(), base_history + 3);

        app.set_crop_preset("passport");
        assert_eq!(app.settings.crop.preset, "passport");
        assert_eq!(app.settings.crop.output_width, 413);
        assert_eq!(app.settings.crop.output_height, 531);
        assert_eq!(app.crop_history.len(), base_history + 4);

        let settings_file = temp_dir.path().join("gui_settings_test.json");
        assert!(
            settings_file.exists(),
            "persisted settings file should be created"
        );
    }

    #[test]
    fn resolved_dimensions_follow_preset_and_custom_settings() {
        let (mut app, _temp_dir, _ctx) = app_with_temp_settings();

        app.set_crop_preset("idcard");
        let (preset_w, preset_h) = app.resolved_output_dimensions();
        assert_eq!((preset_w, preset_h), (332, 498));

        app.settings.crop.preset = "custom".to_string();
        app.settings.crop.output_width = 720;
        app.settings.crop.output_height = 960;

        let (custom_w, custom_h) = app.resolved_output_dimensions();
        assert_eq!((custom_w, custom_h), (720, 960));

        let core_settings = app.build_crop_settings();
        assert_eq!(core_settings.output_width, 720);
        assert_eq!(core_settings.output_height, 960);
    }

    #[test]
    fn undo_and_redo_restore_crop_state_sequence() {
        let (mut app, _temp_dir, _ctx) = app_with_temp_settings();

        let initial_height = app.settings.crop.face_height_pct;

        app.adjust_face_height(5.0);
        let raised_height = app.settings.crop.face_height_pct;
        assert!(raised_height > initial_height);

        app.adjust_horizontal_offset(0.4);
        assert_eq!(app.settings.crop.horizontal_offset, 0.4);

        app.undo_crop_settings();
        assert_eq!(app.settings.crop.horizontal_offset, 0.0);
        assert!((app.settings.crop.face_height_pct - raised_height).abs() < 1e-6);

        app.undo_crop_settings();
        assert_eq!(app.settings.crop.horizontal_offset, 0.0);
        assert!((app.settings.crop.face_height_pct - initial_height).abs() < 1e-6);

        app.redo_crop_settings();
        assert!((app.settings.crop.face_height_pct - raised_height).abs() < 1e-6);
        assert_eq!(app.settings.crop.horizontal_offset, 0.0);

        app.redo_crop_settings();
        assert_eq!(app.settings.crop.horizontal_offset, 0.4);
    }

    #[test]
    fn enhancement_presets_apply_expected_parameters() {
        let (mut app, _temp_dir, _ctx) = app_with_temp_settings();

        app.settings.enhance.preset = "vivid".to_string();
        app.settings.enhance.auto_color = false;
        app.settings.enhance.exposure_stops = 0.0;
        app.settings.enhance.brightness = 0;
        app.settings.enhance.contrast = 1.0;
        app.settings.enhance.saturation = 1.0;
        app.settings.enhance.sharpness = 0.0;
        app.settings.enhance.skin_smooth = 0.0;

        app.apply_enhancement_preset();

        assert!(app.settings.enhance.auto_color);
        assert!((app.settings.enhance.exposure_stops - 0.2).abs() < 1e-6);
        assert_eq!(app.settings.enhance.brightness, 10);
        assert!((app.settings.enhance.contrast - 1.2).abs() < 1e-6);
        assert!((app.settings.enhance.saturation - 1.3).abs() < 1e-6);
        assert!((app.settings.enhance.sharpness - 0.8).abs() < 1e-6);
    }
}
