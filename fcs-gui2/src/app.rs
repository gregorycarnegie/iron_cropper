//! eframe::App implementation for fcs-gui2.

use crate::core::{detection::build_detector, settings::load_settings};
use crate::types::*;
use crate::ui;

use eframe::{App, CreationContext, Frame};
use fcs_utils::{
    WgpuEnhancer,
    config::default_settings_path,
    configure_telemetry,
    gpu::{GpuBatchCropper, GpuContext},
};
use log::info;
use lru::LruCache;
use std::{
    collections::HashSet,
    num::NonZeroUsize,
    path::PathBuf,
    sync::{Arc, mpsc},
};

const SIDEBAR_W:   f32 = 300.0;
const INSPECTOR_W: f32 = 340.0;

// ── App creation ─────────────────────────────────────────────────────────────

impl App2 {
    pub fn new(cc: &CreationContext<'_>) -> Self {
        crate::theme::apply(&cc.egui_ctx);

        let settings_path = default_settings_path();
        let settings = load_settings(&settings_path);
        configure_telemetry(settings.telemetry.enabled, settings.telemetry.level_filter());

        let shared_gpu = cc.wgpu_render_state.as_ref().map(|rs| {
            info!("Sharing GPU context from eframe renderer");
            Arc::new(GpuContext::from_existing(
                None, None,
                rs.device.clone(), rs.queue.clone(),
                rs.adapter.get_info(),
            ))
        });

        let (initial_gpu_status, initial_gpu_context, detector_result) =
            build_detector(&settings, shared_gpu);

        let (gpu_context, gpu_enhancer, gpu_batch_cropper) =
            init_gpu_pipelines(initial_gpu_context);

        let detector = match detector_result {
            Ok(d) => { info!("YuNet model loaded"); Some(Arc::new(d)) }
            Err(err) => { log::warn!("Model unavailable: {err}"); None }
        };

        let status_line = if detector.is_some() {
            "Model ready — select an image to detect faces.".to_owned()
        } else {
            "Model not loaded — configure model path to begin.".to_owned()
        };

        let (job_tx, job_rx) = mpsc::channel();
        let model_path_input = settings.model_path.clone().unwrap_or_default();
        let crop_history = vec![settings.crop.clone()];
        let crop_fill_hex_input = format!("#{:02X}{:02X}{:02X}",
            settings.crop.fill_color.red, settings.crop.fill_color.green, settings.crop.fill_color.blue);

        Self {
            settings,
            settings_path,
            gpu_status: initial_gpu_status,
            gpu_context,
            gpu_enhancer,
            gpu_batch_cropper,
            detector,
            job_tx,
            job_rx,
            preview: PreviewState::default(),
            cache: LruCache::new(NonZeroUsize::new(50).unwrap()),
            crop_preview_cache: LruCache::new(NonZeroUsize::new(500).unwrap()),
            image_cache: LruCache::new(NonZeroUsize::new(20).unwrap()),
            selected_faces: HashSet::new(),
            show_crop_overlay: true,
            crop_history,
            crop_history_index: 0,
            crop_fill_hex_input,
            aspect_ratio_locked: false,
            batch_files: Vec::new(),
            batch_current_index: None,
            mapping: MappingUiState::new(),
            manual_box_draft: None,
            active_bbox_drag: None,
            manual_box_tool_enabled: false,
            sidebar_tab: SidebarTab::Queue,
            inspector_tab: InspectorTab::Crop,
            panel_state: PanelState::default(),
            log_lines: vec![
                LogLine { timestamp: "—".into(), message: "Waiting for image…".into(), kind: LogKind::Info },
            ],
            status_line,
            last_error: None,
            is_busy: false,
            texture_seq: 0,
            job_counter: 0,
            current_job: None,
            model_path_input,
            model_path_dirty: false,
            clipboard_temp_images: Vec::new(),
            webcam_state: WebcamState::default(),
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
        }
    }
}

fn init_gpu_pipelines(
    context: Option<Arc<GpuContext>>,
) -> (Option<Arc<GpuContext>>, Option<Arc<WgpuEnhancer>>, Option<Arc<GpuBatchCropper>>) {
    let Some(ctx) = context else { return (None, None, None); };
    let enhancer = WgpuEnhancer::new(ctx.clone()).ok().map(Arc::new);
    let cropper  = GpuBatchCropper::new(ctx.clone()).ok().map(Arc::new);
    (Some(ctx), enhancer, cropper)
}

// ── eframe::App ───────────────────────────────────────────────────────────────

impl App for App2 {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        self.poll_worker(ctx);
        self.handle_dropped_files(ctx);
        if self.is_busy { ctx.request_repaint(); }
    }

    fn ui(&mut self, root_ui: &mut egui::Ui, _frame: &mut Frame) {
        ui::titlebar::show(root_ui, self);
        ui::menubar::show(root_ui, self);
        ui::toolbar::show(root_ui, self);
        ui::statusbar::show(root_ui, self);

        egui::Panel::left("sidebar")
            .exact_size(SIDEBAR_W)
            .resizable(false)
            .show_inside(root_ui, |ui| {
                ui::sidebar::show(ui, self);
            });

        egui::Panel::right("inspector")
            .exact_size(INSPECTOR_W)
            .resizable(false)
            .show_inside(root_ui, |ui| {
                ui::inspector::show(ui, self);
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(crate::theme::P::BG1))
            .show_inside(root_ui, |ui| {
                ui::canvas::show(ui, self);
            });
    }
}

// ── Worker polling / dropped files ───────────────────────────────────────────

impl App2 {
    fn poll_worker(&mut self, ctx: &egui::Context) {
        let mut updated = false;
        while let Ok(msg) = self.job_rx.try_recv() {
            self.handle_job_message(ctx, msg);
            updated = true;
        }
        if updated { ctx.request_repaint(); }
    }

    fn handle_job_message(&mut self, ctx: &egui::Context, msg: JobMessage) {
        use egui::TextureOptions;
        match msg {
            JobMessage::DetectionFinished { job_id, cache_key, data } => {
                if Some(job_id) != self.current_job { return; }
                self.current_job = None;
                self.is_busy = false;
                self.preview.is_loading = false;

                let tex_name = format!("preview_{}", self.texture_seq);
                self.texture_seq += 1;
                let texture = ctx.load_texture(&tex_name, data.color_image, TextureOptions::default());

                self.preview.texture = Some(texture.clone());
                self.preview.image_size = Some(data.original_size);
                self.preview.detections = data.detections.clone();
                self.preview.source_image = Some(data.original_image.clone());
                self.active_bbox_drag = None;
                self.manual_box_tool_enabled = false;

                let n = self.preview.detections.len();
                self.push_log(format!("Detected {n} face(s)"), LogKind::Ok);
                self.status_line = format!("Detected {n} face(s)");

                // Auto-select all faces
                self.selected_faces = (0..n).collect();

                self.cache.put(cache_key, crate::types::DetectionCacheEntry {
                    texture,
                    detections: data.detections,
                    original_size: data.original_size,
                    source_image: data.original_image,
                });
            }
            JobMessage::DetectionFailed { job_id, error } => {
                if Some(job_id) != self.current_job { return; }
                self.current_job = None;
                self.is_busy = false;
                self.preview.is_loading = false;
                self.last_error = Some(error.clone());
                self.push_log(format!("Detection failed: {error}"), LogKind::Warn);
                self.status_line = "Detection failed.".to_owned();
            }
            JobMessage::BatchProgress { index, status } => {
                if let Some(f) = self.batch_files.get_mut(index) { f.status = status; }
            }
            JobMessage::BatchComplete { completed, failed } => {
                self.is_busy = false;
                self.push_log(format!("Batch done: {completed} ok, {failed} failed"), LogKind::Ok);
                self.status_line = format!("Batch complete: {completed} ok, {failed} failed");
            }
            _ => {}
        }
    }

    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped: Vec<_> = ctx.input(|i| i.raw.dropped_files.clone());
        for file in dropped {
            if let Some(path) = file.path {
                let ext = path.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "webp" | "bmp" | "tiff") {
                    self.load_image_path(path.clone());
                    self.batch_files.push(BatchFile {
                        path,
                        status: BatchFileStatus::Pending,
                        output_override: None,
                    });
                }
            }
        }
    }

    pub fn push_log(&mut self, message: String, kind: LogKind) {
        use std::time::{SystemTime, UNIX_EPOCH};
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let h = (secs / 3600) % 24;
        let m = (secs / 60) % 60;
        let s = secs % 60;
        self.log_lines.push(LogLine {
            timestamp: format!("{h:02}:{m:02}:{s:02}"),
            message,
            kind,
        });
        if self.log_lines.len() > 100 {
            self.log_lines.remove(0);
        }
    }

    pub fn load_image_path(&mut self, path: PathBuf) {
        use crate::core::detection::spawn_detection_job;
        self.preview.begin_loading(path.clone());
        self.is_busy = true;
        let job_id = self.job_counter;
        self.job_counter += 1;
        self.current_job = Some(job_id);
        self.push_log(format!("Loading {}", path.display()), LogKind::Info);
        spawn_detection_job(
            job_id,
            path,
            self.detector.clone(),
            self.settings.clone(),
            self.job_tx.clone(),
        );
    }
}
