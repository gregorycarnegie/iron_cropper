mod theme;

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, mpsc},
};

use anyhow::{Context as AnyhowContext, Result};
use eframe::{App, CreationContext, Frame, NativeOptions, egui};
use egui::{
    CentralPanel, Color32, Context as EguiContext, DragValue, Key, Rect, RichText, ScrollArea,
    SidePanel, Slider, Spinner, Stroke, TextureHandle, TextureOptions, TopBottomPanel, pos2, vec2,
};
use log::{error, info, warn};
use rfd::FileDialog;
use yunet_core::{Detection, PostprocessConfig, PreprocessConfig, YuNetDetector};
use yunet_utils::{config::AppSettings, init_logging, load_image};

fn main() -> eframe::Result<()> {
    init_logging(log::LevelFilter::Info).expect("failed to initialize logging");
    let options = NativeOptions::default();

    eframe::run_native(
        "YuNet Desktop",
        options,
        Box::new(|cc| Ok(Box::new(YuNetApp::new(cc)))),
    )
}

struct YuNetApp {
    settings: AppSettings,
    settings_path: PathBuf,
    status_line: String,
    last_error: Option<String>,
    detector: Option<Arc<YuNetDetector>>,
    job_tx: mpsc::Sender<JobMessage>,
    job_rx: mpsc::Receiver<JobMessage>,
    preview: PreviewState,
    cache: HashMap<CacheKey, DetectionCacheEntry>,
    model_path_input: String,
    model_path_dirty: bool,
    is_busy: bool,
    texture_seq: u64,
    job_counter: u64,
    current_job: Option<u64>,
}

#[derive(Default)]
struct PreviewState {
    image_path: Option<PathBuf>,
    texture: Option<TextureHandle>,
    image_size: Option<(u32, u32)>,
    detections: Vec<Detection>,
}

impl PreviewState {
    fn begin_loading(&mut self, path: PathBuf) {
        self.image_path = Some(path);
        self.texture = None;
        self.image_size = None;
        self.detections.clear();
    }
}

enum JobMessage {
    DetectionFinished {
        job_id: u64,
        cache_key: CacheKey,
        data: DetectionJobSuccess,
    },
    DetectionFailed {
        job_id: u64,
        error: String,
    },
}

struct DetectionJobSuccess {
    path: PathBuf,
    color_image: egui::ColorImage,
    detections: Vec<Detection>,
    original_size: (u32, u32),
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct CacheKey {
    path: PathBuf,
    model_path: Option<String>,
    input_width: u32,
    input_height: u32,
    score_bits: u32,
    nms_bits: u32,
    top_k: usize,
}

struct DetectionCacheEntry {
    texture: TextureHandle,
    image_size: (u32, u32),
    detections: Vec<Detection>,
}

impl YuNetApp {
    fn new(cc: &CreationContext<'_>) -> Self {
        let settings_path = default_settings_path();
        Self::create(&cc.egui_ctx, settings_path)
    }

    pub(crate) fn create(ctx: &egui::Context, settings_path: PathBuf) -> Self {
        theme::apply(ctx);

        info!("Loading GUI settings from {}", settings_path.display());
        let settings = load_settings(&settings_path);
        let (job_tx, job_rx) = mpsc::channel();

        let detector = match build_detector(&settings) {
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

        Self {
            settings,
            settings_path,
            status_line,
            last_error: None,
            detector,
            job_tx,
            job_rx,
            preview: PreviewState::default(),
            cache: HashMap::new(),
            model_path_input,
            model_path_dirty: false,
            is_busy: false,
            texture_seq: 0,
            job_counter: 0,
            current_job: None,
        }
    }

    fn show_status_bar(&mut self, ctx: &EguiContext) {
        TopBottomPanel::top("yunet_status_bar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading(RichText::new("YuNet Face Detection").strong());
                ui.separator();
                ui.label(&self.status_line);
                if let Some(err) = &self.last_error {
                    ui.separator();
                    ui.colored_label(ui.visuals().warn_fg_color, err);
                }
            });
        });
    }

    fn show_configuration_panel(&mut self, ctx: &EguiContext) {
        SidePanel::left("yunet_settings_panel")
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                let mut settings_changed = false;
                let mut requires_detector_reset = false;
                let mut requires_cache_refresh = false;
                ui.heading("Image");
                if ui.button("Open image…").clicked() {
                    self.open_image_dialog();
                }
                if let Some(path) = &self.preview.image_path {
                    ui.label("Selected");
                    ui.monospace(path.to_string_lossy());
                } else {
                    ui.label("No image selected yet.");
                }
                if self.is_busy {
                    ui.horizontal(|ui| {
                        ui.add(Spinner::new());
                        ui.label("Running detection…");
                    });
                }

                ui.separator();
                ui.heading("Detections");
                if self.preview.detections.is_empty() {
                    if self.is_busy {
                        ui.label("Waiting for results…");
                    } else {
                        ui.label("No detections yet.");
                    }
                } else {
                    ui.label(format!(
                        "{} face(s) detected.",
                        self.preview.detections.len()
                    ));
                    ui.add_space(6.0);
                    for (index, detection) in self.preview.detections.iter().enumerate() {
                        let bbox = &detection.bbox;
                        ui.group(|ui| {
                            ui.label(format!("Detection {}", index + 1));
                            ui.label(format!("Score: {:.3}", detection.score));
                            ui.label(format!(
                                "BBox: x {:.0}, y {:.0}, w {:.0}, h {:.0}",
                                bbox.x, bbox.y, bbox.width, bbox.height
                            ));
                        });
                    }
                }

                ui.separator();
                ui.heading("Model Settings");
                ui.label("Model path");
                ui.horizontal(|ui| {
                    let response = ui.text_edit_singleline(&mut self.model_path_input);
                    if response.changed() {
                        self.model_path_dirty = true;
                    }
                    let enter_pressed = ui.input(|i| i.key_pressed(Key::Enter));
                    if self.model_path_dirty && (response.lost_focus() || enter_pressed) {
                        self.apply_model_path_input();
                    }
                    if ui.button("Browse…").clicked() {
                        self.open_model_dialog();
                    }
                });
                if self.model_path_dirty {
                    ui.label(RichText::new("Press Enter to apply model path changes.").weak());
                }

                ui.add_space(6.0);
                ui.label("Input size (W×H)");
                ui.horizontal(|ui| {
                    ui.label("Width");
                    let mut width = self.settings.input.width;
                    if ui
                        .add(DragValue::new(&mut width).range(64..=4096).speed(16.0))
                        .changed()
                    {
                        self.settings.input.width = width;
                        settings_changed = true;
                        requires_detector_reset = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Height");
                    let mut height = self.settings.input.height;
                    if ui
                        .add(DragValue::new(&mut height).range(64..=4096).speed(16.0))
                        .changed()
                    {
                        self.settings.input.height = height;
                        settings_changed = true;
                        requires_detector_reset = true;
                    }
                });

                ui.add_space(6.0);
                ui.label("Detection thresholds");
                let mut score = self.settings.detection.score_threshold;
                if ui
                    .add(Slider::new(&mut score, 0.0..=1.0).text("Score threshold"))
                    .changed()
                {
                    self.settings.detection.score_threshold = score;
                    settings_changed = true;
                    requires_cache_refresh = true;
                }
                let mut nms = self.settings.detection.nms_threshold;
                if ui
                    .add(Slider::new(&mut nms, 0.0..=1.0).text("NMS threshold"))
                    .changed()
                {
                    self.settings.detection.nms_threshold = nms;
                    settings_changed = true;
                    requires_cache_refresh = true;
                }
                let mut top_k = self.settings.detection.top_k as i64;
                if ui
                    .add(
                        DragValue::new(&mut top_k)
                            .range(1..=20_000)
                            .speed(100.0)
                            .suffix(" detections"),
                    )
                    .changed()
                {
                    self.settings.detection.top_k = top_k.max(1) as usize;
                    settings_changed = true;
                    requires_cache_refresh = true;
                }

                if settings_changed {
                    self.apply_settings_changes(requires_detector_reset, requires_cache_refresh);
                }

                ui.add_space(8.0);
                ui.small(
                    RichText::new(format!("Settings file: {}", self.settings_path.display()))
                        .weak(),
                );
            });
    }

    fn show_preview(&mut self, ctx: &EguiContext) {
        CentralPanel::default().show(ctx, |ui| {
            if let Some(texture) = &self.preview.texture {
                ScrollArea::both().show(ui, |ui| {
                    let response = ui.add(egui::Image::new(texture));
                    if let Some(dimensions) = self.preview.image_size {
                        self.paint_detections(ui, response.rect, dimensions);
                    }
                });
            } else {
                ui.vertical_centered(|ui| {
                    ui.add_space(48.0);
                    ui.heading("Image preview will appear here.");
                    ui.label("Select an image to run detection.");
                });
            }
        });
    }

    fn paint_detections(&self, ui: &egui::Ui, image_rect: Rect, image_size: (u32, u32)) {
        let painter = ui.painter().with_clip_rect(image_rect);
        let scale_x = image_rect.width() / image_size.0 as f32;
        let scale_y = image_rect.height() / image_size.1 as f32;

        let bbox_stroke = Stroke::new(2.0, Color32::from_rgb(255, 145, 77));
        let landmark_color = Color32::from_rgb(82, 180, 255);

        for detection in &self.preview.detections {
            let bbox = &detection.bbox;
            let top_left = pos2(
                image_rect.left() + bbox.x * scale_x,
                image_rect.top() + bbox.y * scale_y,
            );
            let size = vec2(bbox.width * scale_x, bbox.height * scale_y);
            let rect = Rect::from_min_size(top_left, size);
            painter.rect_stroke(rect, 0.0, bbox_stroke, egui::StrokeKind::Inside);

            for landmark in &detection.landmarks {
                let center = pos2(
                    image_rect.left() + landmark.x * scale_x,
                    image_rect.top() + landmark.y * scale_y,
                );
                painter.circle_filled(center, 3.0, landmark_color);
            }
        }
    }

    fn apply_settings_changes(
        &mut self,
        requires_detector_reset: bool,
        requires_cache_refresh: bool,
    ) {
        if requires_detector_reset {
            self.invalidate_detector();
        } else if requires_cache_refresh {
            self.clear_cache("Detection parameters updated. Re-run detection.");
        }
        self.persist_settings_with_feedback();
    }

    fn persist_settings_with_feedback(&mut self) {
        if let Err(err) = self.persist_settings() {
            let message = format!("Failed to persist settings: {err}");
            warn!("{message}");
            self.last_error = Some(message);
        }
    }

    fn persist_settings(&self) -> Result<()> {
        if let Some(parent) = self.settings_path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("failed to create settings directory {}", parent.display())
                })?;
            }
        }
        self.settings
            .save_to_path(&self.settings_path)
            .with_context(|| {
                format!(
                    "failed to write settings to {}",
                    self.settings_path.display()
                )
            })
    }

    fn invalidate_detector(&mut self) {
        if self.detector.is_some() {
            info!("Invalidating YuNet detector due to configuration change");
        }
        self.detector = None;
        self.clear_cache("Detector configuration changed. Re-run detection.");
    }

    fn clear_cache(&mut self, status_message: &str) {
        if !self.cache.is_empty() {
            info!("Clearing detection cache ({} entries)", self.cache.len());
        }
        self.cache.clear();
        if let Some(job) = self.current_job.take() {
            info!("Cancelling pending detection job {job} due to configuration change");
        }
        self.is_busy = false;
        self.status_line = status_message.to_owned();
    }

    fn apply_model_path_input(&mut self) {
        let trimmed = self.model_path_input.trim().to_owned();
        self.model_path_input = trimmed.clone();
        self.update_model_path(if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        });
    }

    fn update_model_path(&mut self, new_path: Option<String>) {
        let normalized = new_path.and_then(|value| {
            let trimmed = value.trim().to_owned();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        });
        let resolved = normalized
            .clone()
            .or_else(|| AppSettings::default().model_path.clone());

        if normalized.is_none() {
            if let Some(default_path) = resolved.as_deref() {
                info!("Falling back to default model path: {}", default_path);
            }
        }

        if self.settings.model_path != resolved {
            self.settings.model_path = resolved.clone();
            self.apply_settings_changes(true, false);
        }

        self.model_path_input = resolved.clone().unwrap_or_default();
        self.model_path_dirty = false;
    }

    fn open_image_dialog(&mut self) {
        if let Some(path) = FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp"])
            .pick_file()
        {
            self.start_detection(path);
        }
    }

    fn open_model_dialog(&mut self) {
        if let Some(path) = FileDialog::new()
            .add_filter("ONNX model", &["onnx"])
            .pick_file()
        {
            let display = path.to_string_lossy().to_string();
            self.update_model_path(Some(display));
        }
    }

    fn start_detection(&mut self, path: PathBuf) {
        let cache_key = self.cache_key_for_path(&path);
        if let Some(entry) = self.cache.get(&cache_key) {
            info!("Using cached detections for {}", path.display());
            self.preview.image_path = Some(path.clone());
            self.preview.texture = Some(entry.texture.clone());
            self.preview.image_size = Some(entry.image_size);
            self.preview.detections = entry.detections.clone();
            self.status_line = format!("Loaded cached detections for {}", path.display());
            self.last_error = None;
            self.is_busy = false;
            self.current_job = None;
            return;
        }

        let detector = match self.ensure_detector() {
            Ok(detector) => detector,
            Err(err) => {
                let message = format!("Unable to load YuNet model: {err}");
                self.last_error = Some(message.clone());
                self.status_line = "Failed to load YuNet model.".to_owned();
                warn!("{message}");
                return;
            }
        };

        let job_id = self.next_job_id();
        self.preview.begin_loading(path.clone());
        self.status_line = format!("Running detection for {}", path.display());
        self.last_error = None;
        self.is_busy = true;
        self.current_job = Some(job_id);
        info!("Launching detection job {job_id} for {}", path.display());

        let sender = self.job_tx.clone();
        rayon::spawn(move || {
            let payload = match perform_detection(detector, path.clone()) {
                Ok(data) => JobMessage::DetectionFinished {
                    job_id,
                    cache_key,
                    data,
                },
                Err(err) => JobMessage::DetectionFailed {
                    job_id,
                    error: format!("{err}"),
                },
            };

            if sender.send(payload).is_err() {
                error!("GUI dropped detection result for {}", path.display());
            }
        });
    }

    fn ensure_detector(&mut self) -> Result<Arc<YuNetDetector>> {
        if let Some(detector) = &self.detector {
            return Ok(detector.clone());
        }

        let detector = Arc::new(build_detector(&self.settings)?);
        self.detector = Some(detector.clone());
        Ok(detector)
    }

    fn cache_key_for_path(&self, path: &Path) -> CacheKey {
        CacheKey {
            path: path.to_path_buf(),
            model_path: self.settings.model_path.clone(),
            input_width: self.settings.input.width,
            input_height: self.settings.input.height,
            score_bits: self.settings.detection.score_threshold.to_bits(),
            nms_bits: self.settings.detection.nms_threshold.to_bits(),
            top_k: self.settings.detection.top_k,
        }
    }

    fn poll_worker(&mut self, ctx: &EguiContext) {
        let mut updated = false;
        while let Ok(message) = self.job_rx.try_recv() {
            self.handle_job_message(ctx, message);
            updated = true;
        }

        if updated {
            ctx.request_repaint();
        }
    }

    fn handle_job_message(&mut self, ctx: &EguiContext, message: JobMessage) {
        match message {
            JobMessage::DetectionFinished {
                job_id,
                cache_key,
                data,
            } => {
                if Some(job_id) != self.current_job {
                    info!(
                        "Ignoring stale detection result for {}",
                        data.path.display()
                    );
                    return;
                }

                self.current_job = None;
                self.is_busy = false;

                let DetectionJobSuccess {
                    path,
                    color_image,
                    detections,
                    original_size,
                } = data;

                let texture_name = format!("yunet-image-preview-{}", self.texture_seq);
                self.texture_seq = self.texture_seq.wrapping_add(1);

                let texture = ctx.load_texture(texture_name, color_image, TextureOptions::LINEAR);
                let cache_texture = texture.clone();

                let cached_detections = detections.clone();

                self.preview.texture = Some(texture);
                self.preview.image_size = Some(original_size);
                self.preview.detections = detections;
                self.preview.image_path = Some(path.clone());
                self.status_line = format!(
                    "Detected {} face(s) in {}",
                    self.preview.detections.len(),
                    path.display()
                );
                self.last_error = None;

                self.cache.insert(
                    cache_key,
                    DetectionCacheEntry {
                        texture: cache_texture,
                        image_size: original_size,
                        detections: cached_detections,
                    },
                );
            }
            JobMessage::DetectionFailed { job_id, error } => {
                if Some(job_id) != self.current_job {
                    info!("Ignoring stale detection error: {error}");
                    return;
                }

                self.current_job = None;
                self.is_busy = false;
                self.last_error = Some(error.clone());
                self.status_line = "Detection failed.".to_string();
                self.preview.texture = None;
                self.preview.image_size = None;
                self.preview.detections.clear();
            }
        }
    }

    fn next_job_id(&mut self) -> u64 {
        self.job_counter = self.job_counter.wrapping_add(1);
        if self.job_counter == 0 {
            self.job_counter = 1;
        }
        self.job_counter
    }
}

impl App for YuNetApp {
    fn update(&mut self, ctx: &EguiContext, _frame: &mut Frame) {
        self.poll_worker(ctx);
        self.show_status_bar(ctx);
        self.show_configuration_panel(ctx);
        self.show_preview(ctx);

        if self.is_busy {
            ctx.request_repaint();
        }
    }
}

fn load_settings(path: &Path) -> AppSettings {
    match AppSettings::load_from_path(path) {
        Ok(settings) => settings,
        Err(err) => {
            warn!(
                "Failed to load settings from {}: {err:?}. Falling back to defaults.",
                path.display()
            );
            AppSettings::default()
        }
    }
}

fn default_settings_path() -> PathBuf {
    std::env::current_dir()
        .map(|dir| dir.join("config/gui_settings.json"))
        .unwrap_or_else(|_| PathBuf::from("config/gui_settings.json"))
}

fn build_detector(settings: &AppSettings) -> Result<YuNetDetector> {
    let model_path = settings
        .model_path
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("no model path configured"))?;

    let preprocess: PreprocessConfig = settings.input.into();

    let postprocess: PostprocessConfig = (&settings.detection).into();

    YuNetDetector::new(model_path, preprocess, postprocess).with_context(|| {
        format!(
            "failed to load YuNet model from configured path {}",
            model_path
        )
    })
}

fn perform_detection(detector: Arc<YuNetDetector>, path: PathBuf) -> Result<DetectionJobSuccess> {
    let image = load_image(&path)
        .with_context(|| format!("failed to load image from {}", path.display()))?;
    let detection_output = detector
        .detect_image(&image)
        .with_context(|| format!("YuNet detection failed for {}", path.display()))?;

    let rgba = image.to_rgba8();
    let size = [rgba.width() as usize, rgba.height() as usize];
    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());

    Ok(DetectionJobSuccess {
        path,
        color_image,
        detections: detection_output.detections,
        original_size: detection_output.original_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn smoke_initializes_and_persists_settings() {
        let temp = tempdir().expect("tempdir");
        let settings_path = temp.path().join("config").join("gui_settings_smoke.json");
        let ctx = egui::Context::default();

        let mut app = YuNetApp::create(&ctx, settings_path.clone());
        assert!(app.detector.is_none());
        assert!(
            app.status_line.contains("Model not loaded")
                || app.status_line.contains("Select an image"),
            "status line should mention initial state, got {}",
            app.status_line
        );
        assert_eq!(app.settings_path, settings_path);

        app.settings.detection.score_threshold = 0.42;
        app.persist_settings().expect("persist settings");

        let saved = std::fs::read_to_string(&app.settings_path).expect("read settings");
        let json: serde_json::Value = serde_json::from_str(&saved).expect("parse settings");
        assert_eq!(json["detection"]["score_threshold"], json!(0.42));
    }
}
