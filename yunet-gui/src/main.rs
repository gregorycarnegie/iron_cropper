//! Desktop GUI for YuNet face detection.

mod theme;

use std::{
    collections::{HashMap, HashSet},
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
use yunet_core::{
    CropSettings as CoreCropSettings, Detection, PositioningMode, PostprocessConfig,
    PreprocessConfig, YuNetDetector, calculate_crop_region, preset_by_name,
};
use yunet_utils::{
    config::AppSettings,
    enhance::{EnhancementSettings, apply_enhancements},
    init_logging,
    load_image,
    quality::{Quality, estimate_sharpness}
};

/// Main entry point for the GUI application.
fn main() -> eframe::Result<()> {
    init_logging(log::LevelFilter::Info).expect("failed to initialize logging");
    let options = NativeOptions::default();

    eframe::run_native(
        "YuNet Desktop",
        options,
        Box::new(|cc| Ok(Box::new(YuNetApp::new(cc)))),
    )
}

/// Status of a batch file being processed.
#[derive(Debug, Clone, PartialEq, Eq)]
enum BatchFileStatus {
    Pending,
    Processing,
    Completed { faces_detected: usize, faces_exported: usize },
    Failed { error: String },
}

/// A file in the batch processing queue.
#[derive(Debug, Clone)]
struct BatchFile {
    path: PathBuf,
    status: BatchFileStatus,
}

/// The main application state for the YuNet GUI.
struct YuNetApp {
    /// User-configurable settings.
    settings: AppSettings,
    /// Path to the settings file on disk.
    settings_path: PathBuf,
    /// The current status message displayed in the top bar.
    status_line: String,
    /// The last error message, if any.
    last_error: Option<String>,
    /// The face detector instance.
    detector: Option<Arc<YuNetDetector>>,
    /// Sender for submitting detection jobs to a background thread.
    job_tx: mpsc::Sender<JobMessage>,
    /// Receiver for collecting results from detection jobs.
    job_rx: mpsc::Receiver<JobMessage>,
    /// State of the image preview panel.
    preview: PreviewState,
    /// Cache for detection results to avoid re-running on the same image and settings.
    cache: HashMap<CacheKey, DetectionCacheEntry>,
    /// The current value of the model path text input.
    model_path_input: String,
    /// Flag indicating if the model path input has been modified.
    model_path_dirty: bool,
    /// Flag indicating if a detection job is currently running.
    is_busy: bool,
    /// A counter to generate unique texture names.
    texture_seq: u64,
    /// A counter to generate unique job IDs.
    job_counter: u64,
    /// The ID of the currently running detection job.
    current_job: Option<u64>,
    /// Flag indicating whether to show crop region overlays.
    show_crop_overlay: bool,
    /// Set of selected face indices (for cropping).
    selected_faces: HashSet<usize>,
    /// Batch mode state.
    batch_files: Vec<BatchFile>,
    /// Current index in batch processing.
    batch_current_index: Option<usize>,
}

/// A detection with associated quality score and thumbnail.
#[derive(Clone)]
struct DetectionWithQuality {
    /// The core detection data.
    detection: Detection,
    /// Laplacian variance (sharpness score).
    quality_score: f64,
    /// Quality level (Low, Medium, High).
    quality: Quality,
    /// Thumbnail texture handle for the face region.
    thumbnail: Option<TextureHandle>,
}

/// State related to the image preview panel.
#[derive(Default)]
struct PreviewState {
    /// The path to the currently displayed image.
    image_path: Option<PathBuf>,
    /// The handle to the egui texture for the image.
    texture: Option<TextureHandle>,
    /// The dimensions of the original image.
    image_size: Option<(u32, u32)>,
    /// The list of detections with quality scores for the current image.
    detections: Vec<DetectionWithQuality>,
}

impl PreviewState {
    /// Resets the preview state to a loading state for a new image.
    fn begin_loading(&mut self, path: PathBuf) {
        self.image_path = Some(path);
        self.texture = None;
        self.image_size = None;
        self.detections.clear();
    }
}

/// A message sent from a background detection job to the GUI thread.
enum JobMessage {
    /// Indicates that a detection job has finished successfully.
    DetectionFinished {
        job_id: u64,
        cache_key: CacheKey,
        data: DetectionJobSuccess,
    },
    /// Indicates that a detection job has failed.
    DetectionFailed { job_id: u64, error: String },
}

/// The data returned from a successful detection job.
struct DetectionJobSuccess {
    path: PathBuf,
    color_image: egui::ColorImage,
    detections: Vec<DetectionWithQuality>,
    original_size: (u32, u32),
}

/// A key used to cache detection results.
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

/// An entry in the detection cache.
struct DetectionCacheEntry {
    texture: TextureHandle,
    image_size: (u32, u32),
    detections: Vec<DetectionWithQuality>,
}

impl YuNetApp {
    /// Creates a new `YuNetApp` instance.
    fn new(cc: &CreationContext<'_>) -> Self {
        let settings_path = default_settings_path();
        Self::create(&cc.egui_ctx, settings_path)
    }

    /// Creates a new `YuNetApp` instance with a specific settings path.
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
            show_crop_overlay: true,
            selected_faces: HashSet::new(),
            batch_files: Vec::new(),
            batch_current_index: None,
        }
    }

    /// Renders the top status bar.
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

    /// Renders the left-hand configuration panel.
    fn show_configuration_panel(&mut self, ctx: &EguiContext) {
        SidePanel::left("yunet_settings_panel")
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                let mut settings_changed = false;
                let mut requires_detector_reset = false;
                let mut requires_cache_refresh = false;
                ui.heading("Image");
                ui.horizontal(|ui| {
                    if ui.button("Open image…").clicked() {
                        self.open_image_dialog();
                    }
                    if ui.button("Load multiple…").clicked() {
                        self.open_batch_dialog();
                    }
                });

                if !self.batch_files.is_empty() {
                    ui.label(format!("Batch: {} images loaded", self.batch_files.len()));
                } else if let Some(path) = &self.preview.image_path {
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
                ui.heading("Detected Faces");
                if self.preview.detections.is_empty() {
                    if self.is_busy {
                        ui.label("Waiting for results…");
                    } else {
                        ui.label("No faces detected yet.");
                    }
                } else {
                    ui.label(format!(
                        "{} face(s) detected. Click to select for cropping.",
                        self.preview.detections.len()
                    ));
                    ui.add_space(8.0);

                    ScrollArea::vertical().show(ui, |ui| {
                        for (index, det_with_quality) in self.preview.detections.iter().enumerate() {
                            let is_selected = self.selected_faces.contains(&index);

                            let quality_color = match det_with_quality.quality {
                                Quality::High => Color32::from_rgb(0, 200, 100),
                                Quality::Medium => Color32::from_rgb(255, 180, 0),
                                Quality::Low => Color32::from_rgb(255, 80, 80),
                            };

                            let frame_color = if is_selected {
                                Color32::from_rgb(100, 150, 255)
                            } else {
                                ui.visuals().widgets.noninteractive.bg_fill
                            };

                            let frame_stroke = if is_selected {
                                Stroke::new(3.0, Color32::from_rgb(100, 150, 255))
                            } else {
                                Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color)
                            };

                            let response = ui.group(|ui| {
                                egui::Frame::new()
                                    .fill(frame_color)
                                    .stroke(frame_stroke)
                                    .inner_margin(4.0)
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            // Show thumbnail if available
                                            if let Some(thumbnail) = &det_with_quality.thumbnail {
                                                let thumb_response = ui.add(egui::Image::new(thumbnail).max_width(80.0));
                                                if thumb_response.clicked() {
                                                    if is_selected {
                                                        self.selected_faces.remove(&index);
                                                    } else {
                                                        self.selected_faces.insert(index);
                                                    }
                                                }
                                            }

                                            ui.vertical(|ui| {
                                                ui.label(RichText::new(format!("Face {}", index + 1)).strong());
                                                ui.label(format!("Conf: {:.2}", det_with_quality.detection.score));
                                                ui.horizontal(|ui| {
                                                    ui.label("Quality:");
                                                    ui.colored_label(quality_color, format!("{:?}", det_with_quality.quality));
                                                });
                                                ui.label(format!("Score: {:.0}", det_with_quality.quality_score));

                                                if ui.small_button(if is_selected { "Deselect" } else { "Select" }).clicked() {
                                                    if is_selected {
                                                        self.selected_faces.remove(&index);
                                                    } else {
                                                        self.selected_faces.insert(index);
                                                    }
                                                }
                                            });
                                        });
                                    });
                            });

                            if response.response.clicked() {
                                if is_selected {
                                    self.selected_faces.remove(&index);
                                } else {
                                    self.selected_faces.insert(index);
                                }
                            }
                        }
                    });

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Select All").clicked() {
                            self.selected_faces = (0..self.preview.detections.len()).collect();
                        }
                        if ui.button("Deselect All").clicked() {
                            self.selected_faces.clear();
                        }
                    });

                    ui.add_space(8.0);
                    ui.separator();

                    let num_selected = self.selected_faces.len();
                    if num_selected > 0 {
                        let button_label = format!("Export {} Selected Face{}", num_selected, if num_selected == 1 { "" } else { "s" });
                        if ui.button(button_label).clicked() {
                            self.export_selected_faces();
                        }
                    } else {
                        ui.label(RichText::new("Select faces to enable export").weak());
                    }
                }

                // Batch processing panel
                if !self.batch_files.is_empty() {
                    ui.separator();
                    ui.heading("Batch Processing");

                    let total = self.batch_files.len();
                    let completed = self.batch_files.iter().filter(|f| matches!(f.status, BatchFileStatus::Completed { .. })).count();
                    let failed = self.batch_files.iter().filter(|f| matches!(f.status, BatchFileStatus::Failed { .. })).count();

                    ui.label(format!("Progress: {}/{} files", completed + failed, total));
                    if failed > 0 {
                        ui.label(RichText::new(format!("({} failed)", failed)).color(Color32::from_rgb(255, 80, 80)));
                    }

                    ui.add_space(6.0);
                    ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                        for (idx, batch_file) in self.batch_files.iter().enumerate() {
                            let filename = batch_file.path.file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("unknown");

                            let status_text = match &batch_file.status {
                                BatchFileStatus::Pending => "Pending",
                                BatchFileStatus::Processing => "Processing...",
                                BatchFileStatus::Completed { faces_detected, faces_exported } => {
                                    &format!("{} faces, {} exported", faces_detected, faces_exported)
                                }
                                BatchFileStatus::Failed { error: _ } => "Failed",
                            };

                            let status_color = match &batch_file.status {
                                BatchFileStatus::Pending => Color32::GRAY,
                                BatchFileStatus::Processing => Color32::from_rgb(100, 150, 255),
                                BatchFileStatus::Completed { .. } => Color32::from_rgb(0, 200, 100),
                                BatchFileStatus::Failed { .. } => Color32::from_rgb(255, 80, 80),
                            };

                            ui.horizontal(|ui| {
                                ui.label(format!("{}.", idx + 1));
                                ui.label(filename);
                                ui.colored_label(status_color, status_text);
                            });
                        }
                    });

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Export All Batch Files").clicked() {
                            self.start_batch_export();
                        }
                        if ui.button("Clear Batch").clicked() {
                            self.batch_files.clear();
                            self.batch_current_index = None;
                        }
                    });
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

                ui.separator();
                ui.heading("Crop Settings");

                ui.checkbox(&mut self.show_crop_overlay, "Show crop preview overlay");
                ui.add_space(6.0);

                ui.label("Preset");
                egui::ComboBox::from_label("")
                    .selected_text(&self.settings.crop.preset)
                    .show_ui(ui, |ui| {
                        let presets = [
                            ("linkedin", "LinkedIn (400×400)"),
                            ("passport", "Passport (413×531)"),
                            ("instagram", "Instagram (1080×1080)"),
                            ("idcard", "ID Card (332×498)"),
                            ("avatar", "Avatar (512×512)"),
                            ("headshot", "Headshot (600×800)"),
                            ("custom", "Custom size"),
                        ];
                        for (value, label) in presets {
                            if ui.selectable_label(self.settings.crop.preset == value, label).clicked() {
                                self.settings.crop.preset = value.to_string();
                                settings_changed = true;
                            }
                        }
                    });

                if self.settings.crop.preset == "custom" {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label("Width");
                        let mut width = self.settings.crop.output_width;
                        if ui.add(DragValue::new(&mut width).range(64..=4096).speed(16.0)).changed() {
                            self.settings.crop.output_width = width;
                            settings_changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Height");
                        let mut height = self.settings.crop.output_height;
                        if ui.add(DragValue::new(&mut height).range(64..=4096).speed(16.0)).changed() {
                            self.settings.crop.output_height = height;
                            settings_changed = true;
                        }
                    });
                }

                ui.add_space(6.0);
                let mut face_pct = self.settings.crop.face_height_pct;
                if ui.add(Slider::new(&mut face_pct, 10.0..=100.0).text("Face height %")).changed() {
                    self.settings.crop.face_height_pct = face_pct;
                    settings_changed = true;
                }

                ui.add_space(6.0);
                ui.label("Positioning mode");
                egui::ComboBox::from_label(" ")
                    .selected_text(&self.settings.crop.positioning_mode)
                    .show_ui(ui, |ui| {
                        let modes = [
                            ("center", "Center"),
                            ("rule-of-thirds", "Rule of Thirds"),
                            ("custom", "Custom offsets"),
                        ];
                        for (value, label) in modes {
                            if ui.selectable_label(self.settings.crop.positioning_mode == value, label).clicked() {
                                self.settings.crop.positioning_mode = value.to_string();
                                settings_changed = true;
                            }
                        }
                    });

                if self.settings.crop.positioning_mode == "custom" {
                    ui.add_space(4.0);
                    let mut vert = self.settings.crop.vertical_offset;
                    if ui.add(Slider::new(&mut vert, -1.0..=1.0).text("Vertical offset")).changed() {
                        self.settings.crop.vertical_offset = vert;
                        settings_changed = true;
                    }
                    let mut horiz = self.settings.crop.horizontal_offset;
                    if ui.add(Slider::new(&mut horiz, -1.0..=1.0).text("Horizontal offset")).changed() {
                        self.settings.crop.horizontal_offset = horiz;
                        settings_changed = true;
                    }
                }

                if settings_changed {
                    self.persist_settings_with_feedback();
                }

                ui.separator();
                ui.heading("Enhancement Settings");

                ui.checkbox(&mut self.settings.enhance.enabled, "Enable enhancements");
                ui.add_space(6.0);

                if self.settings.enhance.enabled {
                    ui.label("Preset");
                    egui::ComboBox::from_label("  ")
                        .selected_text(&self.settings.enhance.preset)
                        .show_ui(ui, |ui| {
                            let presets = [
                                ("none", "None (Manual)"),
                                ("natural", "Natural"),
                                ("vivid", "Vivid"),
                                ("professional", "Professional"),
                            ];
                            for (value, label) in presets {
                                if ui.selectable_label(self.settings.enhance.preset == value, label).clicked() {
                                    self.settings.enhance.preset = value.to_string();
                                    self.apply_enhancement_preset();
                                    settings_changed = true;
                                }
                            }
                        });

                    ui.add_space(6.0);
                    ui.checkbox(&mut self.settings.enhance.auto_color, "Auto color correction");

                    ui.add_space(6.0);
                    let mut exp = self.settings.enhance.exposure_stops;
                    if ui.add(Slider::new(&mut exp, -2.0..=2.0).text("Exposure (stops)")).changed() {
                        self.settings.enhance.exposure_stops = exp;
                        settings_changed = true;
                    }

                    let mut bright = self.settings.enhance.brightness;
                    if ui.add(Slider::new(&mut bright, -100..=100).text("Brightness")).changed() {
                        self.settings.enhance.brightness = bright;
                        settings_changed = true;
                    }

                    let mut con = self.settings.enhance.contrast;
                    if ui.add(Slider::new(&mut con, 0.5..=2.0).text("Contrast")).changed() {
                        self.settings.enhance.contrast = con;
                        settings_changed = true;
                    }

                    let mut sat = self.settings.enhance.saturation;
                    if ui.add(Slider::new(&mut sat, 0.0..=2.5).text("Saturation")).changed() {
                        self.settings.enhance.saturation = sat;
                        settings_changed = true;
                    }

                    let mut sharp = self.settings.enhance.sharpness;
                    if ui.add(Slider::new(&mut sharp, 0.0..=2.0).text("Sharpness")).changed() {
                        self.settings.enhance.sharpness = sharp;
                        settings_changed = true;
                    }

                    ui.add_space(6.0);
                    if ui.button("Reset to defaults").clicked() {
                        self.settings.enhance = yunet_utils::config::EnhanceSettings::default();
                        settings_changed = true;
                    }
                }

                if settings_changed {
                    self.persist_settings_with_feedback();
                }

                ui.add_space(8.0);
                ui.small(
                    RichText::new(format!("Settings file: {}", self.settings_path.display()))
                        .weak(),
                );
            });
    }

    /// Renders the main image preview panel.
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

    /// Applies an enhancement preset to the current settings.
    fn apply_enhancement_preset(&mut self) {
        match self.settings.enhance.preset.as_str() {
            "natural" => {
                self.settings.enhance.auto_color = false;
                self.settings.enhance.exposure_stops = 0.0;
                self.settings.enhance.brightness = 5;
                self.settings.enhance.contrast = 1.05;
                self.settings.enhance.saturation = 1.05;
                self.settings.enhance.sharpness = 0.3;
            }
            "vivid" => {
                self.settings.enhance.auto_color = true;
                self.settings.enhance.exposure_stops = 0.2;
                self.settings.enhance.brightness = 10;
                self.settings.enhance.contrast = 1.2;
                self.settings.enhance.saturation = 1.3;
                self.settings.enhance.sharpness = 0.8;
            }
            "professional" => {
                self.settings.enhance.auto_color = false;
                self.settings.enhance.exposure_stops = 0.0;
                self.settings.enhance.brightness = 0;
                self.settings.enhance.contrast = 1.1;
                self.settings.enhance.saturation = 1.0;
                self.settings.enhance.sharpness = 0.6;
            }
            _ => {
                // "none" or unknown - no changes
            }
        }
    }

    /// Builds yunet-core CropSettings from the GUI settings.
    fn build_crop_settings(&self) -> CoreCropSettings {
        let (output_width, output_height) = if self.settings.crop.preset == "custom" {
            (
                self.settings.crop.output_width,
                self.settings.crop.output_height,
            )
        } else if let Some(preset) = preset_by_name(&self.settings.crop.preset) {
            (preset.width, preset.height)
        } else {
            // Fallback to linkedin if preset not found
            (400, 400)
        };

        let positioning_mode = match self.settings.crop.positioning_mode.as_str() {
            "rule-of-thirds" | "rule_of_thirds" => PositioningMode::RuleOfThirds,
            "custom" => PositioningMode::Custom,
            _ => PositioningMode::Center,
        };

        CoreCropSettings {
            output_width,
            output_height,
            face_height_pct: self.settings.crop.face_height_pct,
            positioning_mode,
            horizontal_offset: self.settings.crop.horizontal_offset,
            vertical_offset: self.settings.crop.vertical_offset,
        }
    }

    /// Builds EnhancementSettings from the GUI settings.
    fn build_enhancement_settings(&self) -> EnhancementSettings {
        EnhancementSettings {
            auto_color: self.settings.enhance.auto_color,
            exposure_stops: self.settings.enhance.exposure_stops,
            brightness: self.settings.enhance.brightness,
            contrast: self.settings.enhance.contrast,
            saturation: self.settings.enhance.saturation,
            unsharp_amount: 0.6,  // Base unsharp amount
            unsharp_radius: 1.0,
            sharpness: self.settings.enhance.sharpness,
        }
    }

    /// Exports selected faces to disk.
    fn export_selected_faces(&mut self) {
        // Get output directory from user
        let output_dir = match FileDialog::new()
            .set_title("Select output directory for cropped faces")
            .pick_folder()
        {
            Some(dir) => dir,
            None => {
                info!("Export cancelled by user");
                return;
            }
        };

        // Load source image
        let source_path = match &self.preview.image_path {
            Some(p) => p.clone(),
            None => {
                self.last_error = Some("No image loaded".to_string());
                return;
            }
        };

        let source_image = match load_image(&source_path) {
            Ok(img) => img,
            Err(e) => {
                self.last_error = Some(format!("Failed to load source image: {}", e));
                error!("Failed to load source image: {}", e);
                return;
            }
        };

        let crop_settings = self.build_crop_settings();
        let enhancement_settings = self.build_enhancement_settings();
        let enhance_enabled = self.settings.enhance.enabled;

        let mut export_count = 0;
        let mut error_count = 0;

        // Export each selected face
        for &face_idx in &self.selected_faces {
            if let Some(det_with_quality) = self.preview.detections.get(face_idx) {
                let bbox = &det_with_quality.detection.bbox;

                // Calculate crop region
                let crop_region = calculate_crop_region(
                    source_image.width(),
                    source_image.height(),
                    bbox.clone(),
                    &crop_settings,
                );

                // Extract crop
                let cropped = source_image.crop_imm(
                    crop_region.x,
                    crop_region.y,
                    crop_region.width,
                    crop_region.height,
                );

                // Resize to output dimensions
                let resized = cropped.resize_exact(
                    crop_settings.output_width,
                    crop_settings.output_height,
                    image::imageops::FilterType::Lanczos3,
                );

                // Apply enhancements if enabled
                let final_image = if enhance_enabled {
                    apply_enhancements(&resized, &enhancement_settings)
                } else {
                    resized
                };

                // Generate output filename
                let source_stem = source_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("face");
                let ext = match self.settings.crop.output_format.as_str() {
                    "jpeg" | "jpg" => "jpg",
                    "webp" => "webp",
                    _ => "png",
                };
                let output_filename = format!("{}_face_{:02}.{}", source_stem, face_idx + 1, ext);
                let output_path = output_dir.join(output_filename);

                // Save image
                let save_result = match self.settings.crop.output_format.as_str() {
                    "jpeg" | "jpg" => {
                        let rgb = final_image.to_rgb8();
                        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
                            std::fs::File::create(&output_path).unwrap(),
                            self.settings.crop.jpeg_quality,
                        );
                        encoder.encode(
                            rgb.as_raw(),
                            rgb.width(),
                            rgb.height(),
                            image::ExtendedColorType::Rgb8,
                        )
                    }
                    _ => final_image.save(&output_path),
                };

                match save_result {
                    Ok(_) => {
                        info!("Exported face {} to {}", face_idx + 1, output_path.display());
                        export_count += 1;
                    }
                    Err(e) => {
                        warn!("Failed to save face {}: {}", face_idx + 1, e);
                        error_count += 1;
                    }
                }
            }
        }

        // Update status
        if error_count == 0 {
            self.status_line = format!(
                "Successfully exported {} face{} to {}",
                export_count,
                if export_count == 1 { "" } else { "s" },
                output_dir.display()
            );
            self.last_error = None;
        } else {
            self.status_line = format!(
                "Exported {} face{}, {} error{}",
                export_count,
                if export_count == 1 { "" } else { "s" },
                error_count,
                if error_count == 1 { "" } else { "s" }
            );
            self.last_error = Some(format!("{} export error(s) occurred", error_count));
        }

        info!(
            "Export complete: {} succeeded, {} failed",
            export_count, error_count
        );
    }

    /// Paints the detection bounding boxes and landmarks over the preview image.
    fn paint_detections(&self, ui: &egui::Ui, image_rect: Rect, image_size: (u32, u32)) {
        let painter = ui.painter().with_clip_rect(image_rect);
        let scale_x = image_rect.width() / image_size.0 as f32;
        let scale_y = image_rect.height() / image_size.1 as f32;

        let bbox_stroke = Stroke::new(2.0, Color32::from_rgb(255, 145, 77));
        let landmark_color = Color32::from_rgb(82, 180, 255);
        let crop_stroke = Stroke::new(3.0, Color32::from_rgb(0, 255, 127));
        let selected_stroke = Stroke::new(4.0, Color32::from_rgb(100, 150, 255));

        for (index, det_with_quality) in self.preview.detections.iter().enumerate() {
            let bbox = &det_with_quality.detection.bbox;
            let is_selected = self.selected_faces.contains(&index);

            let top_left = pos2(
                image_rect.left() + bbox.x * scale_x,
                image_rect.top() + bbox.y * scale_y,
            );
            let size = vec2(bbox.width * scale_x, bbox.height * scale_y);
            let rect = Rect::from_min_size(top_left, size);

            // Draw selection highlight if selected
            if is_selected {
                painter.rect_stroke(rect, 2.0, selected_stroke, egui::StrokeKind::Outside);
            }

            // Draw detection bounding box
            painter.rect_stroke(rect, 0.0, bbox_stroke, egui::StrokeKind::Inside);

            for landmark in &det_with_quality.detection.landmarks {
                let center = pos2(
                    image_rect.left() + landmark.x * scale_x,
                    image_rect.top() + landmark.y * scale_y,
                );
                painter.circle_filled(center, 3.0, landmark_color);
            }

            // Paint crop region overlay if enabled
            if self.show_crop_overlay {
                let crop_settings = self.build_crop_settings();
                let crop_region = calculate_crop_region(
                    image_size.0,
                    image_size.1,
                    bbox.clone(),
                    &crop_settings,
                );
                let crop_top_left = pos2(
                    image_rect.left() + crop_region.x as f32 * scale_x,
                    image_rect.top() + crop_region.y as f32 * scale_y,
                );
                let crop_size = vec2(
                    crop_region.width as f32 * scale_x,
                    crop_region.height as f32 * scale_y,
                );
                let crop_rect = Rect::from_min_size(crop_top_left, crop_size);
                painter.rect_stroke(crop_rect, 4.0, crop_stroke, egui::StrokeKind::Inside);
            }
        }
    }

    /// Applies changes to the settings, invalidating the detector or cache as needed.
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

    /// Persists the current settings to disk and provides feedback to the user.
    fn persist_settings_with_feedback(&mut self) {
        if let Err(err) = self.persist_settings() {
            let message = format!("Failed to persist settings: {err}");
            warn!("{message}");
            self.last_error = Some(message);
        }
    }

    /// Saves the current settings to the JSON file.
    fn persist_settings(&self) -> Result<()> {
        if let Some(parent) = self.settings_path.parent()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create settings directory {}", parent.display())
            })?;
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

    /// Invalidates the current detector, forcing it to be rebuilt on the next detection run.
    fn invalidate_detector(&mut self) {
        if self.detector.is_some() {
            info!("Invalidating YuNet detector due to configuration change");
        }
        self.detector = None;
        self.clear_cache("Detector configuration changed. Re-run detection.");
    }

    /// Clears the detection cache and updates the status message.
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

    /// Applies the model path from the text input field.
    fn apply_model_path_input(&mut self) {
        let trimmed = self.model_path_input.trim().to_owned();
        self.model_path_input = trimmed.clone();
        self.update_model_path(if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        });
    }

    /// Updates the model path in the settings and invalidates the detector.
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

        if normalized.is_none()
            && let Some(default_path) = resolved.as_deref()
        {
            info!("Falling back to default model path: {}", default_path);
        }

        if self.settings.model_path != resolved {
            self.settings.model_path = resolved.clone();
            self.apply_settings_changes(true, false);
        }

        self.model_path_input = resolved.clone().unwrap_or_default();
        self.model_path_dirty = false;
    }

    /// Opens a file dialog to select an image.
    fn open_image_dialog(&mut self) {
        if let Some(path) = FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp"])
            .pick_file()
        {
            self.start_detection(path);
        }
    }

    /// Opens a file dialog to select an ONNX model.
    fn open_model_dialog(&mut self) {
        if let Some(path) = FileDialog::new()
            .add_filter("ONNX model", &["onnx"])
            .pick_file()
        {
            let display = path.to_string_lossy().to_string();
            self.update_model_path(Some(display));
        }
    }

    /// Opens a file dialog to select multiple images for batch processing.
    fn open_batch_dialog(&mut self) {
        if let Some(paths) = FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp"])
            .pick_files()
        {
            self.batch_files = paths
                .into_iter()
                .map(|path| BatchFile {
                    path,
                    status: BatchFileStatus::Pending,
                })
                .collect();

            self.batch_current_index = None;
            self.status_line = format!("Loaded {} images for batch processing", self.batch_files.len());
            info!("Loaded {} images for batch processing", self.batch_files.len());
        }
    }

    /// Starts batch export processing.
    fn start_batch_export(&mut self) {
        // Get output directory from user
        let output_dir = match FileDialog::new()
            .set_title("Select output directory for batch export")
            .pick_folder()
        {
            Some(dir) => dir,
            None => {
                info!("Batch export cancelled by user");
                return;
            }
        };

        let crop_settings = self.build_crop_settings();
        let enhancement_settings = self.build_enhancement_settings();
        let enhance_enabled = self.settings.enhance.enabled;
        let output_format = self.settings.crop.output_format.clone();
        let jpeg_quality = self.settings.crop.jpeg_quality;

        // Ensure detector is loaded before processing
        let detector = match self.ensure_detector() {
            Ok(d) => d,
            Err(e) => {
                self.last_error = Some(format!("Failed to load detector: {}", e));
                error!("Failed to load detector: {}", e);
                return;
            }
        };

        // Process each batch file
        for batch_file in &mut self.batch_files {
            batch_file.status = BatchFileStatus::Processing;

            // Load image
            let source_image = match load_image(&batch_file.path) {
                Ok(img) => img,
                Err(e) => {
                    batch_file.status = BatchFileStatus::Failed {
                        error: format!("Failed to load: {}", e),
                    };
                    warn!("Failed to load {}: {}", batch_file.path.display(), e);
                    continue;
                }
            };

            let detection_output = match detector.detect_image(&source_image) {
                Ok(output) => output,
                Err(e) => {
                    batch_file.status = BatchFileStatus::Failed {
                        error: format!("Detection failed: {}", e),
                    };
                    continue;
                }
            };

            let faces_detected = detection_output.detections.len();
            let mut faces_exported = 0;

            // Export all detected faces
            for (face_idx, detection) in detection_output.detections.iter().enumerate() {
                let bbox = &detection.bbox;

                // Calculate crop region
                let crop_region = calculate_crop_region(
                    source_image.width(),
                    source_image.height(),
                    bbox.clone(),
                    &crop_settings,
                );

                // Extract and resize crop
                let cropped = source_image.crop_imm(
                    crop_region.x,
                    crop_region.y,
                    crop_region.width,
                    crop_region.height,
                );

                let resized = cropped.resize_exact(
                    crop_settings.output_width,
                    crop_settings.output_height,
                    image::imageops::FilterType::Lanczos3,
                );

                // Apply enhancements if enabled
                let final_image = if enhance_enabled {
                    apply_enhancements(&resized, &enhancement_settings)
                } else {
                    resized
                };

                // Generate output filename
                let source_stem = batch_file.path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("face");
                let ext = match output_format.as_str() {
                    "jpeg" | "jpg" => "jpg",
                    "webp" => "webp",
                    _ => "png",
                };
                let output_filename = format!("{}_face_{:02}.{}", source_stem, face_idx + 1, ext);
                let output_path = output_dir.join(output_filename);

                // Save image
                let save_result = match output_format.as_str() {
                    "jpeg" | "jpg" => {
                        let rgb = final_image.to_rgb8();
                        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
                            std::fs::File::create(&output_path).unwrap(),
                            jpeg_quality,
                        );
                        encoder.encode(
                            rgb.as_raw(),
                            rgb.width(),
                            rgb.height(),
                            image::ExtendedColorType::Rgb8,
                        )
                    }
                    _ => final_image.save(&output_path),
                };

                if save_result.is_ok() {
                    faces_exported += 1;
                } else {
                    warn!("Failed to save face {} from {}", face_idx + 1, batch_file.path.display());
                }
            }

            batch_file.status = BatchFileStatus::Completed {
                faces_detected,
                faces_exported,
            };
        }

        // Update status with summary
        let total = self.batch_files.len();
        let completed = self.batch_files.iter()
            .filter(|f| matches!(f.status, BatchFileStatus::Completed { .. }))
            .count();
        let total_faces: usize = self.batch_files.iter()
            .filter_map(|f| match &f.status {
                BatchFileStatus::Completed { faces_exported, .. } => Some(*faces_exported),
                _ => None,
            })
            .sum();

        self.status_line = format!(
            "Batch export complete: {}/{} files processed, {} faces exported to {}",
            completed, total, total_faces, output_dir.display()
        );
        info!("Batch export complete: {}/{} files, {} faces exported", completed, total, total_faces);
    }

    /// Starts a new detection job for the given image path.
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

    /// Ensures that the detector is loaded, building it if necessary.
    fn ensure_detector(&mut self) -> Result<Arc<YuNetDetector>> {
        if let Some(detector) = &self.detector {
            return Ok(detector.clone());
        }

        let detector = Arc::new(build_detector(&self.settings)?);
        self.detector = Some(detector.clone());
        Ok(detector)
    }

    /// Creates a cache key for the given image path and current settings.
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

    /// Polls for completed detection jobs and updates the UI.
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

    /// Creates thumbnail textures for detected faces.
    fn create_thumbnails(
        &mut self,
        ctx: &EguiContext,
        detections: &mut [DetectionWithQuality],
    ) {
        // Load the source image to crop face thumbnails
        if let Some(path) = &self.preview.image_path {
            if let Ok(image) = load_image(path) {
                for (index, det) in detections.iter_mut().enumerate() {
                    let bbox = &det.detection.bbox;
                    let x = bbox.x.max(0.0) as u32;
                    let y = bbox.y.max(0.0) as u32;
                    let w = bbox.width.max(1.0) as u32;
                    let h = bbox.height.max(1.0) as u32;

                    // Clamp to image bounds
                    let img_w = image.width();
                    let img_h = image.height();
                    let x = x.min(img_w.saturating_sub(1));
                    let y = y.min(img_h.saturating_sub(1));
                    let w = w.min(img_w.saturating_sub(x));
                    let h = h.min(img_h.saturating_sub(y));

                    let face_region = image.crop_imm(x, y, w, h);
                    // Resize to thumbnail size (96x96)
                    let thumb = face_region.resize(96, 96, image::imageops::FilterType::Lanczos3);
                    let thumb_rgba = thumb.to_rgba8();
                    let thumb_size = [thumb_rgba.width() as usize, thumb_rgba.height() as usize];
                    let thumb_color = egui::ColorImage::from_rgba_unmultiplied(thumb_size, thumb_rgba.as_raw());

                    let texture_name = format!("yunet-face-thumb-{}-{}", self.texture_seq, index);
                    self.texture_seq = self.texture_seq.wrapping_add(1);
                    let texture = ctx.load_texture(texture_name, thumb_color, TextureOptions::LINEAR);
                    det.thumbnail = Some(texture);
                }
            }
        }
    }

    /// Handles a message from a detection job.
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
                    mut detections,
                    original_size,
                } = data;

                let texture_name = format!("yunet-image-preview-{}", self.texture_seq);
                self.texture_seq = self.texture_seq.wrapping_add(1);

                let texture = ctx.load_texture(texture_name, color_image, TextureOptions::LINEAR);
                let cache_texture = texture.clone();

                self.preview.texture = Some(texture);
                self.preview.image_size = Some(original_size);
                self.preview.image_path = Some(path.clone());

                // Create thumbnails for face regions
                self.create_thumbnails(ctx, &mut detections);

                let cached_detections = detections.clone();
                self.preview.detections = detections;

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

    /// Returns the next available job ID.
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

/// Loads application settings from a file, or returns default settings if loading fails.
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

/// Returns the default path for the GUI settings file.
fn default_settings_path() -> PathBuf {
    std::env::current_dir()
        .map(|dir| dir.join("config/gui_settings.json"))
        .unwrap_or_else(|_| PathBuf::from("config/gui_settings.json"))
}

/// Builds a `YuNetDetector` from the given application settings.
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

/// Performs face detection on an image and returns the results.
fn perform_detection(detector: Arc<YuNetDetector>, path: PathBuf) -> Result<DetectionJobSuccess> {
    let image = load_image(&path)
        .with_context(|| format!("failed to load image from {}", path.display()))?;
    let detection_output = detector
        .detect_image(&image)
        .with_context(|| format!("YuNet detection failed for {}", path.display()))?;

    // Calculate quality scores for each detected face
    let detections_with_quality: Vec<DetectionWithQuality> = detection_output
        .detections
        .into_iter()
        .map(|detection| {
            // Crop face region for quality analysis
            let bbox = &detection.bbox;
            let x = bbox.x.max(0.0) as u32;
            let y = bbox.y.max(0.0) as u32;
            let w = bbox.width.max(1.0) as u32;
            let h = bbox.height.max(1.0) as u32;

            // Clamp to image bounds
            let img_w = image.width();
            let img_h = image.height();
            let x = x.min(img_w.saturating_sub(1));
            let y = y.min(img_h.saturating_sub(1));
            let w = w.min(img_w.saturating_sub(x));
            let h = h.min(img_h.saturating_sub(y));

            let face_region = image.crop_imm(x, y, w, h);
            let (quality_score, quality) = estimate_sharpness(&face_region);

            DetectionWithQuality {
                detection,
                quality_score,
                quality,
                thumbnail: None, // Will be created on the GUI thread
            }
        })
        .collect();

    let rgba = image.to_rgba8();
    let size = [rgba.width() as usize, rgba.height() as usize];
    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());

    Ok(DetectionJobSuccess {
        path,
        color_image,
        detections: detections_with_quality,
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
