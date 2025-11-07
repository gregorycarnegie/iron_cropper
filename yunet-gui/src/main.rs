//! Desktop GUI for YuNet face detection.

mod theme;

use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap, HashSet},
    io::Cursor,
    path::{Path, PathBuf},
    sync::{Arc, mpsc},
};

use anyhow::{Context as AnyhowContext, Result, anyhow};
use eframe::{App, CreationContext, Frame, NativeOptions, egui};
use egui::{
    Align, Button, CentralPanel, Color32, ComboBox, Context as EguiContext, CornerRadius,
    DragValue, Key, Layout, Margin, ProgressBar, Rect, Response, Rgba, RichText, ScrollArea, Sense,
    SidePanel, Slider, Spinner, Stroke, TextureHandle, TextureOptions, TopBottomPanel, Ui,
    UiBuilder, pos2, vec2,
};
use egui_extras::{Column as TableColumn, TableBuilder};
use ico::IconDir;
use image::DynamicImage;
use log::{LevelFilter, error, info, warn};
use rayon::prelude::*;
use rfd::FileDialog;
use yunet_core::{
    BoundingBox, CropSettings as CoreCropSettings, Detection, Landmark, PositioningMode,
    PostprocessConfig, PreprocessConfig, YuNetDetector, calculate_crop_region, preset_by_name,
};
use yunet_utils::{
    CropShape, MetadataContext, OutputOptions, PolygonCornerStyle, append_suffix_to_filename,
    apply_shape_mask,
    config::{
        AppSettings, CropSettings as ConfigCropSettings, MetadataMode, default_settings_path,
    },
    configure_telemetry,
    enhance::{EnhancementSettings, apply_enhancements},
    init_logging, load_image,
    mapping::{
        ColumnSelector, MappingCatalog, MappingEntry, MappingFormat, MappingPreview,
        MappingReadOptions, detect_format as detect_mapping_format_utils, inspect_mapping_sources,
        load_mapping_entries, load_mapping_preview,
    },
    outline_points_for_rect,
    quality::{Quality, estimate_sharpness},
    save_dynamic_image,
};

/// Main entry point for the GUI application.
fn main() -> eframe::Result<()> {
    init_logging(log::LevelFilter::Info).expect("failed to initialize logging");
    let mut options = NativeOptions::default();
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

/// Status of a batch file being processed.
#[derive(Debug, Clone, PartialEq, Eq)]
enum BatchFileStatus {
    Pending,
    Processing,
    Completed {
        faces_detected: usize,
        faces_exported: usize,
    },
    Failed {
        error: String,
    },
}

/// A file in the batch processing queue.
#[derive(Debug, Clone)]
struct BatchFile {
    path: PathBuf,
    status: BatchFileStatus,
    output_override: Option<PathBuf>,
}

#[derive(Clone)]
struct BatchJobConfig {
    output_dir: PathBuf,
    crop_settings: CoreCropSettings,
    crop_config: ConfigCropSettings,
    enhancement_settings: EnhancementSettings,
    enhance_enabled: bool,
    output_options: OutputOptions,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct EnhancementSignature {
    auto_color: bool,
    exposure_bits: u32,
    brightness: i32,
    contrast_bits: u32,
    saturation_bits: u32,
    unsharp_amount_bits: u32,
    unsharp_radius_bits: u32,
    sharpness_bits: u32,
    skin_smooth_bits: u32,
    skin_sigma_space_bits: u32,
    skin_sigma_color_bits: u32,
    red_eye_removal: bool,
    red_eye_threshold_bits: u32,
    background_blur: bool,
    background_blur_radius_bits: u32,
    background_blur_mask_bits: u32,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct ShapeSignature {
    kind: u8,
    primary_bits: u32,
    secondary_bits: u32,
    sides: u8,
    rotation_bits: u32,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct CropPreviewKey {
    path: PathBuf,
    face_index: usize,
    output_width: u32,
    output_height: u32,
    positioning_mode: u8,
    face_height_bits: u32,
    horizontal_bits: u32,
    vertical_bits: u32,
    shape: ShapeSignature,
    enhancement: EnhancementSignature,
    enhance_enabled: bool,
}

struct CropPreviewCacheEntry {
    image: Arc<DynamicImage>,
    texture: Option<TextureHandle>,
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
    /// Cached cropped previews keyed by face + crop/enhancement configuration.
    crop_preview_cache: HashMap<CropPreviewKey, CropPreviewCacheEntry>,
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
    /// Undo/redo stack for crop configuration.
    crop_history: Vec<ConfigCropSettings>,
    crop_history_index: usize,
    /// Editable metadata tags input cached for the UI.
    metadata_tags_input: String,
    /// Batch mode state.
    batch_files: Vec<BatchFile>,
    /// Current index in batch processing.
    batch_current_index: Option<usize>,
    /// Mapping import workflow state.
    mapping: MappingUiState,
    /// Normalized anchor (0-1) describing the HUD offset inside the preview image.
    preview_hud_anchor: egui::Vec2,
    /// Whether the preview HUD content is collapsed.
    preview_hud_minimized: bool,
    /// Drag origin for HUD repositioning.
    preview_hud_drag_origin: Option<egui::Pos2>,
    /// Whether manual bounding-box draw mode is active.
    manual_box_tool_enabled: bool,
    /// In-progress manual bounding box draft in image coordinates.
    manual_box_draft: Option<ManualBoxDraft>,
    /// Currently active drag handle for adjusting a bounding box.
    active_bbox_drag: Option<ActiveBoxDrag>,
}

/// Indicates whether a bounding box originated from the detector or the user.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DetectionOrigin {
    Detector,
    Manual,
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
    /// Active bounding box (may diverge from the detector output).
    current_bbox: BoundingBox,
    /// Original bounding box (detector or initial manual draft).
    original_bbox: BoundingBox,
    /// Where this bounding box came from.
    origin: DetectionOrigin,
}

impl DetectionWithQuality {
    fn active_bbox(&self) -> BoundingBox {
        self.current_bbox
    }

    fn reset_bbox(&mut self) {
        self.current_bbox = self.original_bbox;
    }

    fn set_bbox(&mut self, bbox: BoundingBox) {
        self.current_bbox = bbox;
    }

    fn is_manual(&self) -> bool {
        matches!(self.origin, DetectionOrigin::Manual)
    }

    fn is_modified(&self) -> bool {
        self.current_bbox != self.original_bbox
    }
}

#[derive(Clone, Copy)]
struct ManualBoxDraft {
    start: egui::Pos2,
    current: egui::Pos2,
}

#[derive(Clone, Copy)]
struct ActiveBoxDrag {
    index: usize,
    handle: DragHandle,
    start_bbox: BoundingBox,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DragHandle {
    Move,
    NorthWest,
    NorthEast,
    SouthWest,
    SouthEast,
}

#[derive(Clone, Copy, Default)]
struct PointerSnapshot {
    pressed: bool,
    released: bool,
    down: bool,
    press_origin: Option<egui::Pos2>,
    pos: Option<egui::Pos2>,
}

impl PointerSnapshot {
    fn capture(ctx: &EguiContext) -> Self {
        ctx.input(|input| PointerSnapshot {
            pressed: input.pointer.primary_pressed(),
            released: input.pointer.primary_released(),
            down: input.pointer.primary_down(),
            press_origin: input.pointer.press_origin(),
            pos: input.pointer.interact_pos(),
        })
    }
}

#[derive(Clone, Copy)]
struct PreviewSpace {
    rect: Rect,
    image_size: (u32, u32),
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
    /// Whether the preview is currently loading/detecting.
    is_loading: bool,
    /// Cached original image pixels for computing previews without disk I/O.
    source_image: Option<Arc<DynamicImage>>,
}

struct MappingUiState {
    file_path: Option<PathBuf>,
    base_dir: Option<PathBuf>,
    detected_format: Option<MappingFormat>,
    format_override: Option<MappingFormat>,
    has_headers: bool,
    delimiter_input: String,
    sheet_name: String,
    sql_table: String,
    sql_query: String,
    catalog: MappingCatalog,
    preview: Option<MappingPreview>,
    preview_error: Option<String>,
    source_column_idx: Option<usize>,
    output_column_idx: Option<usize>,
    entries: Vec<MappingEntry>,
}

impl MappingUiState {
    fn new() -> Self {
        Self {
            file_path: None,
            base_dir: None,
            detected_format: None,
            format_override: None,
            has_headers: true,
            delimiter_input: ",".to_string(),
            sheet_name: String::new(),
            sql_table: String::new(),
            sql_query: String::new(),
            catalog: MappingCatalog::default(),
            preview: None,
            preview_error: None,
            source_column_idx: None,
            output_column_idx: None,
            entries: Vec::new(),
        }
    }

    fn set_file(&mut self, path: PathBuf) {
        self.file_path = Some(path);
        self.base_dir = self
            .file_path
            .as_ref()
            .and_then(|p| p.parent().map(|parent| parent.to_path_buf()));
        self.detected_format = self
            .file_path
            .as_ref()
            .map(|p| detect_mapping_format_utils(p));
        self.preview = None;
        self.preview_error = None;
        self.source_column_idx = None;
        self.output_column_idx = None;
        self.entries.clear();
        self.sheet_name.clear();
        self.sql_table.clear();
        self.sql_query.clear();
        self.delimiter_input = ",".to_string();
        self.refresh_catalog();
    }

    fn effective_format(&self) -> Option<MappingFormat> {
        self.format_override.or(self.detected_format)
    }

    fn refresh_catalog(&mut self) {
        if let Some(path) = &self.file_path {
            match inspect_mapping_sources(path, &self.read_options()) {
                Ok(catalog) => {
                    self.catalog = catalog;
                    if self.sheet_name.is_empty()
                        && let Some(first) = self.catalog.sheets.first()
                    {
                        self.sheet_name = first.clone();
                    }
                    if self.sql_table.is_empty()
                        && let Some(first) = self.catalog.sql_tables.first()
                    {
                        self.sql_table = first.clone();
                    }
                }
                Err(err) => {
                    self.preview_error = Some(err.to_string());
                }
            }
        } else {
            self.catalog = MappingCatalog::default();
        }
    }

    fn read_options(&self) -> MappingReadOptions {
        let mut options = MappingReadOptions {
            format: self.effective_format(),
            has_headers: Some(self.has_headers),
            delimiter: self.delimiter_input.chars().next().map(|c| c as u8),
            ..Default::default()
        };
        if !self.sheet_name.trim().is_empty() {
            options.sheet_name = Some(self.sheet_name.trim().to_string());
        }
        if !self.sql_table.trim().is_empty() {
            options.sql_table = Some(self.sql_table.trim().to_string());
        }
        if !self.sql_query.trim().is_empty() {
            options.sql_query = Some(self.sql_query.trim().to_string());
        }
        options
    }

    fn selected_column_name(&self, idx: Option<usize>) -> String {
        if let (Some(preview), Some(index)) = (self.preview.as_ref(), idx) {
            preview
                .columns
                .get(index)
                .cloned()
                .unwrap_or_else(|| format!("Column {}", index + 1))
        } else {
            "Select column".to_string()
        }
    }

    fn source_selector(&self) -> Option<ColumnSelector> {
        self.source_column_idx.map(ColumnSelector::Index)
    }

    fn output_selector(&self) -> Option<ColumnSelector> {
        self.output_column_idx.map(ColumnSelector::Index)
    }

    fn reload_preview(&mut self) -> Result<()> {
        let path = self
            .file_path
            .clone()
            .ok_or_else(|| anyhow!("Select a mapping file first"))?;
        match load_mapping_preview(&path, &self.read_options()) {
            Ok(preview) => {
                if self.source_column_idx.is_none() && !preview.columns.is_empty() {
                    self.source_column_idx = Some(0);
                }
                if self.output_column_idx.is_none() && preview.columns.len() > 1 {
                    self.output_column_idx = Some(1);
                }
                if self
                    .source_column_idx
                    .is_some_and(|idx| idx >= preview.columns.len())
                {
                    self.source_column_idx = None;
                }
                if self
                    .output_column_idx
                    .is_some_and(|idx| idx >= preview.columns.len())
                {
                    self.output_column_idx = None;
                }
                self.preview = Some(preview);
                self.preview_error = None;
                self.entries.clear();
                Ok(())
            }
            Err(err) => {
                self.preview_error = Some(err.to_string());
                self.preview = None;
                Err(err)
            }
        }
    }

    fn load_entries(&mut self) -> Result<()> {
        let path = self
            .file_path
            .clone()
            .ok_or_else(|| anyhow!("Select a mapping file first"))?;
        let source = self
            .source_selector()
            .ok_or_else(|| anyhow!("Select a source column"))?;
        let output = self
            .output_selector()
            .ok_or_else(|| anyhow!("Select an output column"))?;
        match load_mapping_entries(&path, &self.read_options(), &source, &output) {
            Ok(entries) => {
                self.entries = entries;
                self.preview_error = None;
                Ok(())
            }
            Err(err) => {
                self.preview_error = Some(err.to_string());
                Err(err)
            }
        }
    }
}

impl PreviewState {
    /// Resets the preview state to a loading state for a new image.
    fn begin_loading(&mut self, path: PathBuf) {
        self.image_path = Some(path);
        self.texture = None;
        self.image_size = None;
        self.detections.clear();
        self.is_loading = true;
        self.source_image = None;
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
    original_image: Arc<DynamicImage>,
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
    source_image: Arc<DynamicImage>,
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

        let crop_history = vec![settings.crop.clone()];
        let crop_history_index = crop_history.len() - 1;
        let metadata_tags_input = Self::format_metadata_tags(&settings.crop.metadata.custom_tags);

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
            crop_preview_cache: HashMap::new(),
            model_path_input,
            model_path_dirty: false,
            is_busy: false,
            texture_seq: 0,
            job_counter: 0,
            current_job: None,
            show_crop_overlay: true,
            selected_faces: HashSet::new(),
            crop_history,
            crop_history_index,
            metadata_tags_input,
            batch_files: Vec::new(),
            batch_current_index: None,
            mapping: MappingUiState::new(),
            preview_hud_anchor: vec2(0.02, 0.02),
            preview_hud_minimized: true,
            preview_hud_drag_origin: None,
            manual_box_tool_enabled: false,
            manual_box_draft: None,
            active_bbox_drag: None,
        }
    }

    /// Renders the top status bar with quick stats and actions.
    fn show_status_bar(&mut self, ctx: &EguiContext) {
        let palette = theme::palette();
        TopBottomPanel::top("yunet_status_bar")
            .frame(
                egui::Frame::new()
                    .fill(palette.panel_dark)
                    .stroke(Stroke::new(1.0, palette.outline))
                    .inner_margin(Margin::symmetric(20, 16)),
            )
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 6.0;
                    ui.horizontal(|ui| {
                        ui.heading(RichText::new("YuNet Studio").size(26.0).strong());
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            self.draw_status_badge(ui, palette);
                        });
                    });

                    ui.label(RichText::new(&self.status_line).color(palette.subtle_text));

                    if let Some(err) = &self.last_error {
                        ui.colored_label(palette.danger, err);
                    } else if self.preview.detections.is_empty() && !self.is_busy {
                        ui.label(
                            RichText::new("Choose an image to begin detecting faces.")
                                .color(palette.subtle_text),
                        );
                    }

                    ui.add_space(6.0);
                    self.draw_status_chips(ui, palette);
                    ui.add_space(10.0);
                    self.draw_quick_actions(ui, palette);
                });
            });
    }

    fn draw_status_badge(&self, ui: &mut Ui, palette: theme::Palette) {
        let (label, color) = if self.is_busy {
            ("Detecting...", palette.accent)
        } else if self.detector.is_none() {
            ("Model Required", palette.warning)
        } else {
            ("Ready", palette.success)
        };

        egui::Frame::new()
            .fill(palette.panel_light)
            .stroke(Stroke::new(1.0, color))
            .corner_radius(CornerRadius::same(64))
            .inner_margin(Margin::symmetric(14, 6))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if self.is_busy {
                        ui.add(Spinner::new().size(16.0));
                    }
                    ui.label(RichText::new(label).size(15.0).strong());
                });
            });
    }

    fn draw_status_chips(&self, ui: &mut Ui, palette: theme::Palette) {
        ui.horizontal_wrapped(|ui| {
            self.status_chip(
                ui,
                palette,
                format!("Faces {}", self.preview.detections.len()),
                palette.accent,
            );
            self.status_chip(
                ui,
                palette,
                format!("Selected {}", self.selected_faces.len()),
                if self.selected_faces.is_empty() {
                    palette.subtle_text
                } else {
                    palette.success
                },
            );

            self.status_chip(
                ui,
                palette,
                format!("Batch {}", self.batch_files.len()),
                if self.batch_files.is_empty() {
                    palette.subtle_text
                } else {
                    palette.warning
                },
            );
        });
    }

    fn draw_quick_actions(&mut self, ui: &mut Ui, palette: theme::Palette) {
        ui.horizontal_wrapped(|ui| {
            if self
                .quick_action_button(ui, palette, "Open Image", "Pick a single file", true)
                .clicked()
            {
                self.open_image_dialog();
            }
            if self
                .quick_action_button(ui, palette, "Load Batch", "Queue multiple images", true)
                .clicked()
            {
                self.open_batch_dialog();
            }

            let export_enabled = !self.selected_faces.is_empty();
            if self
                .quick_action_button(
                    ui,
                    palette,
                    "Export Selected",
                    "Sends crops to disk",
                    export_enabled,
                )
                .clicked()
            {
                self.export_selected_faces();
            }

            let batch_enabled = !self.batch_files.is_empty();
            let subtitle = format!("{} queued", self.batch_files.len());
            if self
                .quick_action_button(ui, palette, "Run Batch", &subtitle, batch_enabled)
                .clicked()
            {
                self.start_batch_export();
            }
        });
    }

    fn quick_action_button(
        &self,
        ui: &mut Ui,
        palette: theme::Palette,
        title: &str,
        subtitle: &str,
        enabled: bool,
    ) -> Response {
        let text = format!("{title}\n{subtitle}");
        ui.add_enabled(
            enabled,
            Button::new(RichText::new(text).size(15.0))
                .wrap()
                .min_size(vec2(150.0, 64.0))
                .fill(if enabled {
                    palette.panel_light
                } else {
                    palette.panel_dark
                })
                .stroke(Stroke::new(1.0, palette.outline))
                .corner_radius(CornerRadius::same(16)),
        )
    }

    fn status_chip(
        &self,
        ui: &mut Ui,
        palette: theme::Palette,
        text: impl Into<String>,
        accent: Color32,
    ) {
        egui::Frame::new()
            .fill(palette.panel_dark)
            .stroke(Stroke::new(1.0, accent))
            .corner_radius(CornerRadius::same(24))
            .inner_margin(Margin::symmetric(12, 4))
            .show(ui, |ui| {
                ui.label(
                    RichText::new(text.into())
                        .size(14.0)
                        .color(palette.subtle_text),
                );
            });
    }

    /// Renders the left-hand configuration panel.
    fn show_configuration_panel(&mut self, ctx: &EguiContext) {
        let palette = theme::palette();
        SidePanel::left("yunet_settings_panel")
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let initial_crop_settings = self.settings.crop.clone();
                        let initial_enhance_settings = self.settings.enhance.clone();
                        let mut settings_changed = false;
                        let mut enhancement_changed = false;
                        let mut preview_invalidated = false;
                        let mut metadata_tags_changed = false;
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
                        ui.horizontal(|ui| {
                            let enabled = self.preview.texture.is_some();
                            let label = if self.manual_box_tool_enabled {
                                "Exit draw mode"
                            } else {
                                "Draw bounding box"
                            };
                            if ui
                                .add_enabled(enabled, Button::new(label))
                                .clicked()
                            {
                                self.manual_box_tool_enabled = !self.manual_box_tool_enabled;
                                if !self.manual_box_tool_enabled {
                                    self.manual_box_draft = None;
                                }
                            }
                            if self.manual_box_tool_enabled {
                                ui.label(
                                    RichText::new("Click and drag in the preview area")
                                        .color(palette.subtle_text),
                                );
                            } else if !enabled {
                                ui.label(
                                    RichText::new("Load an image to draw boxes")
                                        .color(palette.subtle_text),
                                );
                            }
                        });
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

                            let mut pending_removal: Option<usize> = None;
                            ScrollArea::vertical()
                                .id_salt("detected_faces_scroll")
                                .max_height(220.0)
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    let len = self.preview.detections.len();
                                    for index in 0..len {
                                        if pending_removal.is_some() {
                                            break;
                                        }
                                        let (
                                            detection_score,
                                            quality,
                                            quality_score,
                                            thumbnail,
                                            is_manual,
                                            is_modified,
                                        ) = {
                                            let det = &self.preview.detections[index];
                                            (
                                                det.detection.score,
                                                det.quality,
                                                det.quality_score,
                                                det.thumbnail.clone(),
                                                det.is_manual(),
                                                det.is_modified(),
                                            )
                                        };
                                        let is_selected = self.selected_faces.contains(&index);

                                        let quality_color = match quality {
                                            Quality::High => Color32::from_rgb(0, 200, 100),
                                            Quality::Medium => Color32::from_rgb(255, 180, 0),
                                            Quality::Low => Color32::from_rgb(255, 80, 80),
                                        };

                                        let frame_fill = if is_selected {
                                            palette.panel_light
                                        } else {
                                            palette.panel_dark
                                        };
                                        let frame_stroke = if is_selected {
                                            Stroke::new(2.0, palette.accent)
                                        } else {
                                            Stroke::new(1.0, palette.outline)
                                        };

                                        let mut remove_requested = false;
                                        let response = egui::Frame::new()
                                            .fill(frame_fill)
                                            .stroke(frame_stroke)
                                            .corner_radius(CornerRadius::same(12))
                                            .inner_margin(Margin::symmetric(10, 8))
                                            .show(ui, |ui| {
                                                ui.horizontal(|ui| {
                                                        if let Some(texture) =
                                                            self.crop_preview_texture_for(
                                                                ctx, index,
                                                            )
                                                        {
                                                            let preview = egui::Image::new((
                                                                texture.id(),
                                                                texture.size_vec2(),
                                                            ))
                                                            .max_size(vec2(100.0, 100.0));
                                                            let preview_response =
                                                                ui.add(preview);
                                                            if preview_response.clicked() {
                                                                if is_selected {
                                                                    self.selected_faces
                                                                        .remove(&index);
                                                                } else {
                                                                    self.selected_faces
                                                                        .insert(index);
                                                                }
                                                            }
                                                        } else if let Some(thumbnail) = thumbnail {
                                                            let thumb_response = ui.add(
                                                                egui::Image::new((
                                                                    thumbnail.id(),
                                                                    thumbnail.size_vec2(),
                                                                ))
                                                                .max_width(80.0),
                                                            );
                                                            if thumb_response.clicked() {
                                                                if is_selected {
                                                                    self.selected_faces
                                                                        .remove(&index);
                                                                } else {
                                                                    self.selected_faces
                                                                        .insert(index);
                                                                }
                                                            }
                                                        }

                                                        ui.vertical(|ui| {
                                                            let title = if is_manual {
                                                                format!("Manual {}", index + 1)
                                                            } else {
                                                                format!("Face {}", index + 1)
                                                            };
                                                            ui.label(RichText::new(title).strong());
                                                            if is_manual {
                                                                ui.label(
                                                                    RichText::new("User box")
                                                                        .color(palette.accent),
                                                                );
                                                            } else if is_modified {
                                                                ui.label(
                                                                    RichText::new("Adjusted box")
                                                                        .color(palette.accent),
                                                                );
                                                            }
                                                            ui.label(format!(
                                                                "Conf: {:.2}",
                                                                detection_score
                                                            ));
                                                            ui.horizontal(|ui| {
                                                                ui.label("Quality:");
                                                                ui.colored_label(
                                                                    quality_color,
                                                                    format!(
                                                                        "{:?}",
                                                                        quality
                                                                    ),
                                                                );
                                                            });
                                                            ui.label(format!(
                                                                "Score: {:.0}",
                                                                quality_score
                                                            ));

                                                            ui.horizontal(|ui| {
                                                                if ui
                                                                    .small_button(if is_selected {
                                                                        "Deselect"
                                                                    } else {
                                                                        "Select"
                                                                    })
                                                                    .clicked()
                                                                {
                                                                    if is_selected {
                                                                        self.selected_faces
                                                                            .remove(&index);
                                                                    } else {
                                                                        self.selected_faces
                                                                            .insert(index);
                                                                    }
                                                                }

                                                                if ui.small_button("Reset box").clicked() {
                                                                    self.reset_detection_bbox(ctx, index);
                                                                }

                                                                if is_manual
                                                                    && ui.small_button("Remove").clicked()
                                                                {
                                                                    remove_requested = true;
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

                                        if remove_requested {
                                            pending_removal = Some(index);
                                        }
                                    }
                                });

                            if let Some(index) = pending_removal {
                                self.remove_detection(index);
                            }

                            ui.add_space(8.0);
                            ui.horizontal(|ui| {
                                if ui.button("Select All").clicked() {
                                    self.selected_faces =
                                        (0..self.preview.detections.len()).collect();
                                }
                                if ui.button("Deselect All").clicked() {
                                    self.selected_faces.clear();
                                }
                            });

                            ui.add_space(8.0);
                            ui.separator();

                            let num_selected = self.selected_faces.len();
                            if num_selected > 0 {
                                let button_label = format!(
                                    "Export {} Selected Face{}",
                                    num_selected,
                                    if num_selected == 1 { "" } else { "s" }
                                );
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
                            let completed = self
                                .batch_files
                                .iter()
                                .filter(|f| matches!(f.status, BatchFileStatus::Completed { .. }))
                                .count();
                            let failed = self
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
                                    RichText::new(format!("({} failed)", failed))
                                        .color(Color32::from_rgb(255, 80, 80)),
                                );
                            }

                            ui.add_space(6.0);
                            ScrollArea::vertical()
                                .id_salt("batch_files_scroll")
                                .max_height(200.0)
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    for (idx, batch_file) in self.batch_files.iter().enumerate() {
                                        let filename = batch_file
                                            .path
                                            .file_name()
                                            .and_then(|n| n.to_str())
                                            .unwrap_or("unknown");

                                        let (status_text, status_color, tooltip) =
                                            match &batch_file.status {
                                                BatchFileStatus::Pending => (
                                                    "Pending".to_string(),
                                                    Color32::GRAY,
                                                    None,
                                                ),
                                                BatchFileStatus::Processing => (
                                                    "Processing...".to_string(),
                                                    Color32::from_rgb(100, 150, 255),
                                                    None,
                                                ),
                                                BatchFileStatus::Completed {
                                                    faces_detected,
                                                    faces_exported,
                                                } => (
                                                    format!(
                                                        "{} faces, {} exported",
                                                        faces_detected, faces_exported
                                                    ),
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
                                                ui.label(
                                                    RichText::new("Mapping")
                                                        .color(palette.accent),
                                                );
                                            }
                                            let status_resp =
                                                ui.colored_label(status_color, status_text);
                                            if let Some(tip) = tooltip.as_deref() {
                                                status_resp.on_hover_text(tip);
                                            }
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
                        } else {
                            ui.separator();
                            ui.heading("Batch Processing");
                            ui.label("No batch files loaded.");
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
                            ui.label(
                                RichText::new("Press Enter to apply model path changes.").weak(),
                            );
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

                        ui.separator();
                        ui.heading("Diagnostics");
                        let mut telemetry_changed = false;
                        if ui
                            .checkbox(
                                &mut self.settings.telemetry.enabled,
                                "Enable telemetry logging",
                            )
                            .changed()
                        {
                            settings_changed = true;
                            telemetry_changed = true;
                        }

                        let level_options = [
                            (LevelFilter::Error, "Error"),
                            (LevelFilter::Warn, "Warn"),
                            (LevelFilter::Info, "Info"),
                            (LevelFilter::Debug, "Debug"),
                            (LevelFilter::Trace, "Trace"),
                        ];

                        ui.add_enabled_ui(self.settings.telemetry.enabled, |ui| {
                            let current_level = self.settings.telemetry.level_filter();
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
                                self.settings.telemetry.set_level(selected_level);
                                settings_changed = true;
                                telemetry_changed = true;
                            }
                            ui.add(
                                egui::Label::new("Timings are written to the application log.")
                                    .wrap(),
                            );
                        });

                        if telemetry_changed {
                            configure_telemetry(
                                self.settings.telemetry.enabled,
                                self.settings.telemetry.level_filter(),
                            );
                            if self.settings.telemetry.enabled {
                                info!(
                                    "Telemetry logging enabled (level={:?})",
                                    self.settings.telemetry.level_filter()
                                );
                            } else {
                                info!("Telemetry logging disabled");
                            }
                        }

                        if settings_changed {
                            self.apply_settings_changes(
                                requires_detector_reset,
                                requires_cache_refresh,
                            );
                        }

                        ui.separator();
                        ui.heading("Crop Settings");

                        ui.checkbox(&mut self.show_crop_overlay, "Show crop preview overlay")
                            .on_hover_text(
                                "Draws the proposed crop rectangle for every detected face.",
                            );
                        ui.add_space(6.0);

                        let preset_combo = egui::ComboBox::from_label("Preset")
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
                                    if ui
                                        .selectable_label(self.settings.crop.preset == value, label)
                                        .clicked()
                                        && self.settings.crop.preset != value
                                    {
                                        self.settings.crop.preset = value.to_string();
                                        if value != "custom"
                                            && let Some(preset) = preset_by_name(value)
                                        {
                                            self.settings.crop.output_width = preset.width;
                                            self.settings.crop.output_height = preset.height;
                                        }
                                        self.clear_crop_preview_cache();
                                        preview_invalidated = true;
                                        settings_changed = true;
                                    }
                                }
                            });
                        preset_combo
                            .response
                            .on_hover_text("Choose a predefined output size/aspect ratio.");

                        ui.add_space(4.0);
                        let (mut width, mut height) = self.resolved_output_dimensions();
                        let mut dimensions_changed = false;
                        ui.horizontal(|ui| {
                            ui.label("Width")
                                .on_hover_text("Export width in pixels for the crop.");
                            let response = ui
                                .add(DragValue::new(&mut width).range(64..=4096).speed(16.0))
                                .on_hover_text(
                                    "Drag or type to set the output width. Editing switches to the Custom preset.",
                                );
                            if response.changed() {
                                self.settings.crop.output_width = width;
                                dimensions_changed = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Height")
                                .on_hover_text("Export height in pixels for the crop.");
                            let response = ui
                                .add(DragValue::new(&mut height).range(64..=4096).speed(16.0))
                                .on_hover_text(
                                    "Drag or type to set the output height. Editing switches to the Custom preset.",
                                );
                            if response.changed() {
                                self.settings.crop.output_height = height;
                                dimensions_changed = true;
                            }
                        });
                        if dimensions_changed {
                            if self.settings.crop.preset != "custom" {
                                self.settings.crop.preset = "custom".to_string();
                            }
                            self.clear_crop_preview_cache();
                            preview_invalidated = true;
                            settings_changed = true;
                        }

                        ui.add_space(6.0);
                        if self.edit_shape_controls(ui) {
                            self.settings.crop.sanitize();
                            self.clear_crop_preview_cache();
                            preview_invalidated = true;
                            settings_changed = true;
                        }

                        ui.add_space(6.0);
                        let mut face_pct = self.settings.crop.face_height_pct;
                        if ui
                            .add(
                                Slider::new(&mut face_pct, 10.0..=100.0).text("Face height %"),
                            )
                            .on_hover_text(
                                "Controls how tall the detected face should be relative to the output height.",
                            )
                            .changed()
                        {
                            self.settings.crop.face_height_pct = face_pct;
                            settings_changed = true;
                        }

                        ui.add_space(6.0);
                        let positioning_combo = egui::ComboBox::from_label("Positioning mode")
                            .selected_text(&self.settings.crop.positioning_mode)
                            .show_ui(ui, |ui| {
                                let modes = [
                                    ("center", "Center"),
                                    ("rule-of-thirds", "Rule of Thirds"),
                                    ("custom", "Custom offsets"),
                                ];
                                for (value, label) in modes {
                                    if ui
                                        .selectable_label(
                                            self.settings.crop.positioning_mode == value,
                                            label,
                                        )
                                        .clicked()
                                    {
                                        self.settings.crop.positioning_mode = value.to_string();
                                        settings_changed = true;
                                    }
                                }
                            });
                        positioning_combo.response.on_hover_text(
                            "Adjusts how the crop is aligned around the detected face.",
                        );

                        if self.settings.crop.positioning_mode == "custom" {
                            ui.add_space(4.0);
                            let mut vert = self.settings.crop.vertical_offset;
                            if ui
                                .add(
                                    Slider::new(&mut vert, -1.0..=1.0)
                                        .text("Vertical offset"),
                                )
                                .on_hover_text(
                                    "Negative values move the crop up, positive values move it down.",
                                )
                                .changed()
                            {
                                self.settings.crop.vertical_offset = vert;
                                settings_changed = true;
                            }
                            let mut horiz = self.settings.crop.horizontal_offset;
                            if ui
                                .add(
                                    Slider::new(&mut horiz, -1.0..=1.0)
                                        .text("Horizontal offset"),
                                )
                                .on_hover_text(
                                    "Negative values move the crop left, positive values move it right.",
                                )
                                .changed()
                            {
                                self.settings.crop.horizontal_offset = horiz;
                                settings_changed = true;
                            }
                        }

                        ui.separator();
                        ui.label("Quality automation");
                        let mut auto_select =
                            self.settings.crop.quality_rules.auto_select_best_face;
                        if ui
                            .checkbox(&mut auto_select, "Auto-select highest quality face")
                            .changed()
                        {
                            self.settings.crop.quality_rules.auto_select_best_face = auto_select;
                            settings_changed = true;
                        }

                        let mut skip_no_high =
                            self.settings.crop.quality_rules.auto_skip_no_high_quality;
                        if ui
                            .checkbox(&mut skip_no_high, "Skip export when no high-quality faces")
                            .changed()
                        {
                            self.settings.crop.quality_rules.auto_skip_no_high_quality =
                                skip_no_high;
                            settings_changed = true;
                        }

                        let mut suffix_enabled = self.settings.crop.quality_rules.quality_suffix;
                        if ui
                            .checkbox(&mut suffix_enabled, "Append quality suffix to filenames")
                            .changed()
                        {
                            self.settings.crop.quality_rules.quality_suffix = suffix_enabled;
                            settings_changed = true;
                        }

                        ui.horizontal(|ui| {
                            ui.label("Minimum quality to export");
                            let current = self.settings.crop.quality_rules.min_quality;
                            let label = match current {
                                Some(Quality::Low) => "Low",
                                Some(Quality::Medium) => "Medium",
                                Some(Quality::High) => "High",
                                None => "Off",
                            };
                            egui::ComboBox::from_id_salt("min_quality_combo")
                                .selected_text(label)
                                .show_ui(ui, |ui| {
                                    if ui.selectable_label(current.is_none(), "Off").clicked() {
                                        self.settings.crop.quality_rules.min_quality = None;
                                        settings_changed = true;
                                    }
                                    if ui
                                        .selectable_label(current == Some(Quality::Low), "Low")
                                        .clicked()
                                    {
                                        self.settings.crop.quality_rules.min_quality =
                                            Some(Quality::Low);
                                        settings_changed = true;
                                    }
                                    if ui
                                        .selectable_label(
                                            current == Some(Quality::Medium),
                                            "Medium",
                                        )
                                        .clicked()
                                    {
                                        self.settings.crop.quality_rules.min_quality =
                                            Some(Quality::Medium);
                                        settings_changed = true;
                                    }
                                    if ui
                                        .selectable_label(current == Some(Quality::High), "High")
                                        .clicked()
                                    {
                                        self.settings.crop.quality_rules.min_quality =
                                            Some(Quality::High);
                                        settings_changed = true;
                                    }
                                });
                        });

                        ui.separator();
                        ui.label("Output format");
                        egui::ComboBox::from_id_salt("output_format_combo")
                            .selected_text(self.settings.crop.output_format.to_ascii_uppercase())
                            .show_ui(ui, |ui| {
                                for option in ["png", "jpeg", "webp"] {
                                    if ui
                                        .selectable_label(
                                            self.settings.crop.output_format == option,
                                            option.to_ascii_uppercase(),
                                        )
                                        .clicked()
                                    {
                                        self.settings.crop.output_format = option.to_string();
                                        settings_changed = true;
                                    }
                                }
                            });

                        let mut auto_detect = self.settings.crop.auto_detect_format;
                        if ui
                            .checkbox(&mut auto_detect, "Auto-detect format from file extension")
                            .changed()
                        {
                            self.settings.crop.auto_detect_format = auto_detect;
                            settings_changed = true;
                        }

                        ui.horizontal(|ui| {
                            ui.label("PNG compression");
                            let current = self.settings.crop.png_compression.to_ascii_lowercase();
                            let label = match current.as_str() {
                                "fast" => "Fast".to_string(),
                                "default" => "Default".to_string(),
                                "best" => "Best".to_string(),
                                other => format!("Custom ({other})"),
                            };
                            egui::ComboBox::from_id_salt("png_compression_combo")
                                .selected_text(label)
                                .show_ui(ui, |ui| {
                                    for (value, text) in
                                        [("fast", "Fast"), ("default", "Default"), ("best", "Best")]
                                    {
                                        if ui
                                            .selectable_label(
                                                self.settings
                                                    .crop
                                                    .png_compression
                                                    .eq_ignore_ascii_case(value),
                                                text,
                                            )
                                            .clicked()
                                        {
                                            self.settings.crop.png_compression = value.to_string();
                                            settings_changed = true;
                                        }
                                    }
                                });

                            let mut level = self
                                .settings
                                .crop
                                .png_compression
                                .parse::<i32>()
                                .unwrap_or(6);
                            let prev = level;
                            if ui
                                .add(
                                    egui::DragValue::new(&mut level)
                                        .range(0..=9)
                                        .prefix("Level "),
                                )
                                .changed()
                            {
                                level = level.clamp(0, 9);
                                if level != prev {
                                    self.settings.crop.png_compression = level.to_string();
                                    settings_changed = true;
                                }
                            }
                        });

                        let mut jpeg_quality = i32::from(self.settings.crop.jpeg_quality);
                        if ui
                            .add(Slider::new(&mut jpeg_quality, 1..=100).text("JPEG quality"))
                            .changed()
                        {
                            self.settings.crop.jpeg_quality = jpeg_quality as u8;
                            settings_changed = true;
                        }

                        let mut webp_quality = i32::from(self.settings.crop.webp_quality);
                        if ui
                            .add(Slider::new(&mut webp_quality, 0..=100).text("WebP quality"))
                            .changed()
                        {
                            self.settings.crop.webp_quality = webp_quality as u8;
                            settings_changed = true;
                        }

                        ui.separator();
                        ui.label("Metadata");
                        let mode_label = match self.settings.crop.metadata.mode {
                            MetadataMode::Preserve => "Preserve",
                            MetadataMode::Strip => "Strip",
                            MetadataMode::Custom => "Custom",
                        };
                        egui::ComboBox::from_id_salt("metadata_mode_combo")
                            .selected_text(mode_label)
                            .show_ui(ui, |ui| {
                                for (value, text) in [
                                    (MetadataMode::Preserve, "Preserve"),
                                    (MetadataMode::Strip, "Strip"),
                                    (MetadataMode::Custom, "Custom"),
                                ] {
                                    if ui
                                        .selectable_label(
                                            self.settings.crop.metadata.mode == value,
                                            text,
                                        )
                                        .clicked()
                                    {
                                        self.settings.crop.metadata.mode = value;
                                        settings_changed = true;
                                    }
                                }
                            });

                        let mut include_crop = self.settings.crop.metadata.include_crop_settings;
                        if ui
                            .checkbox(&mut include_crop, "Include crop settings metadata")
                            .changed()
                        {
                            self.settings.crop.metadata.include_crop_settings = include_crop;
                            settings_changed = true;
                        }

                        let mut include_quality =
                            self.settings.crop.metadata.include_quality_metrics;
                        if ui
                            .checkbox(&mut include_quality, "Include quality metrics metadata")
                            .changed()
                        {
                            self.settings.crop.metadata.include_quality_metrics = include_quality;
                            settings_changed = true;
                        }

                        if ui
                            .text_edit_multiline(&mut self.metadata_tags_input)
                            .changed()
                        {
                            self.settings.crop.metadata.custom_tags =
                                Self::parse_metadata_tags(&self.metadata_tags_input);
                            settings_changed = true;
                            metadata_tags_changed = true;
                        }
                        ui.label("Enter custom tags as key=value, one per line.")
                    .on_hover_text(
                        "Tags are embedded into output metadata when mode is preserve or custom.",
                    );

                        if settings_changed {
                            self.push_crop_history();
                            self.persist_settings_with_feedback();
                            self.apply_quality_rules_to_preview();
                            if !metadata_tags_changed {
                                self.refresh_metadata_tags_input();
                            }
                        }

                        ui.separator();
                        ui.heading("Enhancement Settings");

                        let enable_response =
                            ui.checkbox(&mut self.settings.enhance.enabled, "Enable enhancements");
                        if enable_response.changed() {
                            settings_changed = true;
                            enhancement_changed = true;
                        }
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
                                        if ui
                                            .selectable_label(
                                                self.settings.enhance.preset == value,
                                                label,
                                            )
                                            .clicked()
                                        {
                                            self.settings.enhance.preset = value.to_string();
                                            self.apply_enhancement_preset();
                                            settings_changed = true;
                                            enhancement_changed = true;
                                        }
                                    }
                                });

                            ui.add_space(6.0);
                            if ui
                                .checkbox(
                                    &mut self.settings.enhance.auto_color,
                                    "Auto color correction",
                                )
                                .changed()
                            {
                                settings_changed = true;
                                enhancement_changed = true;
                            }

                            ui.add_space(6.0);
                            let mut exp = self.settings.enhance.exposure_stops;
                            if ui
                                .add(Slider::new(&mut exp, -2.0..=2.0).text("Exposure (stops)"))
                                .changed()
                            {
                                self.settings.enhance.exposure_stops = exp;
                                settings_changed = true;
                                enhancement_changed = true;
                            }

                            let mut bright = self.settings.enhance.brightness;
                            if ui
                                .add(Slider::new(&mut bright, -100..=100).text("Brightness"))
                                .changed()
                            {
                                self.settings.enhance.brightness = bright;
                                settings_changed = true;
                                enhancement_changed = true;
                            }

                            let mut con = self.settings.enhance.contrast;
                            if ui
                                .add(Slider::new(&mut con, 0.5..=2.0).text("Contrast"))
                                .changed()
                            {
                                self.settings.enhance.contrast = con;
                                settings_changed = true;
                                enhancement_changed = true;
                            }

                            let mut sat = self.settings.enhance.saturation;
                            if ui
                                .add(Slider::new(&mut sat, 0.0..=2.5).text("Saturation"))
                                .changed()
                            {
                                self.settings.enhance.saturation = sat;
                                settings_changed = true;
                                enhancement_changed = true;
                            }

                            let mut sharp = self.settings.enhance.sharpness;
                            if ui
                                .add(Slider::new(&mut sharp, 0.0..=2.0).text("Sharpness"))
                                .changed()
                            {
                                self.settings.enhance.sharpness = sharp;
                                settings_changed = true;
                                enhancement_changed = true;
                            }

                            let mut skin_smooth = self.settings.enhance.skin_smooth;
                            if ui
                                .add(
                                    Slider::new(&mut skin_smooth, 0.0..=1.0).text("Skin Smoothing"),
                                )
                                .changed()
                            {
                                self.settings.enhance.skin_smooth = skin_smooth;
                                settings_changed = true;
                                enhancement_changed = true;
                            }

                            if ui
                                .checkbox(
                                    &mut self.settings.enhance.red_eye_removal,
                                    "Red-Eye Removal",
                                )
                                .changed()
                            {
                                settings_changed = true;
                                enhancement_changed = true;
                            }

                            if ui
                                .checkbox(
                                    &mut self.settings.enhance.background_blur,
                                    "Background Blur",
                                )
                                .changed()
                            {
                                settings_changed = true;
                                enhancement_changed = true;
                            }

                            ui.add_space(6.0);
                            if ui.button("Reset to defaults").clicked() {
                                self.settings.enhance =
                                    yunet_utils::config::EnhanceSettings::default();
                                settings_changed = true;
                                enhancement_changed = true;
                            }
                        }

                        if enhancement_changed {
                            self.clear_crop_preview_cache();
                            if !self.preview.detections.is_empty() {
                                for idx in 0..self.preview.detections.len() {
                                    let _ = self.crop_preview_texture_for(ctx, idx);
                                }
                            }
                            ctx.request_repaint();
                        }

                        if settings_changed {
                            self.persist_settings_with_feedback();
                        }

                        ui.add_space(8.0);
                        ui.small(
                            RichText::new(format!(
                                "Settings file: {}",
                                self.settings_path.display()
                            ))
                            .weak(),
                        );

                        let crop_changed = self.settings.crop != initial_crop_settings;
                        let enhance_changed_now =
                            self.settings.enhance != initial_enhance_settings;

                        if crop_changed && !preview_invalidated {
                            self.clear_crop_preview_cache();
                        }
                        if enhance_changed_now && !enhancement_changed {
                            self.clear_crop_preview_cache();
                        }

                        ui.separator();
                        ui.heading("Mapping Import");
                        self.show_mapping_panel(ui, palette);
                    });
            });
    }

    fn show_mapping_panel(&mut self, ui: &mut Ui, palette: theme::Palette) {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                if ui.button("Select mapping file…").clicked()
                    && let Some(path) = FileDialog::new()
                        .add_filter(
                            "Data files",
                            &[
                                "csv", "tsv", "txt", "xlsx", "xls", "parquet", "db", "sqlite",
                            ],
                        )
                        .pick_file()
                {
                    self.mapping.set_file(path.clone());
                    match self.mapping.reload_preview() {
                        Ok(_) => {
                            self.show_success(format!(
                                "Loaded mapping preview from {}",
                                path.display()
                            ));
                        }
                        Err(err) => {
                            self.show_error("Failed to load mapping preview", err.to_string());
                        }
                    }
                }
                if let Some(path) = &self.mapping.file_path {
                    ui.monospace(path.display().to_string());
                } else {
                    ui.label(RichText::new("No mapping file selected").color(palette.subtle_text));
                }
            });

            if self.mapping.file_path.is_none() {
                ui.label(
                    RichText::new("Import CSV, Excel, Parquet, or SQLite files to drive cropping.")
                        .color(palette.subtle_text),
                );
                return;
            }

            self.mapping_options_ui(ui, palette);

            if ui.button("Reload preview").clicked() {
                match self.mapping.reload_preview() {
                    Ok(_) => self.show_success("Mapping preview updated"),
                    Err(err) => {
                        self.show_error("Failed to load mapping preview", err.to_string());
                    }
                }
            }

            if let Some(err) = &self.mapping.preview_error {
                ui.colored_label(palette.danger, err);
            }

            if let Some(preview) = self.mapping.preview.clone() {
                self.mapping_preview_table(ui, palette, &preview);
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    let ready = self.mapping.source_selector().is_some()
                        && self.mapping.output_selector().is_some();
                    if ui
                        .add_enabled(ready, Button::new("Load mapping entries"))
                        .clicked()
                    {
                        match self.mapping.load_entries() {
                            Ok(_) => self.show_success(format!(
                                "Loaded {} mapping entries",
                                self.mapping.entries.len()
                            )),
                            Err(err) => {
                                self.show_error("Failed to load mapping entries", err.to_string());
                            }
                        }
                    }
                    if ui
                        .add_enabled(
                            !self.mapping.entries.is_empty(),
                            Button::new("Replace batch queue"),
                        )
                        .clicked()
                    {
                        self.push_mapping_to_batch();
                    }
                });

                if !self.mapping.entries.is_empty() {
                    ui.label(format!(
                        "{} mapping row(s) ready for batch export",
                        self.mapping.entries.len()
                    ));
                    for entry in self.mapping.entries.iter().take(3) {
                        ui.monospace(format!("{} → {}", entry.source_path, entry.output_name));
                    }
                    if self.mapping.entries.len() > 3 {
                        ui.label(RichText::new("…").color(palette.subtle_text));
                    }
                }
            } else {
                ui.label(
                    RichText::new("Preview will appear here after loading.")
                        .color(palette.subtle_text),
                );
            }
        });
    }

    fn mapping_options_ui(&mut self, ui: &mut Ui, palette: theme::Palette) {
        let prev_format = self.mapping.effective_format();
        egui::ComboBox::from_label("Format")
            .selected_text(
                self.mapping
                    .format_override
                    .map(|f| f.display_name().to_string())
                    .or_else(|| {
                        self.mapping
                            .detected_format
                            .map(|f| format!("Auto ({})", f.display_name()))
                    })
                    .unwrap_or_else(|| "Auto".to_string()),
            )
            .show_ui(ui, |ui| {
                let auto_label = self
                    .mapping
                    .detected_format
                    .map(|f| format!("Auto ({})", f.display_name()))
                    .unwrap_or_else(|| "Auto".to_string());
                if ui
                    .selectable_label(self.mapping.format_override.is_none(), auto_label)
                    .clicked()
                {
                    self.mapping.format_override = None;
                }
                for format in [
                    MappingFormat::Csv,
                    MappingFormat::Excel,
                    MappingFormat::Parquet,
                    MappingFormat::Sqlite,
                ] {
                    if ui
                        .selectable_label(
                            self.mapping.format_override == Some(format),
                            format.display_name(),
                        )
                        .clicked()
                    {
                        self.mapping.format_override = Some(format);
                    }
                }
            });

        let new_format = self.mapping.effective_format();
        if new_format != prev_format {
            self.mapping.refresh_catalog();
        }

        ui.checkbox(&mut self.mapping.has_headers, "First row contains headers");

        if matches!(new_format, Some(MappingFormat::Csv)) || new_format.is_none() {
            ui.horizontal(|ui| {
                ui.label("Delimiter");
                let response = ui.text_edit_singleline(&mut self.mapping.delimiter_input);
                if response.changed() {
                    if self.mapping.delimiter_input.is_empty() {
                        self.mapping.delimiter_input.push(',');
                    } else {
                        let first = self.mapping.delimiter_input.chars().next().unwrap();
                        self.mapping.delimiter_input = first.to_string();
                    }
                }
                ui.label(
                    RichText::new("Only the first character is used.").color(palette.subtle_text),
                );
            });
        }

        if matches!(new_format, Some(MappingFormat::Excel)) {
            if self.mapping.catalog.sheets.is_empty() {
                ui.label(
                    RichText::new("Reload preview to discover sheet names.")
                        .color(palette.subtle_text),
                );
            } else {
                ComboBox::from_label("Sheet")
                    .selected_text(if self.mapping.sheet_name.is_empty() {
                        self.mapping
                            .catalog
                            .sheets
                            .first()
                            .cloned()
                            .unwrap_or_else(|| "Sheet1".to_string())
                    } else {
                        self.mapping.sheet_name.clone()
                    })
                    .show_ui(ui, |ui| {
                        for sheet in &self.mapping.catalog.sheets {
                            if ui
                                .selectable_label(self.mapping.sheet_name == *sheet, sheet.clone())
                                .clicked()
                            {
                                self.mapping.sheet_name = sheet.clone();
                            }
                        }
                    });
            }
        }

        if matches!(new_format, Some(MappingFormat::Sqlite)) {
            if self.mapping.catalog.sql_tables.is_empty() {
                ui.label(
                    RichText::new("Reload preview to discover tables.").color(palette.subtle_text),
                );
            } else {
                ComboBox::from_label("Table")
                    .selected_text(if self.mapping.sql_table.is_empty() {
                        self.mapping
                            .catalog
                            .sql_tables
                            .first()
                            .cloned()
                            .unwrap_or_else(|| "main".to_string())
                    } else {
                        self.mapping.sql_table.clone()
                    })
                    .show_ui(ui, |ui| {
                        for table in &self.mapping.catalog.sql_tables {
                            if ui
                                .selectable_label(self.mapping.sql_table == *table, table.clone())
                                .clicked()
                            {
                                self.mapping.sql_table = table.clone();
                            }
                        }
                    });
            }
            ui.label("Custom SQL query (optional):");
            ui.text_edit_multiline(&mut self.mapping.sql_query);
        }
    }

    fn mapping_preview_table(
        &mut self,
        ui: &mut Ui,
        palette: theme::Palette,
        preview: &MappingPreview,
    ) {
        ui.add_space(8.0);
        let truncated_note = if preview.truncated {
            " (truncated)"
        } else {
            ""
        };
        ui.label(format!(
            "Previewing {} of {} rows{}",
            preview.rows.len(),
            preview.total_rows,
            truncated_note
        ));

        if preview.columns.is_empty() {
            ui.label(
                RichText::new("No columns detected in the mapping file.").color(palette.danger),
            );
            return;
        }

        ui.horizontal(|ui| {
            ComboBox::from_label("Source column")
                .selected_text(
                    self.mapping
                        .selected_column_name(self.mapping.source_column_idx),
                )
                .show_ui(ui, |ui| {
                    for (idx, name) in preview.columns.iter().enumerate() {
                        if ui
                            .selectable_label(self.mapping.source_column_idx == Some(idx), name)
                            .clicked()
                        {
                            self.mapping.source_column_idx = Some(idx);
                        }
                    }
                });

            ComboBox::from_label("Output column")
                .selected_text(
                    self.mapping
                        .selected_column_name(self.mapping.output_column_idx),
                )
                .show_ui(ui, |ui| {
                    for (idx, name) in preview.columns.iter().enumerate() {
                        if ui
                            .selectable_label(self.mapping.output_column_idx == Some(idx), name)
                            .clicked()
                        {
                            self.mapping.output_column_idx = Some(idx);
                        }
                    }
                });
        });

        let mut table = TableBuilder::new(ui)
            .striped(true)
            .auto_shrink([false, false]);
        for _ in &preview.columns {
            table = table.column(TableColumn::auto().resizable(true));
        }

        table
            .header(24.0, |mut header| {
                for (idx, name) in preview.columns.iter().enumerate() {
                    header.col(|ui| {
                        let mut text = RichText::new(name.clone()).strong();
                        if self.mapping.source_column_idx == Some(idx) {
                            text = text.color(palette.accent);
                        }
                        if self.mapping.output_column_idx == Some(idx) {
                            text = text.color(palette.success);
                        }
                        ui.label(text);
                    });
                }
            })
            .body(|body| {
                body.rows(20.0, preview.rows.len(), |mut row| {
                    let row_idx = row.index();
                    if let Some(values) = preview.rows.get(row_idx) {
                        for value in values {
                            row.col(|ui| {
                                ui.label(value);
                            });
                        }
                    }
                });
            });

        if let (Some(src_idx), Some(out_idx)) = (
            self.mapping.source_column_idx,
            self.mapping.output_column_idx,
        ) {
            ui.add_space(6.0);
            ui.label("Preview mappings:");
            for sample in preview.rows.iter().take(3) {
                if let (Some(src), Some(dest)) = (sample.get(src_idx), sample.get(out_idx)) {
                    let display = Self::display_with_output_ext(
                        dest,
                        self.settings.crop.output_format.as_str(),
                    );
                    ui.monospace(format!("{src} → {display}"));
                }
            }
        }
    }

    fn push_mapping_to_batch(&mut self) {
        if self.mapping.entries.is_empty() {
            self.show_error(
                "No mapping entries loaded",
                "Load entries before queuing batch exports.",
            );
            return;
        }

        let mut files = Vec::with_capacity(self.mapping.entries.len());
        for entry in &self.mapping.entries {
            let raw = PathBuf::from(&entry.source_path);
            let path = if raw.is_absolute() {
                raw
            } else if let Some(base) = &self.mapping.base_dir {
                base.join(raw)
            } else {
                raw
            };
            if !path.exists() {
                warn!(
                    "Mapping entry skipped—source {} not found",
                    entry.source_path
                );
                continue;
            }
            files.push(BatchFile {
                path,
                status: BatchFileStatus::Pending,
                output_override: Some(PathBuf::from(&entry.output_name)),
            });
        }

        if files.is_empty() {
            self.show_error(
                "Mapping entries invalid",
                "All mapping rows referenced missing files.",
            );
            return;
        }

        self.batch_files = files;
        self.batch_current_index = None;
        self.show_success(format!(
            "Queued {} mapping row(s) for batch export",
            self.batch_files.len()
        ));
        info!(
            "Loaded {} mapping rows into batch processing queue",
            self.batch_files.len()
        );
    }

    /// Renders the main image preview panel.
    fn show_preview(&mut self, ctx: &EguiContext) {
        let palette = theme::palette();
        CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(palette.canvas)
                    .inner_margin(Margin::symmetric(16, 16)),
            )
            .show(ctx, |ui| {
                if let Some(texture) = self.preview.texture.clone() {
                    let image_dimensions = self.preview.image_size;
                    let available = ui.available_size();
                    if available.x > 0.0 && available.y > 0.0 {
                        let tex_size = texture.size_vec2();
                        if tex_size.x > 0.0 && tex_size.y > 0.0 {
                            let scale = (available.x / tex_size.x)
                                .min(available.y / tex_size.y)
                                .max(0.0);
                            let scale = if scale.is_finite() && scale > 0.0 {
                                scale
                            } else {
                                1.0
                            };
                            let scaled = tex_size * scale;
                            ui.centered_and_justified(|ui| {
                                let image_widget =
                                    egui::Image::new(&texture).fit_to_exact_size(scaled);
                                let response = ui.add(image_widget);
                                if let Some(dimensions) = image_dimensions {
                                    let image_rect =
                                        Rect::from_center_size(response.rect.center(), scaled);
                                    self.handle_preview_interactions(
                                        ctx, ui, image_rect, dimensions,
                                    );
                                    self.paint_detections(ui, image_rect, dimensions);
                                    self.preview_overlay(ui, image_rect, palette);
                                }
                            });
                        }
                    }
                } else if self.preview.is_loading {
                    ui.vertical_centered(|ui| {
                        ui.add_space(64.0);
                        ui.add(Spinner::new().size(28.0));
                        ui.label(
                            RichText::new("Loading image and running detection...").size(16.0),
                        );
                    });
                } else {
                    ui.vertical_centered(|ui| {
                        ui.add_space(64.0);
                        ui.heading("Drop an image or pick one from Quick Actions.");
                        ui.label("The preview area will light up once detection finishes.");
                    });
                }
            });
    }

    fn preview_overlay(&mut self, ui: &mut Ui, image_rect: Rect, palette: theme::Palette) {
        let overlay_size = self.preview_hud_size();
        if image_rect.width() < overlay_size.x || image_rect.height() < overlay_size.y {
            return;
        }

        let mut overlay_rect = self.preview_hud_rect(image_rect, overlay_size);
        let overlay_id = ui.make_persistent_id("preview_hud_overlay");
        let drag_response = ui.interact(overlay_rect, overlay_id, Sense::drag());

        if drag_response.drag_started() {
            self.preview_hud_drag_origin = Some(overlay_rect.left_top());
        }

        if let Some(origin) = self.preview_hud_drag_origin {
            let desired = origin + drag_response.drag_delta();
            overlay_rect = self.update_hud_anchor_from_top_left(image_rect, overlay_size, desired);
        }

        if self.preview_hud_drag_origin.is_some() && !ui.ctx().input(|i| i.pointer.primary_down()) {
            self.preview_hud_drag_origin = None;
        }

        let hovered = drag_response.hovered() || drag_response.dragged();
        let fill = if hovered {
            palette.panel_dark
        } else {
            translucent_color(palette.panel_dark, 0.7)
        };

        ui.scope_builder(UiBuilder::new().max_rect(overlay_rect), |overlay_ui| {
            egui::Frame::new()
                .fill(fill)
                .stroke(Stroke::new(1.0, palette.outline))
                .corner_radius(CornerRadius::same(16))
                .inner_margin(Margin::symmetric(14, 10))
                .show(overlay_ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Preview HUD").strong());
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            let label = if self.preview_hud_minimized {
                                "Expand"
                            } else {
                                "Minimize"
                            };
                            if ui.small_button(label).clicked() {
                                self.preview_hud_minimized = !self.preview_hud_minimized;
                                ui.ctx().request_repaint();
                            }
                        });
                    });

                    if self.preview_hud_minimized {
                        ui.add_space(4.0);
                        ui.label(format!(
                            "Faces: {}  |  Selected: {}",
                            self.preview.detections.len(),
                            self.selected_faces.len()
                        ));
                        ui.label(
                            RichText::new("Drag to reposition / expand for actions")
                                .color(palette.subtle_text),
                        );
                        return;
                    }

                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        self.status_chip(
                            ui,
                            palette,
                            format!("Faces {}", self.preview.detections.len()),
                            palette.accent,
                        );
                        self.status_chip(
                            ui,
                            palette,
                            format!("Selected {}", self.selected_faces.len()),
                            if self.selected_faces.is_empty() {
                                palette.subtle_text
                            } else {
                                palette.success
                            },
                        );
                    });

                    ui.add_space(6.0);
                    self.quality_legend(ui, palette);
                    ui.add_space(6.0);

                    if self.preview.detections.is_empty() {
                        ui.label(RichText::new("No faces detected yet.").weak());
                    } else {
                        ui.checkbox(&mut self.show_crop_overlay, "Show crop guides");
                    }

                    ui.add_space(6.0);
                    let selected = self.selected_faces.len();
                    if selected > 0 {
                        let button_label = format!(
                            "Export {} face{}",
                            selected,
                            if selected == 1 { "" } else { "s" }
                        );
                        if ui.button(button_label).clicked() {
                            self.export_selected_faces();
                        }
                    } else {
                        ui.label(
                            RichText::new("Select faces from the list to export.")
                                .color(palette.subtle_text),
                        );
                    }
                });
        });
    }

    fn preview_hud_size(&self) -> egui::Vec2 {
        if self.preview_hud_minimized {
            vec2(220.0, 72.0)
        } else {
            vec2(260.0, 190.0)
        }
    }

    fn preview_hud_rect(&self, image_rect: Rect, overlay_size: egui::Vec2) -> Rect {
        let available_width = (image_rect.width() - overlay_size.x).max(0.0);
        let available_height = (image_rect.height() - overlay_size.y).max(0.0);
        let anchor_x = self.preview_hud_anchor.x.clamp(0.0, 1.0);
        let anchor_y = self.preview_hud_anchor.y.clamp(0.0, 1.0);
        let top_left = pos2(
            image_rect.left() + anchor_x * available_width,
            image_rect.top() + anchor_y * available_height,
        );
        Rect::from_min_size(top_left, overlay_size)
    }

    fn update_hud_anchor_from_top_left(
        &mut self,
        image_rect: Rect,
        overlay_size: egui::Vec2,
        desired_top_left: egui::Pos2,
    ) -> Rect {
        let clamped = self.clamp_hud_top_left(image_rect, overlay_size, desired_top_left);
        self.preview_hud_anchor = self.anchor_from_top_left(image_rect, overlay_size, clamped);
        Rect::from_min_size(clamped, overlay_size)
    }

    fn clamp_hud_top_left(
        &self,
        image_rect: Rect,
        overlay_size: egui::Vec2,
        desired: egui::Pos2,
    ) -> egui::Pos2 {
        let min_x = image_rect.left();
        let min_y = image_rect.top();
        let max_x = (image_rect.right() - overlay_size.x).max(min_x);
        let max_y = (image_rect.bottom() - overlay_size.y).max(min_y);
        pos2(desired.x.clamp(min_x, max_x), desired.y.clamp(min_y, max_y))
    }

    fn anchor_from_top_left(
        &self,
        image_rect: Rect,
        overlay_size: egui::Vec2,
        top_left: egui::Pos2,
    ) -> egui::Vec2 {
        let denom_x = (image_rect.width() - overlay_size.x).max(1.0);
        let denom_y = (image_rect.height() - overlay_size.y).max(1.0);
        vec2(
            ((top_left.x - image_rect.left()) / denom_x).clamp(0.0, 1.0),
            ((top_left.y - image_rect.top()) / denom_y).clamp(0.0, 1.0),
        )
    }

    fn handle_preview_interactions(
        &mut self,
        ctx: &EguiContext,
        ui: &mut Ui,
        image_rect: Rect,
        image_size: (u32, u32),
    ) {
        self.update_manual_box_draft(ctx, image_rect, image_size);
        self.handle_bbox_drag_interactions(ctx, ui, image_rect, image_size);
    }

    fn update_manual_box_draft(
        &mut self,
        ctx: &EguiContext,
        image_rect: Rect,
        image_size: (u32, u32),
    ) {
        if !self.manual_box_tool_enabled {
            self.manual_box_draft = None;
            return;
        }

        let pointer = PointerSnapshot::capture(ctx);
        if pointer.pressed
            && let Some(origin) = pointer.press_origin
            && image_rect.contains(origin)
            && !self.pointer_over_any_bbox(image_rect, image_size, origin)
            && let Some(image_pos) = self.screen_pos_to_image(origin, image_rect, image_size)
        {
            self.manual_box_draft = Some(ManualBoxDraft {
                start: image_pos,
                current: image_pos,
            });
        }

        if let Some(mut draft) = self.manual_box_draft {
            if pointer.down
                && let Some(pos) = pointer
                    .pos
                    .and_then(|p| self.screen_pos_to_image(p, image_rect, image_size))
            {
                draft.current = pos;
            }

            if pointer.released {
                if let Some(bbox) = self.draft_to_bbox(draft, image_size) {
                    self.add_manual_detection(ctx, bbox);
                    self.status_line = "Manual bounding box added.".to_string();
                }
                self.manual_box_draft = None;
            } else {
                self.manual_box_draft = Some(draft);
            }
        }
    }

    fn handle_bbox_drag_interactions(
        &mut self,
        ctx: &EguiContext,
        ui: &mut Ui,
        image_rect: Rect,
        image_size: (u32, u32),
    ) {
        let len = self.preview.detections.len();
        let preview_space = PreviewSpace {
            rect: image_rect,
            image_size,
        };
        for index in 0..len {
            let bbox = self.preview.detections[index].active_bbox();
            let rect = self.bbox_to_screen_rect(bbox, image_rect, image_size);
            self.handle_drag_control(ctx, ui, rect, index, preview_space, DragHandle::Move);

            for handle in [
                DragHandle::NorthWest,
                DragHandle::NorthEast,
                DragHandle::SouthWest,
                DragHandle::SouthEast,
            ] {
                let handle_rect = self.handle_rect_for(rect, handle);
                self.handle_drag_control(ctx, ui, handle_rect, index, preview_space, handle);
            }
        }
    }

    fn handle_drag_control(
        &mut self,
        ctx: &EguiContext,
        ui: &mut Ui,
        control_rect: Rect,
        index: usize,
        preview_space: PreviewSpace,
        handle: DragHandle,
    ) {
        let id = ui
            .id()
            .with(("bbox_handle", index, Self::drag_handle_id(handle)));
        let response = ui.interact(control_rect, id, Sense::drag());

        if response.drag_started() {
            let start_bbox = self.preview.detections[index].active_bbox();
            self.active_bbox_drag = Some(ActiveBoxDrag {
                index,
                handle,
                start_bbox,
            });
        }

        if let Some(active) = self.active_bbox_drag
            && active.index == index
            && active.handle == handle
            && response.dragged()
        {
            self.apply_active_drag(
                active,
                response.drag_delta(),
                preview_space.rect,
                preview_space.image_size,
            );
        }

        let drag_active = matches!(
            self.active_bbox_drag,
            Some(active) if active.index == index && active.handle == handle
        );
        if drag_active
            && !ctx.input(|i| i.pointer.primary_down())
            && let Some(active) = self.active_bbox_drag.take()
        {
            self.on_bbox_drag_finished(ctx, active.index);
        }
    }

    fn drag_handle_id(handle: DragHandle) -> u8 {
        match handle {
            DragHandle::Move => 0,
            DragHandle::NorthWest => 1,
            DragHandle::NorthEast => 2,
            DragHandle::SouthWest => 3,
            DragHandle::SouthEast => 4,
        }
    }

    fn handle_rect_for(&self, rect: Rect, handle: DragHandle) -> Rect {
        let size = 12.0;
        let center = match handle {
            DragHandle::NorthWest => pos2(rect.left(), rect.top()),
            DragHandle::NorthEast => pos2(rect.right(), rect.top()),
            DragHandle::SouthWest => pos2(rect.left(), rect.bottom()),
            DragHandle::SouthEast => pos2(rect.right(), rect.bottom()),
            DragHandle::Move => rect.center(),
        };
        Rect::from_center_size(center, vec2(size, size))
    }

    fn apply_active_drag(
        &mut self,
        active: ActiveBoxDrag,
        drag_delta: egui::Vec2,
        image_rect: Rect,
        image_size: (u32, u32),
    ) {
        if image_size.0 == 0 || image_size.1 == 0 {
            return;
        }

        let scale_x = image_rect.width() / image_size.0 as f32;
        let scale_y = image_rect.height() / image_size.1 as f32;
        if scale_x <= 0.0 || scale_y <= 0.0 {
            return;
        }

        let mut bbox = active.start_bbox;
        let delta_x = drag_delta.x / scale_x;
        let delta_y = drag_delta.y / scale_y;

        match active.handle {
            DragHandle::Move => {
                bbox.x += delta_x;
                bbox.y += delta_y;
            }
            DragHandle::NorthWest => {
                bbox.x += delta_x;
                bbox.y += delta_y;
                bbox.width -= delta_x;
                bbox.height -= delta_y;
            }
            DragHandle::NorthEast => {
                bbox.y += delta_y;
                bbox.width += delta_x;
                bbox.height -= delta_y;
            }
            DragHandle::SouthWest => {
                bbox.x += delta_x;
                bbox.width -= delta_x;
                bbox.height += delta_y;
            }
            DragHandle::SouthEast => {
                bbox.width += delta_x;
                bbox.height += delta_y;
            }
        }

        let clamped = self.clamp_bbox_to_image(bbox, image_size);
        if let Some(det) = self.preview.detections.get_mut(active.index) {
            det.set_bbox(clamped);
        }
    }

    fn on_bbox_drag_finished(&mut self, ctx: &EguiContext, index: usize) {
        self.clear_crop_preview_cache();
        self.refresh_detection_thumbnail_at(ctx, index);
    }

    fn pointer_over_any_bbox(
        &self,
        image_rect: Rect,
        image_size: (u32, u32),
        pointer: egui::Pos2,
    ) -> bool {
        if !image_rect.contains(pointer) {
            return false;
        }
        self.preview.detections.iter().any(|det| {
            self.bbox_to_screen_rect(det.active_bbox(), image_rect, image_size)
                .expand(8.0)
                .contains(pointer)
        })
    }

    fn screen_pos_to_image(
        &self,
        pos: egui::Pos2,
        image_rect: Rect,
        image_size: (u32, u32),
    ) -> Option<egui::Pos2> {
        if image_size.0 == 0 || image_size.1 == 0 || image_rect.width() <= 0.0 {
            return None;
        }
        let rel_x = ((pos.x - image_rect.left()) / image_rect.width()).clamp(0.0, 1.0);
        let rel_y = ((pos.y - image_rect.top()) / image_rect.height()).clamp(0.0, 1.0);
        Some(pos2(
            rel_x * image_size.0 as f32,
            rel_y * image_size.1 as f32,
        ))
    }

    fn image_point_to_screen(
        &self,
        point: egui::Pos2,
        image_rect: Rect,
        image_size: (u32, u32),
    ) -> egui::Pos2 {
        if image_size.0 == 0 || image_size.1 == 0 {
            return pos2(image_rect.left(), image_rect.top());
        }
        let scale_x = image_rect.width() / image_size.0 as f32;
        let scale_y = image_rect.height() / image_size.1 as f32;
        pos2(
            image_rect.left() + point.x * scale_x,
            image_rect.top() + point.y * scale_y,
        )
    }

    fn bbox_to_screen_rect(
        &self,
        bbox: BoundingBox,
        image_rect: Rect,
        image_size: (u32, u32),
    ) -> Rect {
        if image_size.0 == 0 || image_size.1 == 0 {
            return Rect::from_min_size(image_rect.left_top(), vec2(0.0, 0.0));
        }
        let scale_x = image_rect.width() / image_size.0 as f32;
        let scale_y = image_rect.height() / image_size.1 as f32;
        let top_left = pos2(
            image_rect.left() + bbox.x * scale_x,
            image_rect.top() + bbox.y * scale_y,
        );
        let size = vec2(bbox.width * scale_x, bbox.height * scale_y);
        Rect::from_min_size(top_left, size)
    }

    fn clamp_bbox_to_image(&self, mut bbox: BoundingBox, image_size: (u32, u32)) -> BoundingBox {
        let img_w = image_size.0 as f32;
        let img_h = image_size.1 as f32;
        let min_size = 8.0;
        bbox.width = bbox.width.max(min_size).min(img_w.max(1.0));
        bbox.height = bbox.height.max(min_size).min(img_h.max(1.0));
        bbox.x = bbox.x.clamp(0.0, (img_w - bbox.width).max(0.0));
        bbox.y = bbox.y.clamp(0.0, (img_h - bbox.height).max(0.0));
        bbox
    }

    fn draft_to_bbox(&self, draft: ManualBoxDraft, image_size: (u32, u32)) -> Option<BoundingBox> {
        let x1 = draft.start.x.min(draft.current.x);
        let x2 = draft.start.x.max(draft.current.x);
        let y1 = draft.start.y.min(draft.current.y);
        let y2 = draft.start.y.max(draft.current.y);
        let width = (x2 - x1).abs();
        let height = (y2 - y1).abs();
        if width < 8.0 || height < 8.0 {
            return None;
        }
        let bbox = BoundingBox {
            x: x1,
            y: y1,
            width,
            height,
        };
        Some(self.clamp_bbox_to_image(bbox, image_size))
    }

    fn add_manual_detection(&mut self, ctx: &EguiContext, bbox: BoundingBox) {
        let landmarks = self.placeholder_landmarks(bbox);
        let detection = Detection {
            bbox,
            landmarks,
            score: 1.0,
        };
        let det = DetectionWithQuality {
            detection,
            quality_score: 0.0,
            quality: Quality::High,
            thumbnail: None,
            current_bbox: bbox,
            original_bbox: bbox,
            origin: DetectionOrigin::Manual,
        };
        self.preview.detections.push(det);
        let index = self.preview.detections.len() - 1;
        self.selected_faces.insert(index);
        self.clear_crop_preview_cache();
        self.refresh_detection_thumbnail_at(ctx, index);
    }

    fn placeholder_landmarks(&self, bbox: BoundingBox) -> [Landmark; 5] {
        let cx = bbox.x + bbox.width * 0.5;
        let cy = bbox.y + bbox.height * 0.5;
        [
            Landmark {
                x: bbox.x + bbox.width * 0.3,
                y: bbox.y + bbox.height * 0.35,
            },
            Landmark {
                x: bbox.x + bbox.width * 0.7,
                y: bbox.y + bbox.height * 0.35,
            },
            Landmark { x: cx, y: cy },
            Landmark {
                x: bbox.x + bbox.width * 0.35,
                y: bbox.y + bbox.height * 0.75,
            },
            Landmark {
                x: bbox.x + bbox.width * 0.65,
                y: bbox.y + bbox.height * 0.75,
            },
        ]
    }

    fn refresh_detection_thumbnail_at(&mut self, ctx: &EguiContext, index: usize) {
        let Some(image) = self.preview.source_image.clone() else {
            return;
        };
        let Some(det) = self.preview.detections.get_mut(index) else {
            return;
        };
        let bbox = det.active_bbox();

        let mut x = bbox.x.max(0.0) as u32;
        let mut y = bbox.y.max(0.0) as u32;
        let mut w = bbox.width.max(1.0) as u32;
        let mut h = bbox.height.max(1.0) as u32;

        let img_w = image.width();
        let img_h = image.height();
        x = x.min(img_w.saturating_sub(1));
        y = y.min(img_h.saturating_sub(1));
        w = w.min(img_w.saturating_sub(x));
        h = h.min(img_h.saturating_sub(y));

        let face_region = image.crop_imm(x, y, w, h);
        let thumb = face_region.resize(96, 96, image::imageops::FilterType::Lanczos3);
        let thumb_rgba = thumb.to_rgba8();
        let thumb_size = [thumb_rgba.width() as usize, thumb_rgba.height() as usize];
        let thumb_color = egui::ColorImage::from_rgba_unmultiplied(thumb_size, thumb_rgba.as_raw());
        let texture_name = format!("yunet-face-thumb-{}-{}", self.texture_seq, index);
        self.texture_seq = self.texture_seq.wrapping_add(1);
        let texture = ctx.load_texture(texture_name, thumb_color, TextureOptions::LINEAR);
        det.thumbnail = Some(texture);
    }

    fn reset_detection_bbox(&mut self, ctx: &EguiContext, index: usize) {
        if let Some(det) = self.preview.detections.get_mut(index) {
            det.reset_bbox();
        }
        self.on_bbox_drag_finished(ctx, index);
    }

    fn remove_detection(&mut self, index: usize) {
        if index >= self.preview.detections.len() {
            return;
        }
        self.preview.detections.remove(index);
        let mut new_selection = HashSet::new();
        for face_idx in self.selected_faces.iter().copied() {
            if face_idx == index {
                continue;
            } else if face_idx > index {
                new_selection.insert(face_idx - 1);
            } else {
                new_selection.insert(face_idx);
            }
        }
        self.selected_faces = new_selection;
        self.active_bbox_drag = None;
        self.clear_crop_preview_cache();
    }

    fn quality_legend(&self, ui: &mut Ui, palette: theme::Palette) {
        ui.horizontal(|ui| {
            self.legend_dot(ui, palette, palette.success, "High");
            ui.add_space(8.0);
            self.legend_dot(ui, palette, palette.warning, "Medium");
            ui.add_space(8.0);
            self.legend_dot(ui, palette, palette.danger, "Low");
        });
    }

    fn legend_dot(&self, ui: &mut Ui, palette: theme::Palette, color: Color32, label: &str) {
        let (rect, _) = ui.allocate_exact_size(vec2(12.0, 12.0), Sense::hover());
        ui.painter().circle_filled(rect.center(), 5.0, color);
        ui.add_space(4.0);
        ui.label(RichText::new(label).color(palette.subtle_text));
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
                self.settings.enhance.skin_smooth = 0.0;
                self.settings.enhance.red_eye_removal = false;
                self.settings.enhance.background_blur = false;
            }
            "vivid" => {
                self.settings.enhance.auto_color = true;
                self.settings.enhance.exposure_stops = 0.2;
                self.settings.enhance.brightness = 10;
                self.settings.enhance.contrast = 1.2;
                self.settings.enhance.saturation = 1.3;
                self.settings.enhance.sharpness = 0.8;
                self.settings.enhance.skin_smooth = 0.0;
                self.settings.enhance.red_eye_removal = false;
                self.settings.enhance.background_blur = false;
            }
            "professional" => {
                self.settings.enhance.auto_color = false;
                self.settings.enhance.exposure_stops = 0.0;
                self.settings.enhance.brightness = 0;
                self.settings.enhance.contrast = 1.1;
                self.settings.enhance.saturation = 1.0;
                self.settings.enhance.sharpness = 0.6;
                self.settings.enhance.skin_smooth = 0.0;
                self.settings.enhance.red_eye_removal = false;
                self.settings.enhance.background_blur = false;
            }
            _ => {
                // "none" or unknown - no changes
            }
        }
    }

    fn resolved_output_dimensions(&self) -> (u32, u32) {
        if self.settings.crop.preset == "custom" {
            (
                self.settings.crop.output_width,
                self.settings.crop.output_height,
            )
        } else if let Some(preset) = preset_by_name(&self.settings.crop.preset) {
            (preset.width, preset.height)
        } else {
            (400, 400)
        }
    }

    /// Builds yunet-core CropSettings from the GUI settings.
    fn build_crop_settings(&self) -> CoreCropSettings {
        let (output_width, output_height) = self.resolved_output_dimensions();

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
            unsharp_amount: 0.6, // Base unsharp amount
            unsharp_radius: 1.0,
            sharpness: self.settings.enhance.sharpness,
            skin_smooth_amount: self.settings.enhance.skin_smooth,
            skin_smooth_sigma_space: 3.0,
            skin_smooth_sigma_color: 25.0,
            red_eye_removal: self.settings.enhance.red_eye_removal,
            red_eye_threshold: 1.5,
            background_blur: self.settings.enhance.background_blur,
            background_blur_radius: 15.0,
            background_blur_mask_size: 0.6,
        }
    }

    fn quality_suffix(&self, quality: Quality) -> Option<&'static str> {
        if !self.settings.crop.quality_rules.quality_suffix {
            return None;
        }
        match quality {
            Quality::High => Some("_highq"),
            Quality::Medium => Some("_medq"),
            Quality::Low => Some("_lowq"),
        }
    }

    fn should_skip_quality(&self, quality: Quality) -> bool {
        if let Some(min) = self.settings.crop.quality_rules.min_quality {
            quality < min
        } else {
            false
        }
    }

    fn apply_quality_rules_to_preview(&mut self) {
        self.selected_faces.clear();
        if self.preview.detections.is_empty() {
            return;
        }

        let rules = &self.settings.crop.quality_rules;
        let best_quality = self.preview.detections.iter().map(|d| d.quality).max();

        if rules.auto_skip_no_high_quality && best_quality != Some(Quality::High) {
            if let Some(path) = &self.preview.image_path {
                self.status_line = format!("No high-quality faces detected in {}", path.display());
            } else {
                self.status_line = "No high-quality faces detected".to_string();
            }
            return;
        }

        if rules.auto_select_best_face
            && self.preview.detections.len() > 1
            && let Some((best_idx, _)) =
                self.preview.detections.iter().enumerate().max_by(|a, b| {
                    a.1.quality.cmp(&b.1.quality).then_with(|| {
                        a.1.quality_score
                            .partial_cmp(&b.1.quality_score)
                            .unwrap_or(Ordering::Equal)
                    })
                })
        {
            self.selected_faces.insert(best_idx);
            return;
        }

        for idx in 0..self.preview.detections.len() {
            self.selected_faces.insert(idx);
        }
    }

    fn format_metadata_tags(map: &BTreeMap<String, String>) -> String {
        map.iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn display_with_output_ext(raw: &str, ext: &str) -> String {
        let mut path = PathBuf::from(raw);
        let trimmed = ext.trim().trim_start_matches('.');
        if trimmed.is_empty() {
            return path.display().to_string();
        }
        path.set_extension(trimmed);
        path.display().to_string()
    }

    fn parse_metadata_tags(text: &str) -> BTreeMap<String, String> {
        let mut map = BTreeMap::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some((key, value)) = trimmed.split_once('=') {
                let key = key.trim();
                if key.is_empty() {
                    continue;
                }
                map.insert(key.to_string(), value.trim().to_string());
            }
        }
        map
    }

    fn refresh_metadata_tags_input(&mut self) {
        self.metadata_tags_input =
            Self::format_metadata_tags(&self.settings.crop.metadata.custom_tags);
    }

    fn push_crop_history(&mut self) {
        if self
            .crop_history
            .last()
            .is_some_and(|last| last == &self.settings.crop)
        {
            return;
        }
        if self.crop_history_index + 1 < self.crop_history.len() {
            self.crop_history.truncate(self.crop_history_index + 1);
        }
        self.crop_history.push(self.settings.crop.clone());
        const MAX_HISTORY: usize = 100;
        if self.crop_history.len() > MAX_HISTORY {
            let remove = self.crop_history.len() - MAX_HISTORY;
            self.crop_history.drain(0..remove);
        }
        self.crop_history_index = self.crop_history.len() - 1;
    }

    fn undo_crop_settings(&mut self) {
        if self.crop_history_index == 0 {
            return;
        }
        self.crop_history_index -= 1;
        self.settings.crop = self.crop_history[self.crop_history_index].clone();
        self.clear_crop_preview_cache();
        self.persist_settings_with_feedback();
        self.apply_quality_rules_to_preview();
        self.refresh_metadata_tags_input();
    }

    fn redo_crop_settings(&mut self) {
        if self.crop_history_index + 1 >= self.crop_history.len() {
            return;
        }
        self.crop_history_index += 1;
        self.settings.crop = self.crop_history[self.crop_history_index].clone();
        self.clear_crop_preview_cache();
        self.persist_settings_with_feedback();
        self.apply_quality_rules_to_preview();
        self.refresh_metadata_tags_input();
    }

    fn adjust_horizontal_offset(&mut self, delta: f32) {
        if delta.abs() < f32::EPSILON {
            return;
        }
        let new_value = (self.settings.crop.horizontal_offset + delta).clamp(-1.0, 1.0);
        if (new_value - self.settings.crop.horizontal_offset).abs() < f32::EPSILON {
            return;
        }
        self.settings.crop.horizontal_offset = new_value;
        self.push_crop_history();
        self.clear_crop_preview_cache();
        self.persist_settings_with_feedback();
    }

    fn adjust_vertical_offset(&mut self, delta: f32) {
        if delta.abs() < f32::EPSILON {
            return;
        }
        let new_value = (self.settings.crop.vertical_offset + delta).clamp(-1.0, 1.0);
        if (new_value - self.settings.crop.vertical_offset).abs() < f32::EPSILON {
            return;
        }
        self.settings.crop.vertical_offset = new_value;
        self.push_crop_history();
        self.clear_crop_preview_cache();
        self.persist_settings_with_feedback();
    }

    fn adjust_face_height(&mut self, delta: f32) {
        if delta.abs() < f32::EPSILON {
            return;
        }
        let new_value = (self.settings.crop.face_height_pct + delta).clamp(10.0, 100.0);
        if (new_value - self.settings.crop.face_height_pct).abs() < f32::EPSILON {
            return;
        }
        self.settings.crop.face_height_pct = new_value;
        self.push_crop_history();
        self.clear_crop_preview_cache();
        self.persist_settings_with_feedback();
    }

    fn set_crop_preset(&mut self, preset: &str) {
        if self.settings.crop.preset == preset {
            return;
        }
        self.settings.crop.preset = preset.to_string();
        if preset != "custom"
            && let Some(info) = preset_by_name(preset)
        {
            self.settings.crop.output_width = info.width;
            self.settings.crop.output_height = info.height;
        }
        self.push_crop_history();
        self.clear_crop_preview_cache();
        self.persist_settings_with_feedback();
    }

    fn handle_shortcuts(&mut self, ctx: &EguiContext) {
        let wants_text = ctx.wants_keyboard_input();

        struct ShortcutActions {
            undo: bool,
            redo: bool,
            horiz_delta: f32,
            vert_delta: f32,
            face_height_delta: f32,
            preset: Option<&'static str>,
            toggle_enhance: bool,
            export: bool,
        }

        let mut actions = ShortcutActions {
            undo: false,
            redo: false,
            horiz_delta: 0.0,
            vert_delta: 0.0,
            face_height_delta: 0.0,
            preset: None,
            toggle_enhance: false,
            export: false,
        };

        ctx.input(|input| {
            let base_step = if input.modifiers.shift { 0.1 } else { 0.05 };
            let face_step = if input.modifiers.shift { 5.0 } else { 1.0 };
            let command = input.modifiers.command;

            if input.key_pressed(Key::Z) && command {
                if input.modifiers.shift {
                    actions.redo = true;
                } else {
                    actions.undo = true;
                }
            }
            if input.key_pressed(Key::Y) && command {
                actions.redo = true;
            }

            if !command && !wants_text {
                if input.key_pressed(Key::ArrowLeft) {
                    actions.horiz_delta -= base_step;
                }
                if input.key_pressed(Key::ArrowRight) {
                    actions.horiz_delta += base_step;
                }
                if input.key_pressed(Key::ArrowUp) {
                    actions.vert_delta -= base_step;
                }
                if input.key_pressed(Key::ArrowDown) {
                    actions.vert_delta += base_step;
                }
                if input.key_pressed(Key::Minus) {
                    actions.face_height_delta -= face_step;
                }
                if input.key_pressed(Key::Equals) {
                    actions.face_height_delta += face_step;
                }

                const PRESETS: [(&str, Key); 6] = [
                    ("linkedin", Key::Num1),
                    ("passport", Key::Num2),
                    ("instagram", Key::Num3),
                    ("idcard", Key::Num4),
                    ("avatar", Key::Num5),
                    ("headshot", Key::Num6),
                ];
                for (preset, key) in PRESETS {
                    if input.key_pressed(key) {
                        actions.preset = Some(preset);
                    }
                }

                if input.key_pressed(Key::Space) {
                    actions.toggle_enhance = true;
                }
                if input.key_pressed(Key::Enter) {
                    actions.export = true;
                }
            }
        });

        if actions.undo {
            self.undo_crop_settings();
            return;
        }
        if actions.redo {
            self.redo_crop_settings();
            return;
        }

        if actions.horiz_delta.abs() > f32::EPSILON {
            self.adjust_horizontal_offset(actions.horiz_delta);
        }
        if actions.vert_delta.abs() > f32::EPSILON {
            self.adjust_vertical_offset(actions.vert_delta);
        }
        if actions.face_height_delta.abs() > f32::EPSILON {
            self.adjust_face_height(actions.face_height_delta);
        }
        if let Some(preset) = actions.preset {
            self.set_crop_preset(preset);
        }
        if actions.toggle_enhance {
            self.settings.enhance.enabled = !self.settings.enhance.enabled;
            self.clear_crop_preview_cache();
            self.persist_settings_with_feedback();
        }
        if actions.export {
            self.export_selected_faces();
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
                self.show_error(
                    "Select an image before exporting.",
                    "No image loaded when export was requested.",
                );
                return;
            }
        };

        let crop_config = self.settings.crop.clone();
        let output_options = OutputOptions::from_crop_settings(&crop_config);

        let mut export_count = 0;
        let mut error_count = 0;
        let mut skipped_quality = 0;

        let selected_faces: Vec<usize> = self.selected_faces.iter().copied().collect();

        // Export each selected face
        for face_idx in selected_faces {
            let Some(det_with_quality) = self.preview.detections.get(face_idx).cloned() else {
                continue;
            };
            let detection_score = det_with_quality.detection.score;
            let quality = det_with_quality.quality;
            let quality_score = det_with_quality.quality_score;

            if self.should_skip_quality(quality) {
                info!(
                    "Skipping face {} due to {:?} quality",
                    face_idx + 1,
                    quality
                );
                skipped_quality += 1;
                continue;
            }

            let (_, preview_image) = match self.ensure_crop_preview_entry(face_idx) {
                Some(entry) => entry,
                None => {
                    warn!("Failed to prepare crop preview for face {}", face_idx + 1);
                    error_count += 1;
                    continue;
                }
            };

            // Generate output filename
            let source_stem = source_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("face");
            let ext = match crop_config.output_format.as_str() {
                "jpeg" | "jpg" => "jpg",
                "webp" => "webp",
                _ => "png",
            };
            let mut output_filename = format!("{}_face_{:02}.{}", source_stem, face_idx + 1, ext);
            if let Some(suffix) = self.quality_suffix(quality) {
                output_filename = append_suffix_to_filename(&output_filename, suffix);
            }
            let output_path = output_dir.join(output_filename);

            let metadata_ctx = MetadataContext {
                source_path: Some(source_path.as_path()),
                crop_settings: Some(&crop_config),
                detection_score: Some(detection_score),
                quality: Some(quality),
                quality_score: Some(quality_score),
            };

            match save_dynamic_image(&preview_image, &output_path, &output_options, &metadata_ctx) {
                Ok(_) => {
                    info!(
                        "Exported face {} to {}",
                        face_idx + 1,
                        output_path.display()
                    );
                    export_count += 1;
                }
                Err(e) => {
                    warn!("Failed to save face {}: {}", face_idx + 1, e);
                    error_count += 1;
                }
            }
        }

        // Update status
        if error_count == 0 {
            if skipped_quality > 0 {
                self.show_success(format!(
                    "Exported {} face{} ({} skipped for quality) to {}",
                    export_count,
                    if export_count == 1 { "" } else { "s" },
                    skipped_quality,
                    output_dir.display()
                ));
            } else {
                self.show_success(format!(
                    "Successfully exported {} face{} to {}",
                    export_count,
                    if export_count == 1 { "" } else { "s" },
                    output_dir.display()
                ));
            }
        } else {
            let mut summary = format!(
                "Exported {} face{}, {} error{}",
                export_count,
                if export_count == 1 { "" } else { "s" },
                error_count,
                if error_count == 1 { "" } else { "s" }
            );
            if skipped_quality > 0 {
                summary.push_str(&format!(", {} skipped for quality", skipped_quality));
            }
            self.show_error(
                summary,
                format!("{error_count} export error(s) occurred while writing files."),
            );
        }

        info!(
            "Export complete: {} succeeded, {} failed, {} skipped for quality",
            export_count, error_count, skipped_quality
        );
    }

    /// Paints the detection bounding boxes and landmarks over the preview image.
    fn paint_detections(&self, ui: &egui::Ui, image_rect: Rect, image_size: (u32, u32)) {
        let painter = ui.painter().with_clip_rect(image_rect);
        let scale_x = image_rect.width() / image_size.0 as f32;
        let scale_y = image_rect.height() / image_size.1 as f32;
        let stroke_scale = scale_x.min(scale_y).max(0.1);

        let base_bbox_color = Color32::from_rgb(255, 145, 77);
        let manual_bbox_color = Color32::from_rgb(255, 214, 142);
        let bbox_width = (2.0 * stroke_scale).clamp(0.5, 6.0);
        let landmark_color = Color32::from_rgb(82, 180, 255);
        let crop_stroke = Stroke::new(
            (3.0 * stroke_scale).clamp(0.5, 6.0),
            Color32::from_rgb(0, 255, 127),
        );
        let selected_stroke = Stroke::new(
            (4.0 * stroke_scale).clamp(0.5, 8.0),
            Color32::from_rgb(100, 150, 255),
        );
        let landmark_radius = (3.0 * stroke_scale).clamp(1.0, 6.0);

        if self.should_show_rule_of_thirds_guides() {
            self.paint_rule_of_thirds_guides(&painter, image_rect, stroke_scale);
        }

        for (index, det_with_quality) in self.preview.detections.iter().enumerate() {
            let bbox = det_with_quality.active_bbox();
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
            let bbox_color = if det_with_quality.is_manual() {
                manual_bbox_color
            } else {
                base_bbox_color
            };
            let bbox_stroke = Stroke::new(bbox_width, bbox_color);
            painter.rect_stroke(rect, 0.0, bbox_stroke, egui::StrokeKind::Inside);

            if !det_with_quality.is_manual() {
                for landmark in &det_with_quality.detection.landmarks {
                    let center = pos2(
                        image_rect.left() + landmark.x * scale_x,
                        image_rect.top() + landmark.y * scale_y,
                    );
                    painter.circle_filled(center, landmark_radius, landmark_color);
                }
            }

            // Draw resize handles
            let handle_size = (10.0 * stroke_scale).clamp(4.0, 12.0);
            let handle_color = Color32::from_rgba_unmultiplied(255, 255, 255, 210);
            for center in [
                pos2(rect.left(), rect.top()),
                pos2(rect.right(), rect.top()),
                pos2(rect.left(), rect.bottom()),
                pos2(rect.right(), rect.bottom()),
            ] {
                let handle_rect = Rect::from_center_size(center, vec2(handle_size, handle_size));
                painter.rect_filled(handle_rect, 2.0, handle_color);
            }

            // Paint crop region overlay if enabled
            if self.show_crop_overlay {
                let crop_settings = self.build_crop_settings();
                let crop_region =
                    calculate_crop_region(image_size.0, image_size.1, bbox, &crop_settings);
                let crop_top_left = pos2(
                    image_rect.left() + crop_region.x as f32 * scale_x,
                    image_rect.top() + crop_region.y as f32 * scale_y,
                );
                let crop_size = vec2(
                    crop_region.width as f32 * scale_x,
                    crop_region.height as f32 * scale_y,
                );
                let crop_rect = Rect::from_min_size(crop_top_left, crop_size);
                let shape_points = outline_points_for_rect(
                    crop_rect.width(),
                    crop_rect.height(),
                    &self.settings.crop.shape,
                );
                if shape_points.len() >= 2 {
                    let mut outline = Vec::with_capacity(shape_points.len());
                    for (x, y) in shape_points {
                        outline.push(pos2(crop_rect.left() + x, crop_rect.top() + y));
                    }
                    painter.add(egui::Shape::closed_line(outline, crop_stroke));
                } else {
                    painter.rect_stroke(crop_rect, 4.0, crop_stroke, egui::StrokeKind::Inside);
                }
            }
        }

        if let Some(draft) = self.manual_box_draft {
            let a = self.image_point_to_screen(draft.start, image_rect, image_size);
            let b = self.image_point_to_screen(draft.current, image_rect, image_size);
            let draft_rect = Rect::from_two_pos(a, b);
            let draft_stroke = Stroke::new(
                (2.5 * stroke_scale).clamp(1.0, 4.0),
                Color32::from_rgb(100, 200, 255),
            );
            painter.rect_stroke(draft_rect, 0.0, draft_stroke, egui::StrokeKind::Inside);
        }
    }

    fn should_show_rule_of_thirds_guides(&self) -> bool {
        if !self.show_crop_overlay {
            return false;
        }
        matches!(
            self.settings.crop.positioning_mode.as_str(),
            "rule-of-thirds" | "rule_of_thirds"
        )
    }

    fn paint_rule_of_thirds_guides(
        &self,
        painter: &egui::Painter,
        image_rect: Rect,
        stroke_scale: f32,
    ) {
        let guide_color = Color32::from_rgba_unmultiplied(255, 255, 255, 110);
        let guide_stroke = Stroke::new((1.5 * stroke_scale).clamp(0.5, 4.0), guide_color);

        for i in 1..=2 {
            let frac = i as f32 / 3.0;
            let x = image_rect.left() + image_rect.width() * frac;
            let y = image_rect.top() + image_rect.height() * frac;
            painter.line_segment(
                [pos2(x, image_rect.top()), pos2(x, image_rect.bottom())],
                guide_stroke,
            );
            painter.line_segment(
                [pos2(image_rect.left(), y), pos2(image_rect.right(), y)],
                guide_stroke,
            );
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

    fn show_success(&mut self, message: impl Into<String>) {
        self.status_line = message.into();
        self.last_error = None;
    }

    fn show_error(&mut self, headline: impl Into<String>, detail: impl Into<String>) {
        self.status_line = headline.into();
        self.last_error = Some(detail.into());
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
        self.preview.is_loading = false;
        self.status_line = status_message.to_owned();
        self.clear_crop_preview_cache();
    }

    fn clear_crop_preview_cache(&mut self) {
        if !self.crop_preview_cache.is_empty() {
            info!(
                "Clearing cached crop previews ({})",
                self.crop_preview_cache.len()
            );
        }
        self.crop_preview_cache.clear();
    }

    fn enhancement_signature(&self, settings: &EnhancementSettings) -> EnhancementSignature {
        EnhancementSignature {
            auto_color: settings.auto_color,
            exposure_bits: settings.exposure_stops.to_bits(),
            brightness: settings.brightness,
            contrast_bits: settings.contrast.to_bits(),
            saturation_bits: settings.saturation.to_bits(),
            unsharp_amount_bits: settings.unsharp_amount.to_bits(),
            unsharp_radius_bits: settings.unsharp_radius.to_bits(),
            sharpness_bits: settings.sharpness.to_bits(),
            skin_smooth_bits: settings.skin_smooth_amount.to_bits(),
            skin_sigma_space_bits: settings.skin_smooth_sigma_space.to_bits(),
            skin_sigma_color_bits: settings.skin_smooth_sigma_color.to_bits(),
            red_eye_removal: settings.red_eye_removal,
            red_eye_threshold_bits: settings.red_eye_threshold.to_bits(),
            background_blur: settings.background_blur,
            background_blur_radius_bits: settings.background_blur_radius.to_bits(),
            background_blur_mask_bits: settings.background_blur_mask_size.to_bits(),
        }
    }

    fn positioning_mode_id(mode: PositioningMode) -> u8 {
        match mode {
            PositioningMode::Center => 0,
            PositioningMode::RuleOfThirds => 1,
            PositioningMode::Custom => 2,
        }
    }

    fn shape_signature(settings: &ConfigCropSettings) -> ShapeSignature {
        let shape = settings.shape.clone().sanitized();
        match shape {
            CropShape::Rectangle => ShapeSignature {
                kind: 0,
                primary_bits: 0,
                secondary_bits: 0,
                sides: 0,
                rotation_bits: 0,
            },
            CropShape::Ellipse => ShapeSignature {
                kind: 1,
                primary_bits: 0,
                secondary_bits: 0,
                sides: 0,
                rotation_bits: 0,
            },
            CropShape::RoundedRectangle { radius_pct } => ShapeSignature {
                kind: 2,
                primary_bits: radius_pct.to_bits(),
                secondary_bits: 0,
                sides: 0,
                rotation_bits: 0,
            },
            CropShape::ChamferedRectangle { size_pct } => ShapeSignature {
                kind: 3,
                primary_bits: size_pct.to_bits(),
                secondary_bits: 0,
                sides: 0,
                rotation_bits: 0,
            },
            CropShape::Polygon {
                sides,
                rotation_deg,
                corner_style,
            } => {
                let (kind, primary, secondary) = match corner_style {
                    PolygonCornerStyle::Sharp => (4_u8, 0_u32, 0_u32),
                    PolygonCornerStyle::Rounded { radius_pct } => {
                        (5_u8, radius_pct.to_bits(), 0_u32)
                    }
                    PolygonCornerStyle::Chamfered { size_pct } => (6_u8, size_pct.to_bits(), 0_u32),
                };
                ShapeSignature {
                    kind,
                    primary_bits: primary,
                    secondary_bits: secondary,
                    sides,
                    rotation_bits: rotation_deg.to_bits(),
                }
            }
        }
    }

    fn edit_shape_controls(&mut self, ui: &mut Ui) -> bool {
        let mut shape = self.settings.crop.shape.clone();
        let mut changed = false;

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Variant {
            Rectangle,
            RoundedRect,
            ChamferRect,
            Ellipse,
            PolygonSharp,
            PolygonRounded,
            PolygonChamfered,
        }

        let mut variant = match &shape {
            CropShape::Rectangle => Variant::Rectangle,
            CropShape::RoundedRectangle { .. } => Variant::RoundedRect,
            CropShape::ChamferedRectangle { .. } => Variant::ChamferRect,
            CropShape::Ellipse => Variant::Ellipse,
            CropShape::Polygon { corner_style, .. } => match corner_style {
                PolygonCornerStyle::Sharp => Variant::PolygonSharp,
                PolygonCornerStyle::Rounded { .. } => Variant::PolygonRounded,
                PolygonCornerStyle::Chamfered { .. } => Variant::PolygonChamfered,
            },
        };

        let current_label = match variant {
            Variant::Rectangle => "Rectangle",
            Variant::RoundedRect => "Rounded rectangle",
            Variant::ChamferRect => "Chamfered rectangle",
            Variant::Ellipse => "Ellipse",
            Variant::PolygonSharp => "Polygon",
            Variant::PolygonRounded => "Polygon (rounded)",
            Variant::PolygonChamfered => "Polygon (chamfered)",
        };

        let mut variant_changed = false;
        egui::ComboBox::from_label("Shape")
            .selected_text(current_label)
            .show_ui(ui, |ui| {
                let mut select_variant = |label: &str, target: Variant| {
                    let selected = variant == target;
                    if ui.selectable_label(selected, label).clicked() && !selected {
                        variant = target;
                        variant_changed = true;
                    }
                };
                select_variant("Rectangle", Variant::Rectangle);
                select_variant("Rounded rectangle", Variant::RoundedRect);
                select_variant("Chamfered rectangle", Variant::ChamferRect);
                select_variant("Ellipse", Variant::Ellipse);
                select_variant("Polygon", Variant::PolygonSharp);
                select_variant("Polygon (rounded)", Variant::PolygonRounded);
                select_variant("Polygon (chamfered)", Variant::PolygonChamfered);
            });

        if variant_changed {
            shape = match variant {
                Variant::Rectangle => CropShape::Rectangle,
                Variant::RoundedRect => CropShape::RoundedRectangle { radius_pct: 0.12 },
                Variant::ChamferRect => CropShape::ChamferedRectangle { size_pct: 0.12 },
                Variant::Ellipse => CropShape::Ellipse,
                Variant::PolygonSharp => CropShape::Polygon {
                    sides: 6,
                    rotation_deg: 0.0,
                    corner_style: PolygonCornerStyle::Sharp,
                },
                Variant::PolygonRounded => CropShape::Polygon {
                    sides: 6,
                    rotation_deg: 0.0,
                    corner_style: PolygonCornerStyle::Rounded { radius_pct: 0.1 },
                },
                Variant::PolygonChamfered => CropShape::Polygon {
                    sides: 6,
                    rotation_deg: 0.0,
                    corner_style: PolygonCornerStyle::Chamfered { size_pct: 0.1 },
                },
            };
            changed = true;
        }

        match &mut shape {
            CropShape::RoundedRectangle { radius_pct } => {
                let mut radius = (*radius_pct * 100.0).clamp(0.0, 50.0);
                if ui
                    .add(Slider::new(&mut radius, 0.0..=50.0).text("Corner radius (%)"))
                    .changed()
                {
                    *radius_pct = (radius / 100.0).clamp(0.0, 0.5);
                    changed = true;
                }
            }
            CropShape::ChamferedRectangle { size_pct } => {
                let mut size = (*size_pct * 100.0).clamp(0.0, 50.0);
                if ui
                    .add(Slider::new(&mut size, 0.0..=50.0).text("Chamfer size (%)"))
                    .changed()
                {
                    *size_pct = (size / 100.0).clamp(0.0, 0.5);
                    changed = true;
                }
            }
            CropShape::Polygon {
                sides,
                rotation_deg,
                corner_style,
            } => {
                let mut sides_u32 = *sides as u32;
                if ui
                    .add(
                        DragValue::new(&mut sides_u32)
                            .range(3..=24)
                            .speed(1.0)
                            .suffix(" sides"),
                    )
                    .changed()
                {
                    *sides = sides_u32.clamp(3, 24) as u8;
                    changed = true;
                }
                if ui
                    .add(Slider::new(rotation_deg, -180.0..=180.0).text("Rotation (°)"))
                    .changed()
                {
                    changed = true;
                }

                match corner_style {
                    PolygonCornerStyle::Sharp => {}
                    PolygonCornerStyle::Rounded { radius_pct } => {
                        let mut radius = (*radius_pct * 100.0).clamp(0.0, 40.0);
                        if ui
                            .add(Slider::new(&mut radius, 0.0..=40.0).text("Corner radius (%)"))
                            .changed()
                        {
                            *radius_pct = (radius / 100.0).clamp(0.0, 0.5);
                            changed = true;
                        }
                    }
                    PolygonCornerStyle::Chamfered { size_pct } => {
                        let mut size = (*size_pct * 100.0).clamp(0.0, 40.0);
                        if ui
                            .add(Slider::new(&mut size, 0.0..=40.0).text("Chamfer size (%)"))
                            .changed()
                        {
                            *size_pct = (size / 100.0).clamp(0.0, 0.5);
                            changed = true;
                        }
                    }
                }
            }
            CropShape::Rectangle | CropShape::Ellipse => {}
        }

        let sanitized = shape.sanitized();
        if sanitized != self.settings.crop.shape {
            self.settings.crop.shape = sanitized;
            changed = true;
        }

        changed
    }

    fn ensure_crop_preview_entry(
        &mut self,
        face_idx: usize,
    ) -> Option<(CropPreviewKey, Arc<DynamicImage>)> {
        let path = self.preview.image_path.clone()?;
        let detection = self.preview.detections.get(face_idx)?;

        let crop_settings = self.build_crop_settings();
        let enhancement_settings = self.build_enhancement_settings();
        let enhance_enabled = self.settings.enhance.enabled;
        let signature = self.enhancement_signature(&enhancement_settings);

        let key = CropPreviewKey {
            path: path.clone(),
            face_index: face_idx,
            output_width: crop_settings.output_width,
            output_height: crop_settings.output_height,
            positioning_mode: Self::positioning_mode_id(crop_settings.positioning_mode),
            face_height_bits: crop_settings.face_height_pct.to_bits(),
            horizontal_bits: crop_settings.horizontal_offset.to_bits(),
            vertical_bits: crop_settings.vertical_offset.to_bits(),
            shape: Self::shape_signature(&self.settings.crop),
            enhancement: signature,
            enhance_enabled,
        };

        if let Some(entry) = self.crop_preview_cache.get(&key) {
            return Some((key, entry.image.clone()));
        }

        let source_image = if let Some(img) = &self.preview.source_image {
            img.clone()
        } else {
            match load_image(&path) {
                Ok(img) => {
                    let arc = Arc::new(img);
                    self.preview.source_image = Some(arc.clone());
                    arc
                }
                Err(err) => {
                    warn!(
                        "Failed to load {} for crop preview: {}",
                        path.display(),
                        err
                    );
                    return None;
                }
            }
        };

        let bbox = detection.active_bbox();
        let crop_region = calculate_crop_region(
            source_image.width(),
            source_image.height(),
            bbox,
            &crop_settings,
        );
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
        let processed = if enhance_enabled {
            apply_enhancements(&resized, &enhancement_settings)
        } else {
            resized
        };
        let mut rgba = processed.to_rgba8();
        apply_shape_mask(&mut rgba, &self.settings.crop.shape);
        let final_image = DynamicImage::ImageRgba8(rgba);
        let arc_image = Arc::new(final_image);
        self.crop_preview_cache.insert(
            key.clone(),
            CropPreviewCacheEntry {
                image: arc_image.clone(),
                texture: None,
            },
        );
        Some((key, arc_image))
    }

    fn load_texture_from_image(
        &mut self,
        ctx: &EguiContext,
        image: &DynamicImage,
    ) -> TextureHandle {
        let rgba = image.to_rgba8();
        let size = [rgba.width() as usize, rgba.height() as usize];
        let color_image = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
        let texture_name = format!("yunet-crop-preview-{}", self.texture_seq);
        self.texture_seq = self.texture_seq.wrapping_add(1);
        ctx.load_texture(texture_name, color_image, TextureOptions::LINEAR)
    }

    fn crop_preview_texture_for(
        &mut self,
        ctx: &EguiContext,
        face_idx: usize,
    ) -> Option<TextureHandle> {
        let (key, image) = self.ensure_crop_preview_entry(face_idx)?;
        if let Some(texture) = self
            .crop_preview_cache
            .get(&key)
            .and_then(|entry| entry.texture.clone())
        {
            return Some(texture);
        }
        let texture = self.load_texture_from_image(ctx, &image);
        if let Some(entry) = self.crop_preview_cache.get_mut(&key) {
            entry.texture = Some(texture.clone());
        }
        Some(texture)
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
                    output_override: None,
                })
                .collect();

            self.batch_current_index = None;
            let loaded = self.batch_files.len();
            self.show_success(format!("Loaded {loaded} images for batch processing"));
            info!("Loaded {loaded} images for batch processing");
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
        let mut crop_config = self.settings.crop.clone();
        crop_config.sanitize();
        let output_options = OutputOptions::from_crop_settings(&crop_config);

        // Ensure detector is loaded before processing
        let detector = match self.ensure_detector() {
            Ok(d) => d,
            Err(e) => {
                let detail = format!("Failed to load detector: {e}");
                self.show_error(
                    "Unable to load the detector for batch export.",
                    detail.clone(),
                );
                error!("{detail}");
                return;
            }
        };

        if self.batch_files.is_empty() {
            self.show_error(
                "No batch files to export.",
                "Load images into the batch list before starting export.",
            );
            return;
        }

        for batch_file in &mut self.batch_files {
            batch_file.status = BatchFileStatus::Processing;
        }

        let tasks: Vec<(usize, PathBuf, Option<PathBuf>)> = self
            .batch_files
            .iter()
            .enumerate()
            .map(|(idx, batch_file)| {
                (
                    idx,
                    batch_file.path.clone(),
                    batch_file.output_override.clone(),
                )
            })
            .collect();

        let job_config = Arc::new(BatchJobConfig {
            output_dir: output_dir.clone(),
            crop_settings,
            crop_config: crop_config.clone(),
            enhancement_settings,
            enhance_enabled,
            output_options,
        });

        let detector_for_jobs = detector.clone();
        let results: Vec<(usize, BatchFileStatus)> = tasks
            .into_par_iter()
            .map(|(idx, path, override_path)| {
                let status = Self::run_batch_job(
                    detector_for_jobs.clone(),
                    path,
                    job_config.clone(),
                    override_path,
                );
                (idx, status)
            })
            .collect();

        for (idx, status) in results {
            if let Some(batch_file) = self.batch_files.get_mut(idx) {
                batch_file.status = status;
            }
        }

        // Update status with summary
        let total = self.batch_files.len();
        let completed = self
            .batch_files
            .iter()
            .filter(|f| matches!(f.status, BatchFileStatus::Completed { .. }))
            .count();
        let failed = self
            .batch_files
            .iter()
            .filter(|f| matches!(f.status, BatchFileStatus::Failed { .. }))
            .count();
        let total_faces: usize = self
            .batch_files
            .iter()
            .filter_map(|f| match &f.status {
                BatchFileStatus::Completed { faces_exported, .. } => Some(*faces_exported),
                _ => None,
            })
            .sum();

        if failed > 0 {
            let details: Vec<String> = self
                .batch_files
                .iter()
                .filter_map(|f| match &f.status {
                    BatchFileStatus::Failed { error } => Some(format!(
                        "{}: {error}",
                        f.path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("unknown file")
                    )),
                    _ => None,
                })
                .take(3)
                .collect();
            let mut detail_text = details.join("\n");
            let remaining = failed.saturating_sub(details.len());
            if remaining > 0 {
                if !detail_text.is_empty() {
                    detail_text.push('\n');
                }
                detail_text.push_str(&format!(
                    "...and {} more failure{}",
                    remaining,
                    if remaining == 1 { "" } else { "s" }
                ));
            }
            if detail_text.is_empty() {
                detail_text = format!("{failed} file(s) failed during export.");
            }
            self.show_error(
                format!(
                    "Batch export finished with {failed} error{} ({} faces saved to {}).",
                    if failed == 1 { "" } else { "s" },
                    total_faces,
                    output_dir.display()
                ),
                detail_text,
            );
        } else {
            self.show_success(format!(
                "Batch export complete: {}/{} files, {} faces exported to {}",
                completed,
                total,
                total_faces,
                output_dir.display()
            ));
        }

        info!(
            "Batch export complete: {}/{} files, {} faces exported, {} failed",
            completed, total, total_faces, failed
        );
    }

    fn run_batch_job(
        detector: Arc<YuNetDetector>,
        path: PathBuf,
        config: Arc<BatchJobConfig>,
        output_override: Option<PathBuf>,
    ) -> BatchFileStatus {
        let source_image = match load_image(&path) {
            Ok(img) => img,
            Err(e) => {
                warn!("Failed to load {}: {}", path.display(), e);
                return BatchFileStatus::Failed {
                    error: format!("Failed to load: {e}"),
                };
            }
        };

        let detection_output = match detector.detect_image(&source_image) {
            Ok(output) => output,
            Err(e) => {
                warn!("Detection failed for {}: {}", path.display(), e);
                return BatchFileStatus::Failed {
                    error: format!("Detection failed: {e}"),
                };
            }
        };

        let faces_detected = detection_output.detections.len();

        struct FaceCandidate {
            index: usize,
            quality: Quality,
            quality_score: f64,
            score: f32,
        }

        let mut candidates = Vec::with_capacity(faces_detected);
        for (face_idx, detection) in detection_output.detections.iter().enumerate() {
            let crop_region = calculate_crop_region(
                source_image.width(),
                source_image.height(),
                detection.bbox,
                &config.crop_settings,
            );

            let cropped = source_image.crop_imm(
                crop_region.x,
                crop_region.y,
                crop_region.width,
                crop_region.height,
            );

            let resized = cropped.resize_exact(
                config.crop_settings.output_width,
                config.crop_settings.output_height,
                image::imageops::FilterType::Lanczos3,
            );

            let processed = if config.enhance_enabled {
                apply_enhancements(&resized, &config.enhancement_settings)
            } else {
                resized
            };
            let mut rgba = processed.to_rgba8();
            apply_shape_mask(&mut rgba, &config.crop_config.shape);
            let final_image = DynamicImage::ImageRgba8(rgba);

            let (quality_score, quality) = estimate_sharpness(&final_image);
            candidates.push(FaceCandidate {
                index: face_idx,
                quality,
                quality_score,
                score: detection.score,
            });
        }

        if candidates.is_empty() {
            return BatchFileStatus::Completed {
                faces_detected,
                faces_exported: 0,
            };
        }

        let quality_rules = &config.crop_config.quality_rules;
        let best_quality = candidates.iter().map(|c| c.quality).max();

        if quality_rules.auto_skip_no_high_quality && best_quality != Some(Quality::High) {
            warn!(
                "Skipping {} - no high-quality faces detected",
                path.display()
            );
            return BatchFileStatus::Completed {
                faces_detected,
                faces_exported: 0,
            };
        }

        let mut exports = candidates;
        if quality_rules.auto_select_best_face
            && exports.len() > 1
            && let Some((best_idx, _)) = exports.iter().enumerate().max_by(|a, b| {
                a.1.quality.cmp(&b.1.quality).then_with(|| {
                    a.1.quality_score
                        .partial_cmp(&b.1.quality_score)
                        .unwrap_or(Ordering::Equal)
                })
            })
        {
            let best_index = exports[best_idx].index;
            exports.retain(|c| c.index == best_index);
        }

        let multi_face = exports.len() > 1;

        let mut faces_exported = 0;
        for candidate in exports.into_iter() {
            let should_skip = if let Some(min) = quality_rules.min_quality {
                candidate.quality < min
            } else {
                false
            };
            if should_skip {
                info!(
                    "Skipping face {} from {} due to {:?} quality",
                    candidate.index + 1,
                    path.display(),
                    candidate.quality
                );
                continue;
            }

            let detection = match detection_output.detections.get(candidate.index) {
                Some(det) => det,
                None => continue,
            };

            let crop_region = calculate_crop_region(
                source_image.width(),
                source_image.height(),
                detection.bbox,
                &config.crop_settings,
            );

            let cropped = source_image.crop_imm(
                crop_region.x,
                crop_region.y,
                crop_region.width,
                crop_region.height,
            );

            let resized = cropped.resize_exact(
                config.crop_settings.output_width,
                config.crop_settings.output_height,
                image::imageops::FilterType::Lanczos3,
            );

            let final_image = if config.enhance_enabled {
                apply_enhancements(&resized, &config.enhancement_settings)
            } else {
                resized
            };

            let source_stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("face");
            let mut ext = config.crop_config.output_format.clone();
            if ext.is_empty() {
                ext = "png".to_string();
            }
            let ext = ext.to_ascii_lowercase();

            let output_path = if let Some(custom) = output_override.as_ref() {
                resolve_mapping_override_path(
                    &config.output_dir,
                    custom,
                    &ext,
                    candidate.index,
                    multi_face,
                )
            } else {
                let mut output_filename =
                    format!("{}_face_{:02}.{}", source_stem, candidate.index + 1, ext);
                if quality_rules.quality_suffix {
                    let suffix = match candidate.quality {
                        Quality::High => Some("_highq"),
                        Quality::Medium => Some("_medq"),
                        Quality::Low => Some("_lowq"),
                    };
                    if let Some(suffix) = suffix {
                        output_filename = append_suffix_to_filename(&output_filename, suffix);
                    }
                }
                config.output_dir.join(output_filename)
            };

            if let Some(parent) = output_path.parent()
                && let Err(err) = std::fs::create_dir_all(parent)
            {
                warn!(
                    "Failed to create output directory {}: {err}",
                    parent.display()
                );
            }

            let metadata_ctx = MetadataContext {
                source_path: Some(path.as_path()),
                crop_settings: Some(&config.crop_config),
                detection_score: Some(candidate.score),
                quality: Some(candidate.quality),
                quality_score: Some(candidate.quality_score),
            };

            match save_dynamic_image(
                &final_image,
                &output_path,
                &config.output_options,
                &metadata_ctx,
            ) {
                Ok(_) => {
                    faces_exported += 1;
                }
                Err(err) => {
                    warn!(
                        "Failed to save face {} from {}: {}",
                        candidate.index + 1,
                        path.display(),
                        err
                    );
                }
            }
        }

        BatchFileStatus::Completed {
            faces_detected,
            faces_exported,
        }
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
            self.preview.is_loading = false;
            self.preview.source_image = Some(entry.source_image.clone());
            self.clear_crop_preview_cache();
            self.show_success(format!("Loaded cached detections for {}", path.display()));
            self.is_busy = false;
            self.current_job = None;
            return;
        }

        let detector = match self.ensure_detector() {
            Ok(detector) => detector,
            Err(err) => {
                let detail = format!("Unable to load YuNet model: {err}");
                self.show_error(
                    "Failed to load the YuNet model. Check the model path in Settings.",
                    detail.clone(),
                );
                warn!("{detail}");
                return;
            }
        };

        let job_id = self.next_job_id();
        self.preview.begin_loading(path.clone());
        self.clear_crop_preview_cache();
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
        image: &DynamicImage,
    ) {
        for (index, det) in detections.iter_mut().enumerate() {
            let bbox = det.active_bbox();
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
            let thumb_color =
                egui::ColorImage::from_rgba_unmultiplied(thumb_size, thumb_rgba.as_raw());

            let texture_name = format!("yunet-face-thumb-{}-{}", self.texture_seq, index);
            self.texture_seq = self.texture_seq.wrapping_add(1);
            let texture = ctx.load_texture(texture_name, thumb_color, TextureOptions::LINEAR);
            det.thumbnail = Some(texture);
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
                self.preview.is_loading = false;

                let DetectionJobSuccess {
                    path,
                    color_image,
                    mut detections,
                    original_size,
                    original_image,
                } = data;

                let texture_name = format!("yunet-image-preview-{}", self.texture_seq);
                self.texture_seq = self.texture_seq.wrapping_add(1);

                let texture = ctx.load_texture(texture_name, color_image, TextureOptions::LINEAR);
                let cache_texture = texture.clone();

                self.preview.texture = Some(texture);
                self.preview.image_size = Some(original_size);
                self.preview.image_path = Some(path.clone());
                self.preview.source_image = Some(original_image.clone());

                // Create thumbnails for face regions
                self.create_thumbnails(ctx, &mut detections, &original_image);

                let cached_detections = detections.clone();
                self.preview.detections = detections;
                self.manual_box_draft = None;
                self.active_bbox_drag = None;
                self.manual_box_tool_enabled = false;

                self.show_success(format!(
                    "Detected {} face(s) in {}",
                    self.preview.detections.len(),
                    path.display()
                ));
                self.apply_quality_rules_to_preview();

                self.cache.insert(
                    cache_key,
                    DetectionCacheEntry {
                        texture: cache_texture,
                        image_size: original_size,
                        detections: cached_detections,
                        source_image: original_image,
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
                self.preview.is_loading = false;
                self.show_error(
                    "Detection failed. Verify the model file and try again.",
                    format!("Detection error: {error}"),
                );
                self.preview.texture = None;
                self.preview.image_size = None;
                self.preview.detections.clear();
                self.preview.source_image = None;
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

fn resolve_mapping_override_path(
    output_dir: &Path,
    override_target: &Path,
    ext: &str,
    face_index: usize,
    multi_face: bool,
) -> PathBuf {
    let cleaned_ext = ext.trim_start_matches('.').to_string();
    let parent = if override_target.is_absolute() {
        override_target
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default()
    } else {
        let rel_parent = override_target.parent().unwrap_or_else(|| Path::new(""));
        output_dir.join(rel_parent)
    };
    let base_name = override_target
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "output".to_string());
    let final_base = if multi_face {
        format!("{base_name}_face{}", face_index + 1)
    } else {
        base_name
    };
    let mut final_path = parent;
    final_path.push(final_base);
    final_path.set_extension(cleaned_ext);
    final_path
}

impl App for YuNetApp {
    fn update(&mut self, ctx: &EguiContext, _frame: &mut Frame) {
        self.poll_worker(ctx);
        self.show_status_bar(ctx);
        self.show_configuration_panel(ctx);
        self.show_preview(ctx);

        self.handle_shortcuts(ctx);

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

fn translucent_color(color: Color32, alpha_multiplier: f32) -> Color32 {
    let rgba: Rgba = color.into();
    let new_alpha = (rgba.a() * alpha_multiplier).clamp(0.0, 1.0);
    Rgba::from_rgba_premultiplied(rgba.r(), rgba.g(), rgba.b(), new_alpha).into()
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
    let image = Arc::new(
        load_image(&path)
            .with_context(|| format!("failed to load image from {}", path.display()))?,
    );
    let detection_output = detector
        .detect_image(&image)
        .with_context(|| format!("YuNet detection failed for {}", path.display()))?;

    // Calculate quality scores for each detected face
    let detections_with_quality: Vec<DetectionWithQuality> = detection_output
        .detections
        .into_iter()
        .map(|detection| {
            // Crop face region for quality analysis
            let bbox = detection.bbox;
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
                current_bbox: bbox,
                original_bbox: bbox,
                origin: DetectionOrigin::Detector,
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
        original_image: image,
    })
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
        let app = YuNetApp::create(&ctx, settings_path);
        (app, temp, ctx)
    }

    #[test]
    fn smoke_initializes_and_persists_settings() {
        let ctx = egui::Context::default();
        let temp = tempdir().expect("tempdir");
        let settings_path = temp.path().join("config").join("gui_settings_smoke.json");

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
