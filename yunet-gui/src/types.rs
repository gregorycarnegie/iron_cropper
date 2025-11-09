//! Type definitions for the YuNet GUI application.

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{Arc, mpsc},
};

use egui::{Context as EguiContext, Rect, TextureHandle};
use image::DynamicImage;
use yunet_core::{BoundingBox, CropSettings as CoreCropSettings, Detection};
use yunet_utils::{
    OutputOptions, WgpuEnhancer,
    config::{AppSettings, CropSettings as ConfigCropSettings, ResizeQuality},
    enhance::EnhancementSettings,
    gpu::{GpuContext, GpuStatusIndicator},
    mapping::{
        ColumnSelector, MappingCatalog, MappingEntry, MappingFormat, MappingPreview,
        MappingReadOptions,
    },
    quality::Quality,
};

/// Status of a batch file being processed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchFileStatus {
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
pub struct BatchFile {
    pub path: PathBuf,
    pub status: BatchFileStatus,
    pub output_override: Option<PathBuf>,
}

/// Configuration for a batch export job.
#[derive(Clone)]
pub struct BatchJobConfig {
    pub output_dir: PathBuf,
    pub crop_settings: CoreCropSettings,
    pub crop_config: ConfigCropSettings,
    pub enhancement_settings: EnhancementSettings,
    pub enhance_enabled: bool,
    pub output_options: OutputOptions,
    pub gpu_enhancer: Option<Arc<WgpuEnhancer>>,
}

/// Fingerprint of enhancement settings for cache invalidation.
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct EnhancementSignature {
    pub auto_color: bool,
    pub exposure_bits: u32,
    pub brightness: i32,
    pub contrast_bits: u32,
    pub saturation_bits: u32,
    pub unsharp_amount_bits: u32,
    pub unsharp_radius_bits: u32,
    pub sharpness_bits: u32,
    pub skin_smooth_bits: u32,
    pub skin_sigma_space_bits: u32,
    pub skin_sigma_color_bits: u32,
    pub red_eye_removal: bool,
    pub red_eye_threshold_bits: u32,
    pub background_blur: bool,
    pub background_blur_radius_bits: u32,
    pub background_blur_mask_bits: u32,
}

/// Fingerprint of shape settings for cache invalidation.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct ShapeSignature {
    pub kind: u8,
    pub primary_bits: u32,
    pub secondary_bits: u32,
    pub sides: u8,
    pub rotation_bits: u32,
}

/// Cache key for crop preview images.
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct CropPreviewKey {
    pub path: PathBuf,
    pub face_index: usize,
    pub output_width: u32,
    pub output_height: u32,
    pub positioning_mode: u8,
    pub face_height_bits: u32,
    pub horizontal_bits: u32,
    pub vertical_bits: u32,
    pub shape: ShapeSignature,
    pub enhancement: EnhancementSignature,
    pub enhance_enabled: bool,
}

/// Cached crop preview entry.
pub struct CropPreviewCacheEntry {
    pub image: Arc<DynamicImage>,
    pub texture: Option<TextureHandle>,
}

/// The main application state for the YuNet GUI.
pub struct YuNetApp {
    /// User-configurable settings.
    pub settings: AppSettings,
    /// Path to the settings file on disk.
    pub settings_path: PathBuf,
    /// The current status message displayed in the top bar.
    pub status_line: String,
    /// The last error message, if any.
    pub last_error: Option<String>,
    /// Snapshot of GPU availability/fallback status.
    pub gpu_status: GpuStatusIndicator,
    /// Active GPU context, if initialized.
    pub gpu_context: Option<Arc<GpuContext>>,
    /// Shared GPU enhancer used for crop previews/exports.
    pub gpu_enhancer: Option<Arc<WgpuEnhancer>>,
    /// The face detector instance.
    pub detector: Option<Arc<yunet_core::YuNetDetector>>,
    /// Sender for submitting detection jobs to a background thread.
    pub job_tx: mpsc::Sender<JobMessage>,
    /// Receiver for collecting results from detection jobs.
    pub job_rx: mpsc::Receiver<JobMessage>,
    /// State of the image preview panel.
    pub preview: PreviewState,
    /// Cache for detection results to avoid re-running on the same image and settings.
    pub cache: HashMap<CacheKey, DetectionCacheEntry>,
    /// Cached cropped previews keyed by face + crop/enhancement configuration.
    pub crop_preview_cache: HashMap<CropPreviewKey, CropPreviewCacheEntry>,
    /// The current value of the model path text input.
    pub model_path_input: String,
    /// Flag indicating if the model path input has been modified.
    pub model_path_dirty: bool,
    /// Flag indicating if a detection job is currently running.
    pub is_busy: bool,
    /// A counter to generate unique texture names.
    pub texture_seq: u64,
    /// A counter to generate unique job IDs.
    pub job_counter: u64,
    /// The ID of the currently running detection job.
    pub current_job: Option<u64>,
    /// Flag indicating whether to show crop region overlays.
    pub show_crop_overlay: bool,
    /// Set of selected face indices (for cropping).
    pub selected_faces: HashSet<usize>,
    /// Undo/redo stack for crop configuration.
    pub crop_history: Vec<ConfigCropSettings>,
    pub crop_history_index: usize,
    /// Editable metadata tags input cached for the UI.
    pub metadata_tags_input: String,
    /// Batch mode state.
    pub batch_files: Vec<BatchFile>,
    /// Current index in batch processing.
    pub batch_current_index: Option<usize>,
    /// Mapping import workflow state.
    pub mapping: MappingUiState,
    /// Normalized anchor (0-1) describing the HUD offset inside the preview image.
    pub preview_hud_anchor: egui::Vec2,
    /// Whether the preview HUD content is collapsed.
    pub preview_hud_minimized: bool,
    /// Drag origin for HUD repositioning.
    pub preview_hud_drag_origin: Option<egui::Pos2>,
    /// Whether manual bounding-box draw mode is active.
    pub manual_box_tool_enabled: bool,
    /// In-progress manual bounding box draft in image coordinates.
    pub manual_box_draft: Option<ManualBoxDraft>,
    /// Currently active drag handle for adjusting a bounding box.
    pub active_bbox_drag: Option<ActiveBoxDrag>,
}

/// Indicates whether a bounding box originated from the detector or the user.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DetectionOrigin {
    Detector,
    Manual,
}

/// A detection with associated quality score and thumbnail.
#[derive(Clone)]
pub struct DetectionWithQuality {
    /// The core detection data.
    pub detection: Detection,
    /// Laplacian variance (sharpness score).
    pub quality_score: f64,
    /// Quality level (Low, Medium, High).
    pub quality: Quality,
    /// Thumbnail texture handle for the face region.
    pub thumbnail: Option<TextureHandle>,
    /// Active bounding box (may diverge from the detector output).
    pub current_bbox: BoundingBox,
    /// Original bounding box (detector or initial manual draft).
    pub original_bbox: BoundingBox,
    /// Where this bounding box came from.
    pub origin: DetectionOrigin,
}

impl DetectionWithQuality {
    pub fn active_bbox(&self) -> BoundingBox {
        self.current_bbox
    }

    pub fn reset_bbox(&mut self) {
        self.current_bbox = self.original_bbox;
    }

    pub fn set_bbox(&mut self, bbox: BoundingBox) {
        self.current_bbox = bbox;
    }

    pub fn is_manual(&self) -> bool {
        matches!(self.origin, DetectionOrigin::Manual)
    }

    pub fn is_modified(&self) -> bool {
        self.current_bbox != self.original_bbox
    }
}

/// In-progress manual bounding box being drawn.
#[derive(Clone, Copy)]
pub struct ManualBoxDraft {
    pub start: egui::Pos2,
    pub current: egui::Pos2,
}

/// Active drag operation on a bounding box.
#[derive(Clone, Copy)]
pub struct ActiveBoxDrag {
    pub index: usize,
    pub handle: DragHandle,
    pub start_bbox: BoundingBox,
}

/// Handle for dragging/resizing a bounding box.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DragHandle {
    Move,
    NorthWest,
    NorthEast,
    SouthWest,
    SouthEast,
}

/// Snapshot of pointer state for interaction handling.
#[derive(Clone, Copy, Default)]
pub struct PointerSnapshot {
    pub pressed: bool,
    pub released: bool,
    pub down: bool,
    pub press_origin: Option<egui::Pos2>,
    pub pos: Option<egui::Pos2>,
}

impl PointerSnapshot {
    pub fn capture(ctx: &EguiContext) -> Self {
        ctx.input(|input| PointerSnapshot {
            pressed: input.pointer.primary_pressed(),
            released: input.pointer.primary_released(),
            down: input.pointer.primary_down(),
            press_origin: input.pointer.press_origin(),
            pos: input.pointer.interact_pos(),
        })
    }
}

/// Screen space metadata for preview panel.
#[derive(Clone, Copy)]
pub struct PreviewSpace {
    pub rect: Rect,
    pub image_size: (u32, u32),
}

/// State related to the image preview panel.
#[derive(Default)]
pub struct PreviewState {
    /// The path to the currently displayed image.
    pub image_path: Option<PathBuf>,
    /// The handle to the egui texture for the image.
    pub texture: Option<TextureHandle>,
    /// The dimensions of the original image.
    pub image_size: Option<(u32, u32)>,
    /// The list of detections with quality scores for the current image.
    pub detections: Vec<DetectionWithQuality>,
    /// Whether the preview is currently loading/detecting.
    pub is_loading: bool,
    /// Cached original image pixels for computing previews without disk I/O.
    pub source_image: Option<Arc<DynamicImage>>,
}

impl PreviewState {
    /// Resets the preview state to a loading state for a new image.
    pub fn begin_loading(&mut self, path: PathBuf) {
        self.image_path = Some(path);
        self.texture = None;
        self.image_size = None;
        self.detections.clear();
        self.is_loading = true;
        self.source_image = None;
    }
}

/// Mapping import workflow state.
pub struct MappingUiState {
    pub file_path: Option<PathBuf>,
    pub base_dir: Option<PathBuf>,
    pub detected_format: Option<MappingFormat>,
    pub format_override: Option<MappingFormat>,
    pub has_headers: bool,
    pub delimiter_input: String,
    pub sheet_name: String,
    pub sql_table: String,
    pub sql_query: String,
    pub catalog: MappingCatalog,
    pub preview: Option<MappingPreview>,
    pub preview_error: Option<String>,
    pub source_column_idx: Option<usize>,
    pub output_column_idx: Option<usize>,
    pub entries: Vec<MappingEntry>,
}

impl Default for MappingUiState {
    fn default() -> Self {
        Self::new()
    }
}

impl MappingUiState {
    pub fn new() -> Self {
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

    pub fn set_file(&mut self, path: PathBuf) {
        use yunet_utils::mapping::detect_format as detect_mapping_format_utils;

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

    pub fn effective_format(&self) -> Option<MappingFormat> {
        self.format_override.or(self.detected_format)
    }

    pub fn refresh_catalog(&mut self) {
        use yunet_utils::mapping::inspect_mapping_sources;

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

    pub fn read_options(&self) -> MappingReadOptions {
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

    pub fn selected_column_name(&self, idx: Option<usize>) -> String {
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

    pub fn source_selector(&self) -> Option<ColumnSelector> {
        self.source_column_idx.map(ColumnSelector::Index)
    }

    pub fn output_selector(&self) -> Option<ColumnSelector> {
        self.output_column_idx.map(ColumnSelector::Index)
    }

    pub fn reload_preview(&mut self) -> anyhow::Result<()> {
        use anyhow::anyhow;
        use yunet_utils::mapping::load_mapping_preview;

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

    pub fn load_entries(&mut self) -> anyhow::Result<()> {
        use anyhow::anyhow;
        use yunet_utils::mapping::load_mapping_entries;

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

/// A message sent from a background detection job to the GUI thread.
pub enum JobMessage {
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
pub struct DetectionJobSuccess {
    pub path: PathBuf,
    pub color_image: egui::ColorImage,
    pub detections: Vec<DetectionWithQuality>,
    pub original_size: (u32, u32),
    pub original_image: Arc<DynamicImage>,
}

/// A key used to cache detection results.
#[derive(Hash, PartialEq, Eq, Clone)]
pub struct CacheKey {
    pub path: PathBuf,
    pub model_path: Option<String>,
    pub input_width: u32,
    pub input_height: u32,
    pub resize_quality: ResizeQuality,
    pub score_bits: u32,
    pub nms_bits: u32,
    pub top_k: usize,
}

/// An entry in the detection cache.
pub struct DetectionCacheEntry {
    pub texture: TextureHandle,
    pub detections: Vec<DetectionWithQuality>,
    pub original_size: (u32, u32),
    pub source_image: Arc<DynamicImage>,
}
