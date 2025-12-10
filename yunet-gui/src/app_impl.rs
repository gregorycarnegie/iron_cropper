//! Additional YuNetApp implementation methods that delegate to module functions.

use crate::PointerSnapshot;
use crate::core::{
    cache, cache::CropPreviewRequest, detection, detection::create_thumbnails, export, quality,
};
use crate::interaction::{bbox_drag, drawing, shortcuts};
use crate::{DetectionCacheEntry, DetectionJobSuccess, GpuStatusMode, JobMessage, YuNetApp};

use crate::core::detection::build_detector;
use crate::core::settings::load_settings;
use crate::types::{
    BatchFile, BatchFileStatus, ColorMode, MappingUiState, WebcamState, WebcamStatus,
};
use anyhow::{Context, anyhow};
#[cfg(not(target_arch = "wasm32"))]
use arboard::{Clipboard, Error as ClipboardError};
use eframe::{App, CreationContext, Frame};
use egui::{Context as EguiContext, DroppedFile, Event, TextureHandle, TextureOptions};
use egui_extras::{Size, StripBuilder};
use image::{DynamicImage, RgbaImage};
use log::info;
use std::sync::{Arc, mpsc};
use std::{
    collections::VecDeque,
    fs,
    path::{Path, PathBuf},
};
use tempfile::Builder;
use yunet_utils::{
    WgpuEnhancer,
    config::default_settings_path,
    configure_telemetry,
    gpu::{GpuBatchCropper, GpuContext, GpuStatusIndicator},
};

type GpuPipelineInit = (
    Option<Arc<GpuContext>>,
    Option<Arc<WgpuEnhancer>>,
    Option<Arc<GpuBatchCropper>>,
);

impl YuNetApp {
    /// Polls the worker thread for completed detection jobs.
    pub(crate) fn poll_worker(&mut self, ctx: &EguiContext) {
        let mut updated = false;
        while let Ok(message) = self.job_rx.try_recv() {
            self.handle_job_message(ctx, message);
            updated = true;
        }

        if updated {
            ctx.request_repaint();
        }
    }

    /// Applies quality rules to the preview detections.
    pub(crate) fn apply_quality_rules_to_preview(&mut self) {
        quality::apply_quality_rules_to_preview(
            &self.preview.detections,
            &mut self.selected_faces,
            self.settings.crop.quality_rules.auto_skip_no_high_quality,
            self.settings.crop.quality_rules.auto_select_best_face,
        );
    }

    /// Refreshes the thumbnail for a specific detection.
    pub(crate) fn refresh_detection_thumbnail_at(&mut self, ctx: &EguiContext, index: usize) {
        if let Some(source) = &self.preview.source_image {
            quality::refresh_detection_thumbnail_at(
                ctx,
                &mut self.preview.detections,
                index,
                source,
                &mut self.texture_seq,
            );
        }
    }

    /// Resets a detection's bounding box to its original state.
    pub(crate) fn reset_detection_bbox(&mut self, ctx: &EguiContext, index: usize) {
        quality::reset_detection_bbox(&mut self.preview.detections, index);
        self.refresh_detection_thumbnail_at(ctx, index);
    }

    /// Removes a detection from the preview.
    pub(crate) fn remove_detection(&mut self, index: usize) {
        quality::remove_detection(
            &mut self.preview.detections,
            &mut self.selected_faces,
            index,
        );
    }

    /// Exports the selected faces to disk.
    pub(crate) fn export_selected_faces(&mut self) {
        export::export_selected_faces(self);
    }

    /// Starts batch export processing.
    pub(crate) fn start_batch_export(&mut self) {
        export::start_batch_export(self);
    }

    /// Handles all preview panel interactions (drawing, dragging).
    pub(crate) fn handle_preview_interactions(
        &mut self,
        ctx: &EguiContext,
        preview_rect: egui::Rect,
        image_size: (u32, u32),
    ) {
        let preview_space = crate::PreviewSpace {
            rect: preview_rect,
            image_size,
        };
        let pointer = PointerSnapshot::capture(ctx);

        // Handle manual box drawing
        if self.manual_box_tool_enabled {
            drawing::update_manual_box_draft(
                &mut self.manual_box_draft,
                &mut self.preview.detections,
                &pointer,
                &preview_space,
            );
        }

        // Handle bbox dragging
        if !self.manual_box_tool_enabled {
            bbox_drag::handle_bbox_drag_interactions(
                &mut self.active_bbox_drag,
                &mut self.preview.detections,
                &pointer,
                &preview_space,
            );
        }
    }

    /// Starts a face detection job for the given image path.
    pub(crate) fn start_detection(&mut self, path: PathBuf) {
        let cache_key = detection::cache_key_for_path(&path, &self.settings);

        if let Some(cached) = self.cache.get(&cache_key) {
            // Use cached result
            self.preview.image_path = Some(path);
            self.preview.texture = Some(cached.texture.clone());
            self.preview.image_size = Some(cached.original_size);
            self.preview.detections = cached.detections.clone();
            self.preview.source_image = Some(cached.source_image.clone());
            self.preview.is_loading = false;
            self.apply_quality_rules_to_preview();
            return;
        }

        let (gpu_status_update, gpu_context_update, detector_result) = detection::ensure_detector(
            &mut self.detector,
            &self.settings,
            self.gpu_context.clone(),
        );
        if let Some(status) = gpu_status_update.as_ref() {
            self.gpu_status = status.clone();
        }
        if let Some(context) = gpu_context_update {
            self.refresh_gpu_pipelines(Some(context));
        } else if let Some(status) = gpu_status_update
            && !matches!(
                status.mode,
                GpuStatusMode::Available | GpuStatusMode::Pending
            )
        {
            self.refresh_gpu_pipelines(None);
        }
        let detector = match detector_result {
            Ok(detector) => detector,
            Err(err) => {
                self.show_error(
                    "Detection failed",
                    format!("Unable to load detector: {err}"),
                );
                return;
            }
        };

        self.preview.begin_loading(path.clone());
        self.is_busy = true;
        self.job_counter += 1;
        let job_id = self.job_counter;
        self.current_job = Some(job_id);

        detection::start_detection(path, cache_key, detector, job_id, self.job_tx.clone());
    }

    /// Gets or creates a texture for a crop preview.
    pub(crate) fn crop_preview_texture_for(
        &mut self,
        ctx: &EguiContext,
        face_idx: usize,
    ) -> Option<TextureHandle> {
        let path = self.preview.image_path.as_ref()?;
        let detection = self.preview.detections.get(face_idx)?;
        let crop_settings = self.build_crop_settings();
        let enhancement_settings = self.build_enhancement_settings();

        let request = CropPreviewRequest {
            path,
            face_idx,
            detection,
            source_image: &mut self.preview.source_image,
            image_cache: &mut self.image_cache,
            crop_settings: &crop_settings,
            crop_config: &self.settings.crop,
            enhancement_settings: &enhancement_settings,
            enhance_enabled: self.settings.enhance.enabled,
            gpu_enhancer: self.gpu_enhancer.clone(),
            gpu_cropper: self.gpu_batch_cropper.clone(),
        };

        cache::crop_preview_texture_for(
            ctx,
            &mut self.crop_preview_cache,
            request,
            &mut self.texture_seq,
        )
    }

    /// Consumes clipboard paste events and file drops to load images quickly.
    pub(crate) fn process_import_payloads(&mut self, ctx: &EguiContext) {
        let dropped_files = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped_files.is_empty() {
            let handled = self.handle_dropped_images(dropped_files);
            ctx.input_mut(|i| i.raw.dropped_files.clear());
            if handled {
                return;
            }
        }

        if ctx.wants_keyboard_input() {
            return;
        }

        let events = ctx.input(|i| i.events.clone());
        for event in events {
            if let Event::Paste(text) = event
                && self.handle_paste_text(&text)
            {
                return;
            }
        }
    }

    fn handle_dropped_images(&mut self, files: Vec<DroppedFile>) -> bool {
        if files.is_empty() {
            return false;
        }
        for file in files {
            match self.try_process_dropped_file(&file) {
                Ok(true) => return true,
                Ok(false) => continue,
                Err(err) => {
                    self.show_error("Drop failed", err.to_string());
                    return true;
                }
            }
        }
        self.show_error(
            "Unsupported drop",
            "Drop files, folders, or clipboard images to load content.",
        );
        false
    }

    fn try_process_dropped_file(&mut self, file: &DroppedFile) -> anyhow::Result<bool> {
        if let Some(path) = file.path.as_ref() {
            if path.is_dir() {
                let images = Self::collect_images_from_directory(path)?;
                return Ok(self.enqueue_batch_paths(images));
            }
            if Self::is_supported_mapping_path(path) {
                self.load_mapping_from_path(path.to_path_buf())?;
                return Ok(true);
            }
            if Self::is_supported_image_path(path) {
                self.start_detection(path.to_path_buf());
                return Ok(true);
            }
            return Ok(false);
        }
        if let Some(bytes) = file.bytes.as_deref() {
            let label = if file.name.is_empty() {
                None
            } else {
                Some(file.name.as_str())
            };
            self.consume_image_bytes(bytes, label)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn handle_paste_text(&mut self, text: &str) -> bool {
        let mut file_paths = Vec::new();
        let mut mapping_paths = Vec::new();
        let mut dir_paths = Vec::new();
        for candidate in Self::paths_from_clipboard_text(text) {
            if candidate.is_dir() {
                dir_paths.push(candidate);
            } else if Self::is_supported_image_path(&candidate) {
                file_paths.push(candidate);
            } else if Self::is_supported_mapping_path(&candidate) {
                mapping_paths.push(candidate);
            }
        }

        if !dir_paths.is_empty() {
            let mut aggregated = Vec::new();
            for dir in dir_paths {
                match Self::collect_images_from_directory(&dir) {
                    Ok(mut images) => aggregated.append(&mut images),
                    Err(err) => {
                        self.show_error(
                            "Paste failed",
                            format!("Failed to inspect {}: {err}", dir.display()),
                        );
                        return true;
                    }
                }
            }
            if self.enqueue_batch_paths(aggregated) {
                return true;
            }
        }

        if let Some(mapping) = mapping_paths.into_iter().next() {
            if let Err(err) = self.load_mapping_from_path(mapping.clone()) {
                self.show_error(
                    "Paste failed",
                    format!("Failed to load mapping {}: {err}", mapping.display()),
                );
            }
            return true;
        }

        if file_paths.len() > 1 {
            if self.enqueue_batch_paths(file_paths) {
                return true;
            }
            return false;
        }

        if let Some(single) = file_paths.into_iter().next() {
            self.start_detection(single);
            return true;
        }

        match self.try_paste_image_from_clipboard() {
            Ok(true) => true,
            Ok(false) => false,
            Err(err) => {
                self.show_error("Paste failed", err.to_string());
                true
            }
        }
    }

    fn consume_image_bytes(&mut self, bytes: &[u8], label: Option<&str>) -> anyhow::Result<()> {
        let image = image::load_from_memory(bytes).with_context(|| {
            format!(
                "Failed to decode dropped image {}",
                label.unwrap_or_default()
            )
        })?;
        self.persist_and_start_from_dynamic(image)
    }

    fn persist_and_start_from_dynamic(&mut self, image: DynamicImage) -> anyhow::Result<()> {
        let path = self.persist_clipboard_image(&image)?;
        self.start_detection(path);
        Ok(())
    }

    fn persist_clipboard_image(&mut self, image: &DynamicImage) -> anyhow::Result<PathBuf> {
        let temp = Builder::new()
            .prefix("iron_cropper_clip_")
            .suffix(".png")
            .tempfile()
            .context("Failed to create temporary file for clipboard image")?;
        image
            .save(temp.path())
            .context("Failed to encode clipboard image")?;
        let temp_path = temp.into_temp_path();
        let path_buf = temp_path.to_path_buf();
        self.clipboard_temp_images.push(temp_path);
        if self.clipboard_temp_images.len() > MAX_CLIPBOARD_IMAGES {
            self.clipboard_temp_images.remove(0);
        }
        Ok(path_buf)
    }

    pub(crate) fn enqueue_batch_paths(&mut self, paths: Vec<PathBuf>) -> bool {
        if paths.is_empty() {
            return false;
        }
        let added = self.enqueue_batch_images(paths);
        if added > 0 {
            let total = self.batch_files.len();
            self.show_success(format!(
                "Added {added} file(s) to the batch queue ({} total)",
                total
            ));
            true
        } else {
            self.show_success("All dropped files were already in the batch queue.");
            false
        }
    }

    pub(crate) fn collect_images_from_directory(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
        let mut images = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(dir.to_path_buf());

        while let Some(current) = queue.pop_front() {
            let entries = fs::read_dir(&current)
                .with_context(|| format!("Failed to read directory {}", current.display()))?;
            for entry in entries {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    queue.push_back(path);
                } else if Self::is_supported_image_path(&path) {
                    images.push(path);
                }
            }
        }

        Ok(images)
    }

    fn load_mapping_from_path(&mut self, path: PathBuf) -> anyhow::Result<()> {
        self.load_mapping_file(path);
        Ok(())
    }

    fn is_supported_mapping_path(path: &Path) -> bool {
        if !path.is_file() {
            return false;
        }
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| {
                matches!(
                    ext.to_ascii_lowercase().as_str(),
                    "csv"
                        | "tsv"
                        | "txt"
                        | "dat"
                        | "xlsx"
                        | "xls"
                        | "xlsm"
                        | "ods"
                        | "parquet"
                        | "pq"
                        | "db"
                        | "sqlite"
                        | "sqlite3"
                )
            })
            .unwrap_or(false)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn try_paste_image_from_clipboard(&mut self) -> anyhow::Result<bool> {
        let mut clipboard =
            Clipboard::new().context("Clipboard access is unavailable on this platform")?;
        match clipboard.get_image() {
            Ok(image) => {
                let width = image.width as u32;
                let height = image.height as u32;
                let rgba = image.bytes.into_owned();
                let buffer = RgbaImage::from_raw(width, height, rgba)
                    .context("Clipboard image had unexpected dimensions")?;
                let dynamic = DynamicImage::ImageRgba8(buffer);
                self.persist_and_start_from_dynamic(dynamic)?;
                Ok(true)
            }
            Err(ClipboardError::ContentNotAvailable) => Ok(false),
            Err(err) => Err(anyhow!("Failed to read clipboard image: {err}")),
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn try_paste_image_from_clipboard(&mut self) -> anyhow::Result<bool> {
        Ok(false)
    }

    fn paths_from_clipboard_text(text: &str) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        if let Some(path) = Self::normalize_clipboard_path(text) {
            paths.push(path);
        }
        for line in text.lines() {
            if let Some(path) = Self::normalize_clipboard_path(line) {
                paths.push(path);
            }
        }
        paths
    }

    fn normalize_clipboard_path(fragment: &str) -> Option<PathBuf> {
        let trimmed = fragment.trim().trim_matches(['\"', '\'']).trim();
        if trimmed.is_empty() {
            return None;
        }
        let without_scheme = if let Some(rest) = trimmed.strip_prefix("file://") {
            if cfg!(windows) {
                rest.trim_start_matches('/')
            } else {
                rest
            }
        } else {
            trimmed
        };
        let candidate = PathBuf::from(without_scheme);
        if candidate.exists() {
            Some(candidate)
        } else {
            None
        }
    }

    fn is_supported_image_path(path: &Path) -> bool {
        if !path.is_file() {
            return false;
        }
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| {
                let lower = ext.to_ascii_lowercase();
                matches!(lower.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "webp")
            })
            .unwrap_or(false)
    }

    /// Handles keyboard shortcuts.
    pub fn handle_shortcuts(&mut self, ctx: &EguiContext) {
        let wants_text = ctx.wants_keyboard_input();
        let actions = shortcuts::capture_shortcut_actions(ctx, wants_text);

        if actions.export && !self.selected_faces.is_empty() && !self.preview.detections.is_empty()
        {
            self.export_selected_faces();
        }
        if actions.undo {
            self.undo_crop_settings();
        }
        if actions.redo {
            self.redo_crop_settings();
        }
        if actions.horiz_delta != 0.0 {
            self.adjust_horizontal_offset(actions.horiz_delta);
        }
        if actions.vert_delta != 0.0 {
            self.adjust_vertical_offset(actions.vert_delta);
        }
        if actions.face_height_delta != 0.0 {
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
    }

    /// Handles a message from a detection job.
    pub(crate) fn handle_job_message(&mut self, ctx: &EguiContext, message: JobMessage) {
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
                create_thumbnails(ctx, &mut detections, &original_image, &mut self.texture_seq);

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
                        detections: cached_detections,
                        original_size,
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
            JobMessage::WebcamFrame {
                image,
                frame_number,
                detections,
            } => {
                self.process_webcam_frame(ctx, image, frame_number, detections);
            }
            JobMessage::WebcamError(error) => {
                self.handle_webcam_error(error);
            }
            JobMessage::WebcamStopped => {
                self.handle_webcam_stopped();
            }
            JobMessage::BatchProgress { index, status } => {
                if let Some(file) = self.batch_files.get_mut(index) {
                    file.status = status;
                }
            }
            JobMessage::BatchComplete { completed, failed } => {
                self.show_success(format!(
                    "Batch export complete: {} succeeded, {} failed",
                    completed, failed
                ));
            }
        }
    }
}

impl YuNetApp {
    /// Creates a new `YuNetApp` instance.
    pub fn new(cc: &CreationContext<'_>) -> Self {
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
        ctx: &EguiContext,
        settings_path: PathBuf,
        shared_gpu_context: Option<Arc<GpuContext>>,
    ) -> Self {
        crate::theme::apply(ctx);

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
                log::warn!("Unable to initialize YuNet model: {err}");
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
            cache: std::collections::HashMap::new(),
            crop_preview_cache: std::collections::HashMap::new(),
            image_cache: std::collections::HashMap::new(),
            model_path_input,
            model_path_dirty: false,
            is_busy: false,
            texture_seq: 0,
            job_counter: 0,
            current_job: None,
            show_crop_overlay: true,
            selected_faces: std::collections::HashSet::new(),
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
            clipboard_temp_images: Vec::new(),
            show_settings_window: false,
            show_batch_window: false,
            show_mapping_window: false,
            show_detection_window: false,
            webcam_state: WebcamState::default(),
            icons: crate::ui::icons::IconSet::new(ctx),
            fill_color_mode: ColorMode::Rgb,
            aspect_ratio_locked: false,
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
                    log::warn!("Failed to initialize GPU enhancer: {err}");
                    None
                }
            };
            let cropper = match GpuBatchCropper::new(ctx.clone()) {
                Ok(cropper) => Some(Arc::new(cropper)),
                Err(err) => {
                    log::warn!("Failed to initialize GPU batch cropper: {err}");
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
    fn update(&mut self, ctx: &EguiContext, _frame: &mut Frame) {
        self.process_import_payloads(ctx);
        self.poll_worker(ctx);
        self.show_status_bar(ctx);

        // Request continuous repaints when webcam is active
        if matches!(
            self.webcam_state.status,
            WebcamStatus::Active | WebcamStatus::Starting
        ) {
            ctx.request_repaint();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            StripBuilder::new(ui)
                .size(Size::exact(283.0)) // Navigation
                .size(Size::remainder()) // Preview
                .size(Size::exact(420.0)) // Configuration
                .horizontal(|mut strip| {
                    strip.cell(|ui| {
                        // Navigation Panel Frame
                        let palette = crate::theme::palette();
                        egui::Frame::new()
                            .fill(palette.panel)
                            .inner_margin(egui::Margin::symmetric(16, 18))
                            .stroke(egui::Stroke::new(1.0, palette.outline))
                            .show(ui, |ui| {
                                self.show_navigation_panel(ui);
                            });
                    });
                    strip.cell(|ui| {
                        // Preview Panel Frame
                        let palette = crate::theme::palette();
                        egui::Frame::new()
                            .fill(palette.canvas)
                            .inner_margin(egui::Margin::symmetric(16, 16))
                            .show(ui, |ui| {
                                self.show_preview(ui, ctx);
                            });
                    });
                    strip.cell(|ui| {
                        // Configuration Panel Frame
                        let palette = crate::theme::palette();
                        egui::Frame::new()
                            .fill(palette.panel)
                            .stroke(egui::Stroke::new(1.0, palette.outline))
                            .inner_margin(egui::Margin::symmetric(16, 18))
                            .show(ui, |ui| {
                                crate::ui::config::panel::show_configuration_panel(self, ui, ctx);
                            });
                    });
                });
        });

        if self.show_settings_window {
            crate::ui::settings_window::show_settings_window(self, ctx);
        }
        if self.show_batch_window {
            crate::ui::batch_window::show_batch_window(self, ctx);
        }
        if self.show_mapping_window {
            self.show_mapping_window(ctx);
        }
        if self.show_detection_window {
            crate::ui::config::detections::show_detection_window(self, ctx);
        }

        self.handle_shortcuts(ctx);

        if self.is_busy {
            ctx.request_repaint();
        }
    }
}

const MAX_CLIPBOARD_IMAGES: usize = 8;

#[cfg(test)]
mod tests {
    use super::*;
    use egui;
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
