//! Additional YuNetApp implementation methods that delegate to module functions.

use crate::{GpuStatusMode, YuNetApp};
use anyhow::{Context, anyhow};
#[cfg(not(target_arch = "wasm32"))]
use arboard::{Clipboard, Error as ClipboardError};
use egui::{Context as EguiContext, DroppedFile, Event};
use image::{DynamicImage, RgbaImage};
use std::{
    collections::VecDeque,
    fs,
    path::{Path, PathBuf},
};
use tempfile::Builder;

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
        use crate::core::quality;

        quality::apply_quality_rules_to_preview(
            &self.preview.detections,
            &mut self.selected_faces,
            self.settings.crop.quality_rules.auto_skip_no_high_quality,
            self.settings.crop.quality_rules.auto_select_best_face,
        );
    }

    /// Refreshes the thumbnail for a specific detection.
    pub(crate) fn refresh_detection_thumbnail_at(&mut self, ctx: &EguiContext, index: usize) {
        use crate::core::quality;

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
        use crate::core::quality;

        quality::reset_detection_bbox(&mut self.preview.detections, index);
        self.refresh_detection_thumbnail_at(ctx, index);
    }

    /// Removes a detection from the preview.
    pub(crate) fn remove_detection(&mut self, index: usize) {
        use crate::core::quality;

        quality::remove_detection(
            &mut self.preview.detections,
            &mut self.selected_faces,
            index,
        );
    }

    /// Exports the selected faces to disk.
    pub(crate) fn export_selected_faces(&mut self) {
        use crate::core::export;

        export::export_selected_faces(self);
    }

    /// Starts batch export processing.
    pub(crate) fn start_batch_export(&mut self) {
        use crate::core::export;

        export::start_batch_export(self);
    }

    /// Handles all preview panel interactions (drawing, dragging).
    pub(crate) fn handle_preview_interactions(
        &mut self,
        ctx: &EguiContext,
        preview_rect: egui::Rect,
        image_size: (u32, u32),
    ) {
        use crate::PointerSnapshot;
        use crate::interaction::{bbox_drag, drawing};

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
    pub(crate) fn start_detection(&mut self, path: std::path::PathBuf) {
        use crate::core::detection;

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
    ) -> Option<egui::TextureHandle> {
        use crate::core::cache;
        use crate::core::cache::CropPreviewRequest;

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

    fn enqueue_batch_paths(&mut self, paths: Vec<PathBuf>) -> bool {
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

    fn collect_images_from_directory(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
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
    pub(crate) fn handle_shortcuts(&mut self, ctx: &EguiContext) {
        use crate::interaction::shortcuts;

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
    pub(crate) fn handle_job_message(&mut self, ctx: &EguiContext, message: crate::JobMessage) {
        use crate::core::detection::create_thumbnails;
        use crate::{DetectionCacheEntry, DetectionJobSuccess, JobMessage};
        use egui::TextureOptions;
        use log::info;

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

const MAX_CLIPBOARD_IMAGES: usize = 8;
