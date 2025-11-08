//! Additional YuNetApp implementation methods that delegate to module functions.

use crate::YuNetApp;
use egui::Context as EguiContext;

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

        let detector = match detection::ensure_detector(&mut self.detector, &self.settings) {
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
            crop_settings: &crop_settings,
            crop_config: &self.settings.crop,
            enhancement_settings: &enhancement_settings,
            enhance_enabled: self.settings.enhance.enabled,
        };

        cache::crop_preview_texture_for(
            ctx,
            &mut self.crop_preview_cache,
            request,
            &mut self.texture_seq,
        )
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
        }
    }
}
