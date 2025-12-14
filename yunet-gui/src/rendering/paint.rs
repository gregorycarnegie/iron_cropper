use crate::interaction;
use crate::types::YuNetApp;

use egui::{Color32, Rect, Stroke, StrokeKind, pos2, vec2};
use yunet_core::calculate_crop_region;
use yunet_utils::outline_points_for_rect;

impl YuNetApp {
    /// Paints the detection bounding boxes and landmarks over the preview image.
    pub(crate) fn paint_detections(&self, ui: &egui::Ui, image_rect: Rect, image_size: (u32, u32)) {
        let painter = ui.painter().with_clip_rect(image_rect);
        let scale_x = image_rect.width() / image_size.0 as f32;
        let scale_y = image_rect.height() / image_size.1 as f32;
        let stroke_scale = scale_x.min(scale_y).max(0.1);

        let base_bbox_color = Color32::from_rgb(255, 145, 77);
        let manual_bbox_color = Color32::from_rgb(255, 214, 142);
        let bbox_width = stroke_scale.clamp(0.25, 3.0);
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
                bbox.x.mul_add(scale_x, image_rect.left()),
                bbox.y.mul_add(scale_y, image_rect.top()),
            );
            let size = vec2(bbox.width * scale_x, bbox.height * scale_y);
            let rect = Rect::from_min_size(top_left, size);

            // Draw selection highlight if selected
            if is_selected {
                painter.rect_stroke(rect, 2.0, selected_stroke, StrokeKind::Outside);
            }

            // Draw detection bounding box
            let bbox_color = if det_with_quality.is_manual() {
                manual_bbox_color
            } else {
                base_bbox_color
            };
            let bbox_stroke = Stroke::new(bbox_width, bbox_color);
            painter.rect_stroke(rect, 0.0, bbox_stroke, StrokeKind::Inside);

            if !det_with_quality.is_manual() {
                for landmark in &det_with_quality.detection.landmarks {
                    let center = pos2(
                        landmark.x.mul_add(scale_x, image_rect.left()),
                        landmark.y.mul_add(scale_y, image_rect.top()),
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
                    (crop_region.x as f32).mul_add(scale_x, image_rect.left()),
                    (crop_region.y as f32).mul_add(scale_y, image_rect.top()),
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
                    painter.rect_stroke(crop_rect, 4.0, crop_stroke, StrokeKind::Inside);
                }
            }
        }

        if let Some(draft) = self.manual_box_draft {
            let a = interaction::coords::image_point_to_screen(draft.start, image_rect, image_size);
            let b =
                interaction::coords::image_point_to_screen(draft.current, image_rect, image_size);
            let draft_rect = Rect::from_two_pos(a, b);
            let draft_stroke = Stroke::new(
                (2.5 * stroke_scale).clamp(1.0, 4.0),
                Color32::from_rgb(100, 200, 255),
            );
            painter.rect_stroke(draft_rect, 0.0, draft_stroke, StrokeKind::Inside);
        }
    }

    pub(crate) fn should_show_rule_of_thirds_guides(&self) -> bool {
        if !self.show_crop_overlay {
            return false;
        }
        matches!(
            self.settings.crop.positioning_mode.as_str(),
            "rule-of-thirds" | "rule_of_thirds"
        )
    }

    pub(crate) fn paint_rule_of_thirds_guides(
        &self,
        painter: &egui::Painter,
        image_rect: Rect,
        stroke_scale: f32,
    ) {
        let guide_color = Color32::from_rgba_unmultiplied(255, 255, 255, 110);
        let guide_stroke = Stroke::new((1.5 * stroke_scale).clamp(0.5, 4.0), guide_color);

        for i in 1..=2 {
            let frac = i as f32 / 3.0;
            let x = image_rect.width().mul_add(frac, image_rect.left());
            let y = image_rect.height().mul_add(frac, image_rect.top());
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
    pub(crate) fn apply_settings_changes(
        &mut self,
        requires_detector_reset: bool,
        requires_cache_refresh: bool,
    ) {
        if requires_detector_reset {
            self.invalidate_detector();
            if let Some(path) = self.preview.image_path.clone() {
                self.start_detection(path);
            } else {
                // If no image is loaded, we still MUST rebuild the detector/GPU context
                // so the app is ready for batch operations.
                let (gpu_status_update, gpu_context_update, detector_result) =
                    crate::core::detection::ensure_detector(
                        &mut self.detector,
                        &self.settings,
                        self.gpu_context.clone(),
                    );
                if let Some(status) = gpu_status_update {
                    self.gpu_status = status;
                }
                if let Some(context) = gpu_context_update {
                    self.refresh_gpu_pipelines(Some(context));
                } else if !matches!(
                    self.gpu_status.mode,
                    crate::GpuStatusMode::Available | crate::GpuStatusMode::Pending
                ) {
                    self.refresh_gpu_pipelines(None);
                }

                // Update status line based on result
                self.status_line = match detector_result {
                    Ok(_) => "Model ready. Select an image to run detection.".to_owned(),
                    Err(e) => format!("Model initialization failed: {e}"),
                };
            }
        } else if requires_cache_refresh {
            self.clear_cache("Detection parameters updated. Re-run detection.");
            if let Some(path) = self.preview.image_path.clone() {
                self.start_detection(path);
            }
        }
        self.persist_settings_with_feedback();
    }

    pub(crate) fn show_success(&mut self, message: impl Into<String>) {
        self.status_line = message.into();
        self.last_error = None;
    }

    pub(crate) fn show_error(&mut self, headline: impl Into<String>, detail: impl Into<String>) {
        self.status_line = headline.into();
        self.last_error = Some(detail.into());
    }
}
