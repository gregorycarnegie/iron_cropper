use crate::types::YuNetApp;
use egui::{Color32, RichText, Sense, Ui, vec2};
use std::collections::BTreeMap;
use std::path::PathBuf;
use yunet_core::{CropSettings as CoreCropSettings, PositioningMode, preset_by_name};
use yunet_utils::{RgbaColor, enhance::EnhancementSettings, quality::Quality};

impl YuNetApp {
    pub(crate) fn quality_legend(&self, ui: &mut Ui, palette: crate::theme::Palette) {
        ui.horizontal(|ui| {
            self.legend_dot(ui, palette, palette.success, "High");
            ui.add_space(8.0);
            self.legend_dot(ui, palette, palette.warning, "Medium");
            ui.add_space(8.0);
            self.legend_dot(ui, palette, palette.danger, "Low");
        });
    }

    pub(crate) fn legend_dot(
        &self,
        ui: &mut Ui,
        palette: crate::theme::Palette,
        color: Color32,
        label: &str,
    ) {
        let (rect, _) = ui.allocate_exact_size(vec2(12.0, 12.0), Sense::hover());
        ui.painter().circle_filled(rect.center(), 5.0, color);
        ui.add_space(4.0);
        ui.label(RichText::new(label).color(palette.subtle_text));
    }

    /// Applies an enhancement preset to the current settings.
    pub(crate) fn apply_enhancement_preset(&mut self) {
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

    pub(crate) fn resolved_output_dimensions(&self) -> (u32, u32) {
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
    pub(crate) fn build_crop_settings(&self) -> CoreCropSettings {
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
            fill_color: self.settings.crop.fill_color,
        }
    }

    /// Builds EnhancementSettings from the GUI settings.
    pub(crate) fn build_enhancement_settings(&self) -> EnhancementSettings {
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

    pub(crate) fn quality_suffix(&self, quality: Quality) -> Option<&'static str> {
        if !self.settings.crop.quality_rules.quality_suffix {
            return None;
        }
        match quality {
            Quality::High => Some("_highq"),
            Quality::Medium => Some("_medq"),
            Quality::Low => Some("_lowq"),
        }
    }

    pub(crate) fn format_metadata_tags(map: &BTreeMap<String, String>) -> String {
        map.iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub(crate) fn format_fill_color_hex(color: RgbaColor) -> String {
        if color.alpha != 255 {
            format!(
                "#{:02X}{:02X}{:02X}{:02X}",
                color.red, color.green, color.blue, color.alpha
            )
        } else {
            format!("#{:02X}{:02X}{:02X}", color.red, color.green, color.blue)
        }
    }

    pub(crate) fn refresh_fill_color_hex_input(&mut self) {
        self.crop_fill_hex_input = Self::format_fill_color_hex(self.settings.crop.fill_color);
    }

    pub(crate) fn set_fill_color(&mut self, color: RgbaColor) -> bool {
        if self.settings.crop.fill_color == color {
            return false;
        }
        self.settings.crop.fill_color = color;
        self.refresh_fill_color_hex_input();
        true
    }

    pub(crate) fn display_with_output_ext(raw: &str, ext: &str) -> String {
        let mut path = PathBuf::from(raw);
        let trimmed = ext.trim().trim_start_matches('.');
        if trimmed.is_empty() {
            return path.display().to_string();
        }
        path.set_extension(trimmed);
        path.display().to_string()
    }

    pub(crate) fn refresh_metadata_tags_input(&mut self) {
        self.metadata_tags_input =
            Self::format_metadata_tags(&self.settings.crop.metadata.custom_tags);
    }

    pub(crate) fn push_crop_history(&mut self) {
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

    pub(crate) fn undo_crop_settings(&mut self) {
        if self.crop_history_index == 0 {
            return;
        }
        self.crop_history_index -= 1;
        self.settings.crop = self.crop_history[self.crop_history_index].clone();
        self.clear_crop_preview_cache();
        self.refresh_fill_color_hex_input();
        self.persist_settings_with_feedback();
        self.apply_quality_rules_to_preview();
        self.refresh_metadata_tags_input();
    }

    pub(crate) fn redo_crop_settings(&mut self) {
        if self.crop_history_index + 1 >= self.crop_history.len() {
            return;
        }
        self.crop_history_index += 1;
        self.settings.crop = self.crop_history[self.crop_history_index].clone();
        self.clear_crop_preview_cache();
        self.refresh_fill_color_hex_input();
        self.persist_settings_with_feedback();
        self.apply_quality_rules_to_preview();
        self.refresh_metadata_tags_input();
    }

    pub(crate) fn adjust_horizontal_offset(&mut self, delta: f32) {
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

    pub(crate) fn adjust_vertical_offset(&mut self, delta: f32) {
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

    pub(crate) fn adjust_face_height(&mut self, delta: f32) {
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

    pub(crate) fn set_crop_preset(&mut self, preset: &str) {
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

    /// Applies the model path from the text input field.
    pub(crate) fn apply_model_path_input(&mut self) {
        let trimmed = self.model_path_input.trim().to_owned();
        self.model_path_input = trimmed.clone();
        self.update_model_path(if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        });
    }

    /// Updates the model path in the settings and invalidates the detector.
    pub(crate) fn update_model_path(&mut self, new_path: Option<String>) {
        use log::info;
        use yunet_utils::config::AppSettings;

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
    pub(crate) fn open_image_dialog(&mut self) {
        use rfd::FileDialog;

        if let Some(path) = FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp"])
            .pick_file()
        {
            self.start_detection(path);
        }
    }

    /// Opens a file dialog to select an ONNX model.
    pub(crate) fn open_model_dialog(&mut self) {
        use rfd::FileDialog;

        if let Some(path) = FileDialog::new()
            .add_filter("ONNX model", &["onnx"])
            .pick_file()
        {
            let display = path.to_string_lossy().to_string();
            self.update_model_path(Some(display));
        }
    }

    /// Opens a file dialog to select multiple images for batch processing.
    pub(crate) fn open_batch_dialog(&mut self) {
        use crate::types::{BatchFile, BatchFileStatus};
        use log::info;
        use rfd::FileDialog;

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

    /// Enqueues additional image paths for batch processing, deduplicating existing entries.
    pub(crate) fn enqueue_batch_images(&mut self, paths: Vec<PathBuf>) -> usize {
        use crate::types::{BatchFile, BatchFileStatus};

        let mut added = 0;
        for path in paths {
            if self
                .batch_files
                .iter()
                .any(|existing| existing.path == path)
            {
                continue;
            }
            self.batch_files.push(BatchFile {
                path,
                status: BatchFileStatus::Pending,
                output_override: None,
            });
            added += 1;
        }
        if added > 0 {
            self.batch_current_index = None;
        }
        added
    }
}
