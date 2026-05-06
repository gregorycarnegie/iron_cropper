//! Export / batch crop logic.

use crate::types::*;
use anyhow::Result;
use fcs_core::{CropSettings as CoreCropSettings, crop_face_from_image};
use fcs_utils::config::AppSettings;
use image::DynamicImage;
use log::info;
use std::path::PathBuf;

/// Export selected faces to a directory.
pub fn export_faces(
    source: &DynamicImage,
    detections: &[DetectionWithQuality],
    selected: &[usize],
    settings: &AppSettings,
    output_dir: &PathBuf,
) -> Result<Vec<PathBuf>> {
    let core_crop = CoreCropSettings {
        output_width: settings.crop.output_width,
        output_height: settings.crop.output_height,
        face_height_pct: settings.crop.face_height_pct,
        positioning_mode: fcs_core::PositioningMode::Center,
        horizontal_offset: settings.crop.horizontal_offset,
        vertical_offset: settings.crop.vertical_offset,
        fill_color: fcs_core::FillColor {
            red:   settings.crop.fill_color.red,
            green: settings.crop.fill_color.green,
            blue:  settings.crop.fill_color.blue,
            alpha: 255,
        },
    };

    let mut saved = Vec::new();
    for &idx in selected {
        let det = &detections[idx];
        let crop = crop_face_from_image(source, &det.detection, &core_crop);

        let stem = output_dir.file_stem().and_then(|s| s.to_str()).unwrap_or("crop");
        let filename = format!("{stem}_face{}.jpg", idx + 1);
        let out_path = output_dir.join(&filename);

        crop.save(&out_path)?;
        info!("Saved crop to {}", out_path.display());
        saved.push(out_path);
    }
    Ok(saved)
}
