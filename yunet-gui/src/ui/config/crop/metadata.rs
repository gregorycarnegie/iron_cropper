use crate::YuNetApp;
use egui::{ComboBox, Ui};
use std::collections::BTreeMap;
use yunet_utils::config::MetadataMode;

/// Shows the metadata settings section.
pub fn show_metadata_section(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    metadata_tags_changed: &mut bool,
) {
    ui.separator();
    ui.label("Metadata");

    let mode_label = match app.settings.crop.metadata.mode {
        MetadataMode::Preserve => "Preserve",
        MetadataMode::Strip => "Strip",
        MetadataMode::Custom => "Custom",
    };
    ComboBox::from_id_salt("metadata_mode_combo")
        .selected_text(mode_label)
        .show_ui(ui, |ui| {
            for (value, text) in [
                (MetadataMode::Preserve, "Preserve"),
                (MetadataMode::Strip, "Strip"),
                (MetadataMode::Custom, "Custom"),
            ] {
                if ui
                    .selectable_label(app.settings.crop.metadata.mode == value, text)
                    .clicked()
                {
                    app.settings.crop.metadata.mode = value;
                    *settings_changed = true;
                }
            }
        });

    let mut include_crop = app.settings.crop.metadata.include_crop_settings;
    if ui
        .checkbox(&mut include_crop, "Include crop settings metadata")
        .changed()
    {
        app.settings.crop.metadata.include_crop_settings = include_crop;
        *settings_changed = true;
    }

    let mut include_quality = app.settings.crop.metadata.include_quality_metrics;
    if ui
        .checkbox(&mut include_quality, "Include quality metrics metadata")
        .changed()
    {
        app.settings.crop.metadata.include_quality_metrics = include_quality;
        *settings_changed = true;
    }

    if ui
        .text_edit_multiline(&mut app.metadata_tags_input)
        .changed()
    {
        app.settings.crop.metadata.custom_tags = parse_metadata_tags(&app.metadata_tags_input);
        *settings_changed = true;
        *metadata_tags_changed = true;
    }
    ui.label("Enter custom tags as key=value, one per line.")
        .on_hover_text("Tags are embedded into output metadata when mode is preserve or custom.");
}

/// Parses metadata tags from text input.
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
