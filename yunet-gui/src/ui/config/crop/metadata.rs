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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_key_value_pairs() {
        let result = parse_metadata_tags("author=Alice\ntitle=Portrait");
        assert_eq!(result["author"], "Alice");
        assert_eq!(result["title"], "Portrait");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn trims_whitespace_around_key_and_value() {
        let result = parse_metadata_tags("  author  =  Alice  ");
        assert_eq!(result["author"], "Alice");
    }

    #[test]
    fn skips_empty_lines() {
        let result = parse_metadata_tags("\n\nauthor=Alice\n\n");
        assert_eq!(result.len(), 1);
        assert_eq!(result["author"], "Alice");
    }

    #[test]
    fn skips_lines_without_equals() {
        let result = parse_metadata_tags("not_a_pair\nauthor=Alice");
        assert_eq!(result.len(), 1);
        assert_eq!(result["author"], "Alice");
    }

    #[test]
    fn skips_lines_with_empty_key() {
        let result = parse_metadata_tags("=no_key\nauthor=Alice");
        assert_eq!(result.len(), 1);
        assert_eq!(result["author"], "Alice");
    }

    #[test]
    fn last_value_wins_for_duplicate_keys() {
        let result = parse_metadata_tags("author=Alice\nauthor=Bob");
        assert_eq!(result["author"], "Bob");
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn value_containing_equals_is_preserved() {
        // Only the first '=' is the delimiter; the rest is part of the value.
        let result = parse_metadata_tags("url=https://example.com?a=1&b=2");
        assert_eq!(result["url"], "https://example.com?a=1&b=2");
    }

    #[test]
    fn empty_input_returns_empty_map() {
        assert!(parse_metadata_tags("").is_empty());
        assert!(parse_metadata_tags("   \n  \n  ").is_empty());
    }

    #[test]
    fn result_is_sorted_by_key() {
        let result = parse_metadata_tags("zebra=last\nalpha=first\nmiddle=mid");
        let keys: Vec<&str> = result.keys().map(String::as_str).collect();
        assert_eq!(keys, ["alpha", "middle", "zebra"]);
    }
}
