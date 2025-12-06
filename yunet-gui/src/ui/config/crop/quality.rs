use crate::YuNetApp;
use egui::{ComboBox, Ui};
use yunet_utils::quality::Quality;

/// Shows the quality automation settings.
pub fn show_quality_automation(app: &mut YuNetApp, ui: &mut Ui, settings_changed: &mut bool) {
    ui.separator();
    ui.label("Quality automation");

    let mut auto_select = app.settings.crop.quality_rules.auto_select_best_face;
    if ui
        .checkbox(&mut auto_select, "Auto-select highest quality face")
        .changed()
    {
        app.settings.crop.quality_rules.auto_select_best_face = auto_select;
        *settings_changed = true;
    }

    let mut skip_no_high = app.settings.crop.quality_rules.auto_skip_no_high_quality;
    if ui
        .checkbox(&mut skip_no_high, "Skip export when no high-quality faces")
        .changed()
    {
        app.settings.crop.quality_rules.auto_skip_no_high_quality = skip_no_high;
        *settings_changed = true;
    }

    let mut suffix_enabled = app.settings.crop.quality_rules.quality_suffix;
    if ui
        .checkbox(&mut suffix_enabled, "Append quality suffix to filenames")
        .changed()
    {
        app.settings.crop.quality_rules.quality_suffix = suffix_enabled;
        *settings_changed = true;
    }

    ui.horizontal(|ui| {
        ui.label("Minimum quality to export");
        let current = app.settings.crop.quality_rules.min_quality;
        let label = match current {
            Some(Quality::Low) => "Low",
            Some(Quality::Medium) => "Medium",
            Some(Quality::High) => "High",
            None => "Off",
        };
        ComboBox::from_id_salt("min_quality_combo")
            .selected_text(label)
            .show_ui(ui, |ui| {
                if ui.selectable_label(current.is_none(), "Off").clicked() {
                    app.settings.crop.quality_rules.min_quality = None;
                    *settings_changed = true;
                }
                if ui
                    .selectable_label(current == Some(Quality::Low), "Low")
                    .clicked()
                {
                    app.settings.crop.quality_rules.min_quality = Some(Quality::Low);
                    *settings_changed = true;
                }
                if ui
                    .selectable_label(current == Some(Quality::Medium), "Medium")
                    .clicked()
                {
                    app.settings.crop.quality_rules.min_quality = Some(Quality::Medium);
                    *settings_changed = true;
                }
                if ui
                    .selectable_label(current == Some(Quality::High), "High")
                    .clicked()
                {
                    app.settings.crop.quality_rules.min_quality = Some(Quality::High);
                    *settings_changed = true;
                }
            });
    });
}
