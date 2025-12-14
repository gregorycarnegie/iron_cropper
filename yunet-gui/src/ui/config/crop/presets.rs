use crate::YuNetApp;
use egui::{ComboBox, Ui};
use yunet_core::preset_by_name;

/// Shows the preset selector combo box.
pub fn show_preset_selector(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    preview_invalidated: &mut bool,
) {
    let preset_combo = ComboBox::from_label("Preset")
        .selected_text(&app.settings.crop.preset)
        .show_ui(ui, |ui| {
            let presets = [
                ("linkedin", "LinkedIn (400×400)"),
                ("passport", "Passport (413×531)"),
                ("instagram", "Instagram (1080×1080)"),
                ("idcard", "ID Card (332×498)"),
                ("avatar", "Avatar (512×512)"),
                ("headshot", "Headshot (600×800)"),
                ("custom", "Custom size"),
            ];
            let icon_size = app.icons.default_size();
            for (value, label) in presets {
                let selected = app.settings.crop.preset == value;
                let icon = match value {
                    "linkedin" => Some(app.icons.linkedin(icon_size)),
                    "passport" => Some(app.icons.passport(icon_size)),
                    "instagram" => Some(app.icons.instagram(icon_size)),
                    "idcard" => Some(app.icons.id_card(icon_size)),
                    "avatar" => Some(app.icons.account(icon_size)),
                    "headshot" => Some(app.icons.portrait(icon_size)),
                    _ => None,
                };

                let clicked = if let Some(icon) = icon {
                    ui.add(egui::Button::image_and_text(icon, label).selected(selected))
                        .clicked()
                } else {
                    ui.selectable_label(selected, label).clicked()
                };

                if clicked && !selected {
                    app.settings.crop.preset = value.to_string();
                    if value != "custom"
                        && let Some(preset) = preset_by_name(value)
                    {
                        app.settings.crop.output_width = preset.width;
                        app.settings.crop.output_height = preset.height;
                    }
                    app.clear_crop_preview_cache();
                    *preview_invalidated = true;
                    *settings_changed = true;
                }
            }
        });
    preset_combo
        .response
        .on_hover_text("Choose a predefined output size/aspect ratio.");
}
