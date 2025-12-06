//! Crop settings UI controls.

use crate::YuNetApp;
use egui::Ui;

mod dimensions;
mod fill;
mod metadata;
mod output;
mod positioning;
mod presets;
mod quality;
mod shape;

pub use self::dimensions::show_dimensions_controls;
pub use self::fill::show_fill_color_controls;
pub use self::metadata::show_metadata_section;
pub use self::output::show_output_format;
pub use self::positioning::show_positioning_controls;
pub use self::presets::show_preset_selector;
pub use self::quality::show_quality_automation;
pub use self::shape::edit_shape_controls;

/// Shows the crop settings section.
pub fn show_crop_section(
    app: &mut YuNetApp,
    ui: &mut Ui,
    settings_changed: &mut bool,
    preview_invalidated: &mut bool,
    metadata_tags_changed: &mut bool,
) {
    ui.separator();
    ui.heading("Crop Settings");

    ui.checkbox(&mut app.show_crop_overlay, "Show crop preview overlay")
        .on_hover_text("Draws the proposed crop rectangle for every detected face.");
    ui.add_space(6.0);

    // Preset selection
    show_preset_selector(app, ui, settings_changed, preview_invalidated);

    // Output dimensions
    show_dimensions_controls(app, ui, settings_changed, preview_invalidated);

    // Shape controls
    if edit_shape_controls(app, ui) {
        app.settings.crop.sanitize();
        app.clear_crop_preview_cache();
        *preview_invalidated = true;
        *settings_changed = true;
    }

    if show_fill_color_controls(app, ui) {
        app.clear_crop_preview_cache();
        *preview_invalidated = true;
        *settings_changed = true;
    }

    // Positioning mode
    show_positioning_controls(app, ui, settings_changed);

    // Quality automation
    show_quality_automation(app, ui, settings_changed);

    // Output format
    show_output_format(app, ui, settings_changed);

    // Metadata
    show_metadata_section(app, ui, settings_changed, metadata_tags_changed);
}
