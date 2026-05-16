//! Enhancement settings construction from CLI arguments.

use fcs_utils::EnhancementSettings;
use log::warn;

use crate::args::DetectArgs;

/// Build EnhancementSettings from CLI args. Returns None when enhancements aren't enabled.
pub fn build_enhancement_settings(args: &DetectArgs) -> Option<EnhancementSettings> {
    // Only construct enhancement settings when `--enhance` is explicitly set to true
    if !args.enhance.unwrap_or(false) {
        return None;
    }

    // Base: defaults or named preset.
    let mut base = match args.enhancement_preset.as_deref() {
        None => EnhancementSettings::default(),
        Some(name) => EnhancementSettings::preset_by_name(name).unwrap_or_else(|| {
            warn!("unknown enhancement preset '{}', using defaults", name);
            EnhancementSettings::default()
        }),
    };

    // Apply explicit overrides if provided
    if let Some(v) = args.unsharp_amount {
        base.unsharp_amount = v;
    }
    if let Some(v) = args.unsharp_radius {
        base.unsharp_radius = v;
    }
    if let Some(v) = args.enhance_contrast {
        base.contrast = v;
    }
    if let Some(v) = args.enhance_exposure {
        base.exposure_stops = v;
    }
    if let Some(v) = args.enhance_brightness {
        base.brightness = v;
    }
    if let Some(v) = args.enhance_saturation {
        base.saturation = v;
    }
    if let Some(v) = args.enhance_auto_color {
        base.auto_color = v;
    }
    if let Some(v) = args.enhance_sharpness {
        base.sharpness = v;
    }
    if let Some(v) = args.enhance_skin_smooth {
        base.skin_smooth_amount = v;
    }
    if let Some(v) = args.enhance_red_eye_removal {
        base.red_eye_removal = v;
    }
    if let Some(v) = args.enhance_background_blur {
        base.background_blur = v;
    }

    Some(base)
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    fn parse_args(args: &[&str]) -> DetectArgs {
        let mut full = vec!["fcs-cli"];
        full.extend_from_slice(args);
        DetectArgs::try_parse_from(full).expect("parse test CLI args")
    }

    #[test]
    fn build_enhancement_settings_returns_none_when_disabled() {
        let args = parse_args(&["--input", "input.jpg"]);
        assert!(build_enhancement_settings(&args).is_none());
    }

    #[test]
    fn build_enhancement_settings_uses_defaults_for_unknown_preset() {
        let args = parse_args(&[
            "--input",
            "input.jpg",
            "--enhance",
            "true",
            "--enhancement-preset",
            "unknown",
        ]);

        let settings = build_enhancement_settings(&args).expect("build enhancement settings");
        assert_eq!(settings.contrast, EnhancementSettings::default().contrast);
        assert_eq!(
            settings.unsharp_amount,
            EnhancementSettings::default().unsharp_amount
        );
    }

    #[test]
    fn build_enhancement_settings_applies_remaining_overrides() {
        let args = parse_args(&[
            "--input",
            "input.jpg",
            "--enhance",
            "true",
            "--unsharp-amount",
            "1.1",
            "--unsharp-radius",
            "2.5",
            "--enhance-skin-smooth",
            "0.75",
            "--enhance-red-eye-removal=true",
            "--enhance-background-blur=true",
        ]);

        let settings =
            build_enhancement_settings(&args).expect("build enhancement settings with overrides");
        assert!((settings.unsharp_amount - 1.1).abs() < f32::EPSILON);
        assert!((settings.unsharp_radius - 2.5).abs() < f32::EPSILON);
        assert!((settings.skin_smooth_amount - 0.75).abs() < f32::EPSILON);
        assert!(settings.red_eye_removal);
        assert!(settings.background_blur);
    }
}
