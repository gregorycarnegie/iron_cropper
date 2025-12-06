//! Enhancement settings construction from CLI arguments.

use log::warn;
use yunet_utils::EnhancementSettings;

use crate::args::DetectArgs;

/// Build EnhancementSettings from CLI args. Returns None when enhancements aren't enabled.
pub fn build_enhancement_settings(args: &DetectArgs) -> Option<EnhancementSettings> {
    // Only construct enhancement settings when `--enhance` is explicitly set to true
    if !args.enhance.unwrap_or(false) {
        return None;
    }

    // Base: defaults or preset
    let mut base = EnhancementSettings::default();
    if let Some(pname) = args.enhancement_preset.as_ref() {
        match pname.as_str() {
            "natural" => {
                base = EnhancementSettings {
                    auto_color: true,
                    exposure_stops: 0.1,
                    brightness: 0,
                    contrast: 1.1,
                    saturation: 1.05,
                    unsharp_amount: 0.6,
                    unsharp_radius: 1.0,
                    sharpness: 0.2,
                    skin_smooth_amount: 0.0,
                    skin_smooth_sigma_space: 3.0,
                    skin_smooth_sigma_color: 25.0,
                    red_eye_removal: false,
                    red_eye_threshold: 1.5,
                    background_blur: false,
                    background_blur_radius: 15.0,
                    background_blur_mask_size: 0.6,
                }
            }
            "vivid" => {
                base = EnhancementSettings {
                    auto_color: false,
                    exposure_stops: 0.3,
                    brightness: 10,
                    contrast: 1.25,
                    saturation: 1.3,
                    unsharp_amount: 0.9,
                    unsharp_radius: 1.2,
                    sharpness: 0.5,
                    skin_smooth_amount: 0.0,
                    skin_smooth_sigma_space: 3.0,
                    skin_smooth_sigma_color: 25.0,
                    red_eye_removal: false,
                    red_eye_threshold: 1.5,
                    background_blur: false,
                    background_blur_radius: 15.0,
                    background_blur_mask_size: 0.6,
                }
            }
            "professional" => {
                base = EnhancementSettings {
                    auto_color: true,
                    exposure_stops: 0.2,
                    brightness: 0,
                    contrast: 1.15,
                    saturation: 1.05,
                    unsharp_amount: 1.2,
                    unsharp_radius: 1.0,
                    sharpness: 0.8,
                    skin_smooth_amount: 0.0,
                    skin_smooth_sigma_space: 3.0,
                    skin_smooth_sigma_color: 25.0,
                    red_eye_removal: false,
                    red_eye_threshold: 1.5,
                    background_blur: false,
                    background_blur_radius: 15.0,
                    background_blur_mask_size: 0.6,
                }
            }
            other => warn!("unknown enhancement preset '{}', using defaults", other),
        }
    }

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
