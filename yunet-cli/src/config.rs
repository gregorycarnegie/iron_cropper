//! Configuration loading and CLI override logic.

use std::{collections::BTreeMap, path::PathBuf};

use anyhow::{Context, Result};
use log::{info, warn};
use yunet_core::preset_by_name;
use yunet_core::{CropSettings, PositioningMode};
use yunet_utils::{
    Quality,
    config::{
        AppSettings, CropSettings as ConfigCropSettings, MetadataMode, default_settings_path,
    },
    normalize_path,
};

use crate::{args::DetectArgs, color::parse_fill_color_spec};

/// Load application settings from a file or use defaults.
pub fn load_settings(config_path: Option<&PathBuf>) -> Result<AppSettings> {
    if let Some(path) = config_path {
        let resolved = normalize_path(path)?;
        let settings = AppSettings::load_from_path(&resolved)?;
        info!("Loaded settings from {}", resolved.display());
        Ok(settings)
    } else {
        let default_path = default_settings_path();
        if default_path.exists() {
            let settings = AppSettings::load_from_path(&default_path).with_context(|| {
                format!(
                    "failed to load default settings from {}",
                    default_path.display()
                )
            })?;
            info!("Loaded settings from {}", default_path.display());
            Ok(settings)
        } else {
            Ok(AppSettings::default())
        }
    }
}

/// Apply command-line arguments to override loaded or default settings.
pub fn apply_cli_overrides(settings: &mut AppSettings, args: &DetectArgs) {
    if args.gpu {
        settings.gpu.enabled = true;
    }
    if args.no_gpu {
        settings.gpu.enabled = false;
        settings.gpu.inference = false;
    }
    if args.gpu_inference {
        settings.gpu.inference = true;
    }
    if let Some(mode) = args.gpu_env {
        settings.gpu.respect_env = mode.respects_env();
    }

    if args.telemetry {
        settings.telemetry.enabled = true;
    }
    if let Some(level) = args.telemetry_level.as_ref() {
        let normalized = level.trim();
        if !normalized.is_empty() {
            let lower = normalized.to_ascii_lowercase();
            settings.telemetry.level = lower.clone();
            if lower == "off" {
                settings.telemetry.enabled = false;
            }
        }
    }

    if let Some(width) = args.width {
        settings.input.width = width;
    }
    if let Some(height) = args.height {
        settings.input.height = height;
    }
    if let Some(mode) = args.resize_quality {
        settings.input.resize_quality = mode;
    }
    if let Some(score) = args.score_threshold {
        settings.detection.score_threshold = score;
    }
    if let Some(nms) = args.nms_threshold {
        settings.detection.nms_threshold = nms;
    }
    if let Some(top_k) = args.top_k {
        settings.detection.top_k = top_k;
    }

    if let Some(preset_name) = args.preset.as_ref() {
        settings.crop.preset = preset_name.to_ascii_lowercase();
        if settings.crop.preset != "custom"
            && let Some(preset) = preset_by_name(preset_name)
            && preset.width > 0
            && preset.height > 0
        {
            settings.crop.output_width = preset.width;
            settings.crop.output_height = preset.height;
        }
    }
    if let Some(width) = args.output_width {
        settings.crop.output_width = width;
        settings.crop.preset = "custom".to_string();
    }
    if let Some(height) = args.output_height {
        settings.crop.output_height = height;
        settings.crop.preset = "custom".to_string();
    }

    settings.crop.face_height_pct = args.face_height_pct;
    settings.crop.horizontal_offset = args.horizontal_offset;
    settings.crop.vertical_offset = args.vertical_offset;
    settings.crop.positioning_mode = args.positioning_mode.replace('_', "-");
    if let Some(ref fill) = args.crop_fill_color {
        match parse_fill_color_spec(fill) {
            Ok(color) => settings.crop.fill_color = color,
            Err(err) => warn!("failed to parse --crop-fill-color '{}': {}", fill, err),
        }
    }
    settings.crop.output_format = args.output_format.to_ascii_lowercase();
    settings.crop.jpeg_quality = args.jpeg_quality;
    if let Some(ref compression) = args.png_compression {
        settings.crop.png_compression = compression.clone();
    }
    if let Some(webp) = args.webp_quality {
        settings.crop.webp_quality = webp;
    }
    if let Some(auto) = args.auto_detect_format {
        settings.crop.auto_detect_format = auto;
    }

    if let Some(auto) = args.auto_select_best {
        settings.crop.quality_rules.auto_select_best_face = auto;
    }
    if let Some(skip) = args.skip_no_high_quality {
        settings.crop.quality_rules.auto_skip_no_high_quality = skip;
    }
    if let Some(flag) = args.quality_suffix {
        settings.crop.quality_rules.quality_suffix = flag;
    }
    if let Some(skip) = args.skip_low_quality {
        if skip {
            settings.crop.quality_rules.min_quality = Some(Quality::Medium);
        } else {
            settings.crop.quality_rules.min_quality = None;
        }
    } else if let Some(ref s) = args.min_quality {
        match s.parse::<Quality>() {
            Ok(q) => {
                settings.crop.quality_rules.min_quality = Some(q);
            }
            Err(_) => warn!("unknown --min-quality value '{}', ignoring", s),
        }
    }

    if let Some(ref mode) = args.metadata_mode {
        match mode.parse::<MetadataMode>() {
            Ok(mode) => settings.crop.metadata.mode = mode,
            Err(err) => warn!("{err}"),
        }
    }
    if let Some(include) = args.metadata_include_crop {
        settings.crop.metadata.include_crop_settings = include;
    }
    if let Some(include) = args.metadata_include_quality {
        settings.crop.metadata.include_quality_metrics = include;
    }
    if !args.metadata_tags.is_empty() {
        settings.crop.metadata.custom_tags = parse_metadata_tags_args(&args.metadata_tags);
    }

    settings.crop.sanitize();
}

pub fn build_core_crop_settings(cfg: &ConfigCropSettings) -> CropSettings {
    CropSettings {
        output_width: cfg.output_width,
        output_height: cfg.output_height,
        face_height_pct: cfg.face_height_pct,
        positioning_mode: parse_positioning_mode(&cfg.positioning_mode),
        horizontal_offset: cfg.horizontal_offset,
        vertical_offset: cfg.vertical_offset,
        fill_color: cfg.fill_color,
    }
}

fn parse_positioning_mode(value: &str) -> PositioningMode {
    match value.to_ascii_lowercase().as_str() {
        "rule_of_thirds" | "rule-of-thirds" | "ruleofthirds" => PositioningMode::RuleOfThirds,
        "custom" => PositioningMode::Custom,
        _ => PositioningMode::Center,
    }
}

fn parse_metadata_tags_args(entries: &[String]) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for entry in entries {
        if let Some((key, value)) = entry.split_once('=') {
            let key = key.trim();
            if key.is_empty() {
                warn!("Ignoring metadata tag with empty key: '{entry}'");
                continue;
            }
            map.insert(key.to_string(), value.trim().to_string());
        } else {
            warn!("Invalid metadata tag '{entry}', expected key=value");
        }
    }
    map
}
