//! Settings persistence and loading.

use anyhow::{Context as AnyhowContext, Result};
use log::warn;
use std::path::Path;
use yunet_utils::config::AppSettings;

/// Loads application settings from a file, or returns default settings if loading fails.
pub fn load_settings(path: &Path) -> AppSettings {
    match AppSettings::load_from_path(path) {
        Ok(settings) => settings,
        Err(err) => {
            warn!(
                "Failed to load settings from {}: {err:?}. Falling back to defaults.",
                path.display()
            );
            AppSettings::default()
        }
    }
}

/// Saves the current settings to the JSON file.
pub fn persist_settings(settings: &AppSettings, settings_path: &Path) -> Result<()> {
    if let Some(parent) = settings_path.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create settings directory {}", parent.display()))?;
    }
    settings
        .save_to_path(settings_path)
        .with_context(|| format!("failed to write settings to {}", settings_path.display()))
}

/// Persists the current settings to disk and provides feedback.
pub fn persist_settings_with_feedback(
    settings: &AppSettings,
    settings_path: &Path,
) -> Result<(), String> {
    persist_settings(settings, settings_path).map_err(|err| {
        let message = format!("Failed to persist settings: {err}");
        warn!("{message}");
        message
    })
}
