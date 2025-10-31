//! Common helpers shared across YuNet crates.

pub mod config;
pub mod fixtures;
pub mod image_utils;

use std::path::Path;

use anyhow::Result;
use log::LevelFilter;

pub use fixtures::{
    fixture_path, fixtures_dir, load_fixture_bytes, load_fixture_image, load_fixture_json,
};
pub use image_utils::{
    compute_resize_scales, dynamic_to_bgr_chw, load_image, resize_image, rgb_to_bgr_chw,
};

/// Initialize logging once for CLI and GUI environments.
pub fn init_logging(default_filter: LevelFilter) -> Result<()> {
    if env_logger::try_init_from_env(
        env_logger::Env::default().default_filter_or(default_filter.as_str()),
    )
    .is_err()
    {
        // Logger already initialized; nothing to do.
    }
    Ok(())
}

/// Validate that a path exists and resolve it to an absolute path.
pub fn normalize_path<P: AsRef<Path>>(path: P) -> Result<std::path::PathBuf> {
    let path = path.as_ref();
    anyhow::ensure!(path.exists(), "path does not exist: {}", path.display());
    Ok(path.canonicalize()?)
}
