//! Common helpers shared across YuNet crates.

/// Application configuration and settings management.
pub mod config;
/// Image enhancement utilities (unsharp mask, contrast, exposure, etc.)
pub mod enhance;
/// Test fixture loading and path resolution.
pub mod fixtures;
/// Shared GPU context initialization and pooling helpers.
pub mod gpu;
/// Image loading, resizing, and tensor conversion.
pub mod image_utils;
/// Data-driven mapping utilities (CSV/Excel/Parquet/SQLite ingestion).
pub mod mapping;
/// Image output helpers (encoding, metadata preservation).
pub mod output;
/// Image quality analysis (Laplacian variance blur detection).
pub mod quality;
/// Shape metadata and masking helpers for custom crop geometry.
pub mod shape;
/// Instrumentation helpers for optional performance tracing.
pub mod telemetry;

use std::path::Path;

use anyhow::Result;
use log::LevelFilter;

pub use enhance::{EnhancementSettings, WgpuEnhancer, apply_enhancements};
pub use fixtures::{
    fixture_path, fixtures_dir, load_fixture_bytes, load_fixture_image, load_fixture_json,
};
pub use gpu::{
    GpuAvailability, GpuContext, GpuContextGuard, GpuContextOptions, GpuContextPool, GpuInitError,
    GpuPoolError,
};
pub use image_utils::{
    compute_resize_scales, dynamic_to_bgr_chw, load_image, resize_image, rgb_to_bgr_chw,
};
pub use mapping::{
    ColumnSelector, MappingCatalog, MappingEntry, MappingFormat, MappingPreview,
    MappingReadOptions, detect_format as detect_mapping_format, inspect_mapping_sources,
    list_sqlite_tables, load_mapping_entries, load_mapping_preview,
};
pub use output::{
    ImageFormatHint, MetadataContext, OutputOptions, append_suffix_to_filename, save_dynamic_image,
};
pub use quality::QualityFilter;
pub use quality::{Quality, estimate_sharpness, laplacian_variance};
pub use shape::{
    CropShape, PolygonCornerStyle, apply_shape_mask, apply_shape_mask_dynamic,
    outline_points_for_rect,
};
pub use telemetry::{
    TimingGuard, configure as configure_telemetry, telemetry_allows, telemetry_enabled,
    telemetry_level, timing_guard, timing_guard_if,
};

/// Initialize logging once for CLI and GUI environments.
///
/// This function respects the `RUST_LOG` environment variable if it is set.
/// Otherwise, it falls back to the provided default filter level.
///
/// # Arguments
///
/// * `default_filter` - The `LevelFilter` to use if `RUST_LOG` is not set.
pub fn init_logging(default_filter: LevelFilter) -> Result<()> {
    let mut builder = env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(default_filter.as_str()),
    );
    builder.filter_module("yunet::telemetry", LevelFilter::Trace);

    if builder.try_init().is_err() {
        // Logger already initialized; nothing to do.
    }
    Ok(())
}

/// Validate that a path exists and resolve it to an absolute path.
///
/// # Arguments
///
/// * `path` - The path to validate and normalize.
pub fn normalize_path<P: AsRef<Path>>(path: P) -> Result<std::path::PathBuf> {
    let path = path.as_ref();
    anyhow::ensure!(path.exists(), "path does not exist: {}", path.display());
    Ok(path.canonicalize()?)
}
