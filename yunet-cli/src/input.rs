//! Input collection and mapping file handling.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use log::{debug, info, warn};
use walkdir::WalkDir;
use yunet_utils::{
    mapping::{
        ColumnSelector, MappingFormat, MappingReadOptions, detect_format as detect_mapping_format,
        load_mapping_entries,
    },
    normalize_path,
};

use crate::args::DetectArgs;

#[derive(Clone)]
pub struct ProcessingItem {
    pub source: PathBuf,
    pub output_override: Option<PathBuf>,
    pub mapping_row: Option<usize>,
}

/// Collect all image paths from a file or directory.
pub fn collect_images(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    if !path.is_dir() {
        anyhow::bail!(
            "input path is neither file nor directory: {}",
            path.display()
        );
    }

    let exts = ["jpg", "jpeg", "png", "bmp", "webp"];
    let mut images = Vec::new();
    for entry in WalkDir::new(path)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
            let ext_lower = ext.to_ascii_lowercase();
            if exts.contains(&ext_lower.as_str()) {
                images.push(entry.path().to_path_buf());
            } else {
                debug!("Skipping non-image file {}", entry.path().display());
            }
        }
    }
    images.sort();
    Ok(images)
}

pub fn collect_standard_targets(input_path: &Path) -> Result<Vec<ProcessingItem>> {
    let images = collect_images(input_path)?;
    if images.is_empty() {
        anyhow::bail!(
            "no images found at {} (supported extensions: jpg, jpeg, png, bmp)",
            input_path.display()
        );
    }
    Ok(images
        .into_iter()
        .map(|path| ProcessingItem {
            source: path,
            output_override: None,
            mapping_row: None,
        })
        .collect())
}

pub fn collect_mapping_targets(
    mapping_file: &Path,
    args: &DetectArgs,
) -> Result<Vec<ProcessingItem>> {
    let mapping_path = normalize_path(mapping_file)?;
    let source_selector = ColumnSelector::parse_token(
        args.mapping_source_col
            .as_deref()
            .ok_or_else(|| anyhow!("--mapping-source-col is required with --mapping-file"))?,
    )?;
    let output_selector = ColumnSelector::parse_token(
        args.mapping_output_col
            .as_deref()
            .ok_or_else(|| anyhow!("--mapping-output-col is required with --mapping-file"))?,
    )?;

    let user_format = match args.mapping_format.as_deref() {
        Some(token) => Some(parse_mapping_format_token(token)?),
        None => None,
    };
    let mut read_options = MappingReadOptions {
        format: user_format,
        has_headers: args.mapping_has_headers,
        delimiter: args.mapping_delimiter.map(|c| c as u8),
        sheet_name: args.mapping_sheet.clone(),
        sql_table: args.mapping_sql_table.clone(),
        sql_query: args.mapping_sql_query.clone(),
        ..Default::default()
    };
    let resolved_format = read_options
        .format
        .unwrap_or_else(|| detect_mapping_format(&mapping_path));
    read_options.format = Some(resolved_format);

    info!(
        "Loading mapping ({}) from {}",
        resolved_format.display_name(),
        mapping_path.display()
    );

    let entries = load_mapping_entries(
        &mapping_path,
        &read_options,
        &source_selector,
        &output_selector,
    )
    .with_context(|| format!("failed to load mapping from {}", mapping_path.display()))?;
    if entries.is_empty() {
        anyhow::bail!(
            "no usable rows found in mapping file {}",
            mapping_path.display()
        );
    }

    let mapping_dir = mapping_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    let mut items = Vec::new();
    for (idx, entry) in entries.into_iter().enumerate() {
        let row_no = idx + 1;
        let raw_source = PathBuf::from(entry.source_path);

        // Security: reject absolute paths from mapping files; sources must be
        // relative to the mapping file's directory.
        if raw_source.is_absolute() {
            warn!(
                "Skipping mapping row {}: absolute source paths are not allowed ({})",
                row_no,
                raw_source.display()
            );
            continue;
        }

        let resolved_source = mapping_dir.join(&raw_source);

        // Verify the resolved path does not escape the mapping directory via `..`.
        if let (Ok(canon_source), Ok(canon_dir)) =
            (resolved_source.canonicalize(), mapping_dir.canonicalize())
            && !canon_source.starts_with(&canon_dir)
        {
            warn!(
                "Skipping mapping row {}: source path escapes the mapping directory ({})",
                row_no,
                raw_source.display()
            );
            continue;
        }

        if !resolved_source.exists() {
            warn!(
                "Skipping mapping row {}: source {} was not found",
                row_no,
                resolved_source.display()
            );
            continue;
        }
        items.push(ProcessingItem {
            source: resolved_source,
            output_override: Some(PathBuf::from(entry.output_name)),
            mapping_row: Some(row_no),
        });
    }

    if items.is_empty() {
        anyhow::bail!(
            "mapping file {} did not produce any usable rows",
            mapping_path.display()
        );
    }

    info!(
        "Loaded {} mapping row(s) from {}",
        items.len(),
        mapping_path.display()
    );

    Ok(items)
}

fn parse_mapping_format_token(token: &str) -> Result<MappingFormat> {
    match token.to_ascii_lowercase().as_str() {
        "csv" | "delimited" | "text" => Ok(MappingFormat::Csv),
        "excel" | "xlsx" | "xls" => Ok(MappingFormat::Excel),
        "parquet" | "pq" => Ok(MappingFormat::Parquet),
        "sqlite" | "sql" | "db" => Ok(MappingFormat::Sqlite),
        other => anyhow::bail!(
            "unknown mapping format '{other}' (supported: csv, excel, parquet, sqlite)"
        ),
    }
}

pub fn resolve_override_output_path(
    output_dir: &Path,
    override_target: &Path,
    ext: &str,
    face_index: usize,
    multi_face: bool,
) -> PathBuf {
    let cleaned_ext = ext.trim_start_matches('.').to_string();

    // Security: reject absolute paths and path traversal components from mapping data.
    let safe_target = sanitize_override_path(override_target);

    let rel_parent = safe_target.parent().unwrap_or_else(|| Path::new(""));
    let parent = output_dir.join(rel_parent);

    let base_name = safe_target
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "output".to_string());
    let final_base = if multi_face {
        format!("{base_name}_face{}", face_index + 1)
    } else {
        base_name
    };
    let mut final_path = parent;
    final_path.push(final_base);
    final_path.set_extension(cleaned_ext);
    final_path
}

/// Strip path traversal components and root prefixes so the result is always
/// a relative path that stays inside the output directory.
fn sanitize_override_path(p: &Path) -> PathBuf {
    use std::path::Component;
    p.components()
        .filter(|c| matches!(c, Component::Normal(_)))
        .collect()
}
