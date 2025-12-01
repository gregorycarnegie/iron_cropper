use anyhow::{Context, Result, anyhow};
use base64::{Engine as _, engine::general_purpose};
use calamine::{Data as ExcelData, Reader as _, open_workbook_auto};
use csv::ReaderBuilder;
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::{Field, Row},
};
use rusqlite::{Connection, types::ValueRef};
use std::{fs::File, path::Path};

/// Default number of rows to show in mapping previews.
pub const DEFAULT_PREVIEW_ROWS: usize = 32;

struct MappingTable {
    columns: Vec<String>,
    rows: Vec<Vec<String>>,
    total_rows: usize,
}

/// Supported mapping formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MappingFormat {
    Csv,
    Excel,
    Parquet,
    Sqlite,
}

impl MappingFormat {
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Csv => "CSV / Delimited",
            Self::Excel => "Excel",
            Self::Parquet => "Parquet",
            Self::Sqlite => "SQLite",
        }
    }
}

/// Column selector used to resolve user selections to a zero-based index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ColumnSelector {
    Index(usize),
    Name(String),
}

impl ColumnSelector {
    pub fn by_index(index: usize) -> Self {
        Self::Index(index)
    }

    pub fn by_name(name: impl Into<String>) -> Self {
        Self::Name(name.into())
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Index(idx) => format!("column #{idx}"),
            Self::Name(name) => format!("column \"{name}\""),
        }
    }

    /// Parses a CLI-style token (`#3` or `3` for indices, any other value for names).
    pub fn parse_token(token: &str) -> Result<Self> {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            anyhow::bail!("column selector cannot be empty");
        }
        let digits = trimmed.strip_prefix('#').unwrap_or(trimmed);
        if digits.chars().all(|c| c.is_ascii_digit()) {
            let idx: usize = digits.parse()?;
            return Ok(Self::Index(idx));
        }
        Ok(Self::Name(trimmed.to_string()))
    }
}

/// Options that influence how a mapping file is read.
#[derive(Clone, Debug)]
pub struct MappingReadOptions {
    pub format: Option<MappingFormat>,
    pub has_headers: Option<bool>,
    pub delimiter: Option<u8>,
    pub sheet_name: Option<String>,
    pub sql_table: Option<String>,
    pub sql_query: Option<String>,
    pub preview_rows: usize,
}

impl Default for MappingReadOptions {
    fn default() -> Self {
        Self {
            format: None,
            has_headers: None,
            delimiter: None,
            sheet_name: None,
            sql_table: None,
            sql_query: None,
            preview_rows: DEFAULT_PREVIEW_ROWS,
        }
    }
}

/// Preview payload shared with the GUI.
#[derive(Clone, Debug)]
pub struct MappingPreview {
    pub format: MappingFormat,
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub total_rows: usize,
    pub truncated: bool,
}

/// Fully materialised mapping entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MappingEntry {
    pub source_path: String,
    pub output_name: String,
}

/// Additional metadata exposed to the UI (e.g. sheet or table names).
#[derive(Clone, Debug, Default)]
pub struct MappingCatalog {
    pub sheets: Vec<String>,
    pub sql_tables: Vec<String>,
}

/// Detects a mapping format from the file extension.
pub fn detect_format(path: &Path) -> MappingFormat {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default()
        .as_str()
    {
        "xlsx" | "xls" | "xlsm" | "ods" => MappingFormat::Excel,
        "parquet" | "pq" => MappingFormat::Parquet,
        "db" | "sqlite" | "sqlite3" => MappingFormat::Sqlite,
        _ => MappingFormat::Csv,
    }
}

/// Builds a preview for the supplied file.
pub fn load_mapping_preview(path: &Path, options: &MappingReadOptions) -> Result<MappingPreview> {
    let format = options.format.unwrap_or_else(|| detect_format(path));
    let limit = options.preview_rows.max(1);
    match format {
        MappingFormat::Csv => preview_csv(path, options, limit),
        MappingFormat::Excel => preview_excel(path, options, limit),
        MappingFormat::Parquet => preview_parquet(path, limit),
        MappingFormat::Sqlite => preview_sqlite(path, options, limit),
    }
}

/// Loads every mapping entry, resolving the selected columns to source/output pairs.
pub fn load_mapping_entries(
    path: &Path,
    options: &MappingReadOptions,
    source: &ColumnSelector,
    output: &ColumnSelector,
) -> Result<Vec<MappingEntry>> {
    let format = options.format.unwrap_or_else(|| detect_format(path));
    let table = match format {
        MappingFormat::Csv => table_csv(path, options)?,
        MappingFormat::Excel => table_excel(path, options)?,
        MappingFormat::Parquet => table_parquet(path)?,
        MappingFormat::Sqlite => table_sqlite(path, options)?,
    };

    let source_idx = resolve_selector(&table.columns, source)?;
    let output_idx = resolve_selector(&table.columns, output)?;

    let entries = table
        .rows
        .into_iter()
        .filter_map(|row| {
            let source_value = row.get(source_idx)?.trim();
            let output_value = row.get(output_idx)?.trim();
            if source_value.is_empty() || output_value.is_empty() {
                return None;
            }
            Some(MappingEntry {
                source_path: source_value.to_string(),
                output_name: output_value.to_string(),
            })
        })
        .collect();
    Ok(entries)
}

/// Lists auxiliary metadata such as sheet/table names for UI drop-downs.
pub fn inspect_mapping_sources(
    path: &Path,
    options: &MappingReadOptions,
) -> Result<MappingCatalog> {
    let format = options.format.unwrap_or_else(|| detect_format(path));
    match format {
        MappingFormat::Excel => {
            let workbook = open_workbook_auto(path)
                .with_context(|| format!("failed to open workbook {}", path.display()))?;
            Ok(MappingCatalog {
                sheets: workbook.sheet_names().to_vec(),
                sql_tables: Vec::new(),
            })
        }
        MappingFormat::Sqlite => Ok(MappingCatalog {
            sheets: Vec::new(),
            sql_tables: list_sqlite_tables(path)?,
        }),
        _ => Ok(MappingCatalog::default()),
    }
}

/// Enumerates SQLite tables in the supplied database.
pub fn list_sqlite_tables(path: &Path) -> Result<Vec<String>> {
    let conn = Connection::open(path)
        .with_context(|| format!("failed to open sqlite database {}", path.display()))?;
    list_sqlite_tables_conn(&conn)
}

// ---------------------------------------------------------------------------
// Preview builders

fn preview_csv(path: &Path, options: &MappingReadOptions, limit: usize) -> Result<MappingPreview> {
    let table = table_csv_internal(path, options, Some(limit))?;
    Ok(to_preview(MappingFormat::Csv, table))
}

fn preview_excel(
    path: &Path,
    options: &MappingReadOptions,
    limit: usize,
) -> Result<MappingPreview> {
    let table = table_excel_internal(path, options, Some(limit))?;
    Ok(to_preview(MappingFormat::Excel, table))
}

fn preview_parquet(path: &Path, limit: usize) -> Result<MappingPreview> {
    let table = table_parquet_internal(path, Some(limit))?;
    Ok(to_preview(MappingFormat::Parquet, table))
}

fn preview_sqlite(
    path: &Path,
    options: &MappingReadOptions,
    limit: usize,
) -> Result<MappingPreview> {
    let table = table_sqlite_internal(path, options, Some(limit))?;
    Ok(to_preview(MappingFormat::Sqlite, table))
}

fn to_preview(format: MappingFormat, table: MappingTable) -> MappingPreview {
    let truncated = table.total_rows > table.rows.len();
    MappingPreview {
        format,
        columns: table.columns,
        total_rows: table.total_rows,
        truncated,
        rows: table.rows,
    }
}

// ---------------------------------------------------------------------------
// Full table loading

fn table_csv(path: &Path, options: &MappingReadOptions) -> Result<MappingTable> {
    table_csv_internal(path, options, None)
}

fn table_excel(path: &Path, options: &MappingReadOptions) -> Result<MappingTable> {
    table_excel_internal(path, options, None)
}

fn table_parquet(path: &Path) -> Result<MappingTable> {
    table_parquet_internal(path, None)
}

fn table_sqlite(path: &Path, options: &MappingReadOptions) -> Result<MappingTable> {
    table_sqlite_internal(path, options, None)
}

// ---------------------------------------------------------------------------
// CSV

fn table_csv_internal(
    path: &Path,
    options: &MappingReadOptions,
    row_limit: Option<usize>,
) -> Result<MappingTable> {
    let has_headers = options.has_headers.unwrap_or(true);
    let delimiter = options.delimiter.unwrap_or(b',');
    let mut reader = ReaderBuilder::new()
        .has_headers(has_headers)
        .delimiter(delimiter)
        .from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;

    let mut columns = if has_headers {
        reader
            .headers()
            .context("failed to read CSV headers")?
            .iter()
            .enumerate()
            .map(|(idx, raw)| format_header(raw, idx))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    let mut rows = Vec::new();
    let mut total_rows = 0usize;
    for record in reader.records() {
        let record = record?;
        let mut row: Vec<String> = record.iter().map(sanitize_value).collect();
        if row.iter().all(|v| v.is_empty()) {
            continue;
        }
        ensure_columns(&mut columns, row.len());
        normalize_row(&mut row, columns.len());
        if row_limit.is_none_or(|limit| rows.len() < limit) {
            rows.push(row);
        }
        total_rows += 1;
    }

    Ok(MappingTable {
        columns,
        rows,
        total_rows,
    })
}

// ---------------------------------------------------------------------------
// Excel

fn table_excel_internal(
    path: &Path,
    options: &MappingReadOptions,
    row_limit: Option<usize>,
) -> Result<MappingTable> {
    let mut workbook = open_workbook_auto(path)
        .with_context(|| format!("failed to open workbook {}", path.display()))?;
    let sheet_name = match options
        .sheet_name
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        Some(explicit) => explicit.to_string(),
        None => workbook
            .sheet_names()
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("workbook {} has no sheets", path.display()))?,
    };

    let has_headers = options.has_headers.unwrap_or(true);
    let range = workbook
        .worksheet_range(&sheet_name)
        .map_err(|e| anyhow!("failed to read sheet {sheet_name}: {e}"))?;

    let mut rows_iter = range.rows();
    let mut columns = if has_headers {
        rows_iter
            .next()
            .map(|header_row| {
                header_row
                    .iter()
                    .enumerate()
                    .map(|(idx, cell)| format_excel_header(cell, idx))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    let mut rows = Vec::new();
    let mut total_rows = 0usize;
    for row in rows_iter {
        let mut values: Vec<String> = row.iter().map(format_excel_cell).collect();
        if values.iter().all(|v| v.is_empty()) {
            continue;
        }
        ensure_columns(&mut columns, values.len());
        normalize_row(&mut values, columns.len());
        if row_limit.is_none_or(|limit| rows.len() < limit) {
            rows.push(values);
        }
        total_rows += 1;
    }

    Ok(MappingTable {
        columns,
        rows,
        total_rows,
    })
}

// ---------------------------------------------------------------------------
// Parquet

fn table_parquet_internal(path: &Path, row_limit: Option<usize>) -> Result<MappingTable> {
    let file = File::open(path)
        .with_context(|| format!("failed to open parquet file {}", path.display()))?;
    let reader =
        SerializedFileReader::new(file).with_context(|| "failed to create parquet reader")?;

    let schema = reader.metadata().file_metadata().schema_descr();
    let mut columns: Vec<String> = schema
        .columns()
        .iter()
        .map(|c| c.name().to_string())
        .collect();
    if columns.is_empty() {
        anyhow::bail!("parquet file {} has no columns", path.display());
    }

    let mut rows = Vec::new();
    let mut total_rows = 0usize;
    let iter = reader
        .get_row_iter(None)
        .with_context(|| "failed to build parquet row iterator")?;
    for row in iter {
        let row: Row = row?;
        let mut values: Vec<String> = row
            .get_column_iter()
            .map(|(_, field)| format_parquet_field(field))
            .collect();
        if values.iter().all(|v| v.is_empty()) {
            continue;
        }
        ensure_columns(&mut columns, values.len());
        normalize_row(&mut values, columns.len());
        if row_limit.is_none_or(|limit| rows.len() < limit) {
            rows.push(values);
        }
        total_rows += 1;
    }

    Ok(MappingTable {
        columns,
        rows,
        total_rows,
    })
}

// ---------------------------------------------------------------------------
// SQLite

fn table_sqlite_internal(
    path: &Path,
    options: &MappingReadOptions,
    row_limit: Option<usize>,
) -> Result<MappingTable> {
    let conn = Connection::open(path)
        .with_context(|| format!("failed to open sqlite database {}", path.display()))?;
    let sql = resolve_sql_query(&conn, options)?;
    let mut stmt = conn.prepare(&sql)?;
    let columns = stmt
        .column_names()
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            let trimmed = name.trim();
            if trimmed.is_empty() {
                format!("Column {}", idx + 1)
            } else {
                trimmed.to_string()
            }
        })
        .collect::<Vec<_>>();
    if columns.is_empty() {
        anyhow::bail!("SQL query returned no columns");
    }

    let mut rows_iter = stmt.query([])?;

    let mut rows = Vec::new();
    let mut total_rows = 0usize;
    while let Some(row) = rows_iter.next()? {
        let mut values = Vec::with_capacity(columns.len());
        for idx in 0..columns.len() {
            let value = row.get_ref(idx)?;
            values.push(format_sql_value(value));
        }
        if values.iter().all(|v| v.is_empty()) {
            continue;
        }
        if row_limit.is_none_or(|limit| rows.len() < limit) {
            rows.push(values);
        }
        total_rows += 1;
    }

    Ok(MappingTable {
        columns,
        rows,
        total_rows,
    })
}

fn resolve_sql_query(conn: &Connection, options: &MappingReadOptions) -> Result<String> {
    if let Some(query) = options
        .sql_query
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        return Ok(query.to_string());
    }

    let table_name = if let Some(explicit) = options
        .sql_table
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        explicit.to_string()
    } else {
        list_sqlite_tables_conn(conn)?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("database does not contain any tables"))?
    };
    Ok(format!("SELECT * FROM {}", quote_identifier(&table_name)))
}

fn list_sqlite_tables_conn(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt =
        conn.prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY LOWER(name)")?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

fn quote_identifier(name: &str) -> String {
    let escaped = name.replace('"', "\"\"");
    format!("\"{escaped}\"")
}

// ---------------------------------------------------------------------------
// Common helpers

fn resolve_selector(columns: &[String], selector: &ColumnSelector) -> Result<usize> {
    match selector {
        ColumnSelector::Index(idx) => {
            if *idx >= columns.len() {
                anyhow::bail!(
                    "{} is out of range ({} column(s) detected)",
                    selector.describe(),
                    columns.len()
                );
            }
            Ok(*idx)
        }
        ColumnSelector::Name(name) => columns
            .iter()
            .position(|col| col.eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                anyhow!(
                    "column named \"{name}\" not found (available: {})",
                    columns.join(", ")
                )
            }),
    }
}

#[inline]
fn sanitize_value(value: &str) -> String {
    value.trim().to_string()
}

fn format_header(raw: &str, idx: usize) -> String {
    let trimmed = sanitize_value(raw);
    if trimmed.is_empty() {
        format!("Column {}", idx + 1)
    } else {
        trimmed
    }
}

fn ensure_columns(columns: &mut Vec<String>, desired: usize) {
    if columns.len() >= desired {
        return;
    }
    let current = columns.len();
    for idx in current..desired {
        columns.push(format!("Column {}", idx + 1));
    }
}

fn normalize_row(row: &mut Vec<String>, len: usize) {
    row.resize(len, String::new());
}

fn format_excel_header(cell: &ExcelData, idx: usize) -> String {
    let text = format_excel_cell(cell);
    if text.is_empty() {
        format!("Column {}", idx + 1)
    } else {
        text
    }
}

fn format_excel_cell(cell: &ExcelData) -> String {
    match cell {
        ExcelData::Empty => String::new(),
        ExcelData::String(s) => s.trim().to_string(),
        ExcelData::Float(f) => {
            if f.fract() == 0.0 {
                format!("{:.0}", f)
            } else {
                f.to_string()
            }
        }
        ExcelData::Int(i) => i.to_string(),
        ExcelData::Bool(b) => b.to_string(),
        ExcelData::Error(_) => String::new(),
        ExcelData::DateTime(dt) => dt.to_string(),
        ExcelData::DateTimeIso(s) => s.to_string(),
        ExcelData::DurationIso(s) => s.to_string(),
    }
}

fn format_parquet_field(field: &Field) -> String {
    match field {
        Field::Null => String::new(),
        Field::Bool(b) => b.to_string(),
        Field::Byte(v) => v.to_string(),
        Field::Short(v) => v.to_string(),
        Field::Int(v) => v.to_string(),
        Field::Long(v) => v.to_string(),
        Field::UByte(v) => v.to_string(),
        Field::UShort(v) => v.to_string(),
        Field::UInt(v) => v.to_string(),
        Field::ULong(v) => v.to_string(),
        Field::Float16(v) => v.to_string(),
        Field::Float(v) => v.to_string(),
        Field::Double(v) => v.to_string(),
        Field::Str(s) => s.trim().to_string(),
        Field::Bytes(bytes) => encode_bytes(bytes.data()),
        Field::Decimal(value) => format!("{value:?}"),
        Field::Date(v) => v.to_string(),
        Field::TimeMillis(v) => v.to_string(),
        Field::TimeMicros(v) => v.to_string(),
        Field::TimestampMillis(v) => v.to_string(),
        Field::TimestampMicros(v) => v.to_string(),
        Field::Group(group) => format!(
            "{{{}}}",
            group
                .get_column_iter()
                .map(|(name, value)| format!("{name}: {}", format_parquet_field(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Field::ListInternal(list) => format!(
            "[{}]",
            list.elements()
                .iter()
                .map(format_parquet_field)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Field::MapInternal(map) => format!("{map:?}"),
    }
}

#[inline]
fn format_sql_value(value: ValueRef<'_>) -> String {
    match value {
        ValueRef::Null => String::new(),
        ValueRef::Integer(i) => i.to_string(),
        ValueRef::Real(r) => r.to_string(),
        ValueRef::Text(text) => String::from_utf8_lossy(text).trim().to_string(),
        ValueRef::Blob(blob) => encode_bytes(blob),
    }
}

fn encode_bytes<B: AsRef<[u8]>>(bytes: B) -> String {
    let slice = bytes.as_ref();
    if slice.is_empty() {
        String::new()
    } else {
        general_purpose::STANDARD.encode(slice)
    }
}
