//! Mapping panel UI components for the YuNet GUI.

use std::path::PathBuf;

use egui::{Button, ComboBox, RichText, Ui};
use egui_extras::{Column as TableColumn, TableBuilder};
use log::warn;
use rfd::FileDialog;

use yunet_utils::mapping::{MappingFormat, MappingPreview};

use crate::{BatchFile, BatchFileStatus, YuNetApp, theme};

impl YuNetApp {
    /// Renders the mapping import window.
    pub fn show_mapping_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_mapping_window;
        egui::Window::new("Mapping Import")
            .open(&mut open)
            .resizable(true)
            .default_width(600.0)
            .default_height(500.0)
            .show(ctx, |ui| {
                let palette = theme::palette();
                self.show_mapping_content(ui, palette);
            });
        self.show_mapping_window = open;
    }

    /// Renders the mapping import panel content.
    pub fn show_mapping_content(&mut self, ui: &mut Ui, palette: theme::Palette) {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                if ui.button("Select mapping file…").clicked()
                    && let Some(path) = FileDialog::new()
                        .add_filter(
                            "Data files",
                            &[
                                "csv", "tsv", "txt", "xlsx", "xls", "parquet", "db", "sqlite",
                            ],
                        )
                        .pick_file()
                {
                    self.mapping.set_file(path.clone());
                    match self.mapping.reload_preview() {
                        Ok(_) => {
                            self.show_success(format!(
                                "Loaded mapping preview from {}",
                                path.display()
                            ));
                        }
                        Err(err) => {
                            self.show_error("Failed to load mapping preview", err.to_string());
                        }
                    }
                }
                if let Some(path) = &self.mapping.file_path {
                    ui.monospace(path.display().to_string());
                } else {
                    ui.label(RichText::new("No mapping file selected").color(palette.subtle_text));
                }
            });

            if self.mapping.file_path.is_none() {
                ui.label(
                    RichText::new("Import CSV, Excel, Parquet, or SQLite files to drive cropping.")
                        .color(palette.subtle_text),
                );
                return;
            }

            self.mapping_options_ui(ui, palette);

            if ui.button("Reload preview").clicked() {
                match self.mapping.reload_preview() {
                    Ok(_) => self.show_success("Mapping preview updated"),
                    Err(err) => {
                        self.show_error("Failed to load mapping preview", err.to_string());
                    }
                }
            }

            if let Some(err) = &self.mapping.preview_error {
                ui.colored_label(palette.danger, err);
            }

            if let Some(preview) = self.mapping.preview.clone() {
                self.mapping_preview_table(ui, palette, &preview);
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    let ready = self.mapping.source_selector().is_some()
                        && self.mapping.output_selector().is_some();
                    if ui
                        .add_enabled(ready, Button::new("Load mapping entries"))
                        .clicked()
                    {
                        match self.mapping.load_entries() {
                            Ok(_) => self.show_success(format!(
                                "Loaded {} mapping entries",
                                self.mapping.entries.len()
                            )),
                            Err(err) => {
                                self.show_error("Failed to load mapping entries", err.to_string());
                            }
                        }
                    }
                    if ui
                        .add_enabled(
                            !self.mapping.entries.is_empty(),
                            Button::new("Replace batch queue"),
                        )
                        .clicked()
                    {
                        self.push_mapping_to_batch();
                    }
                });

                if !self.mapping.entries.is_empty() {
                    ui.label(format!(
                        "{} mapping row(s) ready for batch export",
                        self.mapping.entries.len()
                    ));
                    for entry in self.mapping.entries.iter().take(3) {
                        ui.monospace(format!("{} → {}", entry.source_path, entry.output_name));
                    }
                    if self.mapping.entries.len() > 3 {
                        ui.label(RichText::new("…").color(palette.subtle_text));
                    }
                }
            } else {
                ui.label(
                    RichText::new("Preview will appear here after loading.")
                        .color(palette.subtle_text),
                );
            }
        });
    }

    /// Renders format-specific mapping options UI.
    fn mapping_options_ui(&mut self, ui: &mut Ui, palette: theme::Palette) {
        let prev_format = self.mapping.effective_format();
        egui::ComboBox::from_label("Format")
            .selected_text(
                self.mapping
                    .format_override
                    .map(|f| f.display_name().to_string())
                    .or_else(|| {
                        self.mapping
                            .detected_format
                            .map(|f| format!("Auto ({})", f.display_name()))
                    })
                    .unwrap_or_else(|| "Auto".to_string()),
            )
            .show_ui(ui, |ui| {
                let auto_label = self
                    .mapping
                    .detected_format
                    .map(|f| format!("Auto ({})", f.display_name()))
                    .unwrap_or_else(|| "Auto".to_string());
                if ui
                    .selectable_label(self.mapping.format_override.is_none(), auto_label)
                    .clicked()
                {
                    self.mapping.format_override = None;
                }
                for format in [
                    MappingFormat::Csv,
                    MappingFormat::Excel,
                    MappingFormat::Parquet,
                    MappingFormat::Sqlite,
                ] {
                    if ui
                        .selectable_label(
                            self.mapping.format_override == Some(format),
                            format.display_name(),
                        )
                        .clicked()
                    {
                        self.mapping.format_override = Some(format);
                    }
                }
            });

        let new_format = self.mapping.effective_format();
        if new_format != prev_format {
            self.mapping.refresh_catalog();
        }

        ui.checkbox(&mut self.mapping.has_headers, "First row contains headers");

        if matches!(new_format, Some(MappingFormat::Csv)) || new_format.is_none() {
            ui.horizontal(|ui| {
                ui.label("Delimiter");
                let response = ui.text_edit_singleline(&mut self.mapping.delimiter_input);
                if response.changed() {
                    if self.mapping.delimiter_input.is_empty() {
                        self.mapping.delimiter_input.push(',');
                    } else {
                        let first = self.mapping.delimiter_input.chars().next().unwrap();
                        self.mapping.delimiter_input = first.to_string();
                    }
                }
                ui.label(
                    RichText::new("Only the first character is used.").color(palette.subtle_text),
                );
            });
        }

        if matches!(new_format, Some(MappingFormat::Excel)) {
            if self.mapping.catalog.sheets.is_empty() {
                ui.label(
                    RichText::new("Reload preview to discover sheet names.")
                        .color(palette.subtle_text),
                );
            } else {
                ComboBox::from_label("Sheet")
                    .selected_text(if self.mapping.sheet_name.is_empty() {
                        self.mapping
                            .catalog
                            .sheets
                            .first()
                            .cloned()
                            .unwrap_or_else(|| "Sheet1".to_string())
                    } else {
                        self.mapping.sheet_name.clone()
                    })
                    .show_ui(ui, |ui| {
                        for sheet in &self.mapping.catalog.sheets {
                            if ui
                                .selectable_label(self.mapping.sheet_name == *sheet, sheet.clone())
                                .clicked()
                            {
                                self.mapping.sheet_name = sheet.clone();
                            }
                        }
                    });
            }
        }

        if matches!(new_format, Some(MappingFormat::Sqlite)) {
            if self.mapping.catalog.sql_tables.is_empty() {
                ui.label(
                    RichText::new("Reload preview to discover tables.").color(palette.subtle_text),
                );
            } else {
                ComboBox::from_label("Table")
                    .selected_text(if self.mapping.sql_table.is_empty() {
                        self.mapping
                            .catalog
                            .sql_tables
                            .first()
                            .cloned()
                            .unwrap_or_else(|| "main".to_string())
                    } else {
                        self.mapping.sql_table.clone()
                    })
                    .show_ui(ui, |ui| {
                        for table in &self.mapping.catalog.sql_tables {
                            if ui
                                .selectable_label(self.mapping.sql_table == *table, table.clone())
                                .clicked()
                            {
                                self.mapping.sql_table = table.clone();
                            }
                        }
                    });
            }
            ui.label("Custom SQL query (optional):");
            ui.text_edit_multiline(&mut self.mapping.sql_query);
        }
    }

    /// Renders the mapping preview table with column selection.
    fn mapping_preview_table(
        &mut self,
        ui: &mut Ui,
        palette: theme::Palette,
        preview: &MappingPreview,
    ) {
        ui.add_space(8.0);
        let truncated_note = if preview.truncated {
            " (truncated)"
        } else {
            ""
        };
        ui.label(format!(
            "Previewing {} of {} rows{}",
            preview.rows.len(),
            preview.total_rows,
            truncated_note
        ));

        if preview.columns.is_empty() {
            ui.label(
                RichText::new("No columns detected in the mapping file.").color(palette.danger),
            );
            return;
        }

        ui.horizontal(|ui| {
            ComboBox::from_label("Source column")
                .selected_text(
                    self.mapping
                        .selected_column_name(self.mapping.source_column_idx),
                )
                .show_ui(ui, |ui| {
                    for (idx, name) in preview.columns.iter().enumerate() {
                        if ui
                            .selectable_label(self.mapping.source_column_idx == Some(idx), name)
                            .clicked()
                        {
                            self.mapping.source_column_idx = Some(idx);
                        }
                    }
                });

            ComboBox::from_label("Output column")
                .selected_text(
                    self.mapping
                        .selected_column_name(self.mapping.output_column_idx),
                )
                .show_ui(ui, |ui| {
                    for (idx, name) in preview.columns.iter().enumerate() {
                        if ui
                            .selectable_label(self.mapping.output_column_idx == Some(idx), name)
                            .clicked()
                        {
                            self.mapping.output_column_idx = Some(idx);
                        }
                    }
                });
        });

        let mut table = TableBuilder::new(ui)
            .striped(true)
            .auto_shrink([false, false]);
        for _ in &preview.columns {
            table = table.column(TableColumn::auto().resizable(true));
        }

        table
            .header(24.0, |mut header| {
                for (idx, name) in preview.columns.iter().enumerate() {
                    header.col(|ui| {
                        let mut text = RichText::new(name.clone()).strong();
                        if self.mapping.source_column_idx == Some(idx) {
                            text = text.color(palette.accent);
                        }
                        if self.mapping.output_column_idx == Some(idx) {
                            text = text.color(palette.success);
                        }
                        ui.label(text);
                    });
                }
            })
            .body(|body| {
                body.rows(20.0, preview.rows.len(), |mut row| {
                    let row_idx = row.index();
                    if let Some(values) = preview.rows.get(row_idx) {
                        for value in values {
                            row.col(|ui| {
                                ui.label(value);
                            });
                        }
                    }
                });
            });

        if let (Some(src_idx), Some(out_idx)) = (
            self.mapping.source_column_idx,
            self.mapping.output_column_idx,
        ) {
            ui.add_space(6.0);
            ui.label("Preview mappings:");
            for sample in preview.rows.iter().take(3) {
                if let (Some(src), Some(dest)) = (sample.get(src_idx), sample.get(out_idx)) {
                    let display = Self::display_with_output_ext(
                        dest,
                        self.settings.crop.output_format.as_str(),
                    );
                    ui.monospace(format!("{src} → {display}"));
                }
            }
        }
    }

    /// Converts loaded mapping entries into batch queue items.
    pub fn push_mapping_to_batch(&mut self) {
        if self.mapping.entries.is_empty() {
            self.show_error(
                "No mapping entries loaded",
                "Load entries before queuing batch exports.",
            );
            return;
        }

        let mut files = Vec::with_capacity(self.mapping.entries.len());
        for entry in &self.mapping.entries {
            let raw = PathBuf::from(&entry.source_path);
            let path = if raw.is_absolute() {
                raw
            } else if let Some(base) = &self.mapping.base_dir {
                base.join(raw)
            } else {
                raw
            };
            if !path.exists() {
                warn!(
                    "Mapping entry skipped—source {} not found",
                    entry.source_path
                );
                continue;
            }
            files.push(BatchFile {
                path,
                status: BatchFileStatus::Pending,
                output_override: Some(PathBuf::from(&entry.output_name)),
            });
        }

        if files.is_empty() {
            self.show_error(
                "Mapping entries invalid",
                "All mapping rows referenced missing files.",
            );
            return;
        }

        self.batch_files = files;
        self.batch_current_index = None;
        self.show_success(format!(
            "Queued {} mapping row(s) for batch export",
            self.batch_files.len()
        ));
        log::info!(
            "Loaded {} mapping rows into batch processing queue",
            self.batch_files.len()
        );
    }
}
