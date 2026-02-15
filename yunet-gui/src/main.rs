#![cfg_attr(
    all(target_os = "windows", not(debug_assertions)),
    windows_subsystem = "windows"
)]

//! Desktop GUI for YuNet face detection.

use eframe::{NativeOptions, egui};
use ico::IconDir;
use log::{LevelFilter, warn};
use std::{io::Cursor, sync::Arc};
use yunet_gui::YuNetApp;
use yunet_utils::init_logging;

/// Main entry point for the GUI application.
fn main() -> eframe::Result<()> {
    init_logging(LevelFilter::Info).expect("failed to initialize logging");
    let mut options = NativeOptions::default();

    // Set initial window size to avoid scrunched UI on first launch
    options.viewport = options.viewport.with_inner_size([1280.0, 800.0]);

    if let Some(icon) = load_app_icon() {
        options.viewport = options.viewport.with_icon(Arc::new(icon));
    }

    eframe::run_native(
        "Face Crop Studio",
        options,
        Box::new(|cc| Ok(Box::new(YuNetApp::new(cc)))),
    )
}

/// Load the embedded ICO file and convert it into an `eframe` icon, if possible.
fn load_app_icon() -> Option<egui::IconData> {
    const ICON_BYTES: &[u8] = include_bytes!("../assets/app_icon.ico");

    let dir = match IconDir::read(Cursor::new(ICON_BYTES)) {
        Ok(dir) => dir,
        Err(err) => {
            warn!("Failed to read embedded app icon: {err}");
            return None;
        }
    };

    let mut best: Option<(ico::IconImage, u32)> = None;
    for entry in dir.entries() {
        match entry.decode() {
            Ok(image) => {
                let score = image.width().saturating_mul(image.height());
                if best
                    .as_ref()
                    .is_none_or(|(_, best_score)| score > *best_score)
                {
                    best = Some((image, score));
                }
            }
            Err(err) => warn!("Failed to decode icon entry: {err}"),
        }
    }

    if let Some((image, _)) = best {
        Some(egui::IconData {
            rgba: image.rgba_data().to_vec(),
            width: image.width(),
            height: image.height(),
        })
    } else {
        warn!("Embedded icon did not yield any usable RGBA data");
        None
    }
}
