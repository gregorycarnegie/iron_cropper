//! Central icon loader for the GUI.
//!
//! Wraps the SVG assets under `assets/icons` so they can be reused across
//! panels with consistent sizing.

use egui::{Image, vec2};

/// A reusable set of SVG icons embedded at compile time.
#[derive(Clone)]
pub struct IconSet {
    photo: SvgIcon,
    gallery: SvgIcon,
    folder_open: SvgIcon,
    export: SvgIcon,
    palette: SvgIcon,
    settings: SvgIcon,
    spreadsheet: SvgIcon,
    webcam: SvgIcon,
    default_size: f32,
}

impl IconSet {
    /// Installs SVG loaders and prepares the icon set.
    pub fn new(ctx: &egui::Context) -> Self {
        egui_extras::install_image_loaders(ctx);
        Self {
            photo: SvgIcon::new(
                "bytes://icons/photo.svg",
                include_bytes!("../../assets/icons/gui-picture-svgrepo-com.svg"),
            ),
            gallery: SvgIcon::new(
                "bytes://icons/gallery.svg",
                include_bytes!("../../assets/icons/gui-pictures-svgrepo-com.svg"),
            ),
            folder_open: SvgIcon::new(
                "bytes://icons/folder-open.svg",
                include_bytes!("../../assets/icons/gui-folder-open-svgrepo-com.svg"),
            ),
            export: SvgIcon::new(
                "bytes://icons/export.svg",
                include_bytes!("../../assets/icons/gui-export-svgrepo-com.svg"),
            ),
            palette: SvgIcon::new(
                "bytes://icons/palette.svg",
                include_bytes!("../../assets/icons/gui-palette-svgrepo-com.svg"),
            ),
            settings: SvgIcon::new(
                "bytes://icons/settings.svg",
                include_bytes!("../../assets/icons/gui-settings-svgrepo-com.svg"),
            ),
            spreadsheet: SvgIcon::new(
                "bytes://icons/spreadsheet.svg",
                include_bytes!("../../assets/icons/gui-spreadsheet-svgrepo-com.svg"),
            ),
            webcam: SvgIcon::new(
                "bytes://icons/webcam.svg",
                include_bytes!("../../assets/icons/gui-webcam-svgrepo-com.svg"),
            ),
            default_size: 18.0,
        }
    }

    /// Default icon size to keep buttons consistent.
    pub fn default_size(&self) -> f32 {
        self.default_size
    }

    pub fn photo(&self, size: f32) -> Image<'static> {
        self.photo.image(size)
    }

    pub fn gallery(&self, size: f32) -> Image<'static> {
        self.gallery.image(size)
    }

    pub fn folder_open(&self, size: f32) -> Image<'static> {
        self.folder_open.image(size)
    }

    pub fn export(&self, size: f32) -> Image<'static> {
        self.export.image(size)
    }

    pub fn palette(&self, size: f32) -> Image<'static> {
        self.palette.image(size)
    }

    pub fn settings(&self, size: f32) -> Image<'static> {
        self.settings.image(size)
    }

    pub fn spreadsheet(&self, size: f32) -> Image<'static> {
        self.spreadsheet.image(size)
    }

    pub fn webcam(&self, size: f32) -> Image<'static> {
        self.webcam.image(size)
    }
}

#[derive(Clone, Copy)]
struct SvgIcon {
    uri: &'static str,
    bytes: &'static [u8],
}

impl SvgIcon {
    const fn new(uri: &'static str, bytes: &'static [u8]) -> Self {
        Self { uri, bytes }
    }

    fn image(self, size: f32) -> Image<'static> {
        Image::from_bytes(self.uri, self.bytes).fit_to_exact_size(vec2(size, size))
    }
}
