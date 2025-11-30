//! Central icon loader for the GUI.
//!
//! Wraps the SVG assets under `assets/icons` so they can be reused across
//! panels with consistent sizing.

use egui::{Image, vec2};

macro_rules! declare_icon_set {
    ($($name:ident),* $(,)?) => {
        /// A reusable set of SVG icons embedded at compile time.
        #[derive(Clone)]
        pub struct IconSet {
            $(
                $name: SvgIcon,
            )*
            default_size: f32,
        }

        impl IconSet {
            /// Installs SVG loaders and prepares the icon set.
            pub fn new(ctx: &egui::Context) -> Self {
                egui_extras::install_image_loaders(ctx);
                Self {
                    $(
                        $name: SvgIcon::new(
                            concat!("bytes://icons/", stringify!($name), ".svg"),
                            include_bytes!(concat!("../../assets/icons/gui-", stringify!($name), "-svgrepo-com.svg")),
                        ),
                    )*
                    default_size: 18.0,
                }
            }

            /// Default icon size to keep buttons consistent.
            pub fn default_size(&self) -> f32 {
                self.default_size
            }

            $(
                pub fn $name(&self, size: f32) -> Image<'static> {
                    self.$name.image(size)
                }
            )*
        }
    }
}

declare_icon_set!(
    photo,
    gallery,
    folder_open,
    export,
    palette,
    run,
    settings,
    spreadsheet,
    webcam,
    network,
    batch,
    automation,
    enhance,
    instagram,
    id_card,
    linkedin,
    passport,
    portrait,
    account,
);

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
