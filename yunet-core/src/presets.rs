//! Size presets for cropped outputs.
//!
//! Provides a small collection of commonly used output sizes and helpers.

/// A named crop size preset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CropPreset {
    /// Human-visible name for UI/CLI selection.
    pub name: &'static str,
    /// Output width in pixels (0 for unspecified/custom).
    pub width: u32,
    /// Output height in pixels (0 for unspecified/custom).
    pub height: u32,
    /// Short description for tooling and UI.
    pub description: &'static str,
}

impl CropPreset {
    /// Create a new preset.
    pub const fn new(
        name: &'static str,
        width: u32,
        height: u32,
        description: &'static str,
    ) -> Self {
        Self {
            name,
            width,
            height,
            description,
        }
    }
}

/// Returns a list of standard presets.
static PRESETS: [CropPreset; 7] = [
    CropPreset::new(
        "LinkedIn",
        400,
        400,
        "Square professional profile photo (400×400)",
    ),
    CropPreset::new("Passport", 413, 531, "Passport photo dimensions (413×531)"),
    CropPreset::new("Instagram", 1080, 1080, "Instagram square post (1080×1080)"),
    CropPreset::new("ID Card", 332, 498, "ID card photo size (332×498)"),
    CropPreset::new("Avatar", 512, 512, "Small square avatar (512×512)"),
    CropPreset::new("Headshot", 600, 800, "Vertical headshot (600×800)"),
    CropPreset::new("Custom", 0, 0, "User-defined custom dimensions"),
];

pub fn standard_presets() -> &'static [CropPreset] {
    &PRESETS
}

/// Find a preset by name (case-insensitive). Returns `None` if not found.
pub fn preset_by_name(name: &str) -> Option<CropPreset> {
    let lookup_key = normalize_name(name);
    for p in standard_presets() {
        if normalize_name(p.name) == lookup_key {
            return Some(p.clone());
        }
    }
    None
}

fn normalize_name(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_expected_linkedin_preset() {
        let presets = standard_presets();
        let found = presets
            .iter()
            .find(|p| p.name == "LinkedIn")
            .expect("LinkedIn preset present");
        assert_eq!(found.width, 400);
        assert_eq!(found.height, 400);
    }

    #[test]
    fn preset_lookup_by_name_is_case_insensitive() {
        let p = preset_by_name("instagram").expect("instagram preset");
        assert_eq!(p.width, 1080);
        assert_eq!(p.name, "Instagram");
    }

    #[test]
    fn preset_lookup_ignores_spacing_and_case() {
        let p = preset_by_name("IDCard").expect("ID Card preset");
        assert_eq!(p.width, 332);
        assert_eq!(p.height, 498);
        let p2 = preset_by_name("id card").expect("ID Card preset with space");
        assert_eq!(p2.name, "ID Card");
    }

    #[test]
    fn all_presets_have_expected_structure() {
        let presets = standard_presets();
        assert!(!presets.is_empty());
        // All non-Custom presets must have non-zero width and height
        for p in presets.iter() {
            if p.name != "Custom" {
                assert!(p.width > 0, "preset {} should have width > 0", p.name);
                assert!(p.height > 0, "preset {} should have height > 0", p.name);
            } else {
                // Custom preset is allowed to be zero-sized
                assert_eq!(p.width, 0);
                assert_eq!(p.height, 0);
            }
        }
    }
}
