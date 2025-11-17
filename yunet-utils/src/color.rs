//! Basic color utilities shared across CLI and GUI surfaces.

use serde::{Deserialize, Serialize};

/// Simple RGBA color stored in 8-bit channels.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct RgbaColor {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
    pub alpha: u8,
}

impl RgbaColor {
    /// Constructs an opaque RGB color.
    pub const fn opaque(red: u8, green: u8, blue: u8) -> Self {
        Self {
            red,
            green,
            blue,
            alpha: 255,
        }
    }

    /// Returns the color as normalized HSV tuple (hue in degrees, saturation/value 0.0..1.0).
    pub fn to_hsv(self) -> (f32, f32, f32) {
        rgb_to_hsv(self.red, self.green, self.blue)
    }

    /// Builds a color from HSV values (hue in degrees, saturation/value 0.0..1.0).
    pub fn from_hsv(h: f32, s: f32, v: f32) -> Self {
        let (r, g, b) = hsv_to_rgb(h, s, v);
        Self::opaque(r, g, b)
    }
}

impl Default for RgbaColor {
    fn default() -> Self {
        Self::opaque(0, 0, 0)
    }
}

/// Convert RGB channels (0-255) to HSV (hue in degrees 0-360, saturation/value 0-1).
pub fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;

    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let delta = max - min;

    let hue = if delta.abs() < f32::EPSILON {
        0.0
    } else if (max - rf).abs() < f32::EPSILON {
        60.0 * (((gf - bf) / delta) % 6.0)
    } else if (max - gf).abs() < f32::EPSILON {
        60.0 * (((bf - rf) / delta) + 2.0)
    } else {
        60.0 * (((rf - gf) / delta) + 4.0)
    };

    let hue = if hue < 0.0 { hue + 360.0 } else { hue };
    let saturation = if max.abs() < f32::EPSILON {
        0.0
    } else {
        delta / max
    };
    (hue, saturation, max)
}

/// Convert HSV (hue in degrees, saturation/value 0-1) to RGB channels (0-255).
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    if s <= 0.0 {
        let val = (v * 255.0).round().clamp(0.0, 255.0) as u8;
        return (val, val, val);
    }

    let hue = if h.is_nan() { 0.0 } else { h.rem_euclid(360.0) };
    let c = v * s;
    let x = c * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r1, g1, b1) = match hue {
        h if h < 60.0 => (c, x, 0.0),
        h if h < 120.0 => (x, c, 0.0),
        h if h < 180.0 => (0.0, c, x),
        h if h < 240.0 => (0.0, x, c),
        h if h < 300.0 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    let to_byte = |value: f32| -> u8 { ((value + m) * 255.0).round().clamp(0.0, 255.0) as u8 };

    (to_byte(r1), to_byte(g1), to_byte(b1))
}

/// Parse a hexadecimal color string. Accepts `#RGB`, `#RRGGBB`, `#RRGGBBAA`, with or without `#`.
pub fn parse_hex_color(input: &str) -> Option<RgbaColor> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut hex = trimmed;
    if let Some(stripped) = hex.strip_prefix('#') {
        hex = stripped;
    } else if let Some(stripped) = hex.strip_prefix("0x") {
        hex = stripped;
    }
    let hex = hex.replace('_', "");
    match hex.len() {
        3 => Some(RgbaColor::opaque(
            replicate_nibble(hex.get(0..1)?)?,
            replicate_nibble(hex.get(1..2)?)?,
            replicate_nibble(hex.get(2..3)?)?,
        )),
        4 => Some(RgbaColor {
            red: replicate_nibble(hex.get(0..1)?)?,
            green: replicate_nibble(hex.get(1..2)?)?,
            blue: replicate_nibble(hex.get(2..3)?)?,
            alpha: replicate_nibble(hex.get(3..4)?)?,
        }),
        6 => Some(RgbaColor {
            red: parse_byte(hex.get(0..2)?)?,
            green: parse_byte(hex.get(2..4)?)?,
            blue: parse_byte(hex.get(4..6)?)?,
            alpha: 255,
        }),
        8 => Some(RgbaColor {
            red: parse_byte(hex.get(0..2)?)?,
            green: parse_byte(hex.get(2..4)?)?,
            blue: parse_byte(hex.get(4..6)?)?,
            alpha: parse_byte(hex.get(6..8)?)?,
        }),
        _ => None,
    }
}

fn parse_byte(slice: &str) -> Option<u8> {
    u8::from_str_radix(slice, 16).ok()
}

fn replicate_nibble(slice: &str) -> Option<u8> {
    let nib = u8::from_str_radix(slice, 16).ok()?;
    Some((nib << 4) | nib)
}
