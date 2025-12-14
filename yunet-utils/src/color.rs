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

/// Convert RGB channels (0-255) to HSL (hue in degrees 0-360, saturation 0-1, lightness 0-1).
pub fn rgb_to_hsl(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
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

    let lightness = (max + min) / 2.0;

    let saturation = if delta.abs() < f32::EPSILON {
        0.0
    } else {
        delta / (1.0 - (lightness.mul_add(2.0, -1.0)).abs())
    };

    (hue, saturation, lightness)
}

/// Convert HSL (hue in degrees, saturation 0-1, lightness 0-1) to RGB channels (0-255).
pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let c = (1.0 - (l.mul_add(2.0, -1.0)).abs()) * s;
    let hue = if h.is_nan() { 0.0 } else { h.rem_euclid(360.0) };
    let x = c * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r1, g1, b1) = match hue {
        h if h < 60.0 => (c, x, 0.0),
        h if h < 120.0 => (x, c, 0.0),
        h if h < 180.0 => (0.0, c, x),
        h if h < 240.0 => (0.0, x, c),
        h if h < 300.0 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    let to_byte = |value: f32| -> u8 { ((value + m) * 255.0) as u8 };

    (to_byte(r1), to_byte(g1), to_byte(b1))
}

/// Convert RGB channels (0-255) to CMYK (0-1 for all channels).
pub fn rgb_to_cmyk(r: u8, g: u8, b: u8) -> (f32, f32, f32, f32) {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;

    let k = 1.0 - rf.max(gf).max(bf);
    if (1.0 - k).abs() < f32::EPSILON {
        return (0.0, 0.0, 0.0, 1.0);
    }

    let rgb_channel_to_cymk = |value: f32| -> f32 { (1.0 - value - k) / (1.0 - k) };

    (
        rgb_channel_to_cymk(rf), // cyan
        rgb_channel_to_cymk(gf), // magenta
        rgb_channel_to_cymk(bf), // yellow
        k,                       // black
    )
}

/// Convert CMYK (0-1 for all channels) to RGB channels (0-255).
pub fn cmyk_to_rgb(c: f32, m: f32, y: f32, k: f32) -> (u8, u8, u8) {
    let to_rgb_channel = |value: f32| -> u8 { (255.0 * (1.0 - value) * (1.0 - k)) as u8 };
    (
        to_rgb_channel(c), // red
        to_rgb_channel(m), // green
        to_rgb_channel(y), // blue
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_hsl_and_back() {
        let (r, g, b) = (255, 0, 0); // Red
        let (h, s, l) = rgb_to_hsl(r, g, b);
        assert_eq!(h, 0.0);
        assert_eq!(s, 1.0);
        assert_eq!(l, 0.5);
        let (r2, g2, b2) = hsl_to_rgb(h, s, l);
        assert_eq!((r, g, b), (r2, g2, b2));

        let (r, g, b) = (0, 255, 0); // Green
        let (h, s, l) = rgb_to_hsl(r, g, b);
        assert_eq!(h, 120.0);
        assert_eq!(s, 1.0);
        assert_eq!(l, 0.5);
        let (r2, g2, b2) = hsl_to_rgb(h, s, l);
        assert_eq!((r, g, b), (r2, g2, b2));

        let (r, g, b) = (128, 128, 128); // Gray
        let (h, s, l) = rgb_to_hsl(r, g, b);
        assert_eq!(h, 0.0); // Hue is undefined/0 for grayscale
        assert_eq!(s, 0.0);
        assert!((l - 0.5).abs() < 0.01);
        let (r2, g2, b2) = hsl_to_rgb(h, s, l);
        assert_eq!((r, g, b), (r2, g2, b2));
    }

    #[test]
    fn test_rgb_to_cmyk_and_back() {
        let (r, g, b) = (255, 0, 0); // Red
        let (c, m, y, k) = rgb_to_cmyk(r, g, b);
        assert_eq!(c, 0.0);
        assert_eq!(m, 1.0);
        assert_eq!(y, 1.0);
        assert_eq!(k, 0.0);
        let (r2, g2, b2) = cmyk_to_rgb(c, m, y, k);
        assert_eq!((r, g, b), (r2, g2, b2));

        let (r, g, b) = (0, 0, 0); // Black
        let (c, m, y, k) = rgb_to_cmyk(r, g, b);
        assert_eq!(c, 0.0);
        assert_eq!(m, 0.0);
        assert_eq!(y, 0.0);
        assert_eq!(k, 1.0);
        let (r2, g2, b2) = cmyk_to_rgb(c, m, y, k);
        assert_eq!((r, g, b), (r2, g2, b2));

        let (r, g, b) = (255, 255, 255); // White
        let (c, m, y, k) = rgb_to_cmyk(r, g, b);
        assert_eq!(c, 0.0);
        assert_eq!(m, 0.0);
        assert_eq!(y, 0.0);
        assert_eq!(k, 0.0);
        let (r2, g2, b2) = cmyk_to_rgb(c, m, y, k);
        assert_eq!((r, g, b), (r2, g2, b2));
    }
}
