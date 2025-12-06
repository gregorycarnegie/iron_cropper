//! Color parsing utilities for CLI color specifications.

use yunet_utils::{RgbaColor, hsv_to_rgb, parse_hex_color};

/// Parse fill color specification from CLI argument.
/// Accepts formats: #RRGGBB, rgb(), hsv(), or comma-separated values.
pub fn parse_fill_color_spec(raw: &str) -> Result<RgbaColor, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("fill color value is empty".to_string());
    }

    if let Some(color) = parse_hex_color(trimmed) {
        return Ok(color);
    }
    if let Some(args) = parse_fn_args(trimmed, "rgb") {
        let (r, g, b) = parse_rgb_components(&args)?;
        let alpha = args
            .get(3)
            .map(|value| parse_alpha_value(value))
            .transpose()?
            .unwrap_or(255);
        return Ok(RgbaColor {
            red: r,
            green: g,
            blue: b,
            alpha,
        });
    }
    if let Some(args) = parse_fn_args(trimmed, "hsv") {
        if args.len() < 3 {
            return Err("hsv() requires three values: hue,saturation,value".to_string());
        }
        let hue = parse_hue_value(args[0])?;
        let sat = parse_percentage_value(args[1])?;
        let val = parse_percentage_value(args[2])?;
        let (r, g, b) = hsv_to_rgb(hue, sat, val);
        return Ok(RgbaColor::opaque(r, g, b));
    }

    if trimmed.contains(',') {
        let parts: Vec<_> = trimmed
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        if parts.len() >= 3 {
            let (r, g, b) = parse_rgb_components(&parts)?;
            return Ok(RgbaColor::opaque(r, g, b));
        }
    }

    Err(format!(
        "unrecognized fill color format '{}'; expected #RRGGBB, rgb(), or hsv()",
        trimmed
    ))
}

fn parse_fn_args<'a>(input: &'a str, name: &str) -> Option<Vec<&'a str>> {
    let trimmed = input.trim();
    let open = trimmed.find('(')?;
    let close = trimmed.rfind(')')?;
    if close <= open {
        return None;
    }
    if !trimmed[..open].trim().eq_ignore_ascii_case(name) {
        return None;
    }
    let inner = &trimmed[open + 1..close];
    let args = inner
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    if args.is_empty() { None } else { Some(args) }
}

fn parse_rgb_components(parts: &[&str]) -> Result<(u8, u8, u8), String> {
    if parts.len() < 3 {
        return Err("expected three values for rgb()".to_string());
    }
    Ok((
        parse_rgb_value(parts[0])?,
        parse_rgb_value(parts[1])?,
        parse_rgb_value(parts[2])?,
    ))
}

fn parse_rgb_value(token: &str) -> Result<u8, String> {
    let value: f32 = token
        .parse()
        .map_err(|_| format!("invalid RGB component '{}'", token))?;
    if !(0.0..=255.0).contains(&value) {
        return Err(format!(
            "RGB component '{}' must be between 0 and 255",
            token
        ));
    }
    Ok(value.round() as u8)
}

fn parse_alpha_value(token: &str) -> Result<u8, String> {
    let trimmed = token.trim();
    let normalized = if let Some(stripped) = trimmed.strip_suffix('%') {
        let pct = stripped
            .trim()
            .parse::<f32>()
            .map_err(|_| format!("invalid alpha percentage '{}'", token))?;
        pct / 100.0
    } else {
        let value: f32 = trimmed
            .parse()
            .map_err(|_| format!("invalid alpha value '{}'", token))?;
        if value > 1.0 { value / 255.0 } else { value }
    };
    Ok((normalized.clamp(0.0, 1.0) * 255.0).round() as u8)
}

fn parse_hue_value(token: &str) -> Result<f32, String> {
    let mut raw = token.trim().to_string();
    if raw.len() >= 3 && raw[raw.len() - 3..].eq_ignore_ascii_case("deg") {
        raw.truncate(raw.len() - 3);
        raw = raw.trim_end().to_string();
    }
    if raw.ends_with('°') {
        raw.pop();
        raw = raw.trim_end().to_string();
    }
    let value: f32 = raw
        .parse()
        .map_err(|_| format!("invalid hue '{}'", token))?;
    Ok(value.rem_euclid(360.0))
}

fn parse_percentage_value(token: &str) -> Result<f32, String> {
    let trimmed = token.trim();
    if let Some(stripped) = trimmed.strip_suffix('%') {
        let pct = stripped
            .trim()
            .parse::<f32>()
            .map_err(|_| format!("invalid percentage '{}'", token))?;
        return Ok((pct / 100.0).clamp(0.0, 1.0));
    }
    let value: f32 = trimmed
        .parse()
        .map_err(|_| format!("invalid component '{}'", token))?;
    if value > 1.0 {
        Ok((value / 100.0).clamp(0.0, 1.0))
    } else {
        Ok(value.clamp(0.0, 1.0))
    }
}
