//! Helpers for exporting cropped images with flexible encoding and metadata handling.
//!
//! This module centralizes output-format selection, compression tuning, and metadata
//! preservation so that both the CLI and GUI can share a single implementation.

use crate::{
    config::{CropSettings, MetadataMode, MetadataSettings},
    quality::Quality,
};

use anyhow::{Context, Result};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use crc32fast::Hasher as Crc32;
use image::{
    DynamicImage, ExtendedColorType, ImageEncoder,
    codecs::{
        jpeg::JpegEncoder,
        png::{CompressionType, FilterType, PngEncoder},
        webp::WebPEncoder,
    },
};
use log::{debug, warn};
use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};
use std::{
    fs,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

/// Canonical image formats supported by the exporter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormatHint {
    Png,
    Jpeg,
    Webp,
}

impl ImageFormatHint {
    /// Determine format from a filesystem extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        ext.parse().ok()
    }
}

impl Default for ImageFormatHint {
    fn default() -> Self {
        Self::Png
    }
}

impl std::str::FromStr for ImageFormatHint {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "png" => Ok(Self::Png),
            "jpg" | "jpeg" => Ok(Self::Jpeg),
            "webp" => Ok(Self::Webp),
            other => Err(format!("unknown image format '{other}'")),
        }
    }
}

/// Simplified PNG compression strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PngCompression {
    Fast,
    Default,
    Best,
}

impl PngCompression {
    /// Parse compression string/level into a compression strategy.
    pub fn parse(input: &str) -> Self {
        let normalized = input.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "fast" => Self::Fast,
            "best" => Self::Best,
            "default" => Self::Default,
            _ => {
                if let Ok(level) = normalized.parse::<u8>() {
                    match level {
                        0..=3 => Self::Fast,
                        7..=9 => Self::Best,
                        _ => Self::Default,
                    }
                } else {
                    warn!(
                        "Unknown PNG compression '{}', falling back to default strategy",
                        input
                    );
                    Self::Default
                }
            }
        }
    }

    fn into_image(self) -> CompressionType {
        match self {
            Self::Fast => CompressionType::Fast,
            Self::Default => CompressionType::Default,
            Self::Best => CompressionType::Best,
        }
    }
}

/// Immutable configuration derived from the user's crop settings.
#[derive(Debug, Clone)]
pub struct OutputOptions {
    pub format: Option<ImageFormatHint>,
    pub auto_detect: bool,
    pub jpeg_quality: u8,
    pub png_compression: PngCompression,
    pub webp_quality: u8,
    pub metadata: MetadataSettings,
}

impl OutputOptions {
    /// Build `OutputOptions` from persistent crop settings.
    pub fn from_crop_settings(settings: &CropSettings) -> Self {
        Self {
            format: settings.output_format.parse().ok(),
            auto_detect: settings.auto_detect_format,
            jpeg_quality: settings.jpeg_quality.clamp(1, 100),
            png_compression: PngCompression::parse(&settings.png_compression),
            webp_quality: settings.webp_quality.min(100),
            metadata: settings.metadata.clone(),
        }
    }
}

/// Runtime metadata passed in from the caller when exporting a single crop.
#[derive(Debug, Clone, Default)]
pub struct MetadataContext<'a> {
    pub source_path: Option<&'a Path>,
    pub crop_settings: Option<&'a CropSettings>,
    pub detection_score: Option<f32>,
    pub quality: Option<Quality>,
    pub quality_score: Option<f64>,
}

/// Save an image using the provided options and metadata context.
pub fn save_dynamic_image(
    image: &DynamicImage,
    destination: &Path,
    options: &OutputOptions,
    metadata: &MetadataContext<'_>,
) -> Result<()> {
    if let Some(parent) = destination.parent().filter(|p| !p.exists()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let format = determine_format(destination, options);
    debug!(
        "Saving crop to {} using {:?} format",
        destination.display(),
        format
    );

    let mut encoded = match format {
        ImageFormatHint::Png => encode_png(image, options.png_compression)?,
        ImageFormatHint::Jpeg => encode_jpeg(image, options.jpeg_quality)?,
        ImageFormatHint::Webp => encode_webp(image)?,
    };

    // Prepare metadata payload if applicable.
    let custom_payload = build_custom_metadata_payload(&options.metadata, metadata)?;

    match format {
        ImageFormatHint::Png => {
            if !matches!(options.metadata.mode, MetadataMode::Strip) {
                let mut exif_chunks = Vec::new();
                if matches!(options.metadata.mode, MetadataMode::Preserve) {
                    exif_chunks = load_png_exif_chunks(metadata.source_path);
                }
                encoded = inject_png_metadata(encoded, &exif_chunks, custom_payload.as_deref());
            }
        }
        ImageFormatHint::Jpeg => {
            encoded = inject_jpeg_metadata(
                encoded,
                if matches!(options.metadata.mode, MetadataMode::Preserve) {
                    load_jpeg_exif(metadata.source_path)
                } else {
                    None
                },
                if matches!(options.metadata.mode, MetadataMode::Strip) {
                    None
                } else {
                    custom_payload.as_deref()
                },
            );
        }
        ImageFormatHint::Webp => {
            if matches!(options.metadata.mode, MetadataMode::Preserve) {
                if let Some(exif) = load_jpeg_exif(metadata.source_path) {
                    encoded = inject_webp_exif(encoded, Some(exif), custom_payload.as_deref());
                } else {
                    encoded = inject_webp_exif(encoded, None, custom_payload.as_deref());
                }
            } else if matches!(options.metadata.mode, MetadataMode::Strip) {
                // Nothing extra to embed.
            } else {
                encoded = inject_webp_exif(encoded, None, custom_payload.as_deref());
            }
        }
    }

    write_bytes(destination, &encoded)?;
    Ok(())
}

fn determine_format(path: &Path, options: &OutputOptions) -> ImageFormatHint {
    if !options.auto_detect {
        return options.format.unwrap_or_default();
    }

    if let Some(fmt) = path
        .extension()
        .and_then(|e| e.to_str())
        .and_then(ImageFormatHint::from_extension)
    {
        fmt
    } else {
        options.format.unwrap_or_default()
    }
}

fn encode_png(image: &DynamicImage, compression: PngCompression) -> Result<Vec<u8>> {
    let rgba = image.to_rgba8();
    let mut buffer = Vec::new();
    {
        let encoder = PngEncoder::new_with_quality(
            &mut buffer,
            compression.into_image(),
            FilterType::Adaptive,
        );
        encoder
            .write_image(
                rgba.as_raw(),
                rgba.width(),
                rgba.height(),
                ExtendedColorType::Rgba8,
            )
            .context("failed to encode PNG")?;
    }
    Ok(buffer)
}

fn encode_jpeg(image: &DynamicImage, quality: u8) -> Result<Vec<u8>> {
    let rgb = image.to_rgb8();
    let mut buffer = Vec::new();
    {
        let encoder = JpegEncoder::new_with_quality(&mut buffer, quality);
        encoder
            .write_image(
                rgb.as_raw(),
                rgb.width(),
                rgb.height(),
                ExtendedColorType::Rgb8,
            )
            .context("failed to encode JPEG")?;
    }
    Ok(buffer)
}

fn encode_webp(image: &DynamicImage) -> Result<Vec<u8>> {
    let rgba = image.to_rgba8();
    let mut buffer = Vec::new();
    {
        let encoder = WebPEncoder::new_lossless(&mut buffer);
        encoder
            .write_image(
                rgba.as_raw(),
                rgba.width(),
                rgba.height(),
                ExtendedColorType::Rgba8,
            )
            .context("failed to encode WebP")?;
    }
    Ok(buffer)
}

fn load_png_exif_chunks(source: Option<&Path>) -> Vec<Vec<u8>> {
    let Some(path) = source else {
        return Vec::new();
    };
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("png"))
        != Some(true)
    {
        return Vec::new();
    }

    let Ok(bytes) = fs::read(path) else {
        warn!("Failed to read source PNG metadata from {}", path.display());
        return Vec::new();
    };
    if bytes.len() < 8 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" {
        return Vec::new();
    }

    let mut cursor = 8usize;
    let mut chunks = Vec::new();
    while cursor + 8 <= bytes.len() {
        let length = u32::from_be_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        let chunk_type = &bytes[cursor + 4..cursor + 8];
        let data_start = cursor + 8;
        let data_end = data_start.saturating_add(length);
        if data_end + 4 > bytes.len() {
            break;
        }

        if chunk_type == b"eXIf" {
            chunks.push(bytes[cursor..data_end + 4].to_vec());
        }

        cursor = data_end + 4;
        if chunk_type == b"IEND" {
            break;
        }
    }
    chunks
}

fn load_jpeg_exif(source: Option<&Path>) -> Option<Vec<u8>> {
    let path = source?;
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
        != Some(true)
    {
        return None;
    }

    let bytes = fs::read(path).ok()?;
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None;
    }

    let mut index = 2usize;
    while index + 4 < bytes.len() {
        if bytes[index] != 0xFF {
            break;
        }
        let marker = bytes[index + 1];
        index += 2;

        if marker == 0xDA || marker == 0xD9 {
            break;
        }

        if index + 2 > bytes.len() {
            break;
        }

        let length = u16::from_be_bytes([bytes[index], bytes[index + 1]]) as usize;
        let data_start = index + 2;
        let data_end = data_start.saturating_add(length - 2);
        if data_end > bytes.len() {
            break;
        }

        if marker == 0xE1 && length >= 8 && &bytes[data_start..data_start + 4] == b"Exif" {
            let mut segment = Vec::with_capacity(length + 2);
            segment.extend_from_slice(&[0xFF, 0xE1]);
            segment.extend_from_slice(&bytes[index..index + length]);
            return Some(segment);
        }

        index = data_end;
    }
    None
}

fn inject_png_metadata(
    encoded: Vec<u8>,
    exif_chunks: &[Vec<u8>],
    custom_json: Option<&str>,
) -> Vec<u8> {
    if encoded.len() < 8 {
        return encoded;
    }
    if exif_chunks.is_empty() && custom_json.is_none() {
        return encoded;
    }

    let signature = &encoded[..8];
    let cursor = 8usize;
    if cursor + 8 > encoded.len() {
        return encoded;
    }
    let ihdr_length = u32::from_be_bytes(encoded[cursor..cursor + 4].try_into().unwrap()) as usize;
    let ihdr_total = 8 + ihdr_length + 4;
    if cursor + ihdr_total > encoded.len() {
        return encoded;
    }

    let mut output = Vec::with_capacity(
        encoded.len()
            + exif_chunks.iter().map(|c| c.len()).sum::<usize>()
            + custom_json.map(|_| 64).unwrap_or_default(),
    );
    output.extend_from_slice(signature);
    output.extend_from_slice(&encoded[cursor..cursor + ihdr_total]);

    for chunk in exif_chunks {
        output.extend_from_slice(chunk);
    }

    if let Some(chunk) = custom_json.and_then(|json| build_png_text_chunk("IronCropper", json)) {
        output.extend_from_slice(&chunk);
    }

    output.extend_from_slice(&encoded[cursor + ihdr_total..]);
    output
}

fn build_png_text_chunk(keyword: &str, value: &str) -> Option<Vec<u8>> {
    if keyword.is_empty() || keyword.len() > 79 {
        warn!("PNG text keyword '{keyword}' is invalid; skipping");
        return None;
    }
    if !keyword
        .chars()
        .all(|c| c.is_ascii() && c != '\0' && c != '\n' && c != '\r')
    {
        warn!("PNG text keyword '{keyword}' contains unsupported characters");
        return None;
    }

    let mut data = Vec::with_capacity(keyword.len() + value.len() + 1);
    data.extend_from_slice(keyword.as_bytes());
    data.push(0);
    data.extend_from_slice(value.as_bytes());

    let length = data.len() as u32;
    let mut chunk = Vec::with_capacity(12 + data.len());
    chunk.extend_from_slice(&length.to_be_bytes());
    chunk.extend_from_slice(b"tEXt");
    chunk.extend_from_slice(&data);

    let mut hasher = Crc32::new();
    hasher.update(b"tEXt");
    hasher.update(&data);
    chunk.extend_from_slice(&hasher.finalize().to_be_bytes());
    Some(chunk)
}

fn inject_jpeg_metadata(
    encoded: Vec<u8>,
    exif_segment: Option<Vec<u8>>,
    custom_json: Option<&str>,
) -> Vec<u8> {
    if encoded.len() < 2 || encoded[0] != 0xFF || encoded[1] != 0xD8 {
        return encoded;
    }
    if exif_segment.is_none() && custom_json.is_none() {
        return encoded;
    }

    let mut output = Vec::with_capacity(
        encoded.len()
            + exif_segment.as_ref().map(|s| s.len()).unwrap_or(0)
            + custom_json.map(|json| json.len() + 64).unwrap_or(0),
    );
    output.extend_from_slice(&encoded[..2]);

    if let Some(exif) = exif_segment {
        output.extend_from_slice(&exif);
    }
    if let Some(segment) = custom_json.and_then(build_jpeg_xmp_segment) {
        output.extend_from_slice(&segment);
    }

    output.extend_from_slice(&encoded[2..]);
    output
}

fn build_jpeg_xmp_segment(json: &str) -> Option<Vec<u8>> {
    let encoded = BASE64.encode(json.as_bytes());
    let packet = format!(
        r#"<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description xmlns:iron="https://iron-cropper.app/ns/1.0/">
   <iron:Metadata>{encoded}</iron:Metadata>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#
    );

    let header = b"http://ns.adobe.com/xap/1.0/\0";
    let payload = [header.as_ref(), packet.as_bytes()].concat();
    let total_len = payload.len() + 2;
    if total_len > u16::MAX as usize {
        warn!("XMP payload too large; skipping metadata embed");
        return None;
    }

    let mut segment = Vec::with_capacity(total_len + 2);
    segment.extend_from_slice(&[0xFF, 0xE1]);
    segment.extend_from_slice(&(total_len as u16).to_be_bytes());
    segment.extend_from_slice(&payload);
    Some(segment)
}

fn inject_webp_exif(
    encoded: Vec<u8>,
    exif_segment: Option<Vec<u8>>,
    custom_json: Option<&str>,
) -> Vec<u8> {
    if exif_segment.is_none() && custom_json.is_none() {
        return encoded;
    }

    // WebP metadata requires a RIFF container; we simply append EXIF/XMP chunks when possible.
    let output = encoded;
    if exif_segment.as_ref().is_some_and(|exif| !exif.is_empty()) {
        debug!("Preserving EXIF in WebP is not yet implemented; skipping");
    }
    if let Some(json) = custom_json {
        debug!(
            "Custom metadata for WebP is not implemented; skipping payload {} bytes",
            json.len()
        );
    }
    output
}

fn build_custom_metadata_payload(
    settings: &MetadataSettings,
    ctx: &MetadataContext<'_>,
) -> Result<Option<String>> {
    if matches!(settings.mode, MetadataMode::Strip) {
        return Ok(None);
    }

    let mut root = JsonMap::new();
    for (key, value) in &settings.custom_tags {
        root.insert(key.clone(), JsonValue::String(value.clone()));
    }

    if let Some(crop) = ctx.crop_settings.filter(|_| settings.include_crop_settings) {
        root.insert(
            "crop_settings".to_string(),
            JsonValue::Object({
                let mut crop_map = JsonMap::new();
                crop_map.insert("preset".into(), JsonValue::String(crop.preset.clone()));
                crop_map.insert(
                    "output_width".into(),
                    JsonValue::Number(JsonNumber::from(crop.output_width)),
                );
                crop_map.insert(
                    "output_height".into(),
                    JsonValue::Number(JsonNumber::from(crop.output_height)),
                );
                crop_map.insert(
                    "face_height_pct".into(),
                    JsonValue::Number(
                        JsonNumber::from_f64(crop.face_height_pct as f64)
                            .unwrap_or(JsonNumber::from(0)),
                    ),
                );
                crop_map.insert(
                    "positioning_mode".into(),
                    JsonValue::String(crop.positioning_mode.clone()),
                );
                crop_map.insert(
                    "horizontal_offset".into(),
                    JsonValue::Number(
                        JsonNumber::from_f64(crop.horizontal_offset as f64)
                            .unwrap_or(JsonNumber::from(0)),
                    ),
                );
                crop_map.insert(
                    "vertical_offset".into(),
                    JsonValue::Number(
                        JsonNumber::from_f64(crop.vertical_offset as f64)
                            .unwrap_or(JsonNumber::from(0)),
                    ),
                );
                crop_map
            }),
        );
    }

    if settings.include_quality_metrics {
        if let Some(q) = ctx.quality {
            root.insert("quality".into(), serde_json::to_value(q)?);
        }
        if let Some(num) = ctx.quality_score.and_then(JsonNumber::from_f64) {
            root.insert("quality_score".into(), JsonValue::Number(num));
        }
        if let Some(num) = ctx
            .detection_score
            .and_then(|conf| JsonNumber::from_f64(conf as f64))
        {
            root.insert("face_confidence".into(), JsonValue::Number(num));
        }
    }

    if root.is_empty() {
        return Ok(None);
    }

    root.insert("generator".into(), JsonValue::String("iron-cropper".into()));
    root.insert(
        "generator_version".into(),
        JsonValue::String(env!("CARGO_PKG_VERSION").into()),
    );

    Ok(Some(JsonValue::Object(root).to_string()))
}

/// Append a suffix to a filename, preserving the existing extension.
pub fn append_suffix_to_filename(name: &str, suffix: &str) -> String {
    if suffix.is_empty() {
        return name.to_string();
    }
    if let Some(idx) = name.rfind('.') {
        let (base, ext) = name.split_at(idx);
        format!("{base}{suffix}{ext}")
    } else {
        format!("{name}{suffix}")
    }
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(bytes)
        .with_context(|| format!("failed to write {}", path.display()))?;
    writer.flush().ok();
    Ok(())
}
