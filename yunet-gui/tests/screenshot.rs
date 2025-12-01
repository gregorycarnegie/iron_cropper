use anyhow::{Context, Result, bail};
use image::{ColorType, ImageEncoder, Rgba, RgbaImage, codecs::png::PngEncoder};
use imageproc::{
    drawing::{draw_filled_circle_mut, draw_hollow_rect_mut},
    rect::Rect as ProcRect,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};
use yunet_utils::{fixture_path, load_fixture_json};

const SNAPSHOT_FILE: &str = "tests/snapshots/gui_preview_045.sha256";
const BBOX_COLOR: Rgba<u8> = Rgba([255, 145, 77, 255]);
const LANDMARK_COLOR: Rgba<u8> = Rgba([82, 180, 255, 255]);
const LANDMARK_RADIUS: i32 = 3;
const BBOX_THICKNESS: i32 = 2;

#[test]
fn preview_overlay_matches_snapshot() -> Result<()> {
    let image_path = fixture_path("images/045.jpg")?;
    let fixture: FixtureFile = load_fixture_json("opencv/045.json")?;

    let mut canvas = image::open(&image_path)
        .with_context(|| format!("failed to open {}", image_path.display()))?
        .into_rgba8();

    draw_overlay(&mut canvas, &fixture.detections);

    let png_bytes = encode_png(&canvas)?;
    let actual_hash = sha256_hex(&png_bytes);

    let snapshot_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(SNAPSHOT_FILE);

    if !snapshot_path.exists() {
        bail!(
            "Snapshot missing at {}.\nNew sha256 digest:\n{}",
            snapshot_path.display(),
            actual_hash
        );
    }

    let expected_hash = fs::read_to_string(&snapshot_path)
        .with_context(|| format!("failed to read {}", snapshot_path.display()))?
        .trim()
        .to_owned();

    if actual_hash != expected_hash {
        bail!(
            "Screenshot snapshot mismatch.\nUpdate {} with:\n{}",
            snapshot_path.display(),
            actual_hash
        );
    }

    Ok(())
}

fn draw_overlay(canvas: &mut RgbaImage, detections: &[FixtureDetection]) {
    for det in detections {
        let x = det.bbox[0].round() as i32;
        let y = det.bbox[1].round() as i32;
        let width = det.bbox[2].max(1.0).round() as u32;
        let height = det.bbox[3].max(1.0).round() as u32;

        let rect = ProcRect::at(x, y).of_size(width, height);
        draw_hollow_rect_mut(canvas, rect, BBOX_COLOR);

        if BBOX_THICKNESS > 1 {
            let expanded = ProcRect::at(x - 1, y - 1)
                .of_size(width.saturating_add(2), height.saturating_add(2));
            draw_hollow_rect_mut(canvas, expanded, BBOX_COLOR);
        }

        for landmark in &det.landmarks {
            let cx = landmark[0].round() as i32;
            let cy = landmark[1].round() as i32;
            draw_filled_circle_mut(canvas, (cx, cy), LANDMARK_RADIUS, LANDMARK_COLOR);
        }
    }
}

fn encode_png(image: &RgbaImage) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    {
        let encoder = PngEncoder::new(&mut buffer);
        encoder.write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            ColorType::Rgba8.into(),
        )?;
    }
    Ok(buffer)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{:02x}", b)).collect()
}

#[derive(Debug, Deserialize)]
struct FixtureFile {
    detections: Vec<FixtureDetection>,
}

#[derive(Debug, Deserialize)]
struct FixtureDetection {
    bbox: [f64; 4],
    landmarks: [[f64; 2]; 5],
}
