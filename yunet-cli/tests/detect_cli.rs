use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use assert_cmd::cargo::cargo_bin_cmd;
use image::{ImageBuffer, Rgb};
use serde::Deserialize;
use tempfile::tempdir;
use yunet_utils::{load_fixture_json, normalize_path};

const MODEL_REL_PATH: &str = "models/face_detection_yunet_2023mar.onnx";

#[test]
fn detect_single_image_produces_json_output() -> Result<(), Box<dyn Error>> {
    let Some(model) = ensure_model_path() else {
        return Ok(());
    };

    let work_dir = tempdir()?;
    let image_path = work_dir.path().join("sample.png");
    let json_path = work_dir.path().join("out.json");

    // Generate a simple RGB test image.
    let img = ImageBuffer::from_fn(32, 32, |x, y| {
        let r = ((x + y) % 255) as u8;
        Rgb([r, 128, 255u8.saturating_sub(r)])
    });
    img.save(&image_path)?;

    let empty: Vec<String> = Vec::new();
    let detections = run_cli_detection(&image_path, &json_path, &model, &empty)?;
    assert_eq!(detections.len(), 1, "expected exactly one CLI output entry");
    assert_eq!(
        detections[0].image,
        image_path.display().to_string(),
        "CLI should echo the image path"
    );

    Ok(())
}

#[test]
fn detect_annotate_matches_fixture_when_no_detections() -> Result<(), Box<dyn Error>> {
    let Some(model) = ensure_model_path() else {
        return Ok(());
    };
    let fixture_image = Path::new("fixtures/images/test_pattern.png");
    assert!(
        fixture_image.exists(),
        "fixture image not found at {}",
        fixture_image.display()
    );

    let work_dir = tempdir()?;
    let input_path = work_dir.path().join("pattern.png");
    fs::copy(fixture_image, &input_path)?;
    let json_path = work_dir.path().join("out.json");
    let annotate_dir = work_dir.path().join("annotated");

    let extra = vec![
        "--annotate".to_string(),
        annotate_dir.to_str().unwrap().to_string(),
        "--score-threshold".to_string(),
        "2.0".to_string(),
    ];
    let detections = run_cli_detection(&input_path, &json_path, &model, &extra)?;
    assert_eq!(detections.len(), 1);
    assert!(detections[0].detections.is_empty());

    let annotated_path = annotate_dir.join("pattern.png");
    assert!(
        annotated_path.exists(),
        "annotated image missing at {}",
        annotated_path.display()
    );

    let original = image::open(fixture_image)?.into_rgba8();
    let annotated = image::open(&annotated_path)?.into_rgba8();
    assert_eq!(annotated.dimensions(), original.dimensions());
    assert_eq!(annotated.as_raw(), original.as_raw());

    let expected: FixtureFile = load_fixture_json("opencv/pattern_no_faces.json")?;
    assert!(
        expected.detections.is_empty(),
        "fixture should have no detections"
    );

    Ok(())
}

#[test]
fn cli_detections_match_opencv_parity_samples() -> Result<(), Box<dyn Error>> {
    let Some(model) = ensure_model_path() else {
        return Ok(());
    };
    let cases = [
        ("fixtures/images/006.jpg", "fixtures/opencv/006.json"),
        ("fixtures/images/190_g.jpg", "fixtures/opencv/190_g.json"),
        ("fixtures/images/002_n.jpg", "fixtures/opencv/002_n.json"),
        ("fixtures/images/168_o.jpg", "fixtures/opencv/168_o.json"),
        ("fixtures/images/169_o.jpg", "fixtures/opencv/169_o.json"),
        ("fixtures/images/250_o.jpg", "fixtures/opencv/250_o.json"),
        ("fixtures/images/253_o.jpg", "fixtures/opencv/253_o.json"),
        ("fixtures/images/255_o.jpg", "fixtures/opencv/255_o.json"),
        ("fixtures/images/258_o.webp", "fixtures/opencv/258_o.json"),
    ];

    for (image_rel, fixture_rel) in cases {
        let image_path = Path::new(image_rel);
        assert!(
            image_path.exists(),
            "missing image fixture {}",
            image_path.display()
        );
        let fixture: FixtureFile = load_fixture_json(fixture_rel)?;
        let work_dir = tempdir()?;
        let json_path = work_dir.path().join("out.json");
        let mut extra = Vec::new();
        if let Some(score) = fixture.score_threshold {
            if (score - 0.9).abs() > f64::EPSILON {
                extra.push("--score-threshold".to_string());
                extra.push(score.to_string());
            }
        }
        if let Some(nms) = fixture.nms_threshold {
            if (nms - 0.3).abs() > f64::EPSILON {
                extra.push("--nms-threshold".to_string());
                extra.push(nms.to_string());
            }
        }
        if let Some(top_k) = fixture.top_k {
            if top_k != 5000 {
                extra.push("--top-k".to_string());
                extra.push(top_k.to_string());
            }
        }
        let detections = run_cli_detection(image_path, &json_path, &model, &extra)?;
        assert_eq!(
            detections.len(),
            1,
            "expected single CLI output entry for {}",
            image_rel
        );

        let actual_list = &detections[0].detections;
        assert_detections_close(actual_list, &fixture.detections, 1e-2);
    }

    Ok(())
}

fn ensure_model_path() -> Option<PathBuf> {
    let path = Path::new(MODEL_REL_PATH);
    if !path.exists() {
        eprintln!(
            "skipping test because YuNet model is missing at {}",
            path.display()
        );
        return None;
    }
    Some(normalize_path(path).expect("normalize_path should succeed"))
}

fn run_cli_detection(
    image_path: &Path,
    json_path: &Path,
    model_path: &Path,
    extra_args: &[String],
) -> Result<Vec<CliDetectionRecord>, Box<dyn Error>> {
    let mut cmd = cargo_bin_cmd!("yunet-cli");
    cmd.arg("--input")
        .arg(image_path)
        .arg("--model")
        .arg(model_path)
        .arg("--json")
        .arg(json_path);
    for arg in extra_args {
        cmd.arg(arg);
    }

    cmd.assert().success();
    let payload = fs::read_to_string(json_path)?;
    let parsed: Vec<CliDetectionRecord> = serde_json::from_str(&payload)?;
    Ok(parsed)
}

fn assert_detections_close(actual: &[Detection], expected: &[Detection], tol: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "detection count mismatch (actual={}, expected={})",
        actual.len(),
        expected.len()
    );

    let mut actual_sorted = actual.to_vec();
    actual_sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut expected_sorted = expected.to_vec();
    expected_sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    for (a, e) in actual_sorted.iter().zip(expected_sorted.iter()) {
        assert!(
            (a.score - e.score).abs() <= tol,
            "score mismatch: actual={}, expected={}",
            a.score,
            e.score
        );
        for (idx, (av, ev)) in a.bbox.iter().zip(e.bbox.iter()).enumerate() {
            assert!(
                (av - ev).abs() <= tol,
                "bbox component {} mismatch: actual={}, expected={}",
                idx,
                av,
                ev
            );
        }
        for (landmark_idx, (al, el)) in a.landmarks.iter().zip(e.landmarks.iter()).enumerate() {
            for (coord_idx, (av, ev)) in al.iter().zip(el.iter()).enumerate() {
                assert!(
                    (av - ev).abs() <= tol,
                    "landmark {} coord {} mismatch: actual={}, expected={}",
                    landmark_idx,
                    coord_idx,
                    av,
                    ev
                );
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct CliDetectionRecord {
    image: String,
    detections: Vec<Detection>,
}

#[derive(Clone, Debug, Deserialize)]
struct Detection {
    score: f64,
    bbox: [f64; 4],
    landmarks: [[f64; 2]; 5],
}

#[derive(Debug, Deserialize)]
struct FixtureFile {
    #[serde(default)]
    score_threshold: Option<f64>,
    #[serde(default)]
    nms_threshold: Option<f64>,
    #[serde(default)]
    top_k: Option<usize>,
    detections: Vec<Detection>,
}
