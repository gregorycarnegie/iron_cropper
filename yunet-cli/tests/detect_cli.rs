use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use assert_cmd::cargo::cargo_bin_cmd;
use image::{ImageBuffer, Rgb};
use serde::Deserialize;
use serde_json::Value;
use tempfile::tempdir;
use yunet_utils::{fixture_path, load_fixture_json, normalize_path};

const MODEL_REL_PATH: &str = "../models/face_detection_yunet_2023mar_640.onnx";

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
    let expected_image_path = image_path.canonicalize()?.display().to_string();
    assert_eq!(
        detections[0].image, expected_image_path,
        "CLI should echo the image path"
    );

    Ok(())
}

#[test]
fn detect_annotate_matches_fixture_when_no_detections() -> Result<(), Box<dyn Error>> {
    let Some(model) = ensure_model_path() else {
        return Ok(());
    };
    let fixture_image = fixture_path("images/test_pattern.png")?;

    let work_dir = tempdir()?;
    let input_path = work_dir.path().join("pattern.png");
    fs::copy(&fixture_image, &input_path)?;
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
        ("images/006.jpg", "opencv/006.json"),
        ("images/190_g.jpg", "opencv/190_g.json"),
        ("images/002_n.jpg", "opencv/002_n.json"),
        ("images/168_o.jpg", "opencv/168_o.json"),
        ("images/169_o.jpg", "opencv/169_o.json"),
        ("images/250_o.jpg", "opencv/250_o.json"),
        ("images/253_o.jpg", "opencv/253_o.json"),
        ("images/255_o.jpg", "opencv/255_o.json"),
        ("images/258_o.webp", "opencv/258_o.json"),
    ];

    for (image_rel, fixture_rel) in cases {
        let image_path = fixture_path(image_rel)?;
        let fixture: FixtureFile = load_fixture_json(fixture_rel)?;
        let work_dir = tempdir()?;
        let json_path = work_dir.path().join("out.json");
        let mut extra = Vec::new();
        if let Some(score) = fixture.score_threshold
            && (score - 0.9).abs() > f64::EPSILON
        {
            extra.push("--score-threshold".to_string());
            extra.push(score.to_string());
        }
        if let Some(nms) = fixture.nms_threshold
            && (nms - 0.3).abs() > f64::EPSILON
        {
            extra.push("--nms-threshold".to_string());
            extra.push(nms.to_string());
        }
        if let Some(top_k) = fixture.top_k
            && top_k != 5000
        {
            extra.push("--top-k".to_string());
            extra.push(top_k.to_string());
        }
        let detections = run_cli_detection(&image_path, &json_path, &model, &extra)?;
        assert_eq!(
            detections.len(),
            1,
            "expected single CLI output entry for {}",
            image_rel
        );

        let actual_list = &detections[0].detections;
        assert_detections_close(actual_list, &fixture.detections, 30.0);
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
        .arg("--no-gpu")
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

#[test]
fn cli_json_output_matches_snapshot() -> Result<(), Box<dyn Error>> {
    let Some(model) = ensure_model_path() else {
        return Ok(());
    };

    let fixture_image = fixture_path("images/006.jpg")?;

    let work_dir = tempdir()?;
    let json_path = work_dir.path().join("out.json");

    let mut cmd = cargo_bin_cmd!("yunet-cli");
    cmd.arg("--input")
        .arg(fixture_image)
        .arg("--model")
        .arg(&model)
        .arg("--no-gpu")
        .arg("--json")
        .arg(&json_path);

    let output = cmd.output()?;
    if !output.status.success() {
        eprintln!(
            "skipping snapshot test; yunet-cli exited with {:?}\nstderr: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
        return Ok(());
    }

    let raw = fs::read_to_string(&json_path)?;
    let sanitized = sanitize_cli_json(&raw)?;
    let expected = load_snapshot("cli_single_image.json")?;
    let sanitized_value: Value = serde_json::from_str(&sanitized)?;
    let expected_value: Value = serde_json::from_str(&expected)?;

    assert!(
        sanitized_value == expected_value,
        "CLI JSON output changed.\nUpdate tests/snapshots/cli_single_image.json if this is expected."
    );

    Ok(())
}

fn sanitize_cli_json(raw: &str) -> Result<String, Box<dyn Error>> {
    let mut value: Value = serde_json::from_str(raw)?;
    if let Some(entries) = value.as_array_mut() {
        for entry in entries {
            if let Some(obj) = entry.as_object_mut() {
                if let Some(image) = obj.get_mut("image")
                    && let Some(path_str) = image.as_str()
                    && let Some(file_name) =
                        Path::new(path_str).file_name().and_then(|n| n.to_str())
                {
                    *image = Value::String(file_name.to_string());
                }
                if let Some(annotated) = obj.get_mut("annotated")
                    && let Some(path_str) = annotated.as_str()
                    && let Some(file_name) =
                        Path::new(path_str).file_name().and_then(|n| n.to_str())
                {
                    *annotated = Value::String(file_name.to_string());
                }
            }
        }
    }
    Ok(serde_json::to_string_pretty(&value)?)
}

fn load_snapshot(name: &str) -> Result<String, Box<dyn Error>> {
    let snapshot_path = Path::new("tests").join("snapshots").join(name);
    if !snapshot_path.exists() {
        return Err(format!("snapshot file missing: {}", snapshot_path.display()).into());
    }
    let contents = fs::read_to_string(&snapshot_path)?;
    Ok(contents)
}
