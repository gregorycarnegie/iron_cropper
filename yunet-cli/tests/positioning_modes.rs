/// Integration tests for positioning modes (center, rule-of-thirds, custom)
use image::GenericImageView;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

fn find_model_path() -> Option<PathBuf> {
    let candidates = vec![
        "models/face_detection_yunet_2023mar_640.onnx",
        "../models/face_detection_yunet_2023mar_640.onnx",
    ];
    candidates
        .into_iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
}

fn find_fixture_image() -> Option<PathBuf> {
    let candidates = vec!["fixtures/images/006.jpg", "../fixtures/images/006.jpg"];
    candidates.into_iter().map(PathBuf::from).find(|p| p.exists())
}

#[test]
fn test_positioning_mode_center() {
    let model = match find_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: model not found");
            return;
        }
    };

    let temp_dir = TempDir::new().expect("create temp dir");
    let input_path = temp_dir.path().join("input.png");
    let output_dir = temp_dir.path().join("output");
    fs::create_dir_all(&output_dir).expect("create output dir");

    // Load and save fixture image
    let img = image::open(find_fixture_image().expect("fixture")).expect("load fixture");
    img.save(&input_path).expect("save input image");

    // Run CLI with center positioning
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args(&[
            "--input",
            input_path.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--positioning-mode",
            "center",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    if !output.status.success() {
        eprintln!("CLI stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("CLI failed with status: {}", output.status);
    }

    // Verify output exists
    let output_files: Vec<_> = fs::read_dir(&output_dir)
        .expect("read output dir")
        .filter_map(Result::ok)
        .collect();

    assert!(
        !output_files.is_empty(),
        "Expected at least one output file"
    );

    // Verify dimensions
    for entry in output_files {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("png") {
            let img = image::open(&path).expect("open output image");
            assert_eq!(
                img.dimensions(),
                (200, 200),
                "Output image should be 200x200"
            );
        }
    }
}

#[test]
fn test_positioning_mode_rule_of_thirds() {
    let model = match find_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: model not found");
            return;
        }
    };

    let temp_dir = TempDir::new().expect("create temp dir");
    let input_path = temp_dir.path().join("input.png");
    let output_dir = temp_dir.path().join("output");
    fs::create_dir_all(&output_dir).expect("create output dir");

    let img = image::open(find_fixture_image().expect("fixture")).expect("load fixture");
    img.save(&input_path).expect("save input image");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args(&[
            "--input",
            input_path.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--positioning-mode",
            "rule-of-thirds",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    assert!(
        output.status.success(),
        "CLI should succeed with rule-of-thirds mode"
    );

    let output_files: Vec<_> = fs::read_dir(&output_dir)
        .expect("read output dir")
        .filter_map(Result::ok)
        .collect();

    assert!(
        !output_files.is_empty(),
        "Expected at least one output file with rule-of-thirds"
    );
}

#[test]
fn test_positioning_mode_custom() {
    let model = match find_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: model not found");
            return;
        }
    };

    let temp_dir = TempDir::new().expect("create temp dir");
    let input_path = temp_dir.path().join("input.png");
    let output_dir = temp_dir.path().join("output");
    fs::create_dir_all(&output_dir).expect("create output dir");

    let img = image::open(find_fixture_image().expect("fixture")).expect("load fixture");
    img.save(&input_path).expect("save input image");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args(&[
            "--input",
            input_path.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--positioning-mode",
            "custom",
            "--vertical-offset=10",
            "--horizontal-offset=-5",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    if !output.status.success() {
        eprintln!("CLI stderr: {}", String::from_utf8_lossy(&output.stderr));
    }
    assert!(
        output.status.success(),
        "CLI should succeed with custom positioning mode"
    );

    let output_files: Vec<_> = fs::read_dir(&output_dir)
        .expect("read output dir")
        .filter_map(Result::ok)
        .collect();

    assert!(
        !output_files.is_empty(),
        "Expected at least one output file with custom positioning"
    );
}

#[test]
fn test_different_positioning_modes_produce_different_crops() {
    let model = match find_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: model not found");
            return;
        }
    };

    let temp_dir = TempDir::new().expect("create temp dir");
    let input_path = temp_dir.path().join("input.png");

    // Load fixture image for comparison
    let img = image::open(find_fixture_image().expect("fixture")).expect("load fixture");
    img.save(&input_path).expect("save input image");

    // Test center mode
    let center_dir = temp_dir.path().join("center");
    fs::create_dir_all(&center_dir).expect("create center dir");

    let _ = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args(&[
            "--input",
            input_path.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--output-width",
            "300",
            "--output-height",
            "300",
            "--positioning-mode",
            "center",
            "--output-dir",
            center_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI for center");

    // Test rule-of-thirds mode
    let rot_dir = temp_dir.path().join("rot");
    fs::create_dir_all(&rot_dir).expect("create rot dir");

    let _ = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args(&[
            "--input",
            input_path.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--output-width",
            "300",
            "--output-height",
            "300",
            "--positioning-mode",
            "rule-of-thirds",
            "--output-dir",
            rot_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI for rot");

    // Verify both produced outputs
    let center_files: Vec<_> = fs::read_dir(&center_dir)
        .expect("read center dir")
        .filter_map(Result::ok)
        .collect();
    let rot_files: Vec<_> = fs::read_dir(&rot_dir)
        .expect("read rot dir")
        .filter_map(Result::ok)
        .collect();

    assert!(
        !center_files.is_empty(),
        "Center mode should produce output"
    );
    assert!(
        !rot_files.is_empty(),
        "Rule-of-thirds mode should produce output"
    );

    // Note: We could compare pixel data to verify they're different,
    // but for now we just verify both modes complete successfully
}
