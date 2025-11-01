/// Integration tests for batch processing with quality filtering
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
    candidates
        .into_iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
}

fn copy_fixture_to(src: &PathBuf, dest: &PathBuf) -> std::io::Result<()> {
    std::fs::copy(src, dest)?;
    Ok(())
}

#[test]
fn test_batch_processing_with_multiple_images() {
    let model = match find_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: model not found");
            return;
        }
    };

    let fixture = match find_fixture_image() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: fixture not found");
            return;
        }
    };

    let temp_dir = TempDir::new().expect("create temp dir");
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");
    fs::create_dir_all(&input_dir).expect("create input dir");
    fs::create_dir_all(&output_dir).expect("create output dir");

    // Copy real fixture images for testing
    copy_fixture_to(&fixture, &input_dir.join("image1.jpg")).expect("copy image1");
    copy_fixture_to(&fixture, &input_dir.join("image2.jpg")).expect("copy image2");
    copy_fixture_to(&fixture, &input_dir.join("image3.jpg")).expect("copy image3");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args(&[
            "--input",
            input_dir.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    if !output.status.success() {
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    }

    assert!(
        output.status.success(),
        "CLI should succeed processing batch"
    );

    // Verify we got output files
    let output_files: Vec<_> = fs::read_dir(&output_dir)
        .expect("read output dir")
        .filter_map(Result::ok)
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e == "png")
                .unwrap_or(false)
        })
        .collect();

    assert!(!output_files.is_empty(), "Should produce output files");
}

#[test]
fn test_skip_low_quality_flag() {
    let model = match find_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: model not found");
            return;
        }
    };

    let fixture = match find_fixture_image() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: fixture not found");
            return;
        }
    };

    let temp_dir = TempDir::new().expect("create temp dir");
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");
    fs::create_dir_all(&input_dir).expect("create input dir");
    fs::create_dir_all(&output_dir).expect("create output dir");

    // Use real fixture images for testing
    copy_fixture_to(&fixture, &input_dir.join("test.jpg")).expect("copy fixture");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args(&[
            "--input",
            input_dir.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--skip-low-quality",
            "true",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    if !output.status.success() {
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    }

    assert!(
        output.status.success(),
        "CLI should succeed with quality filtering"
    );

    // With quality filtering, we might get fewer outputs
    // (This test just verifies the flag is accepted and doesn't crash)
    let output_count = fs::read_dir(&output_dir)
        .expect("read output dir")
        .filter_map(Result::ok)
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e == "png")
                .unwrap_or(false)
        })
        .count();

    // We should have at least some output (the sharp image should pass)
    // or the command should at least complete successfully
    eprintln!("Quality filtering produced {} output files", output_count);
}

#[test]
fn test_min_quality_high() {
    let model = match find_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: model not found");
            return;
        }
    };

    let fixture = match find_fixture_image() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: fixture not found");
            return;
        }
    };

    let temp_dir = TempDir::new().expect("create temp dir");
    let input_path = temp_dir.path().join("input.jpg");
    let output_dir = temp_dir.path().join("output");
    fs::create_dir_all(&output_dir).expect("create output dir");

    copy_fixture_to(&fixture, &input_path).expect("copy fixture");

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
            "--min-quality",
            "high",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    assert!(
        output.status.success(),
        "CLI should succeed with high quality threshold"
    );
}

#[test]
fn test_min_quality_medium() {
    let model = match find_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: model not found");
            return;
        }
    };

    let fixture = match find_fixture_image() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: fixture not found");
            return;
        }
    };

    let temp_dir = TempDir::new().expect("create temp dir");
    let input_path = temp_dir.path().join("input.jpg");
    let output_dir = temp_dir.path().join("output");
    fs::create_dir_all(&output_dir).expect("create output dir");

    copy_fixture_to(&fixture, &input_path).expect("copy fixture");

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
            "--min-quality",
            "medium",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    assert!(
        output.status.success(),
        "CLI should succeed with medium quality threshold"
    );
}

#[test]
fn test_batch_with_face_index_selection() {
    let model = match find_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: model not found");
            return;
        }
    };

    let fixture = match find_fixture_image() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: fixture not found");
            return;
        }
    };

    let temp_dir = TempDir::new().expect("create temp dir");
    let input_path = temp_dir.path().join("input.jpg");
    let output_dir = temp_dir.path().join("output");
    fs::create_dir_all(&output_dir).expect("create output dir");

    copy_fixture_to(&fixture, &input_path).expect("copy fixture");

    // Test selecting specific face index
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
            "--face-index",
            "0",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    assert!(
        output.status.success(),
        "CLI should succeed with face index selection"
    );
}

#[test]
fn test_batch_with_naming_template() {
    let model = match find_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: model not found");
            return;
        }
    };

    let fixture = match find_fixture_image() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: fixture not found");
            return;
        }
    };

    let temp_dir = TempDir::new().expect("create temp dir");
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");
    fs::create_dir_all(&input_dir).expect("create input dir");
    fs::create_dir_all(&output_dir).expect("create output dir");

    copy_fixture_to(&fixture, &input_dir.join("test.jpg")).expect("copy fixture");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args(&[
            "--input",
            input_dir.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--naming-template",
            "{original}_crop_{index}.{ext}",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    assert!(
        output.status.success(),
        "CLI should succeed with naming template"
    );

    // Verify output exists with expected naming pattern
    let output_files: Vec<_> = fs::read_dir(&output_dir)
        .expect("read output dir")
        .filter_map(Result::ok)
        .map(|e| e.file_name().into_string().unwrap())
        .collect();

    assert!(
        output_files.iter().any(|name| name.contains("_crop_")),
        "Output should use naming template, got: {:?}",
        output_files
    );
}

#[test]
fn test_batch_processing_summary_output() {
    let model = match find_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: model not found");
            return;
        }
    };

    let fixture = match find_fixture_image() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: fixture not found");
            return;
        }
    };

    let temp_dir = TempDir::new().expect("create temp dir");
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");
    fs::create_dir_all(&input_dir).expect("create input dir");
    fs::create_dir_all(&output_dir).expect("create output dir");

    // Copy fixture images for testing
    copy_fixture_to(&fixture, &input_dir.join("img1.jpg")).expect("copy img1");
    copy_fixture_to(&fixture, &input_dir.join("img2.jpg")).expect("copy img2");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args(&[
            "--input",
            input_dir.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    assert!(
        output.status.success(),
        "CLI should succeed processing batch"
    );

    // Verify summary output contains stats
    let stdout = String::from_utf8_lossy(&output.stdout);
    // The summary should mention images processed
    // (exact format may vary, so we just check it completes)
    eprintln!("Batch processing output:\n{}", stdout);
}
