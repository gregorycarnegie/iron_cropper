/// Integration tests for enhancement pipeline
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

#[test]
fn test_enhancement_preset_natural() {
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

    image::open(find_fixture_image().expect("fixture"))
        .expect("load fixture")
        .save(&input_path)
        .expect("save input");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args([
            "--input",
            input_path.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--enhance",
            "true",
            "--enhancement-preset",
            "natural",
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
        "CLI should succeed with natural enhancement preset"
    );

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

    assert!(!output_files.is_empty(), "Should produce enhanced output");
}

#[test]
fn test_enhancement_preset_vivid() {
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

    image::open(find_fixture_image().expect("fixture"))
        .expect("load fixture")
        .save(&input_path)
        .expect("save input");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args([
            "--input",
            input_path.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--enhance",
            "true",
            "--enhancement-preset",
            "vivid",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    assert!(
        output.status.success(),
        "CLI should succeed with vivid enhancement preset"
    );

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

    assert!(!output_files.is_empty(), "Should produce enhanced output");
}

#[test]
fn test_enhancement_preset_professional() {
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

    image::open(find_fixture_image().expect("fixture"))
        .expect("load fixture")
        .save(&input_path)
        .expect("save input");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args([
            "--input",
            input_path.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--enhance",
            "true",
            "--enhancement-preset",
            "professional",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    assert!(
        output.status.success(),
        "CLI should succeed with professional enhancement preset"
    );

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

    assert!(!output_files.is_empty(), "Should produce enhanced output");
}

#[test]
fn test_individual_enhancement_flags() {
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

    image::open(find_fixture_image().expect("fixture"))
        .expect("load fixture")
        .save(&input_path)
        .expect("save input");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args([
            "--input",
            input_path.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--enhance",
            "true",
            "--enhance-exposure",
            "5",
            "--enhance-contrast",
            "10.0",
            "--enhance-sharpness",
            "0.5",
            "--enhance-brightness",
            "10",
            "--enhance-saturation",
            "1.2",
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
        "CLI should succeed with individual enhancement flags"
    );

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

    assert!(!output_files.is_empty(), "Should produce enhanced output");
}

#[test]
fn test_enhancement_with_auto_color() {
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

    image::open(find_fixture_image().expect("fixture"))
        .expect("load fixture")
        .save(&input_path)
        .expect("save input");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args([
            "--input",
            input_path.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--enhance",
            "true",
            "--enhance-auto-color",
            "true",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    assert!(
        output.status.success(),
        "CLI should succeed with auto-color enhancement"
    );

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

    assert!(!output_files.is_empty(), "Should produce enhanced output");
}

#[test]
fn test_preset_override_with_explicit_flags() {
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

    image::open(find_fixture_image().expect("fixture"))
        .expect("load fixture")
        .save(&input_path)
        .expect("save input");

    // Use a preset but override some settings
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args([
            "--input",
            input_path.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--enhance",
            "true",
            "--enhancement-preset",
            "natural",
            "--enhance-sharpness",
            "1.0", // Override the preset's sharpness
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    assert!(
        output.status.success(),
        "CLI should succeed when overriding preset values"
    );

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

    assert!(!output_files.is_empty(), "Should produce enhanced output");
}
