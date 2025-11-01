/// Integration tests for crop preset sizes
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
fn test_preset_linkedin() {
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

    image::open(find_fixture_image().expect("fixture")).expect("load fixture")
        .save(&input_path)
        .expect("save input");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args(&[
            "--input",
            input_path.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--preset",
            "linkedin",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    assert!(
        output.status.success(),
        "CLI should succeed with LinkedIn preset"
    );

    // Verify output dimensions are 400x400
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

    assert!(!output_files.is_empty(), "Should have output files");

    for entry in output_files {
        let img = image::open(entry.path()).expect("open output");
        assert_eq!(
            img.dimensions(),
            (400, 400),
            "LinkedIn preset should be 400x400"
        );
    }
}

#[test]
fn test_preset_passport() {
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

    image::open(find_fixture_image().expect("fixture")).expect("load fixture")
        .save(&input_path)
        .expect("save input");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args(&[
            "--input",
            input_path.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--preset",
            "passport",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    assert!(
        output.status.success(),
        "CLI should succeed with Passport preset"
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

    assert!(!output_files.is_empty(), "Should have output files");

    for entry in output_files {
        let img = image::open(entry.path()).expect("open output");
        assert_eq!(
            img.dimensions(),
            (413, 531),
            "Passport preset should be 413x531"
        );
    }
}

#[test]
fn test_preset_instagram() {
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

    image::open(find_fixture_image().expect("fixture")).expect("load fixture")
        .save(&input_path)
        .expect("save input");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args(&[
            "--input",
            input_path.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--preset",
            "instagram",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    assert!(
        output.status.success(),
        "CLI should succeed with Instagram preset"
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

    assert!(!output_files.is_empty(), "Should have output files");

    for entry in output_files {
        let img = image::open(entry.path()).expect("open output");
        assert_eq!(
            img.dimensions(),
            (1080, 1080),
            "Instagram preset should be 1080x1080"
        );
    }
}

#[test]
fn test_preset_avatar() {
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

    image::open(find_fixture_image().expect("fixture")).expect("load fixture")
        .save(&input_path)
        .expect("save input");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args(&[
            "--input",
            input_path.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--preset",
            "avatar",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    assert!(
        output.status.success(),
        "CLI should succeed with Avatar preset"
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

    assert!(!output_files.is_empty(), "Should have output files");

    for entry in output_files {
        let img = image::open(entry.path()).expect("open output");
        assert_eq!(
            img.dimensions(),
            (512, 512),
            "Avatar preset should be 512x512"
        );
    }
}

#[test]
fn test_preset_headshot() {
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

    image::open(find_fixture_image().expect("fixture")).expect("load fixture")
        .save(&input_path)
        .expect("save input");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
        .args(&[
            "--input",
            input_path.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--crop",
            "--preset",
            "headshot",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("execute CLI");

    assert!(
        output.status.success(),
        "CLI should succeed with Headshot preset"
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

    assert!(!output_files.is_empty(), "Should have output files");

    for entry in output_files {
        let img = image::open(entry.path()).expect("open output");
        assert_eq!(
            img.dimensions(),
            (600, 800),
            "Headshot preset should be 600x800"
        );
    }
}

#[test]
fn test_all_presets_succeed() {
    let model = match find_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: model not found");
            return;
        }
    };

    let presets = vec!["linkedin", "passport", "instagram", "idcard", "avatar", "headshot"];

    for preset in presets {
        let temp_dir = TempDir::new().expect("create temp dir");
        let input_path = temp_dir.path().join("input.png");
        let output_dir = temp_dir.path().join("output");
        fs::create_dir_all(&output_dir).expect("create output dir");

        image::open(find_fixture_image().expect("fixture")).expect("load fixture")
            .save(&input_path)
            .expect("save input");

        let output = std::process::Command::new(env!("CARGO_BIN_EXE_yunet-cli"))
            .args(&[
                "--input",
                input_path.to_str().unwrap(),
                "--model",
                model.to_str().unwrap(),
                "--crop",
                "--preset",
                preset,
                "--output-dir",
                output_dir.to_str().unwrap(),
            ])
            .output()
            .expect("execute CLI");

        assert!(
            output.status.success(),
            "Preset '{}' should succeed",
            preset
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

        assert!(
            !output_files.is_empty(),
            "Preset '{}' should produce output",
            preset
        );
    }
}
