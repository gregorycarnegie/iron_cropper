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
    candidates
        .into_iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
}

/// Macro to generate preset dimension tests
///
/// This macro creates a test function that:
/// 1. Loads the model and fixture
/// 2. Runs the CLI with the specified preset
/// 3. Verifies the output has the expected dimensions
macro_rules! preset_test {
    ($test_name:ident, $preset:literal, $width:expr, $height:expr, $display_name:literal) => {
        #[test]
        fn $test_name() {
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
                    "--preset",
                    $preset,
                    "--output-dir",
                    output_dir.to_str().unwrap(),
                ])
                .output()
                .expect("execute CLI");

            assert!(
                output.status.success(),
                "CLI should succeed with {} preset",
                $display_name
            );

            // Verify output dimensions
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
                    ($width, $height),
                    "{} preset should be {}x{}",
                    $display_name,
                    $width,
                    $height
                );
            }
        }
    };
}

// Generate test functions for each preset
preset_test!(test_preset_linkedin, "linkedin", 400, 400, "LinkedIn");
preset_test!(test_preset_passport, "passport", 413, 531, "Passport");
preset_test!(test_preset_instagram, "instagram", 1080, 1080, "Instagram");
preset_test!(test_preset_avatar, "avatar", 512, 512, "Avatar");
preset_test!(test_preset_headshot, "headshot", 600, 800, "Headshot");

#[test]
fn test_all_presets_succeed() {
    let model = match find_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: model not found");
            return;
        }
    };

    let presets = vec![
        "linkedin",
        "passport",
        "instagram",
        "idcard",
        "avatar",
        "headshot",
    ];

    for preset in presets {
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
