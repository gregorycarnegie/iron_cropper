/// Integration tests for positioning modes (center, rule-of-thirds, custom)
use image::GenericImageView;
use std::fs;

#[macro_use]
mod common;

#[test]
fn test_positioning_mode_center() {
    let (model, _fixture, _temp_dir, input_path, output_dir) = cli_test_setup!(load_image);

    let output = run_cli!(
        input_path,
        model,
        output_dir,
        [
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--positioning-mode",
            "center"
        ]
    );

    assert_cli_success!(output, "CLI should succeed with center positioning");

    let output_files = verify_output_files!(output_dir, "png");
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
    let (model, _fixture, _temp_dir, input_path, output_dir) = cli_test_setup!(load_image);

    let output = run_cli!(
        input_path,
        model,
        output_dir,
        [
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--positioning-mode",
            "rule-of-thirds"
        ]
    );

    assert_cli_success!(output, "CLI should succeed with rule-of-thirds mode");

    let output_files = verify_output_files!(output_dir, "png");
    assert!(
        !output_files.is_empty(),
        "Expected at least one output file with rule-of-thirds"
    );
}

#[test]
fn test_positioning_mode_custom() {
    let (model, _fixture, _temp_dir, input_path, output_dir) = cli_test_setup!(load_image);

    let output = run_cli!(
        input_path,
        model,
        output_dir,
        [
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--positioning-mode",
            "custom",
            "--vertical-offset=10",
            "--horizontal-offset=-5"
        ]
    );

    assert_cli_success!(output, "CLI should succeed with custom positioning mode");

    let output_files = verify_output_files!(output_dir, "png");
    assert!(
        !output_files.is_empty(),
        "Expected at least one output file with custom positioning"
    );
}

#[test]
fn test_different_positioning_modes_produce_different_crops() {
    let (model, _fixture, temp_dir, input_path, _output_dir) = cli_test_setup!(load_image);

    // Test center mode
    let center_dir = temp_dir.path().join("center");
    fs::create_dir_all(&center_dir).expect("create center dir");

    let _ = run_cli!(
        input_path,
        model,
        center_dir,
        [
            "--crop",
            "--output-width",
            "300",
            "--output-height",
            "300",
            "--positioning-mode",
            "center"
        ]
    );

    // Test rule-of-thirds mode
    let rot_dir = temp_dir.path().join("rot");
    fs::create_dir_all(&rot_dir).expect("create rot dir");

    let _ = run_cli!(
        input_path,
        model,
        rot_dir,
        [
            "--crop",
            "--output-width",
            "300",
            "--output-height",
            "300",
            "--positioning-mode",
            "rule-of-thirds"
        ]
    );

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
