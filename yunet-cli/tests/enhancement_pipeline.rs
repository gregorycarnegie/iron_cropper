/// Integration tests for enhancement pipeline
use std::fs;

#[macro_use]
mod common;

#[test]
fn test_enhancement_preset_natural() {
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
            "--enhance",
            "true",
            "--enhancement-preset",
            "natural"
        ]
    );

    assert_cli_success!(output, "CLI should succeed with natural enhancement preset");
    let output_files = verify_output_files!(output_dir, "png");
    assert!(!output_files.is_empty(), "Should produce enhanced output");
}

#[test]
fn test_enhancement_preset_vivid() {
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
            "--enhance",
            "true",
            "--enhancement-preset",
            "vivid"
        ]
    );

    assert_cli_success!(output, "CLI should succeed with vivid enhancement preset");
    let output_files = verify_output_files!(output_dir, "png");
    assert!(!output_files.is_empty(), "Should produce enhanced output");
}

#[test]
fn test_enhancement_preset_professional() {
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
            "--enhance",
            "true",
            "--enhancement-preset",
            "professional"
        ]
    );

    assert_cli_success!(
        output,
        "CLI should succeed with professional enhancement preset"
    );
    let output_files = verify_output_files!(output_dir, "png");
    assert!(!output_files.is_empty(), "Should produce enhanced output");
}

#[test]
fn test_individual_enhancement_flags() {
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
            "1.2"
        ]
    );

    assert_cli_success!(
        output,
        "CLI should succeed with individual enhancement flags"
    );
    let output_files = verify_output_files!(output_dir, "png");
    assert!(!output_files.is_empty(), "Should produce enhanced output");
}

#[test]
fn test_enhancement_with_auto_color() {
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
            "--enhance",
            "true",
            "--enhance-auto-color",
            "true"
        ]
    );

    assert_cli_success!(output, "CLI should succeed with auto-color enhancement");
    let output_files = verify_output_files!(output_dir, "png");
    assert!(!output_files.is_empty(), "Should produce enhanced output");
}

#[test]
fn test_preset_override_with_explicit_flags() {
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
            "--enhance",
            "true",
            "--enhancement-preset",
            "natural",
            "--enhance-sharpness",
            "1.0"
        ]
    );

    assert_cli_success!(output, "CLI should succeed when overriding preset values");
    let output_files = verify_output_files!(output_dir, "png");
    assert!(!output_files.is_empty(), "Should produce enhanced output");
}
