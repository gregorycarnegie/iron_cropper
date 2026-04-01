/// Integration tests for crop preset sizes
use image::GenericImageView;
use std::fs;

#[macro_use]
mod common;

macro_rules! preset_test {
    ($test_name:ident, $preset:literal, $width:expr, $height:expr, $display_name:literal) => {
        #[test]
        fn $test_name() {
            let (model, _fixture, _temp_dir, input_path, output_dir) =
                cli_test_setup!(load_image);

            let output = run_cli!(
                input_path,
                model,
                output_dir,
                ["--crop", "--preset", $preset]
            );

            assert_cli_success!(output, "CLI should succeed with preset");

            let output_files = verify_output_files!(output_dir, "png");
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

preset_test!(test_preset_linkedin, "linkedin", 400, 400, "LinkedIn");
preset_test!(test_preset_passport, "passport", 413, 531, "Passport");
preset_test!(test_preset_instagram, "instagram", 1080, 1080, "Instagram");
preset_test!(test_preset_avatar, "avatar", 512, 512, "Avatar");
preset_test!(test_preset_headshot, "headshot", 600, 800, "Headshot");

#[test]
fn test_all_presets_succeed() {
    let (model, fixture, _temp_dir, _input_path, _) = cli_test_setup!();

    let presets = vec![
        "linkedin",
        "passport",
        "instagram",
        "idcard",
        "avatar",
        "headshot",
    ];

    for preset in presets {
        let temp_dir = tempfile::TempDir::new().expect("create temp dir");
        let input_path = temp_dir.path().join("input.jpg");
        let output_dir = temp_dir.path().join("output");
        fs::create_dir_all(&output_dir).expect("create output dir");

        image::open(&fixture)
            .expect("load fixture")
            .save(&input_path)
            .expect("save input");

        let output = run_cli!(
            input_path,
            model,
            output_dir,
            ["--crop", "--preset", preset]
        );

        assert!(output.status.success(), "Preset '{}' should succeed", preset);

        let output_files = verify_output_files!(output_dir, "png");
        assert!(!output_files.is_empty(), "Preset '{}' should produce output", preset);
    }
}
