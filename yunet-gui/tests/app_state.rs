use yunet_gui::types::YuNetApp;
use yunet_utils::config::CropSettings;

#[test]
fn test_undo_redo_crop_settings() {
    let mut app = YuNetApp::test_instance();

    // Initial state
    let initial_preset = app.settings.crop.preset.clone();
    assert_eq!(initial_preset, "linkedin");
    assert_eq!(app.crop_history.len(), 1);
    assert_eq!(app.crop_history_index, 0);

    // Make a change
    app.set_crop_preset("passport");
    assert_eq!(app.settings.crop.preset, "passport");
    assert_eq!(app.crop_history.len(), 2);
    assert_eq!(app.crop_history_index, 1);

    // Make another change (manual) using helper
    app.adjust_face_height(10.0);
    assert!(app.settings.crop.face_height_pct > 70.0);
    assert_eq!(app.crop_history.len(), 3);
    assert_eq!(app.crop_history_index, 2);

    // Undo 1 (Back to passport)
    app.undo_crop_settings();
    assert_eq!(app.settings.crop.preset, "passport");
    assert_eq!(app.crop_history_index, 1);

    // Undo 2 (Back to initial)
    app.undo_crop_settings();
    assert_eq!(app.settings.crop.preset, "linkedin");
    assert_eq!(app.crop_history_index, 0);

    // Undo 3 (Should do nothing)
    app.undo_crop_settings();
    assert_eq!(app.settings.crop.preset, "linkedin");
    assert_eq!(app.crop_history_index, 0);

    // Redo 1 (Forward to passport)
    app.redo_crop_settings();
    assert_eq!(app.settings.crop.preset, "passport");
    assert_eq!(app.crop_history_index, 1);
}

#[test]
fn test_crop_preset_updates_dimensions() {
    let mut app = YuNetApp::test_instance();

    // Set to LinkedIn (400x400)
    app.set_crop_preset("linkedin");
    assert_eq!(app.settings.crop.output_width, 400);
    assert_eq!(app.settings.crop.output_height, 400);

    // Set to Instagram (1080x1080)
    app.set_crop_preset("instagram");
    assert_eq!(app.settings.crop.output_width, 1080);
    assert_eq!(app.settings.crop.output_height, 1080);

    // Custom shouldn't change dims automatically if logic holds,
    // but the setter might not reset them.
    // Let's verifying custom behavior.
    app.set_crop_preset("custom");
    // Dims should stay as last set (1080)
    assert_eq!(app.settings.crop.output_width, 1080);
}

#[test]
fn test_model_path_update() {
    let mut app = YuNetApp::test_instance();

    // Set valid path
    app.model_path_input = "models/my_model.onnx".to_string();
    app.apply_model_path_input();
    assert_eq!(
        app.settings.model_path,
        Some("models/my_model.onnx".to_string())
    );

    // Clear input (should fallback)
    app.model_path_input = "".to_string();
    app.apply_model_path_input();
    // Default fallback logic in update_model_path restores default path if available
    assert!(app.settings.model_path.is_some());
    assert!(
        app.settings
            .model_path
            .as_ref()
            .unwrap()
            .contains("face_detection_yunet")
    );
}
