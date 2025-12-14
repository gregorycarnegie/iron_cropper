use std::path::PathBuf;
use yunet_gui::types::MappingUiState;
use yunet_utils::mapping::MappingFormat;

#[test]
fn test_mapping_state_initialization() {
    let state = MappingUiState::new();
    assert!(state.file_path.is_none());
    assert_eq!(state.delimiter_input, ",");
    assert!(state.entries.is_empty());
}

#[test]
fn test_file_upload_detection() {
    let mut state = MappingUiState::new();

    // Simulate setting a CSV file
    let path = PathBuf::from("data/input.csv");
    state.set_file(path.clone());

    assert_eq!(state.file_path, Some(path));
    // detect_format relies on extension, so it should work even if file doesn't exist for this unit test logic
    // (unless set_file calls inspect_mapping_sources which reads the file)
    // Looking at set_file implementation: It calls detects_format_utils (extension based)
    // AND calls refresh_catalog which calls inspect_mapping_sources.
    // inspect_mapping_sources probably fails if file missing.
    // So we might only check format detection if refresh_catalog failure doesn't panic.

    assert_eq!(state.detected_format, Some(MappingFormat::Csv));
    assert_eq!(state.effective_format(), Some(MappingFormat::Csv));
}

#[test]
fn test_column_selection_logic() {
    let mut state = MappingUiState::new();

    assert!(state.source_selector().is_none());

    state.source_column_idx = Some(0);
    assert!(state.source_selector().is_some());

    state.output_column_idx = Some(2);
    assert!(state.output_selector().is_some());
}
