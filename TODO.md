# Project TODO List: YuNet Face Detection

This document outlines the development plan for the YuNet face detection project, broken down into phases.

## Phase 0: Workspace Bootstrap

- [x] **Workspace layout**: Create `yunet-core`, `yunet-gui`, `yunet-cli`, and `yunet-utils` crates and wire them through the root `Cargo.toml`.
- [x] **Dependencies**: Add baseline crates (`tract-onnx`, `image`, `ndarray`, `rayon`, `egui`, `eframe`, `clap`, `serde`, `serde_json`, `log`) with pinned versions.
- [x] **Models**: Download `face_detection_yunet_2023mar.onnx`, place it under `models/`, and document version plus SHA256 in `models/README.md`.
- [x] **Scaffolding**: Stub out crate-level skeletons (`lib.rs`, `main.rs`, module folders) and ensure `cargo check --workspace` passes.
- [x] **Configs**: Add `.cargo/config.toml` (build target defaults) and extend `.gitignore` for model files, fixtures, and GUI caches.

## Phase 1: Core Logic & Utilities

- [ ] **`yunet-core`**: Implement core face detection logic.
    - [x] Pre-processing of input images.
    - [x] ONNX model inference.
    - [x] Post-processing (NMS, score thresholding).
    - [x] Validate output parsing against OpenCV YuNet sample outputs.
- [ ] **`yunet-utils`**: Develop shared utilities.
    - [x] Fixture loading for tests.
    - [x] Image and data conversion helpers.
    - [x] Shared configuration loader (`serde` structs for thresholds, input size).
- [ ] **Testing**: Add unit tests for `yunet-core`.
    - [x] Test pre-processing logic.
    - [x] Test post-processing logic against known outputs.
    - [x] Add golden-dataset regression test comparing to OpenCV baselines.
    - [x] Capture OpenCV YuNet reference outputs (bboxes, landmarks, scores) for parity checks.
- [x] **Models**: Document initial ONNX model version and checksum in `models/README.md`.

## Phase 2: Command-Line Interface (CLI)

- [ ] **`yunet-cli`**: Build the command-line application.
    - [x] Implement the `detect` command.
    - [x] Add arguments for input path, score threshold, NMS threshold, etc.
    - [ ] Handle output features.
        - [x] JSON export / stdout summary.
        - [x] Annotated image rendering.
    - [x] Support batch directory processing and glob patterns.
- [ ] **Testing**: Create integration tests for the CLI.
    - [x] Basic detection flow producing JSON output (skips if model unavailable).
    - [x] Test command with various images from `fixtures/`.
        - [x] Add parity fixtures with expected detections and compare bounding boxes/landmarks to JSON snapshots.
    - [x] Verify output against golden data.
    - [x] Add snapshot tests for JSON output shape and ordering.

## Phase 3: Graphical User Interface (GUI)

- [ ] **`yunet-gui`**: Build the desktop application with `egui`/`eframe`.
    - [ ] Implement file selection for input images.
    - [ ] Create UI controls for model parameters (thresholds, etc.).
    - [ ] Display the output image with bounding boxes overlaid.
    - [ ] Offload heavy inference tasks to a background thread (`rayon`) to keep the UI responsive.
    - [ ] Persist user preferences (last model, thresholds) between sessions.
- [ ] **Styling**: Centralize `egui` styling in `yunet-gui/src/theme.rs`.
- [ ] **Testing**: Add smoke tests for the GUI application.
    - [ ] Test application startup and basic interaction.
    - [ ] Test settings serialization.
    - [ ] Add screenshot regression harness for key panels (optional).

## Phase 4: Integration & Refinement

- [ ] **Workspace**: Ensure all crates work together seamlessly.
- [ ] **Configuration**: Share threshold/input-size config structs between CLI and GUI.
- [ ] **Documentation**: Add `///` doc comments to all public functions and modules.
- [ ] **Linting & Formatting**: Run `cargo clippy` and `cargo fmt` across the workspace to ensure code quality.
- [ ] **Performance**: Profile and optimize the inference pipeline in `yunet-core`.
- [ ] **Telemetry**: Add optional timing/logging hooks in `yunet-utils` for inference tracing.
- [ ] **Parity validation**: Compare detections against OpenCV YuNet on a shared dataset.
    - [ ] Evaluate IoU overlap, detection recall/precision, and score deltas.
    - [ ] Document discrepancies and adjust preprocessing/postprocessing until parity is within tolerance.

## Phase 5: Finalization

- [ ] **CI/CD**: Set up a GitHub Actions workflow to automate checks, tests, and lints.
- [ ] **Packaging**: Produce cross-platform binaries/installers (Windows MSI, macOS `.app`, Linux AppImage).
- [ ] **Release**: Tag `v1.0.0` and create a release on GitHub.
- [ ] **Review**: Review and update all documentation (`README.md`, `AGENTS.md`, `GEMINI.md`).
