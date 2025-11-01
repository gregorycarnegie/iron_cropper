# Iron Cropper

## Project TODO List: YuNet Face Detection

This document outlines the development plan for the YuNet face detection project, broken down into phases.

### Phase 0: Workspace Bootstrap

- [x] **Workspace layout**: Create `yunet-core`, `yunet-gui`, `yunet-cli`, and `yunet-utils` crates and wire them through the root `Cargo.toml`.
- [x] **Dependencies**: Add baseline crates (`tract-onnx`, `image`, `ndarray`, `rayon`, `egui`, `eframe`, `clap`, `serde`, `serde_json`, `log`) with pinned versions.
- [x] **Models**: Download `face_detection_yunet_2023mar.onnx`, place it under `models/`, and document version plus SHA256 in `models/README.md`.
- [x] **Scaffolding**: Stub out crate-level skeletons (`lib.rs`, `main.rs`, module folders) and ensure `cargo check --workspace` passes.
- [x] **Configs**: Add `.cargo/config.toml` (build target defaults) and extend `.gitignore` for model files, fixtures, and GUI caches.

### Phase 1: Core Logic & Utilities

- [x] **`yunet-core`**: Implement core face detection logic.
  - [x] Pre-processing of input images.
  - [x] ONNX model inference.
  - [x] Post-processing (NMS, score thresholding).
  - [x] Validate output parsing against OpenCV YuNet sample outputs.
- [x] **`yunet-utils`**: Develop shared utilities.
  - [x] Fixture loading for tests.
  - [x] Image and data conversion helpers.
  - [x] Shared configuration loader (`serde` structs for thresholds, input size).
- [x] **Testing**: Add unit tests for `yunet-core`.
  - [x] Test pre-processing logic.
  - [x] Test post-processing logic against known outputs.
  - [x] Add golden-dataset regression test comparing to OpenCV baselines.
  - [x] Capture OpenCV YuNet reference outputs (bboxes, landmarks, scores) for parity checks.
- [x] **Models**: Document initial ONNX model version and checksum in `models/README.md`.

### Phase 2: Command-Line Interface (CLI)

- [x] **`yunet-cli`**: Build the command-line application.
  - [x] Implement the `detect` command.
  - [x] Add arguments for input path, score threshold, NMS threshold, etc.
  - [x] Handle output features.
    - [x] JSON export / stdout summary.
    - [x] Annotated image rendering.
  - [x] Support batch directory processing and glob patterns.
- [x] **Testing**: Create integration tests for the CLI.
  - [x] Basic detection flow producing JSON output (skips if model unavailable).
  - [x] Test command with various images from `fixtures/`.
    - [x] Add parity fixtures with expected detections and compare bounding boxes/landmarks to JSON snapshots.
  - [x] Verify output against golden data.
  - [x] Add snapshot tests for JSON output shape and ordering.

### Phase 3: Graphical User Interface (GUI)

- [x] **`yunet-gui`**: Build the desktop application with `egui`/`eframe`.
  - [x] Implement file selection for input images.
  - [x] Create UI controls for model parameters (thresholds, etc.).
  - [x] Display the output image with bounding boxes overlaid.
  - [x] Offload heavy inference tasks to a background thread (`rayon`) to keep the UI responsive.
  - [x] Persist user preferences (last model, thresholds) between sessions.
- [x] **Styling**: Centralize `egui` styling in `yunet-gui/src/theme.rs`.
- [x] **Testing**: Add smoke tests for the GUI application.
  - [x] Test application startup and basic interaction.
  - [x] Test settings serialization.
  - [x] Add screenshot regression harness for key panels (optional).

    ---

## FACE CROPPING APPLICATION TRANSFORMATION

### Phase 4: Core Cropping Logic (`yunet-core`)

- [x] **Crop calculation module** (`yunet-core/src/cropper.rs`)
  - [x] Define `CropSettings` struct with output dimensions, face height %, positioning mode, offsets
  - [x] Implement `PositioningMode` enum: `Center`, `RuleOfThirds`, `Custom`
  - [x] Implement `calculate_crop_region()` function using scale-based approach
    - [x] Calculate scale factor from face height percentage and output height
    - [x] Compute source region dimensions maintaining aspect ratio
    - [x] Apply positioning mode logic (center, rule of thirds at 1/3, custom offsets)
    - [x] Clamp crop region to image boundaries
  - [x] Unit tests for crop calculations with edge cases (faces near boundaries, various aspect ratios)

- [x] **Size presets** (`yunet-core/src/presets.rs`)
  - [x] Define `CropPreset` struct (name, width, height, description)
  - [x] Implement standard presets:
    - [x] LinkedIn (400×400)
    - [x] Passport (413×531)
    - [x] Instagram (1080×1080)
    - [x] ID Card (332×498)
    - [x] Avatar (512×512)
    - [x] Headshot (600×800)
    - [x] Custom (user-defined)

- [x] **Face cropper implementation** (`yunet-core/src/face_cropper.rs`)
  - [x] Implement `crop_face()` method combining detection + crop calculation
  - [x] Extract and resize face region using `image` crate
  - [x] Handle aspect ratio preservation and quality resampling
  - [x] Return cropped `DynamicImage` or save to disk
  - [x] Integration tests using fixture images (unit tests added; integration with fixtures next)

#### Phase 4 — implemented summary

- Implemented modules:
  - `yunet-core/src/cropper.rs` — crop calculation, `CropSettings`, `PositioningMode`, `calculate_crop_region()`.
  - `yunet-core/src/presets.rs` — `CropPreset`, `standard_presets()` and `preset_by_name()`.
  - `yunet-core/src/face_cropper.rs` — `crop_face_from_image()` which extracts and resizes crops using the `image` crate.

- Verification performed:
  - `cargo check --workspace` — passed.
  - `cargo test --manifest-path yunet-core/Cargo.toml` — all `yunet-core` unit tests passed (including cropper/presets/face_cropper tests).

- Notes & next steps for Phase 4:
  - Add integration tests that use real fixture images in `fixtures/` to validate end-to-end cropping behavior.
  - Wire CLI options (`--crop`, `--preset`, `--face-height-pct`, `--positioning-mode`) in `yunet-cli` (Phase 6 work).
  - Implement quality checks and enhancement hooks in `yunet-utils` (Phase 5) so CLI can skip low-quality crops.

### Phase 5: Image Quality & Enhancement (`yunet-utils`)

- [x] **Quality analysis module** (`yunet-utils/src/quality.rs`)
  - [x] Implement Laplacian variance calculation for blur detection
    - [x] Convert to grayscale
    - [x] Apply 3×3 Laplacian kernel
    - [x] Calculate variance (higher = sharper)
  - [x] Define quality levels: `High` (>1000), `Medium` (>300), `Low`
  - [x] Add quality score to detection output (per-detection `quality_score` / `quality` fields)
  - [x] Add `QualityFilter` settings (minimum quality threshold) and helper `should_skip`

- [ ] **Image enhancement pipeline** (`yunet-utils/src/enhance.rs`)
  - [ ] Auto color correction (histogram equalization)
  - [ ] Exposure adjustment (-2 to +2 stops)
  - [ ] Contrast adjustment (0.5 to 2.0x multiplier)
  - [ ] Sharpness enhancement using unsharp mask (0 to 2.0)
  - [ ] Brightness adjustment
  - [ ] Saturation adjustment
  - [ ] Create `EnhancementSettings` struct with all parameters
  - [ ] Implement enhancement pipeline applying transformations in sequence
  - [ ] Unit tests for each enhancement filter

- [ ] **Advanced enhancements** (optional, Phase 7+)
  - [ ] Skin smoothing using bilateral filter
  - [ ] Red-eye removal (detect + desaturate red regions near eyes)
  - [ ] Background blur/segmentation

### Phase 6: CLI Face Cropping Features

- [x] **Extend CLI arguments** (`yunet-cli/src/main.rs`)
  - [x] Add `--crop` flag to enable cropping mode
  - [x] Add `--output-width` and `--output-height` (or `--preset`)
  - [x] Add `--face-height-pct` (default 70%)
  - [x] Add `--positioning-mode` (center/rule-of-thirds/custom)
  - [x] Add `--vertical-offset` and `--horizontal-offset` (for custom mode)
  - [x] Add `--output-format` (png/jpeg/webp)
  - [x] Add `--jpeg-quality` (1-100)
  - [x] Add `--min-quality` for blur filtering (`--skip-low-quality` maps to `--min-quality medium`)

- [x] **Enhancement CLI options**
  - [x] Add `--enhance` flag to enable enhancements (applies default EnhancementSettings)
  - [x] Add `--auto-color` for auto color correction
  - [x] Add `--exposure`, `--contrast` flags (tunable via CLI)
  - [x] Add `--sharpness`, `--brightness`, `--saturation` flags
  - [x] Add `--enhancement-preset` (natural/vivid/professional)

- [x] **Batch cropping features**
  - [x] Add `--output-dir` for batch output location
  - [x] Add `--naming-template` with variables: `{original}`, `{index}`, `{timestamp}`, `{width}`, `{height}`, `{ext}`
    - Implemented: CLI flag in `yunet-cli/src/main.rs` and test `yunet-cli/tests/naming_template.rs` added. Timestamp uses UNIX epoch seconds.
  - [x] Add `--face-index` to select specific face when multiple detected (default: all)
  - [x] Add `--skip-low-quality` flag to auto-skip blurry faces (maps to `min_quality=medium`)
  - [x] Progress reporting with face count and quality stats
    - [x] Convert numeric enhancement CLI flags to `Option<T>` so presets are unambiguous
      - Implemented `build_enhancement_settings()` helper that applies presets and explicit overrides
      - Added unit tests for preset application and explicit override behavior
      - Added end-of-run summary logging using atomic counters (images processed, faces detected, crops saved, crops skipped)
    - [x] Convert boolean flags to `Option<bool>` for explicit presence detection
      - Converted `--enhance-auto-color`, `--enhance`, and `--skip-low-quality` to accept an explicit presence/value and updated CLI wiring
    - [x] Added lightweight integration-like test for preset+enhance
      - Added a synthetic test that crops a generated image, applies preset+enhance, saves, and asserts output exists
    - [x] Print single-line summary to stdout for interactive runs

- [x] **Testing**
  - [x] Integration tests for crop + save workflow (basic cases implemented)
  - [x] Test all positioning modes produce correct crops (4 tests in positioning_modes.rs)
  - [x] Test preset sizes (6 tests in preset_sizes.rs)
  - [x] Test enhancement pipeline end-to-end (6 tests in enhancement_pipeline.rs)
  - [x] Test batch processing with quality filtering (7 tests in batch_quality_filtering.rs)

### Phase 7: GUI Face Cropping Interface

- [ ] **Crop settings panel** (`yunet-gui/src/crop_panel.rs`)
  - [ ] Output size controls (width/height spinners)
  - [ ] Preset size dropdown (LinkedIn, Passport, Instagram, etc.)
  - [ ] Face height percentage slider (10-100%)
  - [ ] Positioning mode selector (radio buttons)
  - [ ] Vertical/horizontal offset sliders (custom mode only)
  - [ ] Live preview showing crop region overlay on original image

- [ ] **Enhancement controls panel** (`yunet-gui/src/enhancement_panel.rs`)
  - [ ] Auto color correction checkbox
  - [ ] Exposure slider (-2 to +2 stops)
  - [ ] Contrast slider (0.5 to 2.0x)
  - [ ] Sharpness slider (0 to 2.0)
  - [ ] Brightness slider (-100 to +100)
  - [ ] Saturation slider (0 to 2.0)
  - [ ] Enhancement preset dropdown (None/Natural/Vivid/Professional)
  - [ ] Reset to defaults button

- [ ] **Face selection UI**
  - [ ] Display all detected faces as thumbnails
  - [ ] Show quality score badge on each face thumbnail
  - [ ] Allow selecting/deselecting faces for cropping
  - [ ] Show quality filter controls (minimum quality threshold)
  - [ ] Highlight selected faces in main preview

- [ ] **Preview & export**
  - [ ] Split view: original (with crop overlay) | cropped result
  - [ ] Real-time preview updates on settings change
  - [ ] Export format selector (PNG/JPEG/WebP)
  - [ ] JPEG quality slider (when JPEG selected)
  - [ ] "Crop Selected Faces" button
  - [ ] "Export All" for batch processing
  - [ ] Save location picker
  - [ ] Filename template editor

- [ ] **Batch processing UI**
  - [ ] Multi-file selection support
  - [ ] Image list with status (pending/processing/done/error)
  - [ ] Progress bar for batch operations
  - [ ] Summary statistics (faces detected, crops saved, skipped due to quality)
  - [ ] Export log/report with quality scores

- [ ] **Settings persistence**
  - [ ] Save last used crop settings
  - [ ] Save enhancement preferences
  - [ ] Save output format and quality preferences
  - [ ] Save last output directory

### Phase 8: Advanced Features

- [ ] **Output format support** (`yunet-utils/src/output.rs`)
  - [ ] PNG encoder with compression level control
  - [ ] JPEG encoder with quality control (1-100)
  - [ ] WebP encoder (if available via `image` crate)
  - [ ] Format auto-detection from file extension

- [ ] **Metadata preservation**
  - [ ] Copy EXIF data from original to cropped image (when possible)
  - [ ] Add custom metadata tags (crop settings, face confidence, quality score)
  - [ ] Option to strip all metadata for privacy

- [ ] **Quality-based auto-cropping**
  - [ ] Auto-select highest quality face when multiple detected
  - [ ] Auto-skip images with no high-quality faces
  - [ ] Quality-based output filename suffixes (e.g., `_highq`, `_mediumq`)

- [ ] **Undo/Redo system** (GUI)
  - [ ] History stack for crop settings changes
  - [ ] Undo/redo shortcuts (Ctrl+Z / Ctrl+Y)

- [ ] **Keyboard shortcuts** (GUI)
  - [ ] Arrow keys for offset adjustments
  - [ ] +/- for face height percentage
  - [ ] Number keys for preset selection
  - [ ] Space to toggle enhancement preview
  - [ ] Enter to export

### Phase 9: Testing & Documentation

- [ ] **Comprehensive testing**
  - [ ] Unit tests for all crop calculation edge cases
  - [ ] Unit tests for quality analysis (blur detection)
  - [ ] Unit tests for each enhancement filter
  - [ ] Integration tests for full crop workflow
  - [ ] GUI smoke tests for crop panel interactions
  - [ ] Performance benchmarks for crop + enhancement pipeline

- [ ] **Documentation**
  - [ ] Document crop calculation algorithm in `yunet-core/src/cropper.rs`
  - [ ] Add examples showing each positioning mode
  - [ ] Document quality scoring methodology
  - [ ] Create user guide for GUI crop features
  - [ ] Create CLI examples for common use cases
  - [ ] Update README with crop features overview
  - [ ] Add ARCHITECTURE.md explaining crop calculation flow

### Phase 10: Integration & Refinement

- [ ] **Workspace**: Ensure all crates work together seamlessly.
- [ ] **Configuration**: Share threshold/input-size config structs between CLI and GUI.
- [ ] **Documentation**: Add `///` doc comments to all public functions and modules.
- [ ] **Linting & Formatting**: Run `cargo clippy` and `cargo fmt` across the workspace to ensure code quality.
- [ ] **Performance**: Profile and optimize the inference pipeline in `yunet-core`.
- [ ] **Telemetry**: Add optional timing/logging hooks in `yunet-utils` for inference tracing.
- [ ] **Parity validation**: Compare detections against OpenCV YuNet on a shared dataset.
  - [ ] Evaluate IoU overlap, detection recall/precision, and score deltas.
  - [ ] Document discrepancies and adjust preprocessing/postprocessing until parity is within tolerance.

### Phase 11: Finalization & Release

- [ ] **Polish & UX refinements**
  - [ ] Add tooltips explaining each crop parameter
  - [ ] Add visual guides for rule of thirds in preview
  - [ ] Error handling with user-friendly messages
  - [ ] Loading states and progress indicators
  - [ ] Success/error notifications for batch operations

- [ ] **Performance optimization**
  - [ ] Cache cropped previews to avoid recomputation
  - [ ] Parallelize batch cropping with rayon
  - [ ] Optimize enhancement pipeline (SIMD if possible)
  - [ ] Memory-efficient processing for large batches

- [ ] **CI/CD**
  - [ ] Set up GitHub Actions workflow
  - [ ] Automated testing on Windows/macOS/Linux
  - [ ] Clippy and format checks
  - [ ] Release binary builds for all platforms

- [ ] **Packaging**
  - [ ] Windows installer (MSI or NSIS)
  - [ ] macOS app bundle (.app)
  - [ ] Linux AppImage or .deb package
  - [ ] Include sample images and presets

- [ ] **Release**
  - [ ] Tag `v1.0.0`
  - [ ] Create GitHub release with binaries
  - [ ] Write release notes documenting all features
  - [ ] Update README with installation instructions
  - [ ] Add screenshots/GIFs demonstrating crop features
