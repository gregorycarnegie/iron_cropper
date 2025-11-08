# Architecture Overview

The Iron Cropper workspace is split into four crates that collaborate to deliver face detection, cropping, and post-processing across CLI and GUI front-ends.

```text
Root Cargo.toml
├── yunet-core     # Detection, cropping, presets, ONNX integration
├── yunet-utils    # Shared config, quality scoring, enhancement & export helpers
├── yunet-cli      # Command-line entry point and batch automation
└── yunet-gui      # eframe/egui desktop application
```

This document focuses on the crop calculation pipeline introduced in Phase 4 and extended through Phase 9.

## Data Flow

1. **Detection** – `YuNetDetector` (in `yunet-core`) runs the ONNX model via `tract-onnx`, returning `DetectionOutput` records that contain bounding boxes, landmarks, and confidence scores.
2. **Crop Derivation** – `calculate_crop_region` in `yunet-core/src/cropper.rs` transforms a detection into a bounded `CropRegion`. The algorithm:
   - clamps the configured face-height percentage to `[1, 100]`,
   - derives a source height that will yield the requested face coverage once resized,
   - mirrors the output aspect ratio to compute the source width,
   - applies positioning logic (center, rule of thirds, custom offsets),
   - and clamps the resulting rectangle to the original image dimensions.
3. **Crop Extraction** – `crop_face_from_image` uses `image::imageops::crop_imm` followed by a Lanczos3 resize to obtain the final output dimensions. This abstraction is reused by both the CLI and GUI.
4. **Enhancement & Quality** – `yunet-utils` kicks in next:
   - `apply_enhancements` runs the optional enhancement pipeline (auto color, exposure, contrast, saturation, unsharp mask, skin smoothing, red-eye removal, and background blur).
   - `estimate_sharpness` computes Laplacian variance to classify the crop as Low/Medium/High quality. These scores inform automation like `QualityFilter::select_best_index` and filename suffixing.
5. **Export** – `save_dynamic_image` encodes the image (PNG/JPEG/WebP), optionally injects metadata (original EXIF, crop configuration, quality metrics), and writes to disk.

The CLI stitches these steps together in synchronous code paths. The GUI pushes detection and enhancement workloads onto background Rayon tasks to keep the egui frame loop responsive, caching results in `DetectionCacheEntry`s keyed by model configuration.

## YuNet Model Loading

`YuNetModel::load` first attempts to run `tract-onnx`'s `into_optimized()` pipeline, which performs operator fusion and constant folding. When tract cannot optimize the graph, the loader logs a warning and falls back to the decluttered graph via `into_decluttered()`. This mode keeps inference functional but roughly doubles end-to-end inference latency because key optimizations are skipped. Watch for the warning in CLI/GUI logs to diagnose unexpected slowdowns or incompatible ONNX exports. The upstream `face_detection_yunet_2023mar.onnx` file, for example, encodes contradictory spatial hints (Conv_0 claims its output is `1×16×160×160` even though the input tensor is `1×3×320×320`), so tract refuses to type-check the network. We ship the sanitized `face_detection_yunet_2023mar_640.onnx` export—which locks the input to `640×640`—as the default model to keep tract on the fast path.

## Crop History & Undo/Redo

The GUI maintains a circular history buffer (max 100 entries) of `CropSettings` snapshots. Interactions that affect framing—preset switches, slider changes, keyboard nudges—call `push_crop_history`, making undo/redo operations deterministic. This behaviour is covered by the smoke tests in `yunet-gui/src/main.rs`.

## Benchmarks

`yunet-core/benches/crop_enhance.rs` provides a Criterion micro-benchmark for the combined crop + enhancement pipeline. It generates a synthetic high-frequency region, runs `crop_face_from_image`, and immediately applies the enhancement stack. Use `cargo bench -p yunet-core crop_enhance` to track latency when tuning algorithms.

## Testing Matrix

| Area                    | Location                                       | Purpose                                           |
|-------------------------|------------------------------------------------|---------------------------------------------------|
| Crop edge cases         | `yunet-core/src/cropper.rs`                    | Unit tests for clamping, aspect ratio, offsets    |
| Face extraction         | `yunet-core/src/face_cropper.rs`               | Ensures resize dimensions match configuration     |
| Quality scoring         | `yunet-utils/src/quality.rs`                   | Threshold bucketing, filters, suffix logic        |
| Enhancement pipeline    | `yunet-utils/src/enhance.rs`                   | Unit + pipeline parity tests                      |
| Full crop workflow      | `yunet-core/tests/full_crop_workflow.rs`       | Integration test from detection to export         |
| CLI scenarios           | `yunet-cli/tests/*.rs`                         | Snapshot, batch, naming, enhancement workflows    |
| GUI smoke tests         | `yunet-gui/src/main.rs` (test module)          | Crop adjustments, undo/redo, preset application   |
| GUI visuals             | `yunet-gui/tests/screenshot.rs`                | Snapshot-based overlay verification               |

These layers give confidence that the crop calculation flow remains stable across both user interfaces while providing hooks to measure and iterate on performance.
