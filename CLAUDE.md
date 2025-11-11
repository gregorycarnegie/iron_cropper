# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YuNet is a pure Rust implementation of face detection using the YuNet ONNX model. This is a Cargo workspace with four crates organized by responsibility:

- **yunet-core**: Core inference logic (preprocessing, ONNX model execution via tract, postprocessing with NMS)
- **yunet-cli**: Command-line interface for batch face detection
- **yunet-gui**: Desktop application built with egui/eframe
- **yunet-utils**: Shared utilities (config loading, fixture management, image helpers)

## Build and Test Commands

```bash
# Quick type check across workspace
cargo check --workspace

# Run all tests (requires models/face_detection_yunet_2023mar_640.onnx)
cargo test --workspace --all-features

# Run CLI tool (defaults to 640x640 resolution)
cargo run -p yunet-cli -- --help
cargo run -p yunet-cli -- --input fixtures/ --model models/face_detection_yunet_2023mar_640.onnx

# Run GUI application (defaults to 640x640 resolution)
cargo run -p yunet-gui

# Format and lint (required before commits)
cargo fmt --all
cargo clippy --workspace -- -D warnings

# Release builds for performance testing
cargo test --release
cargo build --release -p yunet-cli
```

## Architecture

### Inference Pipeline (yunet-core)

1. **Preprocessing** (preprocess.rs): Loads images, resizes to InputSize (default 640x640), converts RGB to normalized CHW tensor format, tracks scale factors for coordinate mapping
2. **Model Execution** (model.rs): YuNetModel wraps tract-onnx SimplePlan, constrains input dimensions at load time, runs inference returning raw output tensor
3. **Postprocessing** (postprocess.rs): Parses model output into Detection structs (bbox, landmarks, score), applies score thresholding and Non-Maximum Suppression (NMS), scales coordinates back to original image dimensions

### Configuration System

- `yunet-utils/src/config.rs` defines AppSettings with nested InputSettings and DetectionSettings structs
- Default resolution: 640x640 (required for tract compatibility with available ONNX model)
- Default thresholds match OpenCV YuNet: `score_threshold=0.9`, `nms_threshold=0.3`, `top_k=5000`
- CLI accepts JSON config file (`--config`) and individual parameter overrides (`--width`, `--score-threshold`, etc.)

### CLI Design

- Accepts single image or directory with recursive traversal (walkdir)
- Outputs JSON array of detections with bbox [x,y,w,h] and 5 landmarks [[x,y],...]
- `--annotate` flag renders bounding boxes (red) and landmarks (green circles) using imageproc
- Batch processing continues on individual failures, logging warnings

### GUI Architecture

- egui/eframe for cross-platform desktop UI
- Inference offloaded to rayon background threads to maintain UI responsiveness
- Should cache preprocessed tensors per image to avoid redundant work
- Settings persistence between sessions

## Model Management

- ONNX models live in `models/` (git-ignored)
- `models/README.md` tracks model versions and SHA256 checksums
- Current model: `face_detection_yunet_2023mar.onnx` (March 2023)
- Verify downloaded models against documented checksums before use

## Testing Strategy

- **Unit tests**: Core preprocessing and postprocessing logic in yunet-core
- **Integration tests**: CLI detection flow using fixtures (tests skip if model unavailable)
- **Parity validation**: Compare against OpenCV YuNet reference outputs (planned)
- **Test fixtures**: Golden inputs in `fixtures/`, loaded via yunet-utils helpers
- Use default thresholds in tests to catch regressions
- Cover edge cases: no faces, single face, multiple overlapping faces

## Development Conventions

### Code Style

- Rust 2024 edition, stable toolchain
- rustfmt with 4-space indentation, trailing commas
- snake_case modules/functions, PascalCase types, SCREAMING_SNAKE_CASE constants
- Keep main.rs minimal, document YuNet-specific behaviors with `///`
- GUI styling centralized in yunet-gui/src/theme.rs

### Task Management

- Update TODO.md when completing scoped tasks
- Tests should fail descriptively with context about expected vs actual values

### Python Scripts

- Place utility scripts under `scripts/`
- Use local virtual environments, never install to system interpreter

## Common Issues

- **Missing model errors**: Download ONNX model to `models/` directory and verify checksum
- **Coordinate scaling**: Always use scale_x and scale_y from PreprocessOutput when mapping detections back to original image space
- **NMS behavior**: IoU calculation and suppression logic in postprocess.rs follows YuNet paper
- **Image loading**: yunet-utils abstracts image loading; use those helpers for consistency
- **Model compatibility**: The 320x320 YuNet model (`face_detection_yunet_2023mar.onnx`) has a tract compatibility issue (fails at Conv_0 node during optimization). Use the 640x640 model (`face_detection_yunet_2023mar_640.onnx`) instead. Attempts to use tract Symbols for dynamic shapes did not resolve the underlying model export issue.

## Python Interpreter Usage

- Always use the local Python interpreter located in the `scripts/` folder when running Python scripts or installing packages with `pip`.
- Do not use the system/global interpreter.
- If the `scripts/` folder does not exist:
  1. Create it at the root of the repository (`mkdir scripts`).
  2. Set up a local virtual environment inside `scripts/` (e.g., `python -m venv scripts/venv`).
  3. Use `scripts/venv/bin/python` (Linux/macOS) or `scripts\venv\Scripts\python.exe` (Windows) as the interpreter path.
- All utility scripts must explicitly reference this interpreter to ensure reproducibility.
