# Iron Cropper

Iron Cropper is a Rust workspace that wraps the YuNet face detector with deterministic cropping, quality analysis, enhancement, and export tooling. The project ships both a command-line workflow and an egui desktop application, backed by shared utilities for image processing, metadata handling, and configuration.

## Crates

- **`yunet-core`** – Loads the YuNet ONNX model with `tract-onnx`, handles preprocessing/postprocessing, and implements the crop calculation logic (see `ARCHITECTURE.md` for details).
- **`yunet-utils`** – Shared helpers: configuration structs, Laplacian-variance quality scoring, enhancement pipeline, and output encoders with metadata support.
- **`yunet-cli`** – Command-line frontend aimed at batch processing and automation. Example invocations are documented in `docs/cli_recipes.md`.
- **`yunet-gui`** – eframe/egui desktop experience with live preview, crop adjustments, enhancements, history/undo, and batch export. A user guide lives in `docs/gui_crop_guide.md`.

## Crop Features Overview

- **Preset sizing** – LinkedIn, Passport, Instagram, ID Card, Avatar, Headshot, plus an explicit “Custom” mode.
- **Face height targeting** – Configure how large the face should appear in the final crop (10–100%). The cropper preserves the requested aspect ratio and clamps within source bounds.
- **Positioning modes** – Center, Rule of Thirds, or fully custom offsets with keyboard nudges and undo/redo support.
- **Quality automation** – Laplacian-variance scoring categorises crops into Low/Medium/High. Filters can auto-select the sharpest face, skip soft captures, and append quality suffixes.
- **Enhancement pipeline** – Optional post-crop adjustments (auto color, exposure, brightness, contrast, saturation, sharpening, skin smoothing, red-eye removal, and portrait background blur) implemented in pure Rust.
- **Metadata & export** – Preserve, strip, or customise metadata. Exports support PNG, JPEG (with quality controls), and WebP.
- **Batch processing** – Both CLI and GUI support multi-image workflows with status tracking, filenames derived from templates, and quality-aware selection.

## Development Tasks

- `cargo check --workspace` – Fast type checking across all crates.
- `cargo test --workspace --all-features` – Run the full test suite (requires the YuNet 640×640 ONNX model under `models/`).
- `cargo run -p yunet-cli -- --help` – View CLI options.
- `cargo run -p yunet-gui` – Launch the GUI with default settings.
- `cargo bench -p yunet-core crop_enhance` – Measure the crop + enhancement micro-benchmark.
- `cargo fmt --all && cargo clippy --workspace -- -D warnings` – Formatting and linting hygiene.

## Configuration

- Settings persist to `config/gui_settings.json`. The GUI saves changes automatically, and the CLI now reads the same file by default when `--config` is omitted, keeping thresholds and input dimensions in sync across surfaces.

## Diagnostics

- CLI: pass `--telemetry` (optionally `--telemetry-level trace`) to log scoped timings under the `yunet::telemetry` target. Use `--telemetry-level off` to disable timing logs for that run.
- GUI: toggle the **Diagnostics → Telemetry logging** checkbox in the settings panel to emit the same timing traces.

## Documentation

- `ARCHITECTURE.md` – End-to-end architecture and crop pipeline notes.
- `docs/gui_crop_guide.md` – Detailed walkthrough of the GUI crop features.
- `docs/cli_recipes.md` – Command-line recipes for common automation scenarios.

Refer to `TODO.md` for the broader roadmap and phase breakdown.
