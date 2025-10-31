# Gemini Project Context: YuNet Face Detection

This document provides context for the `yunet` project, a Rust-based face detection application, intended for use by the Gemini AI assistant.

## Project Overview

This is a multi-crate Rust workspace for face detection using the YuNet ONNX model.

* **Crates:**
  * `yunet-core`: Handles preprocessing, inference, and postprocessing logic.
  * `yunet-gui`: A desktop application built with `egui`/`eframe`.
  * `yunet-cli`: A command-line interface.
  * `yunet-utils`: Shared utilities, likely for handling fixtures.
* **Models:** ONNX models are stored in the `models/` directory (which is git-ignored). A `models/README.md` file tracks model versions and checksums.
* **Fixtures:** Shared test fixtures are located in the `fixtures/` directory.

## Task Management

* After completing a task, you **MUST** update the `TODO.md` file to check off the completed item.

## Build, Test, and Development Commands

* **Quick Check:**

    ```sh
    cargo check --workspace
    ```

* **Run All Tests:**

    ```sh
    cargo test --workspace --all-features
    ```

* **Run GUI:**

    ```sh
    cargo run -p yunet-gui
    ```

* **Run CLI:**

    ```sh
    cargo run -p yunet-cli -- detect --help
    ```

* **Formatting & Linting (Run before push):**

    ```sh
    cargo fmt --all
    cargo clippy --workspace -- -D warnings
    ```

## Development Conventions

### Coding Style & Naming

* **Toolchain:** Use the stable Rust toolchain (`rustup default stable`).
* **Formatting:** Code is formatted with `rustfmt` using 4-space indentation and trailing commas.
* **Naming:**
  * Modules/Functions: `snake_case`
  * Types/Enums: `PascalCase`
  * Constants: `SCREAMING_SNAKE_CASE`
* **Structure:** Keep `main.rs` files minimal. GUI styling is centralized in `yunet-gui/src/theme.rs`.

### Testing

* **Defaults:** Tests should use OpenCV YuNet's default thresholds (`score_threshold=0.9`, `nms_threshold=0.3`, `top_k=5000`) to catch regressions.
* **Fixtures:** Golden inputs/outputs are in `fixtures/` and loaded via `yunet-utils`.
* **Scenarios:** Ensure test coverage for images with no faces, a single face, and multiple faces.

### Commits & Pull Requests

* **Commit Messages:** Use the imperative present tense (e.g., "Implement YuNet NMS").
* **Branching:** Rebase on `main` before pushing; avoid merge commits in PRs.
* **PRs:** Summarize the approach, list verification commands, link issues, and attach artifacts (JSON/PNG) if detection output changes.

## Security

* **Secrets:** Never commit proprietary images or API keys. Use a `.env` file for local overrides.
* **Models:** Verify YuNet ONNX file downloads against the SHA256 checksums in `models/README.md`.
