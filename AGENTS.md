# Repository Guidelines

## Project Structure & Module Organization

- Root `Cargo.toml` manages crates `yunet-core`, `yunet-gui`, `yunet-cli`, and `yunet-utils`; document ownership in each crate README.
- Host preprocessing, inference, and postprocessing in `yunet-core/src/`; store YuNet ONNX files under `models/` (ignored) and record version plus checksum in `models/README.md`.
- Build the desktop app in `yunet-gui/src/` with `egui`/`eframe`, while CLI flows stay in `yunet-cli/src/main.rs` with shared fixtures under `fixtures/`.

## Build, Test, and Development Commands

- `cargo check --workspace` - quick type check.
- `cargo test --workspace --all-features` - run unit plus integration suites (requires `models/face_detection_yunet_2023mar_640.onnx`).
- `cargo run -p yunet-gui` (defaults to 640x640 resolution) or `cargo run -p yunet-cli -- --help` (defaults to 640x640 resolution) - smoke test desktop and CLI surfaces.
- `cargo fmt --all` && `cargo clippy --workspace -- -D warnings` - enforce formatting and lint hygiene before pushing.

## Coding Style & Naming Conventions

- Stay on stable Rust (`rustup default stable`) and let `rustfmt` manage 4-space indentation, trailing commas, and import ordering.
- Use `snake_case` for modules/functions, `PascalCase` for types/enums, and `SCREAMING_SNAKE_CASE` for constants; align file names with module names.
- Keep `main.rs` files thin, document YuNet-specific behaviors with `///`, and centralize `egui` styling in `yunet-gui/src/theme.rs` for consistent spacing and colors.
- When authoring utility scripts, place them under `scripts/` and create a local Python virtual environment per script; never install packages into the system interpreter.

## Testing Guidelines

- Mirror OpenCV YuNet defaults (`score_threshold=0.9`, `nms_threshold=0.3`, `top_k=5000`) with 640x640 input resolution so parity regressions surface quickly.
- Store golden inputs/outputs in `fixtures/` and load them via `yunet-utils`; name tests for user-visible behaviors.
- Run `cargo test --release` for latency checks and ensure coverage for empty, single-face, and crowded images.
- Tests require the 640x640 model (`models/face_detection_yunet_2023mar_640.onnx`). The 320x320 model has tract compatibility issues (fails at Conv_0 node during optimization).

## Commit & Pull Request Guidelines

- Use imperative, present-tense commit messages (`Implement YuNet NMS`) and keep changes scoped.
- Rebase on `main` before pushing; avoid merge commits in PR branches.
- Summarize approach, list verification commands (`cargo test`, `cargo clippy`), link issues, and attach JSON/PNG artifacts when detection output changes; tag TODOs as `// TODO(name):`.
- Update `TODO.md` after finishing any scoped task so the roadmap stays current.

## Security & Configuration Tips

- Never commit proprietary images or API keys; keep overrides in `.env`.
- Verify YuNet ONNX downloads with SHA256 recorded in `models/README.md`, and document new dependencies in the PR rationale.

## GUI Development Guidelines

- Power the desktop shell with `egui`/`eframe`, offloading heavy inference to `rayon` tasks to keep frames responsive.
- Cache textures and normalized tensors per image to reduce uploads; refresh caches when model paths or thresholds change.
- Match CLI capabilities: expose model selection, `input_size`, and YuNet thresholds in the UI, and cover settings serialization with an `app/tests/` smoke test.

## Python Interpreter Usage

- Always run Python scripts and install packages using the interpreter located in the `scripts/` folder.
- Do not rely on the system/global interpreter.
- If the `scripts/` folder does not exist:
  1. Create it (`mkdir scripts`).
  2. Initialize a virtual environment (`python -m venv scripts/venv`).
  3. Use the interpreter path `scripts/venv/bin/python` (Linux/macOS) or `scripts\venv\Scripts\python.exe` (Windows).
- All developer instructions and automation must reference this interpreter to maintain consistency across environments.
