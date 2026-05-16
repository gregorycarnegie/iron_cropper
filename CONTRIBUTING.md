# Contributing to Face Crop Studio

Thanks for helping improve Face Crop Studio. This repository is a Rust workspace with four main crates:

- `fcs-core`: face detection, YuNet model loading, crop math, and GPU inference pieces.
- `fcs-utils`: shared configuration, mapping import, image enhancement, export helpers, and webcam support.
- `fcs-cli`: command-line workflows for batch processing and automation.
- `fcs-gui`: the egui/eframe desktop application.

## Development Setup

Install the usual Rust and Windows build tools first:

- Rust toolchain. CI currently uses Rust `1.95.0`.
- Visual Studio Build Tools with the C++ desktop workload and Windows SDK.
- Git.
- NASM on `PATH`; for example `C:\Program Files\NASM`.
- `pkg-config` and `dav1d` for AVIF support.

On Windows, the project CI uses `pkgconfiglite` and vcpkg:

```powershell
choco install pkgconfiglite --no-progress -y

if (-not (Test-Path "C:\vcpkg\vcpkg.exe")) {
  git clone https://github.com/microsoft/vcpkg C:\vcpkg
  & C:\vcpkg\bootstrap-vcpkg.bat
}

& C:\vcpkg\vcpkg.exe install dav1d:x64-windows-static
```

Set the environment variables for your current terminal:

```powershell
$env:PKG_CONFIG="C:\ProgramData\chocolatey\bin\pkg-config.exe"
$env:PKG_CONFIG_PATH="C:\vcpkg\installed\x64-windows-static\lib\pkgconfig"
$env:PKG_CONFIG_ALL_STATIC="1"
```

To make the vcpkg settings permanent for new terminals:

```powershell
[Environment]::SetEnvironmentVariable("PKG_CONFIG_PATH", "C:\vcpkg\installed\x64-windows-static\lib\pkgconfig", "User")
[Environment]::SetEnvironmentVariable("PKG_CONFIG_ALL_STATIC", "1", "User")
```

The full test suite expects the YuNet model at:

```text
models/face_detection_yunet_2023mar_640.onnx
```

See `models/README.md` for checksum and regeneration details.

## Common Commands

Run a fast workspace check:

```powershell
cargo check --workspace
```

Run the full test suite:

```powershell
cargo test --workspace --all-features
```

Run formatting and lint checks:

```powershell
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
```

Launch the CLI and GUI:

```powershell
cargo run -p fcs-cli -- --help
cargo run -p fcs-gui
```

## Contribution Workflow

1. Create a branch for your change.
2. Keep changes focused and avoid unrelated formatting churn.
3. Add or update tests when changing behavior.
4. Update docs when changing user-facing workflows, setup, configuration, or CLI flags.
5. Run the relevant checks before opening a pull request.

For high-risk changes, prefer smaller pull requests that isolate behavior changes from refactors.

## Dependency Guidance

- Prefer workspace dependencies in the root `Cargo.toml`.
- Update `Cargo.lock` with the manifest change.
- Avoid local `[patch.crates-io]` overrides unless there is no practical alternative.
- If an upstream compatible release is broken, prefer an exact version pin with a short explanation in the pull request.
- Keep dependency upgrades separate from feature work when practical.

The workspace currently pins `nokhwa` exactly because `0.10.11` does not compile with the resolved Windows backend/core combination.

## Code Style

- Follow the existing crate boundaries.
- Keep application-level error context helpful.
- Keep shared helpers small and testable.
- Use `cargo fmt` for formatting.
- Do not commit build outputs from `target/`.

## Reporting Issues

When reporting a bug, include:

- Operating system and GPU model if relevant.
- The command or GUI workflow used.
- The input format involved, such as image type, CSV, Excel, Parquet, or SQLite.
- The full error output or logs.
- Whether GPU acceleration was enabled or disabled.
