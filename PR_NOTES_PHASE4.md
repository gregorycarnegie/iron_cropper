# PR Notes — Phase 4: Core Cropping Logic

## Summary

This change implements Phase 4 of the face-cropping transformation. It adds crop calculation utilities, standard presets, and a simple face-cropping implementation that extracts and resizes detected faces.

Files added/modified

- `yunet-core/src/cropper.rs` — Implements `CropSettings`, `PositioningMode`, and `calculate_crop_region()` with clamping and aspect preservation. Includes unit tests for center and edge cases.
- `yunet-core/src/presets.rs` — Adds `CropPreset`, `standard_presets()`, and `preset_by_name()` with unit tests.
- `yunet-core/src/face_cropper.rs` — Adds `crop_face_from_image()` that extracts a source region using the cropper and resizes with Lanczos3. Includes a unit test using a synthetic image.
- `yunet-core/src/lib.rs` — Exposes the new modules and re-exports key helpers.
- `TODO.md` — Updated Phase 4 checklist and added verification notes.

## Verification

- `cargo check --workspace` — passed.
- `cargo test --manifest-path yunet-core/Cargo.toml` — all `yunet-core` tests passed locally (cropper, presets, face_cropper and existing tests).

## Design notes

- The crop calculation is scale-based: source region height is computed so the face's height occupies the requested percentage of the output height; source width preserves the output aspect ratio.
- Positioning modes supported: `Center`, `RuleOfThirds`, and `Custom` (custom uses offsets relative to half the crop size).
- The face cropper returns an owned `DynamicImage` resized to the requested `output_width` x `output_height`. If either output dimension is 0 the raw crop is returned unchanged.

## Next steps / follow-ups

1. Add integration tests using the `fixtures/` images to validate cropping against known-good results and visual expectations.
2. Wire CLI options in `yunet-cli` (Phase 6): `--crop`, `--preset`, `--face-height-pct`, `--positioning-mode`, offsets, output format/quality, and batch options.
3. Implement quality analysis (Phase 5) in `yunet-utils` so CLI can filter out low-quality crops.
4. Add GUI crop panel and live preview in `yunet-gui` (Phase 7).

## Review checklist

- [ ] Verify design matches UX expectations for presets and default face height (currently default 70%).
- [ ] Validate rule-of-thirds placement visually for multiple fixture images.
- [ ] Confirm resizing quality (Lanczos3) is acceptable; consider exposing the resampling filter as a setting if needed.

## How to test locally

Run the unit tests for the core crate:

```powershell
cargo test --manifest-path yunet-core/Cargo.toml
```

Run workspace check:

```powershell
cargo check --workspace
```

To manually exercise cropping from the CLI once flags are wired, a usage example will be added to the CLI docs (Phase 6).
