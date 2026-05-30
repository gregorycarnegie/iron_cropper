# Test Fixtures

Sample images, golden outputs, and reference metadata for the test suite.
Individual fixture assets should not be committed if they contain proprietary
or sensitive information; prefer synthetic or cleared data.

## Layout

- `fixtures/images/` — raw input frames used by the OpenCV parity test.
  **Local only** (git-ignored): these are real faces, so they are not committed.
  Tests that need them skip gracefully when the directory is absent (e.g. in CI).
- `fixtures/opencv/` — OpenCV YuNet reference detections for parity validation.
  **Local only** (git-ignored), paired with `fixtures/images/`.
- `fixtures/golden/` — **committed** golden outputs. Synthetic, no image data:
  - `crop_regions.json` — expected `CropRegion` for the scenarios in
    `fcs-core/tests/golden_crop_regions.rs`.

## Regenerating golden crop regions

After an intentional change to the crop geometry, refresh and review the diff:

```powershell
$env:UPDATE_GOLDEN = "1"; cargo test -p fcs-core --test golden_crop_regions
```

```bash
UPDATE_GOLDEN=1 cargo test -p fcs-core --test golden_crop_regions
```

When adding fixtures that should be committed, re-include their path in
`.gitignore` (the `fixtures/*` rule ignores fixture contents by default).
