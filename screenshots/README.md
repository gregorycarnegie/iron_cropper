# Screenshot & GIF Asset Checklist

Use this folder for release screenshots/GIFs referenced by the README and GitHub
release notes.

## Static screenshots (`v1.0.0`)

- `gui-main.png` - Main GUI view with loaded image and detection overlays.
- `gui-crop-config.png` - Crop configuration panel (presets, fill color, positioning).
- `gui-batch.png` - Batch queue/export workflow.
- `gui-enhancement.png` - Enhancement controls in use.
- `cli-example.png` - Terminal screenshot of representative CLI run.

## Workflow GIFs (to record)

Short looping clips of the main workflows. The README has a ready-to-uncomment
"Workflow recordings" block; drop these files in and uncomment it.

| File | Shows | Suggested length |
|------|-------|------------------|
| `workflow-detect-crop-export.gif` | Load an image → faces detected → tweak crop/positioning → export. The headline flow. | 8–12 s |
| `workflow-batch.gif` | Drop a folder → queue fills → start batch → per-file statuses update to ok/skip. | 8–12 s |
| `workflow-enhance.gif` | Adjust enhancement sliders (exposure/contrast/smoothing) with the preview updating live. | 6–10 s |
| `workflow-mapping.gif` | Drop a CSV/XLSX → mapping panel → pick source/output columns → preview rows. | 8–12 s |

## Capture guidance

- Prefer 16:9 at readable scale (e.g. 1600x900+); for GIFs, 1280x720 keeps file
  size reasonable.
- Use real, non-sensitive sample images from `fixtures/` or `samples/`.
- Avoid including personal file paths/usernames in frame.
- Keep the UI in a realistic state (model loaded, meaningful detections).
- Keep each GIF under ~5 MB so it loads quickly on GitHub. Trim to a single
  clean pass of the workflow and let it loop.

### Recording tools

- **Windows**: [ScreenToGif](https://www.screentogif.com/) (record a window
  region, edit frames, export GIF directly).
- **Cross-platform**: record MP4 (OBS, or the OS recorder) then convert:

  ```bash
  # MP4 → optimized looping GIF via ffmpeg + a shared palette
  ffmpeg -i workflow.mp4 -vf "fps=15,scale=1280:-1:flags=lanczos,palettegen" palette.png
  ffmpeg -i workflow.mp4 -i palette.png -vf "fps=15,scale=1280:-1:flags=lanczos,paletteuse" workflow-detect-crop-export.gif
  ```
