# GUI Overhaul — fcs-gui2 Todo

Redesign the desktop GUI as a new `fcs-gui2` crate, styled after the Windows-11 mockup
in `fcs-gui2/face-crop-studio-windows.html`. Functionality is wired from `fcs-core` and
`fcs-utils` exactly as in the existing `fcs-gui` crate.

---

## Phase 1 — Scaffolding

- [x] Add `fcs-gui2` to workspace `members` in root `Cargo.toml`
- [x] Create `fcs-gui2/Cargo.toml` (mirrors `fcs-gui` deps + `bytemuck`)
- [x] Create `fcs-gui2/build.rs` (Windows icon resource, same as `fcs-gui`)
- [x] Create `fcs-gui2/src/main.rs` — `eframe::run_native` entry point
- [x] Create `fcs-gui2/src/lib.rs` — re-exports for `main.rs`

## Phase 2 — Theme & Palette

- [x] Create `fcs-gui2/src/theme.rs`
  - [x] Define `Palette` constants matching HTML CSS vars (`--bg`, `--surface`, `--peach`,
        `--cyan`, `--lime`, `--rose`, `--ink`, `--ink-2`, `--ink-3`, `--rule`, `--rule-2`)
  - [x] `apply(ctx)` fn — sets egui `Visuals`, fonts (DM Mono fallback to monospace),
        rounding, spacing, shadows, scrollbar style
  - [x] Helper fn `mono_font()` for inline monospace spans
  - [x] Helper fns `btn_primary_style()`, `btn_peach_style()`, `btn_ghost_style()`,
        `btn_danger_style()` returning `egui::style::WidgetVisuals`

## Phase 3 — App State

- [x] Create `fcs-gui2/src/types.rs`
  - [x] `App2` struct — mirrors `YuNetApp` fields:
        detector, gpu_context, gpu_enhancer, gpu_batch_cropper,
        settings (`AppSettings`), preview state, LRU caches, selected_faces,
        batch_files, webcam_state, UI flags
  - [x] `BatchFile` struct (path, status: Queued/Running/Done/Skip/Error, face_count)
  - [x] `PreviewState` (loaded image, texture handle, zoom, pan)
  - [x] `SidebarTab` enum (Queue, Mapping, History)
  - [x] `InspectorTab` enum (Crop, Output, Enhance)
  - [x] `PanelState` struct (open booleans for each collapsible panel)

## Phase 4 — App Impl (eframe::App)

- [x] Create `fcs-gui2/src/app.rs`
  - [x] `impl App2 { pub fn new(cc) }` — GPU init, detector load, settings load
        (port from `fcs-gui/src/app_impl.rs`)
  - [x] `impl eframe::App for App2 { fn update() }` — top-level layout shell:
        calls title_bar → menu_bar → toolbar → main_area → status_bar
  - [x] Window size / title / icon setup

## Phase 5 — Title Bar

- [x] Create `fcs-gui2/src/ui/titlebar.rs`
  - [x] Conic-gradient logo disc (drawn with egui `Painter`)
  - [x] App name + separator + current filename + unsaved-changes dot
  - [x] Fake Windows 11 min/max/close buttons (right-aligned)
        — actual window control delegated to eframe; buttons are decorative
  - [x] Drag region via `interact` with `egui::Sense::click_and_drag`

## Phase 6 — Menu Bar

- [x] Create `fcs-gui2/src/ui/menubar.rs`
  - [x] File / Edit / View / Detect / Tools / Window / Help menu items
  - [x] Right side: YuNet model status pill with pulsing green dot

## Phase 7 — Toolbar

- [x] Create `fcs-gui2/src/ui/toolbar.rs`
  - [x] "Detect faces →" primary (cyan) button → triggers detection on current image
  - [x] "Export crops" peach button → opens export flow
  - [x] Icon buttons: Open, Save, Undo, Redo
  - [x] Rotation buttons: ↶ 90°, 90° ↷
  - [x] Select All / Select None buttons
  - [x] Clear (danger) button
  - [x] Right side: GPU status pill (wgpu backend + adapter name)

## Phase 8 — Left Sidebar

- [x] Create `fcs-gui2/src/ui/sidebar.rs`
  - [x] Tab bar: Queue | Mapping | History
  - [x] **Queue tab**:
    - [x] Drop zone with dashed border (drag-and-drop via `egui::DroppedFile`)
    - [x] File tree grouped by status (In progress / Queued / Done)
    - [x] Each row: index, person icon, filename, status badge
          (run=peach, ok=lime+count, skip=grey, err=rose)
    - [x] Click row → load that image into canvas
    - [x] `rfd::AsyncFileDialog` for Browse button
  - [x] **Mapping tab** (stub — show existing `ui/mapping` content or placeholder)
  - [x] **History tab** (stub — list `crop_history` entries)

## Phase 9 — Canvas Column

- [x] Create `fcs-gui2/src/ui/canvas.rs`
  - [x] Canvas header: filename, dimensions/size, control chips
        (pad %, aspect, conf threshold, preset name, rotation)
  - [x] **Stage** (image viewport):
    - [x] Display loaded image as egui texture
    - [x] Overlay face bounding boxes (peach/cyan/grey dashed borders)
    - [x] Confidence label badge above each box
    - [x] 5 landmark dots (lime) per face
    - [x] Resize handles on selected face box (4 corner squares)
    - [x] Click face box → toggle selection
    - [x] Drag handles → resize crop region (port from `interaction/bbox_drag.rs`)
  - [x] **Mini-log overlay** (bottom-left of stage):
        shows last 5 pipeline log lines with timestamps and ok/warn colouring
  - [x] **Canvas bottom bar**:
    - [x] Face chips (face_001 · conf, toggleable on/off)
    - [x] Zoom controls: − / 100% / + / fit buttons

## Phase 10 — Right Inspector

- [x] Create `fcs-gui2/src/ui/inspector.rs`
  - [x] Tab bar: Crop | Output | Enhance
  - [x] **Mini stats grid** (always visible, 4 cells):
        Faces (peach), Selected (cyan), Detect ms (lime), Source resolution (rose)
  - [x] **Crop tab** panels (collapsible, port from `ui/config/crop/`):
    - [x] Panel 01 — Crop framing:
          Preset dropdown, aspect-ratio segmented control,
          face-height % slider, padding % slider,
          width/height pixel inputs, confidence-floor slider,
          padding fill color swatch + hex input
    - [x] Panel 02 — Positioning:
          mode segmented (Center/Thirds/Custom),
          offset X/Y px inputs,
          eye-line align toggle, auto-orient EXIF toggle
    - [x] Panel 03 — Crops ready:
          thumbnail cards (face#, filename, size, Save button)
    - [x] Panel 04 — Enhancement (collapsed by default):
          auto-color, skin-smooth, sharpen, red-eye-removal toggles
  - [x] **Output tab** (port from `ui/config/crop/output.rs` + `quality.rs`):
        format selector, JPEG quality slider, naming pattern input,
        output directory picker
  - [x] **Enhance tab** (port from `ui/config/enhancement.rs`):
        full sliders — exposure, brightness, contrast, saturation, sharpness,
        background blur toggle + radius

## Phase 11 — Status Bar

- [x] Create `fcs-gui2/src/ui/statusbar.rs`
  - [x] Status cell (Ready/Running dot + label)
  - [x] Model cell (YuNet 640 · ONNX)
  - [x] GPU cell (wgpu · backend · adapter)
  - [x] Batch progress cell (N / M)
  - [x] Spacer
  - [x] Progress bar (thin, peach→rose gradient)
  - [x] RAM usage cell
  - [x] GPU usage % cell
  - [x] Clock cell (current time HH:MM:SS)

## Phase 12 — Core Logic Wiring

- [x] Create `fcs-gui2/src/core/mod.rs` (re-export submodules)
- [x] Port `fcs-gui/src/core/detection.rs` → `fcs-gui2/src/core/detection.rs`
      (detector.detect → store detections in App2 state)
- [x] Port `fcs-gui/src/core/export.rs` → `fcs-gui2/src/core/export.rs`
      (GPU batch crop via `GpuBatchCropper` + save via `fcs_utils::output`)
- [x] Port `fcs-gui/src/core/cache.rs` → `fcs-gui2/src/core/cache.rs`
      (LRU crop-preview cache)
- [x] Port `fcs-gui/src/core/settings.rs` → `fcs-gui2/src/core/settings.rs`
      (load/save `AppSettings` to platform config dir)
- [x] Port `fcs-gui/src/core/quality.rs` → `fcs-gui2/src/core/quality.rs`
      (Laplacian variance quality gate)

## Phase 13 — Interaction / Rendering Ports

- [x] Port `fcs-gui/src/interaction/coords.rs` → `fcs-gui2/src/interaction/coords.rs`
      (image ↔ screen coordinate transforms for zoom/pan)
- [x] Port `fcs-gui/src/interaction/bbox_drag.rs` → `fcs-gui2/src/interaction/bbox_drag.rs`
      (drag handle logic for resizing crop boxes)
- [x] Port `fcs-gui/src/rendering/paint.rs` → `fcs-gui2/src/rendering/paint.rs`
      (egui Painter helpers for drawing boxes, landmarks, handles)

## Phase 14 — Custom Widgets

- [x] Create `fcs-gui2/src/ui/widgets.rs`
  - [x] `themed_slider()` — peach thumb, dark track, value label
  - [x] `segmented_control()` — grid of toggle buttons, one active (peach bg)
  - [x] `toggle_switch()` — pill switch, cyan when on
  - [x] `color_swatch_input()` — colored square + hex text input
  - [x] `panel_header()` — chevron + title + optional badge, click to collapse
  - [x] `badge()` — small monospace pill in ok/run/skip/err colours
  - [x] `face_chip()` — pill with check-mark and confidence label
  - [x] `gpu_pill()` — lime dot + adapter label

## Phase 15 — Polish & Integration Tests

- [ ] Verify `cargo check -p fcs-gui2` passes with zero errors
- [ ] Verify `cargo clippy -p fcs-gui2 -- -D warnings` is clean
- [ ] Smoke-test: run `cargo run -p fcs-gui2`, open an image, detect faces, export a crop
- [ ] Confirm theme exactly matches HTML palette (spot-check colours in running app)
- [ ] Confirm panels collapse/expand, tabs switch, sliders update settings
- [ ] Test drag-and-drop file loading
- [ ] Test batch queue progress display
- [ ] Update `TODO.md` (workspace-level) to note fcs-gui2 overhaul complete
