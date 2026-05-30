# Screenshot gallery — layout candidates

Preview this file (VS Code: <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>V</kbd>, or open on a
branch in GitHub) and pick the layout you like. Tell me the letter and I'll move it
into `README.md` and delete this file.

All options use the real `screenshots/*.png` files and stay within GitHub's HTML
allowlist (no CSS/lightbox is possible in a GitHub README — that only works if the
page is hosted on your own site).

---

## Option A — Hero + uniform 3-up row

Big hero, then one tidy symmetric row of the three side panels. Clean and modern.

<p align="center">
  <a href="screenshots/gui-main.png"><img src="screenshots/gui-main.png" alt="Desktop workspace" width="900"/></a><br/>
  <sub><strong>Desktop workspace</strong> — detection overlays, crop preview, queue, and export controls.</sub>
</p>

<table>
  <tr>
    <td width="33%" align="center" valign="top">
      <a href="screenshots/gui-crop-config.png"><img src="screenshots/gui-crop-config.png" alt="Crop setup" width="260"/></a>
    </td>
    <td width="33%" align="center" valign="top">
      <a href="screenshots/gui-enhancement.png"><img src="screenshots/gui-enhancement.png" alt="Enhancement controls" width="260"/></a>
    </td>
    <td width="33%" align="center" valign="top">
      <a href="screenshots/gui-batch.png"><img src="screenshots/gui-batch.png" alt="Batch queue" width="260"/></a>
    </td>
  </tr>
  <tr>
    <td align="center" valign="top"><sub><strong>Crop setup</strong><br/>Presets, sizing, shape, padding, positioning.</sub></td>
    <td align="center" valign="top"><sub><strong>Enhancement</strong><br/>Colour, sharpness, smoothing, blur.</sub></td>
    <td align="center" valign="top"><sub><strong>Batch queue</strong><br/>Multi-image status and export.</sub></td>
  </tr>
</table>

<details>
  <summary><strong>CLI automation example</strong></summary>
  <p align="center">
    <a href="screenshots/cli-example.png"><img src="screenshots/cli-example.png" alt="CLI output" width="900"/></a><br/>
    <sub>Detection metadata, crop coordinates, quality scoring, and processing summary.</sub>
  </p>
</details>

---

## Option B — Hero + 2×2 quadrant grid

Hero, then a balanced 2×2 grid that promotes the CLI shot into the grid (no
hidden `<details>`). Everything visible at a glance; more vertical space used.

<p align="center">
  <a href="screenshots/gui-main.png"><img src="screenshots/gui-main.png" alt="Desktop workspace" width="900"/></a><br/>
  <sub><strong>Desktop workspace</strong> — detection overlays, crop preview, queue, and export controls.</sub>
</p>

<table>
  <tr>
    <td width="50%" align="center" valign="top">
      <a href="screenshots/gui-crop-config.png"><img src="screenshots/gui-crop-config.png" alt="Crop setup" width="300"/></a><br/>
      <sub><strong>Crop setup</strong><br/>Presets, sizing, shape, padding, positioning.</sub>
    </td>
    <td width="50%" align="center" valign="top">
      <a href="screenshots/gui-enhancement.png"><img src="screenshots/gui-enhancement.png" alt="Enhancement controls" width="380"/></a><br/>
      <sub><strong>Enhancement</strong><br/>Colour, sharpness, smoothing, background blur.</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center" valign="top">
      <a href="screenshots/gui-batch.png"><img src="screenshots/gui-batch.png" alt="Batch queue" width="380"/></a><br/>
      <sub><strong>Batch queue</strong><br/>Multi-image status and export progress.</sub>
    </td>
    <td width="50%" align="center" valign="top">
      <a href="screenshots/cli-example.png"><img src="screenshots/cli-example.png" alt="CLI output" width="440"/></a><br/>
      <sub><strong>CLI automation</strong><br/>Batch-friendly output with metadata and quality scores.</sub>
    </td>
  </tr>
</table>

---

## Option C — Magazine layout (refined current)

Keeps the asymmetric feel of today's README but tidies widths and captions: tall
crop panel on the left, two stacked panels on the right.

<p align="center">
  <a href="screenshots/gui-main.png"><img src="screenshots/gui-main.png" alt="Desktop workspace" width="900"/></a><br/>
  <sub><strong>Desktop workspace</strong> — detection overlays, crop preview, queue, and export controls.</sub>
</p>

<table>
  <tr>
    <td width="40%" rowspan="2" align="center" valign="top">
      <a href="screenshots/gui-crop-config.png"><img src="screenshots/gui-crop-config.png" alt="Crop setup" width="260"/></a><br/>
      <sub><strong>Crop setup</strong><br/>Presets, target sizing, shape, padding, and positioning.</sub>
    </td>
    <td width="60%" align="center" valign="top">
      <a href="screenshots/gui-enhancement.png"><img src="screenshots/gui-enhancement.png" alt="Enhancement controls" width="340"/></a><br/>
      <sub><strong>Enhancement controls</strong><br/>Post-crop colour, sharpness, smoothing, and blur tuning.</sub>
    </td>
  </tr>
  <tr>
    <td align="center" valign="top">
      <a href="screenshots/gui-batch.png"><img src="screenshots/gui-batch.png" alt="Batch queue" width="340"/></a><br/>
      <sub><strong>Batch queue</strong><br/>Multi-image queue management and export progress.</sub>
    </td>
  </tr>
</table>

<details>
  <summary><strong>CLI automation example</strong></summary>
  <p align="center">
    <a href="screenshots/cli-example.png"><img src="screenshots/cli-example.png" alt="CLI output" width="900"/></a><br/>
    <sub>Detection metadata, crop coordinates, quality scoring, and processing summary.</sub>
  </p>
</details>

---

## Option D — Hero + collapsible sections (compact)

Only the hero shows by default; each panel lives in its own collapsible section.
Keeps the README short and lets readers expand just what they care about.

<p align="center">
  <a href="screenshots/gui-main.png"><img src="screenshots/gui-main.png" alt="Desktop workspace" width="900"/></a><br/>
  <sub><strong>Desktop workspace</strong> — detection overlays, crop preview, queue, and export controls.</sub>
</p>

<details>
  <summary><strong>Crop setup</strong> — presets, sizing, shape, padding, positioning</summary>
  <p align="center"><a href="screenshots/gui-crop-config.png"><img src="screenshots/gui-crop-config.png" alt="Crop setup" width="300"/></a></p>
</details>

<details>
  <summary><strong>Enhancement controls</strong> — colour, sharpness, smoothing, blur</summary>
  <p align="center"><a href="screenshots/gui-enhancement.png"><img src="screenshots/gui-enhancement.png" alt="Enhancement controls" width="420"/></a></p>
</details>

<details>
  <summary><strong>Batch queue</strong> — multi-image status and export</summary>
  <p align="center"><a href="screenshots/gui-batch.png"><img src="screenshots/gui-batch.png" alt="Batch queue" width="420"/></a></p>
</details>

<details>
  <summary><strong>CLI automation</strong> — batch-friendly terminal output</summary>
  <p align="center"><a href="screenshots/cli-example.png"><img src="screenshots/cli-example.png" alt="CLI output" width="900"/></a></p>
</details>

---

## Option E — Filmstrip (equal 4-up row)

Hero, then a single compact row of all four shots at equal width — most
"gallery-like" within GitHub's limits. Thumbnails are small; click to enlarge.

<p align="center">
  <a href="screenshots/gui-main.png"><img src="screenshots/gui-main.png" alt="Desktop workspace" width="900"/></a><br/>
  <sub><strong>Desktop workspace</strong> — detection overlays, crop preview, queue, and export controls.</sub>
</p>

<table>
  <tr>
    <td width="25%" align="center" valign="top"><a href="screenshots/gui-crop-config.png"><img src="screenshots/gui-crop-config.png" alt="Crop setup" width="200"/></a></td>
    <td width="25%" align="center" valign="top"><a href="screenshots/gui-enhancement.png"><img src="screenshots/gui-enhancement.png" alt="Enhancement controls" width="200"/></a></td>
    <td width="25%" align="center" valign="top"><a href="screenshots/gui-batch.png"><img src="screenshots/gui-batch.png" alt="Batch queue" width="200"/></a></td>
    <td width="25%" align="center" valign="top"><a href="screenshots/cli-example.png"><img src="screenshots/cli-example.png" alt="CLI output" width="200"/></a></td>
  </tr>
  <tr>
    <td align="center"><sub><strong>Crop setup</strong></sub></td>
    <td align="center"><sub><strong>Enhancement</strong></sub></td>
    <td align="center"><sub><strong>Batch queue</strong></sub></td>
    <td align="center"><sub><strong>CLI</strong></sub></td>
  </tr>
</table>
