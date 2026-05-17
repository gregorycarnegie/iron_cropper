# Built-in Presets

Face Crop Studio ships with two families of presets baked into the binary. You don't
need to install anything extra to use them — they appear in both the GUI dropdowns and
as `--preset` / `--enhancement-preset` flags on the CLI.

## Crop size presets

Defined in `fcs-core/src/presets.rs`. Selected via the GUI preset dropdown or the CLI
`--preset` flag.

| Preset | Dimensions (W×H) | Use case |
| --- | --- | --- |
| **LinkedIn** | 400 × 400 | Square professional profile photo |
| **Passport** | 413 × 531 | Standard passport photo dimensions |
| **Instagram** | 1080 × 1080 | Square social post |
| **ID Card** | 332 × 498 | Government / corporate ID badges |
| **Avatar** | 512 × 512 | Small square avatar (forums, chat) |
| **Headshot** | 600 × 800 | Vertical 3:4 headshot |
| **Custom** | user-defined | Any width/height entered manually |

All seven presets honor the `--face-height-pct` and `--positioning-mode` flags, so e.g.
the LinkedIn preset can be tuned for a 60% face height with center positioning, or 70%
with rule-of-thirds positioning, without changing the output dimensions.

## Enhancement presets

Defined in `fcs-utils/src/enhance.rs`. Selected via the GUI enhancement dropdown or the
CLI `--enhancement-preset` flag.

### Natural

Gentle baseline correction. Use when the source photo already looks good and you only
want a light polish.

| Parameter | Value |
| --- | --- |
| Auto color correction | enabled |
| Exposure | +0.1 stops |
| Contrast | 1.10× |
| Saturation | 1.05× |
| Sharpness | 0.20 |

### Vivid

Punchier output — warmer exposure, more saturation, stronger sharpening. Best for
casual or social-media-bound portraits where you want the image to "pop".

| Parameter | Value |
| --- | --- |
| Exposure | +0.3 stops |
| Brightness | +10 |
| Contrast | 1.25× |
| Saturation | 1.30× |
| Unsharp mask amount | 0.9 (radius 1.2) |
| Sharpness | 0.50 |

### Professional

Balanced tone with strong detail enhancement. Best for headshots and ID-style portraits
where skin tone realism matters more than visual impact.

| Parameter | Value |
| --- | --- |
| Auto color correction | enabled |
| Exposure | +0.2 stops |
| Contrast | 1.15× |
| Saturation | 1.05× |
| Unsharp mask amount | 1.2 |
| Sharpness | 0.80 |

## CLI examples

```bash
# Crop with the LinkedIn preset and apply the Natural enhancement preset
fcs-cli --input photo.jpg --crop --preset LinkedIn \
    --enhance --enhancement-preset natural --output-dir out/

# Passport photo with custom face height and the Professional enhancement preset
fcs-cli --input photo.jpg --crop --preset Passport \
    --face-height-pct 75 --positioning-mode center \
    --enhance --enhancement-preset professional --output-dir out/
```

## Combining presets

The crop and enhancement presets are independent — any of the seven crop sizes can be
paired with any of the three enhancement looks (or no enhancement at all). The GUI's
"Reset to defaults" button under enhancement settings is useful when switching between
the three enhancement presets to start from a clean baseline each time.
