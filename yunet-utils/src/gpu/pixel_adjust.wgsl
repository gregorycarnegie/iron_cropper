struct PixelAdjustUniforms {
    exposure_multiplier : f32,
    brightness_offset : f32,
    contrast_factor : f32,
    saturation : f32,
    pixel_count : u32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
};

@group(0) @binding(0)
var<storage, read_write> pixels : array<u32>;

@group(0) @binding(1)
var<uniform> params : PixelAdjustUniforms;

fn clamp255(value : f32) -> f32 {
    return clamp(value, 0.0, 255.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.x;
    if (index >= params.pixel_count) {
        return;
    }

    let base = index * 4u;
    var r = f32(pixels[base + 0u]);
    var g = f32(pixels[base + 1u]);
    var b = f32(pixels[base + 2u]);
    let a = pixels[base + 3u];

    // Exposure (multiply)
    r = clamp255(r * params.exposure_multiplier);
    g = clamp255(g * params.exposure_multiplier);
    b = clamp255(b * params.exposure_multiplier);

    // Brightness (offset)
    r = clamp255(r + params.brightness_offset);
    g = clamp255(g + params.brightness_offset);
    b = clamp255(b + params.brightness_offset);

    // Contrast (normalized around 0.5)
    let contrast = params.contrast_factor;
    if (abs(contrast - 1.0) > 0.00001) {
        r = clamp255(((r / 255.0 - 0.5) * contrast + 0.5) * 255.0);
        g = clamp255(((g / 255.0 - 0.5) * contrast + 0.5) * 255.0);
        b = clamp255(((b / 255.0 - 0.5) * contrast + 0.5) * 255.0);
    }

    // Saturation adjustment
    let saturation = params.saturation;
    if (abs(saturation - 1.0) > 0.00001) {
        let gray = 0.299 * r + 0.587 * g + 0.114 * b;
        let inv = 1.0 - saturation;
        r = clamp255(gray * inv + r * saturation);
        g = clamp255(gray * inv + g * saturation);
        b = clamp255(gray * inv + b * saturation);
    }

    pixels[base + 0u] = u32(round(r));
    pixels[base + 1u] = u32(round(g));
    pixels[base + 2u] = u32(round(b));
    pixels[base + 3u] = a;
}
