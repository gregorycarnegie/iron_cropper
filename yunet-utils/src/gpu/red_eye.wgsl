struct RedEyeUniforms {
    pixel_count : u32,
    threshold : f32,
    min_red : f32,
    _pad : f32,
};

@group(0) @binding(0)
var<storage, read_write> pixels : array<u32>;

@group(0) @binding(1)
var<uniform> params : RedEyeUniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.x;
    if (index >= params.pixel_count) {
        return;
    }

    let base = index * 4u;
    let r = f32(pixels[base + 0u]);
    let g = f32(pixels[base + 1u]);
    let b = f32(pixels[base + 2u]);
    let a = pixels[base + 3u];

    let avg_gb = (g + b) * 0.5 + 1e-6;
    let ratio = r / avg_gb;

    var out_r = r;
    if (ratio > params.threshold && r > params.min_red) {
        out_r = avg_gb;
    }

    pixels[base + 0u] = u32(clamp(round(out_r), 0.0, 255.0));
    pixels[base + 1u] = u32(clamp(round(g), 0.0, 255.0));
    pixels[base + 2u] = u32(clamp(round(b), 0.0, 255.0));
    pixels[base + 3u] = a;
}
