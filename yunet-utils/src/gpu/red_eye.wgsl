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

    let pixel = pixels[index];
    let r = f32(pixel & 0xFFu);
    let g = f32((pixel >> 8u) & 0xFFu);
    let b = f32((pixel >> 16u) & 0xFFu);
    let a = (pixel >> 24u) & 0xFFu;

    let avg_gb = fma((g + b), 0.5, 1e-6);
    let ratio = r / avg_gb;

    let out_r = select(r, avg_gb, ratio > params.threshold && r > params.min_red);
    let r_new = u32(clamp(round(out_r), 0.0, 255.0)) & 0xFFu;
    let g_new = u32(clamp(round(g), 0.0, 255.0)) & 0xFFu;
    let b_new = u32(clamp(round(b), 0.0, 255.0)) & 0xFFu;
    pixels[index] = r_new | (g_new << 8u) | (b_new << 16u) | (a << 24u);
}
