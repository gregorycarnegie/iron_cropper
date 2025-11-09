struct HistogramUniforms {
    pixel_count : u32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
};

@group(0) @binding(0)
var<storage, read> pixels : array<u32>;

@group(0) @binding(1)
var<storage, read_write> histogram : array<atomic<u32>>;

@group(0) @binding(2)
var<uniform> hist_params : HistogramUniforms;

@compute @workgroup_size(256)
fn build_histogram(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.x;
    if (index >= hist_params.pixel_count) {
        return;
    }

    let base = index * 4u;
    let r = pixels[base + 0u];
    let g = pixels[base + 1u];
    let b = pixels[base + 2u];

    let r_idx = r & 0xFFu;
    let g_idx = g & 0xFFu;
    let b_idx = b & 0xFFu;

    atomicAdd(&histogram[r_idx + 0u], 1u);
    atomicAdd(&histogram[g_idx + 256u], 1u);
    atomicAdd(&histogram[b_idx + 512u], 1u);
}

struct LutUniforms {
    pixel_count : u32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
};

@group(0) @binding(0)
var<storage, read_write> pixels_apply : array<u32>;

@group(0) @binding(1)
var<storage, read> lut : array<u32>;

@group(0) @binding(2)
var<uniform> lut_params : LutUniforms;

@compute @workgroup_size(256)
fn apply_equalization(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.x;
    if (index >= lut_params.pixel_count) {
        return;
    }

    let base = index * 4u;
    let r = pixels_apply[base + 0u];
    let g = pixels_apply[base + 1u];
    let b = pixels_apply[base + 2u];
    let a = pixels_apply[base + 3u];

    let r_idx = r & 0xFFu;
    let g_idx = g & 0xFFu;
    let b_idx = b & 0xFFu;

    let r_new = lut[r_idx + 0u];
    let g_new = lut[g_idx + 256u];
    let b_new = lut[b_idx + 512u];

    pixels_apply[base + 0u] = r_new & 0xFFu;
    pixels_apply[base + 1u] = g_new & 0xFFu;
    pixels_apply[base + 2u] = b_new & 0xFFu;
    pixels_apply[base + 3u] = a;
}
