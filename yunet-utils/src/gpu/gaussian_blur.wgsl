const MAX_KERNEL_SIZE : u32 = 25u;

struct BlurUniforms {
    width : u32,
    height : u32,
    radius : u32,
    direction : u32,
    kernel_size : u32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
};

@group(0) @binding(0)
var<storage, read> input_pixels : array<u32>;

@group(0) @binding(1)
var<storage, read_write> output_pixels : array<u32>;

@group(0) @binding(2)
var<uniform> params : BlurUniforms;

@group(0) @binding(3)
var<storage, read> weights : array<f32, MAX_KERNEL_SIZE>;

fn unpack_pixel(value : u32) -> vec4<f32> {
    return vec4<f32>(
        f32(value & 0xFFu),
        f32((value >> 8u) & 0xFFu),
        f32((value >> 16u) & 0xFFu),
        f32((value >> 24u) & 0xFFu),
    );
}

fn pack_pixel(color : vec4<f32>) -> u32 {
    let clamped = clamp(round(color), vec4<f32>(0.0), vec4<f32>(255.0));
    let r = u32(clamped.x) & 0xFFu;
    let g = u32(clamped.y) & 0xFFu;
    let b = u32(clamped.z) & 0xFFu;
    let a = u32(clamped.w) & 0xFFu;
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

fn load_pixel(idx : u32) -> vec4<f32> {
    return unpack_pixel(input_pixels[idx]);
}

fn store_pixel(idx : u32, value : vec4<f32>) {
    output_pixels[idx] = pack_pixel(value);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    let dst_index = y * params.width + x;
    let center = load_pixel(dst_index);
    var accum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    let alpha = center.w;

    let radius = i32(params.radius);
    for (var offset : i32 = -radius; offset <= radius; offset = offset + 1) {
        let kernel_index = u32(offset + radius);
        let weight = weights[kernel_index];

        var sample_x = i32(x);
        var sample_y = i32(y);

        if (params.direction == 0u) {
            sample_x = sample_x + offset;
        } else {
            sample_y = sample_y + offset;
        }

        sample_x = clamp(sample_x, 0, i32(params.width) - 1);
        sample_y = clamp(sample_y, 0, i32(params.height) - 1);

        let sample_idx = u32(sample_y) * params.width + u32(sample_x);
        let sample = load_pixel(sample_idx);
        accum = accum + sample.xyz * weight;
        weight_sum = weight_sum + weight;
    }

    let color = accum / max(weight_sum, 1e-6);
    store_pixel(dst_index, vec4<f32>(color, alpha));
}
