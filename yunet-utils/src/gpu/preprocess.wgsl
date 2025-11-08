struct PreprocessUniforms {
    src_size : vec2<u32>,
    dst_size : vec2<u32>,
};

@group(0) @binding(0)
var source_tex : texture_2d<f32>;

@group(0) @binding(1)
var source_sampler : sampler;

@group(0) @binding(2)
var<storage, read_write> output_buffer : array<f32>;

@group(0) @binding(3)
var<uniform> uniforms : PreprocessUniforms;

fn chw_index(x : u32, y : u32, c : u32, width : u32, height : u32) -> u32 {
    return c * width * height + y * width + x;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if (global_id.x >= uniforms.dst_size.x || global_id.y >= uniforms.dst_size.y) {
        return;
    }

    let dst_size_f = vec2<f32>(vec2<u32>(uniforms.dst_size));
    let uv = (vec2<f32>(global_id.xy) + vec2<f32>(0.5, 0.5)) / dst_size_f;
    let color = textureSampleLevel(source_tex, source_sampler, uv, 0.0);

    let width = uniforms.dst_size.x;
    let height = uniforms.dst_size.y;
    let idx = chw_index(global_id.x, global_id.y, 0u, width, height);
    let plane_size = width * height;

    // Convert from normalized [0,1] floats back to 0-255 range and swap RGB->BGR.
    output_buffer[idx] = color.b * 255.0;
    output_buffer[idx + plane_size] = color.g * 255.0;
    output_buffer[idx + plane_size * 2u] = color.r * 255.0;
}
