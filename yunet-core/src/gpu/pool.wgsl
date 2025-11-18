struct PoolUniforms {
    input_width: u32,
    input_height: u32,
    channels: u32,
    output_width: u32,
    output_height: u32,
    kernel: u32,
    stride: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_tensor: array<f32>;
@group(0) @binding(2) var<uniform> params: PoolUniforms;

fn index(c: u32, y: u32, x: u32) -> u32 {
    let channel_height = c * params.input_height + y;
    let flat_index = channel_height * params.input_width + x;
    return flat_index;
}

const WG_SIZE_X: u32 = 8u;
const WG_SIZE_Y: u32 = 8u;

@compute @workgroup_size(WG_SIZE_X, WG_SIZE_Y, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ox = global_id.x;
    let oy = global_id.y;
    let oc = global_id.z;

    if (ox >= params.output_width || oy >= params.output_height || oc >= params.channels) {
        return;
    }

    var max_val = -3.402823466e+38;
    for (var ky: u32 = 0u; ky < params.kernel; ky = ky + 1u) {
        for (var kx: u32 = 0u; kx < params.kernel; kx = kx + 1u) {
            let ix = i32(ox * params.stride + kx) - i32(params.pad);
            let iy = i32(oy * params.stride + ky) - i32(params.pad);
            if (ix < 0 || iy < 0 || ix >= i32(params.input_width) || iy >= i32(params.input_height)) {
                continue;
            }
            let value = input_tensor[index(oc, u32(iy), u32(ix))];
            max_val = select(max_val, value, value > max_val);
        }
    }

    let channel_offset = oc * params.output_height + oy;
    let out_index = channel_offset * params.output_width + ox;
    output_tensor[out_index] = max_val;
}
