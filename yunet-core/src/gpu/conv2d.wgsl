struct Conv2dUniforms {
    input_width: u32,
    input_height: u32,
    input_channels: u32,
    output_width: u32,
    output_height: u32,
    output_channels: u32,
    kernel_width: u32,
    kernel_height: u32,
    stride_x: u32,
    stride_y: u32,
    pad_x: u32,
    pad_y: u32,
    groups: u32,
};

@group(0) @binding(0) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_tensor: array<f32>;
@group(0) @binding(4) var<uniform> params: Conv2dUniforms;

const WORKGROUP_SIZE_X: u32 = 8u;
const WORKGROUP_SIZE_Y: u32 = 8u;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1u)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ox = global_id.x;
    let oy = global_id.y;
    let oc = global_id.z;

    if (ox >= params.output_width || oy >= params.output_height || oc >= params.output_channels) {
        return;
    }

    let group_out = params.output_channels / params.groups;
    let group_in = params.input_channels / params.groups;
    let group_idx = oc / group_out;
    let input_channel_base = group_idx * group_in;

    let kernel_hw = params.kernel_height * params.kernel_width;
    let weights_per_out = group_in * kernel_hw;
    let input_plane = params.input_width * params.input_height;
    let stride = vec2<i32>(i32(params.stride_x), i32(params.stride_y));
    let pad = vec2<i32>(i32(params.pad_x), i32(params.pad_y));
    let dims = vec2<i32>(i32(params.input_width), i32(params.input_height));
    let start_xy = vec2<i32>(i32(ox), i32(oy)) * stride - pad;

    var acc = bias[oc];

    var local_ic: u32 = 0u;
    loop {
        if (local_ic >= group_in) {
            break;
        }
        let channel = input_channel_base + local_ic;
        let channel_base = channel * input_plane;
        let weight_channel_base = oc * weights_per_out + local_ic * kernel_hw;

        var ky: u32 = 0u;
        loop {
            if (ky >= params.kernel_height) {
                break;
            }
            let iy = start_xy.y + i32(ky);
            if (iy < 0 || iy >= dims.y) {
                ky = ky + 1u;
                continue;
            }
            let input_row_base = channel_base + u32(iy) * params.input_width;
            let weight_row_base = weight_channel_base + ky * params.kernel_width;

            var kx: u32 = 0u;
            loop {
                if (kx >= params.kernel_width) {
                    break;
                }
                let ix = start_xy.x + i32(kx);
                if (ix >= 0 && ix < dims.x) {
                    let input_index = input_row_base + u32(ix);
                    let weight_index = weight_row_base + kx;
                    acc = fma(input_tensor[input_index], weights[weight_index], acc);
                }

                kx = kx + 1u;
            }
            ky = ky + 1u;
        }

        local_ic = local_ic + 1u;
    }

    let out_index = (oc * params.output_height + oy) * params.output_width + ox;
    output_tensor[out_index] = acc;
}
