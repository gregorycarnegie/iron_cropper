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
    activation_mode: u32, // 0=None, 1=ReLU, 2=Sigmoid, 3=SiLU
};

@group(0) @binding(0) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_tensor: array<f32>;
@group(0) @binding(4) var<uniform> params: Conv2dUniforms;

const WORKGROUP_SIZE_X: u32 = 8u;
const WORKGROUP_SIZE_Y: u32 = 8u;

const ACT_NONE: u32 = 0u;
const ACT_RELU: u32 = 1u;
const ACT_SIGMOID: u32 = 2u;
const ACT_SILU: u32 = 3u;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1u)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Each thread computes 4 output pixels horizontally
    let ox_start = global_id.x * 4u;
    let oy = global_id.y;
    let oc = global_id.z;

    if ox_start >= params.output_width || oy >= params.output_height || oc >= params.output_channels {
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
    
    // Calculate start positions for 4 pixels
    // ox_vec = [ox, ox+1, ox+2, ox+3]
    let ox_vec = vec4<i32>(i32(ox_start), i32(ox_start) + 1, i32(ox_start) + 2, i32(ox_start) + 3);
    let start_y = i32(oy) * stride.y - pad.y;
    let start_x_vec = ox_vec * stride.x - pad.x;

    // Initialize accumulator with bias
    var acc = vec4<f32>(bias[oc]);

    var local_ic: u32 = 0u;
    loop {
        if local_ic >= group_in {
            break;
        }
        let channel = input_channel_base + local_ic;
        let channel_base = channel * input_plane;
        let weight_channel_base = oc * weights_per_out + local_ic * kernel_hw;

        var ky: u32 = 0u;
        loop {
            if ky >= params.kernel_height {
                break;
            }
            let iy = start_y + i32(ky);
            if iy < 0 || iy >= dims.y {
                ky = ky + 1u;
                continue;
            }
            let input_row_base = channel_base + u32(iy) * params.input_width;
            let weight_row_base = weight_channel_base + ky * params.kernel_width;

            var kx: u32 = 0u;
            loop {
                if kx >= params.kernel_width {
                    break;
                }
                
                // Load weight (scalar broadcast)
                let weight_index = weight_row_base + kx;
                let w = weights[weight_index];
                let w_vec = vec4<f32>(w);

                // Calculate input X coordinates for 4 pixels
                let ix_vec = start_x_vec + i32(kx);
                
                // Gather inputs
                var inputs = vec4<f32>(0.0);
                
                // Unroll manually for 4 components
                // Pixel 0
                if ix_vec.x >= 0 && ix_vec.x < dims.x {
                    inputs.x = input_tensor[input_row_base + u32(ix_vec.x)];
                }
                // Pixel 1
                if ix_vec.y >= 0 && ix_vec.y < dims.x {
                    inputs.y = input_tensor[input_row_base + u32(ix_vec.y)];
                }
                // Pixel 2
                if ix_vec.z >= 0 && ix_vec.z < dims.x {
                    inputs.z = input_tensor[input_row_base + u32(ix_vec.z)];
                }
                // Pixel 3
                if ix_vec.w >= 0 && ix_vec.w < dims.x {
                    inputs.w = input_tensor[input_row_base + u32(ix_vec.w)];
                }

                acc = fma(inputs, w_vec, acc);

                kx = kx + 1u;
            }
            ky = ky + 1u;
        }

        local_ic = local_ic + 1u;
    }

    // Apply fused activation
    if params.activation_mode == ACT_RELU {
        acc = max(acc, vec4<f32>(0.0));
    } else if params.activation_mode == ACT_SIGMOID {
        acc = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-acc));
    } else if params.activation_mode == ACT_SILU {
        acc = acc / (vec4<f32>(1.0) + exp(-acc));
    }

    // Write output
    let out_row_base = (oc * params.output_height + oy) * params.output_width;
    
    // Pixel 0
    if ox_start < params.output_width {
        output_tensor[out_row_base + ox_start] = acc.x;
    }
    // Pixel 1
    if ox_start + 1u < params.output_width {
        output_tensor[out_row_base + ox_start + 1u] = acc.y;
    }
    // Pixel 2
    if ox_start + 2u < params.output_width {
        output_tensor[out_row_base + ox_start + 2u] = acc.z;
    }
    // Pixel 3
    if ox_start + 3u < params.output_width {
        output_tensor[out_row_base + ox_start + 3u] = acc.w;
    }
}
