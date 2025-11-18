struct BatchNormUniforms {
    width: u32,
    height: u32,
    channels: u32,
    epsilon: f32,
};

@group(0) @binding(0) var<storage, read_write> tensor: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read> mean: array<f32>;
@group(0) @binding(4) var<storage, read> variance: array<f32>;
@group(0) @binding(5) var<uniform> params: BatchNormUniforms;

const BN_WORKGROUP_X: u32 = 8u;
const BN_WORKGROUP_Y: u32 = 8u;

@compute @workgroup_size(BN_WORKGROUP_X, BN_WORKGROUP_Y, 1u)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let channel = global_id.z;

    if (x >= params.width || y >= params.height || channel >= params.channels) {
        return;
    }

    let index = (channel * params.height + y) * params.width + x;
    let value = tensor[index];
    let mean_value = mean[channel];
    let var_value = variance[channel];
    let gamma_value = gamma[channel];
    let beta_value = beta[channel];
    let inv_std = inverseSqrt(var_value + params.epsilon);
    let normalized = (value - mean_value) * inv_std;
    tensor[index] = fma(normalized, gamma_value, beta_value);
}
