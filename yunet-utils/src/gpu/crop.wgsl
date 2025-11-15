struct PixelBuffer {
    data: array<u32>,
};

struct CropUniforms {
    src_width: u32,
    src_height: u32,
    crop_x: u32,
    crop_y: u32,
    crop_width: u32,
    crop_height: u32,
    dst_width: u32,
    dst_height: u32,
};

@group(0) @binding(0)
var<storage, read> source_pixels: PixelBuffer;

@group(0) @binding(1)
var<storage, read_write> dest_pixels: PixelBuffer;

@group(0) @binding(2)
var<uniform> params: CropUniforms;

fn load_pixel(x: u32, y: u32) -> vec4<f32> {
    let clamped_x = min(x, params.src_width - 1u);
    let clamped_y = min(y, params.src_height - 1u);
    let index = clamped_y * params.src_width + clamped_x;
    let value = source_pixels.data[index];
    return vec4<f32>(
        f32(value & 0xFFu),
        f32((value >> 8u) & 0xFFu),
        f32((value >> 16u) & 0xFFu),
        f32((value >> 24u) & 0xFFu),
    );
}

fn store_pixel(x: u32, y: u32, color: vec4<f32>) {
    let idx = y * params.dst_width + x;
    let clamped = clamp(round(color), vec4<f32>(0.0), vec4<f32>(255.0));
    let r = u32(clamped.r) & 0xFFu;
    let g = u32(clamped.g) & 0xFFu;
    let b = u32(clamped.b) & 0xFFu;
    let a = u32(clamped.a) & 0xFFu;
    dest_pixels.data[idx] = r | (g << 8u) | (b << 16u) | (a << 24u);
}

fn safe_span(size: u32) -> f32 {
    if size <= 1u {
        return 0.0;
    }
    return f32(size - 1u);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= params.dst_width || global_id.y >= params.dst_height {
        return;
    }

    let dst_span_x = safe_span(params.dst_width);
    let dst_span_y = safe_span(params.dst_height);
    let crop_span_x = safe_span(params.crop_width);
    let crop_span_y = safe_span(params.crop_height);

    var fx = 0.0;
    if dst_span_x > 0.0 {
        fx = f32(global_id.x) / dst_span_x;
    }
    var fy = 0.0;
    if dst_span_y > 0.0 {
        fy = f32(global_id.y) / dst_span_y;
    }

    let src_x = f32(params.crop_x) + fx * crop_span_x;
    let src_y = f32(params.crop_y) + fy * crop_span_y;

    let base_x = u32(floor(src_x));
    let base_y = u32(floor(src_y));
    let next_x = min(base_x + 1u, params.crop_x + params.crop_width - 1u);
    let next_y = min(base_y + 1u, params.crop_y + params.crop_height - 1u);

    let tx = fract(src_x);
    let ty = fract(src_y);

    let p00 = load_pixel(base_x, base_y);
    let p10 = load_pixel(next_x, base_y);
    let p01 = load_pixel(base_x, next_y);
    let p11 = load_pixel(next_x, next_y);

    let top = mix(p00, p10, tx);
    let bottom = mix(p01, p11, tx);
    let color = mix(top, bottom, ty);

    store_pixel(global_id.x, global_id.y, color);
}
