struct ShapeMaskUniforms {
    width : u32,
    height : u32,
    point_count : u32,
    samples : u32,
};

@group(0) @binding(0)
var<storage, read_write> pixels : array<u32>;

@group(0) @binding(1)
var<storage, read> polygon_points : array<vec2<f32>>;

@group(0) @binding(2)
var<uniform> params : ShapeMaskUniforms;

fn point_in_polygon(p : vec2<f32>) -> bool {
    var inside = false;
    var i : u32 = 0u;
    var j : u32 = params.point_count - 1u;
    loop {
        if (i >= params.point_count) {
            break;
        }
        let pi = polygon_points[i];
        let pj = polygon_points[j];
        let intersects =
            ((pi.y > p.y) != (pj.y > p.y)) &&
            (p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x);
        if (intersects) {
            inside = !inside;
        }
        j = i;
        i = i + 1u;
    }
    return inside;
}

fn coverage_for_pixel(x : u32, y : u32) -> f32 {
    let samples = max(params.samples, 1u);
    let inv = 1.0 / f32(samples);
    var covered = 0.0;
    for (var i : u32 = 0u; i < samples; i = i + 1u) {
        let offset = vec2<f32>(f32(i & 1u), f32((i >> 1u) & 1u)) * 0.5 + vec2<f32>(0.25, 0.25);
        let sample = vec2<f32>(f32(x), f32(y)) + offset;
        if (point_in_polygon(sample)) {
            covered = covered + inv;
        }
    }
    return covered;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height || params.point_count < 3u) {
        return;
    }

    let dst = (y * params.width + x) * 4u;
    let coverage = coverage_for_pixel(x, y);

    if (coverage <= 0.0) {
        pixels[dst + 0u] = 0u;
        pixels[dst + 1u] = 0u;
        pixels[dst + 2u] = 0u;
        pixels[dst + 3u] = 0u;
    } else if (coverage >= 0.999) {
        // Keep original pixel but ensure alpha is opaque
        pixels[dst + 3u] = 255u;
    } else {
        let alpha = clamp(coverage * 255.0, 0.0, 255.0);
        pixels[dst + 3u] = u32(round(alpha));
        pixels[dst + 0u] = u32(clamp(round(f32(pixels[dst + 0u]) * coverage), 0.0, 255.0));
        pixels[dst + 1u] = u32(clamp(round(f32(pixels[dst + 1u]) * coverage), 0.0, 255.0));
        pixels[dst + 2u] = u32(clamp(round(f32(pixels[dst + 2u]) * coverage), 0.0, 255.0));
    }
}
