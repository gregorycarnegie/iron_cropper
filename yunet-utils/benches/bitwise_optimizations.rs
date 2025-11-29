use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use image::Rgba;
use std::hint::black_box;
use yunet_utils::image_utils::rgb_to_bgr_chw;

/// Benchmark for RGB→BGR CHW conversion (tests x * 3 optimization)
fn bench_rgb_to_bgr_chw(c: &mut Criterion) {
    let mut group = c.benchmark_group("rgb_to_bgr_chw");

    for size in [256, 512, 640, 1024].iter() {
        let img = image::RgbaImage::from_pixel(*size, *size, Rgba([128, 128, 128, 255]));
        let rgb_img = image::DynamicImage::ImageRgba8(img).to_rgb8();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", size, size)),
            size,
            |b, _| {
                b.iter(|| {
                    let _result = rgb_to_bgr_chw(black_box(&rgb_img));
                });
            },
        );
    }
    group.finish();
}

/// Benchmark for RGBA pixel indexing in saturation (tests lane * 4 optimization)
fn bench_rgba_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("rgba_indexing");

    for size in [256, 512, 640].iter() {
        let data = vec![128u8; size * size * 4];

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}_multiply", size, size)),
            &data,
            |b, d| {
                b.iter(|| {
                    let mut idx = 0;
                    let data_slice = black_box(d);
                    while idx + 16 <= data_slice.len() {
                        // Baseline: multiplication
                        for lane in 0..4 {
                            let base = idx + lane * 4;
                            let _val = data_slice[base];
                        }
                        idx += 16;
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}_bitshift", size, size)),
            &data,
            |b, d| {
                b.iter(|| {
                    let mut idx = 0;
                    let data_slice = black_box(d);
                    while idx + 16 <= data_slice.len() {
                        // Optimized: bit shift
                        for lane in 0..4 {
                            let base = idx + (lane << 2);
                            let _val = data_slice[base];
                        }
                        idx += 16;
                    }
                });
            },
        );
    }
    group.finish();
}

/// Benchmark for alignment function (tests power-of-2 alignment)
fn bench_align_to(c: &mut Criterion) {
    let mut group = c.benchmark_group("align_to");

    // Baseline modulo-based alignment
    fn align_to_modulo(value: usize, alignment: usize) -> usize {
        if value.is_multiple_of(alignment) {
            value
        } else {
            value + (alignment - (value % alignment))
        }
    }

    // Optimized bitwise alignment
    fn align_to_bitwise(value: usize, alignment: usize) -> usize {
        (value + alignment - 1) & !(alignment - 1)
    }

    let test_values = vec![100, 253, 500, 1000, 1920, 3840];
    let alignment = 256; // Typical WGPU alignment

    group.bench_function("modulo", |b| {
        b.iter(|| {
            for &val in &test_values {
                let _result = align_to_modulo(black_box(val), black_box(alignment));
            }
        });
    });

    group.bench_function("bitwise", |b| {
        b.iter(|| {
            for &val in &test_values {
                let _result = align_to_bitwise(black_box(val), black_box(alignment));
            }
        });
    });

    group.finish();
}

/// Benchmark for row stride calculation (tests w * 4 optimization)
fn bench_row_stride_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_stride");

    for width in [256, 512, 640, 1024, 1920].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_multiply", width)),
            width,
            |b, &w| {
                b.iter(|| {
                    // Baseline: multiplication
                    let _stride = black_box(w as usize * 4);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_bitshift", width)),
            width,
            |b, &w| {
                b.iter(|| {
                    // Optimized: bit shift
                    let _stride = black_box((w as usize) << 2);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_rgb_to_bgr_chw,
    bench_rgba_indexing,
    bench_align_to,
    bench_row_stride_calculation
);
criterion_main!(benches);
