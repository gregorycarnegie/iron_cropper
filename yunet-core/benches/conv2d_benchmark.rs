use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::sync::Arc;
use yunet_core::gpu::ops::{Conv2dChannels, Conv2dConfig, GpuInferenceOps, SpatialDims};
use yunet_utils::gpu::{GpuAvailability, GpuContext, GpuContextOptions};

fn setup_gpu() -> Option<Arc<GpuContext>> {
    match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => Some(ctx),
        _ => None,
    }
}

fn benchmark_conv2d_standard_vs_vec4(c: &mut Criterion) {
    let Some(ctx) = setup_gpu() else {
        eprintln!("Skipping GPU benchmark (no adapter available)");
        return;
    };

    let ops = GpuInferenceOps::new(ctx).expect("create ops");

    // Typical YuNet layer parameters
    let configs = vec![
        // (name, batch, in_ch, out_ch, width, height, kernel, stride, pad, groups)
        ("stage0_conv", 1, 3, 16, 640, 640, 3, 2, 1, 1),
        ("stage1_depth", 1, 32, 32, 160, 160, 3, 1, 1, 32),
        ("stage2_point", 1, 32, 64, 160, 160, 1, 1, 0, 1),
        ("head_depth", 1, 64, 64, 40, 40, 3, 1, 1, 64),
    ];

    for (name, batch, in_ch, out_ch, width, height, kernel, stride, pad, groups) in configs {
        let input_len = (batch * in_ch * width * height) as usize;
        let input: Vec<f32> = (0..input_len).map(|i| (i % 100) as f32 / 100.0).collect();

        let weight_len = if groups == 1 {
            (out_ch * in_ch * kernel * kernel) as usize
        } else {
            (out_ch * kernel * kernel) as usize
        };
        let weights: Vec<f32> = (0..weight_len).map(|i| (i % 100) as f32 / 100.0).collect();

        let bias: Vec<f32> = (0..out_ch).map(|i| (i % 100) as f32 / 100.0).collect();

        let config = Conv2dConfig::new(
            batch,
            Conv2dChannels::new(in_ch, out_ch),
            SpatialDims::new(width, height),
            SpatialDims::new(kernel, kernel),
            SpatialDims::new(stride, stride),
            SpatialDims::new(pad, pad),
            groups,
            Some(yunet_core::gpu::ops::ActivationKind::Relu),
        )
        .unwrap();

        let input_gpu = ops
            .upload_tensor(config.input_shape_dims(), &input, Some("bench_input"))
            .unwrap();
        let weight_gpu = ops
            .upload_tensor(config.weight_shape_dims(), &weights, Some("bench_weights"))
            .unwrap();
        let bias_gpu = ops
            .upload_tensor(config.bias_shape_dims(), &bias, Some("bench_bias"))
            .unwrap();

        let mut group = c.benchmark_group(name);

        // Benchmark standard shader
        group.bench_function("standard", |b| {
            b.iter(|| {
                let output = ops
                    .conv2d_tensor(
                        black_box(&input_gpu),
                        black_box(&weight_gpu),
                        black_box(&bias_gpu),
                        black_box(&config),
                    )
                    .unwrap();
                // Force GPU completion by downloading (in real usage, data stays on GPU)
                black_box(output.to_vec().unwrap());
            });
        });

        // Benchmark vectorized shader
        group.bench_function("vec4", |b| {
            b.iter(|| {
                let output = ops
                    .conv2d_vec4_tensor(
                        black_box(&input_gpu),
                        black_box(&weight_gpu),
                        black_box(&bias_gpu),
                        black_box(&config),
                    )
                    .unwrap();
                // Force GPU completion by downloading
                black_box(output.to_vec().unwrap());
            });
        });

        group.finish();
    }
}

criterion_group!(benches, benchmark_conv2d_standard_vs_vec4);
criterion_main!(benches);
