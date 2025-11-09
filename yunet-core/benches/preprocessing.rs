use std::{hint::black_box, sync::Arc};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use image::DynamicImage;
use yunet_core::preprocess::{
    CpuPreprocessor, InputSize, PreprocessConfig, Preprocessor, WgpuPreprocessor,
    preprocess_image_with,
};
use yunet_utils::{
    config::ResizeQuality,
    fixture_path,
    gpu::{GpuAvailability, GpuContext, GpuContextOptions},
    load_fixture_image,
};

const FIXTURE_IMAGE: &str = "images/006.jpg";
const INPUT_SIZE: InputSize = InputSize::new(640, 640);

fn load_benchmark_image() -> DynamicImage {
    load_fixture_image(FIXTURE_IMAGE).expect("fixture image is required for preprocessing bench")
}

fn preprocess_configs() -> Vec<(&'static str, PreprocessConfig)> {
    vec![
        (
            "quality",
            PreprocessConfig {
                input_size: INPUT_SIZE,
                resize_quality: ResizeQuality::Quality,
            },
        ),
        (
            "speed",
            PreprocessConfig {
                input_size: INPUT_SIZE,
                resize_quality: ResizeQuality::Speed,
            },
        ),
    ]
}

fn benchmark_preprocessing(c: &mut Criterion) {
    let image = load_benchmark_image();
    let image_path_buf =
        fixture_path(FIXTURE_IMAGE).expect("fixture path must exist for preprocessing bench");
    let image_path = image_path_buf.as_path();

    let configs = preprocess_configs();
    let preprocessors = available_preprocessors();

    let mut dyn_group = c.benchmark_group("preprocess_dynamic_image");
    for (backend, preprocessor) in preprocessors.iter() {
        for (label, config) in configs.iter() {
            dyn_group.bench_with_input(
                BenchmarkId::new(format!("{backend}_dynamic"), label),
                config,
                |b, cfg| {
                    b.iter(|| {
                        preprocessor
                            .preprocess(black_box(&image), cfg)
                            .expect("dynamic preprocessing should succeed");
                    });
                },
            );
        }
    }
    dyn_group.finish();

    let mut file_group = c.benchmark_group("preprocess_image");
    for (backend, preprocessor) in preprocessors.iter() {
        for (label, config) in configs.iter() {
            file_group.bench_with_input(
                BenchmarkId::new(format!("{backend}_from_disk"), label),
                config,
                |b, cfg| {
                    b.iter(|| {
                        preprocess_image_with(preprocessor.as_ref(), black_box(image_path), cfg)
                            .expect("file preprocessing should succeed");
                    });
                },
            );
        }
    }
    file_group.finish();
}

fn available_preprocessors() -> Vec<(&'static str, Arc<dyn Preprocessor>)> {
    let mut list: Vec<(&'static str, Arc<dyn Preprocessor>)> =
        vec![("cpu", Arc::new(CpuPreprocessor))];

    if let Some(wgpu_pre) = build_gpu_preprocessor() {
        list.push(("gpu", Arc::new(wgpu_pre)));
    } else {
        println!("GPU preprocessing benchmark unavailable (no compatible adapter).");
    }

    list
}

fn build_gpu_preprocessor() -> Option<WgpuPreprocessor> {
    match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => match WgpuPreprocessor::new(ctx.clone()) {
            Ok(pre) => {
                println!(
                    "Benchmarking GPU preprocessing on '{}' ({:?})",
                    ctx.adapter_info().name,
                    ctx.adapter_info().backend
                );
                Some(pre)
            }
            Err(err) => {
                println!("Failed to initialize GPU preprocessor for benchmark: {err}");
                None
            }
        },
        GpuAvailability::Disabled { reason } => {
            println!("GPU benchmark disabled: {reason}");
            None
        }
        GpuAvailability::Unavailable { error } => {
            println!("GPU benchmark unavailable: {error}");
            None
        }
    }
}

criterion_group!(benches, benchmark_preprocessing);
criterion_main!(benches);
