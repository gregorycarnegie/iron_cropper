use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use image::DynamicImage;
use yunet_core::preprocess::{
    InputSize, PreprocessConfig, preprocess_dynamic_image, preprocess_image,
};
use yunet_utils::{config::ResizeQuality, fixture_path, load_fixture_image};

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

    let mut dyn_group = c.benchmark_group("preprocess_dynamic_image");
    for (label, config) in configs.iter() {
        dyn_group.bench_with_input(BenchmarkId::new("dynamic", label), config, |b, cfg| {
            b.iter(|| {
                preprocess_dynamic_image(black_box(&image), cfg)
                    .expect("dynamic preprocessing should succeed");
            });
        });
    }
    dyn_group.finish();

    let mut file_group = c.benchmark_group("preprocess_image");
    for (label, config) in configs.iter() {
        file_group.bench_with_input(BenchmarkId::new("from_disk", label), config, |b, cfg| {
            b.iter(|| {
                preprocess_image(black_box(image_path), cfg)
                    .expect("file preprocessing should succeed");
            });
        });
    }
    file_group.finish();
}

criterion_group!(benches, benchmark_preprocessing);
criterion_main!(benches);
