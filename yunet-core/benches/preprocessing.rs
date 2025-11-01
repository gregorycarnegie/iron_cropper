use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use image::DynamicImage;
use image::{ImageBuffer, Rgb};
use std::hint::black_box;
use yunet_core::preprocess::{InputSize, PreprocessConfig, preprocess_dynamic_image};

fn create_test_image(width: u32, height: u32) -> DynamicImage {
    let mut img = ImageBuffer::<Rgb<u8>, _>::new(width, height);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let r = ((x + y) % 256) as u8;
        let g = ((x * 2 + y) % 256) as u8;
        let b = ((x + y * 2) % 256) as u8;
        *pixel = Rgb([r, g, b]);
    }
    DynamicImage::ImageRgb8(img)
}

fn benchmark_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("preprocessing");

    // Test different image sizes
    for &size in &[320, 640, 1280] {
        let image = create_test_image(size, size);
        let config = PreprocessConfig {
            input_size: InputSize::new(640, 640),
        };

        group.bench_with_input(
            BenchmarkId::new("preprocess", size),
            &(&image, &config),
            |b, (img, cfg)| {
                b.iter(|| {
                    preprocess_dynamic_image(black_box(*img), black_box(*cfg))
                        .expect("preprocessing should succeed")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_preprocessing);
criterion_main!(benches);
