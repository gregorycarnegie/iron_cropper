//! Preprocessing benchmarking functionality.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use image::DynamicImage;
use log::{info, warn};
use serde::Serialize;
use yunet_core::{CpuPreprocessor, PreprocessConfig, Preprocessor, WgpuPreprocessor};
use yunet_utils::{GpuContext, load_image};

use crate::input::ProcessingItem;

#[derive(Serialize)]
pub struct PreprocessBenchmarkSummary {
    label: String,
    samples: usize,
    iterations_per_sample: usize,
    total_ms: f64,
    avg_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

impl PreprocessBenchmarkSummary {
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = label.to_string();
        self
    }
}

pub fn run_preprocess_benchmark(
    items: &[ProcessingItem],
    config: &PreprocessConfig,
    gpu_context: Option<&Arc<GpuContext>>,
) -> Result<()> {
    anyhow::ensure!(
        !items.is_empty(),
        "preprocess benchmark requires at least one input image"
    );

    info!(
        "Running preprocessing benchmark across {} image(s)...",
        items.len()
    );

    let images = load_benchmark_images(items)?;
    let iterations = 3;

    let cpu = CpuPreprocessor;
    let cpu_summary =
        benchmark_preprocessor("cpu", &cpu, &images, config, iterations)?.with_label("cpu");
    info!(
        "preprocess_benchmark {}",
        serde_json::to_string(&cpu_summary)?
    );

    if let Some(ctx) = gpu_context {
        match WgpuPreprocessor::new(ctx.clone()) {
            Ok(gpu) => {
                let summary = benchmark_preprocessor("gpu", &gpu, &images, config, iterations)?
                    .with_label(&format!(
                        "gpu:{}:{:?}",
                        ctx.adapter_info().name,
                        ctx.adapter_info().backend
                    ));
                info!("preprocess_benchmark {}", serde_json::to_string(&summary)?);
            }
            Err(err) => warn!("Skipping GPU benchmark (initialization failed: {err})"),
        }
    } else {
        info!("GPU context unavailable; skipping GPU preprocessing benchmark.");
    }

    Ok(())
}

fn load_benchmark_images(items: &[ProcessingItem]) -> Result<Vec<DynamicImage>> {
    let mut images = Vec::with_capacity(items.len());
    for item in items {
        let img = load_image(&item.source)
            .with_context(|| format!("failed to load benchmark image {}", item.source.display()))?;
        images.push(img);
    }
    Ok(images)
}

fn benchmark_preprocessor<T: Preprocessor + ?Sized>(
    label: &str,
    preprocessor: &T,
    images: &[DynamicImage],
    config: &PreprocessConfig,
    iterations: usize,
) -> Result<PreprocessBenchmarkSummary> {
    let mut timings = Vec::with_capacity(images.len() * iterations);
    for image in images {
        for _ in 0..iterations {
            let start = Instant::now();
            preprocessor.preprocess(image, config)?;
            timings.push(start.elapsed());
        }
    }
    Ok(PreprocessBenchmarkSummary {
        label: label.to_string(),
        samples: images.len(),
        iterations_per_sample: iterations,
        total_ms: sum_durations_ms(&timings),
        avg_ms: avg_duration_ms(&timings),
        min_ms: timings
            .iter()
            .map(|d| duration_to_ms(*d))
            .fold(f64::MAX, f64::min),
        max_ms: timings
            .iter()
            .map(|d| duration_to_ms(*d))
            .fold(0.0, f64::max),
    })
}

fn sum_durations_ms(samples: &[Duration]) -> f64 {
    samples.iter().map(|d| duration_to_ms(*d)).sum()
}

fn avg_duration_ms(samples: &[Duration]) -> f64 {
    if samples.is_empty() {
        0.0
    } else {
        sum_durations_ms(samples) / samples.len() as f64
    }
}

fn duration_to_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
