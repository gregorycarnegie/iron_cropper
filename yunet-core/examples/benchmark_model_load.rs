use std::time::Instant;
use yunet_core::{preprocess::InputSize, YuNetModel};

fn main() -> anyhow::Result<()> {
    let model_path = "models/face_detection_yunet_2023mar_640.onnx";
    let input_size = InputSize::new(640, 640);

    println!("Benchmarking YuNet model loading...");
    println!("Model: {}", model_path);
    println!("Input size: {}x{}\n", input_size.width, input_size.height);

    // Warm up (first load might involve filesystem caching)
    println!("Warm-up load...");
    let start = Instant::now();
    let _model = YuNetModel::load(model_path, input_size)?;
    let warmup_time = start.elapsed();
    println!("  Warm-up: {:.4}s\n", warmup_time.as_secs_f64());

    // Measure multiple iterations
    let iterations = 5;
    let mut times = Vec::with_capacity(iterations);

    println!("Running {} timed iterations:", iterations);
    for i in 0..iterations {
        let start = Instant::now();
        let _model = YuNetModel::load(model_path, input_size)?;
        let elapsed = start.elapsed();
        times.push(elapsed.as_secs_f64());
        println!("  Iteration {}: {:.4}s", i + 1, elapsed.as_secs_f64());
    }

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("\n=== Model Loading Statistics ===");
    println!("  Average: {:.4}s", avg);
    println!("  Min:     {:.4}s", min);
    println!("  Max:     {:.4}s", max);

    Ok(())
}
