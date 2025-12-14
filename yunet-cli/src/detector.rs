//! YuNet detector construction and configuration.

use std::{path::Path, sync::Arc};

use anyhow::Result;
use log::{info, warn};
use yunet_core::{
    CpuPreprocessor, PostprocessConfig, PreprocessConfig, Preprocessor, WgpuPreprocessor,
    YuNetDetector,
};
use yunet_utils::gpu::GpuStatusIndicator;

use crate::gpu::CliGpuRuntime;

pub fn build_cli_detector(
    model_path: &Path,
    preprocess: &PreprocessConfig,
    postprocess: &PostprocessConfig,
    gpu_runtime: &CliGpuRuntime,
    prefer_gpu_inference: bool,
) -> Result<YuNetDetector> {
    if prefer_gpu_inference {
        if let Some(gpu_ctx) = gpu_runtime.context() {
            let preprocessor: Arc<dyn Preprocessor> = match WgpuPreprocessor::new(gpu_ctx.clone()) {
                Ok(pre) => {
                    info!(
                        "Using GPU preprocessing + inference on {} ({:?})",
                        gpu_ctx.adapter_info().name,
                        gpu_ctx.adapter_info().backend
                    );
                    Arc::new(pre)
                }
                Err(err) => {
                    warn!(
                        "GPU preprocessor initialization failed ({err}); using CPU preprocessing for GPU inference."
                    );
                    Arc::new(CpuPreprocessor)
                }
            };

            match YuNetDetector::with_gpu_preprocessor(
                model_path,
                preprocess.clone(),
                postprocess.clone(),
                preprocessor,
            ) {
                Ok(detector) => {
                    info!("GPU inference enabled for CLI detector.");
                    return Ok(detector);
                }
                Err(err) => {
                    warn!("GPU inference initialization failed: {err}; falling back to CPU path.");
                }
            }
        } else {
            warn!(
                "GPU inference requested but no GPU context available; falling back to CPU path."
            );
        }
    }

    if let Some(gpu_ctx) = gpu_runtime.context() {
        match WgpuPreprocessor::new(gpu_ctx.clone()) {
            Ok(pre) => {
                info!(
                    "Using GPU preprocessing on {} ({:?})",
                    gpu_ctx.adapter_info().name,
                    gpu_ctx.adapter_info().backend
                );
                let preprocessor: Arc<dyn Preprocessor> = Arc::new(pre);
                YuNetDetector::with_preprocessor(
                    model_path,
                    preprocess.clone(),
                    postprocess.clone(),
                    preprocessor,
                )
            }
            Err(err) => {
                let info = gpu_ctx.adapter_info();
                let status = GpuStatusIndicator::fallback(
                    format!("Failed to initialize GPU preprocessor: {err}"),
                    Some(info.name.clone()),
                    Some(format!("{:?}", info.backend)),
                );

                // Log the fallback status
                warn!("{:?}", status);
                YuNetDetector::new(model_path, preprocess.clone(), postprocess.clone())
            }
        }
    } else {
        YuNetDetector::new(model_path, preprocess.clone(), postprocess.clone())
    }
}
