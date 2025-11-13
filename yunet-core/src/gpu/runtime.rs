use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use tract_onnx::prelude::Tensor;
use yunet_utils::gpu::{GpuAvailability, GpuContext, GpuContextOptions};

use crate::gpu::graph::{
    self, BACKBONE_STAGES, DETECTION_HEADS, DetectionLevelOutputs, WeightProvider,
};
use crate::gpu::onnx::OnnxInitializerMap;
use crate::gpu::ops::GpuInferenceOps;
use crate::gpu::tensor::GpuTensor;
use crate::model::decode_yunet_outputs;
use crate::preprocess::InputSize;

#[derive(Debug)]
pub struct GpuYuNet {
    ops: Arc<GpuInferenceOps>,
    weights: Arc<HashMap<String, GpuTensor>>,
    input_size: InputSize,
}

impl GpuYuNet {
    pub fn new<P: AsRef<Path>>(model_path: P, input_size: InputSize) -> Result<Self> {
        let model_path = model_path.as_ref();
        let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
            GpuAvailability::Available(ctx) => ctx,
            GpuAvailability::Disabled { reason } => {
                anyhow::bail!("GPU backend disabled by configuration: {reason}")
            }
            GpuAvailability::Unavailable { error } => {
                anyhow::bail!("GPU backend unavailable: {error}")
            }
        };
        let ops = Arc::new(GpuInferenceOps::new(context)?);
        let loader = graph::load_backbone_weights(
            model_path,
            BACKBONE_STAGES.len(),
            true,
            DETECTION_HEADS.len(),
        )?;
        let weight_map = upload_gpu_weights(&ops, loader)?;
        Ok(Self {
            ops,
            weights: Arc::new(weight_map),
            input_size,
        })
    }

    pub fn run(&self, tensor: Tensor) -> Result<Tensor> {
        let dims = tensor.shape().to_vec();
        let data = tensor
            .as_slice::<f32>()
            .context("gpu input must be contiguous f32")?
            .to_vec();
        let input_gpu = self
            .ops
            .upload_tensor(dims, &data, Some("yunet_gpu_input"))
            .context("upload input tensor")?;

        let weight_provider = CachedWeights {
            tensors: Arc::clone(&self.weights),
        };
        let features = graph::run_backbone_features(
            &self.ops,
            &weight_provider,
            &input_gpu,
            BACKBONE_STAGES.len(),
        )?;
        let levels = graph::run_neck_and_heads(&self.ops, &weight_provider, &features)?;
        let outputs = build_decode_tensors(&levels)?;
        decode_yunet_outputs(&outputs, self.input_size)
    }
}

fn upload_gpu_weights(
    ops: &GpuInferenceOps,
    loader: OnnxInitializerMap,
) -> Result<HashMap<String, GpuTensor>> {
    let mut map = HashMap::with_capacity(loader.len());
    for (name, tensor) in loader.into_map() {
        let gpu_tensor = ops.upload_tensor(
            tensor.dims().to_vec(),
            tensor.data(),
            Some(&format!("weight::{name}")),
        )?;
        map.insert(name, gpu_tensor);
    }
    Ok(map)
}

struct CachedWeights {
    tensors: Arc<HashMap<String, GpuTensor>>,
}

impl WeightProvider for CachedWeights {
    fn tensor(&self, name: &str) -> Result<GpuTensor> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("cached weight '{name}' missing"))
    }
}

fn build_decode_tensors(levels: &[DetectionLevelOutputs; 3]) -> Result<Vec<Tensor>> {
    let mut cls_tensors = Vec::with_capacity(3);
    let mut obj_tensors = Vec::with_capacity(3);
    let mut bbox_tensors = Vec::with_capacity(3);
    let mut kps_tensors = Vec::with_capacity(3);

    for level in levels.iter() {
        let shape = level.feature.shape().dims();
        anyhow::ensure!(
            shape.len() == 4,
            "feature map must be NCHW (got {:?})",
            shape
        );
        let height = shape[2];
        let width = shape[3];
        cls_tensors.push(flatten_branch(&level.cls, height, width, 1, true)?);
        obj_tensors.push(flatten_branch(&level.obj, height, width, 1, true)?);
        bbox_tensors.push(flatten_branch(&level.bbox, height, width, 4, false)?);
        kps_tensors.push(flatten_branch(&level.kps, height, width, 10, false)?);
    }

    let mut outputs = Vec::with_capacity(DET_HEAD_OUTPUTS);
    outputs.extend(cls_tensors);
    outputs.extend(obj_tensors);
    outputs.extend(bbox_tensors);
    outputs.extend(kps_tensors);
    Ok(outputs)
}

const DET_HEAD_OUTPUTS: usize = 12;

fn flatten_branch(
    tensor: &GpuTensor,
    height: usize,
    width: usize,
    channels: usize,
    apply_sigmoid: bool,
) -> Result<Tensor> {
    let flat = tensor.to_vec().context("download branch tensor")?;
    let flattened = reorder_hw_major(&flat, channels, height, width, apply_sigmoid);
    let rows = height * width;
    Tensor::from_shape(&[rows, channels], &flattened)
        .map_err(|e| anyhow!("failed to build tensor from branch output: {e}"))
}

fn reorder_hw_major(
    data: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    apply_sigmoid: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; height * width * channels];
    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let src = ((c * height + y) * width + x) as usize;
                let dst = ((y * width + x) * channels + c) as usize;
                let mut value = data[src];
                if apply_sigmoid {
                    value = sigmoid(value);
                }
                out[dst] = value;
            }
        }
    }
    out
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
