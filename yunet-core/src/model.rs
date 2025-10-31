use std::path::Path;

use anyhow::{Context, Result};
use tract_onnx::prelude::{
    Framework, Graph, InferenceModelExt, IntoTensor, SimplePlan, Tensor, TypedFact, TypedOp, tvec,
};

use crate::preprocess::InputSize;

type RunnableModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

const STRIDES: [usize; 3] = [8, 16, 32];
const OUTPUTS_PER_STRIDE: usize = 4; // cls, obj, bbox, kps
const OUTPUT_COLS: usize = 15; // bbox (4) + landmarks (10) + score (1)

/// Wrapper around the YuNet ONNX runnable model.
#[derive(Debug)]
pub struct YuNetModel {
    runnable: RunnableModel,
    input_size: InputSize,
}

impl YuNetModel {
    /// Load and optimize the YuNet ONNX graph for a specific input size.
    pub fn load<P: AsRef<Path>>(model_path: P, input_size: InputSize) -> Result<Self> {
        let path = model_path.as_ref();
        anyhow::ensure!(path.exists(), "model file not found: {}", path.display());

        let runnable = match load_runnable_model(path, input_size, true) {
            Ok(model) => model,
            Err(opt_err) => {
                let optimize_msg = format!("{opt_err}");
                load_runnable_model(path, input_size, false).with_context(|| {
                    format!(
                        "fallback to decluttered YuNet graph failed after optimize error: {optimize_msg}"
                    )
                })?
            }
        };

        Ok(Self {
            runnable: runnable,
            input_size,
        })
    }

    /// Execute YuNet with a preprocessed tensor and return decoded detections.
    ///
    /// The resulting tensor has shape `[N, 15]` where each row is
    /// `[x, y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score]`
    /// in the resized input coordinate space.
    pub fn run(&self, input: &Tensor) -> Result<Tensor> {
        let outputs = self
            .runnable
            .run(tvec![input.clone().into()])
            .map_err(|e| anyhow::anyhow!("YuNet execution failed: {e}"))?;

        let mut tensors: Vec<Tensor> = outputs
            .into_iter()
            .map(|value| value.into_tensor())
            .collect();

        match tensors.len() {
            0 => anyhow::bail!("YuNet model produced no outputs"),
            1 => Ok(tensors
                .pop()
                .ok_or_else(|| anyhow::anyhow!("YuNet model produced no outputs"))?),
            len if len == STRIDES.len() * OUTPUTS_PER_STRIDE => {
                decode_yunet_outputs(&tensors, self.input_size)
            }
            other => anyhow::bail!(
                "unexpected number of YuNet outputs: expected 1 or {}, got {}",
                STRIDES.len() * OUTPUTS_PER_STRIDE,
                other
            ),
        }
    }

    pub fn input_size(&self) -> InputSize {
        self.input_size
    }
}

fn load_runnable_model(
    path: &Path,
    _input_size: InputSize,
    optimized: bool,
) -> Result<RunnableModel> {
    // Load model and let it infer shape from ONNX file
    // The input_size parameter is used for preprocessing and coordinate scaling,
    // but the model itself should match the ONNX file's declared input shape
    let model = tract_onnx::onnx()
        .model_for_path(path)
        .with_context(|| format!("failed to parse ONNX graph from {}", path.display()))?;

    if optimized {
        model
            .into_optimized()
            .map_err(|e| anyhow::anyhow!("unable to optimize YuNet graph: {e}"))?
            .into_runnable()
            .map_err(|e| anyhow::anyhow!("unable to make YuNet graph runnable: {e}"))
    } else {
        model
            .into_typed()
            .map_err(|e| anyhow::anyhow!("unable to type-check YuNet graph: {e}"))?
            .into_decluttered()
            .map_err(|e| anyhow::anyhow!("unable to declutter YuNet graph: {e}"))?
            .into_runnable()
            .map_err(|e| anyhow::anyhow!("unable to make YuNet graph runnable: {e}"))
    }
}

fn decode_yunet_outputs(outputs: &[Tensor], input_size: InputSize) -> Result<Tensor> {
    anyhow::ensure!(
        outputs.len() == STRIDES.len() * OUTPUTS_PER_STRIDE,
        "YuNet decode expects {} tensors, got {}",
        STRIDES.len() * OUTPUTS_PER_STRIDE,
        outputs.len()
    );

    let input_w = input_size.width as usize;
    let input_h = input_size.height as usize;
    let pad_w = align_to(input_w, 32);
    let pad_h = align_to(input_h, 32);

    let mut total_rows = 0usize;
    for &stride in STRIDES.iter() {
        anyhow::ensure!(
            pad_w % stride == 0,
            "input width not divisible by stride {}",
            stride
        );
        anyhow::ensure!(
            pad_h % stride == 0,
            "input height not divisible by stride {}",
            stride
        );
        total_rows += (pad_w / stride) * (pad_h / stride);
    }

    let mut fused = Vec::<f32>::with_capacity(total_rows * OUTPUT_COLS);

    for (stride_index, &stride) in STRIDES.iter().enumerate() {
        let cols = pad_w / stride;
        let rows = pad_h / stride;
        let cell_count = rows * cols;
        let stride_f = stride as f32;

        let cls_tensor = &outputs[stride_index];
        let obj_tensor = &outputs[stride_index + STRIDES.len()];
        let bbox_tensor = &outputs[stride_index + STRIDES.len() * 2];
        let kps_tensor = &outputs[stride_index + STRIDES.len() * 3];

        let cls_slice = cls_tensor
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("cls output not f32: {e}"))?;
        let obj_slice = obj_tensor
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("obj output not f32: {e}"))?;
        let bbox_slice = bbox_tensor
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("bbox output not f32: {e}"))?;
        let kps_slice = kps_tensor
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("kps output not f32: {e}"))?;

        anyhow::ensure!(
            cls_slice.len() == cell_count,
            "cls length mismatch: expected {}, got {}",
            cell_count,
            cls_slice.len()
        );
        anyhow::ensure!(
            obj_slice.len() == cell_count,
            "obj length mismatch: expected {}, got {}",
            cell_count,
            obj_slice.len()
        );
        anyhow::ensure!(
            bbox_slice.len() == cell_count * 4,
            "bbox length mismatch: expected {}, got {}",
            cell_count * 4,
            bbox_slice.len()
        );
        anyhow::ensure!(
            kps_slice.len() == cell_count * 10,
            "kps length mismatch: expected {}, got {}",
            cell_count * 10,
            kps_slice.len()
        );

        for row in 0..rows {
            for col in 0..cols {
                let idx = row * cols + col;
                let cls_score = cls_slice[idx].clamp(0.0, 1.0);
                let obj_score = obj_slice[idx].clamp(0.0, 1.0);
                let mut score = (cls_score * obj_score).sqrt();
                if !score.is_finite() {
                    score = 0.0;
                }

                let bbox_offset = idx * 4;
                let dx = bbox_slice[bbox_offset];
                let dy = bbox_slice[bbox_offset + 1];
                let dw = bbox_slice[bbox_offset + 2];
                let dh = bbox_slice[bbox_offset + 3];

                let cx = (col as f32 + dx) * stride_f;
                let cy = (row as f32 + dy) * stride_f;
                let w = dw.exp() * stride_f;
                let h = dh.exp() * stride_f;
                let x = cx - w * 0.5;
                let y = cy - h * 0.5;

                fused.push(x);
                fused.push(y);
                fused.push(w);
                fused.push(h);

                let kps_offset = idx * 10;
                for lm in 0..5 {
                    let lx = (kps_slice[kps_offset + lm * 2] + col as f32) * stride_f;
                    let ly = (kps_slice[kps_offset + lm * 2 + 1] + row as f32) * stride_f;
                    fused.push(lx);
                    fused.push(ly);
                }

                fused.push(score);
            }
        }
    }

    let rows = fused.len() / OUTPUT_COLS;
    Tensor::from_shape(&[rows, OUTPUT_COLS], &fused)
        .map_err(|e| anyhow::anyhow!("failed to build fused YuNet tensor: {e}"))
}

fn align_to(value: usize, divisor: usize) -> usize {
    ((value + divisor - 1) / divisor) * divisor
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn loading_missing_model_fails() {
        let result = YuNetModel::load("missing.onnx", InputSize::default());
        assert!(result.is_err());
    }

    #[test]
    fn invalid_model_produces_useful_error() {
        let mut temp = NamedTempFile::new().expect("temp file");
        temp.write_all(b"not a real onnx file")
            .expect("write mock model");

        let err = YuNetModel::load(temp.path(), InputSize::default())
            .expect_err("invalid ONNX should fail");
        let message = format!("{err}");
        assert!(
            message.contains("failed to parse ONNX") || message.contains("unable to optimize"),
            "Unexpected error message: {message}"
        );
    }
}
