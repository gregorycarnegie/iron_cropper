use std::{fmt::Write, path::Path};

use anyhow::{Context, Result};
use log::{debug, warn};
use tract_onnx::prelude::{
    Framework, Graph, InferenceModelExt, IntoTensor, SimplePlan, Tensor, TypedFact, TypedOp, tvec,
};

use crate::preprocess::InputSize;

type RunnableModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

const STRIDES: [usize; 3] = [8, 16, 32];
const OUTPUTS_PER_STRIDE: usize = 4; // cls, obj, bbox, kps
const OUTPUT_COLS: usize = 15; // bbox (4) + landmarks (10) + score (1)

#[derive(Clone, Copy)]
struct StrideMeta {
    stride_index: usize,
    stride: usize,
    cols: usize,
    rows: usize,
    cell_count: usize,
    offset: usize,
}

/// Wrapper around the YuNet ONNX runnable model.
///
/// This struct handles loading the ONNX graph, preparing it for execution, and running inference.
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
            Ok(model) => {
                debug!(
                    "YuNet model {} optimized successfully ({}x{})",
                    path.display(),
                    input_size.width,
                    input_size.height
                );
                model
            }
            Err(opt_err) => {
                let optimize_msg = format!("{opt_err}");
                let mut chain_msg = String::new();
                for cause in opt_err.chain() {
                    let _ = writeln!(&mut chain_msg, "  • {cause}");
                }
                warn!(
                    "YuNet model {} failed optimized load ({}); falling back to decluttered graph (~2x slower).\nError chain:\n{}",
                    path.display(),
                    optimize_msg,
                    chain_msg.trim_end()
                );
                let decluttered =
                    load_runnable_model(path, input_size, false).with_context(|| {
                    format!(
                        "fallback to decluttered YuNet graph failed after optimize error: {optimize_msg}"
                    )
                })?;
                debug!(
                    "YuNet model {} running in decluttered mode ({}x{})",
                    path.display(),
                    input_size.width,
                    input_size.height
                );
                decluttered
            }
        };

        Ok(Self {
            runnable,
            input_size,
        })
    }

    /// Execute YuNet with a preprocessed tensor and return decoded detections.
    ///
    /// The resulting tensor has shape `[N, 15]` where each row is
    /// `[x, y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score]`
    /// in the resized input coordinate space.
    pub fn run(&self, input: Tensor) -> Result<Tensor> {
        let outputs = self
            .runnable
            .run(tvec![input.into()])
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

pub(crate) fn decode_yunet_outputs(outputs: &[Tensor], input_size: InputSize) -> Result<Tensor> {
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

    let mut metas = Vec::with_capacity(STRIDES.len());
    let mut total_cells = 0usize;
    let mut offset = 0usize;
    for (stride_index, &stride) in STRIDES.iter().enumerate() {
        anyhow::ensure!(
            pad_w
                .checked_rem(stride)
                .is_some_and(|remainder| remainder == 0),
            "input width not divisible by stride {}",
            stride
        );
        anyhow::ensure!(
            pad_h
                .checked_rem(stride)
                .is_some_and(|remainder| remainder == 0),
            "input height not divisible by stride {}",
            stride
        );
        let cols = pad_w / stride;
        let rows = pad_h / stride;
        let cell_count = rows * cols;
        let len = cell_count * OUTPUT_COLS;
        metas.push(StrideMeta {
            stride_index,
            stride,
            cols,
            rows,
            cell_count,
            offset,
        });
        total_cells += cell_count;
        offset += len;
    }
    let total_capacity = total_cells * OUTPUT_COLS;
    debug_assert_eq!(offset, total_capacity);

    let mut fused = vec![0f32; total_capacity];

    for meta in metas.iter() {
        let StrideMeta {
            stride_index,
            stride,
            cols,
            rows,
            cell_count,
            offset,
        } = *meta;
        let stride_f = stride as f32;

        let cls_slice = outputs[stride_index]
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("cls output not f32: {e}"))?;
        let obj_slice = outputs[stride_index + STRIDES.len()]
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("obj output not f32: {e}"))?;
        let bbox_slice = outputs[stride_index + STRIDES.len() * 2]
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("bbox output not f32: {e}"))?;
        let kps_slice = outputs[stride_index + STRIDES.len() * 3]
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

        let dst = &mut fused[offset..offset + cell_count * OUTPUT_COLS];
        let mut write = 0;

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
                let x = (-0.5f32).mul_add(w, cx);
                let y = (-0.5f32).mul_add(h, cy);

                dst[write] = x;
                dst[write + 1] = y;
                dst[write + 2] = w;
                dst[write + 3] = h;

                let kps_offset = idx * 10;
                let mut write_kps = write + 4;
                for lm in 0..5 {
                    dst[write_kps] = (kps_slice[kps_offset + lm * 2] + col as f32) * stride_f;
                    dst[write_kps + 1] =
                        (kps_slice[kps_offset + lm * 2 + 1] + row as f32) * stride_f;
                    write_kps += 2;
                }

                dst[write + 14] = score;
                write += OUTPUT_COLS;
            }
        }
    }

    let rows = fused.len() / OUTPUT_COLS;
    Tensor::from_shape(&[rows, OUTPUT_COLS], &fused)
        .map_err(|e| anyhow::anyhow!("failed to build fused YuNet tensor: {e}"))
}

fn align_to(value: usize, divisor: usize) -> usize {
    assert!(divisor > 0, "divisor must be non-zero");
    value.div_ceil(divisor) * divisor
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

#[cfg(test)]
mod benches {
    use super::*;
    use anyhow::Result;
    use rayon::prelude::*;
    use std::time::Instant;

    fn decode_yunet_outputs_baseline(outputs: &[Tensor], input_size: InputSize) -> Result<Tensor> {
        let input_w = input_size.width as usize;
        let input_h = input_size.height as usize;
        let pad_w = align_to(input_w, 32);
        let pad_h = align_to(input_h, 32);

        let mut total_cells = 0;
        for &stride in STRIDES.iter() {
            anyhow::ensure!(
                pad_w
                    .checked_rem(stride)
                    .is_some_and(|remainder| remainder == 0),
                "input width not divisible by stride {}",
                stride
            );
            anyhow::ensure!(
                pad_h
                    .checked_rem(stride)
                    .is_some_and(|remainder| remainder == 0),
                "input height not divisible by stride {}",
                stride
            );
            total_cells += (pad_w / stride) * (pad_h / stride);
        }
        let total_capacity = total_cells * OUTPUT_COLS;

        let stride_results: Result<Vec<Vec<f32>>> = STRIDES
            .par_iter()
            .enumerate()
            .map(|(stride_index, &stride)| -> Result<Vec<f32>> {
                let cols = pad_w / stride;
                let rows = pad_h / stride;
                let cell_count = rows * cols;
                let stride_f = stride as f32;

                let cls_slice = outputs[stride_index]
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("cls output not f32: {e}"))?;
                let obj_slice = outputs[stride_index + STRIDES.len()]
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("obj output not f32: {e}"))?;
                let bbox_slice = outputs[stride_index + STRIDES.len() * 2]
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("bbox output not f32: {e}"))?;
                let kps_slice = outputs[stride_index + STRIDES.len() * 3]
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

                let mut stride_data = Vec::with_capacity(cell_count * OUTPUT_COLS);

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
                        let x = (-0.5f32).mul_add(w, cx);
                        let y = (-0.5f32).mul_add(h, cy);

                        stride_data.push(x);
                        stride_data.push(y);
                        stride_data.push(w);
                        stride_data.push(h);

                        let kps_offset = idx * 10;
                        for lm in 0..5 {
                            let lx = (kps_slice[kps_offset + lm * 2] + col as f32) * stride_f;
                            let ly = (kps_slice[kps_offset + lm * 2 + 1] + row as f32) * stride_f;
                            stride_data.push(lx);
                            stride_data.push(ly);
                        }

                        stride_data.push(score);
                    }
                }

                Ok(stride_data)
            })
            .collect();

        let stride_vecs = stride_results?;
        let mut fused = Vec::with_capacity(total_capacity);
        for vec in stride_vecs {
            fused.extend_from_slice(&vec);
        }

        let rows = fused.len() / OUTPUT_COLS;
        Tensor::from_shape(&[rows, OUTPUT_COLS], &fused)
            .map_err(|e| anyhow::anyhow!("failed to build fused YuNet tensor: {e}"))
    }

    fn mock_outputs(input_size: InputSize) -> Vec<Tensor> {
        let input_w = align_to(input_size.width as usize, 32);
        let input_h = align_to(input_size.height as usize, 32);
        let mut cls_tensors = Vec::new();
        let mut obj_tensors = Vec::new();
        let mut bbox_tensors = Vec::new();
        let mut kps_tensors = Vec::new();

        for &stride in STRIDES.iter() {
            let cols = input_w / stride;
            let rows = input_h / stride;
            let cell_count = cols * rows;

            let cls = vec![0.9f32; cell_count];
            let obj = vec![0.8f32; cell_count];
            let bbox: Vec<f32> = (0..cell_count * 4)
                .map(|idx| ((idx % 11) as f32 * 0.01) - 0.05)
                .collect();
            let kps: Vec<f32> = (0..cell_count * 10)
                .map(|idx| ((idx % 7) as f32 * 0.015) - 0.05)
                .collect();

            cls_tensors.push(Tensor::from_shape(&[cell_count], &cls).unwrap());
            obj_tensors.push(Tensor::from_shape(&[cell_count], &obj).unwrap());
            bbox_tensors.push(Tensor::from_shape(&[cell_count, 4], &bbox).unwrap());
            kps_tensors.push(Tensor::from_shape(&[cell_count, 10], &kps).unwrap());
        }

        cls_tensors
            .into_iter()
            .chain(obj_tensors)
            .chain(bbox_tensors)
            .chain(kps_tensors)
            .collect()
    }

    #[test]
    #[ignore]
    fn bench_decode_yunet_outputs() {
        let input_size = InputSize {
            width: 320,
            height: 320,
        };
        let tensors = mock_outputs(input_size);

        for _ in 0..3 {
            let _ = decode_yunet_outputs(&tensors, input_size).unwrap();
            let _ = decode_yunet_outputs_baseline(&tensors, input_size).unwrap();
        }

        let iterations = 20;
        let start_new = Instant::now();
        for _ in 0..iterations {
            let _ = decode_yunet_outputs(&tensors, input_size).unwrap();
        }
        let new_time = start_new.elapsed();

        let start_old = Instant::now();
        for _ in 0..iterations {
            let _ = decode_yunet_outputs_baseline(&tensors, input_size).unwrap();
        }
        let old_time = start_old.elapsed();

        println!(
            "decode_yunet_outputs optimized avg: {:?}, baseline avg: {:?}",
            new_time / iterations,
            old_time / iterations
        );
    }
}
