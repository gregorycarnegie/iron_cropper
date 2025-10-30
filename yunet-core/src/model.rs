use std::path::Path;

use anyhow::{Context, Result};
use tract_onnx::prelude::{
    Datum, Framework, Graph, InferenceFact, InferenceModelExt, IntoTensor, SimplePlan, TValue,
    TVec, Tensor, TypedFact, TypedOp, tvec,
};

use crate::preprocess::InputSize;

type RunnableModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

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

        let model = tract_onnx::onnx()
            .model_for_path(path)
            .with_context(|| format!("failed to parse ONNX graph from {}", path.display()))?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec![1, 3, input_size.height as i64, input_size.width as i64],
                ),
            )
            .map_err(|e| anyhow::anyhow!("unable to constrain input fact: {e}"))?
            .into_optimized()
            .map_err(|e| anyhow::anyhow!("unable to optimize YuNet graph: {e}"))?
            .into_runnable()
            .map_err(|e| anyhow::anyhow!("unable to make YuNet graph runnable: {e}"))?;

        Ok(Self {
            runnable: model,
            input_size,
        })
    }

    /// Execute YuNet with a preprocessed tensor and return the first output tensor.
    pub fn run(&self, input: &Tensor) -> Result<Tensor> {
        let outputs = self
            .runnable
            .run(tvec![input.clone().into()])
            .map_err(|e| anyhow::anyhow!("YuNet execution failed: {e}"))?;

        extract_first_tensor(outputs)
    }

    pub fn input_size(&self) -> InputSize {
        self.input_size
    }
}

fn extract_first_tensor(outputs: TVec<TValue>) -> Result<Tensor> {
    let value = outputs
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("YuNet model produced no outputs"))?;

    Ok(value.into_tensor())
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
