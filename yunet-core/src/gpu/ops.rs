use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable, bytes_of};
use wgpu::util::DeviceExt;
use yunet_utils::gpu::GpuContext;

use super::tensor::GpuTensor;
use yunet_utils::gpu::GpuBufferPool;

const CONV2D_WGSL: &str = include_str!("conv2d.wgsl");
const BATCH_NORM_WGSL: &str = include_str!("batch_norm.wgsl");
const ACTIVATION_WGSL: &str = include_str!("activation.wgsl");
use crate::gpu::{ADD_WGSL, MAX_POOL_WGSL, UPSAMPLE2X_WGSL};

const CONV_WORKGROUP_X: u32 = 8;
const CONV_WORKGROUP_Y: u32 = 8;
const BN_WORKGROUP_X: u32 = 8;
const BN_WORKGROUP_Y: u32 = 8;
const ACTIVATION_WORKGROUP_SIZE: u32 = 256;
const POOL_WORKGROUP_X: u32 = 8;
const POOL_WORKGROUP_Y: u32 = 8;
const ADD_WORKGROUP_SIZE: u32 = 256;
const UPSAMPLE_WORKGROUP_X: u32 = 8;
const UPSAMPLE_WORKGROUP_Y: u32 = 8;

/// Collection of GPU-backed YuNet primitives.
///
/// This type owns the compiled WGSL pipelines for convolution, batch
/// normalization, and activations so callers can reuse them across layers.
#[derive(Debug)]
pub struct GpuInferenceOps {
    context: Arc<GpuContext>,
    buffer_pool: Arc<GpuBufferPool>,
    conv2d: Conv2dPipeline,
    batch_norm: BatchNormPipeline,
    activation: ActivationPipeline,
    max_pool: MaxPoolPipeline,
    add: AddPipeline,
    upsample2x: Upsample2xPipeline,
}

impl GpuInferenceOps {
    /// Create the GPU pipelines from an existing [`GpuContext`].
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();
        let buffer_pool = Arc::new(GpuBufferPool::new(context.clone()));
        Ok(Self {
            conv2d: Conv2dPipeline::new(device, CONV2D_WGSL, 4)?,
            batch_norm: BatchNormPipeline::new(device)?,
            activation: ActivationPipeline::new(device)?,
            max_pool: MaxPoolPipeline::new(device)?,
            add: AddPipeline::new(device)?,
            upsample2x: Upsample2xPipeline::new(device)?,
            buffer_pool,
            context,
        })
    }

    /// Upload host data into a GPU tensor for reuse across layers.
    pub fn upload_tensor<D>(&self, dims: D, data: &[f32], label: Option<&str>) -> Result<GpuTensor>
    where
        D: Into<Vec<usize>>,
    {
        GpuTensor::from_slice_with_pool(
            self.context.clone(),
            Some(self.buffer_pool.clone()),
            dims,
            data,
            label,
        )
    }

    /// Download a tensor back to host memory.
    pub fn download_tensor(&self, tensor: &GpuTensor) -> Result<Vec<f32>> {
        tensor.to_vec()
    }

    fn ensure_same_context(&self, tensor: &GpuTensor, label: &str) -> Result<()> {
        anyhow::ensure!(
            Arc::ptr_eq(tensor.context(), &self.context),
            "{label} tensor was created from a different GPU context"
        );
        Ok(())
    }

    /// Execute a `Conv2D` layer using GPU-resident tensors.
    pub fn conv2d_tensor(
        &self,
        input: &GpuTensor,
        weights: &GpuTensor,
        bias: &GpuTensor,
        config: &Conv2dConfig,
    ) -> Result<GpuTensor> {
        config.validate(
            input.shape().elements(),
            weights.shape().elements(),
            bias.shape().elements(),
        )?;
        self.ensure_same_context(input, "conv2d input")?;
        self.ensure_same_context(weights, "conv2d weights")?;
        self.ensure_same_context(bias, "conv2d bias")?;
        self.conv2d.execute(
            &self.context,
            &self.buffer_pool,
            input,
            weights,
            bias,
            config,
        )
    }

    /// Execute a `Conv2D` layer using GPU-resident tensors with vectorized shader.
    pub fn conv2d_vec4_tensor(
        &self,
        input: &GpuTensor,
        weights: &GpuTensor,
        bias: &GpuTensor,
        config: &Conv2dConfig,
    ) -> Result<GpuTensor> {
        config.validate(
            input.shape().elements(),
            weights.shape().elements(),
            bias.shape().elements(),
        )?;
        self.ensure_same_context(input, "conv2d input")?;
        self.ensure_same_context(weights, "conv2d weights")?;
        self.ensure_same_context(bias, "conv2d bias")?;
        self.conv2d.execute(
            &self.context,
            &self.buffer_pool,
            input,
            weights,
            bias,
            config,
        )
    }

    /// Convenience wrapper that uploads host slices, runs Conv2D, and downloads the result.
    pub fn conv2d(
        &self,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        config: &Conv2dConfig,
    ) -> Result<Vec<f32>> {
        let input_tensor =
            self.upload_tensor(config.input_shape_dims(), input, Some("yunet_conv_input"))?;
        let weight_tensor = self.upload_tensor(
            config.weight_shape_dims(),
            weights,
            Some("yunet_conv_weights"),
        )?;
        let bias_tensor =
            self.upload_tensor(config.bias_shape_dims(), bias, Some("yunet_conv_bias"))?;
        let output = self.conv2d_tensor(&input_tensor, &weight_tensor, &bias_tensor, config)?;
        output.to_vec()
    }

    /// Batch-norm that keeps data on the GPU.
    pub fn batch_norm_tensor(
        &self,
        tensor: &GpuTensor,
        gamma: &GpuTensor,
        beta: &GpuTensor,
        mean: &GpuTensor,
        variance: &GpuTensor,
        config: &BatchNormConfig,
    ) -> Result<GpuTensor> {
        config.validate(
            tensor.shape().elements(),
            gamma.shape().elements(),
            beta.shape().elements(),
            mean.shape().elements(),
            variance.shape().elements(),
        )?;
        self.ensure_same_context(tensor, "batch_norm tensor")?;
        self.ensure_same_context(gamma, "batch_norm gamma")?;
        self.ensure_same_context(beta, "batch_norm beta")?;
        self.ensure_same_context(mean, "batch_norm mean")?;
        self.ensure_same_context(variance, "batch_norm variance")?;
        let tensors = BatchNormBindings {
            tensor,
            gamma,
            beta,
            mean,
            variance,
        };
        self.batch_norm.execute(&self.context, tensors, config)
    }

    pub fn batch_norm(
        &self,
        tensor: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mean: &[f32],
        variance: &[f32],
        config: &BatchNormConfig,
    ) -> Result<Vec<f32>> {
        let tensor_gpu =
            self.upload_tensor(config.tensor_shape_dims(), tensor, Some("yunet_bn_tensor"))?;
        let channels = config.channels as usize;
        let gamma_gpu = self.upload_tensor([channels], gamma, Some("yunet_bn_gamma"))?;
        let beta_gpu = self.upload_tensor([channels], beta, Some("yunet_bn_beta"))?;
        let mean_gpu = self.upload_tensor([channels], mean, Some("yunet_bn_mean"))?;
        let variance_gpu = self.upload_tensor([channels], variance, Some("yunet_bn_variance"))?;
        let output = self.batch_norm_tensor(
            &tensor_gpu,
            &gamma_gpu,
            &beta_gpu,
            &mean_gpu,
            &variance_gpu,
            config,
        )?;
        output.to_vec()
    }

    /// Apply an activation to a GPU tensor (returns a new tensor copy).
    pub fn activation_tensor(&self, tensor: &GpuTensor, kind: ActivationKind) -> Result<GpuTensor> {
        self.ensure_same_context(tensor, "activation tensor")?;
        self.activation.execute(&self.context, tensor, kind)
    }

    pub fn activation(&self, tensor: &[f32], kind: ActivationKind) -> Result<Vec<f32>> {
        let tensor_gpu =
            self.upload_tensor([tensor.len()], tensor, Some("yunet_activation_tensor"))?;
        let output = self.activation_tensor(&tensor_gpu, kind)?;
        output.to_vec()
    }

    /// Max pool on GPU tensors.
    pub fn max_pool_tensor(&self, tensor: &GpuTensor, config: &MaxPoolConfig) -> Result<GpuTensor> {
        self.ensure_same_context(tensor, "max_pool tensor")?;
        self.max_pool
            .execute(&self.context, &self.buffer_pool, tensor, config)
    }

    /// Element-wise addition of two tensors.
    pub fn add_tensors(&self, lhs: &GpuTensor, rhs: &GpuTensor) -> Result<GpuTensor> {
        self.ensure_same_context(lhs, "add lhs")?;
        self.ensure_same_context(rhs, "add rhs")?;
        anyhow::ensure!(
            lhs.shape().dims() == rhs.shape().dims(),
            "add tensors require identical shapes (lhs={:?}, rhs={:?})",
            lhs.shape().dims(),
            rhs.shape().dims()
        );
        self.add.execute(&self.context, &self.buffer_pool, lhs, rhs)
    }

    /// Nearest-neighbour 2x upsample (spatial dimensions doubled).
    pub fn resize2x_tensor(&self, tensor: &GpuTensor) -> Result<GpuTensor> {
        self.ensure_same_context(tensor, "resize tensor")?;
        self.upsample2x
            .execute(&self.context, &self.buffer_pool, tensor)
    }
}

#[derive(Debug)]
struct Conv2dPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pixels_per_thread: u32,
}

impl Conv2dPipeline {
    fn new(device: &wgpu::Device, source: &str, pixels_per_thread: u32) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yunet_conv2d_shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        // Force compilation check (wgpu validates lazily, but we can try to catch it early or rely on pipeline creation)
        // Actually, create_compute_pipeline will fail if shader is invalid.
        // Let's wrap the pipeline creation in a match to print error.

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yunet_conv2d_bgl"),
            entries: &[
                buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                uniform_entry(4),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("yunet_conv2d_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yunet_conv2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            pixels_per_thread,
        })
    }

    fn execute(
        &self,
        context: &Arc<GpuContext>,
        pool: &Arc<GpuBufferPool>,
        input: &GpuTensor,
        weights: &GpuTensor,
        bias: &GpuTensor,
        config: &Conv2dConfig,
    ) -> Result<GpuTensor> {
        let device = context.device();
        let queue = context.queue();
        let uniforms = Conv2dUniforms::from(config);
        let uniform_buffer = create_uniform_buffer(device, "yunet_conv2d_uniforms", &uniforms);
        let output = GpuTensor::uninitialized_with_pool(
            context.clone(),
            Some(pool.clone()),
            config.output_shape_dims(),
            Some("yunet_conv2d_output"),
        )?;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yunet_conv2d_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("yunet_conv2d_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yunet_conv2d_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                config
                    .output_width
                    .div_ceil(CONV_WORKGROUP_X * self.pixels_per_thread),
                config.output_height.div_ceil(CONV_WORKGROUP_Y),
                config.output_channels,
            );
        }

        queue.submit(Some(encoder.finish()));

        Ok(output)
    }
}

#[derive(Debug)]
struct BatchNormPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[derive(Clone, Copy)]
struct BatchNormBindings<'a> {
    tensor: &'a GpuTensor,
    gamma: &'a GpuTensor,
    beta: &'a GpuTensor,
    mean: &'a GpuTensor,
    variance: &'a GpuTensor,
}

impl BatchNormPipeline {
    fn new(device: &wgpu::Device) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yunet_batch_norm_shader"),
            source: wgpu::ShaderSource::Wgsl(BATCH_NORM_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yunet_batch_norm_bgl"),
            entries: &[
                buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: false }),
                buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
                uniform_entry(5),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("yunet_batch_norm_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yunet_batch_norm_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    fn execute(
        &self,
        context: &Arc<GpuContext>,
        tensors: BatchNormBindings<'_>,
        config: &BatchNormConfig,
    ) -> Result<GpuTensor> {
        let device = context.device();
        let queue = context.queue();
        let uniforms = BatchNormUniforms::from(config);
        let uniform_buffer = create_uniform_buffer(device, "yunet_bn_uniforms", &uniforms);
        let tensor_copy = tensors.tensor.duplicate(Some("yunet_bn_tensor_copy"))?;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yunet_bn_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tensor_copy.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tensors.gamma.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tensors.beta.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tensors.mean.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: tensors.variance.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("yunet_batch_norm_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yunet_batch_norm_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                config.width.div_ceil(BN_WORKGROUP_X),
                config.height.div_ceil(BN_WORKGROUP_Y),
                config.channels,
            );
        }

        queue.submit(Some(encoder.finish()));

        Ok(tensor_copy)
    }
}

#[derive(Debug)]
struct ActivationPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ActivationPipeline {
    fn new(device: &wgpu::Device) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yunet_activation_shader"),
            source: wgpu::ShaderSource::Wgsl(ACTIVATION_WGSL.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yunet_activation_bgl"),
            entries: &[
                buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: false }),
                uniform_entry(1),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("yunet_activation_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yunet_activation_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    fn execute(
        &self,
        context: &Arc<GpuContext>,
        tensor: &GpuTensor,
        kind: ActivationKind,
    ) -> Result<GpuTensor> {
        let device = context.device();
        let queue = context.queue();
        let len = tensor.shape().elements();
        let tensor_copy = tensor.duplicate(Some("yunet_activation_tensor_copy"))?;
        let uniforms = ActivationUniforms {
            len: len as u32,
            mode: kind as u32,
        };
        let uniform_buffer = create_uniform_buffer(device, "yunet_activation_uniforms", &uniforms);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yunet_activation_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tensor_copy.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("yunet_activation_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yunet_activation_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                div_ceil_uniform(len as u32, ACTIVATION_WORKGROUP_SIZE),
                1,
                1,
            );
        }

        queue.submit(Some(encoder.finish()));
        Ok(tensor_copy)
    }
}

#[derive(Debug)]
struct MaxPoolPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl MaxPoolPipeline {
    fn new(device: &wgpu::Device) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yunet_max_pool_shader"),
            source: wgpu::ShaderSource::Wgsl(MAX_POOL_WGSL.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yunet_max_pool_bgl"),
            entries: &[
                buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                uniform_entry(2),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("yunet_max_pool_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yunet_max_pool_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    fn execute(
        &self,
        context: &Arc<GpuContext>,
        pool: &Arc<GpuBufferPool>,
        tensor: &GpuTensor,
        config: &MaxPoolConfig,
    ) -> Result<GpuTensor> {
        let device = context.device();
        let queue = context.queue();
        let output = GpuTensor::uninitialized_with_pool(
            context.clone(),
            Some(pool.clone()),
            config.output_dims().to_vec(),
            Some("yunet_max_pool_output"),
        )?;
        let uniforms = MaxPoolUniforms {
            input_width: config.input_width,
            input_height: config.input_height,
            channels: config.channels,
            output_width: config.output_width,
            output_height: config.output_height,
            kernel: config.kernel,
            stride: config.stride,
            pad: config.pad,
        };
        let uniform_buffer = create_uniform_buffer(device, "yunet_max_pool_uniforms", &uniforms);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yunet_max_pool_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tensor.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("yunet_max_pool_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yunet_max_pool_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                config.output_width.div_ceil(POOL_WORKGROUP_X),
                config.output_height.div_ceil(POOL_WORKGROUP_Y),
                config.channels,
            );
        }

        queue.submit(Some(encoder.finish()));
        Ok(output)
    }
}

#[derive(Debug)]
struct AddPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl AddPipeline {
    fn new(device: &wgpu::Device) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yunet_add_shader"),
            source: wgpu::ShaderSource::Wgsl(ADD_WGSL.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yunet_add_bgl"),
            entries: &[
                buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                uniform_entry(3),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("yunet_add_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yunet_add_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    fn execute(
        &self,
        context: &Arc<GpuContext>,
        pool: &Arc<GpuBufferPool>,
        lhs: &GpuTensor,
        rhs: &GpuTensor,
    ) -> Result<GpuTensor> {
        let output = GpuTensor::uninitialized_with_pool(
            context.clone(),
            Some(pool.clone()),
            lhs.shape().dims().to_vec(),
            Some("yunet_add_output"),
        )?;
        let uniforms = AddUniforms {
            len: lhs.shape().elements() as u32,
        };
        let uniform_buffer =
            create_uniform_buffer(context.device(), "yunet_add_uniforms", &uniforms);

        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("yunet_add_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: lhs.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: rhs.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("yunet_add_encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yunet_add_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(uniforms.len.div_ceil(ADD_WORKGROUP_SIZE), 1, 1);
        }

        context.queue().submit(Some(encoder.finish()));
        Ok(output)
    }
}

#[derive(Debug)]
struct Upsample2xPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl Upsample2xPipeline {
    fn new(device: &wgpu::Device) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yunet_resize2x_shader"),
            source: wgpu::ShaderSource::Wgsl(UPSAMPLE2X_WGSL.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yunet_resize2x_bgl"),
            entries: &[
                buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                uniform_entry(2),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("yunet_resize2x_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yunet_resize2x_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    fn execute(
        &self,
        context: &Arc<GpuContext>,
        pool: &Arc<GpuBufferPool>,
        tensor: &GpuTensor,
    ) -> Result<GpuTensor> {
        let dims = tensor.shape().dims();
        anyhow::ensure!(
            dims.len() == 4,
            "resize2x expects NCHW tensor (got {:?})",
            dims
        );
        anyhow::ensure!(
            dims[0] == 1,
            "resize2x currently supports batch size 1 (got batch {})",
            dims[0]
        );
        let channels = dims[1] as u32;
        let height = dims[2] as u32;
        let width = dims[3] as u32;
        let output_dims = [dims[0], dims[1], dims[2] * 2, dims[3] * 2];
        let output = GpuTensor::uninitialized_with_pool(
            context.clone(),
            Some(pool.clone()),
            output_dims,
            Some("yunet_resize2x_output"),
        )?;
        let uniforms = UpsampleUniforms {
            input_width: width,
            input_height: height,
            channels,
            _padding: 0,
        };
        let uniform_buffer =
            create_uniform_buffer(context.device(), "yunet_resize2x_uniforms", &uniforms);

        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("yunet_resize2x_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: tensor.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("yunet_resize2x_encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yunet_resize2x_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let out_width = width * 2;
            let out_height = height * 2;
            pass.dispatch_workgroups(
                out_width.div_ceil(UPSAMPLE_WORKGROUP_X),
                out_height.div_ceil(UPSAMPLE_WORKGROUP_Y),
                channels,
            );
        }

        context.queue().submit(Some(encoder.finish()));
        Ok(output)
    }
}

fn create_uniform_buffer<T: Pod>(device: &wgpu::Device, label: &str, data: &T) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes_of(data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

fn buffer_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn div_ceil_uniform(value: u32, divisor: u32) -> u32 {
    if value == 0 {
        0
    } else {
        value.div_ceil(divisor)
    }
}

/// Supported activation operations.
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum ActivationKind {
    /// Rectified Linear Unit.
    Relu = 0,
    /// Sigmoid.
    Sigmoid = 1,
}

#[derive(Debug, Clone, Copy)]
pub struct Conv2dChannels {
    pub input: u32,
    pub output: u32,
}

impl Conv2dChannels {
    pub const fn new(input: u32, output: u32) -> Self {
        Self { input, output }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpatialDims {
    pub width: u32,
    pub height: u32,
}

impl SpatialDims {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

impl From<(u32, u32)> for SpatialDims {
    fn from(value: (u32, u32)) -> Self {
        Self {
            width: value.0,
            height: value.1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Conv2dOptions {
    pub groups: u32,
    pub activation: Option<ActivationKind>,
}

impl Conv2dOptions {
    pub const fn new(groups: u32, activation: Option<ActivationKind>) -> Self {
        Self { groups, activation }
    }
}

/// Geometry for a convolution layer.
#[derive(Debug, Clone)]
pub struct Conv2dConfig {
    pub batch: u32,
    pub input_channels: u32,
    pub output_channels: u32,
    pub input_width: u32,
    pub input_height: u32,
    pub kernel_width: u32,
    pub kernel_height: u32,
    pub stride_x: u32,
    pub stride_y: u32,
    pub pad_x: u32,
    pub pad_y: u32,
    pub groups: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub activation: Option<ActivationKind>,
}

impl Conv2dConfig {
    /// Create a validated convolution configuration.
    pub fn new(
        batch: u32,
        channels: Conv2dChannels,
        input: SpatialDims,
        kernel: SpatialDims,
        stride: SpatialDims,
        pad: SpatialDims,
        options: Conv2dOptions,
    ) -> Result<Self> {
        let Conv2dOptions {
            groups,
            activation,
        } = options;
        let Conv2dChannels {
            input: input_channels,
            output: output_channels,
        } = channels;
        let SpatialDims {
            width: input_width,
            height: input_height,
        } = input;
        let SpatialDims {
            width: kernel_width,
            height: kernel_height,
        } = kernel;
        let SpatialDims {
            width: stride_x,
            height: stride_y,
        } = stride;
        let SpatialDims {
            width: pad_x,
            height: pad_y,
        } = pad;
        anyhow::ensure!(batch == 1, "only batch size 1 is supported (got {batch})");
        anyhow::ensure!(input_channels > 0, "input channels must be > 0");
        anyhow::ensure!(output_channels > 0, "output channels must be > 0");
        anyhow::ensure!(
            kernel_width > 0 && kernel_height > 0,
            "kernel must be non-zero"
        );
        anyhow::ensure!(stride_x > 0 && stride_y > 0, "stride must be non-zero");
        anyhow::ensure!(groups > 0, "groups must be > 0");
        anyhow::ensure!(
            input_channels.is_multiple_of(groups),
            "input channels ({input_channels}) must be divisible by groups ({groups})"
        );
        anyhow::ensure!(
            output_channels.is_multiple_of(groups),
            "output channels ({output_channels}) must be divisible by groups ({groups})"
        );

        let output_width = compute_output_dim(input_width, pad_x, kernel_width, stride_x)
            .context("invalid convolution width configuration")?;
        let output_height = compute_output_dim(input_height, pad_y, kernel_height, stride_y)
            .context("invalid convolution height configuration")?;

        Ok(Self {
            batch,
            input_channels,
            output_channels,
            input_width,
            input_height,
            kernel_width,
            kernel_height,
            stride_x,
            stride_y,
            pad_x,
            pad_y,
            groups,
            output_width,
            output_height,
            activation,
        })
    }

    fn validate(&self, input_len: usize, weight_len: usize, bias_len: usize) -> Result<()> {
        let expected_input = self.batch as usize
            * self.input_channels as usize
            * self.input_height as usize
            * self.input_width as usize;
        anyhow::ensure!(
            input_len == expected_input,
            "conv input tensor expected {expected_input} elements, got {input_len}"
        );

        let weights_per_out = (self.input_channels / self.groups) as usize
            * self.kernel_width as usize
            * self.kernel_height as usize;
        let expected_weights = self.output_channels as usize * weights_per_out;
        anyhow::ensure!(
            weight_len == expected_weights,
            "conv weights expected {expected_weights} elements, got {weight_len}"
        );
        anyhow::ensure!(
            bias_len == self.output_channels as usize,
            "conv bias expected {} elements, got {bias_len}",
            self.output_channels
        );
        Ok(())
    }

    pub fn input_shape_dims(&self) -> [usize; 4] {
        [
            self.batch as usize,
            self.input_channels as usize,
            self.input_height as usize,
            self.input_width as usize,
        ]
    }

    pub fn output_shape_dims(&self) -> [usize; 4] {
        [
            self.batch as usize,
            self.output_channels as usize,
            self.output_height as usize,
            self.output_width as usize,
        ]
    }

    pub fn weight_shape_dims(&self) -> [usize; 4] {
        [
            self.output_channels as usize,
            (self.input_channels / self.groups) as usize,
            self.kernel_height as usize,
            self.kernel_width as usize,
        ]
    }

    pub fn bias_shape_dims(&self) -> [usize; 1] {
        [self.output_channels as usize]
    }

    #[cfg(test)]
    fn output_element_count(&self) -> usize {
        self.batch as usize
            * self.output_channels as usize
            * self.output_height as usize
            * self.output_width as usize
    }
}

/// Batch-norm tensor dimensions.
#[derive(Debug, Clone)]
pub struct BatchNormConfig {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub epsilon: f32,
}

/// Max pool configuration (batch size is currently restricted to 1).
#[derive(Debug, Clone)]
pub struct MaxPoolConfig {
    pub batch: u32,
    pub channels: u32,
    pub input_width: u32,
    pub input_height: u32,
    pub kernel: u32,
    pub stride: u32,
    pub pad: u32,
    pub output_width: u32,
    pub output_height: u32,
}

impl MaxPoolConfig {
    pub fn new(
        batch: u32,
        channels: u32,
        input_width: u32,
        input_height: u32,
        kernel: u32,
        stride: u32,
        pad: u32,
    ) -> Result<Self> {
        anyhow::ensure!(batch == 1, "only batch size 1 is supported (got {batch})");
        anyhow::ensure!(channels > 0, "channels must be > 0");
        anyhow::ensure!(
            input_width > 0 && input_height > 0,
            "spatial dims must be > 0"
        );
        anyhow::ensure!(kernel > 0, "kernel must be > 0");
        anyhow::ensure!(stride > 0, "stride must be > 0");
        let output_width = compute_output_dim(input_width, pad, kernel, stride)
            .context("pool width config invalid")?;
        let output_height = compute_output_dim(input_height, pad, kernel, stride)
            .context("pool height config invalid")?;
        Ok(Self {
            batch,
            channels,
            input_width,
            input_height,
            kernel,
            stride,
            pad,
            output_width,
            output_height,
        })
    }

    pub fn from_tensor(tensor: &GpuTensor, kernel: u32, stride: u32, pad: u32) -> Result<Self> {
        let dims = tensor.shape().dims();
        anyhow::ensure!(
            dims.len() == 4,
            "max pool only supports 4D tensors (got dims {:?})",
            dims
        );
        Self::new(
            dims[0] as u32,
            dims[1] as u32,
            dims[3] as u32,
            dims[2] as u32,
            kernel,
            stride,
            pad,
        )
    }

    pub fn output_dims(&self) -> [usize; 4] {
        [
            self.batch as usize,
            self.channels as usize,
            self.output_height as usize,
            self.output_width as usize,
        ]
    }
}

impl BatchNormConfig {
    /// Create a validated batch-norm configuration.
    pub fn new(width: u32, height: u32, channels: u32, epsilon: f32) -> Result<Self> {
        anyhow::ensure!(width > 0 && height > 0, "spatial dimensions must be > 0");
        anyhow::ensure!(channels > 0, "channels must be > 0");
        anyhow::ensure!(epsilon >= 0.0, "epsilon must be >= 0");
        Ok(Self {
            width,
            height,
            channels,
            epsilon,
        })
    }

    fn element_count(&self) -> usize {
        self.width as usize * self.height as usize * self.channels as usize
    }

    fn validate(
        &self,
        tensor_len: usize,
        gamma_len: usize,
        beta_len: usize,
        mean_len: usize,
        variance_len: usize,
    ) -> Result<()> {
        let expected_spatial = self.element_count();
        anyhow::ensure!(
            tensor_len == expected_spatial,
            "batch-norm tensor expected {expected_spatial} values, got {tensor_len}"
        );
        let channels = self.channels as usize;
        for (name, len) in [
            ("gamma", gamma_len),
            ("beta", beta_len),
            ("mean", mean_len),
            ("variance", variance_len),
        ] {
            anyhow::ensure!(
                len == channels,
                "batch-norm {name} expected {channels} values, got {len}"
            );
        }
        Ok(())
    }

    pub fn tensor_shape_dims(&self) -> [usize; 3] {
        [
            self.channels as usize,
            self.height as usize,
            self.width as usize,
        ]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Conv2dUniforms {
    input_width: u32,
    input_height: u32,
    input_channels: u32,
    output_width: u32,
    output_height: u32,
    output_channels: u32,
    kernel_width: u32,
    kernel_height: u32,
    stride_x: u32,
    stride_y: u32,
    pad_x: u32,
    pad_y: u32,
    groups: u32,
    activation_mode: u32,
}

impl From<&Conv2dConfig> for Conv2dUniforms {
    fn from(value: &Conv2dConfig) -> Self {
        Self {
            input_width: value.input_width,
            input_height: value.input_height,
            input_channels: value.input_channels,
            output_width: value.output_width,
            output_height: value.output_height,
            output_channels: value.output_channels,
            kernel_width: value.kernel_width,
            kernel_height: value.kernel_height,
            stride_x: value.stride_x,
            stride_y: value.stride_y,
            pad_x: value.pad_x,
            pad_y: value.pad_y,
            groups: value.groups,
            activation_mode: value
                .activation
                .map(|k| match k {
                    ActivationKind::Relu => 1,
                    ActivationKind::Sigmoid => 2,
                })
                .unwrap_or(0),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BatchNormUniforms {
    width: u32,
    height: u32,
    channels: u32,
    epsilon: f32,
}

impl From<&BatchNormConfig> for BatchNormUniforms {
    fn from(value: &BatchNormConfig) -> Self {
        Self {
            width: value.width,
            height: value.height,
            channels: value.channels,
            epsilon: value.epsilon,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ActivationUniforms {
    len: u32,
    mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaxPoolUniforms {
    input_width: u32,
    input_height: u32,
    channels: u32,
    output_width: u32,
    output_height: u32,
    kernel: u32,
    stride: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AddUniforms {
    len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UpsampleUniforms {
    input_width: u32,
    input_height: u32,
    channels: u32,
    _padding: u32,
}

fn compute_output_dim(size: u32, pad: u32, kernel: u32, stride: u32) -> Result<u32> {
    anyhow::ensure!(stride > 0, "stride must be > 0");
    anyhow::ensure!(kernel > 0, "kernel must be > 0");
    let numerator = size
        .checked_add(pad * 2)
        .context("padding overflowed u32")?
        .checked_sub(kernel)
        .context("kernel larger than padded input")?;
    Ok(numerator / stride + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use tract_onnx::prelude::*;
    use yunet_utils::gpu::{GpuAvailability, GpuContextOptions};

    use crate::gpu::{OnnxInitializerMap, OnnxTensor};

    fn gpu_ops() -> Option<GpuInferenceOps> {
        match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
            GpuAvailability::Available(ctx) => Some(GpuInferenceOps::new(ctx).expect("build ops")),
            _ => None,
        }
    }

    fn synthetic_input() -> Vec<f32> {
        let input_shape = 1 * 3 * 640 * 640;
        (0..input_shape)
            .map(|i| ((i % 257) as f32) / 256.0)
            .collect()
    }

    fn upload_onx_tensor(
        ops: &GpuInferenceOps,
        tensor: &OnnxTensor,
        label: &str,
    ) -> Result<GpuTensor> {
        ops.upload_tensor(tensor.dims().to_vec(), tensor.data(), Some(label))
    }

    const MODEL_REL_PATH: &str = "models/face_detection_yunet_2023mar_640.onnx";

    fn model_file_path() -> Option<PathBuf> {
        let mut candidates = Vec::new();
        candidates.push(PathBuf::from(MODEL_REL_PATH));

        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        if let Some(workspace_root) = manifest_dir.parent() {
            candidates.push(workspace_root.join(MODEL_REL_PATH));
        }

        candidates.into_iter().find(|path| path.exists())
    }

    fn reference_tensor(model_path: &Path, node: &str, input: &[f32]) -> (Vec<f32>, Vec<usize>) {
        let mut model = tract_onnx::onnx()
            .model_for_path(model_path)
            .expect("load reference ONNX");
        model
            .set_output_names(&[node])
            .expect("set reference output");
        let plan = model
            .into_optimized()
            .expect("optimize reference")
            .into_runnable()
            .expect("plan reference graph");
        let arr = tract_ndarray::Array4::from_shape_vec((1, 3, 640, 640), input.to_vec()).unwrap();
        let tensor = plan
            .run(tvec!(arr.into_tensor().into()))
            .expect("run reference graph")
            .remove(0)
            .into_tensor()
            .into_array::<f32>()
            .expect("convert reference output");
        let shape = tensor
            .shape()
            .iter()
            .map(|d| *d as usize)
            .collect::<Vec<_>>();
        (tensor.into_raw_vec_and_offset().0, shape)
    }

    fn reference_output(model_path: &Path, node: &str, input: &[f32]) -> Vec<f32> {
        reference_tensor(model_path, node, input).0
    }

    fn assert_tensor_matches(model_path: &Path, node: &str, tensor: &GpuTensor, input: &[f32]) {
        let cpu_vec = reference_output(model_path, node, input);
        let gpu_vec = tensor.to_vec().expect("download tensor");
        assert_eq!(
            gpu_vec.len(),
            cpu_vec.len(),
            "node {node} produced mismatched length"
        );
        let max_diff = gpu_vec
            .iter()
            .zip(cpu_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "node {node} mismatch (max diff {max_diff})"
        );
    }

    fn upload_named(
        ops: &GpuInferenceOps,
        loader: &OnnxInitializerMap,
        name: &str,
    ) -> Result<GpuTensor> {
        let tensor = loader
            .tensor(name)
            .with_context(|| format!("initializer '{name}' missing"))?;
        upload_onx_tensor(ops, tensor, name)
    }

    #[derive(Clone, Copy)]
    struct StageBlock {
        point_weight: &'static str,
        point_bias: &'static str,
        depth_weight: &'static str,
        depth_bias: &'static str,
    }

    struct BackboneStage {
        blocks: &'static [StageBlock],
        pool_before: bool,
    }

    const STAGE1_BLOCKS: [StageBlock; 2] = [
        StageBlock {
            point_weight: "backbone.model1.conv1.conv1.weight",
            point_bias: "backbone.model1.conv1.conv1.bias",
            depth_weight: "426",
            depth_bias: "427",
        },
        StageBlock {
            point_weight: "backbone.model1.conv2.conv1.weight",
            point_bias: "backbone.model1.conv2.conv1.bias",
            depth_weight: "429",
            depth_bias: "430",
        },
    ];

    const STAGE2_BLOCKS: [StageBlock; 2] = [
        StageBlock {
            point_weight: "backbone.model2.conv1.conv1.weight",
            point_bias: "backbone.model2.conv1.conv1.bias",
            depth_weight: "432",
            depth_bias: "433",
        },
        StageBlock {
            point_weight: "backbone.model2.conv2.conv1.weight",
            point_bias: "backbone.model2.conv2.conv1.bias",
            depth_weight: "435",
            depth_bias: "436",
        },
    ];

    const STAGE3_BLOCKS: [StageBlock; 2] = [
        StageBlock {
            point_weight: "backbone.model3.conv1.conv1.weight",
            point_bias: "backbone.model3.conv1.conv1.bias",
            depth_weight: "438",
            depth_bias: "439",
        },
        StageBlock {
            point_weight: "backbone.model3.conv2.conv1.weight",
            point_bias: "backbone.model3.conv2.conv1.bias",
            depth_weight: "441",
            depth_bias: "442",
        },
    ];

    const STAGE4_BLOCKS: [StageBlock; 2] = [
        StageBlock {
            point_weight: "backbone.model4.conv1.conv1.weight",
            point_bias: "backbone.model4.conv1.conv1.bias",
            depth_weight: "444",
            depth_bias: "445",
        },
        StageBlock {
            point_weight: "backbone.model4.conv2.conv1.weight",
            point_bias: "backbone.model4.conv2.conv1.bias",
            depth_weight: "447",
            depth_bias: "448",
        },
    ];

    const STAGE5_BLOCKS: [StageBlock; 2] = [
        StageBlock {
            point_weight: "backbone.model5.conv1.conv1.weight",
            point_bias: "backbone.model5.conv1.conv1.bias",
            depth_weight: "450",
            depth_bias: "451",
        },
        StageBlock {
            point_weight: "backbone.model5.conv2.conv1.weight",
            point_bias: "backbone.model5.conv2.conv1.bias",
            depth_weight: "453",
            depth_bias: "454",
        },
    ];

    const BACKBONE_STAGES: [BackboneStage; 5] = [
        BackboneStage {
            blocks: &STAGE1_BLOCKS,
            pool_before: true,
        },
        BackboneStage {
            blocks: &STAGE2_BLOCKS,
            pool_before: false,
        },
        BackboneStage {
            blocks: &STAGE3_BLOCKS,
            pool_before: true,
        },
        BackboneStage {
            blocks: &STAGE4_BLOCKS,
            pool_before: true,
        },
        BackboneStage {
            blocks: &STAGE5_BLOCKS,
            pool_before: true,
        },
    ];

    const STAGE0_WEIGHT_NAMES: [&str; 6] = [
        "420",
        "421",
        "backbone.model0.conv2.conv1.weight",
        "backbone.model0.conv2.conv1.bias",
        "423",
        "424",
    ];

    const NECK_BLOCKS: [StageBlock; 3] = [
        StageBlock {
            point_weight: "neck.lateral_convs.0.conv1.weight",
            point_bias: "neck.lateral_convs.0.conv1.bias",
            depth_weight: "462",
            depth_bias: "463",
        },
        StageBlock {
            point_weight: "neck.lateral_convs.1.conv1.weight",
            point_bias: "neck.lateral_convs.1.conv1.bias",
            depth_weight: "459",
            depth_bias: "460",
        },
        StageBlock {
            point_weight: "neck.lateral_convs.2.conv1.weight",
            point_bias: "neck.lateral_convs.2.conv1.bias",
            depth_weight: "456",
            depth_bias: "457",
        },
    ];

    struct HeadBlock {
        conv1_weight: &'static str,
        conv1_bias: &'static str,
        conv2_weight: &'static str,
        conv2_bias: &'static str,
    }

    impl HeadBlock {
        fn names(&self) -> [&'static str; 4] {
            [
                self.conv1_weight,
                self.conv1_bias,
                self.conv2_weight,
                self.conv2_bias,
            ]
        }
    }

    struct DetectionHeadConfig {
        cls: HeadBlock,
        obj: HeadBlock,
        bbox: HeadBlock,
        kps: HeadBlock,
    }

    const DETECTION_HEADS: [DetectionHeadConfig; 3] = [
        DetectionHeadConfig {
            cls: HeadBlock {
                conv1_weight: "bbox_head.multi_level_cls.0.conv1.weight",
                conv1_bias: "bbox_head.multi_level_cls.0.conv1.bias",
                conv2_weight: "bbox_head.multi_level_cls.0.conv2.weight",
                conv2_bias: "bbox_head.multi_level_cls.0.conv2.bias",
            },
            obj: HeadBlock {
                conv1_weight: "bbox_head.multi_level_obj.0.conv1.weight",
                conv1_bias: "bbox_head.multi_level_obj.0.conv1.bias",
                conv2_weight: "bbox_head.multi_level_obj.0.conv2.weight",
                conv2_bias: "bbox_head.multi_level_obj.0.conv2.bias",
            },
            bbox: HeadBlock {
                conv1_weight: "bbox_head.multi_level_bbox.0.conv1.weight",
                conv1_bias: "bbox_head.multi_level_bbox.0.conv1.bias",
                conv2_weight: "bbox_head.multi_level_bbox.0.conv2.weight",
                conv2_bias: "bbox_head.multi_level_bbox.0.conv2.bias",
            },
            kps: HeadBlock {
                conv1_weight: "bbox_head.multi_level_kps.0.conv1.weight",
                conv1_bias: "bbox_head.multi_level_kps.0.conv1.bias",
                conv2_weight: "bbox_head.multi_level_kps.0.conv2.weight",
                conv2_bias: "bbox_head.multi_level_kps.0.conv2.bias",
            },
        },
        DetectionHeadConfig {
            cls: HeadBlock {
                conv1_weight: "bbox_head.multi_level_cls.1.conv1.weight",
                conv1_bias: "bbox_head.multi_level_cls.1.conv1.bias",
                conv2_weight: "bbox_head.multi_level_cls.1.conv2.weight",
                conv2_bias: "bbox_head.multi_level_cls.1.conv2.bias",
            },
            obj: HeadBlock {
                conv1_weight: "bbox_head.multi_level_obj.1.conv1.weight",
                conv1_bias: "bbox_head.multi_level_obj.1.conv1.bias",
                conv2_weight: "bbox_head.multi_level_obj.1.conv2.weight",
                conv2_bias: "bbox_head.multi_level_obj.1.conv2.bias",
            },
            bbox: HeadBlock {
                conv1_weight: "bbox_head.multi_level_bbox.1.conv1.weight",
                conv1_bias: "bbox_head.multi_level_bbox.1.conv1.bias",
                conv2_weight: "bbox_head.multi_level_bbox.1.conv2.weight",
                conv2_bias: "bbox_head.multi_level_bbox.1.conv2.bias",
            },
            kps: HeadBlock {
                conv1_weight: "bbox_head.multi_level_kps.1.conv1.weight",
                conv1_bias: "bbox_head.multi_level_kps.1.conv1.bias",
                conv2_weight: "bbox_head.multi_level_kps.1.conv2.weight",
                conv2_bias: "bbox_head.multi_level_kps.1.conv2.bias",
            },
        },
        DetectionHeadConfig {
            cls: HeadBlock {
                conv1_weight: "bbox_head.multi_level_cls.2.conv1.weight",
                conv1_bias: "bbox_head.multi_level_cls.2.conv1.bias",
                conv2_weight: "bbox_head.multi_level_cls.2.conv2.weight",
                conv2_bias: "bbox_head.multi_level_cls.2.conv2.bias",
            },
            obj: HeadBlock {
                conv1_weight: "bbox_head.multi_level_obj.2.conv1.weight",
                conv1_bias: "bbox_head.multi_level_obj.2.conv1.bias",
                conv2_weight: "bbox_head.multi_level_obj.2.conv2.weight",
                conv2_bias: "bbox_head.multi_level_obj.2.conv2.bias",
            },
            bbox: HeadBlock {
                conv1_weight: "bbox_head.multi_level_bbox.2.conv1.weight",
                conv1_bias: "bbox_head.multi_level_bbox.2.conv1.bias",
                conv2_weight: "bbox_head.multi_level_bbox.2.conv2.weight",
                conv2_bias: "bbox_head.multi_level_bbox.2.conv2.bias",
            },
            kps: HeadBlock {
                conv1_weight: "bbox_head.multi_level_kps.2.conv1.weight",
                conv1_bias: "bbox_head.multi_level_kps.2.conv1.bias",
                conv2_weight: "bbox_head.multi_level_kps.2.conv2.weight",
                conv2_bias: "bbox_head.multi_level_kps.2.conv2.bias",
            },
        },
    ];

    fn load_backbone_weights(
        model_path: &Path,
        stage_count: usize,
        include_neck: bool,
        head_levels: usize,
    ) -> Result<OnnxInitializerMap> {
        anyhow::ensure!(
            stage_count <= BACKBONE_STAGES.len(),
            "requested {stage_count} backbone stages but only {} are available",
            BACKBONE_STAGES.len()
        );
        anyhow::ensure!(
            head_levels <= DETECTION_HEADS.len(),
            "requested {head_levels} head levels but only {} are available",
            DETECTION_HEADS.len()
        );
        let mut names: Vec<&str> = Vec::new();
        names.extend_from_slice(&STAGE0_WEIGHT_NAMES);
        for stage in BACKBONE_STAGES.iter().take(stage_count) {
            for block in stage.blocks.iter() {
                names.push(block.point_weight);
                names.push(block.point_bias);
                names.push(block.depth_weight);
                names.push(block.depth_bias);
            }
        }
        if include_neck {
            for block in NECK_BLOCKS.iter() {
                names.push(block.point_weight);
                names.push(block.point_bias);
                names.push(block.depth_weight);
                names.push(block.depth_bias);
            }
        }
        if head_levels > 0 {
            for head in DETECTION_HEADS.iter().take(head_levels) {
                names.extend_from_slice(&head.cls.names());
                names.extend_from_slice(&head.obj.names());
                names.extend_from_slice(&head.bbox.names());
                names.extend_from_slice(&head.kps.names());
            }
        }
        OnnxInitializerMap::load(model_path, &names)
    }

    fn run_stage_blocks(
        ops: &GpuInferenceOps,
        loader: &OnnxInitializerMap,
        input: &GpuTensor,
        blocks: &[StageBlock],
    ) -> Result<GpuTensor> {
        let Some((first, rest)) = blocks.split_first() else {
            anyhow::bail!("stage block list cannot be empty");
        };
        let mut current = run_stage_block(
            ops,
            loader,
            input,
            first.point_weight,
            first.point_bias,
            first.depth_weight,
            first.depth_bias,
        )?;
        for block in rest {
            current = run_stage_block(
                ops,
                loader,
                &current,
                block.point_weight,
                block.point_bias,
                block.depth_weight,
                block.depth_bias,
            )?;
        }
        Ok(current)
    }

    fn run_backbone_to_stage(
        ops: &GpuInferenceOps,
        loader: &OnnxInitializerMap,
        input: &GpuTensor,
        stage_count: usize,
    ) -> Result<GpuTensor> {
        anyhow::ensure!(stage_count > 0, "stage_count must be > 0");
        anyhow::ensure!(
            stage_count <= BACKBONE_STAGES.len(),
            "requested stage {} exceeds available {}",
            stage_count,
            BACKBONE_STAGES.len()
        );
        let mut current = run_stage0_block(ops, loader, input)?;
        for stage in BACKBONE_STAGES.iter().take(stage_count) {
            if stage.pool_before {
                current = pool_tensor(ops, &current)?;
            }
            current = run_stage_blocks(ops, loader, &current, stage.blocks)?;
        }
        Ok(current)
    }

    fn run_stage0_block(
        ops: &GpuInferenceOps,
        loader: &OnnxInitializerMap,
        input: &GpuTensor,
    ) -> Result<GpuTensor> {
        let conv0_weight = upload_named(ops, loader, "420")?;
        let conv0_bias = upload_named(ops, loader, "421")?;
        let pw_weight = upload_named(ops, loader, "backbone.model0.conv2.conv1.weight")?;
        let pw_bias = upload_named(ops, loader, "backbone.model0.conv2.conv1.bias")?;
        let dw_weight = upload_named(ops, loader, "423")?;
        let dw_bias = upload_named(ops, loader, "424")?;

        let conv_cfg = Conv2dConfig::new(
            1,
            Conv2dChannels::new(3, 16),
            SpatialDims::new(640, 640),
            SpatialDims::new(3, 3),
            SpatialDims::new(2, 2),
            SpatialDims::new(1, 1),
            Conv2dOptions::new(1, Some(ActivationKind::Relu)),
        )
        .expect("conv config");
        let relu0 = ops
            .conv2d_tensor(input, &conv0_weight, &conv0_bias, &conv_cfg)
            .expect("stage0 conv");

        let point_cfg = Conv2dConfig::new(
            1,
            Conv2dChannels::new(16, 16),
            SpatialDims::new(320, 320),
            SpatialDims::new(1, 1),
            SpatialDims::new(1, 1),
            SpatialDims::new(0, 0),
            Conv2dOptions::new(1, None),
        )
        .unwrap();
        let point = ops
            .conv2d_tensor(&relu0, &pw_weight, &pw_bias, &point_cfg)
            .expect("stage0 pw");

        let depth_cfg = Conv2dConfig::new(
            1,
            Conv2dChannels::new(16, 16),
            SpatialDims::new(320, 320),
            SpatialDims::new(3, 3),
            SpatialDims::new(1, 1),
            SpatialDims::new(1, 1),
            Conv2dOptions::new(16, Some(ActivationKind::Relu)),
        )
        .unwrap();
        ops.conv2d_tensor(&point, &dw_weight, &dw_bias, &depth_cfg)
    }

    fn run_backbone_features(
        ops: &GpuInferenceOps,
        loader: &OnnxInitializerMap,
        input: &GpuTensor,
        stage_count: usize,
    ) -> Result<Vec<GpuTensor>> {
        anyhow::ensure!(
            stage_count <= BACKBONE_STAGES.len(),
            "requested {stage_count} backbone outputs but only {} exist",
            BACKBONE_STAGES.len()
        );
        let mut features = Vec::with_capacity(stage_count);
        let mut current = run_stage0_block(ops, loader, input)?;
        for stage in BACKBONE_STAGES.iter().take(stage_count) {
            if stage.pool_before {
                current = pool_tensor(ops, &current)?;
            }
            current = run_stage_blocks(ops, loader, &current, stage.blocks)?;
            features.push(current.clone());
        }
        Ok(features)
    }

    fn pool_tensor(ops: &GpuInferenceOps, tensor: &GpuTensor) -> Result<GpuTensor> {
        let cfg = MaxPoolConfig::from_tensor(tensor, 2, 2, 0)?;
        ops.max_pool_tensor(tensor, &cfg)
    }

    struct DetectionLevelOutputs {
        feature: GpuTensor,
        cls: GpuTensor,
        obj: GpuTensor,
        bbox: GpuTensor,
        kps: GpuTensor,
    }

    fn run_neck_and_heads(
        ops: &GpuInferenceOps,
        loader: &OnnxInitializerMap,
        features: &[GpuTensor],
    ) -> Result<[DetectionLevelOutputs; 3]> {
        anyhow::ensure!(
            features.len() >= 5,
            "need at least five backbone outputs (got {})",
            features.len()
        );
        let c3 = features[2].clone();
        let c4 = features[3].clone();
        let c5 = features[4].clone();

        let p5_raw = run_stage_blocks(ops, loader, &c5, &NECK_BLOCKS[2..3])?;
        let level2 = run_detection_level(ops, loader, p5_raw.clone(), &DETECTION_HEADS[2])?;

        let up_p5 = ops.resize2x_tensor(&p5_raw)?;
        let merged_p4_input = ops.add_tensors(&up_p5, &c4)?;
        let p4_raw = run_stage_blocks(ops, loader, &merged_p4_input, &NECK_BLOCKS[1..2])?;
        let level1 = run_detection_level(ops, loader, p4_raw.clone(), &DETECTION_HEADS[1])?;

        let up_p4 = ops.resize2x_tensor(&p4_raw)?;
        let merged_p3_input = ops.add_tensors(&up_p4, &c3)?;
        let p3_raw = run_stage_blocks(ops, loader, &merged_p3_input, &NECK_BLOCKS[0..1])?;
        let level0 = run_detection_level(ops, loader, p3_raw.clone(), &DETECTION_HEADS[0])?;

        Ok([level0, level1, level2])
    }

    fn run_detection_level(
        ops: &GpuInferenceOps,
        loader: &OnnxInitializerMap,
        feature: GpuTensor,
        head: &DetectionHeadConfig,
    ) -> Result<DetectionLevelOutputs> {
        let cls = run_head_branch(ops, loader, &feature, &head.cls)?;
        let obj = run_head_branch(ops, loader, &feature, &head.obj)?;
        let bbox = run_head_branch(ops, loader, &feature, &head.bbox)?;
        let kps = run_head_branch(ops, loader, &feature, &head.kps)?;
        Ok(DetectionLevelOutputs {
            feature,
            cls,
            obj,
            bbox,
            kps,
        })
    }

    fn run_head_branch(
        ops: &GpuInferenceOps,
        loader: &OnnxInitializerMap,
        input: &GpuTensor,
        branch: &HeadBlock,
    ) -> Result<GpuTensor> {
        let point_weight = upload_named(ops, loader, branch.conv1_weight)?;
        let point_bias = upload_named(ops, loader, branch.conv1_bias)?;
        let depth_weight = upload_named(ops, loader, branch.conv2_weight)?;
        let depth_bias = upload_named(ops, loader, branch.conv2_bias)?;

        let dims = input.shape().dims();
        anyhow::ensure!(
            dims.len() == 4,
            "head branch expects NCHW tensor (got {:?})",
            dims
        );
        let batch = dims[0] as u32;
        let in_channels = dims[1] as u32;
        let height = dims[2] as u32;
        let width = dims[3] as u32;
        let point_out = point_weight.shape().dims()[0] as u32;

        let point_cfg = Conv2dConfig::new(
            batch,
            Conv2dChannels::new(in_channels, point_out),
            SpatialDims::new(width, height),
            SpatialDims::new(1, 1),
            SpatialDims::new(1, 1),
            SpatialDims::new(0, 0),
            Conv2dOptions::new(1, None),
        )?;
        let reduced = ops.conv2d_tensor(input, &point_weight, &point_bias, &point_cfg)?;

        let depth_out = depth_weight.shape().dims()[0] as u32;
        anyhow::ensure!(
            depth_out == point_out,
            "depthwise conv expects {} channels but got {}",
            point_out,
            depth_out
        );
        let depth_cfg = Conv2dConfig::new(
            batch,
            Conv2dChannels::new(point_out, depth_out),
            SpatialDims::new(width, height),
            SpatialDims::new(3, 3),
            SpatialDims::new(1, 1),
            SpatialDims::new(1, 1),
            Conv2dOptions::new(depth_out, None),
        )?;
        ops.conv2d_tensor(&reduced, &depth_weight, &depth_bias, &depth_cfg)
    }

    fn run_separable_block(
        ops: &GpuInferenceOps,
        input: &GpuTensor,
        point_weight: &GpuTensor,
        point_bias: &GpuTensor,
        depth_weight: &GpuTensor,
        depth_bias: &GpuTensor,
    ) -> Result<GpuTensor> {
        let dims = input.shape().dims();
        anyhow::ensure!(
            dims.len() == 4,
            "expected NCHW tensor for separable block (got {:?})",
            dims
        );
        let batch = dims[0] as u32;
        let channels = dims[1] as u32;
        let height = dims[2] as u32;
        let width = dims[3] as u32;

        let point_shape = point_weight.shape().dims();
        anyhow::ensure!(
            point_shape.len() == 4,
            "pointwise weights must be 4D (got {:?})",
            point_shape
        );
        let point_out = point_shape[0] as u32;
        let point_kernel_h = point_shape[2] as u32;
        let point_kernel_w = point_shape[3] as u32;
        anyhow::ensure!(
            point_kernel_h == 1 && point_kernel_w == 1,
            "pointwise kernels must be 1x1 (got {}x{})",
            point_kernel_h,
            point_kernel_w
        );

        let point_cfg = Conv2dConfig::new(
            batch,
            Conv2dChannels::new(channels, point_out),
            SpatialDims::new(width, height),
            SpatialDims::new(1, 1),
            SpatialDims::new(1, 1),
            SpatialDims::new(0, 0),
            Conv2dOptions::new(1, None),
        )?;
        let point = ops.conv2d_tensor(input, point_weight, point_bias, &point_cfg)?;

        let depth_shape = depth_weight.shape().dims();
        anyhow::ensure!(
            depth_shape.len() == 4,
            "depthwise weights must be 4D (got {:?})",
            depth_shape
        );
        let depth_out = depth_shape[0] as u32;
        anyhow::ensure!(
            depth_out == point_out,
            "depthwise output ({depth_out}) must match pointwise output ({point_out})"
        );
        anyhow::ensure!(
            depth_shape[1] as u32 == 1,
            "depthwise weights expect channel multiplier 1 (got {})",
            depth_shape[1]
        );
        let depth_kernel_h = depth_shape[2] as u32;
        let depth_kernel_w = depth_shape[3] as u32;
        anyhow::ensure!(
            depth_kernel_h == depth_kernel_w,
            "depthwise kernels must be square (got {}x{})",
            depth_kernel_h,
            depth_kernel_w
        );
        let pad = depth_kernel_w / 2;

        let depth_cfg = Conv2dConfig::new(
            batch,
            Conv2dChannels::new(point_out, depth_out),
            SpatialDims::new(width, height),
            SpatialDims::new(depth_kernel_w, depth_kernel_h),
            SpatialDims::new(1, 1),
            SpatialDims::new(pad, pad),
            Conv2dOptions::new(depth_out, Some(ActivationKind::Relu)),
        )?;
        ops.conv2d_tensor(&point, depth_weight, depth_bias, &depth_cfg)
    }

    fn run_stage_block(
        ops: &GpuInferenceOps,
        loader: &OnnxInitializerMap,
        input: &GpuTensor,
        point_weight: &str,
        point_bias: &str,
        depth_weight: &str,
        depth_bias: &str,
    ) -> Result<GpuTensor> {
        let pw = upload_named(ops, loader, point_weight)?;
        let pb = upload_named(ops, loader, point_bias)?;
        let dw = upload_named(ops, loader, depth_weight)?;
        let db = upload_named(ops, loader, depth_bias)?;
        run_separable_block(ops, input, &pw, &pb, &dw, &db)
    }

    #[test]
    fn activation_matches_cpu() {
        let Some(ops) = gpu_ops() else {
            eprintln!("Skipping activation GPU test (no adapter)");
            return;
        };
        let tensor: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
        let relu_gpu = ops.activation(&tensor, ActivationKind::Relu).unwrap();
        let relu_cpu: Vec<f32> = tensor.iter().map(|v| v.max(0.0)).collect();
        assert_eq!(relu_gpu, relu_cpu);

        let sigmoid_gpu = ops.activation(&tensor, ActivationKind::Sigmoid).unwrap();
        let sigmoid_cpu: Vec<f32> = tensor.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
        assert!(
            sigmoid_gpu
                .iter()
                .zip(sigmoid_cpu.iter())
                .all(|(a, b)| (a - b).abs() < 1e-4),
            "sigmoid mismatch"
        );
    }

    #[test]
    fn batch_norm_matches_cpu() {
        let Some(ops) = gpu_ops() else {
            eprintln!("Skipping batch-norm GPU test (no adapter)");
            return;
        };
        let config = BatchNormConfig::new(4, 2, 3, 1e-5).unwrap();
        let tensor: Vec<f32> = (0..24).map(|i| (i % 7) as f32 * 0.25).collect();
        let gamma: Vec<f32> = vec![1.0, 0.75, 1.25];
        let beta: Vec<f32> = vec![0.0, 0.1, -0.1];
        let mean: Vec<f32> = vec![0.5, 0.4, 0.3];
        let variance: Vec<f32> = vec![0.2, 0.3, 0.1];
        let gpu = ops
            .batch_norm(&tensor, &gamma, &beta, &mean, &variance, &config)
            .unwrap();
        let cpu = batch_norm_cpu(&tensor, &gamma, &beta, &mean, &variance, &config);
        assert!(
            gpu.iter()
                .zip(cpu.iter())
                .all(|(a, b)| (a - b).abs() < 1e-4),
            "batch-norm mismatch"
        );
    }

    #[test]
    fn conv2d_matches_cpu_groups() {
        let Some(ops) = gpu_ops() else {
            eprintln!("Skipping conv2d GPU test (no adapter)");
            return;
        };
        let config = Conv2dConfig::new(
            1,
            Conv2dChannels::new(4, 4),
            SpatialDims::new(4, 4),
            SpatialDims::new(3, 3),
            SpatialDims::new(1, 1),
            SpatialDims::new(1, 1),
            Conv2dOptions::new(2, None),
        )
        .unwrap();
        let input: Vec<f32> = (0..(4 * 4 * 4))
            .map(|i| ((i * 13 % 17) as f32) * 0.1)
            .collect();
        let weights_len = (config.output_channels as usize)
            * ((config.input_channels / config.groups) as usize)
            * config.kernel_width as usize
            * config.kernel_height as usize;
        let weights: Vec<f32> = (0..weights_len)
            .map(|i| ((i * 7 % 19) as f32) * 0.05)
            .collect();
        let bias: Vec<f32> = (0..config.output_channels)
            .map(|i| i as f32 * 0.1 - 0.2)
            .collect();
        let gpu = ops.conv2d(&input, &weights, &bias, &config).unwrap();
        let cpu = conv2d_cpu(&input, &weights, &bias, &config);
        assert!(
            gpu.iter()
                .zip(cpu.iter())
                .all(|(a, b)| (a - b).abs() < 1e-3),
            "conv2d mismatch"
        );
    }

    fn conv2d_cpu(input: &[f32], weights: &[f32], bias: &[f32], cfg: &Conv2dConfig) -> Vec<f32> {
        let mut output = vec![0.0; cfg.output_element_count()];
        let in_c = cfg.input_channels as usize;
        let out_c = cfg.output_channels as usize;
        let in_w = cfg.input_width as usize;
        let in_h = cfg.input_height as usize;
        let k_w = cfg.kernel_width as usize;
        let k_h = cfg.kernel_height as usize;
        let stride_x = cfg.stride_x as usize;
        let stride_y = cfg.stride_y as usize;
        let pad_x = cfg.pad_x as isize;
        let pad_y = cfg.pad_y as isize;
        let groups = cfg.groups as usize;
        let group_in = in_c / groups;
        let group_out = out_c / groups;
        let weights_per_out = group_in * k_w * k_h;
        let out_w = cfg.output_width as usize;
        let out_h = cfg.output_height as usize;

        for oc in 0..out_c {
            let group_idx = oc / group_out;
            let in_start = group_idx * group_in;
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut acc = bias[oc];
                    for ic_local in 0..group_in {
                        let ic = in_start + ic_local;
                        for ky in 0..k_h {
                            for kx in 0..k_w {
                                let ix = ox * stride_x + kx;
                                let iy = oy * stride_y + ky;
                                let ix = ix as isize - pad_x;
                                let iy = iy as isize - pad_y;
                                if ix < 0 || iy < 0 || ix >= in_w as isize || iy >= in_h as isize {
                                    continue;
                                }
                                let input_index = (ic * in_h + iy as usize) * in_w + ix as usize;
                                let weight_index =
                                    oc * weights_per_out + ic_local * k_h * k_w + ky * k_w + kx;
                                acc = input[input_index].mul_add(weights[weight_index], acc);
                            }
                        }
                    }
                    let out_index = (oc * out_h + oy) * out_w + ox;
                    output[out_index] = acc;
                }
            }
        }
        output
    }

    fn batch_norm_cpu(
        tensor: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mean: &[f32],
        variance: &[f32],
        cfg: &BatchNormConfig,
    ) -> Vec<f32> {
        let mut out = tensor.to_vec();
        let width = cfg.width as usize;
        let height = cfg.height as usize;
        let channels = cfg.channels as usize;
        let plane = width * height;
        for c in 0..channels {
            let gamma_c = gamma[c];
            let beta_c = beta[c];
            let mean_c = mean[c];
            let var_c = variance[c];
            let inv_std = 1.0 / (var_c + cfg.epsilon).sqrt();
            for idx in 0..plane {
                let offset = c * plane + idx;
                let gain = inv_std * gamma_c;
                out[offset] = gain.mul_add(out[offset] - mean_c, beta_c);
            }
        }
        out
    }

    #[test]
    fn gpu_tensor_chain_remains_on_device() {
        let Some(ops) = gpu_ops() else {
            eprintln!("Skipping GPU tensor chain test (no adapter available)");
            return;
        };

        let config = Conv2dConfig::new(
            1,
            Conv2dChannels::new(4, 4),
            SpatialDims::new(4, 4),
            SpatialDims::new(3, 3),
            SpatialDims::new(1, 1),
            SpatialDims::new(1, 1),
            Conv2dOptions::new(2, None),
        )
        .unwrap();
        let input: Vec<f32> = (0..(4 * 4 * 4))
            .map(|i| ((i * 17 % 23) as f32) * 0.05)
            .collect();
        let weights_len = (config.output_channels as usize)
            * ((config.input_channels / config.groups) as usize)
            * config.kernel_width as usize
            * config.kernel_height as usize;
        let weights: Vec<f32> = (0..weights_len)
            .map(|i| ((i * 11 % 29) as f32) * 0.03)
            .collect();
        let bias: Vec<f32> = (0..config.output_channels)
            .map(|i| i as f32 * 0.02 - 0.1)
            .collect();

        let input_tensor = ops
            .upload_tensor(config.input_shape_dims(), &input, Some("chain_input"))
            .unwrap();
        let weight_tensor = ops
            .upload_tensor(config.weight_shape_dims(), &weights, Some("chain_weights"))
            .unwrap();
        let bias_tensor = ops
            .upload_tensor(config.bias_shape_dims(), &bias, Some("chain_bias"))
            .unwrap();

        let conv_gpu = ops
            .conv2d_tensor(&input_tensor, &weight_tensor, &bias_tensor, &config)
            .unwrap();
        let relu_gpu = ops
            .activation_tensor(&conv_gpu, ActivationKind::Relu)
            .unwrap();
        let gpu_output = relu_gpu.to_vec().unwrap();

        let cpu_conv = conv2d_cpu(&input, &weights, &bias, &config);
        let relu_cpu: Vec<f32> = cpu_conv.into_iter().map(|v| v.max(0.0)).collect();

        assert!(
            gpu_output
                .iter()
                .zip(relu_cpu.iter())
                .all(|(a, b)| (a - b).abs() < 1e-3),
            "GPU tensor chain diverged from CPU reference"
        );
    }

    #[test]
    fn first_conv_relu_matches_onnx() {
        let Some(ops) = gpu_ops() else {
            eprintln!("Skipping ONNX block parity test (no adapter available)");
            return;
        };

        let Some(model_path) = model_file_path() else {
            eprintln!("Skipping ONNX block parity test (model file {MODEL_REL_PATH} missing)");
            return;
        };

        let loader = OnnxInitializerMap::load(
            &model_path,
            &[
                "420",
                "421",
                "backbone.model0.conv2.conv1.weight",
                "backbone.model0.conv2.conv1.bias",
                "423",
                "424",
            ],
        )
        .expect("load stage0 weights");

        let input_data = synthetic_input();
        let cpu_vec = reference_output(&model_path, "188", &input_data);

        let input_gpu = ops
            .upload_tensor([1usize, 3, 640, 640], &input_data, Some("stage0_input"))
            .expect("upload input");
        let relu_gpu = run_stage0_block(&ops, &loader, &input_gpu).expect("stage0 output");
        let gpu_vec = relu_gpu.to_vec().expect("download gpu relu");

        assert_eq!(gpu_vec.len(), cpu_vec.len(), "shape mismatch");
        let max_diff = gpu_vec
            .iter()
            .zip(cpu_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "first conv+relu mismatch (max diff {max_diff})"
        );
    }

    #[test]
    fn conv_depthwise_block_matches_onnx() {
        let Some(ops) = gpu_ops() else {
            eprintln!("Skipping block parity test (no adapter available)");
            return;
        };

        let Some(model_path) = model_file_path() else {
            eprintln!("Skipping block parity test (model file {MODEL_REL_PATH} missing)");
            return;
        };

        let loader = OnnxInitializerMap::load(
            &model_path,
            &[
                "420",
                "421",
                "backbone.model0.conv2.conv1.weight",
                "backbone.model0.conv2.conv1.bias",
                "423",
                "424",
            ],
        )
        .expect("load block weights");

        let input_data = synthetic_input();
        let cpu_vec = reference_output(&model_path, "188", &input_data);

        let input_gpu = ops
            .upload_tensor([1usize, 3, 640, 640], &input_data, Some("stage0_input"))
            .expect("upload input");
        let relu = run_stage0_block(&ops, &loader, &input_gpu).expect("stage0 block");
        let gpu_vec = relu.to_vec().expect("download block output");

        assert_eq!(gpu_vec.len(), cpu_vec.len(), "shape mismatch");
        let max_diff = gpu_vec
            .iter()
            .zip(cpu_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "conv-depthwise block mismatch (max diff {max_diff})"
        );
    }

    #[test]
    fn pooled_stage_conv_matches_onnx() {
        let Some(ops) = gpu_ops() else {
            eprintln!("Skipping pooled-stage test (no adapter)");
            return;
        };

        let Some(model_path) = model_file_path() else {
            eprintln!("Skipping pooled-stage test (model file {MODEL_REL_PATH} missing)");
            return;
        };

        let loader = load_backbone_weights(&model_path, 1, false, 0).expect("load stage1 weights");

        let input_data = synthetic_input();
        let cpu_vec = reference_output(&model_path, "Relu_8", &input_data);

        let input_gpu = ops
            .upload_tensor([1usize, 3, 640, 640], &input_data, Some("stage1_input"))
            .expect("upload input");
        let stage0 = run_stage0_block(&ops, &loader, &input_gpu).expect("stage0");
        let pooled = pool_tensor(&ops, &stage0).expect("max pool");
        let relu =
            run_stage_blocks(&ops, &loader, &pooled, &STAGE1_BLOCKS[..1]).expect("stage1 block0");
        let gpu_vec = relu.to_vec().expect("download stage1 output");

        assert_eq!(gpu_vec.len(), cpu_vec.len(), "shape mismatch");
        let max_diff = gpu_vec
            .iter()
            .zip(cpu_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "stage1 pooled block mismatch (max diff {max_diff})"
        );
    }

    #[test]
    fn pooled_stage_two_block_matches_onnx() {
        let Some(ops) = gpu_ops() else {
            eprintln!("Skipping pooled-stage two-block test (no adapter)");
            return;
        };

        let Some(model_path) = model_file_path() else {
            eprintln!("Skipping pooled-stage two-block test (model file {MODEL_REL_PATH} missing)");
            return;
        };

        let loader = load_backbone_weights(&model_path, 1, false, 0).expect("load stage1 weights");

        let input_data = synthetic_input();
        let cpu_vec = reference_output(&model_path, "Relu_11", &input_data);

        let input_gpu = ops
            .upload_tensor([1usize, 3, 640, 640], &input_data, Some("stage1_input"))
            .expect("upload input");
        let stage1 = run_backbone_to_stage(&ops, &loader, &input_gpu, 1).expect("stage1 output");
        let gpu_vec = stage1.to_vec().expect("download stage1 block2");

        assert_eq!(gpu_vec.len(), cpu_vec.len(), "shape mismatch");
        let max_diff = gpu_vec
            .iter()
            .zip(cpu_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "stage1 two-block mismatch (max diff {max_diff})"
        );
    }

    #[test]
    fn stage2_blocks_match_onnx() {
        let Some(ops) = gpu_ops() else {
            eprintln!("Skipping stage2 test (no adapter)");
            return;
        };

        let Some(model_path) = model_file_path() else {
            eprintln!("Skipping stage2 test (model file {MODEL_REL_PATH} missing)");
            return;
        };

        let loader = load_backbone_weights(&model_path, 2, false, 0).expect("load stage2 weights");

        let input_data = synthetic_input();
        let cpu_vec = reference_output(&model_path, "Relu_17", &input_data);

        let input_gpu = ops
            .upload_tensor([1usize, 3, 640, 640], &input_data, Some("stage2_input"))
            .expect("upload input");
        let stage2 = run_backbone_to_stage(&ops, &loader, &input_gpu, 2).expect("stage2");
        let gpu_vec = stage2.to_vec().expect("download stage2 block2");

        assert_eq!(gpu_vec.len(), cpu_vec.len(), "shape mismatch");
        let max_diff = gpu_vec
            .iter()
            .zip(cpu_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "stage2 two-block mismatch (max diff {max_diff})"
        );
    }

    #[test]
    fn stage2_pooled_matches_onnx() {
        let Some(ops) = gpu_ops() else {
            eprintln!("Skipping stage2 pooled test (no adapter)");
            return;
        };

        let Some(model_path) = model_file_path() else {
            eprintln!("Skipping stage2 pooled test (model file {MODEL_REL_PATH} missing)");
            return;
        };

        let loader =
            load_backbone_weights(&model_path, 2, false, 0).expect("load stage2 pooled weights");

        let input_data = synthetic_input();
        let cpu_vec = reference_output(&model_path, "MaxPool_18", &input_data);

        let input_gpu = ops
            .upload_tensor(
                [1usize, 3, 640, 640],
                &input_data,
                Some("stage2_pool_input"),
            )
            .expect("upload input");
        let stage2 = run_backbone_to_stage(&ops, &loader, &input_gpu, 2).expect("stage2");
        let stage2_pooled = pool_tensor(&ops, &stage2).expect("stage2 pool");
        let gpu_vec = stage2_pooled.to_vec().expect("download stage2 pool");

        assert_eq!(gpu_vec.len(), cpu_vec.len(), "shape mismatch");
        let max_diff = gpu_vec
            .iter()
            .zip(cpu_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "stage2 pooled mismatch (max diff {max_diff})"
        );
    }

    #[test]
    fn stage3_blocks_match_onnx() {
        let Some(ops) = gpu_ops() else {
            eprintln!("Skipping stage3 test (no adapter)");
            return;
        };

        let Some(model_path) = model_file_path() else {
            eprintln!("Skipping stage3 test (model file {MODEL_REL_PATH} missing)");
            return;
        };

        let loader = load_backbone_weights(&model_path, 3, false, 0).expect("load stage3 weights");

        let input_data = synthetic_input();
        let cpu_vec = reference_output(&model_path, "Relu_24", &input_data);

        let input_gpu = ops
            .upload_tensor([1usize, 3, 640, 640], &input_data, Some("stage3_input"))
            .expect("upload input");
        let stage3 = run_backbone_to_stage(&ops, &loader, &input_gpu, 3).expect("stage3 output");
        let gpu_vec = stage3.to_vec().expect("download stage3 block2");

        assert_eq!(gpu_vec.len(), cpu_vec.len(), "shape mismatch");
        let max_diff = gpu_vec
            .iter()
            .zip(cpu_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "stage3 two-block mismatch (max diff {max_diff})"
        );
    }

    #[test]
    fn stage3_pooled_matches_onnx() {
        let Some(ops) = gpu_ops() else {
            eprintln!("Skipping stage3 pooled test (no adapter)");
            return;
        };

        let Some(model_path) = model_file_path() else {
            eprintln!("Skipping stage3 pooled test (model file {MODEL_REL_PATH} missing)");
            return;
        };

        let loader = load_backbone_weights(&model_path, 3, false, 0).expect("load stage3 weights");

        let input_data = synthetic_input();
        let cpu_vec = reference_output(&model_path, "MaxPool_25", &input_data);

        let input_gpu = ops
            .upload_tensor(
                [1usize, 3, 640, 640],
                &input_data,
                Some("stage3_pool_input"),
            )
            .expect("upload input");
        let stage3 = run_backbone_to_stage(&ops, &loader, &input_gpu, 3).expect("stage3 output");
        let pooled = pool_tensor(&ops, &stage3).expect("stage3 pool");
        let gpu_vec = pooled.to_vec().expect("download stage3 pool");

        assert_eq!(gpu_vec.len(), cpu_vec.len(), "shape mismatch");
        let max_diff = gpu_vec
            .iter()
            .zip(cpu_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "stage3 pooled mismatch (max diff {max_diff})"
        );
    }

    #[test]
    fn stage4_blocks_match_onnx() {
        let Some(ops) = gpu_ops() else {
            eprintln!("Skipping stage4 test (no adapter)");
            return;
        };

        let Some(model_path) = model_file_path() else {
            eprintln!("Skipping stage4 test (model file {MODEL_REL_PATH} missing)");
            return;
        };

        let loader = load_backbone_weights(&model_path, 4, false, 0).expect("load stage4 weights");

        let input_data = synthetic_input();
        let cpu_vec = reference_output(&model_path, "Relu_31", &input_data);

        let input_gpu = ops
            .upload_tensor([1usize, 3, 640, 640], &input_data, Some("stage4_input"))
            .expect("upload input");
        let stage4 = run_backbone_to_stage(&ops, &loader, &input_gpu, 4).expect("stage4 output");
        let gpu_vec = stage4.to_vec().expect("download stage4 block2");

        assert_eq!(gpu_vec.len(), cpu_vec.len(), "shape mismatch");
        let max_diff = gpu_vec
            .iter()
            .zip(cpu_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "stage4 two-block mismatch (max diff {max_diff})"
        );
    }

    #[test]
    fn stage4_pooled_matches_onnx() {
        let Some(ops) = gpu_ops() else {
            eprintln!("Skipping stage4 pooled test (no adapter)");
            return;
        };

        let Some(model_path) = model_file_path() else {
            eprintln!("Skipping stage4 pooled test (model file {MODEL_REL_PATH} missing)");
            return;
        };

        let loader = load_backbone_weights(&model_path, 4, false, 0).expect("load stage4 weights");

        let input_data = synthetic_input();
        let cpu_vec = reference_output(&model_path, "MaxPool_32", &input_data);

        let input_gpu = ops
            .upload_tensor(
                [1usize, 3, 640, 640],
                &input_data,
                Some("stage4_pool_input"),
            )
            .expect("upload input");
        let stage4 = run_backbone_to_stage(&ops, &loader, &input_gpu, 4).expect("stage4 output");
        let pooled = pool_tensor(&ops, &stage4).expect("stage4 pool");
        let gpu_vec = pooled.to_vec().expect("download stage4 pool");

        assert_eq!(gpu_vec.len(), cpu_vec.len(), "shape mismatch");
        let max_diff = gpu_vec
            .iter()
            .zip(cpu_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "stage4 pooled mismatch (max diff {max_diff})"
        );
    }

    #[test]
    fn stage5_blocks_match_onnx() {
        let Some(ops) = gpu_ops() else {
            eprintln!("Skipping stage5 test (no adapter)");
            return;
        };

        let Some(model_path) = model_file_path() else {
            eprintln!("Skipping stage5 test (model file {MODEL_REL_PATH} missing)");
            return;
        };

        let loader = load_backbone_weights(&model_path, 5, false, 0).expect("load stage5 weights");

        let input_data = synthetic_input();
        let cpu_vec = reference_output(&model_path, "Relu_38", &input_data);

        let input_gpu = ops
            .upload_tensor([1usize, 3, 640, 640], &input_data, Some("stage5_input"))
            .expect("upload input");
        let stage5 = run_backbone_to_stage(&ops, &loader, &input_gpu, 5).expect("stage5 output");
        let gpu_vec = stage5.to_vec().expect("download stage5 block2");

        assert_eq!(gpu_vec.len(), cpu_vec.len(), "shape mismatch");
        let max_diff = gpu_vec
            .iter()
            .zip(cpu_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "stage5 two-block mismatch (max diff {max_diff})"
        );
    }

    #[test]
    fn neck_and_detection_heads_match_onnx() {
        let Some(ops) = gpu_ops() else {
            eprintln!("Skipping neck/head test (no adapter)");
            return;
        };

        let Some(model_path) = model_file_path() else {
            eprintln!("Skipping neck/head test (model file {MODEL_REL_PATH} missing)");
            return;
        };

        let loader = load_backbone_weights(
            &model_path,
            BACKBONE_STAGES.len(),
            true,
            DETECTION_HEADS.len(),
        )
        .expect("load detection weights");

        let input_data = synthetic_input();
        let input_gpu = ops
            .upload_tensor([1usize, 3, 640, 640], &input_data, Some("neck_input"))
            .expect("upload input");
        let backbone_features =
            run_backbone_features(&ops, &loader, &input_gpu, BACKBONE_STAGES.len())
                .expect("backbone outputs");
        let levels =
            run_neck_and_heads(&ops, &loader, &backbone_features).expect("neck + head outputs");

        let feature_checks = [
            ("Relu_53", &levels[0].feature),
            ("Relu_47", &levels[1].feature),
            ("Relu_41", &levels[2].feature),
        ];
        for (node, tensor) in feature_checks {
            assert_tensor_matches(&model_path, node, tensor, &input_data);
        }

        let branch_checks = [
            ("Conv_55", &levels[0].cls),
            ("Conv_57", &levels[1].cls),
            ("Conv_59", &levels[2].cls),
            ("Conv_61", &levels[0].bbox),
            ("Conv_63", &levels[1].bbox),
            ("Conv_65", &levels[2].bbox),
            ("Conv_67", &levels[0].obj),
            ("Conv_69", &levels[1].obj),
            ("Conv_71", &levels[2].obj),
            ("Conv_73", &levels[0].kps),
            ("Conv_75", &levels[1].kps),
            ("Conv_77", &levels[2].kps),
        ];
        for (node, tensor) in branch_checks {
            assert_tensor_matches(&model_path, node, tensor, &input_data);
        }
    }

    #[test]
    fn conv2d_vec4_matches_standard() {
        println!("Starting conv2d_vec4_matches_standard test");
        let Some(ops) = gpu_ops() else {
            eprintln!("Skipping conv2d_vec4 test (no adapter)");
            return;
        };

        let batch = 1;
        let input_channels = 16;
        let output_channels = 32;
        let width = 64;
        let height = 64;
        let kernel = 3;
        let stride = 1;
        let pad = 1;

        let input_len = (batch * input_channels * width * height) as usize;
        let input: Vec<f32> = (0..input_len).map(|i| (i % 100) as f32 / 100.0).collect();

        let weight_len = (output_channels * input_channels * kernel * kernel) as usize;
        let weights: Vec<f32> = (0..weight_len).map(|i| (i % 100) as f32 / 100.0).collect();

        let bias_len = output_channels as usize;
        let bias: Vec<f32> = (0..bias_len).map(|i| (i % 100) as f32 / 100.0).collect();

        let config = Conv2dConfig::new(
            batch,
            Conv2dChannels::new(input_channels, output_channels),
            SpatialDims::new(width, height),
            SpatialDims::new(kernel, kernel),
            SpatialDims::new(stride, stride),
            SpatialDims::new(pad, pad),
            Conv2dOptions::new(1, Some(ActivationKind::Relu)),
        )
        .unwrap();

        let input_gpu = ops
            .upload_tensor(config.input_shape_dims(), &input, Some("input"))
            .unwrap();
        let weight_gpu = ops
            .upload_tensor(config.weight_shape_dims(), &weights, Some("weights"))
            .unwrap();
        let bias_gpu = ops
            .upload_tensor(config.bias_shape_dims(), &bias, Some("bias"))
            .unwrap();

        let standard_out = ops
            .conv2d_tensor(&input_gpu, &weight_gpu, &bias_gpu, &config)
            .unwrap();
        let vec4_out = ops
            .conv2d_vec4_tensor(&input_gpu, &weight_gpu, &bias_gpu, &config)
            .unwrap();

        let standard_vec = standard_out.to_vec().unwrap();
        let vec4_vec = vec4_out.to_vec().unwrap();

        assert_eq!(standard_vec.len(), vec4_vec.len(), "Output length mismatch");

        let mut max_diff = 0.0f32;
        let mut mismatch_count = 0;
        for (i, (a, b)) in standard_vec.iter().zip(vec4_vec.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 1e-4 {
                if mismatch_count < 10 {
                    eprintln!(
                        "Mismatch at index {}: standard={}, vec4={}, diff={}",
                        i, a, b, diff
                    );
                }
                mismatch_count += 1;
            }
        }

        eprintln!("Total mismatches: {}", mismatch_count);
        eprintln!("Max diff between standard and vec4: {}", max_diff);
        assert!(
            max_diff < 1e-4,
            "Vectorized implementation output mismatch (max diff {})",
            max_diff
        );
    }

    #[test]
    #[ignore] // Run with: cargo test -p yunet-core benchmark_conv2d_performance -- --ignored --nocapture
    fn benchmark_conv2d_performance() {
        use std::time::Instant;

        let Some(ops) = gpu_ops() else {
            eprintln!("Skipping benchmark (no GPU adapter)");
            return;
        };

        let configs = vec![
            ("stage0_conv", 1, 3, 16, 640, 640, 3, 2, 1, 1),
            ("stage1_depth", 1, 32, 32, 160, 160, 3, 1, 1, 32),
            ("stage2_point", 1, 32, 64, 160, 160, 1, 1, 0, 1),
            ("head_depth", 1, 64, 64, 40, 40, 3, 1, 1, 64),
        ];

        println!("\n========== Conv2D Performance: Standard vs Vec4 ==========");

        for (name, batch, in_ch, out_ch, width, height, kernel, stride, pad, groups) in configs {
            let input_len = (batch * in_ch * width * height) as usize;
            let input: Vec<f32> = (0..input_len).map(|i| (i % 100) as f32 / 100.0).collect();

            let weight_len = if groups == 1 {
                (out_ch * in_ch * kernel * kernel) as usize
            } else {
                (out_ch * kernel * kernel) as usize
            };
            let weights: Vec<f32> = (0..weight_len).map(|i| (i % 100) as f32 / 100.0).collect();
            let bias: Vec<f32> = (0..out_ch).map(|i| (i % 100) as f32 / 100.0).collect();

            let config = Conv2dConfig::new(
                batch,
                Conv2dChannels::new(in_ch, out_ch),
                SpatialDims::new(width, height),
                SpatialDims::new(kernel, kernel),
                SpatialDims::new(stride, stride),
                SpatialDims::new(pad, pad),
                Conv2dOptions::new(groups, Some(ActivationKind::Relu)),
            )
            .unwrap();

            let input_gpu = ops
                .upload_tensor(config.input_shape_dims(), &input, None)
                .unwrap();
            let weight_gpu = ops
                .upload_tensor(config.weight_shape_dims(), &weights, None)
                .unwrap();
            let bias_gpu = ops
                .upload_tensor(config.bias_shape_dims(), &bias, None)
                .unwrap();

            // Warmup
            for _ in 0..5 {
                let _ = ops
                    .conv2d_tensor(&input_gpu, &weight_gpu, &bias_gpu, &config)
                    .unwrap();
            }

            // Benchmark standard
            let iterations = 50;
            let start = Instant::now();
            for _ in 0..iterations {
                let output = ops
                    .conv2d_tensor(&input_gpu, &weight_gpu, &bias_gpu, &config)
                    .unwrap();
                let _ = output.to_vec().unwrap();
            }
            let standard_avg = start.elapsed().as_micros() as f64 / iterations as f64;

            // Warmup vec4
            for _ in 0..5 {
                let _ = ops
                    .conv2d_vec4_tensor(&input_gpu, &weight_gpu, &bias_gpu, &config)
                    .unwrap();
            }

            // Benchmark vec4
            let start = Instant::now();
            for _ in 0..iterations {
                let output = ops
                    .conv2d_vec4_tensor(&input_gpu, &weight_gpu, &bias_gpu, &config)
                    .unwrap();
                let _ = output.to_vec().unwrap();
            }
            let vec4_avg = start.elapsed().as_micros() as f64 / iterations as f64;

            let speedup_pct = (standard_avg / vec4_avg - 1.0) * 100.0;

            println!(
                "{:<15} Standard: {:>7.1}μs | Vec4: {:>7.1}μs | Speedup: {:>+5.1}%",
                name, standard_avg, vec4_avg, speedup_pct
            );
        }

        println!("==========================================================\n");
    }
}
