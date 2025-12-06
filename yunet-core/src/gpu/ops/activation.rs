use super::utils::{buffer_entry, create_uniform_buffer, div_ceil_uniform, uniform_entry};
use crate::gpu::GpuTensor;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use yunet_utils::gpu::GpuContext;

const ACTIVATION_WGSL: &str = include_str!("../activation.wgsl");
const ACTIVATION_WORKGROUP_SIZE: u32 = 256;

/// Supported activation operations.
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum ActivationKind {
    /// Rectified Linear Unit.
    Relu = 0,
    /// Sigmoid.
    Sigmoid = 1,
}

#[derive(Debug)]
pub(super) struct ActivationPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ActivationPipeline {
    pub(super) fn new(device: &wgpu::Device) -> Result<Self> {
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

    pub(super) fn execute(
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ActivationUniforms {
    len: u32,
    mode: u32,
}
