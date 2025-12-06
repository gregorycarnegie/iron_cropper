use super::utils::{buffer_entry, create_uniform_buffer, uniform_entry};
use crate::gpu::GpuTensor;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use yunet_utils::gpu::{GpuBufferPool, GpuContext};

const ADD_WORKGROUP_SIZE: u32 = 256;

#[derive(Debug)]
pub(super) struct AddPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl AddPipeline {
    pub(super) fn new(device: &wgpu::Device) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yunet_add_shader"),
            source: wgpu::ShaderSource::Wgsl(super::ADD_WGSL.into()),
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

    pub(super) fn execute(
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AddUniforms {
    len: u32,
}
