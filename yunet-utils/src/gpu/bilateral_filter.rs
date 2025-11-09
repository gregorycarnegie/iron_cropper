use std::sync::{Arc, mpsc};

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use super::{BILATERAL_FILTER_WGSL, GpuContext};

const MAX_RADIUS: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BilateralUniforms {
    width: u32,
    height: u32,
    radius: u32,
    pixel_count: u32,
    sigma_space: f32,
    sigma_color: f32,
    amount: f32,
    _pad: f32,
}

#[derive(Clone)]
pub struct GpuBilateralFilter {
    context: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuBilateralFilter {
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yunet_bilateral_filter_shader"),
            source: wgpu::ShaderSource::Wgsl(BILATERAL_FILTER_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yunet_bilateral_filter_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("yunet_bilateral_filter_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yunet_bilateral_filter_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            context,
            pipeline,
            bind_group_layout,
        })
    }

    pub fn smooth(
        &self,
        image: &DynamicImage,
        amount: f32,
        sigma_space: f32,
        sigma_color: f32,
    ) -> Result<DynamicImage> {
        let amount = amount.clamp(0.0, 1.0);
        if amount <= 0.0 {
            return Ok(image.clone());
        }

        let sigma_space = sigma_space.max(0.1);
        let sigma_color = sigma_color.max(0.1);
        let radius = ((sigma_space * 2.0).ceil() as u32).clamp(1, MAX_RADIUS);

        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        let pixel_count = (width as usize) * (height as usize);
        let mut data_u32 = Vec::with_capacity(pixel_count * 4);
        data_u32.extend(rgba.as_raw().iter().map(|b| *b as u32));

        let device = self.context.device();
        let queue = self.context.queue();
        let buffer_size = (data_u32.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_bilateral_filter_input"),
            contents: cast_slice(&data_u32),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yunet_bilateral_filter_output"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yunet_bilateral_filter_readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniforms = BilateralUniforms {
            width,
            height,
            radius,
            pixel_count: (width as u32) * (height as u32),
            sigma_space,
            sigma_color,
            amount,
            _pad: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_bilateral_filter_uniforms"),
            contents: bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yunet_bilateral_filter_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("yunet_bilateral_filter_encoder"),
        });
        {
            let workgroups_x = div_ceil(width, 8);
            let workgroups_y = div_ceil(height, 8);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yunet_bilateral_filter_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, buffer_size);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });

        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|err| anyhow::anyhow!("device poll failed: {err}"))?;
        receiver
            .recv()
            .map_err(|_| anyhow::anyhow!("GPU bilateral map callback dropped"))?
            .map_err(|err| anyhow::anyhow!("GPU bilateral map error: {err}"))?;

        let mapped = slice.get_mapped_range();
        let result_u32: Vec<u32> = cast_slice(&mapped).to_vec();
        drop(mapped);
        readback.unmap();

        anyhow::ensure!(
            result_u32.len() == data_u32.len(),
            "unexpected bilateral output size (expected {}, got {})",
            data_u32.len(),
            result_u32.len()
        );

        let mut out_bytes = Vec::with_capacity(result_u32.len());
        for value in result_u32 {
            out_bytes.push((value & 0xFF) as u8);
        }

        let image = RgbaImage::from_raw(width, height, out_bytes)
            .context("failed to build smoothed image")?;
        Ok(DynamicImage::ImageRgba8(image))
    }
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    (value + divisor - 1) / divisor
}
