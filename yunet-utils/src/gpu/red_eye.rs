use std::sync::{Arc, mpsc};

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use super::{GpuContext, RED_EYE_WGSL};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RedEyeUniforms {
    pixel_count: u32,
    threshold: f32,
    min_red: f32,
    _pad: f32,
}

#[derive(Clone)]
pub struct GpuRedEyeRemoval {
    context: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuRedEyeRemoval {
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yunet_red_eye_shader"),
            source: wgpu::ShaderSource::Wgsl(RED_EYE_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yunet_red_eye_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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
            label: Some("yunet_red_eye_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yunet_red_eye_pipeline"),
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

    pub fn apply(&self, image: &DynamicImage, threshold: f32) -> Result<DynamicImage> {
        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        let pixel_count = (width as usize) * (height as usize);
        if pixel_count == 0 {
            return Ok(image.clone());
        }
        let mut data_u32 = Vec::with_capacity(pixel_count * 4);
        data_u32.extend(rgba.as_raw().iter().map(|b| *b as u32));

        let device = self.context.device();
        let queue = self.context.queue();

        let buffer_size = (data_u32.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        let storage = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_red_eye_storage"),
            contents: cast_slice(&data_u32),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yunet_red_eye_readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniforms = RedEyeUniforms {
            pixel_count: pixel_count as u32,
            threshold,
            min_red: 80.0,
            _pad: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_red_eye_uniforms"),
            contents: bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yunet_red_eye_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("yunet_red_eye_encoder"),
        });
        {
            let dispatch = div_ceil(pixel_count as u32, 256);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yunet_red_eye_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, buffer_size);
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
            .map_err(|_| anyhow::anyhow!("GPU red-eye map callback dropped"))?
            .map_err(|err| anyhow::anyhow!("GPU red-eye map error: {err}"))?;

        let mapped = slice.get_mapped_range();
        let result_u32: Vec<u32> = cast_slice(&mapped).to_vec();
        drop(mapped);
        readback.unmap();

        anyhow::ensure!(
            result_u32.len() == data_u32.len(),
            "unexpected red-eye output size (expected {}, got {})",
            data_u32.len(),
            result_u32.len()
        );

        let mut out_bytes = Vec::with_capacity(result_u32.len());
        for value in result_u32 {
            out_bytes.push((value & 0xFF) as u8);
        }

        let image = RgbaImage::from_raw(width, height, out_bytes)
            .context("failed to build red-eye image")?;
        Ok(DynamicImage::ImageRgba8(image))
    }
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    (value + divisor - 1) / divisor
}
