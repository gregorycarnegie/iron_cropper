use std::sync::{Arc, mpsc};

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use super::{BACKGROUND_BLUR_WGSL, GpuContext};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BackgroundBlurUniforms {
    width: u32,
    height: u32,
    mask_size: f32,
    _pad: f32,
}

#[derive(Clone)]
pub struct GpuBackgroundBlur {
    context: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuBackgroundBlur {
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yunet_background_blur_shader"),
            source: wgpu::ShaderSource::Wgsl(BACKGROUND_BLUR_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yunet_background_blur_bgl"),
            entries: &[
                buffer_entry(0, true),
                buffer_entry(1, false),
                buffer_entry(2, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
            label: Some("yunet_background_blur_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yunet_background_blur_pipeline"),
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

    pub fn blend(
        &self,
        sharp: &DynamicImage,
        blurred: &DynamicImage,
        mask_size: f32,
    ) -> Result<DynamicImage> {
        let sharp_rgba = sharp.to_rgba8();
        let blur_rgba = blurred.to_rgba8();
        let (width, height) = sharp_rgba.dimensions();

        if blur_rgba.dimensions() != (width, height) {
            anyhow::bail!("background blur images must have matching dimensions");
        }

        let pixel_count = (width as usize) * (height as usize);
        let mut sharp_u32 = Vec::with_capacity(pixel_count * 4);
        sharp_u32.extend(sharp_rgba.as_raw().iter().map(|b| *b as u32));
        let mut blur_u32 = Vec::with_capacity(pixel_count * 4);
        blur_u32.extend(blur_rgba.as_raw().iter().map(|b| *b as u32));

        let device = self.context.device();
        let queue = self.context.queue();
        let buffer_size = (sharp_u32.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

        let sharp_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_background_blur_sharp"),
            contents: cast_slice(&sharp_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let blur_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_background_blur_blur"),
            contents: cast_slice(&blur_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yunet_background_blur_output"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yunet_background_blur_readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniforms = BackgroundBlurUniforms {
            width,
            height,
            mask_size,
            _pad: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_background_blur_uniforms"),
            contents: bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yunet_background_blur_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sharp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: blur_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("yunet_background_blur_encoder"),
        });
        {
            let workgroups_x = div_ceil(width, 16);
            let workgroups_y = div_ceil(height, 16);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yunet_background_blur_pass"),
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
            .map_err(|_| anyhow::anyhow!("GPU background blur map callback dropped"))?
            .map_err(|err| anyhow::anyhow!("GPU background blur map error: {err}"))?;

        let mapped = slice.get_mapped_range();
        let result_u32: Vec<u32> = cast_slice(&mapped).to_vec();
        drop(mapped);
        readback.unmap();

        anyhow::ensure!(
            result_u32.len() == sharp_u32.len(),
            "unexpected background blur output size (expected {}, got {})",
            sharp_u32.len(),
            result_u32.len()
        );

        let mut out_bytes = Vec::with_capacity(result_u32.len());
        for value in result_u32 {
            out_bytes.push((value & 0xFF) as u8);
        }

        let image = RgbaImage::from_raw(width, height, out_bytes)
            .context("failed to build blurred image")?;
        Ok(DynamicImage::ImageRgba8(image))
    }
}

fn buffer_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    (value + divisor - 1) / divisor
}
