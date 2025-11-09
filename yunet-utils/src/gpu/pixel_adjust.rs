use std::sync::{Arc, mpsc};

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use crate::enhance::EnhancementSettings;

use super::{GpuContext, PIXEL_ADJUST_WGSL};

const EPSILON: f32 = 1e-6;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PixelAdjustUniforms {
    exposure_multiplier: f32,
    brightness_offset: f32,
    contrast: f32,
    saturation: f32,
    pixel_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(Clone)]
pub struct GpuPixelAdjust {
    context: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuPixelAdjust {
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yunet_pixel_adjust_shader"),
            source: wgpu::ShaderSource::Wgsl(PIXEL_ADJUST_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yunet_pixel_adjust_bgl"),
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
            label: Some("yunet_pixel_adjust_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yunet_pixel_adjust_pipeline"),
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

    pub fn needs_adjustment(settings: &EnhancementSettings) -> bool {
        settings.exposure_stops.abs() >= EPSILON
            || settings.brightness != 0
            || (settings.contrast - 1.0).abs() >= EPSILON
            || (settings.saturation - 1.0).abs() >= EPSILON
    }

    pub fn apply(
        &self,
        image: &DynamicImage,
        settings: &EnhancementSettings,
    ) -> Result<DynamicImage> {
        if !Self::needs_adjustment(settings) {
            return Ok(image.clone());
        }

        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        let pixel_count = (width as usize) * (height as usize);
        let mut data_u32: Vec<u32> = Vec::with_capacity(pixel_count * 4);
        for byte in rgba.as_raw() {
            data_u32.push(*byte as u32);
        }

        let device = self.context.device();
        let queue = self.context.queue();

        let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_pixel_adjust_storage"),
            contents: cast_slice(&data_u32),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = PixelAdjustUniforms {
            exposure_multiplier: if settings.exposure_stops.abs() < EPSILON {
                1.0
            } else {
                2f32.powf(settings.exposure_stops)
            },
            brightness_offset: settings.brightness as f32,
            contrast: settings.contrast.clamp(0.5, 2.0),
            saturation: settings.saturation.clamp(0.0, 2.5),
            pixel_count: pixel_count as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_pixel_adjust_uniforms"),
            contents: bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yunet_pixel_adjust_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let buffer_size_bytes =
            (data_u32.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yunet_pixel_adjust_readback"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("yunet_pixel_adjust_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yunet_pixel_adjust_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let dispatch = div_ceil(pixel_count as u32, 256);
            pass.dispatch_workgroups(dispatch, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&storage_buffer, 0, &readback, 0, buffer_size_bytes);
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
            .map_err(|_| anyhow::anyhow!("GPU map callback dropped"))?
            .map_err(|err| anyhow::anyhow!("GPU map error: {err}"))?;

        let mapped = slice.get_mapped_range();
        let output_u32: Vec<u32> = cast_slice(&mapped).to_vec();
        drop(mapped);
        readback.unmap();

        anyhow::ensure!(
            output_u32.len() == data_u32.len(),
            "unexpected GPU pixel count (expected {}, got {})",
            data_u32.len(),
            output_u32.len()
        );

        let mut out_bytes = Vec::with_capacity(output_u32.len());
        for value in output_u32 {
            out_bytes.push((value & 0xFF) as u8);
        }

        let image =
            RgbaImage::from_raw(width, height, out_bytes).context("failed to build RGBA image")?;
        Ok(DynamicImage::ImageRgba8(image))
    }
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}
