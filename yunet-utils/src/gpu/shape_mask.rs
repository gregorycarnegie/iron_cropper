use std::sync::{Arc, mpsc};

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use crate::shape::CropShape;
use crate::shape::outline_points_for_rect;

use super::{GpuContext, SHAPE_MASK_WGSL};

const MAX_POINTS: usize = 512;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShapeMaskUniforms {
    width: u32,
    height: u32,
    point_count: u32,
    samples: u32,
}

#[derive(Clone)]
pub struct GpuShapeMask {
    context: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuShapeMask {
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yunet_shape_mask_shader"),
            source: wgpu::ShaderSource::Wgsl(SHAPE_MASK_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yunet_shape_mask_bgl"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            label: Some("yunet_shape_mask_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yunet_shape_mask_pipeline"),
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

    pub fn apply(&self, image: &DynamicImage, shape: &CropShape) -> Result<Option<DynamicImage>> {
        if matches!(shape, CropShape::Rectangle) {
            return Ok(None);
        }
        let width = image.width();
        let height = image.height();

        let points = outline_points_for_rect(width as f32, height as f32, shape);
        if points.len() < 3 {
            return Ok(None);
        }
        let clamped = points
            .iter()
            .take(MAX_POINTS)
            .map(|(x, y)| [*x, *y])
            .collect::<Vec<_>>();
        if clamped.len() < 3 {
            return Ok(None);
        }

        let rgba = image.to_rgba8();
        let pixel_count = (width as usize) * (height as usize);
        let mut pixels_u32 = Vec::with_capacity(pixel_count * 4);
        pixels_u32.extend(rgba.as_raw().iter().map(|b| *b as u32));

        let device = self.context.device();
        let queue = self.context.queue();
        let buffer_size = (pixels_u32.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

        let storage = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_shape_mask_pixels"),
            contents: cast_slice(&pixels_u32),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yunet_shape_mask_readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let points_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_shape_mask_points"),
            contents: cast_slice(&clamped),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let uniforms = ShapeMaskUniforms {
            width,
            height,
            point_count: clamped.len() as u32,
            samples: 4,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_shape_mask_uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yunet_shape_mask_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: points_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("yunet_shape_mask_encoder"),
        });
        {
            let workgroups_x = div_ceil(width, 16);
            let workgroups_y = div_ceil(height, 16);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yunet_shape_mask_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
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
            .map_err(|_| anyhow::anyhow!("shape mask map callback dropped"))?
            .map_err(|err| anyhow::anyhow!("shape mask map error: {err}"))?;

        let mapped = slice.get_mapped_range();
        let result_u32: Vec<u32> = cast_slice(&mapped).to_vec();
        drop(mapped);
        readback.unmap();

        anyhow::ensure!(
            result_u32.len() == pixels_u32.len(),
            "unexpected shape mask output size (expected {}, got {})",
            pixels_u32.len(),
            result_u32.len()
        );

        let mut out_bytes = Vec::with_capacity(result_u32.len());
        for value in result_u32 {
            out_bytes.push((value & 0xFF) as u8);
        }

        let masked = RgbaImage::from_raw(width, height, out_bytes)
            .context("failed to build masked image")?;
        Ok(Some(DynamicImage::ImageRgba8(masked)))
    }
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    (value + divisor - 1) / divisor
}
