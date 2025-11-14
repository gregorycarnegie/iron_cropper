use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::cast_slice;
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use crate::shape::CropShape;
use crate::shape::outline_points_for_rect;
use crate::{create_gpu_pipeline, gpu_readback, gpu_uniforms, storage_buffer_entry, uniform_buffer_entry};

use super::{GpuContext, SHAPE_MASK_WGSL};

const MAX_POINTS: usize = 512;

gpu_uniforms!(ShapeMaskUniforms, 0, {
    width: u32,
    height: u32,
    point_count: u32,
    samples: u32,
});

#[derive(Clone)]
pub struct GpuShapeMask {
    context: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuShapeMask {
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();

        let (pipeline, bind_group_layout) = create_gpu_pipeline!(
            device,
            "shape_mask",
            SHAPE_MASK_WGSL,
            [
                storage_buffer_entry!(0, read_write),
                storage_buffer_entry!(1, read_only),
                uniform_buffer_entry!(2),
            ]
        );

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
            __padding: [0; 0],
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
            let workgroups_x = width.div_ceil(16);
            let workgroups_y = height.div_ceil(16);
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

        let out_bytes = gpu_readback!(readback, device, pixels_u32.len(), "shape mask")?;

        let masked = RgbaImage::from_raw(width, height, out_bytes)
            .context("failed to build masked image")?;
        Ok(Some(DynamicImage::ImageRgba8(masked)))
    }
}
