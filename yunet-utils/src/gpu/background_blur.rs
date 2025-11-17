use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use super::{BACKGROUND_BLUR_WGSL, GpuContext, pack_rgba_pixels, unpack_rgba_pixels};
use crate::{
    create_gpu_pipeline, gpu_readback, gpu_uniforms, storage_buffer_entry, uniform_buffer_entry,
};

gpu_uniforms!(BackgroundBlurUniforms, 1, {
    width: u32,
    height: u32,
    mask_size: f32,
});

#[derive(Clone)]
pub struct GpuBackgroundBlur {
    context: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuBackgroundBlur {
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();

        let (pipeline, bind_group_layout) = create_gpu_pipeline!(
            device,
            "background_blur",
            BACKGROUND_BLUR_WGSL,
            [
                storage_buffer_entry!(0, read_only),
                storage_buffer_entry!(1, read_only),
                storage_buffer_entry!(2, read_write),
                uniform_buffer_entry!(3),
            ]
        );

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

        let sharp_u32 = pack_rgba_pixels(sharp_rgba.as_raw());
        let blur_u32 = pack_rgba_pixels(blur_rgba.as_raw());

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
            __padding: [0; 1],
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
            let workgroups_x = width.div_ceil(16);
            let workgroups_y = height.div_ceil(16);
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

        let out_pixels = gpu_readback!(readback, device, sharp_u32.len(), "background blur")?;
        let out_bytes = unpack_rgba_pixels(&out_pixels);

        let image = RgbaImage::from_raw(width, height, out_bytes)
            .context("failed to build blurred image")?;
        Ok(DynamicImage::ImageRgba8(image))
    }
}
