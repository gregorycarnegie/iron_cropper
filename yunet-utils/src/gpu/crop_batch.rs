use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use crate::{
    create_gpu_pipeline, gpu_readback, gpu_uniforms, storage_buffer_entry, uniform_buffer_entry,
};

use super::{CROP_WGSL, GpuContext};

gpu_uniforms!(CropUniforms, 0, {
    src_width: u32,
    src_height: u32,
    crop_x: u32,
    crop_y: u32,
    crop_width: u32,
    crop_height: u32,
    dst_width: u32,
    dst_height: u32,
});

/// Describes a single crop-and-resize job inside a batch.
#[derive(Debug, Clone, Copy)]
pub struct BatchCropRequest {
    /// Top-left source coordinate of the crop rectangle.
    pub source_x: u32,
    /// Top-left source coordinate of the crop rectangle.
    pub source_y: u32,
    /// Width of the source crop region.
    pub source_width: u32,
    /// Height of the source crop region.
    pub source_height: u32,
    /// Desired output width in pixels.
    pub output_width: u32,
    /// Desired output height in pixels.
    pub output_height: u32,
}

impl BatchCropRequest {
    fn validate(&self, src_width: u32, src_height: u32) -> Result<()> {
        anyhow::ensure!(
            self.output_width > 0 && self.output_height > 0,
            "GPU batch crop requires non-zero output dimensions"
        );
        anyhow::ensure!(
            self.source_width > 0 && self.source_height > 0,
            "GPU batch crop source region must be non-empty"
        );
        anyhow::ensure!(
            self.source_x < src_width && self.source_y < src_height,
            "GPU batch crop outside image bounds"
        );
        let end_x = self.source_x.saturating_add(self.source_width);
        let end_y = self.source_y.saturating_add(self.source_height);
        anyhow::ensure!(
            end_x <= src_width && end_y <= src_height,
            "GPU batch crop rectangle exceeds image dimensions"
        );
        Ok(())
    }
}

/// GPU compute pipeline that crops and resizes multiple face regions without re-uploading the source image.
#[derive(Clone)]
pub struct GpuBatchCropper {
    context: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuBatchCropper {
    /// Initialize the compute pipeline.
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();
        let (pipeline, bind_group_layout) = create_gpu_pipeline!(
            device,
            "batch_crop",
            CROP_WGSL,
            [
                storage_buffer_entry!(0, read_only),
                storage_buffer_entry!(1, read_write),
                uniform_buffer_entry!(2),
            ]
        );

        Ok(Self {
            context,
            pipeline,
            bind_group_layout,
        })
    }

    /// Execute the batch crop on the provided image and requests.
    pub fn crop(
        &self,
        source: &DynamicImage,
        requests: &[BatchCropRequest],
    ) -> Result<Vec<DynamicImage>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let rgba = source.to_rgba8();
        let (src_width, src_height) = rgba.dimensions();
        anyhow::ensure!(
            src_width > 0 && src_height > 0,
            "source image must be non-empty"
        );

        let pixel_count = (src_width as usize) * (src_height as usize);
        let mut source_data = Vec::with_capacity(pixel_count * 4);
        source_data.extend(rgba.as_raw().iter().map(|b| *b as u32));

        let device = self.context.device();
        let queue = self.context.queue();

        let source_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_batch_crop_source"),
            contents: cast_slice(&source_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let mut outputs = Vec::with_capacity(requests.len());

        for req in requests {
            req.validate(src_width, src_height)?;
            let crop_width = req.source_width.min(src_width - req.source_x).max(1);
            let crop_height = req.source_height.min(src_height - req.source_y).max(1);
            let dst_pixels = (req.output_width as usize) * (req.output_height as usize);
            let buffer_len_bytes =
                (dst_pixels * 4 * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("yunet_batch_crop_output"),
                size: buffer_len_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("yunet_batch_crop_readback"),
                size: buffer_len_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let uniforms = CropUniforms {
                src_width,
                src_height,
                crop_x: req.source_x,
                crop_y: req.source_y,
                crop_width,
                crop_height,
                dst_width: req.output_width,
                dst_height: req.output_height,
                __padding: [],
            };

            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("yunet_batch_crop_uniforms"),
                contents: bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("yunet_batch_crop_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: source_buffer.as_entire_binding(),
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
                label: Some("yunet_batch_crop_encoder"),
            });
            {
                let workgroups_x = req.output_width.div_ceil(16);
                let workgroups_y = req.output_height.div_ceil(16);
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("yunet_batch_crop_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups_x.max(1), workgroups_y.max(1), 1);
            }
            encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, buffer_len_bytes);
            queue.submit(std::iter::once(encoder.finish()));

            let out_bytes = gpu_readback!(readback, device, dst_pixels * 4, "batch crop")?;
            let image = RgbaImage::from_raw(req.output_width, req.output_height, out_bytes)
                .context("failed to build GPU crop image")?;
            outputs.push(DynamicImage::ImageRgba8(image));
        }

        Ok(outputs)
    }
}
