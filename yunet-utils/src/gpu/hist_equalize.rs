use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use super::{GpuContext, HIST_EQUALIZE_WGSL};
use crate::{gpu_readback, gpu_uniforms, storage_buffer_entry, uniform_buffer_entry};

gpu_uniforms!(HistogramUniforms, 3, {
    pixel_count: u32,
});

gpu_uniforms!(CdfUniforms, 3, {
    total_pixels: u32,
});

gpu_uniforms!(LutUniforms, 3, {
    pixel_count: u32,
});

#[derive(Clone)]
pub struct GpuHistogramEqualizer {
    context: Arc<GpuContext>,
    histogram_pipeline: wgpu::ComputePipeline,
    cdf_pipeline: wgpu::ComputePipeline,
    apply_pipeline: wgpu::ComputePipeline,
    histogram_bgl: wgpu::BindGroupLayout,
    cdf_bgl: wgpu::BindGroupLayout,
    apply_bgl: wgpu::BindGroupLayout,
}

impl GpuHistogramEqualizer {
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yunet_hist_equalize_shader"),
            source: wgpu::ShaderSource::Wgsl(HIST_EQUALIZE_WGSL.into()),
        });

        let histogram_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yunet_histogram_bgl"),
            entries: &[
                storage_buffer_entry!(0, read_only),
                storage_buffer_entry!(1, read_write),
                uniform_buffer_entry!(2),
            ],
        });
        let cdf_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yunet_hist_cdf_bgl"),
            entries: &[
                storage_buffer_entry!(0, read_only),
                storage_buffer_entry!(1, read_write),
                uniform_buffer_entry!(2),
            ],
        });
        let apply_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yunet_hist_apply_bgl"),
            entries: &[
                storage_buffer_entry!(0, read_write),
                storage_buffer_entry!(1, read_only),
                uniform_buffer_entry!(2),
            ],
        });

        let histogram_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("yunet_histogram_layout"),
            bind_group_layouts: &[&histogram_bgl],
            push_constant_ranges: &[],
        });
        let cdf_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("yunet_hist_cdf_layout"),
            bind_group_layouts: &[&cdf_bgl],
            push_constant_ranges: &[],
        });
        let apply_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("yunet_hist_apply_layout"),
            bind_group_layouts: &[&apply_bgl],
            push_constant_ranges: &[],
        });

        let histogram_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yunet_histogram_pipeline"),
            layout: Some(&histogram_layout),
            module: &module,
            entry_point: Some("build_histogram"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let cdf_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yunet_hist_cdf_pipeline"),
            layout: Some(&cdf_layout),
            module: &module,
            entry_point: Some("compute_lut"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let apply_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yunet_hist_apply_pipeline"),
            layout: Some(&apply_layout),
            module: &module,
            entry_point: Some("apply_equalization"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            context,
            histogram_pipeline,
            cdf_pipeline,
            apply_pipeline,
            histogram_bgl,
            cdf_bgl,
            apply_bgl,
        })
    }

    pub fn equalize(&self, image: &DynamicImage) -> Result<DynamicImage> {
        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        let pixel_count = (width as usize) * (height as usize);
        if pixel_count == 0 {
            return Ok(image.clone());
        }

        let mut data_u32 = Vec::with_capacity(pixel_count * 4);
        data_u32.extend(rgba.as_raw().iter().map(|b| *b as u32));

        let (lut_buffer, pixel_buffer) =
            self.build_histogram_and_lut(&data_u32, pixel_count as u32)?;

        self.apply_lut(pixel_buffer, lut_buffer, width, height)
    }

    fn build_histogram_and_lut(
        &self,
        pixels: &[u32],
        pixel_count: u32,
    ) -> Result<(wgpu::Buffer, wgpu::Buffer)> {
        let device = self.context.device();
        let queue = self.context.queue();

        let pixel_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_hist_pixels"),
            contents: cast_slice(pixels),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let histogram_size = (256 * 3 * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        let histogram_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yunet_histogram_buffer"),
            size: histogram_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &histogram_buffer,
            0,
            vec![0u8; histogram_size as usize].as_slice(),
        );

        let uniform = HistogramUniforms {
            pixel_count,
            __padding: [0; 3],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_hist_uniform"),
            contents: bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yunet_histogram_bg"),
            layout: &self.histogram_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pixel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let lut_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yunet_hist_lut_buffer"),
            size: (256 * 3 * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cdf_uniform = CdfUniforms {
            total_pixels: pixel_count,
            __padding: [0; 3],
        };
        let cdf_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_hist_cdf_uniform"),
            contents: bytes_of(&cdf_uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let cdf_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yunet_hist_cdf_bg"),
            layout: &self.cdf_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lut_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cdf_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("yunet_hist_encoder"),
        });
        {
            let dispatch = pixel_count.div_ceil(256);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yunet_hist_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.histogram_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yunet_hist_cdf_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cdf_pipeline);
            pass.set_bind_group(0, &cdf_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        Ok((lut_buffer, pixel_buffer))
    }

    fn apply_lut(
        &self,
        pixel_buffer: wgpu::Buffer,
        lut_buffer: wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> Result<DynamicImage> {
        let device = self.context.device();
        let queue = self.context.queue();
        let pixel_count = (width as usize * height as usize) as u32;
        let buffer_size =
            (pixel_count as usize * 4 * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

        let uniform = LutUniforms {
            pixel_count,
            __padding: [0; 3],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_hist_apply_uniform"),
            contents: bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yunet_hist_apply_bg"),
            layout: &self.apply_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pixel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lut_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("yunet_hist_apply_encoder"),
        });
        {
            let dispatch = pixel_count.div_ceil(256);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yunet_hist_apply_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.apply_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch, 1, 1);
        }

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yunet_hist_apply_readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&pixel_buffer, 0, &readback, 0, buffer_size);
        queue.submit(std::iter::once(encoder.finish()));

        let expected_len = (pixel_count as usize) * 4;
        let bytes = gpu_readback!(readback, device, expected_len, "histogram equalization")?;

        let result =
            RgbaImage::from_raw(width, height, bytes).context("failed to build equalized image")?;
        Ok(DynamicImage::ImageRgba8(result))
    }
}
