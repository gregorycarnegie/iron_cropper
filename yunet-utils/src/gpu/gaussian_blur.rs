use std::sync::{Arc, mpsc};

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use super::{GAUSSIAN_BLUR_WGSL, GpuContext};

const MAX_RADIUS: u32 = 12;
const MAX_KERNEL_SIZE: usize = (MAX_RADIUS as usize * 2) + 1;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurUniforms {
    width: u32,
    height: u32,
    radius: u32,
    direction: u32,
    kernel_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    weights: [f32; MAX_KERNEL_SIZE],
}

#[derive(Clone)]
pub struct GpuGaussianBlur {
    context: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuGaussianBlur {
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yunet_gaussian_blur_shader"),
            source: wgpu::ShaderSource::Wgsl(GAUSSIAN_BLUR_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yunet_gaussian_blur_bgl"),
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
            label: Some("yunet_gaussian_blur_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yunet_gaussian_blur_pipeline"),
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

    pub fn blur(&self, image: &DynamicImage, radius: f32) -> Result<DynamicImage> {
        let radius = radius.ceil() as i32;
        if radius <= 0 {
            return Ok(image.clone());
        }
        let radius = radius.clamp(1, MAX_RADIUS as i32) as u32;
        let kernel_size = radius * 2 + 1;
        let weights = build_kernel(radius);

        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        let pixel_count = (width as usize) * (height as usize);
        let mut data_u32 = Vec::with_capacity(pixel_count * 4);
        data_u32.extend(rgba.as_raw().iter().map(|b| *b as u32));

        let device = self.context.device();
        let queue = self.context.queue();
        let buffer_size = (data_u32.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_gaussian_blur_input"),
            contents: cast_slice(&data_u32),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yunet_gaussian_blur_temp"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yunet_gaussian_blur_readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("yunet_gaussian_blur_encoder"),
        });

        // Horizontal pass
        let horizontal_uniforms = BlurUniforms {
            width,
            height,
            radius,
            direction: 0,
            kernel_size,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            weights,
        };
        let horizontal_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_gaussian_blur_uniform_horizontal"),
            contents: bytes_of(&horizontal_uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let horizontal_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yunet_gaussian_blur_bg_horizontal"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: horizontal_buffer.as_entire_binding(),
                },
            ],
        });
        dispatch_blur(&mut encoder, &self.pipeline, &horizontal_bg, width, height);

        // Vertical pass
        let vertical_uniforms = BlurUniforms {
            direction: 1,
            ..horizontal_uniforms
        };
        let vertical_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yunet_gaussian_blur_uniform_vertical"),
            contents: bytes_of(&vertical_uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let vertical_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yunet_gaussian_blur_bg_vertical"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vertical_buffer.as_entire_binding(),
                },
            ],
        });
        dispatch_blur(&mut encoder, &self.pipeline, &vertical_bg, width, height);

        encoder.copy_buffer_to_buffer(&input_buffer, 0, &readback, 0, buffer_size);
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
            .map_err(|err| anyhow::anyhow!("device poll failed during blur: {err}"))?;
        receiver
            .recv()
            .map_err(|_| anyhow::anyhow!("GPU blur map callback dropped"))?
            .map_err(|err| anyhow::anyhow!("GPU blur map error: {err}"))?;

        let mapped = slice.get_mapped_range();
        let result_u32: Vec<u32> = cast_slice(&mapped).to_vec();
        drop(mapped);
        readback.unmap();

        anyhow::ensure!(
            result_u32.len() == data_u32.len(),
            "unexpected GPU blur output size (expected {}, got {})",
            data_u32.len(),
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

fn dispatch_blur(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    width: u32,
    height: u32,
) {
    let workgroups_x = div_ceil(width, 16);
    let workgroups_y = div_ceil(height, 16);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("yunet_gaussian_blur_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    (value + divisor - 1) / divisor
}

fn build_kernel(radius: u32) -> [f32; MAX_KERNEL_SIZE] {
    let sigma = (radius.max(1) as f32) * 0.5 + 0.5;
    let kernel_size = radius * 2 + 1;
    let mut weights = [0.0f32; MAX_KERNEL_SIZE];
    let mut sum = 0.0;
    for i in 0..kernel_size {
        let distance = i as i32 - radius as i32;
        let weight = gaussian(distance as f32, sigma);
        weights[i as usize] = weight;
        sum += weight;
    }
    if sum > 0.0 {
        for i in 0..kernel_size as usize {
            weights[i] /= sum;
        }
    }
    weights
}

fn gaussian(distance: f32, sigma: f32) -> f32 {
    let two_sigma_sq = 2.0 * sigma * sigma;
    (-distance * distance / two_sigma_sq).exp()
}
