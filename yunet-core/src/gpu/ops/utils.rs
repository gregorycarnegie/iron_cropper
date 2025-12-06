use anyhow::{Context, Result};
use bytemuck::{Pod, bytes_of};

pub(super) fn create_uniform_buffer(
    device: &wgpu::Device,
    label: &str,
    data: &impl Pod,
) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes_of(data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

pub(super) fn buffer_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(super) fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(super) fn div_ceil_uniform(value: u32, divisor: u32) -> u32 {
    if value == 0 {
        0
    } else {
        value.div_ceil(divisor)
    }
}

pub(super) fn compute_output_dim(size: u32, pad: u32, kernel: u32, stride: u32) -> Result<u32> {
    anyhow::ensure!(stride > 0, "stride must be > 0");
    anyhow::ensure!(kernel > 0, "kernel must be > 0");
    let numerator = size
        .checked_add(pad * 2)
        .context("padding overflowed u32")?
        .checked_sub(kernel)
        .context("kernel larger than padded input")?;
    Ok(numerator / stride + 1)
}
