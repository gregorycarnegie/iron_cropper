//! Preprocessing utilities for preparing images for YuNet inference.
//!
//! The helpers in this module resize images, convert them into the expected tensor layout, and
//! return the scale factors necessary to map detections back to the source image.

use std::{
    borrow::Cow,
    path::Path,
    sync::{Arc, Mutex, mpsc},
};

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable, bytes_of};
use image::{DynamicImage, GenericImageView, RgbImage, imageops::FilterType};
use tract_onnx::prelude::Tensor;
use yunet_utils::gpu::{GpuContext, PREPROCESS_WGSL};
use yunet_utils::telemetry::timing_guard;
use yunet_utils::{
    compute_resize_scales,
    config::{InputDimensions, ResizeQuality},
    load_image, resize_image, rgb_to_bgr_chw,
};

/// Desired input resolution for YuNet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputSize {
    /// The width of the input tensor.
    pub width: u32,
    /// The height of the input tensor.
    pub height: u32,
}

impl InputSize {
    /// Creates a new `InputSize`.
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

impl Default for InputSize {
    fn default() -> Self {
        Self {
            width: 640,
            height: 640,
        }
    }
}

/// Configuration for preprocessing an image before inference.
#[derive(Debug, Clone, Default)]
pub struct PreprocessConfig {
    /// The target input size for the model.
    pub input_size: InputSize,
    /// Resize filter preference controlling the quality vs speed trade-off.
    pub resize_quality: ResizeQuality,
}

impl PreprocessConfig {
    fn resize_filter(&self) -> FilterType {
        match self.resize_quality {
            ResizeQuality::Quality => FilterType::Triangle,
            ResizeQuality::Speed => FilterType::Nearest,
        }
    }
}

/// Output of preprocessing: tensor plus metadata for rescaling detections.
#[derive(Debug)]
pub struct PreprocessOutput {
    /// The preprocessed image tensor, ready for inference.
    pub tensor: Tensor,
    /// The horizontal scale factor to convert detection coordinates to the original image space.
    pub scale_x: f32,
    /// The vertical scale factor to convert detection coordinates to the original image space.
    pub scale_y: f32,
    /// The original dimensions of the input image.
    pub original_size: (u32, u32),
}

/// Preprocess an image file into a YuNet-ready tensor in `[1, 3, H, W]` (CHW) BGR format matching OpenCV's `blobFromImage`.
///
/// # Arguments
///
/// * `path` - The path to the image file.
/// * `config` - The configuration for preprocessing.
pub fn preprocess_image<P: AsRef<Path>>(
    path: P,
    config: &PreprocessConfig,
) -> Result<PreprocessOutput> {
    let default_cpu = CpuPreprocessor;
    preprocess_image_with(&default_cpu, path, config)
}

/// Preprocess an image from disk using a specific preprocessor implementation.
///
/// This is primarily useful for injecting GPU-backed preprocessors in tests/benchmarks.
pub fn preprocess_image_with<P, T>(
    preprocessor: &T,
    path: P,
    config: &PreprocessConfig,
) -> Result<PreprocessOutput>
where
    P: AsRef<Path>,
    T: Preprocessor + ?Sized,
{
    let _guard = timing_guard("yunet_core::preprocess_image", log::Level::Debug);
    let path_ref = path.as_ref();
    anyhow::ensure!(
        path_ref.exists(),
        "input image does not exist: {}",
        path_ref.display()
    );

    let image = load_image(path_ref)
        .with_context(|| format!("failed to load image from {}", path_ref.display()))?;
    preprocessor.preprocess(&image, config)
}

/// Preprocess an in-memory image (useful for tests).
///
/// # Arguments
///
/// * `image` - The dynamic image to process.
/// * `config` - The configuration for preprocessing.
pub fn preprocess_dynamic_image(
    image: &DynamicImage,
    config: &PreprocessConfig,
) -> Result<PreprocessOutput> {
    let cpu = CpuPreprocessor;
    cpu.preprocess(image, config)
}

impl From<InputDimensions> for InputSize {
    fn from(dimensions: InputDimensions) -> Self {
        InputSize::new(dimensions.width, dimensions.height)
    }
}

impl From<&InputDimensions> for InputSize {
    fn from(dimensions: &InputDimensions) -> Self {
        (*dimensions).into()
    }
}

impl From<InputDimensions> for PreprocessConfig {
    fn from(dimensions: InputDimensions) -> Self {
        let InputDimensions {
            width,
            height,
            resize_quality,
        } = dimensions;
        PreprocessConfig {
            input_size: InputSize::new(width, height),
            resize_quality,
        }
    }
}

impl From<&InputDimensions> for PreprocessConfig {
    fn from(dimensions: &InputDimensions) -> Self {
        PreprocessConfig {
            input_size: (*dimensions).into(),
            resize_quality: dimensions.resize_quality,
        }
    }
}

/// Abstraction over preprocessing backends (CPU, GPU).
pub trait Preprocessor: Send + Sync + std::fmt::Debug {
    /// Convert the provided dynamic image into a YuNet-ready tensor.
    fn preprocess(
        &self,
        image: &DynamicImage,
        config: &PreprocessConfig,
    ) -> Result<PreprocessOutput>;
}

/// Default CPU implementation backed by `image` + ndarray utilities.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuPreprocessor;

impl Preprocessor for CpuPreprocessor {
    fn preprocess(
        &self,
        image: &DynamicImage,
        config: &PreprocessConfig,
    ) -> Result<PreprocessOutput> {
        cpu_preprocess(image, config)
    }
}

fn cpu_preprocess(image: &DynamicImage, config: &PreprocessConfig) -> Result<PreprocessOutput> {
    let _guard = timing_guard("yunet_core::preprocess_dynamic_image", log::Level::Trace);
    let input_w = config.input_size.width;
    let input_h = config.input_size.height;
    anyhow::ensure!(
        input_w > 0 && input_h > 0,
        "input dimensions must be greater than zero"
    );

    let (orig_w, orig_h) = image.dimensions();
    anyhow::ensure!(
        orig_w > 0 && orig_h > 0,
        "source image dimensions must be greater than zero"
    );
    let resized_rgb: Cow<'_, RgbImage> = if orig_w == input_w && orig_h == input_h {
        match image.as_rgb8() {
            Some(rgb) => Cow::Borrowed(rgb),
            None => Cow::Owned(image.to_rgb8()),
        }
    } else {
        Cow::Owned(resize_image(
            image,
            input_w,
            input_h,
            config.resize_filter(),
        ))
    };
    let chw = rgb_to_bgr_chw(&resized_rgb);

    let shape = [1usize, 3, input_h as usize, input_w as usize];
    let (data, offset) = chw.into_raw_vec_and_offset();
    debug_assert_eq!(offset, Some(0), "expected contiguous array");
    let tensor = Tensor::from_shape(&shape, &data)
        .map_err(|e| anyhow::anyhow!("failed to build tensor: {e}"))?;

    let (scale_x, scale_y) = compute_resize_scales((orig_w, orig_h), (input_w, input_h))?;

    Ok(PreprocessOutput {
        tensor,
        scale_x,
        scale_y,
        original_size: (orig_w, orig_h),
    })
}

/// GPU-backed preprocessor that uses `wgpu` compute shaders for resize + color conversion.
#[derive(Clone)]
pub struct WgpuPreprocessor {
    context: Arc<GpuContext>,
    pipeline: Arc<WgpuPreprocessPipeline>,
    pool: Arc<Mutex<GpuResourcePool>>,
}

impl std::fmt::Debug for WgpuPreprocessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuPreprocessor")
            .field("adapter", self.context.adapter_info())
            .finish()
    }
}

impl WgpuPreprocessor {
    /// Create a GPU preprocessor from an existing `GpuContext`.
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let pipeline = WgpuPreprocessPipeline::new(context.device())?;
        Ok(Self {
            context,
            pipeline: Arc::new(pipeline),
            pool: Arc::new(Mutex::new(GpuResourcePool::default())),
        })
    }
}

impl Preprocessor for WgpuPreprocessor {
    fn preprocess(
        &self,
        image: &DynamicImage,
        config: &PreprocessConfig,
    ) -> Result<PreprocessOutput> {
        gpu_preprocess(
            image,
            config,
            self.context.as_ref(),
            &self.pipeline,
            self.pool.as_ref(),
        )
    }
}

struct WgpuPreprocessPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    sampler: wgpu::Sampler,
}

impl WgpuPreprocessPipeline {
    fn new(device: &wgpu::Device) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yunet_preprocess_shader"),
            source: wgpu::ShaderSource::Wgsl(PREPROCESS_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yunet_preprocess_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
            label: Some("yunet_preprocess_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yunet_preprocess_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("yunet_preprocess_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Ok(Self {
            bind_group_layout,
            pipeline,
            sampler,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PreprocessUniforms {
    src_size: [u32; 2],
    dst_size: [u32; 2],
}

#[derive(Default)]
struct GpuResourcePool {
    idle: Vec<GpuWorkBuffers>,
}

struct GpuWorkBuffers {
    texture: wgpu::Texture,
    extent: wgpu::Extent3d,
    storage: wgpu::Buffer,
    storage_size: u64,
    readback: wgpu::Buffer,
    readback_size: u64,
    uniform: wgpu::Buffer,
    staging: Vec<u8>,
}

const UNIFORM_BUFFER_SIZE: u64 = std::mem::size_of::<PreprocessUniforms>() as u64;

impl GpuResourcePool {
    fn acquire(
        &mut self,
        device: &wgpu::Device,
        extent: wgpu::Extent3d,
        output_bytes: u64,
    ) -> GpuWorkBuffers {
        if let Some(mut buffers) = self.idle.pop() {
            buffers.ensure_texture(device, extent);
            buffers.ensure_output_buffers(device, output_bytes);
            buffers
        } else {
            GpuWorkBuffers::new(device, extent, output_bytes)
        }
    }

    fn recycle(&mut self, buffers: GpuWorkBuffers) {
        self.idle.push(buffers);
    }
}

impl GpuWorkBuffers {
    fn new(device: &wgpu::Device, extent: wgpu::Extent3d, output_bytes: u64) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("yunet_preprocess_input_texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yunet_preprocess_output_storage"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yunet_preprocess_readback"),
            size: output_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yunet_preprocess_uniforms"),
            size: UNIFORM_BUFFER_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            texture,
            extent,
            storage,
            storage_size: output_bytes,
            readback,
            readback_size: output_bytes,
            uniform,
            staging: Vec::new(),
        }
    }

    fn ensure_texture(&mut self, device: &wgpu::Device, extent: wgpu::Extent3d) {
        if self.extent == extent {
            return;
        }
        self.texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("yunet_preprocess_input_texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.extent = extent;
    }

    fn ensure_output_buffers(&mut self, device: &wgpu::Device, size: u64) {
        if self.storage_size < size {
            self.storage = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("yunet_preprocess_output_storage"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.storage_size = size;
        }
        if self.readback_size < size {
            self.readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("yunet_preprocess_readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.readback_size = size;
        }
    }

    fn uniform_buffer(&self) -> &wgpu::Buffer {
        &self.uniform
    }

    fn storage_buffer(&self) -> &wgpu::Buffer {
        &self.storage
    }

    fn readback_buffer(&self) -> &wgpu::Buffer {
        &self.readback
    }

    fn prepare_upload<'a>(&'a mut self, data: &'a [u8], width: u32) -> (&'a [u8], u32) {
        let bytes_per_row = 4 * width as usize;
        let aligned = align_to(bytes_per_row, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
        if aligned == bytes_per_row {
            return (data, bytes_per_row as u32);
        }

        let rows = data.len() / bytes_per_row;
        let required = aligned * rows;
        self.staging.resize(required, 0);
        for row in 0..rows {
            let src_start = row * bytes_per_row;
            let dst_start = row * aligned;
            self.staging[dst_start..dst_start + bytes_per_row]
                .copy_from_slice(&data[src_start..src_start + bytes_per_row]);
        }
        (self.staging.as_slice(), aligned as u32)
    }
}

fn gpu_preprocess(
    image: &DynamicImage,
    config: &PreprocessConfig,
    context: &GpuContext,
    pipeline: &WgpuPreprocessPipeline,
    pool: &Mutex<GpuResourcePool>,
) -> Result<PreprocessOutput> {
    let input_w = config.input_size.width;
    let input_h = config.input_size.height;
    anyhow::ensure!(
        input_w > 0 && input_h > 0,
        "input dimensions must be greater than zero"
    );

    let (orig_w, orig_h) = image.dimensions();
    let rgba = image.to_rgba8();
    let device = context.device();
    let queue = context.queue();

    let src_size = wgpu::Extent3d {
        width: orig_w,
        height: orig_h,
        depth_or_array_layers: 1,
    };

    let output_pixels = (input_w * input_h) as usize;
    let output_f32_len = output_pixels * 3;
    let output_size_bytes = (output_f32_len * std::mem::size_of::<f32>()) as u64;

    let mut pool_guard = pool.lock().expect("gpu resource pool poisoned");
    let mut buffers = pool_guard.acquire(device, src_size, output_size_bytes);
    drop(pool_guard);

    let texture_handle = buffers.texture.clone();
    let texture_view = texture_handle.create_view(&wgpu::TextureViewDescriptor::default());
    let storage_buffer = buffers.storage_buffer().clone();
    let readback_buffer = buffers.readback_buffer().clone();
    let uniform_buffer = buffers.uniform_buffer().clone();

    let (input_bytes, bytes_per_row) = buffers.prepare_upload(rgba.as_raw(), orig_w);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture_handle,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        input_bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(orig_h),
        },
        src_size,
    );
    let uniforms = PreprocessUniforms {
        src_size: [orig_w, orig_h],
        dst_size: [input_w, input_h],
    };
    queue.write_buffer(&uniform_buffer, 0, bytes_of(&uniforms));

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("yunet_preprocess_bind_group"),
        layout: &pipeline.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&pipeline.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("yunet_preprocess_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("yunet_preprocess_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups_x = input_w.div_ceil(8);
        let workgroups_y = input_h.div_ceil(8);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &readback_buffer, 0, output_size_bytes);
    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = readback_buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| anyhow::anyhow!("device poll failed during preprocessing: {e}"))?;
    receiver
        .recv()
        .map_err(|_| anyhow::anyhow!("GPU map callback was dropped"))?
        .map_err(|e| anyhow::anyhow!("failed to map GPU preprocessing buffer: {e}"))?;
    let data = buffer_slice.get_mapped_range();
    let floats: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback_buffer.unmap();

    let mut pool_guard = pool.lock().expect("gpu resource pool poisoned");
    pool_guard.recycle(buffers);

    anyhow::ensure!(
        floats.len() == output_f32_len,
        "unexpected GPU output size (expected {}, got {})",
        output_f32_len,
        floats.len()
    );

    let shape = [1usize, 3, input_h as usize, input_w as usize];
    let tensor = Tensor::from_shape(&shape, &floats)
        .map_err(|e| anyhow::anyhow!("failed to build tensor: {e}"))?;

    let (scale_x, scale_y) = compute_resize_scales((orig_w, orig_h), (input_w, input_h))?;

    Ok(PreprocessOutput {
        tensor,
        scale_x,
        scale_y,
        original_size: (orig_w, orig_h),
    })
}

fn align_to(value: usize, alignment: usize) -> usize {
    if value.is_multiple_of(alignment) {
        value
    } else {
        value + (alignment - (value % alignment))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};
    use yunet_utils::config::{InputDimensions, ResizeQuality};

    #[test]
    fn preprocess_generates_bgr_tensor() {
        let mut img = ImageBuffer::<Rgb<u8>, _>::new(4, 4);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            let value = ((x + y) * 32) as u8;
            *pixel = Rgb([value, value / 2, 255]);
        }

        let dynamic = DynamicImage::ImageRgb8(img);
        let config = PreprocessConfig {
            input_size: InputSize::new(2, 2),
            ..Default::default()
        };

        let output =
            preprocess_dynamic_image(&dynamic, &config).expect("preprocess should succeed");

        assert_eq!(output.original_size, (4, 4));
        assert_eq!(output.scale_x, 2.0);
        assert_eq!(output.scale_y, 2.0);
        assert_eq!(output.tensor.shape(), &[1, 3, 2, 2]);

        let data = output.tensor.as_slice::<f32>().unwrap();
        assert!(data.iter().all(|v| *v >= 0.0 && *v <= 255.0));
    }

    #[test]
    fn converts_dimensions_into_configs() {
        let dims = InputDimensions {
            width: 320,
            height: 240,
            resize_quality: ResizeQuality::Quality,
        };

        let size: InputSize = dims.into();
        assert_eq!(size.width, 320);
        assert_eq!(size.height, 240);

        let config: PreprocessConfig = dims.into();
        assert_eq!(config.input_size.width, 320);
        assert_eq!(config.input_size.height, 240);
    }

    #[test]
    fn cpu_preprocessor_trait_matches_helpers() {
        let mut img = ImageBuffer::<Rgb<u8>, _>::new(2, 2);
        for (i, pixel) in img.pixels_mut().enumerate() {
            *pixel = Rgb([(i * 10) as u8, 0, 255]);
        }
        let dynamic = DynamicImage::ImageRgb8(img);
        let config = PreprocessConfig {
            input_size: InputSize::new(2, 2),
            ..Default::default()
        };

        let cpu = CpuPreprocessor;
        let trait_output = cpu.preprocess(&dynamic, &config).expect("trait preprocess");
        let helper_output =
            preprocess_dynamic_image(&dynamic, &config).expect("function preprocess");

        assert_eq!(trait_output.original_size, helper_output.original_size);
        assert_eq!(trait_output.scale_x, helper_output.scale_x);
        assert_eq!(trait_output.scale_y, helper_output.scale_y);
        assert_eq!(trait_output.tensor.shape(), helper_output.tensor.shape());

        let trait_data = trait_output.tensor.as_slice::<f32>().unwrap();
        let helper_data = helper_output.tensor.as_slice::<f32>().unwrap();
        assert_eq!(trait_data, helper_data);
    }
}
