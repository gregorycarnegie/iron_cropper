use std::fmt;
use std::sync::{Arc, Mutex};

use yunet_utils::gpu::GpuContext;

struct BufferEntry {
    buffer: wgpu::Buffer,
    size: u64,
    usage: wgpu::BufferUsages,
}

/// Simple best-fit pool for `wgpu::Buffer` allocations so inference layers can
/// recycle storage instead of thrashing the driver with create/destroy calls.
pub struct GpuBufferPool {
    context: Arc<GpuContext>,
    idle: Mutex<Vec<BufferEntry>>,
}

impl GpuBufferPool {
    pub fn new(context: Arc<GpuContext>) -> Self {
        Self {
            context,
            idle: Mutex::new(Vec::new()),
        }
    }

    pub fn acquire(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> wgpu::Buffer {
        if let Some(entry) = self.take_best_fit(size, usage) {
            return entry.buffer;
        }
        self.context
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label,
                size,
                usage,
                mapped_at_creation: false,
            })
    }

    pub fn recycle(&self, buffer: wgpu::Buffer, size: u64, usage: wgpu::BufferUsages) {
        match self.idle.lock() {
            Ok(mut idle) => idle.push(BufferEntry {
                buffer,
                size,
                usage,
            }),
            Err(poisoned) => poisoned.into_inner().push(BufferEntry {
                buffer,
                size,
                usage,
            }),
        }
    }

    pub fn available(&self) -> usize {
        self.idle
            .lock()
            .map(|buffers| buffers.len())
            .unwrap_or_else(|poisoned| poisoned.into_inner().len())
    }

    fn take_best_fit(&self, size: u64, usage: wgpu::BufferUsages) -> Option<BufferEntry> {
        let mut idle = match self.idle.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut best_index = None;
        let mut best_size = u64::MAX;
        for (index, entry) in idle.iter().enumerate() {
            if entry.usage != usage || entry.size < size {
                continue;
            }
            if entry.size < best_size {
                best_size = entry.size;
                best_index = Some(index);
                if entry.size == size {
                    break;
                }
            }
        }
        best_index.map(|index| idle.swap_remove(index))
    }
}

impl fmt::Debug for GpuBufferPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let idle_len = self
            .idle
            .lock()
            .map(|idle| idle.len())
            .unwrap_or_else(|poisoned| poisoned.into_inner().len());
        f.debug_struct("GpuBufferPool")
            .field("idle_buffers", &idle_len)
            .finish()
    }
}
