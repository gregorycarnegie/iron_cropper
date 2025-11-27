use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::gpu::GpuContext;

struct BufferEntry {
    buffer: wgpu::Buffer,
    size: u64,
}

/// Best-fit pool for `wgpu::Buffer` allocations organized by usage flags.
/// Buffers are grouped by usage to improve search performance and avoid
/// iterating through incompatible buffers.
pub struct GpuBufferPool {
    context: Arc<GpuContext>,
    idle: Mutex<HashMap<wgpu::BufferUsages, Vec<BufferEntry>>>,
}

impl GpuBufferPool {
    pub fn new(context: Arc<GpuContext>) -> Self {
        Self {
            context,
            idle: Mutex::new(HashMap::new()),
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
            Ok(mut idle) => {
                idle.entry(usage)
                    .or_default()
                    .push(BufferEntry { buffer, size });
            }
            Err(poisoned) => {
                poisoned
                    .into_inner()
                    .entry(usage)
                    .or_default()
                    .push(BufferEntry { buffer, size });
            }
        }
    }

    pub fn available(&self) -> usize {
        self.idle
            .lock()
            .map(|map| map.values().map(|v| v.len()).sum())
            .unwrap_or_else(|poisoned| poisoned.into_inner().values().map(|v| v.len()).sum())
    }

    fn take_best_fit(&self, size: u64, usage: wgpu::BufferUsages) -> Option<BufferEntry> {
        let mut idle = match self.idle.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        // Only search buffers with matching usage flags
        let buffers = idle.get_mut(&usage)?;

        let mut best_index = None;
        let mut best_size = u64::MAX;

        for (index, entry) in buffers.iter().enumerate() {
            if entry.size < size {
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

        best_index.map(|index| buffers.swap_remove(index))
    }
}

impl fmt::Debug for GpuBufferPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let idle_count = self.available();
        f.debug_struct("GpuBufferPool")
            .field("idle_buffers", &idle_count)
            .finish()
    }
}
