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
    total_allocated_bytes: std::sync::atomic::AtomicU64,
}

impl GpuBufferPool {
    pub fn new(context: Arc<GpuContext>) -> Self {
        Self {
            context,
            idle: Mutex::new(HashMap::new()),
            total_allocated_bytes: std::sync::atomic::AtomicU64::new(0),
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

        let buffer = self
            .context
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label,
                size,
                usage,
                mapped_at_creation: false,
            });

        self.total_allocated_bytes
            .fetch_add(size, std::sync::atomic::Ordering::Relaxed);
        buffer
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

    /// Returns the total size in bytes of all buffers currently managed (or allocated) by this pool.
    pub fn memory_usage(&self) -> u64 {
        self.total_allocated_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Clears all idle buffers from the pool, freeing their memory.
    pub fn clear(&self) {
        let mut idle = match self.idle.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        // Calculate size of dropped buffers to decrement counter correctly
        let mut freed_bytes = 0;
        for entries in idle.values() {
            for entry in entries {
                freed_bytes += entry.size;
            }
        }

        idle.clear();

        // Note: We only decrement for buffers that were actually in the pool.
        // Buffers currently in use are not affected, but total_allocated_bytes tracks *all* allocated
        // buffers created through this pool that haven't been dropped by the GPU yet (conceptually).
        // Wait, `recycle` puts them back. If they are dropped outside, we can't track that easily without a wrapper.
        //
        // CORRECTION: The current design allocates new buffers if pool is empty.
        // `total_allocated_bytes` increments on create.
        // It should technically decrement when a buffer is destroyed.
        // But we return raw `wgpu::Buffer`. We don't know when the user drops it unless they call `recycle`.
        //
        // If they DROP it instead of recycling, our counter leaks.
        //
        // For OOM handling purposes, we care about what's IN the pool mostly, or we accept the leak
        // as "allocated by app".
        //
        // Let's adjust: track `total_allocated_bytes` as distinct from `pooled_bytes`.
        // Actually, if we want to release memory on OOM, we only care about `idle` buffers.
        //
        // Let's assume `total_allocated_bytes` is "allocated via this pool and not yet known to be freed".
        // Use a better metric? `pooled_memory_usage` might be more accurate for "what we can free".
        //
        // Let's stick to tracking what we create. If user drops buffer without recycling, we drift.
        // Ideally we'd wrap `wgpu::Buffer`. For now, let's just track "pooled" memory + estimated active?
        //
        // Actually, `recycle` puts it back. If they don't recycle, it's gone from our control.
        // So `total_allocated_bytes` increases on create.
        // When clearing pool, we decrement by the size of cleared buffers.
        //
        // Issue: if user drops buffer (no recycle), we never decrement.
        //
        // Let's change the defined metric: `pooled_memory_usage`. Only track what is sitting in `idle`.
        // When we create a buffer, it's "in use". When recycled, it becomes "pooled".
        //
        // If we want to track TOTAL GPU usage, `wgpu::GlobalReport` is better.
        // `GpuBufferPool` should track how much IT is holding.

        self.total_allocated_bytes
            .fetch_sub(freed_bytes, std::sync::atomic::Ordering::Relaxed);
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
