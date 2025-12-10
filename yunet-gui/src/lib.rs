//! Desktop GUI for YuNet face detection (Library).

pub mod app;
pub mod app_impl;
pub mod core;
pub mod interaction;
pub mod rendering;
pub mod theme;
pub mod types;
pub mod ui;

// Re-export types for convenience
pub use types::*;
pub use yunet_utils::gpu::GpuStatusMode;
