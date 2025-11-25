//! Core functionality modules for the YuNet GUI application.
//!
//! This module organizes the core business logic into separate concerns:
//! - `detection`: Face detection workflow and detector management
//! - `export`: Export and batch processing operations
//! - `cache`: Caching logic for crop previews
//! - `settings`: Settings persistence and loading
//! - `quality`: Quality assessment and filtering helpers
//! - `webcam`: Webcam capture and real-time detection

pub mod cache;
pub mod detection;
pub mod export;
pub mod quality;
pub mod settings;
pub mod webcam;
