//! LRU cache helpers.

use crate::types::*;
use image::DynamicImage;
use lru::LruCache;
use std::{num::NonZeroUsize, path::PathBuf, sync::Arc};

pub fn make_detection_cache() -> LruCache<CacheKey, DetectionCacheEntry> {
    LruCache::new(NonZeroUsize::new(50).unwrap())
}

pub fn make_preview_cache() -> LruCache<CropPreviewKey, CropPreviewCacheEntry> {
    // Each entry holds a full-resolution DynamicImage plus a texture handle; 500 entries
    // can hit several GB with a few large source images. Key includes float-bit offsets
    // so slider drags thrash the cache anyway — a small ceiling is enough.
    LruCache::new(NonZeroUsize::new(64).unwrap())
}

pub fn make_image_cache() -> LruCache<PathBuf, Arc<DynamicImage>> {
    LruCache::new(NonZeroUsize::new(20).unwrap())
}
