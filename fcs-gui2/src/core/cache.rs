//! LRU cache helpers.

use crate::types::*;
use image::DynamicImage;
use lru::LruCache;
use std::{num::NonZeroUsize, path::PathBuf, sync::Arc};

pub fn make_detection_cache() -> LruCache<CacheKey, DetectionCacheEntry> {
    LruCache::new(NonZeroUsize::new(50).unwrap())
}

pub fn make_preview_cache() -> LruCache<CropPreviewKey, CropPreviewCacheEntry> {
    LruCache::new(NonZeroUsize::new(500).unwrap())
}

pub fn make_image_cache() -> LruCache<PathBuf, Arc<DynamicImage>> {
    LruCache::new(NonZeroUsize::new(20).unwrap())
}
