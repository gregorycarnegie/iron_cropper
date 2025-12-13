use image::DynamicImage;
use std::{path::PathBuf, sync::Arc};
use yunet_gui::YuNetApp;

#[test]
fn image_cache_respects_capacity() {
    let mut app = YuNetApp::test_instance();

    // Capacity should be 20 for image_cache as per test_instance()
    assert_eq!(app.image_cache.cap().get(), 20);

    // Fill the cache
    for i in 0..20 {
        let path = PathBuf::from(format!("test_image_{}.png", i));
        let img = Arc::new(DynamicImage::new_rgb8(1, 1));
        app.image_cache.put(path, img);
    }

    assert_eq!(app.image_cache.len(), 20);

    // Add 21st item (should evict 0)
    let path_20 = PathBuf::from("test_image_20.png");
    let img_20 = Arc::new(DynamicImage::new_rgb8(1, 1));
    app.image_cache.put(path_20.clone(), img_20);

    // Verify length is still 20
    assert_eq!(app.image_cache.len(), 20);

    // Verify 0 is gone
    let path_0 = PathBuf::from("test_image_0.png");
    assert!(app.image_cache.get(&path_0).is_none());

    // Verify 20 is present
    assert!(app.image_cache.get(&path_20).is_some());
}

#[test]
fn crop_preview_cache_respects_capacity() {
    let app = YuNetApp::test_instance();

    // Capacity 500
    assert_eq!(app.crop_preview_cache.cap().get(), 500);
}
