use yunet_gui::{GpuStatusMode, YuNetApp};

#[test]
fn handle_oom_clears_caches() {
    // 1. Setup app
    let mut app = YuNetApp::test_instance();

    // 2. Trigger OOM
    app.handle_oom_event();

    let (img_cache, prev_cache) = app.debug_cache_stats();
    assert_eq!(img_cache, 0);
    assert_eq!(prev_cache, 0);

    // Check status
    assert!(matches!(app.gpu_status.mode, GpuStatusMode::Error));
    assert_eq!(
        app.gpu_status.detail.as_deref(),
        Some("OOM - Caches Cleared")
    );
}
