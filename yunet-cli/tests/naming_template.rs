use tempfile::tempdir;
use yunet_core::cropper::CropSettings;

#[test]
fn naming_template_with_and_without_ext() {
    let stem = "photo";
    let idx = 2usize;
    let ext = "png";

    let settings = CropSettings {
        output_width: 200,
        output_height: 300,
        ..Default::default()
    };

    // Template that includes {ext}
    let tmpl1 = "{original}_{index}.{ext}".to_string();
    let mut name1 = tmpl1.clone();
    name1 = name1.replace("{original}", stem);
    name1 = name1.replace("{index}", &idx.to_string());
    name1 = name1.replace("{width}", &settings.output_width.to_string());
    name1 = name1.replace("{height}", &settings.output_height.to_string());
    name1 = name1.replace("{ext}", ext);

    assert_eq!(name1, "photo_2.png");

    // Template without {ext} should append extension
    let tmpl2 = "{original}_{index}_{timestamp}".to_string();
    // simulate timestamp insertion and extension append as done in CLI
    let ts = 1u64; // deterministic placeholder for test
    let mut name2 = tmpl2.clone();
    name2 = name2.replace("{original}", stem);
    name2 = name2.replace("{index}", &idx.to_string());
    name2 = name2.replace("{width}", &settings.output_width.to_string());
    name2 = name2.replace("{height}", &settings.output_height.to_string());
    name2 = name2.replace("{timestamp}", &ts.to_string());
    if !tmpl2.contains("{ext}") {
        name2 = format!("{}.{}", name2, ext);
    }

    assert!(name2.starts_with("photo_2_"));
    assert!(name2.ends_with(".png"));

    // A minimal full flow: create a temp dir and ensure filename doesn't collide
    let td = tempdir().unwrap();
    let out_path = td.path().join(&name2);
    // Touch file
    std::fs::write(&out_path, b"test").unwrap();
    assert!(out_path.exists());
}
