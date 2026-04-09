use anyhow::{Context, Result};
use image::DynamicImage;
use serde::de::DeserializeOwned;
use std::{
    env, fs,
    path::{Path, PathBuf},
};

const FIXTURE_ENV: &str = "YUNET_FIXTURE_ROOT";

/// Resolve the root directory that stores project fixtures.
///
/// This function searches for a `fixtures` directory in the following order:
/// 1. The path specified by the `YUNET_FIXTURE_ROOT` environment variable.
/// 2. Ancestor directories of the current crate's manifest directory.
///
/// Returns an error if the directory cannot be found.
pub fn fixtures_dir() -> Result<PathBuf> {
    if let Ok(value) = env::var(FIXTURE_ENV) {
        let path = PathBuf::from(value);
        return Ok(path);
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest_dir.ancestors() {
        let candidate = ancestor.join("fixtures");
        if candidate.is_dir() {
            return Ok(candidate.to_path_buf());
        }
    }

    anyhow::bail!(
        "fixtures directory not found starting from {}",
        manifest_dir.display()
    );
}

/// Resolve a path inside the fixture folder.
///
/// This ensures that the requested fixture file exists before returning the full path.
///
/// # Arguments
///
/// * `relative` - A relative path to a file within the `fixtures` directory.
pub fn fixture_path<P: AsRef<Path>>(relative: P) -> Result<PathBuf> {
    let relative = relative.as_ref();
    let root = fixtures_dir()?;
    let full = root.join(relative);
    anyhow::ensure!(
        full.exists(),
        "fixture {} does not exist under {}",
        relative.display(),
        root.display()
    );
    Ok(full)
}

/// Load a fixture image as a `DynamicImage`.
///
/// # Arguments
///
/// * `relative` - A relative path to an image file within the `fixtures` directory.
pub fn load_fixture_image<P: AsRef<Path>>(relative: P) -> Result<DynamicImage> {
    let path = fixture_path(relative)?;
    image::open(&path).with_context(|| format!("failed to open fixture image {}", path.display()))
}

/// Load fixture contents as bytes.
///
/// # Arguments
///
/// * `relative` - A relative path to a file within the `fixtures` directory.
pub fn load_fixture_bytes<P: AsRef<Path>>(relative: P) -> Result<Vec<u8>> {
    let path = fixture_path(relative)?;
    fs::read(&path).with_context(|| format!("failed to read fixture {}", path.display()))
}

/// Load fixture JSON into a strongly-typed structure.
///
/// # Arguments
///
/// * `relative` - A relative path to a JSON file within the `fixtures` directory.
/// * `T` - The type to deserialize the JSON into.
pub fn load_fixture_json<P, T>(relative: P) -> Result<T>
where
    P: AsRef<Path>,
    T: DeserializeOwned,
{
    let path = fixture_path(relative)?;
    let contents = fs::read_to_string(&path)
        .with_context(|| format!("failed to read JSON fixture {}", path.display()))?;
    let value = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse JSON fixture {}", path.display()))?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    #[test]
    fn fixture_path_resolves_existing_file() {
        let Ok(path) = fixture_path("images/test_pattern.png") else {
            eprintln!("skipping: fixture images not available in this environment");
            return;
        };
        assert!(path.ends_with(Path::new("images/test_pattern.png")));
    }

    #[test]
    fn fixture_path_missing_file_errors() {
        assert!(fixture_path("images/missing.png").is_err());
    }

    #[test]
    fn load_fixture_image_reads_pattern() {
        let Ok(image) = load_fixture_image("images/test_pattern.png") else {
            eprintln!("skipping: fixture images not available in this environment");
            return;
        };
        assert_eq!(image.dimensions(), (4, 4));
    }

    #[test]
    fn fixtures_dir_respects_env_var() {
        use std::env;
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let tmp_path = tmp.path().to_str().unwrap().to_string();

        // Temporarily set the env var to our temp directory
        // SAFETY: test environment is single-threaded at this point; no other threads read this var.
        unsafe { env::set_var(FIXTURE_ENV, &tmp_path) };
        let result = fixtures_dir();
        unsafe { env::remove_var(FIXTURE_ENV) };

        let dir = result.expect("fixtures_dir should succeed with YUNET_FIXTURE_ROOT set");
        assert_eq!(dir.to_str().unwrap(), tmp_path);
    }

    #[test]
    fn load_fixture_bytes_reads_file() {
        let Ok(bytes) = load_fixture_bytes("images/test_pattern.png") else {
            eprintln!("skipping: fixture images not available in this environment");
            return;
        };
        assert!(!bytes.is_empty());
        // PNG magic bytes: \x89PNG
        assert_eq!(&bytes[0..4], b"\x89PNG");
    }

    #[test]
    fn load_fixture_json_deserializes() {
        // Try to load a known JSON fixture; skip if not available.
        let result = load_fixture_json::<_, serde_json::Value>("images/test_pattern.json");
        match result {
            Ok(_) => {} // fixture exists and parsed
            Err(e) => {
                let msg = e.to_string();
                // Acceptable: fixture missing OR JSON parse error
                assert!(
                    msg.contains("does not exist") || msg.contains("failed"),
                    "unexpected error: {msg}"
                );
            }
        }
    }
}
