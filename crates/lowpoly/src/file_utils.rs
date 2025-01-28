use std::path::{Path, PathBuf};

/// Validates the source image path, ensuring it exists and is a file.
pub fn validate_image_source(source: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let abs_path = source
        .canonicalize()
        .map_err(|_| format!("Invalid source path: {}", source.display()))?;

    if !abs_path.exists() {
        return Err(format!("Source path does not exist: {}", abs_path.display()).into());
    }
    if abs_path.is_dir() {
        return Err(format!(
            "Source path is a directory, not an image: {}",
            abs_path.display()
        )
        .into());
    }

    Ok(abs_path)
}

/// Parses or infers the destination directory:
///   - If `destination` is provided, use it.
///   - Else, create <source.parent>/lowpoly
pub fn parse_or_infer_destination(
    destination: &Option<PathBuf>,
    source: &Path,
    subdir: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    match destination {
        Some(dest) => {
            // Attempt to canonicalize; if it fails, just return the original path
            let can = dest.canonicalize().unwrap_or_else(|_| dest.to_path_buf());
            Ok(can)
        }
        None => {
            let parent = source
                .parent()
                .ok_or_else(|| format!("Cannot infer parent from: {}", source.display()))?
                .join(subdir);
            Ok(parent)
        }
    }
}
