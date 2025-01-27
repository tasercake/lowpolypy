use image::ImageFormat;
use log::{error, info};
use std::{fs, path::PathBuf};

mod point_generators;
use point_generators::generate_points_from_sobel;
mod file_utils;
use file_utils::{parse_or_infer_destination, validate_image_source};

pub fn to_lowpoly(
    source: PathBuf,
    destination: Option<PathBuf>,
    num_points: u32,
    output_size: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Validate & expand paths
    let source = validate_image_source(&source)?;
    let destination_dir = parse_or_infer_destination(&destination, &source, "lowpoly")?;
    fs::create_dir_all(&destination_dir)?;

    // 2. Construct output filename
    let source_stem = source
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let output_path = destination_dir.join(format!("{}.png", source_stem));

    // 3. Load image
    info!("Processing {} ...", source.display());
    let image = image::open(&source)
        .map_err(|e| format!("Failed to read image {}: {}", source.display(), e))?;

    // You can check if it's "empty" (in the sense of zero dimensions) if needed:
    if image.width() == 0 || image.height() == 0 {
        error!("Error: The loaded image has zero width or height.");
        std::process::exit(1);
    }

    // 4. Run the lowpoly transformation
    let points = generate_points_from_sobel(&image, num_points);
    info!("Generated {} anchor points.", points.len());

    // TODO: Resize to `output_size`
    info!("Resizing to {}x{} ...", output_size, output_size);

    // 5. Save the output
    info!("Writing output to: {}", output_path.display());
    // By default, `DynamicImage::save` picks format from file extension (PNG here).
    image.save_with_format(&output_path, ImageFormat::Png)?;

    info!("Done.");
    Ok(())
}
