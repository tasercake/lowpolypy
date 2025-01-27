use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageBuffer, ImageFormat, Rgba};
use imageproc::drawing::draw_antialiased_polygon;
use imageproc::pixelops::interpolate;
use imageproc::point::Point;

use log::{error, info};
use std::{fs, path::PathBuf};

mod point_generators;
use point_generators::generate_points_from_sobel;
mod polygon_generators;
use polygon_generators::get_delaunay_polygons;
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

    let polygons = get_delaunay_polygons(&points);
    info!("Generated {} polygons.", polygons.len());

    // Create a new image to draw the polygons on
    let (width, height) = image.dimensions();
    let mut polygon_vis = ImageBuffer::from_pixel(width, height, Rgba([0, 0, 0, 255]));
    // Draw each polygon in white
    for poly in &polygons {
        // Convert your polygon's points into imageproc's Point<i32> type
        let points: Vec<Point<i32>> = poly
            .iter()
            .map(|p| Point::new(p.0 as i32, p.1 as i32))
            .collect();

        // Fill it with white (use `draw_hollow_polygon_mut` if you prefer just an outline)
        draw_antialiased_polygon(
            &mut polygon_vis,
            &points,
            Rgba([255, 255, 255, 255]),
            interpolate,
        );
    }

    // Resize to `output_size` but preserve aspect ratio
    info!("Resizing to {}x{} ...", output_size, output_size);
    let resized = image::imageops::resize(
        &DynamicImage::ImageRgba8(polygon_vis),
        output_size,
        output_size,
        FilterType::CatmullRom,
    );

    // 5. Save the output
    info!("Writing output to: {}", output_path.display());
    // By default, `DynamicImage::save` picks format from file extension (PNG here).
    resized.save_with_format(&output_path, ImageFormat::Png)?;

    info!("Done.");
    Ok(())
}
