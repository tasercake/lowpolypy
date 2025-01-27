use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage};
use imageproc::drawing::{draw_antialiased_line_segment_mut, draw_filled_circle_mut};

use imageproc::pixelops::interpolate;
use log::{error, info};

mod point_generators;
use point_generators::{generate_points_from_sobel, generate_random_points, SobelResult};
mod polygon_generators;
use polygon_generators::get_delaunay_polygons;

pub mod file_utils;

/// Return struct for the `to_lowpoly` function.
/// # Fields
/// * `original` - The original image.
/// * `points` - The anchor points sampled from the image.
/// * `polygons` - The Delaunay triangulation polygons.
/// * `lowpoly` - The low-poly version of the image.
pub struct LowPolyResult {
    pub original_image: DynamicImage,
    pub points: Vec<(u32, u32)>,
    pub polygons: Vec<[(f64, f64); 3]>,
    pub debug_images: Vec<DynamicImage>,
    pub lowpoly_image: RgbImage,
}

/// Generate a low-poly version of an image.
///
/// # Arguments
/// * `image` - The source image to generate a low-poly version of.
/// * `num_points` - The number of anchor points to sample from the image. Default is 1000.
/// * `num_random_points` - The number of random points to generate. Default is 100.
/// * `output_size` - The size (longest side) of the final output image.
pub fn to_lowpoly(
    image: DynamicImage,
    num_points: Option<u32>,
    num_random_points: Option<u32>,
    output_size: u32,
) -> Result<LowPolyResult, Box<dyn std::error::Error>> {
    // Check if it's "empty" (in the sense of zero dimensions):
    if image.width() == 0 || image.height() == 0 {
        error!("Error: The loaded image has zero width or height.");
        std::process::exit(1);
    }

    // Generate anchor points from the image
    let SobelResult {
        sobel_image,
        mut points,
    } = generate_points_from_sobel(&image, num_points.unwrap_or(1000));
    info!("Generated {} anchor points.", points.len());
    // Generate a few more random points around the image
    let random_points = generate_random_points(
        image.width(),
        image.height(),
        num_random_points.unwrap_or(100),
    );
    info!("Generated {} random points.", random_points.len());
    // Combine the points into one vector
    points.extend(random_points.into_iter());

    // Add image corners as anchor points
    let (width, height) = image.dimensions();
    let corners = vec![
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
    ];
    points.extend(corners.into_iter());

    // Create a new target image to draw the low-poly version on
    let (width, height) = image.dimensions();
    let mut debug_image_buffer = RgbImage::from_pixel(width, height, Rgb([0, 0, 0]));
    let lowpoly_image_buffer = RgbImage::from_pixel(width, height, Rgb([0, 0, 0]));

    draw_points(&mut debug_image_buffer, &points);

    let polygons = get_delaunay_polygons(&points);
    info!("Generated {} polygons.", polygons.len());

    draw_polygons(&mut debug_image_buffer, &polygons);

    // Resize to `output_size` but preserve aspect ratio.
    // `output_size` constrains the longest side of the generated image.
    let (new_width, new_height) = if width > height {
        (
            output_size,
            (output_size as f32 * height as f32 / width as f32) as u32,
        )
    } else {
        (
            (output_size as f32 * width as f32 / height as f32) as u32,
            output_size,
        )
    };
    info!("Resizing to {}x{} ...", new_width, new_height);
    let resized = image::imageops::resize(
        &lowpoly_image_buffer,
        new_width,
        new_height,
        FilterType::CatmullRom,
    );

    info!("Done.");
    Ok(LowPolyResult {
        original_image: image,
        points,
        polygons,
        debug_images: vec![
            DynamicImage::ImageRgb8(debug_image_buffer),
            DynamicImage::ImageLuma16(sobel_image),
        ],
        lowpoly_image: resized,
    })
}

pub fn draw_points(image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, points: &[(u32, u32)]) {
    for point in points {
        draw_filled_circle_mut(image, (point.0 as i32, point.1 as i32), 8, Rgb([255, 0, 0]));
    }
}

pub fn draw_polygons(image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, polygons: &Vec<[(f64, f64); 3]>) {
    for polygon in polygons {
        let (p1, p2, p3) = (polygon[0], polygon[1], polygon[2]);
        draw_antialiased_line_segment_mut(
            image,
            (p1.0 as i32, p1.1 as i32),
            (p2.0 as i32, p2.1 as i32),
            Rgb([255, 255, 255]),
            interpolate,
        );
        draw_antialiased_line_segment_mut(
            image,
            (p2.0 as i32, p2.1 as i32),
            (p3.0 as i32, p3.1 as i32),
            Rgb([255, 255, 255]),
            interpolate,
        );
        draw_antialiased_line_segment_mut(
            image,
            (p3.0 as i32, p3.1 as i32),
            (p1.0 as i32, p1.1 as i32),
            Rgb([255, 255, 255]),
            interpolate,
        );
    }
}
