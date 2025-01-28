use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage, Rgba, RgbaImage};
use imageproc::drawing::{
    draw_antialiased_line_segment_mut, draw_antialiased_polygon_mut, draw_filled_circle_mut,
};
use imageproc::pixelops::interpolate;
use imageproc::point::Point;
use log::{error, info};

pub mod colors;
pub mod file_utils;
pub mod point_generators;
pub mod polygon_generators;
pub mod polygon_utils;

use colors::find_dominant_color_kmeans;
use point_generators::{generate_points_from_sobel, generate_random_points, SobelResult};
use polygon_generators::get_delaunay_polygons;
use polygon_utils::pixels_in_triangles;

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
    pub lowpoly_image: RgbaImage,
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
    // Generate a few more random points around the image
    let random_points = generate_random_points(
        image.width(),
        image.height(),
        num_random_points.unwrap_or(100),
    );
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
    let mut lowpoly_image_buffer = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 0]));

    draw_points(&mut debug_image_buffer, &points);

    let polygons = get_delaunay_polygons(&points);

    // Compute the fill color for each polygon
    let pixels_per_polygon = pixels_in_triangles(&polygons, &image);
    // For each polygon, compute the average color of the pixels within it
    let polygon_colors = pixels_per_polygon
        .iter()
        .map(|pixels| find_dominant_color_kmeans(pixels));
    draw_polygons_filled(&mut lowpoly_image_buffer, &polygons, polygon_colors);

    draw_polygon_edges(&mut debug_image_buffer, &polygons);

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

pub fn draw_polygon_edges(
    image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    polygons: &Vec<[(f64, f64); 3]>,
) {
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

/// Draws each polygon from `polygons` filled with the corresponding color from `colors`.
///
/// - `image` is your target image buffer.
/// - `polygons` is a list of triangles or polygons where each polygon is a slice of (f64,f64) coords.
/// - `colors` is an iterator (e.g. a `Vec<Rgb<u8>>` or anything that can yield `Rgb<u8>`) providing
///   the fill color for each polygon.
///
/// The i32 cast in the points is necessary because `draw_filled_polygon_mut` expects integer pixel coordinates.
pub fn draw_polygons_filled<C>(
    image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    polygons: &Vec<[(f64, f64); 3]>,
    colors: C,
) where
    C: IntoIterator<Item = Rgba<u8>>,
{
    // Zip the polygons together with their respective colors.
    for (polygon, color) in polygons.iter().zip(colors) {
        // Convert each (f64,f64) into an `imageproc::point::Point<i32>`.
        let pts: Vec<Point<i32>> = polygon
            .iter()
            .map(|&(x, y)| Point::new(x as i32, y as i32))
            .collect();

        draw_antialiased_polygon_mut(image, &pts, color, interpolate);
    }
}
