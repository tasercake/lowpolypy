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

use colors::find_dominant_color_median_cut;
use num_traits::{Num, NumCast};
use point_generators::{generate_points_from_sobel, generate_random_points, SobelResult};
use polygon_generators::get_delaunay_polygons;
use polygon_utils::pixels_in_triangles;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Return struct for the `to_lowpoly` function.
/// # Fields
/// * `original` - The original image.
/// * `points` - The anchor points sampled from the image.
/// * `polygons` - The Delaunay triangulation polygons.
/// * `debug_images` - Intermediate images for debugging. Only returned if `debug` is true.
/// * `lowpoly` - The low-poly version of the image.
pub struct LowPolyResult<T: Num> {
    pub points: Vec<(T, T)>,
    pub polygons: Vec<[(T, T); 3]>,
    pub debug_images: Vec<Option<DynamicImage>>,
    pub lowpoly_image: RgbaImage,
}

/// Generate a low-poly version of an image.
///
/// # Arguments
/// * `image` - The source image to generate a low-poly version of.
/// * `num_points` - The number of anchor points to sample from the image. Default is 1000.
/// * `sharpness` - The emphasis placed on edges in the images. Higher values make the edges more prominent. Default is 2.2.
/// * `num_random_points` - The number of random points to generate.
/// * `output_size` - The size (longest side) of the final output image.
/// * `debug` - Whether to draw and return intermediate images for debugging.
pub fn to_lowpoly(
    image: &DynamicImage,
    num_points: u32,
    sharpness: f32,
    num_random_points: u32,
    output_size: u32,
    debug: bool,
) -> Result<LowPolyResult<f32>, Box<dyn std::error::Error>> {
    // Check if it's "empty" (in the sense of zero dimensions):
    if image.width() == 0 || image.height() == 0 {
        error!("Error: The loaded image has zero width or height.");
        std::process::exit(1);
    }

    // Generate anchor points from the image
    let SobelResult {
        sobel_image,
        points,
    } = generate_points_from_sobel::<f32>(&image, num_points, sharpness);
    // Generate a few more random points around the image
    let random_points =
        generate_random_points::<f32>(image.width(), image.height(), num_random_points);
    let (width, height) = (image.width() as f32, image.height() as f32);
    let corner_points = vec![
        (0.0, 0.0),
        (width - 1.0, 0.0),
        (0.0, height - 1.0),
        (width - 1.0, height - 1.0),
    ];
    let points: Vec<(f32, f32)> = points
        .into_iter()
        .chain(corner_points.into_iter())
        .chain(random_points.into_iter())
        .collect();

    // Remove the points_vec collection and pass iterator directly
    let polygons = get_delaunay_polygons(&points);

    // Create a new target image to draw the low-poly version on
    let (width, height) = image.dimensions();
    let mut debug_image_buffer = if debug {
        Some(RgbImage::from_pixel(width, height, Rgb([0, 0, 0])))
    } else {
        None
    };
    let mut lowpoly_image_buffer = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 0]));

    // For drawing, use the original points iterator
    match &mut debug_image_buffer {
        Some(buffer) => draw_points(buffer, &points),
        None => (),
    }

    // Compute the fill color for each polygon
    let pixels_per_polygon = pixels_in_triangles(&polygons, image);
    // For each polygon, compute the average color of the pixels within it
    let polygon_colors = pixels_per_polygon
        .into_par_iter()
        .map(|pixels| find_dominant_color_median_cut(&pixels))
        .collect();
    draw_polygons_filled(&mut lowpoly_image_buffer, &polygons, &polygon_colors);

    match &mut debug_image_buffer {
        Some(buffer) => draw_polygon_edges(buffer, &polygons),
        None => (),
    }

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
        points: points,
        polygons: polygons,
        debug_images: vec![
            match debug_image_buffer {
                Some(buffer) => Some(DynamicImage::ImageRgb8(buffer)),
                None => None,
            },
            Some(DynamicImage::ImageLuma16(sobel_image)),
        ],
        lowpoly_image: resized,
    })
}

fn draw_points<T>(image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, points: &Vec<(T, T)>)
where
    T: NumCast + Copy,
{
    points.into_iter().for_each(|(x, y)| {
        draw_filled_circle_mut(
            image,
            (x.to_i32().unwrap(), y.to_i32().unwrap()),
            8,
            Rgb([255, 0, 0]),
        );
    });
}

fn draw_polygon_edges<T>(image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, polygons: &Vec<[(T, T); 3]>)
where
    T: NumCast + Copy,
{
    for polygon in polygons {
        let (p1, p2, p3) = (polygon[0], polygon[1], polygon[2]);
        draw_antialiased_line_segment_mut(
            image,
            (p1.0.to_i32().unwrap(), p1.1.to_i32().unwrap()),
            (p2.0.to_i32().unwrap(), p2.1.to_i32().unwrap()),
            Rgb([255, 255, 255]),
            interpolate,
        );
        draw_antialiased_line_segment_mut(
            image,
            (p2.0.to_i32().unwrap(), p2.1.to_i32().unwrap()),
            (p3.0.to_i32().unwrap(), p3.1.to_i32().unwrap()),
            Rgb([255, 255, 255]),
            interpolate,
        );
        draw_antialiased_line_segment_mut(
            image,
            (p3.0.to_i32().unwrap(), p3.1.to_i32().unwrap()),
            (p1.0.to_i32().unwrap(), p1.1.to_i32().unwrap()),
            Rgb([255, 255, 255]),
            interpolate,
        );
    }
}

/// Draws each polygon from `polygons` filled with the corresponding color from `colors`.
///
/// - `image` is your target image buffer.
/// - `polygons` is an iterator of triangles or polygons where each polygon is a slice of (f32,f32) coords.
/// - `colors` is an iterator (e.g. a `Vec<Rgb<u8>>` or anything that can yield `Rgb<u8>`) providing
///   the fill color for each polygon.
///
/// The i32 cast in the points is necessary because `draw_filled_polygon_mut` expects integer pixel coordinates.
fn draw_polygons_filled(
    image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    polygons: &Vec<[(f32, f32); 3]>,
    colors: &Vec<Rgba<u8>>,
) {
    // Zip the polygons together with their respective colors.
    polygons
        .into_iter()
        .zip(colors)
        .for_each(|(polygon, color)| {
            // Convert each (f64,f64) into an `imageproc::point::Point<i32>`.
            let pts: Vec<Point<i32>> = polygon
                .iter()
                .map(|&(x, y)| Point::new(x as i32, y as i32))
                .collect();
            draw_antialiased_polygon_mut(image, &pts, *color, interpolate);
        });
}
