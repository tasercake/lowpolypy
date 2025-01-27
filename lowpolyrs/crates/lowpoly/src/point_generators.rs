use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use imageproc::gradients::sobel_gradients;
use log::debug;
use rand::{seq::SliceRandom, thread_rng, Rng};
use std::time::Instant;

/// Function to generate an array of points based on the Sobel filter applied to an image.
///
/// # Arguments
/// * `image` - A reference to a DynamicImage that represents the input image.
/// * `num_points` - The number of points to sample randomly from the Sobel gradient image.
///
/// # Returns
/// * `Vec<(u32, u32)>` - A vector of (x, y) coordinates representing sampled points of interest.
pub fn generate_points_from_sobel(image: &DynamicImage, num_points: u32) -> Vec<(u32, u32)> {
    let mut start = Instant::now();
    // Convert the image to grayscale
    let grayscale: GrayImage = image.to_luma8();
    debug!("Time after converting to grayscale: {:?}", start.elapsed());
    start = Instant::now();

    // Apply the Sobel filter to detect edges
    let sobel_gradient: ImageBuffer<Luma<u16>, Vec<u16>> = sobel_gradients(&grayscale);
    debug!("Time after sobel_gradients: {:?}", start.elapsed());
    start = Instant::now();

    // Threshold to extract significant edge points
    let threshold = 100; // This value can be tuned based on desired sensitivity
    let mut points = Vec::new();

    for (x, y, pixel) in sobel_gradient.enumerate_pixels() {
        if pixel[0] as u16 > threshold {
            points.push((x, y));
        }
    }
    debug!("Time after thresholding edges: {:?}", start.elapsed());
    start = Instant::now();

    // Randomly sample `num_points` from the collected points
    let mut rng = thread_rng();
    points.shuffle(&mut rng);
    debug!("Time after random sampling: {:?}", start.elapsed());
    start = Instant::now();

    let sampled_points = points
        .into_iter()
        .take(num_points.try_into().unwrap())
        .collect();
    debug!("Time after taking sampled points: {:?}", start.elapsed());

    sampled_points
}

/// Generate `num_points` random points within the given dimensions.
/// # Arguments
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `num_points` - The number of random points to generate.
pub fn generate_random_points(width: u32, height: u32, num_points: u32) -> Vec<(u32, u32)> {
    let mut rng = thread_rng();
    (0..num_points)
        .map(|_| (rng.gen_range(0..width), rng.gen_range(0..height)))
        .collect()
}
