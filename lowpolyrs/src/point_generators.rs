use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use imageproc::gradients::sobel_gradients;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Function to generate an array of points based on the Sobel filter applied to an image.
///
/// # Arguments
/// * `image` - A reference to a DynamicImage that represents the input image.
/// * `num_points` - The number of points to sample randomly from the Sobel gradient image.
///
/// # Returns
/// * `Vec<(u32, u32)>` - A vector of (x, y) coordinates representing sampled points of interest.
pub fn generate_points_from_sobel(image: &DynamicImage, num_points: u32) -> Vec<(u32, u32)> {
    // Convert the image to grayscale
    let grayscale: GrayImage = image.to_luma8();

    // Apply the Sobel filter to detect edges
    let sobel_gradient: ImageBuffer<Luma<u16>, Vec<u16>> = sobel_gradients(&grayscale);

    // Threshold to extract significant edge points
    let threshold = 100; // This value can be tuned based on desired sensitivity
    let mut points = Vec::new();

    for (x, y, pixel) in sobel_gradient.enumerate_pixels() {
        if pixel[0] as u16 > threshold {
            points.push((x, y));
        }
    }

    // Randomly sample `num_points` from the collected points
    let mut rng = thread_rng();
    points.shuffle(&mut rng);

    points
        .into_iter()
        .take(num_points.try_into().unwrap())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageReader;

    #[test]
    fn test_generate_points_from_sobel() {
        // Load a sample image
        let image = ImageReader::open("../images/bird1.jpg")
            .unwrap()
            .decode()
            .unwrap();

        // Generate points using Sobel filter
        let num_points = 100;
        let points = generate_points_from_sobel(&image, num_points);

        // Assert that points are non-empty and within the limit
        assert!(!points.is_empty());
        assert!(points.len() <= num_points.try_into().unwrap());

        // Optionally, print some points for debugging
        println!("Generated points: {:?}", &points[0..10.min(points.len())]);
    }
}
