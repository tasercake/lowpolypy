use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use imageproc::gradients::sobel_gradients;
use log::debug;
use num_traits::NumCast;
use rand::{seq::SliceRandom, thread_rng, Rng};

pub struct SobelResult<T> {
    pub sobel_image: ImageBuffer<Luma<u16>, Vec<u16>>,
    pub points: Box<dyn Iterator<Item = (T, T)>>,
}

/// Function to generate an iterator of points based on the Sobel filter applied to an image.
///
/// # Arguments
/// * `image` - A reference to a DynamicImage that represents the input image.
/// * `num_points` - The number of points to sample randomly from the Sobel gradient image.
///
/// # Type Parameters
/// * `T` - The numeric type for point coordinates. Can be integers (signed/unsigned) or floats.
///
/// # Returns
/// * `SobelResult<T>` - Contains the Sobel gradient image and an iterator of sampled points of interest.
pub fn generate_points_from_sobel<T>(image: &DynamicImage, num_points: u32) -> SobelResult<T>
where
    T: NumCast + Copy + 'static,
{
    // Convert the image to grayscale
    let grayscale: GrayImage = image.to_luma8();

    // Apply the Sobel filter to detect edges
    let sobel_gradient: ImageBuffer<Luma<u16>, Vec<u16>> = sobel_gradients(&grayscale);
    // Print the min & max values of the Sobel gradient image
    let min = sobel_gradient.pixels().map(|p| p[0]).min().unwrap();
    let max = sobel_gradient.pixels().map(|p| p[0]).max().unwrap();
    debug!("Sobel gradient min: {}, max: {}", min, max);

    // Threshold to extract significant edge points
    let threshold = 100; // This value can be tuned based on desired sensitivity
    let mut points = Vec::new();

    for (x, y, pixel) in sobel_gradient.enumerate_pixels() {
        if pixel[0] as u16 > threshold {
            // Convert coordinates to target type T
            if let (Some(tx), Some(ty)) = (T::from(x), T::from(y)) {
                points.push((tx, ty));
            }
        }
    }

    // Randomly sample `num_points` from the collected points
    let mut rng = thread_rng();
    points.shuffle(&mut rng);

    let points_iter = points.into_iter().take(num_points.try_into().unwrap());

    SobelResult {
        sobel_image: sobel_gradient,
        points: Box::new(points_iter),
    }
}

/// Generate an iterator of `num_points` random points within the given dimensions.
/// # Arguments
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `num_points` - The number of random points to generate.
pub fn generate_random_points<T>(
    width: u32,
    height: u32,
    num_points: u32,
) -> impl Iterator<Item = (T, T)>
where
    T: NumCast + Copy + 'static,
{
    let mut rng = thread_rng();
    (0..num_points).map(move |_| {
        (
            T::from(rng.gen_range(0..width)).unwrap(),
            T::from(rng.gen_range(0..height)).unwrap(),
        )
    })
}
