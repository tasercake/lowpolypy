use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use imageproc::gradients::sobel_gradients;
use num_traits::NumCast;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::{thread_rng, Rng};
use rayon::prelude::*;

pub struct SobelResult<P> {
    pub sobel_image: ImageBuffer<Luma<u16>, Vec<u16>>,
    pub points: P,
}

/// Generate points based on the Sobel filter applied to an image.
///
/// # Arguments
/// * `image` - A reference to a DynamicImage that represents the input image.
/// * `num_points` - The number of points to sample randomly from the Sobel gradient image.
/// * `sharpness` - The sharpness of the Sobel gradient. Default is 1.0 for linear. >1.0 is more focused on edges. <1.0 is more random.
///
/// # Type Parameters
/// * `T` - The numeric type for point coordinates. Can be integers (signed/unsigned) or floats.
///
/// # Returns
/// * `SobelResult<T, impl ParallelIterator<Item = (T, T)>>` - Contains the Sobel gradient image and an iterator of sampled points of interest.
pub fn generate_points_from_sobel<T>(
    image: &DynamicImage,
    num_points: u32,
    sharpness: f32,
) -> SobelResult<impl ParallelIterator<Item = (T, T)>>
where
    T: NumCast + Send,
{
    let width = image.width();

    // Convert the image to grayscale
    let grayscale: GrayImage = image.to_luma8();

    // Apply the Sobel filter to detect edges
    let sobel_gradient: ImageBuffer<Luma<u16>, Vec<u16>> = sobel_gradients(&grayscale);

    // Raise each pixel's Sobel magnitude to `sharpness` power
    let pixel_weights: Vec<f32> = sobel_gradient
        .par_pixels()
        .map(|p| {
            let mag = p[0] as f32;
            mag.powf(sharpness)
        })
        .collect();
    // Normalize the weights to the range [0, 1]
    let max_weight = pixel_weights
        .par_iter()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap();
    let pixel_weights: Vec<f32> = pixel_weights.par_iter().map(|w| w / max_weight).collect();

    // Create a weighted distribution using the adjusted magnitudes.
    let dist = WeightedIndex::new(&pixel_weights)
        .expect("WeightedIndex failed: all weights were zero or invalid.");

    let points = (0..num_points).into_par_iter().map_init(
        || thread_rng(),
        move |rng, _| {
            let i = dist.sample(rng);
            let x = i as u32 % width;
            let y = i as u32 / width;
            (T::from(x).unwrap(), T::from(y).unwrap())
        },
    );

    SobelResult {
        sobel_image: sobel_gradient,
        points,
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
) -> impl ParallelIterator<Item = (T, T)>
where
    T: NumCast + Send,
{
    (0..num_points).into_par_iter().map_init(
        || thread_rng(),
        move |rng, _| {
            (
                T::from(rng.gen_range(0..width)).unwrap(),
                T::from(rng.gen_range(0..height)).unwrap(),
            )
        },
    )
}
