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
/// * `SobelResult<T, impl ParallelIterator<Item = (T, T)>>` - Contains the Sobel gradient image and an iterator of sampled points of interest.
pub fn generate_points_from_sobel<T>(
    image: &DynamicImage,
    num_points: u32,
) -> SobelResult<impl ParallelIterator<Item = (T, T)>>
where
    T: NumCast + Send,
{
    let width = image.width();

    // Convert the image to grayscale
    let grayscale: GrayImage = image.to_luma8();

    // Apply the Sobel filter to detect edges
    let sobel_gradient: ImageBuffer<Luma<u16>, Vec<u16>> = sobel_gradients(&grayscale);

    let pixel_weights: Vec<u32> = sobel_gradient.par_pixels().map(|p| p[0] as u32).collect();

    let dist = WeightedIndex::new(&pixel_weights).unwrap();
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
