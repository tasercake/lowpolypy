use std::cmp::{max, min};

use image::GenericImageView;

/// Returns, for each triangle, a list of the pixels that lie within it.
///
/// Ensures that even if the triangle's bounding box or coverage
/// produces no enclosed pixels, the triangle will contribute exactly one pixel
/// (from its centroid).
pub fn pixels_in_triangles<'a, I, T>(
    triangles: T,
    image: &'a I,
) -> impl Iterator<Item = Vec<I::Pixel>> + 'a
where
    I: GenericImageView,
    T: IntoIterator<Item = [(f32, f32); 3]> + 'a,
{
    let (width, height) = image.dimensions();

    triangles.into_iter().map(move |triangle| {
        let [(x1, y1), (x2, y2), (x3, y3)] = triangle;
        let (min_x, max_x, min_y, max_y) = bounding_box_i32(x1, y1, x2, y2, x3, y3, width, height);
        let mut pixels = Vec::with_capacity((max_x - min_x + 1) as usize);

        // Keep the pixel scanning loop sequential per triangle
        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let px_center = px as f32 + 0.5;
                let py_center = py as f32 + 0.5;
                if point_in_triangle(px_center, py_center, (x1, y1), (x2, y2), (x3, y3)) {
                    let pixel = image.get_pixel(px as u32, py as u32);
                    pixels.push(pixel);
                }
            }
        }

        // Ensure at least one pixel is returned for each triangle
        if pixels.is_empty() {
            // Fallback: sample from triangle's centroid
            let cx = ((x1 + x2 + x3) / 3.0).round();
            let cy = ((y1 + y2 + y3) / 3.0).round();
            let cx_clamped = clamp_to_bounds(cx as i32, 0, width as i32 - 1) as u32;
            let cy_clamped = clamp_to_bounds(cy as i32, 0, height as i32 - 1) as u32;
            pixels.push(image.get_pixel(cx_clamped, cy_clamped));
        }

        pixels
    })
}

/// Check if a point (px, py) lies within the triangle formed by (x1,y1), (x2,y2), (x3,y3)
/// using barycentric coordinates.
fn point_in_triangle(
    px: f32,
    py: f32,
    (x1, y1): (f32, f32),
    (x2, y2): (f32, f32),
    (x3, y3): (f32, f32),
) -> bool {
    // Replace with optimized barycentric coordinate version
    let area = 0.5 * (-y2 * x3 + y1 * (-x2 + x3) + x1 * (y2 - y3) + x2 * y3);
    let s = (y1 * x3 - x1 * y3 + (y3 - y1) * px + (x1 - x3) * py) * area.signum();
    let t = (x1 * y2 - y1 * x2 + (y1 - y2) * px + (x2 - x1) * py) * area.signum();
    s > 0.0 && t > 0.0 && (s + t) < 2.0 * area.abs()
}

/// Clamps value to the given min/max inclusive.
fn clamp_to_bounds(value: i32, min_val: i32, max_val: i32) -> i32 {
    max(min_val, min(value, max_val))
}

fn bounding_box_i32(
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    x3: f32,
    y3: f32,
    width: u32,
    height: u32,
) -> (i32, i32, i32, i32) {
    let min_xf = x1.min(x2.min(x3));
    let max_xf = x1.max(x2.max(x3));
    let min_x = (min_xf.floor() as i32).clamp(0, (width - 1) as i32);
    let max_x = (max_xf.ceil() as i32).clamp(0, (width - 1) as i32);
    let min_yf = y1.min(y2.min(y3));
    let max_yf = y1.max(y2.max(y3));
    let min_y = (min_yf.floor() as i32).clamp(0, (height - 1) as i32);
    let max_y = (max_yf.ceil() as i32).clamp(0, (height - 1) as i32);
    (min_x, max_x, min_y, max_y)
}
