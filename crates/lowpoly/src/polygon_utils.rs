use std::cmp::{max, min};

use image::GenericImageView;

/// Returns, for each triangle, a list of the pixels that lie within it.
///
/// Ensures that even if the triangle's bounding box or coverage
/// produces no enclosed pixels, the triangle will contribute exactly one pixel
/// (from its centroid).
pub fn pixels_in_triangles<I>(triangles: &Vec<[(f64, f64); 3]>, image: &I) -> Vec<Vec<I::Pixel>>
where
    I: GenericImageView,
{
    let (width, height) = image.dimensions();
    let mut result = Vec::with_capacity(triangles.len());

    for &triangle in triangles {
        let [(x1, y1), (x2, y2), (x3, y3)] = triangle;

        let (min_x, max_x, min_y, max_y) = bounding_box_i32(x1, y1, x2, y2, x3, y3, width, height);

        let mut pixels = Vec::new();

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                // sample near the pixel center
                let px_f = px as f64 + 0.5;
                let py_f = py as f64 + 0.5;
                if point_in_triangle(px_f, py_f, (x1, y1), (x2, y2), (x3, y3)) {
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

        result.push(pixels);
    }

    result
}

/// Check if a point (px, py) lies within the triangle formed by (x1,y1), (x2,y2), (x3,y3)
/// using half-plane checks.
fn point_in_triangle(
    px: f64,
    py: f64,
    (x1, y1): (f64, f64),
    (x2, y2): (f64, f64),
    (x3, y3): (f64, f64),
) -> bool {
    // Half-plane checks:
    let sign = |(ax, ay), (bx, by), (cx, cy)| (ax - cx) * (by - cy) - (bx - cx) * (ay - cy);
    let d1 = sign((px, py), (x1, y1), (x2, y2));
    let d2 = sign((px, py), (x2, y2), (x3, y3));
    let d3 = sign((px, py), (x3, y3), (x1, y1));
    let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
    !(has_neg && has_pos)
}

/// Clamps value to the given min/max inclusive.
fn clamp_to_bounds(value: i32, min_val: i32, max_val: i32) -> i32 {
    max(min_val, min(value, max_val))
}

fn bounding_box_i32(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    x3: f64,
    y3: f64,
    width: u32,
    height: u32,
) -> (i32, i32, i32, i32) {
    let (min_x, max_x) = {
        let (min_xf, max_xf) = (x1.min(x2).min(x3), x1.max(x2).max(x3));
        (
            clamp_to_bounds(min_xf as i32, 0, width as i32 - 1),
            clamp_to_bounds(max_xf as i32, 0, width as i32 - 1),
        )
    };
    let (min_y, max_y) = {
        let (min_yf, max_yf) = (y1.min(y2).min(y3), y1.max(y2).max(y3));
        (
            clamp_to_bounds(min_yf as i32, 0, height as i32 - 1),
            clamp_to_bounds(max_yf as i32, 0, height as i32 - 1),
        )
    };
    (min_x, max_x, min_y, max_y)
}
