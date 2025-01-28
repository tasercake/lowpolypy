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

        // Compute the bounding box of the triangle, clamped to image bounds
        let min_x = clamp_to_bounds(
            triangle.iter().map(|(x, _)| *x as i32).min().unwrap_or(0),
            0,
            width as i32 - 1,
        );
        let max_x = clamp_to_bounds(
            triangle.iter().map(|(x, _)| *x as i32).max().unwrap_or(0),
            0,
            width as i32 - 1,
        );
        let min_y = clamp_to_bounds(
            triangle.iter().map(|(_, y)| *y as i32).min().unwrap_or(0),
            0,
            height as i32 - 1,
        );
        let max_y = clamp_to_bounds(
            triangle.iter().map(|(_, y)| *y as i32).max().unwrap_or(0),
            0,
            height as i32 - 1,
        );

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
/// using barycentric coordinates.
fn point_in_triangle(
    px: f64,
    py: f64,
    (x1, y1): (f64, f64),
    (x2, y2): (f64, f64),
    (x3, y3): (f64, f64),
) -> bool {
    // Using the barycentric technique:
    //
    // 1) Compute vectors:
    //    v0 = C - A
    //    v1 = B - A
    //    v2 = P - A
    //
    // 2) Compute dot products:
    //    dot00 = v0 · v0
    //    dot01 = v0 · v1
    //    dot02 = v0 · v2
    //    dot11 = v1 · v1
    //    dot12 = v1 · v2
    //
    // 3) Compute barycentric coordinates:
    //    denom = dot00 * dot11 - dot01 * dot01
    //    invDenom = 1 / denom
    //    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    //    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    //
    // 4) Check if point is in triangle:
    //    (u >= 0) && (v >= 0) && (u + v < 1)

    let (ax, ay) = (x1, y1);
    let (bx, by) = (x2, y2);
    let (cx, cy) = (x3, y3);

    let v0 = (cx - ax, cy - ay);
    let v1 = (bx - ax, by - ay);
    let v2 = (px - ax, py - ay);

    let dot00 = v0.0 * v0.0 + v0.1 * v0.1;
    let dot01 = v0.0 * v1.0 + v0.1 * v1.1;
    let dot02 = v0.0 * v2.0 + v0.1 * v2.1;
    let dot11 = v1.0 * v1.0 + v1.1 * v1.1;
    let dot12 = v1.0 * v2.0 + v1.1 * v2.1;

    let denom = dot00 * dot11 - dot01 * dot01;
    if denom.abs() < f64::EPSILON {
        // Degenerate triangle => treat as "no coverage."
        return false;
    }

    let inv_denom = 1.0 / denom;
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    (u >= 0.0) && (v >= 0.0) && (u + v < 1.0)
}

/// Clamps value to the given min/max inclusive.
fn clamp_to_bounds(value: i32, min_val: i32, max_val: i32) -> i32 {
    max(min_val, min(value, max_val))
}
