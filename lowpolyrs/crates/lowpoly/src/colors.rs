use image::{Pixel, Rgba};
use log::warn;
use rand::{seq::SliceRandom, thread_rng};

pub fn find_mean_color(pixels: &Vec<Rgba<u8>>) -> Rgba<u8> {
    let (r, g, b) = pixels.iter().fold((0, 0, 0), |acc, pixel| {
        let channels = pixel.channels();
        (
            acc.0 + channels[0] as u32,
            acc.1 + channels[1] as u32,
            acc.2 + channels[2] as u32,
        )
    });
    let num_pixels = pixels.len() as u32;
    // Skip if the triangle has no pixels
    if num_pixels == 0 {
        warn!("Detected a triangle with no pixels when filling polygons.");
        return Rgba([0, 0, 0, 0]);
    }
    Rgba([
        (r / num_pixels) as u8,
        (g / num_pixels) as u8,
        (b / num_pixels) as u8,
        255,
    ])
}

/// Returns the 'dominant' color in a collection of RGBA pixels by:
/// 1. Choosing k random centroids
/// 2. Iterating the k-means assignment & update steps
/// 3. Picking the centroid of the largest cluster
pub fn find_dominant_color(pixels: &Vec<Rgba<u8>>) -> Rgba<u8> {
    // If we have no pixels, return a default color (transparent black here).
    if pixels.is_empty() {
        return Rgba([0, 0, 0, 0]);
    }

    // Number of clusters
    let k = 3;
    // If the number of pixels is smaller than the number of clusters, skip k-means
    if pixels.len() < k {
        return find_mean_color(pixels);
    }
    // Number of iterations (tweak as necessary)
    let max_iterations = 10;

    let mut rng = thread_rng();
    // Pick k initial centroids randomly from the pixels
    let mut centroids: Vec<Rgba<u8>> = pixels.choose_multiple(&mut rng, k).cloned().collect();

    // Repeatedly perform assignment and update
    for _ in 0..max_iterations {
        // Assign each pixel to the nearest centroid
        let mut cluster_assignments = vec![0; pixels.len()];
        for (i, px) in pixels.iter().enumerate() {
            let mut min_dist = std::u32::MAX;
            let mut assigned_cluster = 0;
            for (c_idx, c) in centroids.iter().enumerate() {
                let dist = color_distance_squared(px, c);
                if dist < min_dist {
                    min_dist = dist;
                    assigned_cluster = c_idx;
                }
            }
            cluster_assignments[i] = assigned_cluster;
        }

        // Prepare accumulators to compute new centroids
        let mut sum_rgba = vec![(0u64, 0u64, 0u64, 0u64); k];
        let mut count = vec![0u64; k];

        // Sum up colors in each cluster
        for (px, &cluster) in pixels.iter().zip(cluster_assignments.iter()) {
            let [r, g, b, a] = px.0;
            let (sr, sg, sb, sa) = sum_rgba[cluster];
            sum_rgba[cluster] = (sr + r as u64, sg + g as u64, sb + b as u64, sa + a as u64);
            count[cluster] += 1;
        }

        // Compute new centroids (mean color per cluster)
        let mut new_centroids = Vec::with_capacity(k);
        for c_idx in 0..k {
            if count[c_idx] == 0 {
                // If a cluster has no members, keep the old centroid or reinitialize
                new_centroids.push(centroids[c_idx]);
            } else {
                let (sr, sg, sb, sa) = sum_rgba[c_idx];
                let n = count[c_idx];
                new_centroids.push(Rgba([
                    (sr / n) as u8,
                    (sg / n) as u8,
                    (sb / n) as u8,
                    (sa / n) as u8,
                ]));
            }
        }

        // Check for convergence (no change) to break early
        if new_centroids == centroids {
            break;
        }
        centroids = new_centroids;
    }

    // Final assignment to figure out which cluster is largest
    let mut cluster_size = vec![0; k];
    for px in pixels {
        let mut min_dist = std::u32::MAX;
        let mut assigned_cluster = 0;
        for (c_idx, c) in centroids.iter().enumerate() {
            let dist = color_distance_squared(px, c);
            if dist < min_dist {
                min_dist = dist;
                assigned_cluster = c_idx;
            }
        }
        cluster_size[assigned_cluster] += 1;
    }

    // The 'dominant' cluster is the one with the largest membership
    let (dominant_cluster, _) = cluster_size
        .iter()
        .enumerate()
        .max_by_key(|&(_, &size)| size)
        .unwrap();

    centroids[dominant_cluster]
}

/// Computes the squared distance between two RGBA<u8> values in 4D space.
/// This avoids floating-point operations (faster for simple k-means).
fn color_distance_squared(c1: &Rgba<u8>, c2: &Rgba<u8>) -> u32 {
    let [r1, g1, b1, a1] = c1.0;
    let [r2, g2, b2, a2] = c2.0;
    let dr = r1 as i32 - r2 as i32;
    let dg = g1 as i32 - g2 as i32;
    let db = b1 as i32 - b2 as i32;
    let da = a1 as i32 - a2 as i32;
    (dr * dr + dg * dg + db * db + da * da) as u32
}
