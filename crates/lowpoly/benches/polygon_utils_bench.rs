use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::{ImageBuffer, Rgba};

use lowpoly::polygon_utils::pixels_in_triangles;

const IMAGE_DIMENSIONS: (u32, u32) = (400, 400);

fn bench_pixels_in_triangles(c: &mut Criterion) {
    // A small dummy image:
    let img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(IMAGE_DIMENSIONS.0, IMAGE_DIMENSIONS.1);

    // Example dummy data:
    let polygons_small = vec![
        [(0.0, 0.0), (10.0, 0.0), (5.0, 8.0)],
        [(2.0, 2.0), (12.0, 2.0), (7.0, 10.0)],
    ];

    c.bench_function("pixels_in_triangles", |b| {
        b.iter(|| {
            let _ = pixels_in_triangles(black_box(&polygons_small), black_box(&img));
        });
    });

    // Test with a larger set of 1000 random triangles:
    let polygons_large = (0..1000)
        .map(|_| {
            let x1 = rand::random::<f64>() * IMAGE_DIMENSIONS.0 as f64;
            let y1 = rand::random::<f64>() * IMAGE_DIMENSIONS.1 as f64;
            let x2 = rand::random::<f64>() * IMAGE_DIMENSIONS.0 as f64;
            let y2 = rand::random::<f64>() * IMAGE_DIMENSIONS.1 as f64;
            let x3 = rand::random::<f64>() * IMAGE_DIMENSIONS.0 as f64;
            let y3 = rand::random::<f64>() * IMAGE_DIMENSIONS.1 as f64;
            [(x1, y1), (x2, y2), (x3, y3)]
        })
        .collect::<Vec<[(f64, f64); 3]>>();
    c.bench_function("pixels_in_triangles", |b| {
        b.iter(|| {
            let _ = pixels_in_triangles(black_box(&polygons_large), black_box(&img));
        });
    });
}

criterion_group!(benches, bench_pixels_in_triangles);
criterion_main!(benches);
