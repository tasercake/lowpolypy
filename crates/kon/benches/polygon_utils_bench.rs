use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::{ImageBuffer, Rgba};

use kon::polygon_utils::pixels_in_triangles;

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
            let _: Vec<Vec<Rgba<u8>>> =
                pixels_in_triangles(black_box(&polygons_small), black_box(&img));
        });
    });

    // Test with a larger set of 1000 deterministic triangles of varying (but deterministic) sizes:
    let polygons_large: Vec<[(f32, f32); 3]> = (0..1000)
        .map(|i| {
            let size = (i % 100) + 1;
            let x = (i % 20) as f32 * 20.0;
            let y = (i / 20) as f32;
            [
                (x, y),
                ((x + size as f32) % IMAGE_DIMENSIONS.0 as f32, y),
                (
                    (x + (size as f32 / 2.0)) % IMAGE_DIMENSIONS.0 as f32,
                    (y + size as f32) % IMAGE_DIMENSIONS.1 as f32,
                ),
            ]
        })
        .collect();
    c.bench_function("pixels_in_triangles_large", |b| {
        b.iter(|| {
            let _: Vec<Vec<Rgba<u8>>> =
                pixels_in_triangles(black_box(&polygons_large), black_box(&img));
        });
    });
}

criterion_group!(benches, bench_pixels_in_triangles);
criterion_main!(benches);
