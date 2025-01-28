use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::{ImageBuffer, Rgba};

use lowpoly::polygon_utils::pixels_in_triangles;

fn bench_pixels_in_triangles(c: &mut Criterion) {
    // Example dummy data:
    let triangles = vec![
        [(0.0, 0.0), (10.0, 0.0), (5.0, 8.0)],
        [(2.0, 2.0), (12.0, 2.0), (7.0, 10.0)],
    ];
    // A small dummy image:
    let img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(400, 400);

    c.bench_function("pixels_in_triangles", |b| {
        b.iter(|| {
            let _ = pixels_in_triangles(black_box(&triangles), black_box(&img));
        });
    });
}

criterion_group!(benches, bench_pixels_in_triangles);
criterion_main!(benches);
