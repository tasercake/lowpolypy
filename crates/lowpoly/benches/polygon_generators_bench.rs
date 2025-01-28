use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lowpoly::polygon_generators::get_delaunay_polygons;

fn bench_get_delaunay_polygons(c: &mut Criterion) {
    // Example dummy data:
    let points = vec![
        (0, 0),
        (10, 0),
        (5, 8),
        (2, 2),
        (12, 2),
        (7, 10),
        (15, 15),
        (20, 20),
        (25, 25),
        (30, 30),
        (35, 35),
    ];

    c.bench_function("get_delaunay_polygons", |b| {
        b.iter(|| {
            let _ = get_delaunay_polygons(black_box(&points));
        });
    });

    // Test with a larger set of 1000 random points
    let points_large: Vec<(u32, u32)> = (0..1000)
        .map(|_| (rand::random::<u32>() % 1000, rand::random::<u32>() % 1000))
        .collect();
    c.bench_function("get_delaunay_polygons", |b| {
        b.iter(|| {
            let _ = get_delaunay_polygons(black_box(&points_large));
        });
    });
}

criterion_group!(benches, bench_get_delaunay_polygons);
criterion_main!(benches);
