use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lowpoly::polygon_generators::get_delaunay_polygons;

fn bench_get_delaunay_polygons(c: &mut Criterion) {
    // Example dummy data:
    let points: Vec<(f32, f32)> = vec![
        (0.0, 0.0),
        (10.0, 0.0),
        (5.0, 8.0),
        (2.0, 2.0),
        (12.0, 2.0),
        (7.0, 10.0),
        (15.0, 15.0),
        (20.0, 20.0),
        (25.0, 25.0),
        (30.0, 30.0),
        (35.0, 35.0),
    ];

    c.bench_function("get_delaunay_polygons", |b| {
        b.iter(|| {
            let _ = get_delaunay_polygons(black_box(points.clone()));
        });
    });

    // Test with a larger set of 1000 random points
    let points_large: Vec<(f32, f32)> = (0..1000)
        .map(|_| {
            (
                rand::random::<f32>() % 1000.0,
                rand::random::<f32>() % 1000.0,
            )
        })
        .collect();
    c.bench_function("get_delaunay_polygons", |b| {
        b.iter(|| {
            let _ = get_delaunay_polygons(black_box(points_large.clone()));
        });
    });
}

criterion_group!(benches, bench_get_delaunay_polygons);
criterion_main!(benches);
