use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::Rgba;
use lowpoly::colors::{
    find_dominant_color_kmeans, find_dominant_color_median_cut, find_mean_color,
};
use rand::Rng;

fn bench_find_mean_color(c: &mut Criterion) {
    let num_pixels = 1000;
    let mut rng = rand::thread_rng();
    let pixels: Vec<Rgba<u8>> = (0..num_pixels)
        .map(|_| Rgba([rng.gen(), rng.gen(), rng.gen(), 255]))
        .collect();

    c.bench_function("find_mean_color", |b| {
        b.iter(|| {
            let _color = find_mean_color(black_box(&pixels));
        })
    });
}

fn bench_dominant_color_comparison(c: &mut Criterion) {
    let num_pixels = 10000;
    let mut rng = rand::thread_rng();
    let pixels: Vec<Rgba<u8>> = (0..num_pixels)
        .map(|_| Rgba([rng.gen(), rng.gen(), rng.gen(), 255]))
        .collect();

    let mut group = c.benchmark_group("dominant_color_comparison");

    group.bench_function("find_dominant_color_kmeans", |b| {
        b.iter(|| {
            let _color = find_dominant_color_kmeans(black_box(&pixels));
        })
    });

    group.bench_function("find_dominant_color_median_cut", |b| {
        b.iter(|| {
            let _color = find_dominant_color_median_cut(black_box(&pixels));
        })
    });

    group.finish();
}

fn bench_median_cut(c: &mut Criterion) {
    let num_pixels = 1000;
    let mut rng = rand::thread_rng();
    let pixels: Vec<Rgba<u8>> = (0..num_pixels)
        .map(|_| Rgba([rng.gen(), rng.gen(), rng.gen(), 255]))
        .collect();

    c.bench_function("median_cut", |b| {
        b.iter(|| {
            let _color = find_dominant_color_median_cut(black_box(&pixels));
        })
    });
}

criterion_group!(
    benches,
    bench_find_mean_color,
    bench_dominant_color_comparison,
    bench_median_cut
);
criterion_main!(benches);
