use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::{DynamicImage, GenericImage};
use lowpoly::point_generators::{generate_points_from_sobel, generate_random_points};
use rand::Rng;

fn bench_generate_points_from_sobel(c: &mut Criterion) {
    let (height, width) = (400, 400);
    let num_points = 1000;

    // Generate an image containing random noise
    let mut img = DynamicImage::new_rgb8(width, height);
    let mut rng = rand::thread_rng();
    for y in 0..height {
        for x in 0..width {
            let r: u8 = rng.gen();
            let g: u8 = rng.gen();
            let b: u8 = rng.gen();
            img.put_pixel(x, y, image::Rgba([r, g, b, 255]));
        }
    }

    c.bench_function("generate_points_from_sobel", |b| {
        b.iter(|| {
            let _res = generate_points_from_sobel::<u16>(
                black_box(&img),
                black_box(num_points),
                black_box(2.2),
            );
        })
    });
}

fn bench_generate_random_points(c: &mut Criterion) {
    c.bench_function("generate_random_points", |b| {
        b.iter(|| {
            let _: Vec<(f32, f32)> =
                generate_random_points(black_box(1000), black_box(1000), black_box(500));
        })
    });
}

criterion_group!(
    benches,
    bench_generate_points_from_sobel,
    bench_generate_random_points
);
criterion_main!(benches);
