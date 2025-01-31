use criterion::{criterion_group, criterion_main, Criterion};
use image::open;
use image::GenericImageView;
use lowpoly::to_lowpoly;

fn bench_to_lowpoly(c: &mut Criterion) {
    // Load the image
    let path = "benches/sample_images/bird.jpg";
    let image = open(path).expect("Failed to open image");

    let (width, height) = image.dimensions();
    let mut config = c.benchmark_group("to_lowpoly");
    config.sample_size(10);
    config.bench_function("to_lowpoly", |b| {
        b.iter_with_large_drop(|| {
            let _result = to_lowpoly(image.clone(), 2500, 2.2, 1500, width.max(height), false)
                .expect("Failed to generate low-poly image");
        });
    });
    config.finish();
}

criterion_group!(benches, bench_to_lowpoly);
criterion_main!(benches);
