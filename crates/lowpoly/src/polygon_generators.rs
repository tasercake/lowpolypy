use num_traits::NumCast;
use rayon::prelude::*;
use spade::{DelaunayTriangulation, Point2, SpadeNum, Triangulation};

/// Returns a vector of triangles, where each triangle is represented by
/// three `(f64, f64)` coordinates.
pub fn get_delaunay_polygons<T>(points: impl IntoIterator<Item = (T, T)>) -> Vec<[(T, T); 3]>
where
    T: NumCast + SpadeNum + Send + Sync,
{
    // Build a DelaunayTriangulation with floating-point coordinates
    let mut delaunay = DelaunayTriangulation::<Point2<T>>::new();

    // Insert the points into the triangulation
    for (x, y) in points {
        let _ = delaunay.insert(Point2::new(
            NumCast::from(x).unwrap(),
            NumCast::from(y).unwrap(),
        ));
    }

    delaunay
        .inner_faces()
        .par_bridge()
        .map(|face| {
            let v0 = face.vertices()[0].position();
            let v1 = face.vertices()[1].position();
            let v2 = face.vertices()[2].position();
            [(v0.x, v0.y), (v1.x, v1.y), (v2.x, v2.y)]
        })
        .collect()
}
