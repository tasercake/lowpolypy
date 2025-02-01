use num_traits::NumCast;
use spade::{DelaunayTriangulation, Point2, SpadeNum, Triangulation};

/// Returns a vector of triangles, where each triangle is represented by
/// three `(f64, f64)` coordinates.
pub fn get_delaunay_polygons<T>(points: &Vec<(T, T)>) -> Vec<[(T, T); 3]>
where
    T: NumCast + SpadeNum,
{
    // Build a DelaunayTriangulation with floating-point coordinates
    let delaunay_points: Vec<Point2<T>> = points.iter().map(|(x, y)| Point2::new(*x, *y)).collect();
    let delaunay = DelaunayTriangulation::<Point2<T>>::bulk_load_stable(delaunay_points).unwrap();

    delaunay
        .inner_faces()
        // .par_bridge()
        .map(|face| {
            let v0 = face.vertices()[0].position();
            let v1 = face.vertices()[1].position();
            let v2 = face.vertices()[2].position();
            [(v0.x, v0.y), (v1.x, v1.y), (v2.x, v2.y)]
        })
        .collect()
}
