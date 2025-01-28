use num_traits::NumCast;
use spade::{DelaunayTriangulation, Point2, SpadeNum, Triangulation};

/// Returns a vector of triangles, where each triangle is represented by
/// three `(f64, f64)` coordinates.
pub fn get_delaunay_polygons<T>(points: &[(T, T)]) -> Vec<[(T, T); 3]>
where
    T: NumCast + Copy + SpadeNum,
{
    // Build a DelaunayTriangulation with floating-point coordinates
    let mut delaunay = DelaunayTriangulation::<Point2<T>>::new();

    // Insert the points into the triangulation
    for &(x, y) in points {
        let _ = delaunay.insert(Point2::new(
            NumCast::from(x).unwrap(),
            NumCast::from(y).unwrap(),
        ));
    }

    let mut polygons = Vec::new();
    polygons.reserve(delaunay.num_inner_faces());

    // Spade distinguishes between "inner" and "outer" faces,
    // so we iterate only over the valid (inner) triangular faces.
    for face in delaunay.inner_faces() {
        let v0 = face.vertices()[0].position();
        let v1 = face.vertices()[1].position();
        let v2 = face.vertices()[2].position();

        // Collect these three points into one triangle representation.
        polygons.push([(v0.x, v0.y), (v1.x, v1.y), (v2.x, v2.y)]);
    }

    polygons
}
