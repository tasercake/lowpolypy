use spade::{DelaunayTriangulation, Point2, Triangulation};

/// Returns a vector of triangles, where each triangle is represented by
/// three `(f64, f64)` coordinates.
pub fn get_delaunay_polygons(points: &[(u32, u32)]) -> Vec<[(f64, f64); 3]> {
    // Build a DelaunayTriangulation with floating-point coordinates
    let mut delaunay = DelaunayTriangulation::<Point2<f64>>::new();

    // Insert the points into the triangulation
    for &(x, y) in points {
        let _ = delaunay.insert(Point2::new(x as f64, y as f64));
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
