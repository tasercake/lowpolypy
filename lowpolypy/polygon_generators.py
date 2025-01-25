import numpy as np
import shapely
from shapely import MultiPoint, Point, Polygon
from shapely.ops import triangulate, unary_union


def generate_delaunay_polygons(*, points: MultiPoint) -> list[shapely.Polygon]:
    triangles = triangulate(points)
    return triangles


# TODO: Implement `generate_voronoi_polygons`
def generate_voronoi_polygons(*, points: list[Point]) -> list[shapely.Polygon]:
    raise NotImplementedError("Voronoi not implemented yet.")


def simplify(polygons):
    threshold_area = 10.0
    small_polygons = [p for p in polygons if p.area < threshold_area]
    large_polygons = [p for p in polygons if p.area >= threshold_area]
    if small_polygons:
        merged_small = unary_union(small_polygons)
        merged_all = unary_union(large_polygons + [merged_small])
        return (
            list(merged_all) if merged_all.geom_type == "MultiPolygon" else [merged_all]
        )
    return polygons


def rescale_polygons(
    polygons: list[Polygon], size: tuple[float, float]
) -> list[Polygon]:
    """
    Rescales [0, 1] normalized polygons to the specified Width x Height.
    """
    return [
        Polygon(
            [
                (x * size[0], y * size[1])
                for x, y in zip(p.exterior.xy[0], p.exterior.xy[1])
            ]
        )
        for p in polygons
    ]
