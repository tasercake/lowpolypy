import cv2
import numpy as np
from scipy import signal
from shapely import MultiPoint, Point, box
from shapely.ops import unary_union
from scipy.spatial import cKDTree


def random_points(*, num_points: int) -> MultiPoint:
    coordinates = np.random.rand(num_points, 2)
    points = MultiPoint(coordinates)
    return points


def conv_points(
    *,
    image: np.ndarray,
    num_points: int = 1000,
    num_filler_points: int = 50,
    weight_filler_points: bool = True,
) -> MultiPoint:
    points: list[Point] = []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.array(
        [
            [-3 - 3j, 0 - 10j, +3 - 3j],
            [-10 + 0j, 0 + 0j, +10 + 0j],
            [-3 + 3j, 0 + 10j, +3 + 3j],
        ]
    )
    grad = signal.convolve2d(gray, kernel, boundary="symm", mode="same")
    mag = np.absolute(grad)
    mag = mag / mag.max()
    mag[mag <= 0.1] = 0
    mag = (mag * 255).astype(np.uint8)
    mag = cv2.equalizeHist(mag)
    weights = np.ravel(mag.astype(np.float32) / mag.sum())
    coordinates = np.arange(0, weights.size, dtype=np.uint32)
    choices = np.random.choice(coordinates, size=num_points, replace=False, p=weights)
    raw_points = np.unravel_index(choices, image.shape[:2])
    conv_points = np.stack(raw_points, axis=-1) / image.shape[:2]
    points.extend(MultiPoint(conv_points[..., ::-1]).geoms)

    if num_filler_points:
        inverse = 255 - cv2.dilate(mag, np.ones((5, 5), np.uint8), iterations=3)
        inverse = cv2.blur(inverse, ksize=(13, 13))
        weights = np.ravel(inverse.astype(np.float32) / inverse.sum())
        coordinates = np.arange(0, weights.size, dtype=np.uint32)
        choices = np.random.choice(
            coordinates,
            size=num_filler_points,
            replace=False,
            p=weights if weight_filler_points else None,
        )
        raw_points = np.unravel_index(choices, image.shape[:2])
        filler_points = np.stack(raw_points, axis=-1) / image.shape[:2]
        points.extend(MultiPoint(filler_points[..., ::-1]).geoms)

    return MultiPoint(points)


def with_boundary_points(points: MultiPoint) -> MultiPoint:
    edge_box = box(0, 0, 1, 1)
    edges = edge_box.exterior

    edge_points: list[Point] = []
    hull = unary_union(points).convex_hull
    hull_points = [Point(c) for c in hull.exterior.coords]
    edge_points = [edges.interpolate(edges.project(p)) for p in hull_points]

    corners = [Point(c) for c in edges.coords[:-1]]

    return MultiPoint(list(points.geoms) + edge_points + corners)


def remove_duplicate_points(points: MultiPoint, tolerance: float = 1e-3) -> MultiPoint:
    coordinates = np.array([p.xy for p in points.geoms]).squeeze(-1)

    tree = cKDTree(coordinates, compact_nodes=len(points.geoms) > 1000)
    close_pairs = tree.query_pairs(r=tolerance, output_type="ndarray")

    unsafe_indices = np.unique(close_pairs)
    safe_indices = np.ones(len(points.geoms), dtype=np.bool)
    safe_indices[unsafe_indices] = 0
    # safe_coordinates = coordinates[safe_indices]

    neighbors = {}
    for i, j in close_pairs:
        neighbors.setdefault(i, set()).add(j)
        neighbors.setdefault(j, set()).add(i)
    discard: set[int] = set()
    for node in range(len(points.geoms)):
        if node not in discard:
            discard.update(neighbors.get(node, set()))
    return MultiPoint([p for i, p in enumerate(points.geoms) if i not in discard])


def rescale_points(points: MultiPoint, image_size: tuple[int, int]) -> MultiPoint:
    """
    Rescales [0, 1] normalized points to the specified Width x Height.
    """
    coordinates = (
        np.array([p.xy for p in points.geoms]).squeeze(-1) * image_size
    ).round()
    return MultiPoint(coordinates)
