import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy import signal
from abc import ABCMeta, abstractmethod

import shapely
from shapely.geometry import box, Point, MultiPoint, asMultiPoint

from .utils import registry


class PointGenerator(metaclass=ABCMeta):
    """
    Takes in a PIL image and returns a list of normalized shapely points
    """
    @abstractmethod
    def forward(self, image, *args, **kwargs):
        pass

    def __call__(self, image, *args, **kwargs):
        points = self.forward(image, *args, **kwargs)
        points = self.rescale_points(points, image.size)
        points = self.remove_duplicates(points, 4)
        points = self.with_boundary_points(points, image.size)
        return points

    @staticmethod
    def rescale_points(points, image_size):
        """
        Rescales [0, 1] normalized points to the specified Width x Height.
        """
        coordinates = (np.array([p.xy for p in points]).squeeze(-1) * image_size).round()
        return list(asMultiPoint(coordinates))

    @staticmethod
    def with_boundary_points(points, image_size):
        edge_box = box(0, 0, *image_size)
        edges = edge_box.exterior

        edge_points = []
        hull = shapely.ops.unary_union(points).convex_hull
        hull_points = [Point(c) for c in hull.exterior.coords]
        edge_points = [edges.interpolate(edges.project(p)) for p in hull_points]

        corners = [Point(c) for c in edges.coords[:-1]]

        return points + edge_points + corners

    @staticmethod
    def remove_duplicates(points, tolerance):
        coordinates = np.array([p.xy for p in points]).squeeze(-1)

        tree = cKDTree(coordinates, compact_nodes=len(points) > 1000)
        close_pairs = tree.query_pairs(r=tolerance, output_type="ndarray")

        unsafe_indices = np.unique(close_pairs)
        safe_indices = np.ones(len(points), dtype=np.bool)
        safe_indices[unsafe_indices] = 0
        # safe_coordinates = coordinates[safe_indices]

        neighbors = {}
        for i, j in close_pairs:
            neighbors.setdefault(i, set()).add(j)
            neighbors.setdefault(j, set()).add(i)
        discard = set()
        keep = []
        for node in range(len(points)):
            if node not in discard:
                keep.append(node)
                discard.update(neighbors.get(node, set()))
        points = [p for i, p in enumerate(points) if i not in discard]
        return points


@registry.register("PointGenerator", "RandomPoints")
class RandomPoints(PointGenerator):
    def __init__(self, num_points=100):
        super().__init__()
        self.num_points = num_points

    def forward(self, image):
        coordinates = np.random.rand(self.num_points, 2)
        points = [Point(c) for c in coordinates]
        return points


@registry.register("PointGenerator", "ConvPoints")
class ConvPoints(PointGenerator):
    def __init__(self, num_points=1000, num_filler_points=50, weight_filler_points=True):
        super().__init__()
        self.num_points = num_points
        self.num_filler_points = num_filler_points
        self.weight_filler_points = weight_filler_points

    def forward(self, image):
        points = []

        image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel = np.array([
            [-3 - 3j, 0 - 10j, +3 - 3j],
            [-10 + 0j, 0 + 0j, +10 + 0j],
            [-3 + 3j, 0 + 10j, +3 + 3j],
        ])
        grad = signal.convolve2d(gray, kernel, boundary='symm', mode='same')
        mag = np.absolute(grad)
        mag = mag / mag.max()
        mag[mag <= 0.1] = 0
        mag = cv2.equalizeHist((mag * 255).astype(np.uint8))
        weights = np.ravel(mag.astype(np.float32) / mag.sum())
        coordinates = np.arange(0, weights.size, dtype=np.uint32)
        choices = np.random.choice(coordinates, size=self.num_points, replace=False, p=weights)
        raw_points = np.unravel_index(choices, image.shape[:2])
        conv_points = np.stack(raw_points, axis=-1) / image.shape[:2]
        points += list(MultiPoint(conv_points[..., ::-1]))

        if self.num_filler_points:
            inverse = 255 - cv2.dilate(mag, np.ones((5, 5), np.uint8), iterations=3)
            inverse = cv2.blur(inverse, ksize=(13, 13))
            weights = np.ravel(inverse.astype(np.float32) / inverse.sum())
            coordinates = np.arange(0, weights.size, dtype=np.uint32)
            choices = np.random.choice(coordinates, size=self.num_filler_points, replace=False, p=weights if self.weight_filler_points else None)
            raw_points = np.unravel_index(choices, image.shape[:2])
            filler_points = np.stack(raw_points, axis=-1) / image.shape[:2]
            points += list(MultiPoint(filler_points[..., ::-1]))

        return points
