"""
This module does the heavy lifting and is responsible for converting the input image into a low-poly stylized image.
"""

from typing import List
import cv2
from PIL import Image
import numpy as np
from scipy import spatial


# Data structures
# TODO: Image
# TODO: Point
# TODO: Polygon

# Data flow
# TODO: pre-process image (Image) -> Image
# TODO: get polygons from image (Image) -> List[Polygon]
# TODO: shade polygons (List[Point], Image) -> Image
# TODO: post-process image (image) -> Image

class Point:
    def __init__(self, x, y, **kwargs):
        self.__dict__.update(kwargs)
        self.x = x
        self.y = y

    def as_array(self):
        return np.array([self.x, self.y], dtype=np.uint16)


class Polygon:
    def __init__(self, points: List[Point], **kwargs):
        self.__dict__.update(kwargs)
        assert len(points) >= 3
        self.points = points

    def as_array(self) -> np.ndarray:
        point_arrays = [point.as_array() for point in self.points]
        return np.array(point_arrays)


class LowPolyfier:
    def __init__(self, **kwargs):
        self.options = kwargs

    def rescale_image(self, image: np.ndarray, longest_edge: int):
        height, width = image.shape[:2]
        if height >= width:
            ratio = longest_edge / height
            new_height = longest_edge
            new_width = int(width * ratio)
        else:
            ratio = longest_edge / width
            new_width = longest_edge
            new_height = int(height * ratio)
        return cv2.resize(image, (new_width, new_height))

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        longest_edge = self.options.get('longest_edge', 600)
        image = self.rescale_image(image, longest_edge)
        image = cv2.bilateralFilter(image, 9, 200, 200)
        return image

    def check_bounds(self, image, polygons):
        height, width = image.shape[:2]
        for polygon in polygons:
            for point in polygon.points:
                assert 0 <= point.x < width
                assert 0 <= point.y < height

    def get_keypoints(self, image: np.ndarray) -> List[Point]:
        num_points = self.options.get('num_points', 800)
        points = [(0, 0), (image.shape[1] - 1, 0), (0, image.shape[0] - 1), (image.shape[1] - 1, image.shape[0] - 1)]
        points += [np.random.uniform((0, 0), image.shape[:2]).astype(np.uint16)[::-1] for _ in range(num_points)]
        points = [Point(point[0], point[1]) for point in points]
        return points

    def get_polygons(self, image: np.ndarray) -> List[Polygon]:
        points = self.get_keypoints(image)
        points_array = np.array([point.as_array() for point in points])
        polygons = spatial.Delaunay(points_array)
        polygons = [Polygon([points[i] for i in simplex]) for simplex in polygons.simplices]
        self.check_bounds(image, polygons)
        return polygons

    def shade(self, polygons: List[Polygon], image: np.ndarray) -> np.ndarray:
        output_image = np.zeros(image.shape, dtype=np.uint8)
        # TODO: parallelize this
        for polygon in polygons:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            points = polygon.as_array()
            cv2.fillConvexPoly(mask, points.astype(np.int32), (255,))
            output_image[mask > 0] = cv2.mean(image, mask)[:3]
        return output_image

    def postprocess(self, image: np.ndarray) -> np.ndarray:
        return image

    def lowpolyfy(self, image: Image) -> Image:
        image = np.array(image)[:, :, ::-1].copy()
        pre_processed_image = self.preprocess(image)
        polygons = self.get_polygons(pre_processed_image)
        shaded_image = self.shade(polygons, pre_processed_image)
        post_processed_image = self.postprocess(shaded_image)
        return Image.fromarray(post_processed_image[:, :, ::-1])
