"""
This module does the heavy lifting and is responsible for converting the input image into a low-poly stylized image.
"""

from typing import List
import cv2
from PIL import Image
import numpy as np


# Data structures
# TODO: Image
# TODO: Point
# TODO: Polygon

# Data flow
# TODO: pre-process image (Image) -> Image
# TODO: obtain image keypoints (Image) -> List[Point]
# TODO: join keypoints into polygons (List[Point], Image) -> List[Polygon]
# TODO: shade polygons (List[Point], Image) -> Image


class Point:
    def __init__(self, x, y, color=None, **kwargs):
        self.__dict__.update(kwargs)
        self.x = x
        self.y = y
        self.color = color


class Polygon:
    def __init__(self, points: List[Point]):
        assert len(points) >= 3
        self.points = points

    def as_array(self) -> np.ndarray:
        return np.array(self.points + [self.points[0]])


class LowPolyfier:
    def __init__(self, **kwargs):
        self.options = kwargs

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        processed_image = cv2.bilateralFilter(image, 9, 20, 100)
        return processed_image

    def get_keypoints(self, image: np.ndarray) -> List[Point]:
        return []

    def get_polygons(self, keypoints: List[Point], image: np.ndarray) -> List[Polygon]:
        return []

    def shade(self, polygons: List[Polygon], image: np.ndarray) -> np.ndarray:
        return image

    def postprocess(self, image: np.ndarray) -> np.ndarray:
        return image

    def lowpolyfy(self, image: Image) -> Image:
        image = np.array(image)[:, :, ::-1].copy()
        pre_processed_image = self.preprocess(image)
        keypoints = self.get_keypoints(pre_processed_image)
        polygons = self.get_polygons(keypoints, pre_processed_image)
        shaded_image = self.shade(polygons, pre_processed_image)
        post_processed_image = self.postprocess(shaded_image)
        return Image.fromarray(post_processed_image)
