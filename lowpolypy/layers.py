import cv2
import numpy as np
import skimage.draw
from abc import ABCMeta, abstractmethod
from scipy.spatial import cKDTree
from scipy import signal
from loguru import logger

import shapely
from shapely.geometry import box, Point, MultiPoint, asMultiPoint
from shapely.ops import triangulate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import albumentations as A
from torchvision import transforms
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from .utils import registry


class Layer(nn.Module):
    @abstractmethod
    def forward(self, data, *args, **kwargs):
        pass

    def __call__(self, data, *args, **kwargs):
        data = dict(data)
        data.update(super().__call__(data, *args, **kwargs))
        return data

    @staticmethod
    def rescale_points(points, image_size):
        """
        Rescales [0, 1] normalized points to the specified Width x Height.
        """
        coordinates = (
            np.array([p.xy for p in points]).squeeze(-1) * image_size
        ).round()
        return list(asMultiPoint(coordinates))

    @staticmethod
    def with_boundary_points(points):
        edge_box = box(0, 0, 1, 1)
        edges = edge_box.exterior

        edge_points = []
        hull = shapely.ops.unary_union(points).convex_hull
        hull_points = [Point(c) for c in hull.exterior.coords]
        edge_points = [edges.interpolate(edges.project(p)) for p in hull_points]

        corners = [Point(c) for c in edges.coords[:-1]]

        return points + edge_points + corners

    @staticmethod
    def remove_duplicate_points(points, tolerance):
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

    @staticmethod
    def simplify(polygons):
        # TODO: Merge small polygons
        return polygons


class Pipeline(nn.Module):
    def __init__(self, stages):
        super().__init__()
        self.stages = nn.ModuleList(stages)

    def forward(self, data):
        for stage in self.stages:
            data = stage(data)
        return data


@registry.register("LowPolyStage", "ResizeImage")
class ResizeImage(Layer):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, data):
        image = data["image"]
        image_dims = image.shape[1::-1]
        aspect_ratio = image_dims[0] / image_dims[1]
        new_size = (int(round(self.size * aspect_ratio)), self.size)
        logger.debug(f"Resized dimensions: {new_size}")
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
        return {"image": resized}


@registry.register("LowPolyStage", "CNNPoints")
class CNNPoints(Layer):
    def __init__(
        self,
        num_points=100,
    ):
        super().__init__()
        self.num_points = num_points
        self.model = smp.DeepLabV3("efficientnet-b0", in_channels=3)

    def forward(self, data):
        image = data["image"]
        point_matrix = self.model(image).cpu().detach().numpy()
        coordinates = np.argwhere(point_matrix >= 0.5)
        points = [Point(c) for c in coordinates]
        points = self.remove_duplicate_points(points, 1e-4)
        points = self.with_boundary_points(points)
        return {"points": points, "point_matrix": point_matrix}


@registry.register("LowPolyStage", "RandomPoints")
class RandomPoints(Layer):
    def __init__(self, num_points=100):
        super().__init__()
        self.num_points = num_points

    def forward(self, data):
        coordinates = np.random.rand(self.num_points, 2)
        points = [Point(c) for c in coordinates]
        points = self.remove_duplicate_points(points, 1e-4)
        points = self.with_boundary_points(points)
        return {"points": points}


@registry.register("LowPolyStage", "ConvPoints")
class ConvPoints(Layer):
    def __init__(
        self, num_points=1000, num_filler_points=50, weight_filler_points=True
    ):
        super().__init__()
        self.num_points = num_points
        self.num_filler_points = num_filler_points
        self.weight_filler_points = weight_filler_points

    def forward(self, data):
        points = []
        image = data["image"]
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
        choices = np.random.choice(
            coordinates, size=self.num_points, replace=False, p=weights
        )
        raw_points = np.unravel_index(choices, image.shape[:2])
        conv_points = np.stack(raw_points, axis=-1) / image.shape[:2]
        points += list(MultiPoint(conv_points[..., ::-1]))

        if self.num_filler_points:
            inverse = 255 - cv2.dilate(mag, np.ones((5, 5), np.uint8), iterations=3)
            inverse = cv2.blur(inverse, ksize=(13, 13))
            weights = np.ravel(inverse.astype(np.float32) / inverse.sum())
            coordinates = np.arange(0, weights.size, dtype=np.uint32)
            choices = np.random.choice(
                coordinates,
                size=self.num_filler_points,
                replace=False,
                p=weights if self.weight_filler_points else None,
            )
            raw_points = np.unravel_index(choices, image.shape[:2])
            filler_points = np.stack(raw_points, axis=-1) / image.shape[:2]
            points += list(MultiPoint(filler_points[..., ::-1]))

        points = self.remove_duplicate_points(points, 3e-4)
        points = self.with_boundary_points(points)

        return {"points": points}


@registry.register("LowPolyStage", "DelaunayTriangulator")
class DelaunayTriangulator(Layer):
    def forward(self, data):
        points = self.rescale_points(data["points"], data["image"].shape[1::-1])
        if not isinstance(points, MultiPoint):
            points = MultiPoint(points)
        triangles = triangulate(points)
        return {"polygons": triangles}


@registry.register("LowPolyStage", "MeanShader")
class MeanShader(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        image, polygons = data["image"], data["polygons"]
        shaded = image
        mask = np.zeros((*image.shape[:2], 1), dtype=np.uint8)
        for polygon in polygons:
            coords = np.array(polygon.exterior.coords)
            mask[:] = 0
            cv2.fillPoly(mask, [(coords * 256).astype(np.int32)], 255, shift=8)
            mean_color = np.array(cv2.mean(image, mask=mask)[:3], dtype=np.uint8)
            cv2.fillPoly(
                shaded,
                [(coords * 256).astype(int)],
                mean_color.tolist(),
                lineType=cv2.LINE_AA,
                shift=8,
            )
        return {"image": shaded}


@registry.register("LowPolyStage", "KmeansShader")
class KmeansShader(Layer):
    def __init__(self, num_clusters=3, num_attempts=3):
        super().__init__()
        self.num_clusters = num_clusters
        self.num_attempts = num_attempts

    @staticmethod
    def get_dominant_color(pixels, num_clusters, attempts):
        """
        Given a (N, Channels) array of pixel values, compute the dominant color via K-means
        """
        if len(pixels) == 1:
            return pixels[0]
        num_clusters = min(num_clusters, len(pixels))
        flags = cv2.KMEANS_RANDOM_CENTERS
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10)
        _, labels, centroids = cv2.kmeans(
            pixels.astype(np.float32), num_clusters, None, criteria, attempts, flags
        )
        _, counts = np.unique(labels, return_counts=True)
        dominant = centroids[np.argmax(counts)]
        return dominant

    def forward(self, data):
        image, polygons = data["image"], data["polygons"]
        shaded = np.zeros_like(image)
        for polygon in polygons:
            coords = np.array(polygon.exterior.coords)
            rr, cc = skimage.draw.polygon(*coords.T[::-1])
            region = image[rr, cc]
            if len(region) == 0:
                continue
            mean_color = self.get_dominant_color(
                region, self.num_clusters, self.num_attempts
            )
            cv2.fillPoly(
                shaded,
                [(coords * 256).astype(int)],
                mean_color.tolist(),
                lineType=cv2.LINE_AA,
                shift=8,
            )
        return {"image": shaded}
