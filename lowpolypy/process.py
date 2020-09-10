"""
This module does the heavy lifting and is responsible for converting the input image into a low-poly stylized image.
"""
import cv2
from PIL import Image
import numpy as np
import time
from scipy import spatial


# Data flow
# pre-process image (Image) -> Image
# get polygons from image (Image) -> List[Polygon]
# shade polygons (List[Point], Image) -> Image
# post-process image (image) -> Image


class LowPolyfier:
    def __init__(self, **kwargs):
        self.options = {k: v for k, v in kwargs.items() if v is not None}

    @staticmethod
    def rescale_image(image: np.ndarray, longest_edge: int):
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
        if image.shape[-1] == 4:
            float_image = image.astype(np.float64) / 255.0
            float_image[..., :3] *= float_image[..., -1, np.newaxis]
            float_image = float_image[..., :3]
            image = (float_image * 255).astype(np.uint8)
        image = image[..., ::-1]
        longest_edge = self.options["longest_edge"]
        if longest_edge != "auto":
            image = self.rescale_image(image, longest_edge)
        # image = cv2.bilateralFilter(image, 9, 200, 200)
        return image

    @staticmethod
    def visualize_image(image: np.ndarray, name=None):
        name = name or str(time.time())
        cv2.imshow(name, image)
        cv2.waitKey(0)

    def visualize_points(self, image: np.ndarray, points: np.ndarray, name=None):
        image = image.copy() // 2
        points = (*np.squeeze(np.split(points, 2, axis=1))[::-1],)
        image[points] = (0, 255, 0)
        self.visualize_image(image, name=name)

    def visualize_canny(self, image: np.ndarray, canny: np.ndarray):
        image = image.copy()
        image[np.where(canny)] = (0, 255, 0)
        self.visualize_image(image, name="canny")

    @staticmethod
    def get_random_points(image: np.ndarray, num_points=100) -> np.ndarray:
        if num_points <= 0:
            return np.zeros((0, 2), dtype=np.uint8)
        return np.array(
            [
                np.random.uniform((0, 0), image.shape[:2]).astype(np.int32)[::-1]
                for _ in range(num_points)
            ]
        )

    def get_canny_points(
        self, image: np.ndarray, low_thresh: int, high_thresh: int, num_points=500
    ) -> np.ndarray:
        if num_points <= 0:
            return np.zeros((0, 2), dtype=np.uint8)
        canny = cv2.Canny(image, low_thresh, high_thresh)
        if self.options["visualize_canny"]:
            self.visualize_canny(image, canny)
        canny_points = np.argwhere(canny)[..., ::-1]
        step_size = len(canny_points) // num_points
        canny_points = canny_points[::step_size]
        return canny_points

    def get_laplace_points(self, image: np.ndarray, num_points=500) -> np.ndarray:
        if num_points <= 0:
            return np.zeros((0, 2), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (15, 15), 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.uint8(np.absolute(cv2.Laplacian(image, cv2.CV_64F, 19)))
        image = cv2.GaussianBlur(image, (15, 15), 0)
        image = (image * (255 / image.max())).astype(np.uint8)
        image = image.astype(np.float32) / image.sum()
        if self.options["visualize_laplace"]:
            self.visualize_image(image, "laplace")
        weights = np.ravel(image)
        coordinates = np.arange(0, weights.size, dtype=np.uint32)
        choices = np.random.choice(
            coordinates, size=num_points, replace=False, p=weights
        )
        raw_points = np.unravel_index(choices, image.shape)
        points = np.stack(raw_points, axis=-1)[..., ::-1]
        return points

    def randomize_points(
        self, image: np.ndarray, points: np.ndarray, ratio: float
    ) -> None:
        num_random_points = int(len(points) * ratio)
        if num_random_points:
            replace_indices = np.random.uniform(
                0, len(points), num_random_points
            ).astype(np.int32)
            points[replace_indices] = self.get_random_points(
                image, num_points=num_random_points
            )

    def jiggle_keypoints(
        self, image: np.ndarray, points: np.ndarray, jiggle_ratio: float = 0.01
    ) -> None:
        ratios = np.random.uniform(-jiggle_ratio, jiggle_ratio, points.shape)
        jiggles = (image.shape[:2] * ratios).astype(np.int32)
        points += jiggles
        self.constrain_points(image.shape[:2], points)

    def get_keypoints(self, image: np.ndarray, include_corners=True) -> np.ndarray:
        canny_low_threshold = self.options["canny_low_threshold"]
        canny_high_threshold = self.options["canny_high_threshold"]
        random_replace_ratio = self.options["random_replace_ratio"]
        num_random_points = self.options["num_random_points"]
        num_canny_points = self.options["num_canny_points"]
        num_laplace_points = self.options["num_laplace_points"]
        jiggle_ratio = self.options["jiggle_ratio"]
        # TODO: add points to image edges
        canny_points = self.get_canny_points(
            image,
            canny_low_threshold,
            canny_high_threshold,
            num_points=num_canny_points,
        )
        laplace_points = self.get_laplace_points(image, num_points=num_laplace_points)
        random_points = self.get_random_points(image, num_random_points)
        points = np.concatenate((canny_points, laplace_points, random_points))
        self.randomize_points(image, points, random_replace_ratio)
        self.jiggle_keypoints(image, points, jiggle_ratio=jiggle_ratio)
        if include_corners:
            corners = np.array(
                [
                    (0, 0),  # top left
                    (image.shape[1] - 1, 0),  # top right
                    (0, image.shape[0] - 1),  # bottom left
                    (image.shape[1] - 1, image.shape[0] - 1),  # bottom right
                ]
            )
            points = np.concatenate((points, corners))
        if self.options["visualize_points"]:
            self.visualize_points(image, points)
        return points

    @staticmethod
    def validate_polygons(max_dims: tuple, polygons):
        height, width = max_dims
        if isinstance(polygons, np.ndarray):
            assert len(polygons.shape) == 3
            assert polygons.shape[1] >= 3
            assert polygons.shape[2] == 2
        elif isinstance(polygons, list):
            raise NotImplementedError(
                "Expressing polygons as a list is currently unsupported. "
                "Convert it to a numpy array instead."
            )
        try:
            assert polygons[..., 0].min() >= -1
            assert polygons[..., 1].min() >= -1
            assert polygons[..., 0].max() < width
            assert polygons[..., 1].max() < height
        except AssertionError as e:
            print(
                "ERROR: Some polygon coordinates are out of image bounds. This is a bug. Please report this."
            )
            print(
                "(Xmin, Ymin, Xmax, Ymax) = ({}, {}, {}, {})".format(
                    polygons[..., 0].min(),
                    polygons[..., 1].min(),
                    polygons[..., 0].max(),
                    polygons[..., 1].max(),
                )
            )
            print("Max dims: {}".format(max_dims))
            raise e

    @staticmethod
    def constrain_points(max_dims: tuple, points: np.ndarray):
        height, width = max_dims
        if isinstance(points, np.ndarray):
            points[..., 0][points[..., 0] < 0] = 0
            points[..., 0][points[..., 0] >= width] = width - 1
            points[..., 1][points[..., 1] < 0] = 0
            points[..., 1][points[..., 1] >= height] = height - 1
        elif isinstance(points, list):
            raise NotImplementedError(
                "Expressing points as a list is currently unsupported."
            )
        return points

    @staticmethod
    def finitize_voronoi(vor, radius=None):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite regions.

        Args:
            vor:
            radius:

        Returns:

        """

        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max()

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge
                t = vor.points[p2] - vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.array(new_vertices, dtype=np.int32)

    def get_delaunay_triangles(self, image: np.ndarray):
        points = self.get_keypoints(image)
        delaunay = spatial.Delaunay(points)
        return np.array(
            [[points[i] for i in simplex] for simplex in delaunay.simplices]
        )

    def get_voronoi_polygons(self, image: np.ndarray):
        points = self.get_keypoints(image, include_corners=False)
        voronoi = spatial.Voronoi(points)
        regions, vertices = self.finitize_voronoi(voronoi)
        self.constrain_points(image.shape[:2], vertices)
        max_polygon_points = len(max(regions, key=lambda x: len(x)))
        polygons = np.full((len(regions), max_polygon_points, 2), -1)
        for i, region in enumerate(regions):
            polygon_points = vertices[region]
            polygons[i][: polygon_points.shape[0]] = polygon_points
        return polygons

    def get_polygons(self, image: np.ndarray) -> np.ndarray:
        polygon_method = self.options["polygon_method"]
        if polygon_method == "delaunay":
            polygons = self.get_delaunay_triangles(image)
        elif polygon_method == "voronoi":
            polygons = self.get_voronoi_polygons(image)
        else:
            raise ValueError(f"Unknown Polygonization method '{polygon_method}'")
        self.validate_polygons(image.shape[:2], polygons)
        return polygons

    def get_output_dimensions(self, image):
        longest_edge = self.options["output_size"]
        ratio = image.shape[1] / image.shape[0]
        return int(longest_edge / ratio), longest_edge, 3

    @staticmethod
    def strip_negative_points(polygon):
        return polygon[polygon.min(axis=1) >= 0]

    @staticmethod
    def get_dominant_color(pixels, clusters, attempts):
        """
        Given a (N, Channels) array of pixel values, compute the dominant color via K-means
        """
        clusters = min(clusters, len(pixels))
        flags = cv2.KMEANS_RANDOM_CENTERS
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10)
        _, labels, centroids = cv2.kmeans(
            pixels.astype(np.float32), clusters, None, criteria, attempts, flags
        )
        _, counts = np.unique(labels, return_counts=True)
        dominant = centroids[np.argmax(counts)]
        return dominant

    def shade(self, polygons: np.ndarray, image: np.ndarray) -> np.ndarray:
        canvas_dimensions = self.get_output_dimensions(image)
        scale_factor = max(canvas_dimensions) / max(image.shape)
        scaled_polygons = polygons * scale_factor
        output_image = np.zeros(canvas_dimensions, dtype=np.uint8)
        for polygon, scaled_polygon in zip(polygons, scaled_polygons):
            polygon = self.strip_negative_points(polygon)
            scaled_polygon = self.strip_negative_points(scaled_polygon)
            if len(polygon) < 3:
                continue
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, polygon, (255,))
            color = self.get_dominant_color(image[mask > 0], 3, 3).tolist()
            # color = cv2.mean(image, mask)[:3]
            cv2.fillConvexPoly(
                output_image,
                scaled_polygon.astype(np.int32),
                color,
                lineType=cv2.LINE_AA,
            )
        return output_image

    @staticmethod
    def saturation(image: np.ndarray, alpha: float, beta: float):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = cv2.convertScaleAbs(hsv[..., 1], alpha=alpha, beta=beta)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return image

    @staticmethod
    def brightness_contrast(image: np.ndarray, alpha, beta):
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return image

    def postprocess(self, image: np.ndarray) -> np.ndarray:
        saturation = self.options["post_saturation"]
        contrast = self.options["post_contrast"]
        brightness = self.options["post_brightness"]
        image = self.saturation(image, 1 + saturation, 0)
        image = self.brightness_contrast(image, 1 + contrast, 255 * brightness)
        return image

    def lowpolyfy(self, image):
        image = np.array(image).copy()
        pre_processed_image = self.preprocess(image)
        polygons = self.get_polygons(pre_processed_image)
        shaded_image = self.shade(polygons, pre_processed_image)
        post_processed_image = self.postprocess(shaded_image)
        return Image.fromarray(post_processed_image[..., ::-1])
