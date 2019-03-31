"""
This module does the heavy lifting and is responsible for converting the input image into a low-poly stylized image.
"""
import cv2
from PIL import Image
import numpy as np
from scipy import spatial


# Data flow
# pre-process image (Image) -> Image
# get polygons from image (Image) -> List[Polygon]
# shade polygons (List[Point], Image) -> Image
# post-process image (image) -> Image

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
        # image = cv2.bilateralFilter(image, 9, 200, 200)
        return image

    def check_bounds(self, image, polygons):
        height, width = image.shape[:2]
        for polygon in polygons:
            for point in polygon:
                try:
                    assert 0 <= point[0] < width
                    assert 0 <= point[1] < height
                except AssertionError as e:
                    print("(X, Y) = {}, {}".format(point[0], point[1]))
                    print("Image dims: {}".format(image.shape))
                    raise e

    def get_canny_points(self, image: np.ndarray, low_thresh: int, high_thresh: int, num_points=500) -> np.ndarray:
        canny = cv2.Canny(image, low_thresh, high_thresh)
        overlay = np.add(image, cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR))
        print(overlay.shape)
        print(image.shape)
        cv2.imshow('canny', overlay)
        cv2.waitKey(0)
        canny_points = np.argwhere(canny)[..., ::-1]
        random_sample = np.random.choice(canny_points.shape[0], size=num_points, replace=False)
        canny_points = canny_points[random_sample]
        return canny_points

    def get_laplace_points(self, image: np.ndarray, num_points=500) -> np.ndarray:
        image = cv2.GaussianBlur(image, (15, 15), 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.Laplacian(image, cv2.CV_8U, 19)
        image = cv2.GaussianBlur(image, (15, 15), 0)
        image = (image * (255 / image.max())).astype(np.uint8)
        image = image.astype(np.float32) / image.sum()
        weights = np.ravel(image)
        coordinates = np.arange(0, weights.size, dtype=np.uint32)
        choices = np.random.choice(coordinates, size=num_points, replace=False, p=weights)
        raw_points = np.unravel_index(choices, image.shape)
        points = np.stack(raw_points, axis=-1)
        return points[..., ::-1]

    def get_random_points(self, image: np.ndarray, num_points=100) -> np.ndarray:
        return np.array([np.random.uniform((0, 0), image.shape[:2]).astype(np.uint16)[::-1] for _ in range(num_points)])

    def randomize_points(self, image: np.ndarray, points: np.ndarray, ratio: float) -> None:
        num_random_points = int(len(points) * ratio)
        if num_random_points:
            replace_indices = np.random.uniform(0, len(points), num_random_points).astype(np.uint16)
            points[replace_indices] = self.get_random_points(image, num_points=num_random_points)

    def get_keypoints(self, image: np.ndarray) -> np.ndarray:
        random_points_ratio = self.options.get('random_points_ratio', 0.1)
        num_canny_points = self.options.get('num_canny_points', 500)
        num_laplace_points = self.options.get('num_laplace_points', 500)
        corners = np.array([
            (0, 0),  # top left
            (image.shape[1] - 1, 0),  # top right
            (0, image.shape[0] - 1),  # bottom left
            (image.shape[1] - 1, image.shape[0] - 1)  # bottom right
        ])
        canny_points = self.get_canny_points(image, 0, 100, num_points=num_canny_points)
        laplace_points = self.get_laplace_points(image, num_points=num_laplace_points)
        points = np.concatenate((canny_points, laplace_points))
        self.randomize_points(image, points, random_points_ratio)
        points = np.concatenate((points, corners))
        print("Num keypoints", len(points))
        return points

    def validate_polygons(self, polygons: np.ndarray):
        assert len(polygons.shape) == 3
        assert polygons.shape[1] >= 3
        assert polygons.shape[2] == 2

    def get_polygons(self, image: np.ndarray) -> np.ndarray:
        points = self.get_keypoints(image)
        delaunay = spatial.Delaunay(points)
        polygons = np.array([[points[i] for i in simplex] for simplex in delaunay.simplices])
        self.validate_polygons(polygons)
        self.check_bounds(image, polygons)
        return polygons

    def shade(self, polygons: np.ndarray, image: np.ndarray) -> np.ndarray:
        output_image = np.zeros(image.shape, dtype=np.uint8)
        # TODO: parallelize this
        for polygon in polygons:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, polygon.astype(np.int32), (255,))
            mean = cv2.mean(image, mask)[:3]
            # output_image[mask > 0] = mean
            cv2.fillConvexPoly(output_image, polygon.astype(np.int32), mean, lineType=cv2.LINE_AA)
        return output_image

    def postprocess(self, image: np.ndarray) -> np.ndarray:
        return image

    def lowpolyfy(self, image):
        image = np.array(image)[..., ::-1].copy()
        pre_processed_image = self.preprocess(image)
        polygons = self.get_polygons(pre_processed_image)
        shaded_image = self.shade(polygons, pre_processed_image)
        post_processed_image = self.postprocess(shaded_image)
        return Image.fromarray(post_processed_image[..., ::-1])
