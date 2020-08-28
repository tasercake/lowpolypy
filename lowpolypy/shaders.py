import cv2
import numpy as np
import skimage.draw
from PIL import Image
from abc import ABCMeta, abstractmethod

from .utils import registry


class Shader(metaclass=ABCMeta):
    """
    Takes in a PIL image and (optionally) a list of points and (optionally) a list of polygons and returns a shaded image.
    Output should be independent of the order of points or polygons.
    """

    @abstractmethod
    def forward(self, image=None, points=None, polygons=None, *args, **kwargs):
        pass

    def __call__(self, image=None, points=None, polygons=None, *args, **kwargs):
        output = self.forward(image, points, polygons, *args, **kwargs)
        output.setdefault("points", points)
        output.setdefault("polygons", polygons)
        return output


@registry.register("Shader", "MeanShader")
class MeanShader(Shader):
    def __init__(self):
        super().__init__()

    def forward(self, image=None, points=None, polygons=None):
        image = np.array(image)
        shaded = np.array(image)
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
        return {"image": Image.fromarray(shaded)}


@registry.register("Shader", "KmeansShader")
class KmeansShader(Shader):
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

    def forward(self, image=None, points=None, polygons=None):
        image = np.array(image)
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
        return {"image": Image.fromarray(shaded)}
