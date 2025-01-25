import cv2
import numpy as np
import shapely
import skimage


def shade_mean(*, image: np.ndarray, polygons: list[shapely.Polygon]) -> np.ndarray:
    shaded = np.zeros_like(image)
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
    return shaded


def _get_dominant_color(
    pixels: np.ndarray, num_clusters: int, attempts: int
) -> np.ndarray:
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


def shade_kmeans(
    *,
    image: np.ndarray,
    polygons: list[shapely.Polygon],
    num_clusters: int = 3,
    num_attempts: int = 3,
) -> np.ndarray:
    shaded = np.zeros_like(image)
    for polygon in polygons:
        coords = np.array(polygon.exterior.coords)
        rr, cc = skimage.draw.polygon(*coords.T[::-1])
        region = image[rr, cc]
        if len(region) == 0:
            continue
        mean_color = _get_dominant_color(region, num_clusters, num_attempts)
        cv2.fillPoly(
            shaded,
            [(coords * 256).astype(int)],
            mean_color.tolist(),
            lineType=cv2.LINE_AA,
            shift=8,
        )
    return shaded
