import logging
import numpy as np

from lowpolypy.image_utils import resize_image
from lowpolypy.shaders import shade_kmeans

from .point_generators import (
    conv_points,
    remove_duplicate_points,
    with_boundary_points,
)
from .polygon_generators import generate_delaunay_polygons, rescale_polygons

logger = logging.getLogger(__name__)


def run(
    *,
    image: np.ndarray,
    conv_points_num_points: int,
    conv_points_num_filler_points: int,
    weight_filler_points: bool,
    output_size: int,
) -> np.ndarray:
    # Run the pipeline
    logger.info("Computing points...")
    points = conv_points(
        image=image,
        num_points=conv_points_num_points,
        num_filler_points=conv_points_num_filler_points,
        weight_filler_points=weight_filler_points,
    )
    points = with_boundary_points(points)
    points = remove_duplicate_points(points, 3e-4)

    logger.info("Generating polygon mesh from points...")
    polygons = generate_delaunay_polygons(points=points)

    logger.info("Shading polygons...")
    # Resize the image only before shading. This saves memory & compute when generating points and polygons.
    resized_image = resize_image(image=image, size=output_size)
    resized_polygons = rescale_polygons(
        polygons=polygons, size=(resized_image.shape[1] - 1, resized_image.shape[0] - 1)
    )
    shaded_image = shade_kmeans(image=resized_image, polygons=resized_polygons)

    return shaded_image
