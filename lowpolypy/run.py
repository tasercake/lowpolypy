import cv2
from pathlib import Path
from loguru import logger
import numpy as np

from lowpolypy.image_utils import resize_image
from lowpolypy.shaders import shade_kmeans

from .point_generators import (
    conv_points,
    remove_duplicate_points,
    with_boundary_points,
)
from .polygon_generators import generate_delaunay_polygons, rescale_polygons


def run(
    *,
    source: Path,
    destination: Path | None = None,
    conv_points_num_points: int,
    conv_points_num_filler_points: int,
    weight_filler_points: bool,
    output_size: int,
) -> np.ndarray:
    # Validate & expand paths
    source = validate_image_source(source)
    destination_dir = parse_or_infer_destination(destination, source=source)
    destination_dir.mkdir(exist_ok=True, parents=True)
    output_filename = source.stem
    output_path = (destination_dir / output_filename).with_suffix(".png")

    # Load image
    logger.info(f"Processing {source}...")
    image = cv2.imread(str(source))

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

    # Save the output
    logger.info(f"Writing output to: {output_path}")
    cv2.imwrite(
        str(output_path),
        shaded_image,
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )
    logger.info("Done.")
    return shaded_image


def parse_or_infer_destination(
    destination: Path | None, source: Path | None = None, subdir="lowpoly"
):
    if destination:
        destination = destination.expanduser().resolve()
    elif source:
        destination = (source if source.is_dir() else source.parent) / subdir
    else:
        raise ValueError("Destination and source can't both be null.")
    return destination


def validate_image_source(source: Path) -> Path:
    source = source.expanduser().resolve()
    if not source.exists():
        raise ValueError(f"Invalid source path: {source}")
    return source
