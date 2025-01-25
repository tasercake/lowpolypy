"""
Defines the CLI commands for the app.

To run, invoke it using the script command defined in `pyproject.toml`.
"""

import logging
from pathlib import Path

import cv2
import typer

from .run import run

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = typer.Typer(name="lowpolypy", help="LowPolyPy CLI")


@app.command()
def main(
    source: Path,
    destination: Path | None = None,
    conv_points_num_points: int = 1000,
    conv_points_num_filler_points: int = 300,
    weight_filler_points: bool = True,
    output_size: int = 2560,
):
    # Validate & expand paths
    source = _validate_image_source(source)

    destination_dir = _parse_or_infer_destination(destination, source=source)
    destination_dir.mkdir(exist_ok=True, parents=True)
    output_filename = source.stem
    output_path = (destination_dir / output_filename).with_suffix(".png")

    # Load image
    logger.info(f"Processing {source}...")
    image = cv2.imread(str(source))

    lowpoly_image = run(
        image=image,
        conv_points_num_points=conv_points_num_points,
        conv_points_num_filler_points=conv_points_num_filler_points,
        weight_filler_points=weight_filler_points,
        output_size=output_size,
    )

    # Save the output
    logger.info(f"Writing output to: {output_path}")
    cv2.imwrite(
        str(output_path),
        lowpoly_image,
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )
    logger.info("Done.")


def _validate_image_source(source: Path) -> Path:
    source = source.expanduser().resolve()
    if not source.exists():
        raise ValueError(f"Invalid source path: {source}")
    return source


def _parse_or_infer_destination(
    destination: Path | None, source: Path | None = None, subdir="lowpoly"
):
    if destination:
        destination = destination.expanduser().resolve()
    elif source:
        destination = (source if source.is_dir() else source.parent) / subdir
    else:
        raise ValueError("Destination and source can't both be null.")
    return destination
