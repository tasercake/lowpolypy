"""
Defines the CLI commands for the app.

To run, invoke it using the script command defined in `pyproject.toml`.
"""

import logging
from pathlib import Path

import typer

from .run import run

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
    run(
        source=source,
        destination=destination,
        conv_points_num_points=conv_points_num_points,
        conv_points_num_filler_points=conv_points_num_filler_points,
        weight_filler_points=weight_filler_points,
        output_size=output_size,
    )
