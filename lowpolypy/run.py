import os
import cv2
import json
from pathlib import Path
from PIL import Image
from typing import Dict
from loguru import logger
from collections import ChainMap
from copy import deepcopy

import joblib
import contextlib
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from .helpers import OPTIONS, iter_options, get_experiment_dir_name
from .layers import Pipeline
from .utils import registry


def parse_or_infer_destination(destination, source=None, subdir="lowpoly"):
    if destination:
        destination = Path(destination).expanduser().resolve()
    elif source:
        destination = (source if source.is_dir() else source.parent) / subdir
    else:
        raise ValueError("Destination and source can't both be null.")
    return destination


def parse_image_source(source, recursive=False, extensions=None):
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        input_files = [source]
    elif source.is_dir():
        input_files = list(source.rglob("*") if recursive else source.glob("*"))
        if extensions:
            input_files = [f for f in input_files if f.suffix in extensions]
    return input_files


def single(config):
    image_extensions = set(config.files.image_extensions)
    recursive_search = config.files.recursive
    run_index = config.run.index

    # Parse image source (file or directory)
    source = Path(config.files.source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    input_files = parse_image_source(
        source, recursive=recursive_search, extensions=image_extensions
    )
    logger.debug(f"Found images: {input_files}")

    # Parse or infer
    destination = parse_or_infer_destination(config.files.destination, source=source)
    destination.mkdir(exist_ok=True, parents=True)
    logger.info(f"Destination directory: {destination}")

    pipeline = Pipeline(
        [
            registry.get("LowPolyStage", stage_name)(**stage_options)
            for stage_name, stage_options in config.pipeline.items()
        ]
    )

    results = {}
    for file in input_files:
        logger.info(f"Processing {file}...")
        output_filename = (
            f"{file.stem}_{run_index}" if run_index is not None else file.stem
        )
        output_path = (destination / output_filename).with_suffix(".png")
        logger.info(f"Destination file: {output_path}")

        image = cv2.imread(str(file))
        data = {"image": image, "pipeline": pipeline}
        result = pipeline(data)
        cv2.imwrite(
            str(output_path),
            result["image"],
            [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )
        results[file] = result
    return results


def repeat(config):
    """
    Create multiple lowpoly images for each source image.
    Uses the same config for every run.
    """
    iterations = config.run.iterations
    logger.info(f"Repeating lowpolyfication {iterations} times...")
    image_extensions = set(config.files.image_extensions)
    recursive_search = config.files.recursive
    source = Path(config.files.source).expanduser().resolve()
    input_files = parse_image_source(
        source, recursive=recursive_search, extensions=image_extensions
    )

    for input_file in input_files:
        for i in tqdm(range(iterations)):
            single_config = deepcopy(config)
            single_config.run.index = i + 1
            single_config.files.source = str(input_file)
            single(single_config)


def run(config) -> Dict[str, dict]:
    """
    Lowpolyfy one or more images using default or user-provided config.

    Returns (Dict[str, dict]): A mapping from file paths to the config used to generate them.
    """
    if config.run.mode == "single":
        results = single(config)
    elif config.run.mode == "repeat":
        results = repeat(config)
    else:
        raise ValueError(f"Unknown mode '{config.mode}'")
    return results


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()


def experiment(options: dict) -> Dict[str, dict]:
    """
    Lowpolyfy a single image multiple times using a different options each time.

    Returns (Dict[str, dict]): A mapping from file paths to the options used to generate them.
    """
    image_path = options["image_path"]
    output_dir = options.get("output_dir", None) or get_experiment_dir_name(image_path)
    os.makedirs(output_dir, exist_ok=True)

    def run_once(opts):
        opts["image_paths"] = [image_path]
        opts["output_dir"] = output_dir
        filename_to_options = run(opts)  # Also saves image to disk
        return filename_to_options

    options_list = list(iter_options(OPTIONS))
    with tqdm_joblib(tqdm(total=len(options_list))) as pbar:
        results = Parallel(n_jobs=6)(delayed(run_once)(opts) for opts in options_list)
    with open(os.path.join(output_dir, "map.json"), "w") as f:
        configs = dict(ChainMap(*results))
        json.dump(configs, f, indent=2)
    return configs
