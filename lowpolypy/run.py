import os
import cv2
import json
from pathlib import Path
from PIL import Image
from typing import Dict
from loguru import logger
from collections import ChainMap

import joblib
import contextlib
from tqdm import tqdm
from joblib import Parallel, delayed

from .helpers import OPTIONS, iter_options, get_experiment_dir_name
from .layers import Pipeline
from .utils import registry


def run(config) -> Dict[str, dict]:
    """
    Lowpolyfy one or more images using default or user-provided config.

    Returns (Dict[str, dict]): A mapping from file paths to the config used to generate them.
    """
    source = Path(config.files.source)
    if not source.is_file():
        raise ValueError(f"source must be an image file. Got '{source}'")
    image = cv2.imread(source)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    destination = Path(config.files.destination or source.parent / "lowpoly")
    if destination.is_dir():
        destination.mkdir(exist_ok=True, parents=True)
        destination = destination / source.name
    else:
        destination.parent.mkdir(exist_ok=True, parents=True)

    pipeline = Pipeline([
        registry.get("LowPolyStage", stage_name)(**stage_options)
        for stage_name, stage_options in config.pipeline.items()
    ])
    data = pipeline({"image": image, "stages": pipeline.stages})

    shaded = data["image"]
    shaded.save(destination, quality=95)
    return data

    # lowpolyfier = LowPolyfier(**config)
    # for image_path in image_paths:
    #     output_path = get_output_name(image_path, output_dir=destination)
    #     output_files[output_path] = config
    #     image = Image.open(image_path)
    #     low_poly_image = lowpolyfier.lowpolyfy(image)
    #     low_poly_image.save(output_path, quality=100)
    # return output_files


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
