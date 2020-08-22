import os
import json
from PIL import Image
from typing import Dict
from collections import ChainMap

import joblib
import contextlib
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib.parallel import BatchCompletionCallBack

from .process import LowPolyfier
from .helpers import get_output_name, OPTIONS, iter_options, get_experiment_dir_name


def run(options: dict) -> Dict[str, dict]:
    """
    Lowpolyfy one or more images using default or user-provided options.

    Returns (Dict[str, dict]): A mapping from file paths to the options used to generate them.
    """
    image_paths = options['image_paths']
    output_dir = options.get('output_dir', None)
    output_files = {}
    lowpolyfier = LowPolyfier(**options)
    for image_path in image_paths:
        output_path = get_output_name(image_path, output_dir=output_dir)
        output_files[output_path] = options
        image = Image.open(image_path)
        low_poly_image = lowpolyfier.lowpolyfy(image)
        low_poly_image.save(output_path, quality=100)
    return output_files


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
    image_path = options['image_path']
    output_dir = options.get('output_dir', None) or get_experiment_dir_name(image_path)
    os.makedirs(output_dir, exist_ok=True)

    def run_once(opts):
        opts["image_paths"] = [image_path]
        opts["output_dir"] = output_dir
        filename_to_options = run(opts)  # Also saves image to disk
        return filename_to_options

    options_list = list(iter_options(OPTIONS))
    with tqdm_joblib(tqdm(total=len(options_list))) as pbar:
        results = Parallel(n_jobs=6)(delayed(run_once)(opts) for opts in options_list)
    with open(os.path.join(output_dir, 'map.json'), 'w') as f:
        configs = dict(ChainMap(*results))
        json.dump(configs, f, indent=2)
    return configs
