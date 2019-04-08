import os
import json
import multiprocessing
from collections import ChainMap
from typing import Dict
from PIL import Image
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


def experiment(options: dict) -> Dict[str, dict]:
    """
    Lowpolyfy a single image multiple times using a different options each time.

    Returns (Dict[str, dict]): A mapping from file paths to the options used to generate them.
    """
    image_path = options['image_path']
    output_dir = options.get('output_dir', None) or get_experiment_dir_name(image_path)
    os.makedirs(output_dir, exist_ok=True)
    results = []
    pool = multiprocessing.Pool()
    for opts in iter_options(OPTIONS):
        opts['image_paths'] = [image_path]
        opts['output_dir'] = output_dir
        result = pool.apply_async(run, (opts,))
        results.append(result)
    pool.close()
    pool.join()
    with open(os.path.join(output_dir, 'map.json'), 'w') as configs:
        output_files = dict(ChainMap(*[result.get() for result in results]))
        json.dump(output_files, configs, indent=2)
    return output_files
