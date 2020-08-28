import os
from uuid import uuid4
import datetime
from itertools import product


def get_default_options(options: dict):
    kwargs = {}
    for name, value in options.items():
        if isinstance(value, (list, tuple)):
            value = value[0]
        kwargs[name] = value
    return kwargs


def iter_options(options: dict):
    kwargs = options.copy()
    for name, value in kwargs.items():
        if not isinstance(value, (list, tuple)):
            kwargs[name] = [value]
    keys = kwargs.keys()
    values = kwargs.values()
    for instance in product(*values):
        yield dict(zip(keys, instance))


OPTIONS = {
    "longest_edge": [800],
    "output_size": [2500],
    "polygon_method": ["delaunay", "voronoi"],
    "num_canny_points": [2500],
    "num_laplace_points": [0, 250, 500],
    "num_random_points": [0, 100],
    "canny_low_threshold": [150, 100],
    "canny_high_threshold": [200, 150],
    "random_replace_ratio": [0.01],
    "jiggle_ratio": [0.003],
    "gradient": ["none", "random"],
    "post_saturation": [0.0],
    "post_brightness": [0.0],
    "post_contrast": [0.0],
    "visualize_canny": False,
    "visualize_laplace": False,
    "visualize_points": False,
}


def get_output_name(input_name, suffix="lowpoly", output_dir=None):
    filename, extension = os.path.splitext(input_name)
    dirname, filename = os.path.split(filename)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        dirname = output_dir
    filename = os.path.join(dirname, filename)
    now = (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-" + uuid4().hex[-10:]
    )
    output_file_name = "{}_{}_{}{}".format(filename, suffix, now, extension)
    return os.path.abspath(output_file_name)


def get_experiment_dir_name(input_name):
    dirname, filename = os.path.split(input_name)
    filename, extension = os.path.splitext(filename)
    subdir = os.path.join(dirname, filename + "_lowpoly")
    return subdir
