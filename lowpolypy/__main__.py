import argparse
from .helpers import OPTIONS, get_default_options
from .run import experiment, run
from loguru import logger
import hydra
from omegaconf import DictConfig

# parser = argparse.ArgumentParser()
# subparsers = parser.add_subparsers(dest='mode')
#
# # single output parser
# single_parser = subparsers.add_parser('run')
# single_parser.add_argument('image_paths', nargs='+')
# single_parser.add_argument('--output_dir', required=False)
# for name, default in get_default_options(OPTIONS).items():
#     single_parser.add_argument('--{}'.format(name), required=False, type=type(default), default=default)
# single_parser.set_defaults(func=run)
#
# # experiment parser
# experiment_parser = subparsers.add_parser('experiment')
# experiment_parser.add_argument('image_path')
# experiment_parser.set_defaults(func=experiment)
# experiment_parser.add_argument('--output_dir', required=False)
#
# arguments = parser.parse_args()


@hydra.main(config_name="config")
@logger.catch()
def main(config):
    logger.info(config)
    run(config)


if __name__ == "__main__":
    main()
