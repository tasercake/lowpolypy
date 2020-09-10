from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf

from .run import run
from .utils import load_config


@logger.catch()
def main():
    config = load_config()
    logger.info(config)
    run(config)


if __name__ == "__main__":
    main()
