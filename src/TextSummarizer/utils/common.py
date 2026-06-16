import os
from box.exceptions import BoxValueError
import yaml
from TextSummarizer.custom_logging import logger
from box import ConfigBox
from pathlib import Path
from typing import Union


def read_yaml(path_to_yaml: Union[str, Path]) -> ConfigBox:
    """Read a YAML file and return a ConfigBox.

    Args:
        path_to_yaml (str | Path): Path to the yaml file.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        Exception: For any other read/parse errors.
    """
    path = Path(path_to_yaml)
    try:
        logger.info(f">>> Trying to open: {path}")
        if not path.exists():
            logger.error(f"YAML file not found: {path}")
            raise FileNotFoundError(f"YAML file not found: {path}")

        with path.open("r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path} loaded successfully")
            return ConfigBox(content)

    except BoxValueError as e:
        logger.error(f"Error while converting YAML to ConfigBox: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading YAML file: {e}")
        raise


def create_directories(path_to_directories):
    """Creates directories if they do not exist"""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Directory created at: {path}")


def get_size(path) -> str:
    """Returns file size in KB."""
    try:
        size_in_kb = round(path.stat().st_size / 1024, 2)
        logger.info(f"File size: {size_in_kb} KB")
        return f"{size_in_kb} KB"
    except Exception as e:
        logger.error(f"Error getting file size: {e}")
        raise
