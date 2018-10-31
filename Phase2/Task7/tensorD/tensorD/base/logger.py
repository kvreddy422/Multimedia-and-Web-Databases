# Created by ay27 at 17/3/4
import logging
from logging.config import fileConfig
from os import path

DEFAULT_TYPE = 'DEBUG'

config_file = path.join(path.dirname(path.abspath(__file__)), '../conf/logging_config.ini')
fileConfig(config_file)


def create_logger(level=DEFAULT_TYPE):
    """

    Parameters
    ----------
    level: str
        DEBUG or RELEASE or None

    Returns
    -------
    logger

    """
    tmp =logging.getLogger(level)
    tmp.setLevel(logging.ERROR)
    return tmp
