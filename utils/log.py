"""
Created on May 10, 2023

@author: mning
"""
import logging

from EEGFeaturesExtraction.utils import logFilter


def setup_custom_logger(name: str, loglevel: int = logging.INFO) -> logging:
    formatter = logging.Formatter(fmt="\n%(levelname)s - %(module)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(logFilter.logFilter(loglevel))

    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    logger.addHandler(handler)
    return logger
