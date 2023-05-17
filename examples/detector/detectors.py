import torch
import logging
from plato.config import Config
from scipy.stats import norm


def get():

    detector_type = (
        Config().server.detector_type
        if hasattr(Config().server, "detector_type")
        else None
    )

    if detector_type is None:
        logging.info("No defence is applied.")
        return lambda x: x

    if detector_type in registered_detectors:
        registered_defence = registered_detectors[detector_type]
        return registered_defence

    raise ValueError(f"No such defence: {detector_type}")


def fl_detector(weights_attacked):
    """https://arxiv.org/pdf/2207.09209.pdf"""

    return clean_weights


registered_detectors = {
    "FLDetector": fl_detector,
}
