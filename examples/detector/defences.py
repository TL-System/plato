import torch
import logging
from plato.config import Config
from scipy.stats import norm

registered_defences = {}


def get():

    defence_type = (
        Config().server.defence_type
        if hasattr(Config().server, "defence_type")
        else None
    )

    if defence_type is None:
        logging.info("No defence is applied.")
        return lambda x: x

    if defence_type in registered_defences:
        registered_defence = registered_defences[defence_type]
        return registered_defence

    raise ValueError(f"No such defence: {defence_type}")
