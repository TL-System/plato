#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base class for sampling modality so that a multi-modal data can be divided across the clients.
"""
import os
from abc import abstractmethod

from plato.config import Config


class Sampler:
    """Base class for modalities samplers so that the multi-modal dataset is divided into
    partitions across the clients."""
    def __init__(self):
        if hasattr(Config().data, 'random_seed'):
            # Keeping random seed the same across the clients
            # so that the experiments are reproducible
            self.random_seed = Config().data.random_seed
        else:
            # The random seed will be different across different
            # runs if it is not provided.
            self.random_seed = os.getpid()

    @abstractmethod
    def get(self):
        """Obtains an instance of the sampler. """

    @abstractmethod
    def modality_size(self):
        """Returns the length of the dataset after sampling. """