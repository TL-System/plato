"""
Membership Inference Attack (MIA) utilities for federated unlearning.

References:

Liu et al., "FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models,"
in IWQoS 2021.

Shokri et al., "Membership Inference Attacks Against Machine Learning Models," in IEEE S&P 2017.

https://ieeexplore.ieee.org/document/9521274
https://arxiv.org/pdf/1610.05820.pdf
"""

from .mia import launch_attack, train_attack_model
from .mia_client import Client
from .mia_server import Server

__all__ = ["launch_attack", "train_attack_model", "Client", "Server"]
