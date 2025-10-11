"""
Implement new algorithm: personalized federarted NAS.

Reference Search Space: NASVIT: https://openreview.net/pdf?id=Qaw16njk6L.

The code: https://github.com/facebookresearch/NASViT.
"""

import fednas_algorithm
import fednas_client
import fednas_server
import fednas_trainer
from nasvit_wrapper.architect import Architect
from nasvit_wrapper.attentive_nas_dynamic_model import (
    AttentiveNasDynamicModel,
)


def main():
    """
    A Plato federated learning training session using PerFedRLNAS, paper unpublished.
    """
    supernet = AttentiveNasDynamicModel
    client = fednas_client.Client(
        model=supernet,
        algorithm=fednas_algorithm.ClientAlgorithm,
        trainer=fednas_trainer.Trainer,
    )
    server = fednas_server.Server(
        model=Architect,
        algorithm=fednas_algorithm.ServerAlgorithm,
        trainer=fednas_trainer.Trainer,
    )

    server.run(client)


if __name__ == "__main__":
    main()
