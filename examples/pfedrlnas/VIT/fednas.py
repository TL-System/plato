"""
Implement new algorithm: personalized federarted NAS.

Reference Search Space: NASVIT: NEURAL ARCHITECTURE SEARCH FOR EFFICIENT VISION TRANSFORMERS WITH GRADIENT CONFLICT-AWARE SUPERNET TRAINING.

The code: https://github.com/facebookresearch/NASViT.
"""

import fednas_server
import fednas_client
import fednas_algorithm

from NASVIT.models.attentive_nas_dynamic_model import AttentiveNasDynamicModel
from NASVIT.architect import Architect
import fednas_trainer


def main():
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
