"""
Implement new algorithm: personalized federarted NAS

Search Space: https://github.com/facebookresearch/NASViT
"""

import fednas_server
import fednas_client
import fednas_algorithm
from NASVIT.models.attentive_nas_dynamic_model import AttentiveNasDynamicModel
from NASVIT.architect import Architect
import fednas_trainer

# import torch

# torch.multiprocessing.set_sharing_strategy("file_system")


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
