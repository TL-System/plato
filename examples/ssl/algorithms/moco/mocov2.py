"""
An implementation of the MoCoV2 algorithm.

Kaiming He, et al., Momentum Contrast for Unsupervised Visual Representation Learning, 
CVPR 2020. https://arxiv.org/abs/1911.05722.

Xinlei Chen, et al., Improved Baselines with Momentum Contrastive Learning, ArXiv, 2020.
https://arxiv.org/abs/2003.04297.

Source code: https://github.com/facebookresearch/moco.

"""
from self_supervised_learning import ssl_client
from self_supervised_learning import ssl_datasources

import mocov2_model
import mocov2_trainer

from plato.servers import fedavg_personalized as personalized_server


def main():
    """
    A personalized federated learning session with MoCoV2.
    """
    client = ssl_client.Client(
        model=mocov2_model.MoCoV2,
        datasource=ssl_datasources.SSLDataSource,
        trainer=mocov2_trainer.Trainer,
    )
    server = personalized_server.Server(
        model=mocov2_model.MoCoV2,
        trainer=mocov2_trainer.Trainer,
    )

    server.run(client)


if __name__ == "__main__":
    main()
