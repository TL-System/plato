"""
The implementation for the MoCoV2 [2] method, which is the enhanced version of MoCoV1 [1],
for personalized federated learning.

Reference:
[1]. Kaiming He, et al., Momentum Contrast for Unsupervised Visual Representation Learning, 
CVPR 2020. https://arxiv.org/abs/1911.05722.

[2]. Xinlei Chen, et al., Improved Baselines with Momentum Contrastive Learning, ArXiv, 2020.
https://arxiv.org/abs/2003.04297.

The official code: https://github.com/facebookresearch/moco.

"""

from plato.servers import fedavg_personalized as personalized_server

from self_supervised_learning import ssl_client
from self_supervised_learning import ssl_datasources

import mocov2_model
import mocov2_trainer


def main():
    """
    A personalized federated learning session for BYOL approach.
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
