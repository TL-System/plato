"""
The implementation for our contrastive adaptation method.

The official code: https://github.com/google-research/simclr

The third-party code: https://github.com/PatrickHua/SimSiam

The structure of our SimCLR and the classifier is the same as the ones used in
the work https://github.com/spijkervet/SimCLR.git.

Reference:

[1]. https://arxiv.org/abs/2002.05709

"""

import contras_adap_net

from plato.trainers import contrastive_ssl as ssl_trainer
from plato.clients import ssl_simple as ssl_client
from plato.servers import fedavg


def main():
    """ A Plato federated learning training session using the SimCLR algorithm.
        This implementation of simclr utilizes the general setting, i.e.,
        removing the final fully-connected layers of model defined by
        the 'model_name' in config file.
    """
    trainer = ssl_trainer.Trainer
    contras_adap_model = contras_adap_net.ContrasAdapNet()
    client = ssl_client.Client(model=contras_adap_model, trainer=trainer)
    server = fedavg.Server(model=contras_adap_model, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
