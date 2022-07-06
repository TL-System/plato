"""
The implementation for our contrastive adaptation method.

"""

import pFLCMA_net
import pFLCMA_server
import pFLCMA_client
import pFLCMA_trainer
import pFLCMA_algorithm


def main():
    """ A Plato federated learning training session using the SimCLR algorithm.
        This implementation of simclr utilizes the general setting, i.e.,
        removing the final fully-connected layers of model defined by
        the 'model_name' in config file.
    """
    algorithm = pFLCMA_algorithm.Algorithm
    trainer = pFLCMA_trainer.Trainer
    contras_adap_model = pFLCMA_net.pFLCMANet
    client = pFLCMA_client.Client(model=contras_adap_model,
                                  trainer=trainer,
                                  algorithm=algorithm)
    server = pFLCMA_server.Server(model=contras_adap_model,
                                  trainer=trainer,
                                  algorithm=algorithm)

    server.run(client)


if __name__ == "__main__":
    main()
