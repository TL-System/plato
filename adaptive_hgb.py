"""
Hierarchical Gradient Blending (HGB)

Reference:

This is source of our work "Towards Optimal Multi-modal Federated Learning on Non-IID Data with Hierarchical Gradient Blending"

https://github.com/iQua/sijia-infocom22
"""
import os

os.environ['config_file'] = 'configs/Kinetics/kinetics_mm_CSN.py'

from plato.config import Config
from plato.models.multimodal import multimodal_net
from plato.datasources.multimodal import kinetics

from adaptive_hgb import adaptive_hgb_client
from adaptive_hgb import adaptive_hgb_server
from adaptive_hgb import adaptive_hgb_trainer


def main():
    """ A Plato federated learning training session using the HGB algorithm. """
    _ = Config()
    print(Config)
    print(ok)

    kinetics_data_source = kinetics.DataSource()

    multi_model = multimodal_net.MM3F(
        multimoda_model_configs=Config.multimodal_data_model)

    trainer = adaptive_hgb_trainer.Trainer(model=multi_model)

    client = adaptive_hgb_client.Client(model=multi_model,
                                        datasource=kinetics_data_source,
                                        trainer=trainer)

    server = adaptive_hgb_server.Server(model=multi_model, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
