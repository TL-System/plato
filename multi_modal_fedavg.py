#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

os.environ['config_file'] = 'configs/Kinetics/kinetics_mm.py'

from plato.config import Config
from plato.models.multimodal import multimodal_net
from plato.datasources.multimodal import kinetics

from plato.trainers import basic
from plato.clients import simple
from plato.servers import fedavg


def main():
    """ A Plato federated learning training session using the HGB algorithm. """
    _ = Config()
    print(Config)
    print(ok)

    kinetics_data_source = kinetics.DataSource()

    multi_model = multimodal_net.MM3F(
        multimoda_model_configs=Config.multimodal_data_model)

    trainer = basic.Trainer(model=multi_model)

    client = simple.Client(model=multi_model,
                           datasource=kinetics_data_source,
                           trainer=trainer)

    server = fedavg.Server(model=multi_model, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()