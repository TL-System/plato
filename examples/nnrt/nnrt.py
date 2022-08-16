"""
Federated Learning session with NNRT.
"""
from functools import partial

import nnrt_datasource_yolo as nnrt_datasource
import nnrt_trainer_yolo as nnrt_trainer
from nnrt_algorithms import mistnet
from nnrt_models import acl_inference
from plato.config import Config
from plato.clients.mistnet import Client


def main():
    """A Plato mistnet training session using an nnrt yolo model, datasource and trainer."""
    model = partial(
        acl_inference.Inference,
        int(Config().trainer.deviceID),
        Config().trainer.om_path,
        Config().data.input_height,
        Config().data.input_width,
    )
    datasource = nnrt_datasource.DataSource  # special datasource for yolo model
    trainer = nnrt_trainer.Trainer
    algorithm = mistnet.Algorithm

    client = Client(
        model=model, datasource=datasource, algorithm=algorithm, trainer=trainer
    )

    client.load_data()
    __, __ = client.train()


if __name__ == "__main__":
    main()
