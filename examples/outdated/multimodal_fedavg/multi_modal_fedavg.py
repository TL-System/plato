import os

os.environ["config_file"] = "configs/Kinetics/kinetics_mm.py"

from plato.datasources.multimodal import kinetics

from plato.clients import simple
from plato.config import Config
from plato.models.multimodal import multimodal_module
from plato.servers import fedavg
from plato.trainers import basic


def main():
    """A Plato federated learning training session using the HGB algorithm."""
    _ = Config()
    support_modalities = ["rgb", "flow", "audio"]

    kinetics_data_source = kinetics.DataSource()

    # define the sub-nets to be untrained
    for modality_nm in support_modalities:
        modality_model_nm = modality_nm + "_model"
        if modality_model_nm in Config.multimodal_nets_configs.keys():
            modality_net = Config.multimodal_nets_configs[modality_model_nm]
            modality_net["backbone"]["pretrained"] = None
            if "pretrained2d" in list(modality_net["backbone"].keys()):
                modality_net["backbone"]["pretrained2d"] = False

    # define the model
    multi_model = multimodal_module.DynamicMultimodalModule(
        support_modality_names=support_modalities,
        multimodal_nets_configs=Config.multimodal_nets_configs,
        is_fused_head=True,
    )

    trainer = basic.Trainer(model=multi_model)

    client = simple.Client(
        model=multi_model, datasource=kinetics_data_source, trainer=trainer
    )

    server = fedavg.Server(model=multi_model, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
