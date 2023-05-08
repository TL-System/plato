"""
Customize callbacks for trainer performing mixture optimization of the global model
and personalized models. Typical approaches are APFL, Ditto, LG-FedAvg.

As the global model and personalized models are jointly trained during the normal 
federated training process, the personalized model and its test accuracy will be 
saved after the local update.
"""

import os

from plato.callbacks import trainer as plato_trainer
from plato.utils.filename_formatter import NameFormatter
from plato.config import Config

from bases.trainer_utils import checkpoint_personalized_accuracy


class PersonalizedModelStatusCallback(plato_trainer.TrainerCallback):
    """
    A trainer callback to record learning status, including the
    personalized model and additional variables after learning.
    """

    def on_train_run_end(self, trainer, config, **kwargs):
        """Recording the personalized model and the updated alpha"""
        super().on_train_run_end(trainer, config, **kwargs)

        if not trainer.personalized_learning:
            if "max_concurrency" in config:
                current_round = trainer.current_round

                learning_dict = (
                    kwargs["learning_dict"] if "learning_dict" in kwargs else {}
                )
                personalized_model_name = trainer.personalized_model_name
                save_location = trainer.get_checkpoint_dir_path()
                filename = NameFormatter.get_format_name(
                    client_id=trainer.client_id,
                    model_name=personalized_model_name,
                    round_n=current_round,
                    run_id=None,
                    prefix=trainer.personalized_model_checkpoint_prefix,
                    ext="pth",
                )
                os.makedirs(save_location, exist_ok=True)
                trainer.save_personalized_model(
                    filename=filename,
                    location=save_location,
                    learning_dict=learning_dict,
                )


class PersonalizedModelMetricCallback(plato_trainer.TrainerCallback):
    """A trainer callback to compute and record the test accuracy of the
    personalized model."""

    def on_train_run_end(self, trainer, config, **kwargs):
        super().on_train_run_end(trainer, config, **kwargs)
        # perform test for the personalized model
        result_path = Config().params["result_path"]

        if not trainer.personalized_learning:
            test_outputs = trainer.test_personalized_model(config)

            checkpoint_personalized_accuracy(
                result_path,
                client_id=trainer.client_id,
                accuracy=test_outputs["accuracy"],
                current_round=trainer.current_round,
                epoch=trainer.current_epoch,
                run_id=None,
            )

            trainer.personalized_model.to(trainer.device)
            trainer.personalized_model.train()
