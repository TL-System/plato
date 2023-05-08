"""
Customize callbacks to record the personalized model and alpha for APFL approach.

"""
import os

from plato.callbacks import trainer as plato_trainer
from plato.utils.filename_formatter import NameFormatter


class LearningStatusCallback(plato_trainer.TrainerCallback):
    """
    A callback to record learning status, including the
    personalized model and updated alpha after learning.
    """

    def on_train_run_end(self, trainer, config, **kwargs):
        """Recording the personalized model and the updated alpha"""
        super().on_train_run_end(trainer, config, **kwargs)

        if "max_concurrency" in config:
            current_round = trainer.current_round

            learning_dict = {"alpha": trainer.alpha}
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
                filename=filename, location=save_location, learning_dict=learning_dict
            )
