"""
Customize callbacks to record the personalized model and alpha for APFL approach.

"""
import os
import logging

from plato.callbacks import trainer as plato_trainer
from plato.utils.filename_formatter import NameFormatter


class LearningStatusCallback(plato_trainer.TrainerCallback):
    """
    A callback to
    1). update alpha in each learning iteration.
    2). record learning status, including  the personalized model
    and updated alpha after learning.
    """

    def on_train_run_start(self, trainer, config, **kwargs):
        """Defining the variables and the required personalized optimizer
        for APFL."""
        super().on_train_run_start(trainer, config, **kwargs)

        # define the personalized optimizer
        trainer.personalized_optimizer = trainer.get_personalized_optimizer()

        # initialize the optimizer for personalization
        trainer.personalized_model.to(trainer.device)
        trainer.personalized_model.train()

        # initialize the alpha
        initial_alpha = config["alpha"]
        trainer.is_adaptive_alpha = config["is_adaptive_alpha"]
        trainer.alpha = initial_alpha if trainer.alpha == 0.0 else trainer.alpha

    def on_train_epoch_start(self, trainer, config, **kwargs):
        """Assign the learning rate to personalized optimizer."""
        super().on_train_epoch_start(trainer, config, **kwargs)
        trainer.personalized_optimizer.param_groups[0][
            "lr"
        ] = trainer.optimizer.param_groups[0]["lr"]

    def on_train_step_end(self, trainer, config, batch, loss, **kwargs):
        """Updating the alpha."""
        super().on_train_step_end(trainer, config, batch, loss, **kwargs)

        # update alpha based on the Eq. 10 of the paper.
        if trainer.is_adaptive_alpha and trainer.current_epoch == 1 and batch == 0:
            # 0.1/np.sqrt(1+args.local_index))
            lr = trainer.lr_scheduler.get_lr()[0]
            previous_alpha = trainer.alpha
            trainer.update_alpha(lr)
            logging.info(
                "[Client #%d] in round#%d Update alpha from %.6f to %.6f.",
                trainer.client_id,
                trainer.current_round,
                previous_alpha,
                trainer.alpha,
            )

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
