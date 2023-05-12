"""
Customize the model saving callback for Ditto approach.
"""

from bases.trainer_callbacks import mixing_trainer_callbacks


class DittoStatusCallback(mixing_trainer_callbacks.PersonalizedModelStatusCallback):
    """
    A callback to additional record the updated alpha for APFL..
    """

    def on_train_run_end(self, trainer, config, **kwargs):
        """Recording the personalized model and the updated alpha"""
        learning_dict = {"ditto_lambda": trainer.ditto_lambda}
        super().on_train_run_end(trainer, config, learning_dict=learning_dict)
