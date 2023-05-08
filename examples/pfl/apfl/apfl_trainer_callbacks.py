"""
Customize callbacks to record the personalized model and alpha for APFL approach.

"""


from bases.trainer_callbacks import mixing_trainer_callbacks


class APFLStatusCallback(mixing_trainer_callbacks.PersonalizedModelStatusCallback):
    """
    A callback to additional record the updated alpha for APFL.
    """

    def on_train_run_end(self, trainer, config, **kwargs):
        """Recording the personalized model and the updated alpha"""
        learning_dict = {"alpha": trainer.alpha}
        super().on_train_run_end(trainer, config, learning_dict=learning_dict)
