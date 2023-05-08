"""
Customize callbacks for trainer performing separate optimization of the global model
and personalized models. Typical approaches are FedBABU, FedRep.
"""


from bases.trainer_callbacks import base_callbacks


class PersonalizedModelStatusCallback(base_callbacks.PersonalizedModelCallback):
    """
    A trainer callback to record learning status, including the
    personalized model and additional variables after learning.
    """

    def on_train_run_start(self, trainer, config, **kwargs):
        """Performing the test for the personalized model only during
        personalization."""
        if trainer.personalized_learning:
            super().on_train_run_start(trainer, config, **kwargs)
