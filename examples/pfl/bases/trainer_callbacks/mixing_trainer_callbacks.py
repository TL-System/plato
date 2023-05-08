"""
Customize callbacks for trainer performing mixture optimization of the global model
and personalized models. Typical approaches are APFL, Ditto, LG-FedAvg.

As the global model and personalized models are jointly trained during the normal 
federated training process, the personalized model and its test accuracy will be 
saved after the local update.
"""


from bases.trainer_callbacks import base_callbacks


class PersonalizedModelStatusCallback(base_callbacks.PersonalizedModelCallback):
    """
    A trainer callback to record personalized learning status,
    1). at the end of each round of normal federated training.
    """

    def on_train_epoch_end(self, trainer, config, **kwargs):
        """Do not record the personalized model in each epoch."""

    def on_train_run_end(self, trainer, config, **kwargs):
        """Ensuring point 1)."""
        if not trainer.personalized_learning:
            learning_dict = kwargs["learning_dict"] if "learning_dict" in kwargs else {}
            super().on_train_run_end(trainer, config, learning_dict=learning_dict)


class PersonalizedModelMetricCallback(base_callbacks.PersonalizedMetricCallback):
    """A trainer callback to compute and record personalized metrics
    1). at the end of each round of normal federated training.
    2). at the start of any local update."""

    def on_train_epoch_end(self, trainer, config, **kwargs):
        """Do not perform test for the personalized model during training."""

    def on_train_run_end(self, trainer, config, **kwargs):
        """Ensuring point 1)."""

        if not trainer.personalized_learning:
            super().on_train_run_end(trainer, config, **kwargs)
