"""
Customize callbacks for trainer performing mixture optimization of the global model
and personalized models. Typical approaches are APFL, Ditto, LG-FedAvg.

As the global model and personalized models are jointly trained during the normal 
federated training process, the personalized model and its test metrics will be 
saved after the local update.

Therefore,
1. personalized model will be saved and tested during the normal federated learning
process.
2. no personalization model will be optimized during the personalization.
"""


from pflbases.trainer_callbacks import base_callbacks
from plato.config import Config


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

    def on_train_run_start(self, trainer, config, **kwargs):
        if (
            hasattr(Config().algorithm.personalization, "do_test_per_epoch")
            and Config().algorithm.personalization.do_test_per_epoch
        ):
            return super().on_train_run_start(trainer, config, **kwargs)

    def on_train_epoch_end(self, trainer, config, **kwargs):
        """Do not perform test for the personalized model during training."""

    def on_train_run_end(self, trainer, config, **kwargs):
        """Ensuring point 1)."""

        if not trainer.personalized_learning and (
            hasattr(Config().algorithm.personalization, "do_test_per_epoch")
            and Config().algorithm.personalization.do_test_per_epoch
        ):
            super().on_train_run_end(trainer, config, **kwargs)
