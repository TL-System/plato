"""
Customize trainer callbacks for FedRep approach.
"""


from plato.config import Config

from bases.trainer_callbacks import base_callbacks


class PersonalizedModelStatusCallback(base_callbacks.PersonalizedModelCallback):
    """
    A trainer callback to record personalized learning status
    1). at the end of each training round of normal federated learning.
    2). every epochs interval only in the final personalization.
    """

    def on_train_epoch_end(self, trainer, config, **kwargs):
        """Ensuring point 2)."""
        if (
            trainer.personalized_learning
            and trainer.current_round == Config().trainer.rounds + 1
        ):
            super().on_train_epoch_end(trainer, config, **kwargs)


class PersonalizedModelMetricCallback(base_callbacks.PersonalizedMetricCallback):
    """A trainer callback to compute and record the personalized metrics,
        1). at the start of training process.
        2). at each epoch only in the final personalization.
        3). at the end of each round of normal federated learning.
    ."""

    def on_train_epoch_end(self, trainer, config, **kwargs):
        """Ensuring point 2)."""
        if (
            trainer.personalized_learning
            and trainer.current_round == Config().trainer.rounds + 1
        ):
            super().on_train_epoch_end(trainer, config, **kwargs)
