"""
Customize callbacks for trainer performing separate optimization of the global model
and personalized models. Typical approaches are FedBABU, FedAvg-finetuning.

Therefore,
1. During the normal federated learning process, the personalized model is 
to be tested only at the begininng of training and no personalized model 
will besaved. 
2. After the normal federated learning process, each client will perform 
personalization for mulitple epochs to train the personalized model.
"""


from plato.config import Config

from pflbases.trainer_callbacks import base_callbacks


class PersonalizedModelStatusCallback(base_callbacks.PersonalizedModelCallback):
    """
    A trainer callback to record personalized learning status
    1). at the end of any personalization.
    2). every epochs interval only in the final personalization.
    """

    def on_train_epoch_end(self, trainer, config, **kwargs):
        """Ensuring point 2)."""
        if (
            trainer.personalized_learning
            and trainer.current_round == Config().trainer.rounds + 1
        ):
            super().on_train_epoch_end(trainer, config, **kwargs)

    def on_train_run_end(self, trainer, config, **kwargs):
        """Ensuring point 1)."""
        if trainer.personalized_learning:
            learning_dict = kwargs["learning_dict"] if "learning_dict" in kwargs else {}
            super().on_train_run_end(trainer, config, learning_dict=learning_dict)


class PersonalizedModelMetricCallback(base_callbacks.PersonalizedMetricCallback):
    """A trainer callback to compute and record the personalized metrics,
        1). at the start of training process.
        2). at each epoch only in the final personalization.
        3). at the end of each personalization.
    ."""

    def on_train_epoch_end(self, trainer, config, **kwargs):
        """Ensuring point 2)."""
        if (
            trainer.personalized_learning
            and trainer.current_round == Config().trainer.rounds + 1
        ):
            super().on_train_epoch_end(trainer, config, **kwargs)

    def on_train_run_end(self, trainer, config, **kwargs):
        """Ensuring point 3)."""

        if trainer.personalized_learning:
            super().on_train_run_end(trainer, config, **kwargs)
