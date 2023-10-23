"""
Implementation of the trainer for Calibre algorithm.

"""


from pflbases import ssl_trainer

from calibre_loss import CalibreLoss

from plato.config import Config


class Trainer(ssl_trainer.Trainer):
    """A trainer for the Calibre method."""

    def plato_ssl_loss_wrapper(self):
        """A wrapper to connect ssl loss with plato."""

        loss_criterion_name = (
            Config().trainer.loss_criterion
            if hasattr(Config.trainer, "loss_criterion")
            else "CrossEntropyLoss"
        )
        loss_criterion_params = (
            Config().parameters.loss_criterion._asdict()
            if hasattr(Config.parameters, "loss_criterion")
            else {}
        )

        auxiliary_losses = (
            Config().algorithm.auxiliary_loss_criterions
            if hasattr(Config.algorithm, "auxiliary_loss_criterions")
            else []
        )
        auxiliary_losses_params = (
            Config().algorithm.auxiliary_loss_criterions_param._asdict()
            if hasattr(Config.algorithm, "auxiliary_loss_criterions_param")
            else {}
        )

        losses_weight = (
            Config().algorithm.losses_weight
            if hasattr(Config.algorithm, "losses_weight")
            else {}
        )

        defined_ssl_loss = CalibreLoss(
            main_loss=loss_criterion_name,
            main_loss_params=loss_criterion_params,
            auxiliary_losses=auxiliary_losses,
            auxiliary_losses_params=auxiliary_losses_params,
            losses_weight=losses_weight,
        )

        def compute_plato_loss(outputs, labels):
            if isinstance(outputs, (list, tuple)):
                return defined_ssl_loss(*outputs, labels=labels)
            else:
                return defined_ssl_loss(outputs, labels=labels)

        return compute_plato_loss
