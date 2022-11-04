from plato.trainers import basic
from NASVIT.misc.config import get_config
from timm.loss import LabelSmoothingCrossEntropy
import torch.optim as optim
import torch
import copy
from plato.config import Config
import time


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or "rescale" in name
            or "bn" in name
            or (name in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [
        {"params": has_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


class Trainer(basic.Trainer):  # WithTimmScheduler):
    def get_loss_criterion(self):
        return LabelSmoothingCrossEntropy(smoothing=get_config().MODEL.LABEL_SMOOTHING)

    def get_optimizer(self, model):
        config = get_config()

        skip = {"rescale", "bn", "absolute_pos_embed"}
        skip_keywords = {"relative_position_bias_table"}
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()
        if hasattr(model, "no_weight_decay_keywords"):
            skip_keywords = model.no_weight_decay_keywords()

        # add weight decay before gamma (double check!!)
        parameters = set_weight_decay(model, skip, skip_keywords)
        base_opt = optim.AdamW
        optimizer = base_opt(
            parameters,
            eps=config.TRAIN.OPTIMIZER.EPS,
            betas=config.TRAIN.OPTIMIZER.BETAS,
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )
        return optimizer

    def _adjust_lr(self, config, lr_scheduler, optimizer) -> torch.optim.Optimizer:
        """Returns an optimizer with an initial learning rate that has been
        adjusted according to the current round, so that learning rate
        schedulers can be effective throughout the communication rounds."""

        if "global_lr_scheduler" in config and config["global_lr_scheduler"]:
            global_lr_scheduler = copy.deepcopy(lr_scheduler)

            t = 0
            for __ in range(self.current_round - 1):
                for __ in range(Config().trainer.epochs):
                    global_lr_scheduler.step(t)
                    t += 1

            if Config().trainer.lr_scheduler == "timm":
                initial_lr = global_lr_scheduler._get_lr(t)
            else:
                initial_lr = global_lr_scheduler.get_last_lr()
            optimizer.param_groups[0]["lr"] = initial_lr[0]

        return optimizer
