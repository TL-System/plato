"""Common subfunctions to multiple attacks."""
import torch


def optimizer_lookup(params, optim_name, step_size, scheduler=None, warmup=0, max_iterations=10_000):
    if optim_name.lower() == "adam":
        optimizer = torch.optim.Adam(params, lr=step_size)
    elif optim_name.lower() == "adam-safe":
        optimizer = torch.optim.Adam(params, lr=step_size, betas=(0.5, 0.99), eps=1e-4)
    elif optim_name.lower() == "bert-adam":
        # This is "almost" bert adam see more discussion at https://github.com/huggingface/transformers/issues/420
        optimizer = torch.optim.AdamW(params, lr=step_size, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
    elif optim_name.lower() == "momgd":
        optimizer = torch.optim.SGD(params, lr=step_size, momentum=0.9, nesterov=True)
    elif optim_name.lower() == "gd":
        optimizer = torch.optim.SGD(params, lr=step_size, momentum=0.0)
    elif optim_name.lower() == "l-bfgs":
        optimizer = torch.optim.LBFGS(params, lr=step_size)
    else:
        raise ValueError(f"Invalid optimizer {optim_name} given.")

    if scheduler == "step-lr":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[max_iterations // 2.667, max_iterations // 1.6, max_iterations // 1.142], gamma=0.1
        )
    elif scheduler == "cosine-decay":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iterations, eta_min=0.0)
    elif scheduler == "linear":

        def lr_lambda(current_step: int):
            return max(0.0, float(max_iterations - current_step) / float(max(1, max_iterations)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1)

    if warmup > 0:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=warmup, after_scheduler=scheduler)

    return optimizer, scheduler


"""The following code block is part of https://github.com/ildoonet/pytorch-gradual-warmup-lr.


MIT License

Copyright (c) 2019 Ildoo Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        after_scheduler_dict = {
            key: value for key, value in self.after_scheduler.__dict__.items() if key != "optimizer"
        }
        state_dict = {key: value for key, value in self.__dict__.items() if key != "optimizer"}
        state_dict["after_scheduler"] = after_scheduler_dict
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        after_scheduler_dict = state_dict.pop("after_scheduler")
        self.after_scheduler.__dict__.update(after_scheduler_dict)
        self.__dict__.update(state_dict)
