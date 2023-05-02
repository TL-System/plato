"""
The implemetation of the trainer for SMoG approach.
"""

import logging

from lightly.loss.memory_bank import MemoryBankModule
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule

from plato.config import Config
from plato.utils.filename_formatter import NameFormatter
from plato.trainers import basic_ssl
from plato.utils import checkpoint_operator


class Trainer(basic_ssl.Trainer):
    """A personalized federated learning trainer with self-supervised learning."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        self.momentum_val = 0

        # training steps
        self.global_step = 0
        self.local_step = 0

        # memory bank because we reset the group features every 300 iterations
        self.reset_interval = (
            Config().trainer.reset_interval
            if hasattr(Config().trainer, "reset_interval")
            else 300
        )
        memory_bank_size = self.reset_interval * Config().trainer.batch_size
        self.memory_bank = MemoryBankModule(size=memory_bank_size)

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform training."""
        self.optimizer.zero_grad()

        outputs = self.model(examples)

        encoded_samples = outputs[-1]

        loss = self._loss_criterion(outputs[:2], labels)
        self._loss_tracker.update(loss, labels.size(0))

        self.memory_bank(encoded_samples, update=True)

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        self.optimizer.step()

        return loss

    def train_epoch_start(self, config):
        """Operations before starting one epoch."""
        super().train_epoch_start(config)
        epoch = self.current_epoch
        total_epochs = config["epochs"] * config["rounds"]
        self.momentum_val = cosine_schedule(epoch, total_epochs, 0.996, 1)

    def train_step_start(self, config, batch=None):
        """Operations before starting one iteration."""
        super().train_step_start(config)

        # update the global step
        self.global_step += batch

        if self.global_step > 0 and self.global_step % self.reset_interval == 0:
            # reset group features and weights every 300 iterations
            self.model.reset_group_features(memory_bank=self.memory_bank)
            self.model.reset_momentum_weights()
        else:
            update_momentum(
                self.model.encoder, self.model.encoder_momentum, m=self.momentum_val
            )
            update_momentum(
                self.model.projection_head,
                self.model.projection_head_momentum,
                m=self.momentum_val,
            )

        # update the local iteration
        self.model.n_iteration = batch

    def rollback_memory_bank(self, config):
        """Load the memory bank."""
        desired_round = self.current_round - 1
        checkpoint_dir_path = self.get_checkpoint_dir_path()
        filename, ckpt_oper = checkpoint_operator.load_client_checkpoint(
            client_id=self.client_id,
            checkpoints_dir=checkpoint_dir_path,
            model_name="memory_bank",
            current_round=desired_round,
            run_id=None,
            epoch=None,
            prefix="personalized",
            anchor_metric="round",
            mask_words=["epoch"],
            use_latest=True,
        )
        if filename is not None:  
            rollback_status = ckpt_oper.load_checkpoint(checkpoint_name=filename)
            memory_bank_status = rollback_status["model"]
            self.memory_bank.bank = memory_bank_status["bank"]
            self.memory_bank.bank_ptr = memory_bank_status["bank_ptr"]
            logging.info(
                "[Client #%d] Rolled back the Memory Bank from %s under %s.",
                self.client_id,
                filename,
                checkpoint_dir_path,
            )

    def save_memory_bank(self, config):
        """Save the memory bank locally."""
        current_round = self.current_round

        save_location = self.get_checkpoint_dir_path()
        filename = NameFormatter.get_format_name(
            client_id=self.client_id,
            model_name="memory_bank",
            round_n=current_round,
            run_id=None,
            prefix="personalized",
            ext="pth",
        )
        ckpt_oper = checkpoint_operator.CheckpointsOperator(
            checkpoints_dir=save_location
        )
        ckpt_oper.save_checkpoint(
            model_state_dict={
                "bank": self.memory_bank.bank,
                "bank_ptr": self.memory_bank.bank_ptr,
            },
            checkpoints_name=[filename],
        )

        logging.info(
            "[Client #%d] Saved Memory Bank model to %s under %s.",
            self.client_id,
            filename,
            save_location,
        )

    def train_run_start(self, config):
        super().train_run_start(config)
        self.rollback_memory_bank(config)

    def train_run_end(self, config):
        super().train_run_end(config)

        self.save_memory_bank(config)
