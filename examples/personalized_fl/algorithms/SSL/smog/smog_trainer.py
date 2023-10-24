"""
The implemetation of the trainer for SMoG approach.
"""

import logging

from lightly.loss.memory_bank import MemoryBankModule
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule

from plato.config import Config
from pflbases.filename_formatter import NameFormatter


from pflbases import ssl_trainer
from pflbases import checkpoint_operator


class Trainer(ssl_trainer.Trainer):
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

    def model_forward(self, examples):
        """Forward the input examples to the model."""
        outputs = self.model(examples)
        encoded_samples = outputs[-1]
        self.memory_bank(encoded_samples, update=True)
        return outputs[:2]

    def train_epoch_start(self, config):
        """Operations before starting one epoch."""
        super().train_epoch_start(config)
        epoch = self.current_epoch
        total_epochs = config["epochs"] * config["rounds"]
        if not self.do_final_personalization:
            self.momentum_val = cosine_schedule(epoch, total_epochs, 0.996, 1)

    def train_step_start(self, config, batch=None):
        """Operations before starting one iteration."""
        super().train_step_start(config)

        if not self.do_final_personalization:
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

    def get_memory_bank(self, config):
        """Load the memory bank."""
        desired_round = self.current_round - 1
        checkpoint_dir_path = self.get_checkpoint_dir_path()
        filename = NameFormatter.get_format_name(
            client_id=self.client_id,
            model_name="memory_bank",
            round_n=desired_round,
            run_id=None,
            prefix="personalized",
            ext="pth",
        )
        filename, is_searched = checkpoint_operator.search_checkpoint_file(
            filename=filename,
            checkpoints_dir=checkpoint_dir_path,
            key_words=["memory_bank", "personalized"],
            anchor_metric="round",
            mask_words=["epoch"],
            use_latest=True,
        )

        if is_searched:
            ckpt_oper = checkpoint_operator.CheckpointsOperator(checkpoint_dir_path)
            rollback_status = ckpt_oper.load_checkpoint(checkpoint_name=filename)
            memory_bank_status = rollback_status["model"]
            self.memory_bank.bank = memory_bank_status["bank"]
            self.memory_bank.bank_ptr = memory_bank_status["bank_ptr"]
            logging.info(
                "[Client #%d] Got the Memory Bank from %s under %s.",
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
        if not self.do_final_personalization:
            self.get_memory_bank(config)

    def train_run_end(self, config):
        super().train_run_end(config)
        if not self.do_final_personalization:
            self.save_memory_bank(config)
