"""
Implementation of the server for Federated Textual Inversion
"""
import os
import logging

import torch
from plato.servers import fedavg
from plato.config import Config


class Server(fedavg.Server):
    """A server to perform the SplitFed."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        # the prompts and corresponding encodings
        # with shape, [n_prompts, prompt_length, n_encodings]
        self.global_soft_prompts: torch.Tensor = None

        # the type of payload, it can be
        # 1. parameters
        # 2. prompts
        # 3. both
        self.payload_type = (
            Config().algorithm.payload_type
            if hasattr(Config().algorithm, "payload_type")
            else "parameters"
        )

    def init_trainer(self) -> None:
        """Creating placeholder for the model of the trainer."""
        super().init_trainer()

        if not hasattr(self.trainer, "placeholder_token"):
            return

        if self.trainer.placeholder_token is None:
            self.trainer.create_what_to_teach()
            self.trainer.create_concept()
            self.trainer.create_placeholder_token()
            self.trainer.create_initializer_token()

            n_added = self.trainer.model.tokenizer_add_placeholder(
                placeholder_token=self.trainer.placeholder_token
            )
            # this is a new placeholder token for current prompt
            # learner,
            if n_added == 1:
                logging.info(
                    "[Server #%d] Initialize placeholder embedding with token: %s, ",
                    os.getpid(),
                    self.trainer.initializer_token,
                )
                self.trainer.model.initial_placeholder_embed(
                    initializer_token=self.trainer.initializer_token
                )

            logging.info(
                "[Server #%d] concept_name: %s, what_to_teach: %s, placeholder_token: %s, initializer_token: %s, ",
                os.getpid(),
                self.trainer.concept_name,
                self.trainer.what_to_teach,
                self.trainer.placeholder_token,
                self.trainer.initializer_token,
            )
