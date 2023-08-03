"""
Implementation of the client for Prompt Federated Learning.
"""
import logging

from auxfl.clients import fed_prompt_client


class Client(fed_prompt_client.Client):
    """A client for Prompt Federated Learning."""

    def configure(self) -> None:
        """Defining the prompts."""
        super().configure()

        if self.prompts is None and self.customize_prompts is not None:
            self.prompts = self.customize_prompts

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
                "[Client #%d] Initialize placeholder embedding with token: %s, ",
                self.client_id,
                self.trainer.initializer_token,
            )
            self.trainer.model.initial_placeholder_embed(
                initializer_token=self.trainer.initializer_token
            )

        logging.info(
            "[Client #%d] concept_name: %s, what_to_teach: %s, placeholder_token: %s, initializer_token: %s, ",
            self.client_id,
            self.trainer.concept_name,
            self.trainer.what_to_teach,
            self.trainer.placeholder_token,
            self.trainer.initializer_token,
        )
