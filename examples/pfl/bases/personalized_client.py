"""
A basic personalized federated learning client who performs the 
global learning and local learning.

"""
import os
import logging
from types import SimpleNamespace, Any, Tuple

from plato.clients import simple
from plato.config import Config
from plato.utils import fonts
from plato.utils.filename_formatter import NameFormatter


class Client(simple.Client):
    """A basic personalized federated learning client."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        personalized_model=None,
    ):
        # pylint:disable=too-many-arguments
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        # the personalized model here corresponds to the client's
        # personal needs.
        self.custom_personalized_model = personalized_model
        self.personalized_model = None

        # the learning model to be performed in this client
        # by default, performing `normal` fl's local update
        # there are two options:
        # 1.- normal
        # 2.- personalization
        self.learning_mode = "normal"

        # which group that client belongs to
        # there are two options:
        # 1. participant
        # 2. nonparticipant
        self.client_group = "participant"

        # whether this client contains the corresponding
        # personalized model
        self.novel_client = False

    def process_server_response(self, server_response) -> None:
        """Additional client-specific processing on the server response."""

        super().process_server_response(server_response)
        self.learning_mode = server_response["learning_mode"]

    def configure(self) -> None:
        """Performing the general client's configure and then initialize the
        personalized model for the client."""
        super().configure()

        # jump out if no personalized model is set
        if not hasattr(Config().algorithm, "personalization"):
            return None

        # define the personalized model
        if (
            self.personalized_model is None
            and self.custom_personalized_model is not None
        ):
            self.personalized_model = self.custom_personalized_model

        if self.trainer.personalized_model is None:
            self.trainer.define_personalized_model(self.personalized_model)

        self.trainer.set_training_mode(personalized_mode=self.is_personalized_learn())

    def load_personalized_model(self):
        """Load the personalized model.

        This function is necessary for personalized federated learning in
        Plato. Because, in general, when one client is called the first time,
        its personalized model should be randomly intialized. Howerver,
        Plato utilizes the `process` to simulate the client and only the client
        id of each `process` is changed.

        Therefore, in each round, the selected client (each `process`) should load
        its personalized model instead of using the current self.personalized_model
        trained by others.

        By default,
        1. the personalized model will be loaded from the initialized one.
        2. load the latest persisted personalized model.
        """
        personalized_model_name = self.trainer.personalized_model_name
        logging.info(
            fonts.colourize(
                "[Client #%d] Loading its personalized model named %s.", colour="blue"
            ),
            self.client_id,
            personalized_model_name,
        )
        filename = self.is_novel_client()

        if not self.novel_client:
            self.trainer.create_unique_personalized_model(filename)

        # when `persist_personalized_model` is set to be True, it means
        # that each client want to load its latest trained personalzied
        # model instead of using the initial one.
        if (
            hasattr(Config().clients, "persist_personalized_model")
            and Config().clients.persist_personalized_model
        ):
            # load the client's latest personalized model
            desired_round = self.current_round - 1

            logging.info(
                fonts.colourize(
                    "[Client #%d] Loading latest personalized model.", colour="blue"
                ),
                self.client_id,
            )
        else:
            # client does not want to use its trained personalzied model
            # thus, load the initial personalized model saved by
            # `self.persist_initial_personalized_model`
            # i.e., rollback
            desired_round = 0
            logging.info(
                fonts.colourize(
                    "[Client #%d] Loading initial personalized model.", colour="blue"
                ),
                self.client_id,
            )

        checkpoint_dir_path = self.trainer.get_checkpoint_dir_path()
        loaded_status = self.trainer.rollback_model(
            rollback_round=desired_round,
            location=checkpoint_dir_path,
        )
        self.trainer.personalized_model.load_state_dict(
            loaded_status["model"], strict=True
        )
        return loaded_status

    def inbound_received(self, inbound_processor):
        """Reloading the personalized model for this client before any operations."""
        if self.is_personalized_learn() and self.personalized_model is not None:
            self.load_personalized_model()

    async def inbound_processed(
        self, processed_inbound_payload: Any
    ) -> Tuple[SimpleNamespace, Any]:

        if self.is_personalized_learn():
            report, outbound_payload = await self.personalized_train(
                processed_inbound_payload
            )
        else:
            report, outbound_payload = await self._start_training(
                processed_inbound_payload
            )

        return report, outbound_payload

    def is_personalized_learn(self):
        """Whether this client will perform personalization."""
        return self.learning_mode == "personalization"

    def is_participant_group(self):
        """Whether this client participants the federated training."""
        return self.client_group == "participant"

    def is_novel_client(self):
        """Whether this client is a novel one, which is never selected by the
        server, as a result, no unique personalized model is maintained."""
        checkpoint_dir_path = self.trainer.get_checkpoint_dir_path()

        filename = NameFormatter.get_format_name(
            model_name=self.trainer.personalized_model_name,
            client_id=self.client_id,
            round_n=0,
            epoch_n=None,
            run_id=None,
            prefix=self.trainer.personalized_model_checkpoint_prefix,
            ext="pth",
        )
        checkpoint_file_path = os.path.join(checkpoint_dir_path, filename)

        self.novel_client = os.path.exists(checkpoint_file_path)

        return filename
