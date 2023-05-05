"""
A basic personalized federated learning client who performs the 
global learning and local learning.

"""

import os
import time
import logging
from types import SimpleNamespace, Any, Tuple

from plato.clients import simple
from plato.config import Config
from plato.models import registry as models_registry
from plato.utils import fonts
from plato.utils.filename_formatter import NameFormatter

from personalized_trainer import Trainer


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
        personalized_trainer=None,
        personalized_trainer_callbacks=None,
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

        # the personalized trainer
        self.custom_personalized_trainer = personalized_trainer
        self.personalized_trainer = None

        self.personalized_trainer_callbacks = personalized_trainer_callbacks

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

    def get_personalized_model_params(self):
        """Get the params of the personalized model."""
        return Config().parameters.personalized_model._asdict()

    def configure(self) -> None:
        """Performing the general client's configure and then initialize the
        personalized model for the client."""
        super().configure()

        # jump out if no personalized model is set
        if not hasattr(Config().trainer, "personalized_model_name"):
            return None

        pers_model_name = Config().trainer.personalized_model_name
        pers_model_type = (
            Config().trainer.personalized_model_type
            if hasattr(Config().trainer, "personalized_model_type")
            else pers_model_name.split("_")[0]
        )
        # assign the personalized model to the client
        if self.personalized_model is None and self.custom_personalized_model is None:

            pers_model_params = self.get_personalized_model_params()
            self.personalized_model = models_registry.get(
                model_name=pers_model_name,
                model_type=pers_model_type,
                model_params=pers_model_params,
            )
        elif (
            self.personalized_model is None
            and self.custom_personalized_model is not None
        ):
            self.personalized_model = self.custom_personalized_model()

        logging.info(
            "[Client #%d] defined the personalized model: %s",
            self.client_id,
            pers_model_name,
        )

        if (
            self.personalized_trainer is None
            and self.custom_personalized_trainer is None
        ):
            self.personalized_trainer = Trainer(
                model=self.personalized_model,
                callbacks=self.personalized_trainer_callbacks,
            )
        elif (
            self.personalized_trainer is None
            and self.custom_personalized_trainer is not None
        ):
            self.personalized_trainer = self.custom_personalized_trainer(
                model=self.personalized_model,
                callbacks=self.personalized_trainer_callbacks,
            )
        logging.info(
            "[Client #%d] defined the personalized trainer. %s", self.client_id
        )
        self.personalized_trainer.set_client_id(self.client_id)

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
        personalized_model_name = self.personalized_trainer.model_name
        logging.info(
            fonts.colourize(
                "[Client #%d] Loading its personalized model named %s.", colour="blue"
            ),
            self.client_id,
            personalized_model_name,
        )
        is_existed, filename = self.personalized_trainer.is_exist_unique_initial_model()

        if not is_existed:
            self.personalized_trainer.create_unique_initial_model(filename)
            self.novel_client = True
        else:
            self.novel_client = False

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
        return loaded_status

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

    async def personalized_train(self):
        """The machine learning training workload on a client for personalization."""

        logging.info(
            fonts.colourize(
                f"[{self}] Started personalized training in the communication round #{self.current_round}.",
                colour="blue",
            )
        )
        # Perform personalized model training
        try:
            if hasattr(self.personalized_trainer, "current_round"):
                self.personalized_trainer.current_round = self.current_round
            training_time = self.personalized_trainer.train(self.trainset, self.sampler)

        except ValueError as exc:
            logging.info(
                fonts.colourize(f"[{self}] Error occurred during training: {exc}")
            )
            await self.sio.disconnect()

        # Extract model weights and biases
        # this will obtain the parameters of self.model, which
        # should not be trained during this process
        weights = self.algorithm.extract_weights()

        if (hasattr(Config().clients, "do_test") and Config().clients.do_test) and (
            hasattr(Config().clients, "test_interval")
            and self.current_round % Config().clients.test_interval == 0
        ):
            accuracy = self.personalized_trainer.test(
                self.testset, self.testset_sampler
            )
        else:
            accuracy = 0

        # Generate a report for the server, performing model testing if applicable
        if accuracy == -1:
            # The testing process failed, disconnect from the server
            await self.sio.disconnect()

        # Do not print the accuracy if it is not computed
        if accuracy != 0:
            if hasattr(Config().trainer, "target_perplexity"):
                logging.info("[%s] Personalized Test perplexity: %.2f", self, accuracy)
            else:
                logging.info(
                    "[%s] Personalized Test accuracy: %.2f%%", self, 100 * accuracy
                )

        comm_time = time.time()

        report = SimpleNamespace(
            client_id=self.client_id,
            num_samples=self.sampler.num_samples(),
            accuracy=accuracy,
            training_time=training_time,
            comm_time=comm_time,
            update_response=False,
        )

        self._report = self.customize_report(report)

        return self._report, weights

    def is_personalized_learn(self):
        """Whether this client will perform personalization."""
        return self.learning_mode == "personalization"

    def is_participant_group(self):
        """Whether this client participants the federated training."""
        return self.client_group == "participant"
