"""
A basic personalized federated learning client who performs the 
global learning and local learning.

"""

import os
import time
import logging
from types import SimpleNamespace

from plato.clients import simple
from plato.config import Config
from plato.models import registry as models_registry
from plato.utils import fonts
from plato.utils.filename_formatter import NameFormatter
from plato.trainers import registry as trainers_registry


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
        logging.info(
            "[Client #%d] defines the personalized trainer. %s", self.client_id
        )
        if (
            self.personalized_trainer is None
            and self.custom_personalized_trainer is None
        ):
            self.personalized_trainer = trainers_registry.get(
                model=self.personalized_model,
                callbacks=self.personalized_trainer_callbacks,
                type=Config().trainer.personalized_type,
            )
        elif (
            self.personalized_trainer is None
            and self.custom_personalized_trainer is not None
        ):
            self.personalized_trainer = self.custom_personalized_trainer(
                model=self.personalized_model,
                callbacks=self.personalized_trainer_callbacks,
            )

        self.personalized_trainer.set_client_id(self.client_id)

        # assign the client's personalized model to its trainer
        # we need to know that in Plato, the personalized model here
        # makes no sense and it is only initialized to hold the model
        # structure/parameters.
        # The main reason is that Plato simulates the client with multiple
        # `processes`, which are started at the beginning of running. These
        # The server will build fake connection with these `processes` - when
        # the `processes` receive information from the server, the the
        # `configuration()` function is called the first time to perform
        # necessary initialization (model, trainer, algorithm, personalized model).
        # However, only when the actual clients are selected by the server,
        # these `processes` will be assigned meaningful client id.
        # At that time, the parameters of model and personalized model of each client
        # corresponding to one `process` will be assigned with received payloads
        # or initialized for current client - see function `_load_payload`

        # to save space and time, the personalized model of the trainer will be
        # assigned only during the first time - the process is created.
        if self.trainer.personalized_model is None:
            self.trainer.set_client_personalized_model(self.personalized_model)

    def persist_initial_personalized_model(self):
        """Persist the initial model of one client."""
        pers_model_name = Config().trainer.personalized_model_name
        # save the defined personalized model as the initial one
        checkpoint_dir_path = self.trainer.get_checkpoint_dir_path()

        filename = NameFormatter.get_format_name(
            model_name=pers_model_name,
            client_id=self.client_id,
            round_n=0,
            epoch_n=None,
            run_id=None,
            prefix="personalized",
            ext="pth",
        )
        checkpoint_file_path = os.path.join(checkpoint_dir_path, filename)
        self.novel_client = False
        # if the personalized model for current client does
        # not exist - this client is selected the first time
        if not os.path.exists(checkpoint_file_path):
            logging.info(
                fonts.colourize(
                    "First-time Selection of [Client #%d] for personalization.",
                    colour="blue",
                ),
                self.client_id,
            )
            logging.info(
                fonts.colourize(
                    "[Client #%d] Creating its unique personalized parameters by resetting weights.",
                    colour="blue",
                ),
                self.client_id,
            )
            # reset the personalized model for this client
            # thus, different clients have different init parameters
            self.personalized_model.apply(self.trainer.reset_weight)
            self.trainer.save_personalized_model(
                filename=filename,
                location=checkpoint_dir_path,
            )
            # set this client to be the novel one
            self.novel_client = True

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
        personalized_model_name = Config().trainer.personalized_model_name
        logging.info(
            fonts.colourize(
                "[Client #%d] Loading its personalized model named %s.", colour="blue"
            ),
            self.client_id,
            personalized_model_name,
        )
        # when `persist_personalized_model` is set to be True, it means
        # that each client want to load its latest trained personalzied
        # model instead of using the initial one.
        if (
            hasattr(Config().clients, "persist_personalized_model")
            and Config().clients.persist_personalized_model
        ):
            # load the client's latest personalized model
            # we should know that only when this client is selected the first time,
            # the initial personalized model saved by `self.persist_initial_model`
            # will be loaded. Otherwise, the latest trained personalized model
            # will be loaded
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
            model_name=personalized_model_name,
            modelfile_prefix="personalized",
            rollback_round=desired_round,
            location=checkpoint_dir_path,
        )
        return loaded_status

    def _load_payload(self, server_payload) -> None:
        """Load the server model onto this client.

        By default, each client will
        1. load the received global model to its trainer's model
        2. load its personalized model locally.
        """
        logging.info(
            "[Client #%d] Received the payload containing modules: %s.",
            self.client_id,
            self.algorithm.extract_modules_name(list(server_payload.keys())),
        )
        # load the model
        self.algorithm.load_weights(server_payload)

        if self.is_personalized_learn() and self.personalized_model is not None:
            # This operation is important to the personalized FL
            # under Plato
            # Because, in general, when one client is called the first time,
            # its personalized model should be randomly intialized.
            # Howerver, Plato utilizes the `process` to simulate the
            # client and only the client id of each `process` is changed.
            # Thus, even a unseen client is selected, its personalized model
            # is the one trained by other previous clients.
            # Here, the function aims to
            #  1. initial the personalized model for this client
            #  2. persist the initialized personalized model
            # when the client is selected the first time.
            self.persist_initial_personalized_model()

            # load the personalized model.
            self.load_personalized_model()

    async def _train(self):
        """The machine learning training workload on a client.

        To make it flexible, there are three training mode.

        """

        accuracy = -1
        training_time = 0.0

        if hasattr(self.trainer, "current_round"):
            self.trainer.current_round = self.current_round

        if self.is_personalized_learn():
            logging.info(
                fonts.colourize(
                    f"[{self}] Started personalized training in the communication round #{self.current_round}.",
                    colour="blue",
                )
            )
            try:
                training_time, accuracy = self.trainer.personalized_train(
                    self.trainset,
                    self.sampler,
                    testset=self.testset,
                    testset_sampler=self.testset_sampler,
                )
            except ValueError:
                await self.sio.disconnect()
        else:
            logging.info(
                fonts.colourize(
                    f"[{self}] Started training in communication round #{self.current_round}."
                )
            )
            try:
                training_time = self.trainer.train(self.trainset, self.sampler)
            except ValueError:
                await self.sio.disconnect()

            if (hasattr(Config().clients, "do_test") and Config().clients.do_test) and (
                hasattr(Config().clients, "test_interval")
                and self.current_round % Config().clients.test_interval == 0
            ):
                accuracy = self.trainer.test(self.testset, self.testset_sampler)
            else:
                accuracy = 0

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        if accuracy == -1:
            # The testing process failed, disconnect from the server
            await self.sio.disconnect()

        # Do not print the accuracy if it is not computed
        if accuracy != 0:
            if hasattr(Config().trainer, "target_perplexity"):
                logging.info("[%s] Test perplexity: %.2f", self, accuracy)
            else:
                logging.info("[%s] Test accuracy: %.2f%%", self, 100 * accuracy)

        comm_time = time.time()

        if (
            hasattr(Config().clients, "sleep_simulation")
            and Config().clients.sleep_simulation
        ):
            sleep_seconds = Config().client_sleep_times[self.client_id - 1]
            avg_training_time = Config().clients.avg_training_time

            training_time = (
                avg_training_time + sleep_seconds
            ) * Config().trainer.epochs

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
