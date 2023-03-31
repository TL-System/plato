"""
A basic personalized federated learning client
who performs the global learning and local learning.

"""

import time
import logging
from types import SimpleNamespace

from plato.clients import simple
from plato.config import Config
from plato.models import registry as models_registry
from plato.utils import checkpoint_operator
from plato.utils.filename_formatter import get_format_name
from plato.utils import fonts


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

    def persist_initial_model(self):
        """Persist the initial model of one client."""
        pers_model_name = Config().trainer.personalized_model_name
        # save the defined personalized model as the initial one
        checkpoint_path = Config.params["checkpoint_path"]
        cpk_oper = checkpoint_operator.CheckpointsOperator(
            checkpoints_dir=checkpoint_path
        )
        filename = get_format_name(
            model_name=pers_model_name,
            client_id=self.client_id,
            round_n=0,
            prefix="personalized",
            ext="pth",
        )

        if not cpk_oper.vaild_checkpoint_file(filename):
            # reset the personalized model for this client
            self.personalized_model.apply(cpk_oper.reset_weight)
            cpk_oper.save_checkpoint(
                model_state_dict=self.personalized_model.state_dict(),
                checkpoints_name=[filename],
            )
            logging.info(
                "Client[%d] saves the initial personalized model to %s under %d",
                self.client_id,
                filename,
                checkpoint_path,
            )

    def configure(self) -> None:
        """Performing the general client's configure and then initialize the
        personalized model for the client."""
        super().configure()

        # jump out if no personalized model is set
        if not hasattr(Config().trainer, "personalized_model_name"):
            return None

        # assign the personalized model to the client
        if self.custom_personalized_model is not None:
            self.personalized_model = self.custom_personalized_model
            self.custom_personalized_model = None
        pers_model_name = Config().trainer.personalized_model_name
        pers_model_type = Config().trainer.personalized_model_type
        if self.personalized_model is None:

            pers_model_params = Config().parameters.personalized_model._asdict()
            self.personalized_model = models_registry.get(
                model_name=pers_model_name,
                model_type=pers_model_type,
                model_params=pers_model_params,
            )
        else:
            self.personalized_model = self.personalized_model()

            logging.info(
                "Client[%d] defines the personalized model: %s",
                self.client_id,
                pers_model_name,
            )

        # This operation is important to the personalized FL
        # under Plato
        # Because, in general, when one client is called the first time,
        # its personalized model should be randomly intialized.
        # Howerver, Plato utilizes the `process` to simulate the
        # client and only the client id of each `process` is changed.
        # Thus, even a unseen client is selected, its personalized model
        # is the one trained by other previous clients.
        # Here, the function aims to persist the initial personalized model
        # when the client is selected the first time.
        self.persist_initial_model()

        # assign the client's personalized model to its trainer
        if (
            hasattr(self.trainer, "set_client_personalized_model")
            and self.trainer.personalized_model is None
        ):
            self.trainer.set_client_personalized_model(self.personalized_model)

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
        """

        # model_name = Config().trainer.model_name
        personalized_model_name = Config().trainer.personalized_model_name
        logging.info(
            "[Client #%d] loading its personalized model [%s].",
            self.client_id,
            personalized_model_name,
        )

        # when `persist_personalized_model` is set to be True, it means
        # that each client want to load its latest trained personalzied
        # model instead of using the initial one.
        if (
            hasattr(Config().trainer, "persist_personalized_model")
            and Config().trainer.persist_personalized_model
        ):
            # load the client's latest personalized model
            # we should know that only when this client is selected the first time,
            # the initial personalized model saved by `self.persist_initial_model`
            # will be loaded. Otherwise, the latest trained personalized model
            # will be loaded
            desired_round = self.current_round - 1

            logging.info(
                "[Client #%d] loads latest personalized model.",
                self.client_id,
            )
        else:
            # client does not want to use its trained personalzied model
            # thus, load the initial personalized model saved by
            # `self.persist_initial_model`
            desired_round = 0
            logging.info(
                "[Client #%d] loads initial personalized model.",
                self.client_id,
            )

        filename, ckpt_oper = checkpoint_operator.load_client_checkpoint(
            client_id=self.client_id,
            model_name=personalized_model_name,
            current_round=desired_round,
            run_id=None,
            epoch=None,
            prefix="personalized",
            anchor_metric="round",
            mask_words=["epoch"],
            use_latest=True,
        )
        loaded_weights = ckpt_oper.load_checkpoint()["model"]
        self.trainer.personalized_model.load_state_dict(loaded_weights, strict=True)

        logging.info(
            "[Client #%d] loads the personalized model from %s.",
            self.client_id,
            filename,
        )

    def _load_payload(self, server_payload) -> None:
        """Load the server model onto this client.

        The server will first assign its model with the server payload
        """
        logging.info(
            "[Client #%d] Received the model [%s].",
            self.client_id,
            Config().trainer.model_name,
        )
        # load the model
        self.algorithm.load_weights(server_payload, strict=True)
        # load the personalized model.
        self.load_personalized_model()

    async def _train(self):
        """The machine learning training workload on a client.

        To make it flexible, there are three training mode.

        """

        rounds = Config().trainer.rounds
        accuracy = -1
        training_time = 0.0

        if hasattr(self.trainer, "current_round"):
            self.trainer.current_round = self.current_round

        # visit personalized learning conditions
        is_pfl_mode = (
            hasattr(Config().clients, "do_personalization_interval")
            and Config.clients.do_personalization_interval != 0
        )
        # do pfl after the final round
        final_pfl = (
            self.current_round == rounds
            and is_pfl_mode
            and Config.clients.do_personalization_interval == -1
        )
        # do pfl during the training
        middle_pfl = (
            self.current_round < rounds
            and is_pfl_mode
            and self.current_round % Config().clients.do_personalization_interval == 0
        )

        normal_train = self.current_round < rounds and not middle_pfl

        # Perform model training
        if normal_train:
            logging.info(
                fonts.colourize(
                    f"[{self}] Started training in communication round #{self.current_round}."
                )
            )
            try:
                training_time = self.trainer.train(self.trainset, self.sampler)
            except ValueError:
                await self.sio.disconnect()

        if final_pfl or middle_pfl:
            logging.info(
                fonts.colourize(
                    f"[{self}] Started personalized training in the communication round #{self.current_round}."
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

        elif (hasattr(Config().clients, "do_test") and Config().clients.do_test) and (
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
            num_samples=self.sampler.num_samples(),
            accuracy=accuracy,
            training_time=training_time,
            comm_time=comm_time,
            update_response=False,
        )

        self._report = self.customize_report(report)

        return self._report, weights
