"""
A simple federated learning server capable of
1.- performing personalized training.
2.- utilizing a subset of clients for federated training while
others for evaluation.

The total clients are divided into two parts, referred to as
1.- participant clients
2.- nonparticipant clients

"""

from typing import List
import random
import logging

from plato.servers import fedavg
from plato.config import Config
from plato.utils import fonts


class Server(fedavg.Server):
    """Federated learning server for personalization and partial client selection."""

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

        # two types of client groups
        # utilized clients during federated training
        self.participant_clients = 0
        # unused clients during federated training
        self.nonparticipant_clients = 0

        # participanted clients as a percentage of all clients
        # by default, all clients will participant
        self.participant_clients_ratio = 1.0

        # clients id of two types of clients
        self.participant_clients_pool = None
        self.nonparticipant_clients_pool = None

        # client pool of three groups of clients
        # i.e., total, participant, nonparticipant
        self.client_groups_pool = None

        # the flag denoting whether all clients
        # perform personalization.
        # two special value -1 and 0,
        # if -1, do personalization after final round
        # if 0, ignore personalization. Default.
        self.do_personalization_interval = 0
        self.do_personalization_group = "participant"

        # these two variables are used for reminder purposes
        self.personalization_status_info = {}
        self.personalization_group_type_info = {}

        # the flag denoting whether the personalization
        # has been started or terminated
        self.performing_personalization = False
        # whether stop the terminate personalization
        # afterwards
        self.to_terminate_personalization = False

        self.initialize_personalization()
        self.check_hyper_parameters()

    def check_hyper_parameters(self):
        """Check whether hyper-parameters are set correctly."""
        loaded_config = Config()

        # `the participant_clients_pool` and `participant_clients_ratio`
        # if these two hyper-parameters are set simutaneously
        #  they should correspond to each other.
        if hasattr(loaded_config.clients, "participant_clients_ratio") and hasattr(
            loaded_config.clients, "participant_clients_pool"
        ):
            assert self.participant_clients == len(self.participant_clients_pool)

        assert (
            self.total_clients == self.participant_clients + self.nonparticipant_clients
        )

    def initialize_personalization(self):
        """Initialize two types of clients."""
        # set participant and nonparticipant clients
        loaded_config = Config()

        ## 1. initialize participanting clients
        self.participant_clients_ratio = (
            loaded_config.clients.participant_clients_ratio
            if hasattr(loaded_config.clients, "participant_clients_ratio")
            else 1.0
        )

        #  clients will / will not participant in federated training
        self.participant_clients = int(
            self.total_clients * self.participant_clients_ratio
        )
        self.nonparticipant_clients = int(self.total_clients - self.participant_clients)

        logging.info(
            fonts.colourize(
                "[%s] Total clients (%d), participanting clients (%d), nonparticipant_clients (%d). participanting ratio (%.3f).",
                colour="blue",
            ),
            self,
            self.total_clients,
            self.participant_clients,
            self.nonparticipant_clients,
            self.participant_clients_ratio,
        )

        ## 2. initialize personalization interval and types
        # if the value is set to be -1, thus the personalization
        # will be perform at the end of full communication rounds.
        # if do_personalization_interval is 0, no personalization
        # will be performed
        # by default, this value is set to be 0.
        self.do_personalization_interval = (
            loaded_config.server.do_personalization_interval
            if hasattr(loaded_config.server, "do_personalization_interval")
            else 0
        )
        self.do_personalization_group = (
            loaded_config.server.do_personalization_group
            if hasattr(loaded_config.server, "do_personalization_group")
            else "total"
        )

        pers_interval = self.do_personalization_interval
        self.personalization_status_info = dict(
            {
                0: "No personalization required.",
                -1: "Personalization after the final round.",
            }
        )
        if pers_interval not in self.personalization_status_info:
            self.personalization_status_info[pers_interval] = (
                "Personalization every {} rounds."
            ).format(pers_interval)

        self.personalization_group_type_info = {
            "total": "Personalization on total clients",
            "participant": "Personalization on participanting clients",
            "nonparticipant": "Personalization on non-participanting clients",
        }
        self.client_groups_pool = {
            "total": self.clients_pool,
            "participant": self.participant_clients_pool,
            "nonparticipant": self.nonparticipant_clients_pool,
        }

        if pers_interval == 0:
            logging.info(
                fonts.colourize(
                    "[%s] No personalization will be performed.", colour="blue"
                ),
                self,
            )
        elif pers_interval == -1:
            logging.info(
                fonts.colourize(
                    "[%s] Personalization will be performed after final rounds.",
                    colour="blue",
                ),
                self,
            )
            logging.info(
                fonts.colourize("[%s] %s.", colour="blue"),
                self,
                self.personalization_group_type_info[
                    self.do_personalization_group.lower()
                ],
            )
        else:
            logging.info(
                fonts.colourize(
                    "[%s] Personalization will be performed every %d rounds.",
                    colour="blue",
                ),
                self,
                pers_interval,
            )
            logging.info(
                fonts.colourize("[%s] %s.", colour="blue"),
                self,
                self.personalization_group_type_info[
                    self.do_personalization_group.lower()
                ],
            )

    def set_various_clients_pool(self, clients_pool: List[int]):
        """Set various clients pool utilized in federated learning.

        :param clients_pool: A list holding the id of all possible
            clients.

        Note, the participant clients pool will be set in the first round and no
            modification is performed afterwards.
        """
        loaded_config = Config()

        assert self.participant_clients <= len(clients_pool)

        # only set the clients pool when they are empty.

        # load the participant clients pool from configuration
        # if this is not provided,
        # The first `self.participant_clients` in clients_pool will
        # be utilized as participant clients.
        if self.participant_clients_pool is None:
            self.participant_clients_pool = (
                loaded_config.clients.participant_clients_pool
                if hasattr(loaded_config.clients, "participant_clients_pool")
                else clients_pool[: self.participant_clients]
            )
            self.client_groups_pool["participant"] = self.participant_clients_pool
            logging.info(
                fonts.colourize(
                    "[%s] Prepared participanting clients pool: %s", colour="blue"
                ),
                self,
                self.participant_clients_pool,
            )
        # load the non-participant clients pool from configuration
        # if this is not provided,
        # Then reminding clients of clients_pool apart from the
        # participant_clients will be utilized as non-participant clients.
        if self.nonparticipant_clients_pool is None:
            self.nonparticipant_clients_pool = [
                client_id
                for client_id in clients_pool
                if client_id not in self.participant_clients_pool
            ]

            logging.info(
                fonts.colourize(
                    "[%s] Prepared non-participanting clients pool: %s", colour="blue"
                ),
                self,
                self.nonparticipant_clients_pool,
            )
            self.client_groups_pool["nonparticipant"] = self.nonparticipant_clients_pool

        if not self.client_groups_pool["total"]:
            self.client_groups_pool["total"] = self.clients_pool

    def perform_normal_training(self, clients_pool: List[int], clients_count: int):
        """Operations to guarantee general federated learning without personalization."""
        # always set the performing_personalization to close
        # personalization
        self.performing_personalization = False

        # reset `clients_per_round` to the predefined hyper-parameter
        self.clients_per_round = Config().clients.per_round

        # set the clients_pool to be participant_clients_pool
        clients_pool = self.participant_clients_pool

        # However, as we modified the membership `clients_per_round` previously,
        # the clients_count here may be the modified `clients_per_round`.
        # We need to convert it back to `self.clients_per_round`.
        # For example, if the previous round performs personalization,
        # the `clients_per_round` was modified to the size of the client group
        # that performs personalization, making the current round should change it back.
        clients_count = (
            clients_count
            if clients_count <= self.clients_per_round
            else self.clients_per_round
        )

        # by default, we run the general federated training
        # the clients pool should be participant clients
        assert clients_count <= len(self.participant_clients_pool)

        return clients_pool, clients_count

    def perform_intermediate_personalization(
        self, clients_pool: List[int], clients_count: int
    ):
        """Operations to guarantee the personalization during federated training.

        This is generally utilized as the evaluation purpose.

        In current code version, we only support the client sampling method:
            randomly select #`per_round` clients from the desired
            `do_personalization_group`.
        """

        if self.current_round % self.do_personalization_interval == 0:

            logging.info(
                fonts.colourize(
                    "Starting Personalization mode during round %d", colour="blue"
                ),
                self.current_round,
            )
            # open the personalization status flag
            self.performing_personalization = True
            # set the clients pool based on which group is setup
            # to do personalization
            clients_pool = self.client_groups_pool[
                self.do_personalization_group.lower()
            ]
            # change the clients_per_round to be the whole set
            # of clients for personalization
            self.clients_per_round = len(clients_pool)
            clients_count = self.clients_per_round

        return clients_pool, clients_count

    def perform_final_personalization(
        self, clients_pool: List[int], clients_count: int
    ):
        """Operations to guarantee the personalization in the final."""

        if self.current_round > Config().trainer.rounds:

            logging.info(
                fonts.colourize(
                    "Starting Personalization mode after the final round", colour="blue"
                )
            )

            # set the clients pool based on which group is setup
            # to do personalization
            clients_pool = self.client_groups_pool[
                self.do_personalization_group.lower()
            ]

            # set clients for personalization
            self.clients_per_round = len(clients_pool)
            clients_count = self.clients_per_round

            # open the personalization status flag
            self.performing_personalization = True

            # to terminate the personalization afterwards
            self.to_terminate_personalization = True

        return clients_pool, clients_count

    def before_clients_sampling(
        self, clients_pool: List[int], clients_count: int, **kwargs
    ):
        """Determine clients pool and clients count before samling clients."""

        # perform normal training
        clients_pool, clients_count = self.perform_normal_training(
            clients_pool, clients_count
        )

        # personalization after final round is required
        if self.do_personalization_interval == -1:
            clients_pool, clients_count = self.perform_final_personalization(
                clients_pool, clients_count
            )
        # personalization during federated training
        if self.do_personalization_interval > 0:
            clients_pool, clients_count = self.perform_intermediate_personalization(
                clients_pool, clients_count
            )

        return clients_pool, clients_count

    def choose_clients(self, clients_pool: List[int], clients_count: int):
        """Chooses a subset of the clients to participate in each round.

        In plato, this input `clients_pool` contains total clients
        id by default.
        """
        # set required clients pool when possible
        self.set_various_clients_pool(clients_pool)

        clients_pool, clients_count = self.before_clients_sampling(
            clients_pool, clients_count
        )

        random.setstate(self.prng_state)

        # Select clients randomly
        selected_clients = random.sample(clients_pool, clients_count)

        self.prng_state = random.getstate()
        if selected_clients == len(clients_pool):
            logging.info("[%s] Selected all %d clients", self, len(selected_clients))
        else:
            logging.info("[%s] Selected clients: %s", self, selected_clients)

        return selected_clients

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        """Customizes the server response with any additional information."""
        super().customize_server_response(server_response, client_id)

        if self.performing_personalization:
            server_response["learning_mode"] = "personalization"
        else:
            server_response["learning_mode"] = "normal"

        server_response["client_group"] = (
            "participant"
            if client_id in self.client_groups_pool["participant"]
            else "nonparticipant"
        )
        return server_response

    async def wrap_up(self):
        """Wrapping up when each round of training is done.

        This function is required to be extended if
        the user want to support:
            perform the evaluation stage when the
            learning reach the target_accuracy
            or target_perplexity.

        """

        self.save_to_checkpoint()

        if self.current_round >= Config().trainer.rounds:
            logging.info("Target number of training rounds reached.")

            if (
                self.do_personalization_interval >= 0
                or self.to_terminate_personalization
            ):

                logging.info(
                    "%s Completed.",
                    self.personalization_status_info[self.do_personalization_interval],
                )
                await self._close()
