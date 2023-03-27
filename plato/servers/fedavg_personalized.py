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
from collections import UserDict
from collections.abc import Iterable
import random
import logging

from plato.servers import fedavg
from plato.config import Config


class TupleAsKeyDict(UserDict):
    """A customized dict with tuple as the key value."""

    def key_status(self, key):
        target_key = [
            data_key
            for data_key in self.data
            if key == data_key or (isinstance(data_key, Iterable) and key in data_key)
        ]

        if target_key:
            return target_key[0], True

        return None, False

    def __getitem__(self, key):
        """Get value in response to the key."""
        target_key, is_key_existed = self.key_status(key)

        if is_key_existed:
            return self.data[target_key]

        if hasattr(self.__class__, "__missing__"):
            return self.__class__.__missing__(self, key)
        raise KeyError(key)

    def __contains__(self, key):
        """Whether a key contained in the dict."""
        return self.key_status(key)[-1]


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

        # clients that have completed
        # personalization
        self.personalization_done_clients_pool = []

        # the flag denoting whether the personalization
        # has been started
        self.personalization_started = False
        # the flag denoting whether the personalization
        # has been terminated
        self.personalization_terminated = False

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
            "[%s] Total clients (%d), participanting clients (%d) ratio (%.3f), nonparticipant_clients (%d)",
            self,
            self.total_clients,
            self.participant_clients,
            self.participant_clients_ratio,
            self.nonparticipant_clients,
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
            else "Selected"
        )

        self.personalization_status_info = TupleAsKeyDict(
            {
                0: "No personalization required.",
                -1: "Personalization after the final round.",
                tuple(range(1, loaded_config.trainer.rounds)): (
                    "Personalization every {} rounds."
                ).format(self.do_personalization_interval),
            }
        )

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

        if self.do_personalization_interval == 0:
            logging.info("[%s] No personalization will be performed.", self)
        elif self.do_personalization_interval == -1:
            logging.info(
                "[%s] Personalization will be performed after final rounds.", self
            )
            logging.info(
                "[%s] %s.",
                self,
                self.personalization_group_type_info[
                    self.do_personalization_group.lower()
                ],
            )
        else:
            logging.info(
                "[%s] Personalization will be performed every %d rounds.",
                self,
                self.do_personalization_interval,
            )
            logging.info(
                "[%s] %s.",
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
                "[%s] Prepared participanting clients pool: %s",
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
                "[%s] Prepared non-participanting clients pool: %s",
                self,
                self.nonparticipant_clients_pool,
            )
            self.client_groups_pool["nonparticipant"] = self.nonparticipant_clients_pool

        if not self.client_groups_pool["total"]:
            self.client_groups_pool["total"] = self.clients_pool

    def perform_normal_training(self, clients_pool: List[int], clients_count: int):
        """Operations to guarantee general federated learning without personalization."""

        # by default, we run the general federated training
        # the clients pool should be participant clients
        clients_pool = self.participant_clients_pool
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
            # set the clients pool based on which group is setup
            # to do personalization
            clients_pool = self.client_groups_pool[
                self.do_personalization_group.lower()
            ]

        return clients_pool, clients_count

    def perform_final_personalization(
        self, clients_pool: List[int], clients_count: int
    ):
        """Operations to guarantee the personalization in the final."""

        if self.current_round > Config().trainer.rounds:

            # set the clients pool based on which group is setup
            # to do personalization
            clients_pool = self.client_groups_pool[
                self.do_personalization_group.lower()
            ]

            # open the personalization status flag
            self.personalization_started = True

            # maintain current round to be final round
            self.current_round = Config().trainer.rounds

            # the number of clients have not been visited
            non_visited_clients_count = len(clients_pool) - len(
                self.personalization_done_clients_pool
            )
            if non_visited_clients_count <= clients_count:
                # if the non visited clients is less than the
                # required clients per round,
                # select all left non visited clients
                # then personalization on all clients has
                # been terminated
                clients_count = non_visited_clients_count

                # we must change the clients_per_round to be
                # the number of clients_count, i.e., how many
                # clients will be selected in this round.
                # By doing so, the server can know how many updates
                # to be received for aggregation.
                self.clients_per_round = non_visited_clients_count

                # personalization has been terminated
                self.personalization_terminated = True

            # remove the visited clients from the clients_pool
            clients_pool = [
                client_id
                for client_id in clients_pool
                if client_id not in self.personalization_done_clients_pool
            ]

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

    def after_clients_sampling(self, selected_clients: List[int], **kwargs):
        """Perform operations after clients sampling."""
        # add clients who has been selected for personalization
        # to the `personalization_done_clients_pool`
        # thus, they will not be selected then.
        if self.personalization_started:
            self.personalization_done_clients_pool += selected_clients

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

        self.after_clients_sampling(selected_clients)

        self.prng_state = random.getstate()
        if selected_clients == len(clients_pool):
            logging.info("[%s] Selected all %d clients", self, len(selected_clients))
        else:
            logging.info("[%s] Selected clients: %s", self, selected_clients)

        return selected_clients

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

            if self.do_personalization_interval >= 0 or self.personalization_terminated:
                logging.info(
                    "%s Completed.",
                    self.personalization_status_info[self.do_personalization_interval],
                )

                await self._close()
